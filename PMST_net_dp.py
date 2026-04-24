import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import os
import shutil
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import joblib
import math
import warnings
import gc
import sys
import datetime
import time
import random

warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局配置
# ==========================================
CONFIG = {
    'BASE_PATH': "/public/home/putianshu/vis_mlp",
    
    # 第一阶段：ERA5
    'S1_DATA_DIR': "/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_opt", 
    'S1_SUFFIX': "_9h_pmst_v2",   
    
    # 第二阶段：Forecast
    'S2_DATA_DIR': "/public/home/putianshu/vis_mlp/ml_dataset_pmst_finetune_v1", 
    'S2_SUFFIX': "_9h_forecast_v1", 
    
    # --- 训练参数 ---
    'S1_TOTAL_STEPS': 60000,     
    'S1_VAL_INTERVAL': 2000,      
    'S1_BATCH_SIZE': 512,         
    'S1_GRAD_ACCUM': 2,
    'S1_POS_RATIO': 0.40,  
    
    'S2_TOTAL_STEPS': 10000,
    'S2_VAL_INTERVAL': 500,
    'S2_BATCH_SIZE': 256,
    'S2_GRAD_ACCUM': 1,
    'S2_POS_RATIO': 0.40,
}

# ==========================================
# 1. 基础工具与分布式初始化
# ==========================================

def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    torch.cuda.set_device(local_rank)
    
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl", 
                init_method='env://', 
                timeout=datetime.timedelta(minutes=120)
            )
    return local_rank, global_rank, world_size

def copy_to_local(src_path, local_rank):
    filename = os.path.basename(src_path)
    target_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    local_path = os.path.join(target_dir, filename)
    lock_path = local_path + ".lock"

    if local_rank == 0:
        need_copy = True
        if os.path.exists(local_path):
            if os.path.getsize(local_path) == os.path.getsize(src_path):
                need_copy = False
                print(f"[Node Local-0] Found valid cache: {local_path}, skipping copy.", flush=True)

        if need_copy:
            print(f"[Node Local-0] Copying {src_path} to {local_path}...", flush=True)
            try:
                with open(lock_path, 'w') as f: f.write("copying")
                shutil.copyfile(src_path, local_path)
                print(f"[Node Local-0] Copy finished.", flush=True)
            except Exception as e:
                print(f"[Node Local-0] Copy FAILED: {e}. Fallback to NFS.", flush=True)
                local_path = src_path
            finally:
                if os.path.exists(lock_path): os.remove(lock_path)
    
    if dist.is_initialized():
        dist.barrier()
        
    if not os.path.exists(local_path):
        return src_path
    return local_path

# ==========================================
# 2. 改进的采样器
# ==========================================

class InfiniteBalancedSampler(Sampler):
    def __init__(self, dataset, batch_size, pos_ratio=0.40, rank=0, world_size=1, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        
        y = np.array(dataset.y_cls)
        # Class 0: Fog (<500m), Class 1: Mist (500-1000m), Class 2: Clear
        all_pos = np.where(y <= 1)[0]
        all_neg = np.where(y == 2)[0]
        
        # 确保每个 rank 获得不同的子集
        np.random.seed(seed + rank)
        np.random.shuffle(all_pos)
        np.random.shuffle(all_neg)
        
        self.pos_indices = np.array_split(all_pos, world_size)[rank]
        self.neg_indices = np.array_split(all_neg, world_size)[rank]
        
        self.n_pos = int(batch_size * pos_ratio)
        self.n_neg = batch_size - self.n_pos
        
        if rank == 0:
            print(f"[Sampler] Total Pos (Fog/Mist): {len(all_pos)}, Total Neg: {len(all_neg)}")
            print(f"[Sampler] This Rank Pos: {len(self.pos_indices)}, Neg: {len(self.neg_indices)}")
            print(f"[Sampler] Batch Layout: {self.n_pos} Pos ({pos_ratio*100:.1f}%) + {self.n_neg} Neg")

    def __iter__(self):
        # 使用时间戳保证每个 epoch 的随机性不同
        epoch_seed = self.seed + self.rank + int(time.time() * 1000) % 10000
        g = torch.Generator()
        g.manual_seed(epoch_seed)
        
        while True:
            pos_batch = torch.randint(0, len(self.pos_indices), (self.n_pos,), generator=g).numpy()
            neg_batch = torch.randint(0, len(self.neg_indices), (self.n_neg,), generator=g).numpy()
            
            indices = np.concatenate([self.pos_indices[pos_batch], self.neg_indices[neg_batch]])
            np.random.shuffle(indices)
            yield indices.tolist()

    def __len__(self):
        return 2147483647

# ==========================================
# 3. 新的 Loss 函数
# ==========================================

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        
        if alpha is None:
            self.alpha = torch.tensor([10.0, 3.0, 1.0])
        else:
            self.alpha = torch.tensor(alpha)
        
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.device != self.alpha.device:
            self.alpha = self.alpha.to(inputs.device)
        
        # 数值稳定性保护
        inputs = torch.clamp(inputs, min=-20, max=20)
            
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        
        focal_weight = torch.ones_like(targets, dtype=torch.float)
        
        # 正样本 (Fog/Mist)
        pos_mask = targets <= 1
        focal_weight[pos_mask] = (1 - pt[pos_mask] + 1e-6) ** self.gamma_pos
        
        # 负样本 (Clear)
        neg_mask = targets == 2
        focal_weight[neg_mask] = (1 - pt[neg_mask] + 1e-6) ** self.gamma_neg
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class HardMiningWrapper(nn.Module):
    def __init__(self, base_loss, keep_ratio_start=1.0, keep_ratio_end=0.3, total_steps=60000):
        super().__init__()
        self.base_loss = base_loss
        self.keep_ratio_start = keep_ratio_start
        self.keep_ratio_end = keep_ratio_end
        self.total_steps = total_steps
        self.current_step = 0

    def forward(self, inputs, targets):
        loss_per_sample = self.base_loss(inputs, targets)
        
        progress = min(self.current_step / self.total_steps, 1.0)
        current_ratio = self.keep_ratio_start + (self.keep_ratio_end - self.keep_ratio_start) * progress
        
        k = max(1, int(len(loss_per_sample) * current_ratio))
        topk_loss, _ = torch.topk(loss_per_sample, k)
        
        self.current_step += 1
        return topk_loss.mean()


class PhysicsConstrainedRegLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, reg_pred, reg_target, raw_vis):
        # 只对低能见度样本计算回归 loss
        mask_low_vis = raw_vis < 2000
        
        if mask_low_vis.sum() == 0:
            return torch.tensor(0.0, device=reg_pred.device)
        
        reg_flat = reg_pred.view(-1)
        loss = F.huber_loss(reg_flat[mask_low_vis], reg_target[mask_low_vis], delta=1.0)
        
        return self.alpha * loss

# ==========================================
# 4. 双流特征提取架构
# ==========================================

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=4):
        super(ChebyKANLayer, self).__init__()
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.xavier_normal_(self.cheby_coeffs)
        self.layer_norm = nn.LayerNorm(input_dim) 
        self.base_activation = nn.SiLU()

    def forward(self, x):
        # 增加数值保护，防止输入过大导致 NaN
        x = torch.clamp(x, -10, 10)
        x = self.layer_norm(x)
        x = torch.tanh(x)
        cheby_values = [torch.ones_like(x), x]
        for i in range(2, self.degree + 1):
            next_t = 2 * x * cheby_values[-1] - cheby_values[-2]
            cheby_values.append(next_t)
        stacked_cheby = torch.stack(cheby_values, dim=-1)
        y = torch.einsum("bid,iod->bo", stacked_cheby, self.cheby_coeffs)
        return self.base_activation(y)


class MultiScaleTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 7]):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = (k - 1) // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=pad),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(0.25) 
            ))
        self.proj = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, 1)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        return self.proj(torch.cat(outs, dim=1))


class PhysicalStateEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.encoder(x) + self.shortcut(x)


class TemporalEvolutionEncoder(nn.Module):
    def __init__(self, n_vars, n_steps, hidden_dim):
        super().__init__()
        self.embed = nn.Conv1d(n_vars, hidden_dim, 1)
        self.tcn = nn.Sequential(
            MultiScaleTCNBlock(hidden_dim, hidden_dim, kernel_sizes=[3, 5, 7]),
            MultiScaleTCNBlock(hidden_dim, hidden_dim, kernel_sizes=[3, 5])
        )
        self.temporal_attn = nn.Parameter(torch.linspace(0.5, 1.0, n_steps))

    def forward(self, x):
        x = x.permute(0, 2, 1)  # → [B, n_vars, n_steps]
        x = self.embed(x)        # → [B, hidden_dim, n_steps]
        x = self.tcn(x)          # → [B, hidden_dim, n_steps]
        
        weights = F.softmax(self.temporal_attn, dim=0).view(1, 1, -1).to(x.device)
        x = (x * weights).sum(dim=2)  # → [B, hidden_dim]
        return x


class DualStreamPMSTNet(nn.Module):
    def __init__(self, dyn_vars_count=25, window_size=9, 
                 static_cont_dim=5, veg_num_classes=21, veg_emb_dim=16,
                 hidden_dim=512, num_classes=3):
        super().__init__()
        self.dyn_vars = dyn_vars_count
        self.window = window_size
        self.static_cont_dim = static_cont_dim
        
        # 1. 静态变量编码
        self.veg_embedding = nn.Embedding(veg_num_classes, veg_emb_dim)
        total_static_dim = static_cont_dim + veg_emb_dim
        
        self.static_encoder = nn.Sequential(
            ChebyKANLayer(total_static_dim, 256, degree=3),
            nn.LayerNorm(256),
            nn.Linear(256, hidden_dim // 2) 
        )
        
        # ============================================================
        # 2. 双流编码器 (FIX 2: 更新索引)
        # ============================================================
        # 物理流：RH2M(0), T2M(1), LCC(10), RH_925(12), Q_925(19/20?), W_925(20/21?), DPD(22), INVERSION(23)
        # 严格按照用户提供的 231 维列表：
        # 0:RH2M, 1:T2M, 2:PRECIP, 3:MSLP, 4:SW_RAD, 5:U10, 6:WSPD10, 7:V10
        # 10:LCC, 12:RH_925, 20:Q_925, 21:W_925, 22:W_1000, 23:DPD, 24:INVERSION
        
        self.physical_vars_indices = [0, 1, 10, 12, 20, 21, 23, 24]
        physical_dim = len(self.physical_vars_indices)
        self.physical_stream = PhysicalStateEncoder(physical_dim, hidden_dim)
        
        # 时序流：PRECIP(2), MSLP(3), SW_RAD(4), U10(5), WSPD10(6), V10(7)
        self.temporal_vars_indices = [2, 3, 4, 5, 6, 7]
        temporal_dim = len(self.temporal_vars_indices)
        self.temporal_stream = TemporalEvolutionEncoder(temporal_dim, window_size, hidden_dim)
        
        # 3. 特征融合
        fusion_dim = hidden_dim * 2 + hidden_dim // 2
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            ChebyKANLayer(hidden_dim, hidden_dim, degree=3)
        )
        
        # 4. 输出头
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.cls_head[-1].bias.data = torch.tensor([-2.0, -1.0, 1.0])
        
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        split_dyn = self.dyn_vars * self.window  
        split_static = split_dyn + self.static_cont_dim  
        
        x_dyn_flat = x[:, :split_dyn]              
        x_stat_cont = x[:, split_dyn:split_static] 
        x_veg_id = x[:, -1].long()                 
        
        x_dyn_seq = x_dyn_flat.view(-1, self.window, self.dyn_vars)
        
        veg_vec = self.veg_embedding(x_veg_id)
        x_static_full = torch.cat([x_stat_cont, veg_vec], dim=1)
        static_feat = self.static_encoder(x_static_full)
        
        x_current = x_dyn_seq[:, -1, :] 
        x_physical = x_current[:, self.physical_vars_indices] 
        physical_feat = self.physical_stream(x_physical) 
        
        x_temporal = x_dyn_seq[:, :, self.temporal_vars_indices] 
        temporal_feat = self.temporal_stream(x_temporal) 
        
        combined_feat = torch.cat([physical_feat, temporal_feat, static_feat], dim=1)
        embedding = self.fusion_layer(combined_feat)
        
        logits = self.cls_head(embedding)    
        reg_out = self.reg_head(embedding)   
        
        return logits, reg_out

# ==========================================
# 5. Data & Processing (FIX 1: 数据截断)
# ==========================================

class PMSTDataset(Dataset):
    def __init__(self, X_path, y_cls, y_reg, y_raw, scaler=None):
        self.X = np.load(X_path, mmap_mode='r')
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.as_tensor(y_reg, dtype=torch.float32)
        self.y_raw = torch.as_tensor(y_raw, dtype=torch.float32)
        
        self.has_scaler = scaler is not None
        if self.has_scaler:
            self.center = scaler.center_.astype(np.float32)
            self.scale = scaler.scale_.astype(np.float32)
            self.scale = np.where(self.scale == 0, 1.0, self.scale)
            
        # FIX 1: 定义截断范围
        self.clip_min = -10.0
        self.clip_max = 10.0

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        features = row[:-1].astype(np.float32)
        veg_id = row[-1]
        
        # 处理 NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.has_scaler:
            features = (features - self.center) / self.scale
            
        # FIX 1: 强力截断，防止梯度爆炸
        features = np.clip(features, self.clip_min, self.clip_max)
            
        features = np.append(features, veg_id)
        return torch.from_numpy(features).float(), self.y_cls[idx], self.y_reg[idx], self.y_raw[idx]


def load_data_and_scale(data_dir, scaler=None, rank=0, device=None, reuse_scaler=False):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0: print(f"Loading Metadata from: {data_dir}", flush=True)
    
    raw_train_path = os.path.join(data_dir, f'X_train.npy')
    raw_val_path = os.path.join(data_dir, f'X_val.npy')
    
    train_path = copy_to_local(raw_train_path, local_rank)
    val_path = copy_to_local(raw_val_path, local_rank)
    
    y_train_raw = np.load(os.path.join(data_dir, f'y_train.npy'))
    y_val_raw = np.load(os.path.join(data_dir, f'y_val.npy'))
    
    y_train_m = y_train_raw * 1000.0 if np.max(y_train_raw) < 100 else y_train_raw
    y_val_m = y_val_raw * 1000.0 if np.max(y_val_raw) < 100 else y_val_raw
    
    def to_class(y_m):
        cls = np.zeros_like(y_m, dtype=np.int64)
        cls[y_m >= 500] = 1  
        cls[y_m >= 1000] = 2 
        return cls
        
    y_train_cls = to_class(y_train_m)
    y_val_cls = to_class(y_val_m)
    y_train_log = np.log1p(y_train_m).astype(np.float32)
    y_val_log = np.log1p(y_val_m).astype(np.float32)
    
    if scaler is None or not reuse_scaler:
        if rank == 0:
            print(f"  [Rank 0] Fitting RobustScaler...", flush=True)
            scaler = RobustScaler()
            X_temp = np.load(train_path, mmap_mode='r')
            subset_size = min(500000, len(X_temp))
            indices = np.random.choice(len(X_temp), subset_size, replace=False)
            indices.sort()
            X_subset = X_temp[indices, :-1]
            X_subset = np.nan_to_num(X_subset, nan=0.0)
            scaler.fit(X_subset)
            del X_subset, X_temp
            
            center_tensor = torch.from_numpy(scaler.center_).float().to(device)
            scale_tensor = torch.from_numpy(scaler.scale_).float().to(device)
            dim_tensor = torch.tensor([len(scaler.center_)], device=device)
        else:
            dim_tensor = torch.tensor([0], device=device)
            center_tensor = None
            scale_tensor = None

        if dist.is_initialized(): dist.broadcast(dim_tensor, src=0)
        feat_dim = dim_tensor.item()
        
        if rank != 0:
            center_tensor = torch.zeros(feat_dim, device=device)
            scale_tensor = torch.zeros(feat_dim, device=device)
        
        if dist.is_initialized():
            dist.broadcast(center_tensor, src=0)
            dist.broadcast(scale_tensor, src=0)
            
        if rank != 0:
            scaler = RobustScaler()
            scaler.center_ = center_tensor.cpu().numpy()
            scaler.scale_ = scale_tensor.cpu().numpy()
            
    train_ds = PMSTDataset(train_path, y_train_cls, y_train_log, y_train_m, scaler=scaler)
    val_ds = PMSTDataset(val_path, y_val_cls, y_val_log, y_val_m, scaler=scaler)
    
    return train_ds, val_ds, scaler

# ==========================================
# 6. 评估函数
# ==========================================

def calculate_meteorological_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    ts = tp / (tp + fp + fn + 1e-6)
    
    num = 2 * (tp * tn - fp * fn)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = num / (den + 1e-6)
    
    hits_rand = (tp + fn) * (tp + fp) / (tp + fn + fp + tn)
    ets = (tp - hits_rand) / (tp + fn + fp - hits_rand + 1e-6)
    
    return ts, hss, ets

def evaluate_multi_metrics(model, loader, device, return_probs=False):
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for bx, by_cls, _, _ in loader:
            bx = bx.to(device, non_blocking=True)
            with autocast():
                logits, _ = model(bx)
            all_logits.append(logits.float().cpu())
            all_targets.append(by_cls)
            
    all_logits = torch.cat(all_logits)
    probs = F.softmax(all_logits, dim=1).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    prob_c0 = probs[:, 0]
    y_true_c0 = (all_targets == 0).astype(int)
    
    best_ts = 0.0
    best_res = {}
    
    thresholds = np.concatenate([
        np.arange(0.001, 0.01, 0.001),
        np.arange(0.01, 0.1, 0.01),
        np.arange(0.1, 0.5, 0.05),
        np.arange(0.5, 0.9, 0.1)
    ])
    
    candidate_results = []
    
    for t in thresholds:
        y_pred = (prob_c0 > t).astype(int)
        
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        
        ts, hss, ets = calculate_meteorological_metrics(y_true_c0, y_pred)
        rec = recall_score(y_true_c0, y_pred, zero_division=0)
        prec = precision_score(y_true_c0, y_pred, zero_division=0)
        
        candidate_results.append({
            'thresh': t,
            'ts': ts,
            'ets': ets,
            'hss': hss,
            'recall': rec,
            'precision': prec
        })
        
        if ts > best_ts:
            best_ts = ts
            best_res = {
                'score': ts,
                'ts_score': ts,
                'ets_score': ets,
                'hss_score': hss,
                'recall_c0': rec,
                'prec_c0': prec,
                'best_thresh': t
            }
    
    if best_ts == 0.0:
        best_res = {
            'score': 0.0, 
            'ts_score': 0.0, 
            'ets_score': 0.0, 
            'hss_score': 0.0,
            'recall_c0': 0.0, 
            'prec_c0': 0.0, 
            'best_thresh': 0.5
        }
    
    if return_probs:
        best_res['probs'] = probs
        best_res['targets'] = all_targets
        best_res['candidates'] = sorted(candidate_results, key=lambda x: x['ts'], reverse=True)[:5]
    
    return best_res

# ==========================================
# 7. 统一训练流程 (FIX 3: 熔断与Unscale修复)
# ==========================================

def train_unified_stream(stage_name, model, train_ds, val_loader, optimizer, 
                        criterions, scaler_amp, device, rank, world_size,
                        total_steps, val_interval, batch_size, grad_accum, pos_ratio):
    
    sampler = InfiniteBalancedSampler(
        train_ds, 
        batch_size=batch_size, 
        pos_ratio=pos_ratio,
        rank=rank, 
        world_size=world_size
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_sampler=sampler, 
        num_workers=6, 
        pin_memory=True
    )
    
    train_iter = iter(train_loader)
    model.train()
    
    crit_cls, crit_reg = criterions
    
    best_score = -1
    avg_loss_cls = 0
    avg_loss_reg = 0
    start_time = time.time()
    
    optimizer.zero_grad()
    
    for step in range(1, total_steps + 1):
        try:
            bx, by_cls, by_log, by_raw = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            bx, by_cls, by_log, by_raw = next(train_iter)
        
        bx = bx.to(device, non_blocking=True)
        by_cls = by_cls.to(device, non_blocking=True)
        by_log = by_log.to(device, non_blocking=True)
        by_raw = by_raw.to(device, non_blocking=True)
        
        # FIX 3: Loss NaN 检查与熔断
        valid_step = True
        
        with autocast():
            logits, reg_pred = model(bx)
            l_cls = crit_cls(logits, by_cls)
            l_reg = crit_reg(reg_pred, by_log, by_raw)
            loss = l_cls + 0.1 * l_reg
            loss = loss / grad_accum
            
        # 立即检查 Loss 是否正常
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0:
                print(f"\n[WARNING] Step {step}: Loss is NaN/Inf. Skipping batch.", flush=True)
            optimizer.zero_grad()
            valid_step = False
            # 跳过 backward
        
        if valid_step:
            scaler_amp.scale(loss).backward()
            
            avg_loss_cls += l_cls.item()
            avg_loss_reg += l_reg.item()
            
            if step % grad_accum == 0:
                # -----------------------------------------------------------
                # FIX START: 修正 AMP 更新逻辑
                # -----------------------------------------------------------
                
                # 1. 首先反缩放梯度
                scaler_amp.unscale_(optimizer)
                
                # 2. 裁剪梯度 (此时梯度已经是真实的数值，不是缩放后的)
                # 即使梯度中有 Inf/NaN，clip_grad_norm 也会处理（返回 NaN/Inf），不会报错
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 3. 关键修改：无论梯度是否正常，都必须调用 scaler 的 step 和 update
                # 如果 grad_norm 是 Inf/NaN，scaler.step 会自动跳过 optimizer.step()
                scaler_amp.step(optimizer)
                
                # 4. 必须调用 update，它负责更新 scale factor 并【清除 unscale 状态】
                scaler_amp.update()
                
                # 5. 清空梯度
                optimizer.zero_grad()
                
                # (可选) 如果你想监控梯度爆炸情况，可以在这里打印，但不要改变控制流
                if rank == 0 and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    print(f"\n[Warning] Step {step}: Grad norm is {grad_norm.item()}. Scaler will skip update and decrease scale.", flush=True)

                # -----------------------------------------------------------
                # FIX END
                # -----------------------------------------------------------
        
        if rank == 0 and step % 100 == 0:
            elapsed = time.time() - start_time
            avg_loss_cls /= 100
            avg_loss_reg /= 100
            
            print(f"\r[{stage_name}] Step {step}/{total_steps} | "
                  f"L_cls: {avg_loss_cls:.4f} | L_reg: {avg_loss_reg:.4f} | "
                  f"Speed: {100/elapsed:.1f} steps/s", 
                  end="", flush=True)
            
            avg_loss_cls = 0
            avg_loss_reg = 0
            start_time = time.time()
            
        if step % val_interval == 0:
            if rank == 0:
                print(f"\n[{stage_name} Validation @ Step {step}]", flush=True)
            
            met = evaluate_multi_metrics(model, val_loader, device, return_probs=True)
            
            if rank == 0:
                print(f"  Best Result: TS={met['ts_score']:.4f} | "
                      f"Recall={met['recall_c0']:.1%} | Precision={met['prec_c0']:.1%} | "
                      f"Thresh={met['best_thresh']:.4f}", flush=True)
                
                if 'candidates' in met:
                    print("  Top-3 Candidates:")
                    for i, cand in enumerate(met['candidates'][:3], 1):
                        print(f"    {i}. Thresh={cand['thresh']:.4f} → "
                              f"TS={cand['ts']:.4f}, R={cand['recall']:.2%}, P={cand['precision']:.2%}")
                
                if met['score'] > best_score:
                    best_score = met['score']
                    save_name = f"pmst_stage1_best_v2.pth" if stage_name == 'S1' else f"pmst_stage2_final_v2.pth"
                    save_path = os.path.join(CONFIG['BASE_PATH'], f"model/{save_name}")
                    torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), save_path)
                    print(f"  ✓ Checkpoint saved: {save_path} (TS={best_score:.4f})")
            
            model.train()
    
    if rank == 0:
        print(f"\n[{stage_name}] Training completed. Best TS: {best_score:.4f}", flush=True)
    
    del train_iter, train_loader
    if dist.is_initialized():
        dist.barrier()

# ==========================================
# 8. Main Execution
# ==========================================

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    miopen_cache_dir = f"/tmp/miopen_cache_rank_{local_rank}"
    os.makedirs(miopen_cache_dir, exist_ok=True)
    os.environ["MIOPEN_USER_DB_PATH"] = miopen_cache_dir
    
    local_rank, global_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    scaler_amp = GradScaler()
    
    base_path = CONFIG['BASE_PATH']
    if global_rank == 0:
        os.makedirs(os.path.join(base_path, "model"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "scalers"), exist_ok=True)
        print("="*60)
        print("  Visibility Forecasting - Fixed & Robust Pipeline")
        print("="*60)
        print(f"  Configuration:")
        print(f"    - S1 Steps: {CONFIG['S1_TOTAL_STEPS']}")
        print(f"    - S1 Batch Size: {CONFIG['S1_BATCH_SIZE']}")
        print(f"    - Architecture: DualStreamPMSTNet (Indices Corrected)")
        print(f"    - Safety: Loss Clamping & NaN Check Enabled")
        print("="*60)

    # -------------------------------------------------------
    # Stage 1: ERA5 Pre-training
    # -------------------------------------------------------
    if global_rank == 0:
        print("\n[STAGE 1] ERA5 Pre-training")
        print("-" * 60)
    
    s1_train_ds, s1_val_ds, scaler_s1 = load_data_and_scale(
        CONFIG['S1_DATA_DIR'], rank=global_rank, device=device, reuse_scaler=False
    )
    
    if global_rank == 0:
        joblib.dump(scaler_s1, os.path.join(base_path, "scalers/scaler_pmst_v2.pkl"))
        print(f"  ✓ Scaler saved")
    
    s1_val_loader = DataLoader(
        s1_val_ds, 
        batch_size=CONFIG['S1_BATCH_SIZE'], 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    model = DualStreamPMSTNet(
        dyn_vars_count=25, 
        window_size=9,
        hidden_dim=512, 
        num_classes=3
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    
    base_focal = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=4.0, alpha=[10.0, 3.0, 1.0], reduction='none')
    crit_cls = HardMiningWrapper(base_focal, keep_ratio_start=1.0, keep_ratio_end=0.3, 
                                  total_steps=CONFIG['S1_TOTAL_STEPS'])
    crit_reg = PhysicsConstrainedRegLoss(alpha=1.0)
    
    train_unified_stream(
        'S1', model, s1_train_ds, s1_val_loader, optimizer, 
        (crit_cls, crit_reg), scaler_amp, device, global_rank, world_size,
        CONFIG['S1_TOTAL_STEPS'], CONFIG['S1_VAL_INTERVAL'], 
        CONFIG['S1_BATCH_SIZE'], CONFIG['S1_GRAD_ACCUM'],
        CONFIG['S1_POS_RATIO']
    )
    
    del s1_train_ds, s1_val_ds, s1_val_loader
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------
    # Stage 2: Forecast Fine-tuning
    # -------------------------------------------------------
    if global_rank == 0:
        print("\n[STAGE 2] Forecast Fine-tuning")
        print("-" * 60)
    
    s2_train_ds, s2_val_ds, _ = load_data_and_scale(
        CONFIG['S2_DATA_DIR'], 
        scaler=scaler_s1, 
        rank=global_rank, 
        device=device, 
        reuse_scaler=True
    )
    
    s2_val_loader = DataLoader(
        s2_val_ds, 
        batch_size=CONFIG['S2_BATCH_SIZE'], 
        shuffle=False, 
        num_workers=4
    )
    
    params = [
        {'params': [p for n, p in model.named_parameters() if "head" not in n], 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if "head" in n], 'lr': 5e-5}
    ]
    s2_optimizer = optim.AdamW(params, weight_decay=1e-2)
    
    s2_crit_cls = HardMiningWrapper(base_focal, keep_ratio_start=0.5, keep_ratio_end=0.3,
                                     total_steps=CONFIG['S2_TOTAL_STEPS'])
    
    train_unified_stream(
        'S2', model, s2_train_ds, s2_val_loader, s2_optimizer,
        (s2_crit_cls, crit_reg), scaler_amp, device, global_rank, world_size,
        CONFIG['S2_TOTAL_STEPS'], CONFIG['S2_VAL_INTERVAL'],
        CONFIG['S2_BATCH_SIZE'], CONFIG['S2_GRAD_ACCUM'],
        CONFIG['S2_POS_RATIO']
    )
    
    if global_rank == 0:
        print("\n" + "="*60)
        print("  Training Pipeline Completed")
        print("="*60)
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
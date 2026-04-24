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
# 0. 全局配置 (已修改以支持动态窗口)
# ==========================================

# --- [关键修改] 在此处设置时间窗口大小 ---
TARGET_WINDOW_SIZE = 3
# ------------------------------------

# 根据窗口大小自动选择路径
BASE_PATH = "/public/home/putianshu/vis_mlp"


S1_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_{TARGET_WINDOW_SIZE}h"
S2_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_finetune_v1_{TARGET_WINDOW_SIZE}h"

CONFIG = {
    'BASE_PATH': BASE_PATH,
    'WINDOW_SIZE': TARGET_WINDOW_SIZE, # 新增配置项
    
    # 第一阶段: ERA5
    'S1_DATA_DIR': S1_DIR, 
    'S1_SUFFIX': f"_{TARGET_WINDOW_SIZE}h_pmst_v2",   
    
    # 第二阶段: Forecast
    'S2_DATA_DIR': S2_DIR, 
    'S2_SUFFIX': f"_{TARGET_WINDOW_SIZE}h_forecast_v1", 
    
    # --- 训练参数 ---
    'S1_TOTAL_STEPS': 80000,      
    'S1_VAL_INTERVAL': 2000,      
    'S1_BATCH_SIZE': 512,         
    'S1_GRAD_ACCUM': 2,
    'S1_POS_RATIO': 0.25,
    
    'S2_TOTAL_STEPS': 15000,       
    'S2_VAL_INTERVAL': 500,
    'S2_BATCH_SIZE': 256,
    'S2_GRAD_ACCUM': 1,
    'S2_POS_RATIO': 0.25,
    
    # === 评估参数 ===
    'MIN_PRECISION': 0.15,
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

def copy_to_local(src_path, local_rank, device_id=None):
    filename = os.path.basename(src_path)
    target_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    local_path = os.path.join(target_dir, filename)
    
    if local_rank == 0:
        need_copy = True
        if os.path.exists(local_path):
            if os.path.getsize(local_path) == os.path.getsize(src_path):
                need_copy = False
        
        if need_copy:
            print(f"[Node Local-0] Copying {filename} to RAM...", flush=True)
            try:
                tmp_path = local_path + ".tmp"
                shutil.copyfile(src_path, tmp_path)
                os.rename(tmp_path, local_path)
            except Exception as e:
                print(f"[Node Local-0] Copy FAILED: {e}. Fallback to NFS.", flush=True)
                local_path = src_path

    if dist.is_initialized():
        if device_id is not None:
            dist.barrier(device_ids=[device_id])
        else:
            dist.barrier()
        
    if not os.path.exists(local_path):
        return src_path
    return local_path

# ==========================================
# 2. 采样器
# ==========================================

class InfiniteBalancedSampler(Sampler):
    def __init__(self, dataset, batch_size, pos_ratio=0.25, rank=0, world_size=1, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        
        y = np.array(dataset.y_cls)
        # Class 0: Fog, Class 1: Mist, Class 2: Clear
        all_pos = np.where(y <= 1)[0]
        all_neg = np.where(y == 2)[0]
        
        np.random.seed(seed + rank)
        np.random.shuffle(all_pos)
        np.random.shuffle(all_neg)
        
        self.pos_indices = np.array_split(all_pos, world_size)[rank]
        self.neg_indices = np.array_split(all_neg, world_size)[rank]
        
        self.n_pos = int(batch_size * pos_ratio)
        self.n_neg = batch_size - self.n_pos
        
        if rank == 0:
            print(f"[Sampler] Total Pos (Fog/Mist): {len(all_pos)}, Total Neg: {len(all_neg)}")
            print(f"[Sampler] Batch Layout: {self.n_pos} Pos ({pos_ratio*100:.1f}%) + {self.n_neg} Neg")

    def __iter__(self):
        epoch_seed = self.seed + self.rank + int(time.time() * 1000) % 10000
        g = torch.Generator()
        g.manual_seed(epoch_seed)
        
        while True:
            # 循环采样，防止正样本耗尽
            pos_batch = torch.randint(0, len(self.pos_indices), (self.n_pos,), generator=g).numpy()
            neg_batch = torch.randint(0, len(self.neg_indices), (self.n_neg,), generator=g).numpy()
            
            indices = np.concatenate([self.pos_indices[pos_batch], self.neg_indices[neg_batch]])
            np.random.shuffle(indices)
            yield indices.tolist()

    def __len__(self):
        return 2147483647

# ==========================================
# 3. 改进的 Loss 函数
# ==========================================

class RecallFocusedLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=3.0, alpha=None, 
                 recall_weight=1.0, reduction='mean'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        
        if alpha is None:
            # Fog(0), Mist(1), Clear(2)
            self.alpha = torch.tensor([5.0, 2.0, 1.0])
        else:
            self.alpha = torch.tensor(alpha)
        
        self.recall_weight = recall_weight 
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.device != self.alpha.device:
            self.alpha = self.alpha.to(inputs.device)
        
        inputs = torch.clamp(inputs, min=-20, max=20)
            
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        
        focal_weight = torch.ones_like(targets, dtype=torch.float)
        
        pos_mask = targets <= 1
        focal_weight[pos_mask] = (1 - pt[pos_mask] + 1e-6) ** self.gamma_pos
        
        neg_mask = targets == 2
        focal_weight[neg_mask] = (1 - pt[neg_mask] + 1e-6) ** self.gamma_neg
        
        focal_loss = focal_weight * ce_loss
        
        if self.recall_weight > 1.0:
            probs = F.softmax(inputs, dim=1)
            pos_pred_as_neg = pos_mask & (probs[:, 2] > 0.5)
            if pos_pred_as_neg.sum() > 0:
                focal_loss[pos_pred_as_neg] *= self.recall_weight
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PhysicsConstrainedRegLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, reg_pred, reg_target_log, raw_vis):
        mask_low_vis = raw_vis < 2000
        if mask_low_vis.sum() == 0:
            return torch.tensor(0.0, device=reg_pred.device)
        
        reg_flat = reg_pred.view(-1)
        loss = F.huber_loss(reg_flat[mask_low_vis], reg_target_log[mask_low_vis], delta=1.0)
        return self.alpha * loss

# ==========================================
# 4. 改进的架构 (Complex MLP Variant)
# ==========================================

class ResidualMLPBlock(nn.Module):
    """
    带有残差连接和层归一化的 MLP 块
    结构: Input -> LN -> Linear(Expand) -> GELU -> Dropout -> Linear(Project) -> Dropout -> + Input
    """
    def __init__(self, hidden_dim, dropout=0.2, expansion_factor=2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 扩展维度以捕捉高维特征交互
        expanded_dim = int(hidden_dim * expansion_factor)
        
        self.fc1 = nn.Linear(hidden_dim, expanded_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(expanded_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        
        out = self.norm(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        
        return residual + out

class ComplexMLP(nn.Module):
    def __init__(self, dyn_vars_count=25, window_size=9, 
                 static_cont_dim=5, veg_num_classes=21, veg_emb_dim=16,
                 hidden_dim=1024, num_blocks=6, num_classes=3):
        super().__init__()
        
        self.dyn_vars = dyn_vars_count
        self.window = window_size
        self.static_cont_dim = static_cont_dim
        
        # 1. 计算总输入维度
        # 动态变量展开: vars * time_steps
        # 静态连续变量: 5
        # 植被嵌入: 16
        self.input_dim = (dyn_vars_count * window_size) + static_cont_dim + veg_emb_dim
        
        # 2. 植被 Embedding
        self.veg_embedding = nn.Embedding(veg_num_classes, veg_emb_dim)
        
        # 3. 输入投影层 (Input Projection)
        # 将原始特征映射到隐层维度
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 4. 深层残差 MLP 主干 (Backbone)
        # 堆叠多个 Residual Block 进行复杂的非线性特征提取
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout=0.2, expansion_factor=2)
            for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # 5. 分类头 (Classification Head)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 初始化 Bias 以应对类别不平衡
        # log(0.25/0.75) ≈ -1.1 (Fog, Mist), log(0.5/0.5) = 0 (Clear)
        self.cls_head[-1].bias.data = torch.tensor([-1.1, -1.1, 0.0])
        
        # 6. 回归头 (Regression Head)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: [Batch, (Dyn*Win) + Static + 1]
        
        # --- 数据切分与预处理 ---
        split_dyn = self.dyn_vars * self.window
        split_static = split_dyn + self.static_cont_dim
        
        x_dyn_flat = x[:, :split_dyn]          # [Batch, Dyn*Win]
        x_stat_cont = x[:, split_dyn:split_static] # [Batch, 5]
        x_veg_id = x[:, -1].long()             # [Batch]
        
        # --- Embedding ---
        veg_vec = self.veg_embedding(x_veg_id) # [Batch, 16]
        
        # --- 特征拼接 (Flatten All) ---
        # MLP 的核心思想：将时间、空间、静态变量视为一个大的特征向量
        x_all = torch.cat([x_dyn_flat, x_stat_cont, veg_vec], dim=1)
        
        # --- 网络前向传播 ---
        feature = self.input_proj(x_all)
        
        for block in self.blocks:
            feature = block(feature)
            
        feature = self.final_norm(feature)
        
        # --- 输出 ---
        logits = self.cls_head(feature)
        reg_out = self.reg_head(feature)
        
        return logits, reg_out

# ==========================================
# 5. 数据处理
# ==========================================

class PMSTDataset(Dataset):
    def __init__(self, X_path, y_cls, y_reg, y_raw, scaler=None, apply_log_transform=True, window_size=9):
        self.X = np.load(X_path, mmap_mode='r')
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.as_tensor(y_reg, dtype=torch.float32)
        self.y_raw = torch.as_tensor(y_raw, dtype=torch.float32)
        self.window_size = window_size
        
        self.has_scaler = scaler is not None
        if self.has_scaler:
            self.center = scaler.center_.astype(np.float32)
            self.scale = scaler.scale_.astype(np.float32)
            self.scale = np.where(self.scale == 0, 1.0, self.scale)
        
        self.apply_log_transform = apply_log_transform
        
        feat_dim = self.X.shape[1] - 1
        self.log_mask = np.zeros(feat_dim, dtype=bool)
        
        # === 修改: 循环次数依赖 window_size ===
        if apply_log_transform:
            for t in range(self.window_size):
                offset = t * 25
                indices = [offset + 2, offset + 4, offset + 9] 
                for idx in indices:
                    if idx < feat_dim:
                        self.log_mask[idx] = True
        
        self.clip_min = -10.0
        self.clip_max = 10.0

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        features = row[:-1].astype(np.float32)
        veg_id = row[-1]
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.apply_log_transform:
            np.maximum(features, 0, out=features, where=self.log_mask)
            np.log1p(features, out=features, where=self.log_mask)
        
        if self.has_scaler:
            features = (features - self.center) / self.scale
        
        features = np.clip(features, self.clip_min, self.clip_max)
        features = np.append(features, veg_id)
        
        return torch.from_numpy(features).float(), self.y_cls[idx], self.y_reg[idx], self.y_raw[idx]


def load_data_and_scale(data_dir, scaler=None, rank=0, device=None, reuse_scaler=False, window_size=9):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0: print(f"Loading Metadata from: {data_dir} (Window Size: {window_size})", flush=True)
    
    raw_train_path = os.path.join(data_dir, 'X_train.npy')
    raw_val_path = os.path.join(data_dir, 'X_val.npy')
    raw_y_train_path = os.path.join(data_dir, 'y_train.npy')
    raw_y_val_path = os.path.join(data_dir, 'y_val.npy')
    
    train_path = copy_to_local(raw_train_path, local_rank, device_id=local_rank)
    if rank == 0: print(f"  ✓ X_train loaded", flush=True)
    
    val_path = copy_to_local(raw_val_path, local_rank, device_id=local_rank)
    if rank == 0: print(f"  ✓ X_val loaded", flush=True)
    
    local_y_train = copy_to_local(raw_y_train_path, local_rank, device_id=local_rank)
    local_y_val = copy_to_local(raw_y_val_path, local_rank, device_id=local_rank)
    if rank == 0: print(f"  ✓ Labels loaded to local", flush=True)
    
    y_train_raw = np.load(local_y_train)
    y_val_raw = np.load(local_y_val)
    
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
            print(f"  [Rank 0] Fitting RobustScaler (on log-transformed subset)...", flush=True)
            scaler = RobustScaler()
            X_temp = np.load(train_path, mmap_mode='r')
            subset_size = min(300000, len(X_temp))
            indices = np.random.choice(len(X_temp), subset_size, replace=False)
            indices.sort()
            
            X_subset = X_temp[indices, :-1].copy()
            X_subset = np.nan_to_num(X_subset, nan=0.0)
            
            log_mask = np.zeros(X_subset.shape[1], dtype=bool)
            # === 修改: 循环次数依赖 window_size ===
            for t in range(window_size):
                offset = t * 25
                for idx in [offset + 2, offset + 4, offset + 9]:
                    if idx < X_subset.shape[1]: log_mask[idx] = True
            
            np.maximum(X_subset, 0, out=X_subset, where=log_mask)
            np.log1p(X_subset, out=X_subset, where=log_mask)
            
            scaler.fit(X_subset)
            print(f"  [Rank 0] Scaler fitted.", flush=True)
            del X_subset, X_temp
            
            center_tensor = torch.from_numpy(scaler.center_).float().to(device)
            scale_tensor = torch.from_numpy(scaler.scale_).float().to(device)
            dim_tensor = torch.tensor([len(scaler.center_)], device=device)
        else:
            dim_tensor = torch.tensor([0], device=device)
            center_tensor = None
            scale_tensor = None

        if dist.is_initialized():
            dist.broadcast(dim_tensor, src=0)
        
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
    
    train_ds = PMSTDataset(train_path, y_train_cls, y_train_log, y_train_m, scaler=scaler, window_size=window_size)
    val_ds = PMSTDataset(val_path, y_val_cls, y_val_log, y_val_m, scaler=scaler, window_size=window_size)
    
    return train_ds, val_ds, scaler

# ==========================================
# 6. 改进的评估函数
# ==========================================

def calculate_meteorological_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    ts = tp / (tp + fp + fn + 1e-6)
    
    num = 2 * (tp * tn - fp * fn)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = num / (den + 1e-6)
    
    hits_rand = (tp + fn) * (tp + fp) / (tp + fn + fp + tn)
    ets = (tp - hits_rand) / (tp + fn + fp - hits_rand + 1e-6)
    
    return ts, hss, ets

def evaluate_multi_metrics(model, loader, device, return_probs=False, min_precision=0.15):
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
    
    # 扫描阈值，关注正常范围
    thresholds = np.concatenate([
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
        tn, fp, fn, tp = confusion_matrix(y_true_c0, y_pred).ravel()
        
        rec = tp / (tp + fn + 1e-6)
        prec = tp / (tp + fp + 1e-6)
        
        f2 = 5 * (prec * rec) / (4 * prec + rec + 1e-6)
        
        candidate_results.append({
            'thresh': t,
            'ts': ts,
            'ets': ets,
            'hss': hss,
            'recall': rec,
            'precision': prec,
            'f2': f2
        })
    
    valid_candidates = [c for c in candidate_results if c['precision'] >= min_precision]
    
    if valid_candidates:
        best_res = max(valid_candidates, key=lambda x: x['f2'])
        selection_note = f"Selected by Max F2 (Prec>={min_precision:.0%})"
    else:
        if candidate_results:
            best_res = max(candidate_results, key=lambda x: x['ts'])
            selection_note = f"Fallback: Max TS (Precision < {min_precision:.0%})"
        else:
            best_res = {
                'thresh': 0.5, 'ts': 0.0, 'ets': 0.0, 'hss': 0.0,
                'recall': 0.0, 'precision': 0.0, 'f2': 0.0
            }
            selection_note = "No valid threshold found"
    
    result = {
        'score': best_res['ts'], 
        'ts_score': best_res['ts'],
        'ets_score': best_res['ets'],
        'hss_score': best_res['hss'],
        'recall_c0': best_res['recall'],
        'prec_c0': best_res['precision'],
        'f2_score': best_res['f2'],
        'best_thresh': best_res['thresh'],
        'selection_note': selection_note
    }
    
    if return_probs:
        result['probs'] = probs
        result['targets'] = all_targets
        result['candidates'] = sorted(candidate_results, key=lambda x: x['f2'], reverse=True)[:10]
    
    return result

# ==========================================
# 7. 训练流程
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
    
    best_ts = -1
    best_f2 = -1
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
        
        valid_step = True
        
        with autocast():
            logits, reg_pred = model(bx)
            l_cls = crit_cls(logits, by_cls)
            l_reg = crit_reg(reg_pred, by_log, by_raw)
            loss = l_cls + 0.1 * l_reg
            loss = loss / grad_accum
        
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0:
                print(f"\n[WARNING] Step {step}: Loss is NaN/Inf. Skipping batch.", flush=True)
            optimizer.zero_grad()
            valid_step = False
        
        if valid_step:
            scaler_amp.scale(loss).backward()
            
            avg_loss_cls += l_cls.item()
            avg_loss_reg += l_reg.item()
            
            if step % grad_accum == 0:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad()
        
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
            
            met = evaluate_multi_metrics(model, val_loader, device, return_probs=True,
                                        min_precision=CONFIG['MIN_PRECISION'])
            
            if rank == 0:
                print(f"  {met['selection_note']}")
                print(f"  Best Result: TS={met['ts_score']:.4f} | F2={met['f2_score']:.4f} | "
                      f"Recall={met['recall_c0']:.1%} | Precision={met['prec_c0']:.1%} | "
                      f"Thresh={met['best_thresh']:.4f}", flush=True)
                
                if 'candidates' in met:
                    print("  Top-5 F2 Candidates (with Prec > Min):")
                    for i, cand in enumerate(met['candidates'][:5], 1):
                        print(f"    {i}. Thresh={cand['thresh']:.4f} → "
                              f"F2={cand['f2']:.4f}, R={cand['recall']:.2%}, P={cand['precision']:.2%}, TS={cand['ts']:.4f}")
                
                # 保存时加上时间窗口后缀
                win_suffix = f"_{CONFIG['WINDOW_SIZE']}h_"
                
                if met['f2_score'] > best_f2:
                    best_f2 = met['f2_score']
                    save_name = f"mlp_{stage_name.lower()}{win_suffix}best_f2.pth"
                    save_path = os.path.join(CONFIG['BASE_PATH'], f"model/{save_name}")
                    torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), save_path)
                    print(f"  ✓ Best F2 Checkpoint: {save_path} (F2={best_f2:.4f})")
                
                if met['ts_score'] > best_ts:
                    best_ts = met['ts_score']
                    save_name = f"mlp_{stage_name.lower()}{win_suffix}best_ts.pth"
                    save_path = os.path.join(CONFIG['BASE_PATH'], f"model/{save_name}")
                    torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), save_path)
                    print(f"  ✓ Best TS Checkpoint: {save_path} (TS={best_ts:.4f})")
            
            model.train()
    
    if rank == 0:
        print(f"\n[{stage_name}] Training completed. Best TS: {best_ts:.4f}, Best F2: {best_f2:.4f}", flush=True)
    
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
    window_size = CONFIG['WINDOW_SIZE']
    
    if global_rank == 0:
        os.makedirs(os.path.join(base_path, "model"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "scalers"), exist_ok=True)
        print("="*60)
        print(f"  Visibility Forecasting - BALANCED PIPELINE (V3) - {window_size}h Window")
        print("="*60)
        print(f"  Data S1: {CONFIG['S1_DATA_DIR']}")
        print(f"  Data S2: {CONFIG['S2_DATA_DIR']}")
        print(f"  Key Adjustments:")
        print(f"    ✓ Window Size: {window_size}")
        print(f"    ✓ Sampling Ratio: 25% Pos")
        print("="*60)

    # Stage 1: ERA5 Pre-training
    if global_rank == 0:
        print("\n[STAGE 1] ERA5 Pre-training")
        print("-" * 60)
    
    s1_train_ds, s1_val_ds, scaler_s1 = load_data_and_scale(
        CONFIG['S1_DATA_DIR'], rank=global_rank, device=device, reuse_scaler=False, window_size=window_size
    )
    
    if global_rank == 0:
        scaler_name = f"scaler_mlp_{window_size}h.pkl"
        joblib.dump(scaler_s1, os.path.join(base_path, f"scalers/{scaler_name}"))
    
    s1_val_loader = DataLoader(
        s1_val_ds, 
        batch_size=CONFIG['S1_BATCH_SIZE'], 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    model = ComplexMLP(
        dyn_vars_count=25, 
        window_size=window_size, 
        hidden_dim=1024,   # MLP 通常需要比 RNN 更宽的隐层
        num_blocks=6,      # 深度
        num_classes=3
    ).to(device)
    
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    
    crit_cls = RecallFocusedLoss(
        gamma_pos=1.0,           
        gamma_neg=3.0, 
        alpha=[5.0, 2.0, 1.0],   
        recall_weight=1.0        
    )
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

    # Stage 2: Forecast Fine-tuning
    if global_rank == 0:
        print("\n[STAGE 2] Forecast Fine-tuning")
        print("-" * 60)
    
    s2_train_ds, s2_val_ds, _ = load_data_and_scale(
        CONFIG['S2_DATA_DIR'], 
        scaler=scaler_s1, 
        rank=global_rank, 
        device=device, 
        reuse_scaler=True,
        window_size=window_size
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
    
    train_unified_stream(
        'S2', model, s2_train_ds, s2_val_loader, s2_optimizer,
        (crit_cls, crit_reg), scaler_amp, device, global_rank, world_size,
        CONFIG['S2_TOTAL_STEPS'], CONFIG['S2_VAL_INTERVAL'],
        CONFIG['S2_BATCH_SIZE'], CONFIG['S2_GRAD_ACCUM'],
        CONFIG['S2_POS_RATIO']
    )
    
    if global_rank == 0:
        print("\n" + "="*60)
        print("  Training Pipeline Completed (V3 Balanced)")
        print("="*60)
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
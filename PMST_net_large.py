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
    
    'S2_TOTAL_STEPS': 10000,
    'S2_VAL_INTERVAL': 500,
    'S2_BATCH_SIZE': 256,
    'S2_GRAD_ACCUM': 1,
}

# ==========================================
# 1. 基础工具与分布式初始化 (保持不变)
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
# 2. 采样器 (保持不变，这部分逻辑是正确的)
# ==========================================

class InfiniteBalancedSampler(Sampler):
    def __init__(self, dataset, batch_size, pos_ratio=0.3, rank=0, world_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.rank = rank
        self.world_size = world_size
        
        y = np.array(dataset.y_cls)
        # Class 0: Fog (<500m), Class 1: Mist (500-1000m), Class 2: Clear
        all_pos = np.where(y <= 1)[0]
        all_neg = np.where(y == 2)[0]
        
        self.pos_indices = np.array_split(all_pos, world_size)[rank]
        self.neg_indices = np.array_split(all_neg, world_size)[rank]
        
        self.n_pos = int(batch_size * pos_ratio)
        self.n_neg = batch_size - self.n_pos
        
        if rank == 0:
            print(f"[Sampler] Total Pos (Fog/Mist): {len(all_pos)}, Total Neg: {len(all_neg)}")
            print(f"[Sampler] Batch Layout: {self.n_pos} Pos + {self.n_neg} Neg")

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(42 + self.rank + int(time.time() % 1000)) 
        
        while True:
            pos_batch = torch.randint(0, len(self.pos_indices), (self.n_pos,), generator=g).numpy()
            neg_batch = torch.randint(0, len(self.neg_indices), (self.n_neg,), generator=g).numpy()
            
            indices = np.concatenate([self.pos_indices[pos_batch], self.neg_indices[neg_batch]])
            np.random.shuffle(indices)
            yield indices.tolist()

    def __len__(self):
        return 2147483647

# ==========================================
# 3. New Loss Functions (核心修改：Focal + SoftF1)
# ==========================================

class FocalLoss(nn.Module):
    """
    Focal Loss: 专注于难分样本，减少简单负样本(Clear Sky)的权重影响
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            # 默认给 Class 0 (Fog) 最高的权重，Class 2 (Clear) 最低
            self.alpha = torch.tensor([5.0, 2.0, 1.0])
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.device != self.alpha.device:
            self.alpha = self.alpha.to(inputs.device)
            
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss) # p_t
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MacroSoftF1Loss(nn.Module):
    """
    直接优化 F1-Score 的 Loss。
    针对召回率低的问题，这个 Loss 会强迫模型去覆盖正样本。
    """
    def __init__(self, num_classes, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # logits: [B, C], targets: [B]
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # 计算每一类的 TP, FP, FN
        # dim=0 表示在 batch 维度上求和
        tp = (probs * target_onehot).sum(dim=0)
        fp = (probs * (1 - target_onehot)).sum(dim=0)
        fn = ((1 - probs) * target_onehot).sum(dim=0)
        
        # 计算 F1 (Soft)
        f1 = 2 * tp / (2 * tp + fp + fn + self.epsilon)
        
        # 我们主要关心 Class 0 (Fog) 和 Class 1 (Mist)
        # Class 2 (Clear) 的 F1 通常很高，不需要特意优化
        # 这里的权重 [0.7, 0.3, 0.0] 显性地告诉模型：Fog 的召回/精确最重要
        loss = 1 - (0.7 * f1[0] + 0.3 * f1[1]) 
        
        return loss

# ==========================================
# 4. Model Architecture (核心修改：Head & Shortcut)
# ==========================================

class ChebyKANLayer(nn.Module):
    # 保持不变，这是你原本的特征提取模块
    def __init__(self, input_dim, output_dim, degree=4):
        super(ChebyKANLayer, self).__init__()
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.xavier_normal_(self.cheby_coeffs)
        self.layer_norm = nn.LayerNorm(input_dim) 
        self.base_activation = nn.SiLU()

    def forward(self, x):
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
    # 保持不变
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

class StaticDynamicFusion(nn.Module):
    # 保持不变
    def __init__(self, dyn_dim, static_dim, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(dyn_dim, hidden_dim)
        self.kv_kan = ChebyKANLayer(static_dim, hidden_dim, degree=3)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_dyn, x_static):
        Q = self.q_proj(x_dyn)
        kv = self.kv_kan(x_static).unsqueeze(1)
        attn_out, _ = self.attn(query=Q, key=kv, value=kv)
        return self.norm(Q + self.dropout(attn_out))

class PMSTNet(nn.Module):
    def __init__(self, dyn_vars_count=24, window_size=9, 
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
            ChebyKANLayer(total_static_dim, 256, degree=4),
            nn.LayerNorm(256),
            nn.Linear(256, hidden_dim)
        )
        
        # 2. 动态时序编码 (TCN)
        self.dyn_embed = nn.Conv1d(dyn_vars_count, hidden_dim, 1)
        self.tcn = nn.Sequential(
            MultiScaleTCNBlock(hidden_dim, hidden_dim),
            MultiScaleTCNBlock(hidden_dim, hidden_dim)
        )
        
        # 3. 融合模块
        self.fusion_module = StaticDynamicFusion(hidden_dim, hidden_dim, hidden_dim)
        
        # 4. Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, 
                                                 dim_feedforward=1024, 
                                                 batch_first=True, dropout=0.2,
                                                 norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 5. [新增] 当前时刻特征强化模块
        # 我们提取 T=0 (最新时刻) 的特征，通过一个小的 MLP 编码，作为 shortcut
        self.current_state_encoder = nn.Sequential(
            nn.Linear(dyn_vars_count, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim)
        )
        
        # 6. 后处理 KAN
        # 输入维度翻倍，因为我们要把 (时序特征 + 当前强化特征) 结合
        self.post_fusion_kan = ChebyKANLayer(hidden_dim * 2, hidden_dim, degree=3)
        
        # 7. [修改] 回归简单的 Linear Head
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        # 初始化 Bias，让模型初始预测稍微偏向 Clear，避免初始 Loss 过大
        # 假设 0:Fog, 1:Mist, 2:Clear. 给 Class 2 一个大的 bias
        self.cls_head.bias.data = torch.tensor([-1.0, -1.0, 1.0]) 

        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        split_dyn = self.dyn_vars * self.window 
        split_static = split_dyn + self.static_cont_dim 
        
        # A. 数据切分
        x_dyn_flat = x[:, :split_dyn]
        x_stat_cont = x[:, split_dyn:split_static]
        x_veg_id = x[:, -1].long()
        
        # B. 静态特征处理
        veg_vec = self.veg_embedding(x_veg_id) 
        x_static_full = torch.cat([x_stat_cont, veg_vec], dim=1) 
        static_feat = self.static_encoder(x_static_full)
        
        # C. 动态时序处理
        # [Batch, Window, Vars] -> [Batch, Vars, Window] (for Conv1d)
        x_dyn_seq_raw = x_dyn_flat.view(-1, self.window, self.dyn_vars)
        
        # [新增] 提取当前时刻 (假设时序是 t-8, ..., t-0，取最后一个)
        x_current_raw = x_dyn_seq_raw[:, -1, :] 
        current_feat = self.current_state_encoder(x_current_raw)
        
        x_dyn_seq = x_dyn_seq_raw.permute(0, 2, 1)
        dyn_feat = self.dyn_embed(x_dyn_seq) 
        tcn_out = self.tcn(dyn_feat) # [Batch, Hidden, Window]
        
        # D. 融合
        tcn_seq = tcn_out.permute(0, 2, 1) # [Batch, Window, Hidden]
        fused_seq = self.fusion_module(tcn_seq, static_feat)
        
        # E. Transformer
        trans_out = self.transformer(fused_seq)
        trans_pool = trans_out[:, -1, :] # 取最后一个时间步的 Transformer 输出
        
        # F. [核心修改] 特征强结合
        # 将时序提取的特征与当前物理状态特征直接相加或拼接
        # 这里选择拼接后通过 KAN 融合
        combined_feat = torch.cat([trans_pool, current_feat], dim=1)
        embedding = self.post_fusion_kan(combined_feat)
        
        # G. 输出
        logits = self.cls_head(embedding) 
        reg_out = self.reg_head(embedding)
            
        return logits, reg_out

# ==========================================
# 5. Data & Processing (保持不变)
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

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx] 
        features = row[:-1].astype(np.float32)
        veg_id = row[-1]
        features = np.nan_to_num(features, nan=0.0)
        
        if self.has_scaler:
            features = (features - self.center) / self.scale
            
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
# 6. Evaluation Logic (保持不变)
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

def evaluate_multi_metrics(model, loader, device):
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
    
    # 扩大搜索范围，降低起始阈值
    thresholds = np.arange(0.10, 0.90, 0.05)
    
    for t in thresholds:
        y_pred = (prob_c0 > t).astype(int)
        if y_pred.sum() == 0: continue
        
        ts, hss, ets = calculate_meteorological_metrics(y_true_c0, y_pred)
        
        if ts > best_ts:
            best_ts = ts
            rec = recall_score(y_true_c0, y_pred)
            prec = precision_score(y_true_c0, y_pred)
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
        best_res = {'score': 0.0, 'ts_score': 0.0, 'ets_score': 0.0, 'hss_score': 0.0, 'recall_c0': 0.0, 'prec_c0': 0.0, 'best_thresh': 0.5}

    return best_res

# ==========================================
# 7. Training Loops (Unified Stream)
# ==========================================

def train_unified_stream(stage_name, model, train_ds, val_loader, optimizer, 
                        criterions, scaler_amp, device, rank, world_size,
                        total_steps, val_interval, batch_size, grad_accum):
    
    # 保持正样本比例在 25% - 30% 之间，让 Focal Loss 发挥作用
    pos_ratio = 0.25 
    
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
    
    # 解包新的 Loss
    crit_focal, crit_f1 = criterions
    
    best_score = -1
    avg_loss = 0
    start_time = time.time()
    
    optimizer.zero_grad()
    
    for step in range(1, total_steps + 1):
        try:
            bx, by_cls, by_log, by_raw = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            bx, by_cls, by_log, by_raw = next(train_iter)
        
        bx, by_cls = bx.to(device, non_blocking=True), by_cls.to(device, non_blocking=True)
        by_log, by_raw = by_log.to(device, non_blocking=True), by_raw.to(device, non_blocking=True)
        
        with autocast():
            logits, reg_pred = model(bx)
            
            # --- New Loss Calculation Strategy ---
            # 1. Focal Loss (处理 Imbalance, 挖掘 Hard Example)
            l_focal = crit_focal(logits, by_cls)
            
            # 2. Soft F1 Loss (直接优化 Recall/Precision)
            l_f1 = crit_f1(logits, by_cls)
            
            # 3. Regression Loss (辅助物理约束)
            reg_flat = reg_pred.view(-1)
            mask_fog = by_raw < 2000 
            if mask_fog.sum() > 0:
                l_reg = F.huber_loss(reg_flat[mask_fog], by_log[mask_fog])
            else:
                l_reg = torch.tensor(0.0, device=device)
            
            # 总 Loss: 强调用 F1 和 Focal
            # 权重建议: Focal 和 F1 是主导，Reg 是辅助
            loss = l_focal + 0.5 * l_f1 + 0.1 * l_reg
            
            loss = loss / grad_accum
            
        scaler_amp.scale(loss).backward()
        avg_loss += loss.item() * grad_accum
        
        if step % grad_accum == 0:
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            optimizer.zero_grad()
        
        if rank == 0 and step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"\r[{stage_name} Steps] {step}/{total_steps} | Loss: {avg_loss/100:.4f} | Speed: {100/elapsed:.1f} steps/s", end="", flush=True)
            avg_loss = 0
            start_time = time.time()
            
        if step % val_interval == 0:
            if rank == 0: print(f"\n[{stage_name} Validation @ {step}] Eval...", flush=True)
            met = evaluate_multi_metrics(model, val_loader, device)
            
            if rank == 0:
                print(f"  Result: TS: {met['ts_score']:.4f} | ETS: {met['ets_score']:.4f} | HSS: {met['hss_score']:.4f} | R: {met['recall_c0']:.1%} | P: {met['prec_c0']:.1%}", flush=True)
                
                if met['score'] > best_score:
                    best_score = met['score']
                    save_name = f"pmst_stage1_best_large.pth" if stage_name == 'S1' else f"pmst_stage2_final_large.pth"
                    save_path = os.path.join(CONFIG['BASE_PATH'], f"model/{save_name}")
                    torch.save(model.module.state_dict(), save_path)
                    print(f"  [Checkpoint] Saved Best to {save_path}")
            
            model.train()
    
    if rank == 0: print(f"\n[{stage_name}] Finished.", flush=True)
    del train_iter, train_loader
    if dist.is_initialized(): dist.barrier()

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

    # -------------------------------------------------------
    # Stage 1: ERA5 Pre-training (High Recall Focus)
    # -------------------------------------------------------
    if global_rank == 0: print("\n=== Stage 1: ERA5 Pre-training ===", flush=True)
    
    s1_train_ds, s1_val_ds, scaler_s1 = load_data_and_scale(
        CONFIG['S1_DATA_DIR'], rank=global_rank, device=device, reuse_scaler=False
    )
    
    if global_rank == 0:
        joblib.dump(scaler_s1, os.path.join(base_path, "scalers/scaler_pmst_large.pkl"))
    
    s1_val_loader = DataLoader(s1_val_ds, batch_size=CONFIG['S1_BATCH_SIZE'], shuffle=False, 
                              num_workers=4, pin_memory=True)
    
    model = PMSTNet(dyn_vars_count=24, hidden_dim=512, num_classes=3).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    
    # 损失函数组合：Focal (专注难例) + SoftF1 (专注指标)
    crit_focal = FocalLoss(alpha=[5.0, 2.0, 1.0], gamma=2.0)
    crit_f1 = MacroSoftF1Loss(num_classes=3)
    
    train_unified_stream('S1', model, s1_train_ds, s1_val_loader, optimizer, 
                        (crit_focal, crit_f1), 
                        scaler_amp, device, global_rank, world_size,
                        CONFIG['S1_TOTAL_STEPS'], CONFIG['S1_VAL_INTERVAL'], 
                        CONFIG['S1_BATCH_SIZE'], CONFIG['S1_GRAD_ACCUM'])
    
    del s1_train_ds, s1_val_ds, s1_val_loader
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------
    # Stage 2: Forecast Fine-tuning
    # -------------------------------------------------------
    if global_rank == 0: print("\n=== Stage 2: Forecast Fine-tuning ===", flush=True)
    
    s2_train_ds, s2_val_ds, _ = load_data_and_scale(
        CONFIG['S2_DATA_DIR'], 
        scaler=scaler_s1,  
        rank=global_rank, 
        device=device, 
        reuse_scaler=True
    )
    
    s2_val_loader = DataLoader(s2_val_ds, batch_size=CONFIG['S2_BATCH_SIZE'], shuffle=False, num_workers=4)
    
    params = [
        {'params': [p for n, p in model.named_parameters() if "head" not in n], 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if "head" in n], 'lr': 5e-5}
    ]
    s2_optimizer = optim.AdamW(params, weight_decay=1e-2)
    
    # Stage 2 继续使用 Focal + F1，因为它们对长尾问题同样有效
    # 相比 LDAM，这套组合在微调阶段更稳定，不需要调整复杂的 margin schedule
    train_unified_stream('S2', model, s2_train_ds, s2_val_loader, s2_optimizer,
                        (crit_focal, crit_f1),
                        scaler_amp, device, global_rank, world_size,
                        CONFIG['S2_TOTAL_STEPS'], CONFIG['S2_VAL_INTERVAL'],
                        CONFIG['S2_BATCH_SIZE'], CONFIG['S2_GRAD_ACCUM'])
                          
    if world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    main()
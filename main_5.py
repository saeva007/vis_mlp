import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import os
import shutil
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
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
# 1. Basic Tools & Distributed Init
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
                timeout=datetime.timedelta(minutes=60) 
            )
    
    return local_rank, global_rank, world_size


def copy_to_local(src_path, local_rank):
    filename = os.path.basename(src_path)
    target_dir = "/tmp" 
    local_path = os.path.join(target_dir, filename)
    lock_path = local_path + ".lock"

    if local_rank == 0:
        need_copy = True
        if os.path.exists(local_path):
            if os.path.getsize(local_path) == os.path.getsize(src_path):
                need_copy = False
                print(f"[Node Local-0] Found valid cache: {local_path}, skipping copy.", flush=True)

        if need_copy:
            time.sleep(random.randint(0, 5))
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
        dist.barrier(device_ids=[local_rank])
        
    if not os.path.exists(local_path):
        return src_path
        
    return local_path

# ==========================================
# 2. Sampler & Loss
# ==========================================

class FastDistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, alpha=0.7):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() else 0
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        
        targets = np.array(dataset.y_cls)
        
        self.cls_indices = {}
        classes = np.unique(targets)
        for c in classes:
            self.cls_indices[c] = np.where(targets == c)[0]
            
        class_counts = np.bincount(targets, minlength=3)
        class_counts = np.where(class_counts == 0, 1, class_counts)
        
        # alpha controls resampling strength
        weights = 1.0 / (np.power(class_counts, alpha))
        
        total_samples = len(dataset)
        weight_sum = np.sum(weights * class_counts)
        prob_per_class = (weights * class_counts) / weight_sum
        
        self.samples_per_class = (prob_per_class * total_samples).astype(int)
        self.num_samples = int(math.ceil(total_samples / self.num_replicas))

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.rank * 1000)
        
        indices_local = []
        for c, indices in self.cls_indices.items():
            local_count_per_cls = int(self.samples_per_class[c] / self.num_replicas)
            
            if local_count_per_cls > 0:
                if len(indices) < local_count_per_cls:
                    rand_idx = torch.randint(0, len(indices), (local_count_per_cls,), generator=g).numpy()
                else:
                    rand_idx = torch.randperm(len(indices), generator=g)[:local_count_per_cls].numpy()
                indices_local.append(indices[rand_idx])
        
        if len(indices_local) > 0:
            indices_local = np.concatenate(indices_local)
        else:
            indices_local = np.array([], dtype=np.int64)

        np.random.seed(self.epoch + self.rank * 1000)
        np.random.shuffle(indices_local)
        
        if len(indices_local) < self.num_samples:
            diff = self.num_samples - len(indices_local)
            if len(indices_local) > 0:
                padding = np.random.choice(indices_local, diff)
                indices_local = np.concatenate([indices_local, padding])
            else:
                indices_local = np.zeros((self.num_samples,), dtype=np.int64)
        else:
            indices_local = indices_local[:self.num_samples]
            
        return iter(indices_local.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, device='cuda'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list).to(device)
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.float32)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index.bool(), x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

def get_cb_weights(cls_num_list, beta=0.9999):
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    return torch.FloatTensor(per_cls_weights)

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1 
        
        loss = - (mask * log_prob).sum(1) / sum_mask
        return loss.mean()

# ==========================================
# 3. Model Architecture
# ==========================================

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=4):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outputdim = output_dim
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

class CosinePrototypeHead(nn.Module):
    def __init__(self, in_dim, num_classes, init_scale=16.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.FloatTensor(num_classes, in_dim))
        nn.init.normal_(self.prototypes, mean=0, std=0.01)
        self.scale = nn.Parameter(torch.tensor(init_scale))
        
    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
        w_norm = F.normalize(self.prototypes, p=2, dim=1, eps=1e-8)
        cosine_sim = F.linear(x_norm, w_norm)
        scale_clamped = torch.clamp(self.scale, min=1.0, max=30.0) 
        return cosine_sim * scale_clamped

class MultiScaleTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5]):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = (k - 1) // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=pad),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(0.2)
            ))
        self.proj = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, 1)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        return self.proj(torch.cat(outs, dim=1))

class StaticDynamicFusion(nn.Module):
    def __init__(self, dyn_dim, static_dim, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(dyn_dim, hidden_dim)
        self.kv_kan = ChebyKANLayer(static_dim, hidden_dim, degree=3)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_dyn, x_static):
        Q = self.q_proj(x_dyn)
        kv = self.kv_kan(x_static).unsqueeze(1)
        attn_out, _ = self.attn(query=Q, key=kv, value=kv)
        return self.norm(Q + self.dropout(attn_out))

class PMSTNet(nn.Module):
    def __init__(self, dyn_vars_count=24, window_size=9, 
                 static_cont_dim=5, veg_num_classes=21, veg_emb_dim=8,
                 hidden_dim=256, num_classes=3):
        super().__init__()
        self.dyn_vars = dyn_vars_count
        self.window = window_size
        self.static_cont_dim = static_cont_dim
        
        self.veg_embedding = nn.Embedding(veg_num_classes, veg_emb_dim)
        total_static_dim = static_cont_dim + veg_emb_dim
        
        self.static_encoder = nn.Sequential(
            ChebyKANLayer(total_static_dim, 128, degree=4),
            nn.LayerNorm(128),
            nn.Linear(128, hidden_dim)
        )
        
        self.dyn_embed = nn.Conv1d(dyn_vars_count, hidden_dim, 1)
        self.tcn = nn.Sequential(
            MultiScaleTCNBlock(hidden_dim, hidden_dim),
            MultiScaleTCNBlock(hidden_dim, hidden_dim)
        )
        
        self.fusion_module = StaticDynamicFusion(hidden_dim, hidden_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, 
                                                 dim_feedforward=512, 
                                                 batch_first=True, dropout=0.2,
                                                 norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.post_fusion_kan = ChebyKANLayer(hidden_dim * 2, hidden_dim, degree=3)
        
        self.cls_head = CosinePrototypeHead(hidden_dim, num_classes, init_scale=16.0)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.proj_head = nn.Linear(hidden_dim, 64)

    def forward(self, x, return_proj=True):
        split_dyn = self.dyn_vars * self.window 
        split_static = split_dyn + self.static_cont_dim 
        
        x_dyn_flat = x[:, :split_dyn]
        x_stat_cont = x[:, split_dyn:split_static]
        x_veg_id = x[:, -1].long()
        
        veg_vec = self.veg_embedding(x_veg_id) 
        x_static_full = torch.cat([x_stat_cont, veg_vec], dim=1) 
        static_feat = self.static_encoder(x_static_full)
        
        x_dyn_seq = x_dyn_flat.view(-1, self.window, self.dyn_vars).permute(0, 2, 1)
        dyn_feat = self.dyn_embed(x_dyn_seq) 
        tcn_out = self.tcn(dyn_feat)
        tcn_pool = F.adaptive_max_pool1d(tcn_out, 1).squeeze(-1)
        
        tcn_seq = tcn_out.permute(0, 2, 1)
        fused_seq = self.fusion_module(tcn_seq, static_feat)
        
        trans_out = self.transformer(fused_seq)
        trans_pool = trans_out[:, -1, :]
        
        combined = torch.cat([tcn_pool, trans_pool], dim=1)
        embedding = self.post_fusion_kan(combined)
        
        logits = self.cls_head(embedding) 
        reg_out = self.reg_head(embedding)
        
        proj_feat = None
        if return_proj:
            proj_feat = F.normalize(self.proj_head(embedding), dim=1)
            
        return logits, reg_out, proj_feat

# ==========================================
# 4. Data Processing
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

def load_data_and_scale(data_dir, suffix, scaler=None, rank=0, device=None):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if rank == 0:
        print(f"Loading Metadata: {data_dir} (Suffix: {suffix})", flush=True)
    
    raw_train_path = os.path.join(data_dir, f'X_train{suffix}.npy')
    raw_val_path = os.path.join(data_dir, f'X_val{suffix}.npy')
    
    train_path = copy_to_local(raw_train_path, local_rank)
    val_path = copy_to_local(raw_val_path, local_rank)
    
    y_train_raw = np.load(os.path.join(data_dir, f'y_train{suffix}.npy'))
    y_val_raw = np.load(os.path.join(data_dir, f'y_val{suffix}.npy'))
    
    y_train_m = y_train_raw * 1000.0
    y_val_m = y_val_raw * 1000.0
    
    def to_class(y_m):
        cls = np.zeros_like(y_m, dtype=np.int64)
        cls[y_m >= 500] = 1
        cls[y_m >= 1000] = 2
        return cls
        
    y_train_cls = to_class(y_train_m)
    y_val_cls = to_class(y_val_m)
    y_train_log = np.log1p(y_train_m).astype(np.float32)
    y_val_log = np.log1p(y_val_m).astype(np.float32)
    
    if scaler is None:
        if rank == 0:
            print("  [Rank 0] Fitting RobustScaler (local mmap subset)...", flush=True)
            scaler = RobustScaler()
            X_temp = np.load(train_path, mmap_mode='r')
            subset_size = min(200000, len(X_temp))
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
            
    cnt = Counter(y_train_cls)
    cls_num_list = [cnt.get(0, 0) + 1, cnt.get(1, 0) + 1, cnt.get(2, 0) + 1]
    if rank == 0: print(f"  Class Dist: {cnt}", flush=True)
    
    train_ds = PMSTDataset(train_path, y_train_cls, y_train_log, y_train_m, scaler=scaler)
    val_ds = PMSTDataset(val_path, y_val_cls, y_val_log, y_val_m, scaler=scaler)
    
    return train_ds, val_ds, cls_num_list, scaler

# ==========================================
# 5. Training Loop
# ==========================================

def train_one_epoch(model, loader, optimizer, criterions, device, epoch, scaler_amp, rank=0):
    model.train()
    crit_cls, crit_reg, crit_con = criterions
    avg_loss = 0
    start_time = time.time()
    
    for i, (bx, by_cls, by_log, by_raw) in enumerate(loader):
        bx = bx.to(device, non_blocking=True)
        by_cls = by_cls.to(device, non_blocking=True)
        by_log = by_log.to(device, non_blocking=True)
        by_raw = by_raw.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            logits, reg_pred, proj_feat = model(bx, return_proj=True)
            
            l_cls = crit_cls(logits, by_cls)
            
            reg_flat = reg_pred.view(-1)
            mask_fog = by_raw < 2000 
            if mask_fog.sum() > 0:
                l_reg = F.huber_loss(reg_flat[mask_fog], by_log[mask_fog])
            else:
                l_reg = reg_pred.sum() * 0.0
                
            l_con = crit_con(proj_feat, by_cls)
            
            loss = l_cls + 0.1 * l_reg + 0.1 * l_con
        
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        
        avg_loss += loss.item()
        
        if rank == 0 and i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"\r  [Ep{epoch}] Step {i}/{len(loader)} Loss: {loss.item():.4f} ({i/elapsed:.1f} it/s)", end="", flush=True)
            
    if rank == 0: print("", flush=True)
    return avg_loss / len(loader), 0

# ==========================================
# 6. Evaluation Logic (TS/CSI Optimized)
# ==========================================

def evaluate_multi_metrics(model, loader, device):
    """
    Revised evaluation logic: Prioritize TS (CSI) Score.
    This is the "Gold Standard" for meteorological events (like Fog).
    """
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for bx, by_cls, _, _ in loader:
            bx = bx.to(device, non_blocking=True)
            with autocast():
                logits, _, _ = model(bx, return_proj=False)
            all_logits.append(logits.float().cpu()) 
            all_targets.append(by_cls)
            
    all_logits = torch.cat(all_logits)
    probs = F.softmax(all_logits, dim=1).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    # Class 0 (Fog) Analysis
    prob_c0 = probs[:, 0]
    y_true_c0 = (all_targets == 0).astype(int)
    
    best_ts = 0.0
    best_thresh = 0.5
    final_metrics = {}
    
    # Search range: 0.20 to 0.95 (Meteorology often accepts lower prob if CSI is high)
    thresholds = np.arange(0.20, 0.96, 0.02)
    
    for t in thresholds:
        y_pred = (prob_c0 > t).astype(int)
        
        if y_pred.sum() == 0: continue
        
        # Calculate confusion matrix elements for Class 0
        tp = ((y_pred == 1) & (y_true_c0 == 1)).sum()
        fp = ((y_pred == 1) & (y_true_c0 == 0)).sum()
        fn = ((y_pred == 0) & (y_true_c0 == 1)).sum()
        
        # TS (CSI) Formula
        denom = tp + fp + fn
        if denom == 0:
            ts = 0.0
        else:
            ts = tp / denom
            
        # Safety guard: Precision shouldn't be abysmal
        prec = tp / (tp + fp + 1e-6)
        if prec < 0.15: 
            ts = 0.0
            
        if ts > best_ts:
            best_ts = ts
            best_thresh = t
            
            # Calculate auxiliary metrics for logging
            rec = tp / (tp + fn + 1e-6)
            f1 = 2 * rec * prec / (rec + prec + 1e-6)
            
            final_metrics = {
                'score': ts,      # Use TS as the main score
                'ts_score': ts,
                'f1_score': f1,
                'recall_c0': rec,
                'prec_c0': prec,
            }

    # Fallback
    if best_ts == 0.0:
        final_metrics = {'score': 0.0, 'ts_score': 0.0, 'f1_score': 0.0, 'recall_c0': 0.0, 'prec_c0': 0.0}
        best_thresh = 0.5

    final_metrics['best_thresh'] = best_thresh
    return final_metrics

def save_checkpoint(model, path, metrics, threshold):
    to_save = {
        'state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'metrics': metrics,
        'threshold': threshold
    }
    torch.save(to_save, path)
    print(f"  [Checkpoint] Saved to {path} (Thresh={threshold:.2f}, TS={metrics['ts_score']:.4f})")

# ==========================================
# 7. Main Execution (Complete Workflow)
# ==========================================

def main():
    # --- 新增：解决 MIOpen 缓存冲突问题 ---
    # 获取本地 rank，如果还没初始化 DDP，通常从环境变量获取
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 1. 设置 MIOpen 缓存目录到 /tmp 或 /dev/shm (内存盘)，并按 rank 隔离
    # 注意：确保使用了绝对路径
    miopen_cache_dir = f"/tmp/miopen_cache_rank_{local_rank}"
    os.makedirs(miopen_cache_dir, exist_ok=True)
    os.environ["MIOPEN_USER_DB_PATH"] = miopen_cache_dir
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = miopen_cache_dir

    # 2. (可选) 如果上述无效，可以尝试禁用 MIOpen 编译缓存（会增加启动时的编译时间）
    # os.environ["MIOPEN_DISABLE_CACHE"] = "1"
    
    # 3. 增加 NCCL 容错性
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_IB_TIMEOUT"] = "22"  # 增加 InfiniBand 超时时间
    # ------------------------------------

    local_rank, global_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    scaler_amp = GradScaler()
    
    base_path = "/public/home/putianshu/vis_mlp"
    
    if global_rank == 0:
        os.makedirs(os.path.join(base_path, "model"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "scalers"), exist_ok=True)
        print(f"Distributed Init: Rank {global_rank}, World {world_size}", flush=True)

    if world_size > 1: dist.barrier()

    # ==========================
    # Stage 1: Pre-training
    # ==========================
    if global_rank == 0: print("\n=== Stage 1: Pre-training ===", flush=True)
    
    # 1.1 Data Load
    suffix_s1 = "_9h_pmst_v2" 
    train_ds, val_ds, cls_cnts, scaler = load_data_and_scale(
        os.path.join(base_path, 'data/pmst_preprocessed'), suffix_s1, 
        rank=global_rank, device=device
    )
    
    if global_rank == 0:
        joblib.dump(scaler, os.path.join(base_path, f"scalers/pmst_scaler_v4.pkl"))

    if world_size > 1: dist.barrier()

    # 1.2 Setup (Sampler Alpha = 0.6)
    sampler = FastDistributedWeightedSampler(train_ds, num_replicas=world_size, rank=local_rank, alpha=0.6)
    train_loader = DataLoader(train_ds, batch_size=256, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=3)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=3)
    
    model = PMSTNet(dyn_vars_count=24, window_size=9, static_cont_dim=5, 
                   veg_num_classes=21, hidden_dim=256).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    
    cls_weights = get_cb_weights(cls_cnts).to(device)
    crit_cls = LDAMLoss(cls_cnts, weight=cls_weights, device=device)
    crit_reg = nn.HuberLoss()
    crit_con = SupervisedContrastiveLoss()
    
    s1_ckpt_path = os.path.join(base_path, "model/pmst_stage1_best_v4.pth")
    run_stage1 = True
    
    if os.path.exists(s1_ckpt_path):
        if global_rank == 0: print(f"Found Stage 1 Checkpoint, Loading...", flush=True)
        ckpt = torch.load(s1_ckpt_path, map_location=device)
        if 'state_dict' in ckpt: model.module.load_state_dict(ckpt['state_dict']) if world_size > 1 else model.load_state_dict(ckpt['state_dict'])
        else: model.module.load_state_dict(ckpt) if world_size > 1 else model.load_state_dict(ckpt)
        run_stage1 = False 
        
    # 1.3 Train S1
    if run_stage1:
        best_score_s1 = -1
        for epoch in range(200): 
            sampler.set_epoch(epoch)
            loss, _ = train_one_epoch(model, train_loader, optimizer, (crit_cls, crit_reg, crit_con), 
                                      device, epoch, scaler_amp, rank=global_rank)
            
            if global_rank == 0 and epoch % 1 == 0:
                met = evaluate_multi_metrics(model, val_loader, device)
                # Print TS as main metric
                print(f"[S1 Ep{epoch}] TS(CSI): {met['ts_score']:.4f} | R: {met['recall_c0']:.1%} P: {met['prec_c0']:.1%}", flush=True)
                
                if met['score'] > best_score_s1:
                    best_score_s1 = met['score']
                    save_checkpoint(model, s1_ckpt_path, met, threshold=met['best_thresh'])

    # 1.4 Safe Transition (Avoid Task Exit)
    if world_size > 1: dist.barrier()
    if global_rank == 0: print("Transitioning to Stage 2 (Cleanly)...", flush=True)
    
    # Clear memory but do NOT destroy process group
    del train_loader, val_loader, sampler, train_ds, val_ds
    gc.collect()
    
    # Save Stage 1 weights to CPU dict for Stage 2 init
    if world_size > 1:
        s1_state_dict = {k: v.cpu() for k, v in model.module.state_dict().items()}
    else:
        s1_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        
    # ==========================
    # Stage 2: Fine-tuning
    # ==========================
    if global_rank == 0: print("\n=== Stage 2: Fine-tuning (Airport) ===", flush=True)
    
    suffix_s2 = "_9h_airport_pmst_v2"
    
    scaler_path = os.path.join(base_path, f"scalers/pmst_scaler_v4.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    ft_train_ds, ft_val_ds, ft_cnts, _ = load_data_and_scale(
        os.path.join(base_path, 'data/airport_pmst_processed'), suffix_s2, 
        scaler=scaler, rank=global_rank, device=device
    )
    
    # [STRATEGY] Alpha 0.3 (Milder balancing)
    ft_sampler = FastDistributedWeightedSampler(ft_train_ds, num_replicas=world_size, rank=local_rank, alpha=0.3)
    
    ft_loader = DataLoader(ft_train_ds, batch_size=128, sampler=ft_sampler, num_workers=4, pin_memory=True, prefetch_factor=3)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=3)
    
    # Re-build Model
    model = PMSTNet(dyn_vars_count=24, window_size=9, static_cont_dim=5, 
                   veg_num_classes=21, hidden_dim=256).to(device)
    
    # Load Backbone Weights
    model.load_state_dict(s1_state_dict, strict=False)
    
    # [STRATEGY] Reset Heads
    if global_rank == 0: print("  [Stage 2] Resetting Heads for Airport adaptation...", flush=True)
    model.cls_head = CosinePrototypeHead(256, 3, init_scale=16.0).to(device)
    model.reg_head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        
    # Setup Phase 1: Freeze Backbone
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True
            head_params.append(param)
        else:
            param.requires_grad = False
            backbone_params.append(param)
            
    if global_rank == 0: print("  [Phase 1] Training Heads Only (Warmup)...", flush=True)
    ft_optim = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-2)
    ft_cls_w = get_cb_weights(ft_cnts, beta=0.99).to(device)
    ft_crit = LDAMLoss(ft_cnts, weight=ft_cls_w, device=device)
    
    for epoch in range(20):
        ft_sampler.set_epoch(epoch)
        train_one_epoch(model, ft_loader, ft_optim, (ft_crit, crit_reg, crit_con), device, epoch, scaler_amp, rank=global_rank)
        
    # Setup Phase 2: Unfreeze Backbone (Low LR)
    if global_rank == 0: print("  [Phase 2] Unfreezing Backbone (Low LR Fine-tuning)...", flush=True)
    
    for param in model.parameters():
        param.requires_grad = True
        
    param_groups = [
        {'params': backbone_params, 'lr': 1e-5}, # Very low LR for backbone
        {'params': head_params, 'lr': 5e-4}      # Normal LR for heads
    ]
    ft_optim = optim.AdamW(param_groups, weight_decay=1e-2)
    
    best_s2_score = -1
    patience = 0
    
    for epoch in range(100):
        real_epoch = epoch + 20
        ft_sampler.set_epoch(real_epoch)
        train_one_epoch(model, ft_loader, ft_optim, (ft_crit, crit_reg, crit_con), device, real_epoch, scaler_amp, rank=global_rank)
        
        if epoch % 1 == 0 and global_rank == 0:
            met = evaluate_multi_metrics(model, ft_val_loader, device)
            
            # Log TS (CSI) prominently
            print(f"[S2 Ep{real_epoch}] TS(CSI): {met['ts_score']:.4f} | F1: {met['f1_score']:.4f} | R: {met['recall_c0']:.1%} P: {met['prec_c0']:.1%} | Thresh: {met['best_thresh']:.2f}", flush=True)
            
            if met['score'] > best_s2_score:
                best_s2_score = met['score']
                save_checkpoint(model, os.path.join(base_path, "model/pmst_airport_best_v5.pth"), met, threshold=met['best_thresh'])
                patience = 0
            else:
                patience += 1
                
            if patience > 25:
                print("Early stopping triggered due to no TS improvement.")
                break

    if world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    main()
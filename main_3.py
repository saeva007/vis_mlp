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
    
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl", 
                init_method='env://', 
                timeout=datetime.timedelta(minutes=30) 
            )
        torch.cuda.set_device(local_rank)
    
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
            time.sleep(random.randint(0, 30))
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
# 2. Optimized Sampler (Alpha=0.6)
# ==========================================

class FastDistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, alpha=0.6):
        """
        alpha=0.6: 针对 1:400 分布，采样比例约为 1:15，保留足够负样本以保证 Precision。
        """
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
                rand_idx = torch.randint(0, len(indices), (local_count_per_cls,), generator=g).numpy()
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

class FiLMLayer(nn.Module):
    def __init__(self, static_dim, dynamic_channels):
        super().__init__()
        self.scale_gen = nn.Linear(static_dim, dynamic_channels)
        self.shift_gen = nn.Linear(static_dim, dynamic_channels)
    def forward(self, x_dyn, x_static):
        scale = self.scale_gen(x_static).unsqueeze(-1) 
        shift = self.shift_gen(x_static).unsqueeze(-1) 
        return x_dyn * (1 + scale) + shift

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
            nn.Linear(total_static_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, hidden_dim)
        )
        self.dyn_embed = nn.Conv1d(dyn_vars_count, hidden_dim, 1)
        self.film = FiLMLayer(hidden_dim, hidden_dim)
        self.tcn = nn.Sequential(
            MultiScaleTCNBlock(hidden_dim, hidden_dim),
            MultiScaleTCNBlock(hidden_dim, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=512, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.reg_head = nn.Linear(hidden_dim, 1)
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
        modulated_feat = self.film(dyn_feat, static_feat)
        tcn_out = self.tcn(modulated_feat)
        tcn_pool = F.adaptive_max_pool1d(tcn_out, 1).squeeze(-1) 
        trans_in = modulated_feat.permute(0, 2, 1) 
        trans_out = self.transformer(trans_in)
        trans_pool = trans_out[:, -1, :] 
        combined = torch.cat([tcn_pool, trans_pool], dim=1)
        embedding = self.fusion_gate(combined)
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
    if rank == 0: print(f"Loading Metadata: {data_dir} (Suffix: {suffix})", flush=True)
    
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
            
    cnt = Counter(y_train_cls)
    cls_num_list = [cnt.get(0, 0) + 1, cnt.get(1, 0) + 1, cnt.get(2, 0) + 1]
    
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
            loss = l_cls + 0.5 * l_reg + 0.1 * l_con
        
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
# 6. Evaluation with Threshold Search
# ==========================================

def evaluate_multi_metrics(model, loader, device):
    """
    计算多个指标，并搜索最佳阈值 (Threshold Moving)
    """
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for bx, by_cls, _, _ in loader:
            bx = bx.to(device, non_blocking=True)
            with autocast():
                logits, _, _ = model(bx, return_proj=False)
            all_logits.append(logits.cpu())
            all_targets.append(by_cls)
            
    all_logits = torch.cat(all_logits)       # [N, 3]
    all_targets = torch.cat(all_targets).numpy() # [N]
    
    # --- 1. 标准指标 (基于 Argmax) ---
    preds_argmax = torch.argmax(all_logits, dim=1).numpy()
    
    acc = accuracy_score(all_targets, preds_argmax)
    f1_macro = f1_score(all_targets, preds_argmax, average='macro', zero_division=0)
    recalls = recall_score(all_targets, preds_argmax, average=None, zero_division=0)
    if len(recalls) < 3: recalls = np.pad(recalls, (0, 3 - len(recalls)), 'constant')
    
    score_weighted = recalls[0] * 0.4 + f1_macro * 0.4 + acc * 0.2
    
    # --- 2. 阈值搜索 (Threshold Adjustment for Class 0) ---
    probs = F.softmax(all_logits, dim=1).numpy()
    prob_c0 = probs[:, 0]
    y_true_c0 = (all_targets == 0).astype(int)
    
    best_constrained_recall = 0.0
    constrained_threshold = 0.5
    current_prec_at_constraint = 0.0
    max_precision_c0 = 0.0
    
    # 遍历阈值
    thresholds = np.arange(0.01, 0.95, 0.01)
    
    for t in thresholds:
        y_pred_c0 = (prob_c0 > t).astype(int)
        
        tp = np.sum((y_pred_c0 == 1) & (y_true_c0 == 1))
        fp = np.sum((y_pred_c0 == 1) & (y_true_c0 == 0))
        fn = np.sum((y_pred_c0 == 0) & (y_true_c0 == 1))
        
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        
        if rec > 0.01 and prec > max_precision_c0:
            max_precision_c0 = prec
            
        # 寻找满足 Precision >= 0.10 的最大 Recall
        if prec >= 0.10:
            if rec > best_constrained_recall:
                best_constrained_recall = rec
                constrained_threshold = t
                current_prec_at_constraint = prec
                
    return {
        'score_weighted': score_weighted,
        'f1_macro': f1_macro,
        'acc': acc,
        'recall_constrained': best_constrained_recall, 
        'precision_max': max_precision_c0,             
        'best_thresh_c0': constrained_threshold,       
        'prec_at_constraint': current_prec_at_constraint 
    }

# ==========================================
# 7. Helper: Inference with Threshold
# ==========================================

class Predictor:
    """
    辅助类：展示如何加载模型和阈值进行预测
    """
    def __init__(self, model_class, checkpoint_path, device='cuda'):
        self.device = device
        self.model = model_class().to(device)
        self.threshold = 0.5 # 默认值
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 【关键】兼容处理：如果 checkpoint 包含 metadata 则读取，否则只读权重
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                if 'threshold' in checkpoint:
                    self.threshold = checkpoint['threshold']
                    print(f"Loaded Optimized Threshold: {self.threshold:.3f}")
            else:
                self.model.load_state_dict(checkpoint) # 旧格式
                
            self.model.eval()
        else:
            print("Checkpoint not found!")

    def predict(self, x_tensor):
        """
        输入: x_tensor [batch, dim]
        输出: classes [batch]
        """
        with torch.no_grad():
            logits, _, _ = self.model(x_tensor.to(self.device))
            probs = F.softmax(logits, dim=1)
            
            # 应用阈值逻辑
            preds = torch.argmax(logits, dim=1) # 默认
            
            # 针对 Class 0 (Fog) 使用优化后的阈值覆盖默认结果
            is_fog = probs[:, 0] > self.threshold
            preds[is_fog] = 0
            
            return preds

# ==========================================
# 8. Main Execution
# ==========================================

def save_checkpoint(model, path, metrics, threshold=None):
    """
    【修改】保存模型权重以及关键的阈值信息
    """
    to_save = {
        'state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'metrics': metrics,
        'threshold': threshold if threshold is not None else 0.5
    }
    torch.save(to_save, path)

def main():
    local_rank, global_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    scaler_amp = GradScaler()
    
    base_path = "/public/home/putianshu/vis_mlp"
    
    if global_rank == 0:
        os.makedirs(os.path.join(base_path, "model"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "scalers"), exist_ok=True)
        print(f"Distributed Init: Rank {global_rank}, World {world_size}, AMP Enabled", flush=True)

    if world_size > 1: dist.barrier()

    # === Stage 1: Pre-training ===
    if global_rank == 0: print("\n=== Stage 1: Pre-training ===", flush=True)
    suffix_s1 = "_9h_pmst_v2" 
    
    train_ds, val_ds, cls_cnts, scaler = load_data_and_scale(
        os.path.join(base_path, 'data/pmst_preprocessed'), suffix_s1, 
        rank=global_rank, device=device
    )
    
    if global_rank == 0:
        joblib.dump(scaler, os.path.join(base_path, f"scalers/pmst_scaler_opt_v3.pkl"))

    if world_size > 1: dist.barrier()

    sampler = FastDistributedWeightedSampler(train_ds, num_replicas=world_size, rank=local_rank, alpha=0.6)
    
    train_loader = DataLoader(
        train_ds, batch_size=256, sampler=sampler, 
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=3 
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=3
    )
    
    model = PMSTNet(dyn_vars_count=24, window_size=9, static_cont_dim=5, 
                   veg_num_classes=21, hidden_dim=256).to(device)
    
    try:
        if global_rank == 0: print("  [Info] Trying torch.compile...", flush=True)
        model = torch.compile(model)
    except: pass

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    
    cls_weights = get_cb_weights(cls_cnts).to(device)
    crit_cls = LDAMLoss(cls_cnts, max_m=0.5, weight=cls_weights, device=device) 
    crit_reg = nn.HuberLoss()
    crit_con = SupervisedContrastiveLoss()
    
    # 记录四个指标的最佳值
    best_records = {
        'score': -1.0,
        'f1': -1.0,
        'recall_c0': -1.0,
        'precision_c0': -1.0
    }
    
    epochs_s1 = 500 
    
    for epoch in range(epochs_s1):
        sampler.set_epoch(epoch)
        loss, _ = train_one_epoch(model, train_loader, optimizer, (crit_cls, crit_reg, crit_con), 
                                      device, epoch, scaler_amp, rank=global_rank)
        
        if epoch % 1 == 0 and global_rank == 0:
            met = evaluate_multi_metrics(model, val_loader, device)
            
            print(f"[Ep{epoch}] Recall(C0): {met['recall_constrained']:.1%} (Prec: {met['prec_at_constraint']:.1%}) | "
                  f"F1: {met['f1_macro']:.1%} | Score: {met['score_weighted']:.4f}", flush=True)
            
            # 【修改】保存逻辑，把 threshold 存进去
            
            # 1. 召回率最高 (带阈值)
            if met['recall_constrained'] > best_records['recall_c0']:
                best_records['recall_c0'] = met['recall_constrained']
                save_checkpoint(model, os.path.join(base_path, "model/pmst_best_recall_constrained_v3.pth"), 
                                met, threshold=met['best_thresh_c0'])
                print(f"  >>> Saved Best Recall (Thresh={met['best_thresh_c0']:.2f})", flush=True)
            
            # 2. 总体分数最高 (带阈值，虽通常用0.5，但也存一份最佳的)
            if met['score_weighted'] > best_records['score']:
                best_records['score'] = met['score_weighted']
                save_checkpoint(model, os.path.join(base_path, "model/pmst_best_score_v3.pth"), 
                                met, threshold=met['best_thresh_c0'])
                
            # 3. F1最高
            if met['f1_macro'] > best_records['f1']:
                best_records['f1'] = met['f1_macro']
                save_checkpoint(model, os.path.join(base_path, "model/pmst_best_f1_v3.pth"), 
                                met, threshold=0.5)
                
            # 4. 精确率最高
            if met['precision_max'] > best_records['precision_c0']:
                best_records['precision_c0'] = met['precision_max']
                save_checkpoint(model, os.path.join(base_path, "model/pmst_best_precision_v3.pth"), 
                                met, threshold=0.9) # 精确率优先模型通常阈值很高

    if world_size > 1: dist.barrier()
    
    # === Clean up ===
    if global_rank == 0: print("Cleaning up Stage 1...", flush=True)
    del train_loader, val_loader, sampler, train_ds, val_ds
    
    if world_size > 1:
        final_state = model.module.state_dict()
    else:
        final_state = model.state_dict()
        
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    if world_size > 1: dist.barrier()
    
    # === Stage 2: Fine-tuning ===
    if global_rank == 0: print("\n=== Stage 2: Fine-tuning (Airport) ===", flush=True)
    suffix_s2 = "_9h_airport_pmst_v2"
    
    ft_train_ds, ft_val_ds, ft_cnts, _ = load_data_and_scale(
        os.path.join(base_path, 'data/airport_pmst_processed'), suffix_s2, 
        scaler=scaler, rank=global_rank, device=device
    )
    
    ft_sampler = FastDistributedWeightedSampler(ft_train_ds, num_replicas=world_size, rank=local_rank, alpha=0.6)
    ft_loader = DataLoader(ft_train_ds, batch_size=128, sampler=ft_sampler, num_workers=4, pin_memory=True, prefetch_factor=3)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=3)
    
    new_model = PMSTNet(dyn_vars_count=24, window_size=9, static_cont_dim=5, 
                   veg_num_classes=21, hidden_dim=256).to(device)
    
    # 加载 Stage 1 模型 (注意处理字典结构)
    pretrain_path = os.path.join(base_path, "model/pmst_best_recall_constrained_v3.pth")
    if os.path.exists(pretrain_path):
        if global_rank == 0: print(f"Loading Stage 1 Best Recall Model from {pretrain_path}", flush=True)
        ckpt = torch.load(pretrain_path, map_location=device)
        if 'state_dict' in ckpt:
            new_model.load_state_dict(ckpt['state_dict'])
        else:
            new_model.load_state_dict(ckpt)
    else:
        new_model.load_state_dict(final_state)
    
    for name, param in new_model.tcn.named_parameters():
        param.requires_grad = False
    
    try: new_model = torch.compile(new_model)
    except: pass

    if world_size > 1:
        model = DDP(new_model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        model = new_model
            
    ft_optim = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-2)
    ft_cls_w = get_cb_weights(ft_cnts).to(device)
    ft_crit = LDAMLoss(ft_cnts, weight=ft_cls_w, device=device)
    
    best_records_s2 = {'score': -1.0, 'f1': -1.0, 'recall_c0': -1.0, 'precision_c0': -1.0}
    epochs_s2 = 300
    
    for epoch in range(epochs_s2):
        ft_sampler.set_epoch(epoch)
        loss, _ = train_one_epoch(model, ft_loader, ft_optim, (ft_crit, crit_reg, crit_con), 
                                  device, epoch, scaler_amp, rank=global_rank)
        
        if epoch % 1 == 0 and global_rank == 0:
            met = evaluate_multi_metrics(model, ft_val_loader, device)
            print(f"[S2 Ep{epoch}] Recall(C0): {met['recall_constrained']:.1%} | Score: {met['score_weighted']:.4f}", flush=True)
            
            # Stage 2 同样保存字典
            if met['recall_constrained'] > best_records_s2['recall_c0']:
                best_records_s2['recall_c0'] = met['recall_constrained']
                save_checkpoint(model, os.path.join(base_path, "model/pmst_airport_best_recall_v3.pth"), 
                                met, threshold=met['best_thresh_c0'])
                
            if met['score_weighted'] > best_records_s2['score']:
                best_records_s2['score'] = met['score_weighted']
                save_checkpoint(model, os.path.join(base_path, "model/pmst_airport_best_score_v3.pth"), 
                                met, threshold=met['best_thresh_c0'])

    if world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    main()
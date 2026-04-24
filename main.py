import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import os
from sklearn.metrics import recall_score, accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import Counter
import joblib
import math
import warnings
import gc
import sys
import datetime
import time

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
                # 适当延长超时时间，防止 NFS 偶尔卡顿导致超时
                timeout=datetime.timedelta(minutes=120) 
            )
        torch.cuda.set_device(local_rank)
    
    return local_rank, global_rank, world_size

# ==========================================
# 2. Optimized Sampler (Crucial for Performance)
# ==========================================

class FastDistributedWeightedSampler(Sampler):
    """
    优化版采样器：
    1. 不生成巨大的权重 Tensor (省内存)。
    2. 使用 randint (O(1)) 代替 multinomial (O(N)) (省 CPU)。
    3. 需要 dataset.y_cls 在内存中，不要 mmap 标签。
    """
    def __init__(self, dataset, num_replicas=None, rank=None, alpha=0.8):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() else 0
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        
        # 确保 targets 是内存中的 numpy array，读取速度极快
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
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.rank * 1000)
        
        indices_local = []
        for c, indices in self.cls_indices.items():
            # 计算当前 Rank 应采样的数量
            local_count_per_cls = int(self.samples_per_class[c] / self.num_replicas)
            if local_count_per_cls > 0:
                rand_idx = torch.randint(0, len(indices), (local_count_per_cls,), generator=g).numpy()
                indices_local.append(indices[rand_idx])
        
        if len(indices_local) > 0:
            indices_local = np.concatenate(indices_local)
        else:
            indices_local = np.array([], dtype=np.int64)

        # 打乱
        np.random.seed(self.epoch + self.rank * 1000)
        np.random.shuffle(indices_local)
        
        # 补齐长度
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
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1 
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
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
# 4. Data Processing (MMAP + MEM Optimization)
# ==========================================

class PMSTDataset(Dataset):
    def __init__(self, X_path, y_cls, y_reg, y_raw, scaler=None):
        # [内存优化关键]
        # X: 使用 mmap 模式 'r'。
        # 这样数据保留在磁盘上，不占用 RAM。
        # Linux 内核会自动管理 Page Cache，只缓存最近访问的数据块。
        self.X = np.load(X_path, mmap_mode='r')
        
        # [性能优化关键]
        # Y: 标签数据很小 (1300万 int/float 约 50MB)，直接加载进内存。
        # 这样 Sampler 在计算权重和索引时，访问的是 RAM，不会产生 IO。
        # 如果 Y 也在 mmap 上，随机采样会导致 IOPS 爆炸。
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
        # 1. 只有这一步会触发磁盘/NFS读取
        # 因为我们用了 mmap，且 batch 里的 index 是离散的，可能会有一定延迟，
        # 但 num_workers 会掩盖这个延迟。
        row = self.X[idx] 
        
        # 2. 转换和计算都在本地内存进行
        features = row[:-1].astype(np.float32)
        veg_id = row[-1]
        
        features = np.nan_to_num(features, nan=0.0)
        
        if self.has_scaler:
            features = (features - self.center) / self.scale
            
        features = np.append(features, veg_id)
        return torch.from_numpy(features).float(), self.y_cls[idx], self.y_reg[idx], self.y_raw[idx]

def load_data_and_scale(data_dir, suffix, scaler=None, rank=0, device=None):
    if rank == 0:
        print(f"Loading Metadata: {data_dir} (Suffix: {suffix})", flush=True)
    
    train_path = os.path.join(data_dir, f'X_train{suffix}.npy')
    val_path = os.path.join(data_dir, f'X_val{suffix}.npy')
    
    # 将标签加载进内存 (RAM)
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
    
    # --- Robust Scaler Logic ---
    if scaler is None:
        # 仅 Rank 0 计算 Scaler，避免 80 个进程同时读取数据
        if rank == 0:
            print("  [Rank 0] Fitting RobustScaler (subsample mmap)...", flush=True)
            scaler = RobustScaler()
            
            # 使用 mmap 读取，但只读很小一部分
            X_temp = np.load(train_path, mmap_mode='r')
            # 采样 20万条数据即可，不用全量，极大减少 IO
            subset_size = min(200000, len(X_temp))
            indices = np.random.choice(len(X_temp), subset_size, replace=False)
            indices.sort() # 排序以利用磁盘顺序读取
            
            X_subset = X_temp[indices, :-1]
            X_subset = np.nan_to_num(X_subset, nan=0.0)
            scaler.fit(X_subset)
            
            del X_subset, X_temp
            
            # 准备广播
            center_tensor = torch.from_numpy(scaler.center_).float().to(device)
            scale_tensor = torch.from_numpy(scaler.scale_).float().to(device)
            dim_tensor = torch.tensor([len(scaler.center_)], device=device)
        else:
            dim_tensor = torch.tensor([0], device=device)
            center_tensor = None
            scale_tensor = None

        # 广播参数
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
            
    # Calculate Class Distribution
    cnt = Counter(y_train_cls)
    cls_num_list = [cnt.get(0, 0) + 1, cnt.get(1, 0) + 1, cnt.get(2, 0) + 1]
    if rank == 0: print(f"  Class Dist: {cnt}", flush=True)
    
    # Dataset 内部使用 mmap 处理 X，内存处理 Y
    train_ds = PMSTDataset(train_path, y_train_cls, y_train_log, y_train_m, scaler=scaler)
    val_ds = PMSTDataset(val_path, y_val_cls, y_val_log, y_val_m, scaler=scaler)
    
    return train_ds, val_ds, cls_num_list, scaler

# ==========================================
# 5. Training Loop
# ==========================================

def train_one_epoch(model, loader, optimizer, criterions, device, epoch, rank=0):
    model.train()
    crit_cls, crit_reg, crit_con = criterions
    avg_loss = 0
    avg_cls = 0
    
    if rank == 0:
        print(f"  [Epoch {epoch}] Start iterating...", flush=True)
    
    start_time = time.time()
    
    for i, (bx, by_cls, by_log, by_raw) in enumerate(loader):
        # 简单的性能监控
        if i == 5 and rank == 0:
            print(f"  [Perf] First 5 batches took {time.time()-start_time:.2f}s", flush=True)

        bx = bx.to(device, non_blocking=True)
        by_cls = by_cls.to(device, non_blocking=True)
        by_log = by_log.to(device, non_blocking=True)
        by_raw = by_raw.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        logits, reg_pred, proj_feat = model(bx, return_proj=True)
        
        l_cls = crit_cls(logits, by_cls)
        
        reg_flat = reg_pred.view(-1)
        mask_fog = by_raw < 2000 
        if mask_fog.sum() > 0:
            l_reg = F.huber_loss(reg_flat[mask_fog], by_log[mask_fog])
        else:
            l_reg = torch.tensor(0.0, device=device)
            
        l_con = crit_con(proj_feat, by_cls)
            
        loss = l_cls + 0.5 * l_reg + 0.1 * l_con
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        avg_loss += loss.item()
        avg_cls += l_cls.item()
        
        if rank == 0 and i % 100 == 0 and i > 0:
            print(f"\r  [Ep{epoch}] Step {i}/{len(loader)} Loss: {loss.item():.4f}", end="", flush=True)
            
    if rank == 0: print("", flush=True)
    return avg_loss / len(loader), avg_cls / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds, trues, reg_errs = [], [], []
    
    with torch.no_grad():
        for bx, by_cls, _, by_raw in loader:
            bx = bx.to(device, non_blocking=True)
            logits, reg_p, _ = model(bx, return_proj=False)
            
            p_cls = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(p_cls)
            trues.extend(by_cls.numpy())
            
            r_m_pred = torch.expm1(reg_p).cpu().numpy().flatten()
            r_m_true = by_raw.numpy().flatten()
            
            mask = r_m_true < 500
            if mask.any():
                reg_errs.extend(np.abs(r_m_pred[mask] - r_m_true[mask]))
                
    recalls = recall_score(trues, preds, average=None, zero_division=0)
    if len(recalls) < 3:
        recalls = np.pad(recalls, (0, 3 - len(recalls)), 'constant')
        
    f1s = f1_score(trues, preds, average=None, zero_division=0)
    if len(f1s) < 3:
        f1s = np.pad(f1s, (0, 3 - len(f1s)), 'constant')
    
    acc = accuracy_score(trues, preds)
    mae = np.mean(reg_errs) if reg_errs else 9999
    
    score = recalls[0] * 0.4 + f1s[0] * 0.4 + acc * 0.2
    
    metrics = {
        'acc': acc, 'r0': recalls[0], 'f1_0': f1s[0], 
        'r1': recalls[1], 'mae': mae, 'score': score
    }
    return metrics

# ==========================================
# 6. Main Execution
# ==========================================

def main():
    local_rank, global_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    base_path = "/public/home/putianshu/vis_mlp"
    
    if global_rank == 0:
        os.makedirs(os.path.join(base_path, "model"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "scalers"), exist_ok=True)
        print(f"Distributed Init: Rank {global_rank}, World {world_size}", flush=True)

    if world_size > 1: dist.barrier()

    # ==========================
    # Stage 1
    # ==========================
    if global_rank == 0: print("\n=== Stage 1: Pre-training (National Data) ===", flush=True)
    
    suffix_s1 = "_9h_pmst_v2" 
    
    # 纯 mmap 加载
    train_ds, val_ds, cls_cnts, scaler = load_data_and_scale(
        os.path.join(base_path, 'data/pmst_preprocessed'), suffix_s1, 
        rank=global_rank, device=device
    )
    
    if global_rank == 0:
        joblib.dump(scaler, os.path.join(base_path, f"scalers/pmst_scaler_s1_15node.pkl"))
        print("  [Rank 0] Scaler saved. Initializing Sampler...", flush=True)

    if world_size > 1: dist.barrier()

    # 使用 Fast Sampler，避免在 mmap 上进行随机 IO
    sampler = FastDistributedWeightedSampler(train_ds, num_replicas=world_size, rank=local_rank, alpha=0.7)
    
    # num_workers: 
    # 使用 mmap 时，4个 worker 通常是安全的。如果依然 OOM，可以降为 2。
    # persistent_workers=True 极其重要，避免每个 epoch 重新 fork 导致 mmap 重新映射。
    train_loader = DataLoader(
        train_ds, batch_size=256, sampler=sampler, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    model = PMSTNet(dyn_vars_count=24, window_size=9, static_cont_dim=5, 
                   veg_num_classes=21, hidden_dim=256).to(device)
    
    if global_rank == 0: print("  [Rank 0] Model created. Wrapping DDP...", flush=True)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    
    cls_weights = get_cb_weights(cls_cnts).to(device)
    crit_cls = LDAMLoss(cls_cnts, weight=cls_weights, device=device)
    crit_reg = nn.HuberLoss()
    crit_con = SupervisedContrastiveLoss()
    
    best_score = -1
    epochs_s1 = 500 
    
    if global_rank == 0: print("  [Rank 0] Start Training Loop...", flush=True)
    
    if world_size > 1: dist.barrier()

    for epoch in range(epochs_s1):
        sampler.set_epoch(epoch)
        loss, l_cls = train_one_epoch(model, train_loader, optimizer, (crit_cls, crit_reg, crit_con), device, epoch, rank=global_rank)
        
        if global_rank == 0:
            print(f"[S1 Ep{epoch}] Loss: {loss:.4f} (Cls: {l_cls:.4f})", end=" ", flush=True)
            if epoch % 1 == 0:
                met = evaluate(model, val_loader, device)
                print(f"| R0: {met['r0']:.1%} F1_0: {met['f1_0']:.1%} Acc: {met['acc']:.1%}", flush=True)
                
                if met['score'] > best_score:
                    best_score = met['score']
                    to_save = model.module.state_dict() if world_size > 1 else model.state_dict()
                    torch.save(to_save, os.path.join(base_path, "model/pmst_stage1_best_15node.pth"))

    if world_size > 1: 
        dist.barrier()
    
    # ==========================
    # Stage 2
    # ==========================
    if global_rank == 0: print("\n=== Stage 2: Fine-tuning (Airport Data) ===", flush=True)
    
    suffix_s2 = "_9h_airport_pmst_v2"
    
    ft_train_ds, ft_val_ds, ft_cnts, _ = load_data_and_scale(
        os.path.join(base_path, 'data/airport_pmst_processed'), suffix_s2, 
        scaler=scaler, rank=global_rank, device=device
    )
    
    ft_sampler = FastDistributedWeightedSampler(ft_train_ds, num_replicas=world_size, rank=local_rank, alpha=0.9)
    
    ft_loader = DataLoader(ft_train_ds, batch_size=128, sampler=ft_sampler, num_workers=4, pin_memory=True, persistent_workers=True)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    load_path = os.path.join(base_path, "model/pmst_stage1_best_15node.pth")
    
    raw_model = model.module if world_size > 1 else model
    raw_model.load_state_dict(torch.load(load_path, map_location=device))
    
    for name, param in raw_model.tcn.named_parameters():
        param.requires_grad = False
    
    if world_size > 1:
        del model 
        model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        model = raw_model
            
    ft_optim = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-2)
    
    ft_cls_w = get_cb_weights(ft_cnts).to(device)
    ft_crit = LDAMLoss(ft_cnts, weight=ft_cls_w, device=device)
    
    best_ft_score = -1
    epochs_s2 = 300
    
    for epoch in range(epochs_s2):
        ft_sampler.set_epoch(epoch)
        loss, _ = train_one_epoch(model, ft_loader, ft_optim, (ft_crit, crit_reg, crit_con), device, epoch, rank=global_rank)
        
        if global_rank == 0:
            met = evaluate(model, ft_val_loader, device)
            print(f"[S2 Ep{epoch}] Loss: {loss:.4f} | R0: {met['r0']:.1%} F1_0: {met['f1_0']:.1%} MAE: {met['mae']:.0f}m", flush=True)
            
            if met['score'] > best_ft_score:
                best_ft_score = met['score']
                save_path = os.path.join(base_path, "model/pmst_airport_best_15node.pth")
                to_save = model.module.state_dict() if world_size > 1 else model.state_dict()
                torch.save(to_save, save_path)
                print(f"  --> S2 Best Saved! (R0={met['r0']:.2f})", flush=True)
    
    if world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import os
import shutil
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
    # 从环境变量获取 Local Rank (节点内的卡号) 和 Global Rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl", 
                init_method='env://', 
                # 超时改回合理的 30 分钟，避免死锁时空转太久
                timeout=datetime.timedelta(minutes=30) 
            )
        torch.cuda.set_device(local_rank)
    
    return local_rank, global_rank, world_size

# [CRITICAL FIX] 将数据拷贝到本地节点，解决 OOM 和 NFS 卡顿
def copy_to_local(src_path, local_rank):
    """
    将数据从 NFS 拷贝到本地 /tmp 目录。
    1. 解决 OOM: 本地文件 mmap 可以利用 OS Page Cache 在多进程间共享内存。
    2. 解决 IO 卡顿: 本地 SSD 随机读取速度远超 NFS。
    """
    filename = os.path.basename(src_path)
    # 使用 /tmp 目录，一般计算节点都有较大的 /tmp 或 /scratch
    target_dir = "/tmp" 
    local_path = os.path.join(target_dir, filename)
    lock_path = local_path + ".lock"

    # 只有该节点的 Rank 0 负责拷贝，避免多进程写入冲突
    if local_rank == 0:
        if not os.path.exists(local_path):
            print(f"[Node Local-0] Copying {src_path} to {local_path}...", flush=True)
            try:
                # 创建锁文件 (简单标记)
                with open(lock_path, 'w') as f: f.write("copying")
                
                shutil.copyfile(src_path, local_path)
                print(f"[Node Local-0] Copy finished.", flush=True)
            except Exception as e:
                print(f"[Node Local-0] Copy FAILED: {e}. Fallback to NFS.", flush=True)
                # 拷贝失败则回退到原始路径，防止程序崩溃
                local_path = src_path
            finally:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
        else:
            print(f"[Node Local-0] Found local cache: {local_path}", flush=True)
    
    # 所有进程等待 Local Rank 0 完成拷贝
    if dist.is_initialized():
        dist.barrier()
        
    # 如果拷贝失败（文件不存在），所有进程回退到 NFS 路径
    if not os.path.exists(local_path):
        return src_path
        
    return local_path

# ==========================================
# 2. Optimized Sampler
# ==========================================

class FastDistributedWeightedSampler(Sampler):
    """
    优化版采样器：省内存、省 CPU
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
        
        # y_cls 已经在内存中，读取很快
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
# 4. Data Processing (FIXED for OOM & IO)
# ==========================================

class PMSTDataset(Dataset):
    def __init__(self, X_path, y_cls, y_reg, y_raw, scaler=None):
        # [CRITICAL FIX] 恢复 mmap_mode='r'
        # 此时 X_path 指向本地 SSD (/tmp)，所以随机读取非常快。
        # 同时，OS 会合并重复内存页，彻底解决 44GB OOM 的问题。
        self.X = np.load(X_path, mmap_mode='r')
        
        # 标签数据较小，读入 RAM 无妨
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
        # 本地 IO 读取，速度极快
        row = self.X[idx] 
        
        # 转换和计算
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
    
    # 原始 NFS 路径
    raw_train_path = os.path.join(data_dir, f'X_train{suffix}.npy')
    raw_val_path = os.path.join(data_dir, f'X_val{suffix}.npy')
    
    # [CRITICAL FIX] 拷贝大文件到本地 SSD，解决 IO 瓶颈和 OOM
    train_path = copy_to_local(raw_train_path, local_rank)
    val_path = copy_to_local(raw_val_path, local_rank)
    
    # 标签数据直接读内存
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
    
    # --- Scaler Logic ---
    if scaler is None:
        if rank == 0:
            print("  [Rank 0] Fitting RobustScaler (local mmap subset)...", flush=True)
            scaler = RobustScaler()
            
            # 读取本地 mmap 文件的子集来拟合
            X_temp = np.load(train_path, mmap_mode='r')
            subset_size = min(200000, len(X_temp))
            indices = np.random.choice(len(X_temp), subset_size, replace=False)
            indices.sort()
            
            X_subset = X_temp[indices, :-1]
            X_subset = np.nan_to_num(X_subset, nan=0.0)
            scaler.fit(X_subset)
            
            del X_subset, X_temp
            
            # 准备广播参数
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
            
    # Calculate Class Distribution
    cnt = Counter(y_train_cls)
    cls_num_list = [cnt.get(0, 0) + 1, cnt.get(1, 0) + 1, cnt.get(2, 0) + 1]
    if rank == 0: print(f"  Class Dist: {cnt}", flush=True)
    
    # Dataset 初始化
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
        
        # 前向传播
        logits, reg_pred, proj_feat = model(bx, return_proj=True)
        
        # Loss 计算
        l_cls = crit_cls(logits, by_cls)
        
        reg_flat = reg_pred.view(-1)
        mask_fog = by_raw < 2000 
        
        # 即使 mask_fog 为空，Loss 也要参与计算图，乘以 0 即可
        # 这对于 find_unused_parameters=True 并不是严格必须，但有助于数值稳定
        if mask_fog.sum() > 0:
            l_reg = F.huber_loss(reg_flat[mask_fog], by_log[mask_fog])
        else:
            l_reg = reg_flat.sum() * 0.0
            
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
    
    # 加载数据 (包含自动拷贝到 /tmp)
    train_ds, val_ds, cls_cnts, scaler = load_data_and_scale(
        os.path.join(base_path, 'data/pmst_preprocessed'), suffix_s1, 
        rank=global_rank, device=device
    )
    
    if global_rank == 0:
        joblib.dump(scaler, os.path.join(base_path, f"scalers/pmst_scaler_s1_15node.pkl"))
        print("  [Rank 0] Scaler saved. Initializing Sampler...", flush=True)

    if world_size > 1: dist.barrier()

    # Sampler 读取内存中的 label，非常快
    sampler = FastDistributedWeightedSampler(train_ds, num_replicas=world_size, rank=local_rank, alpha=0.7)
    
    # num_workers=4 配合本地 SSD mmap 非常安全
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
        # [CRITICAL FIX] find_unused_parameters=True 解决 Regression 分支导致的死锁
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        
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
            print(f"[S1 Ep{epoch}] Loss: {loss:.4f} (Cls: {l_cls:.4f})", flush=True)
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
    # Transition: Clean Memory
    # ==========================
    # [CRITICAL FIX] 清理 Stage 1 占用的资源，防止 Stage 2 OOM
    if global_rank == 0: print("Cleaning up Stage 1 memory...", flush=True)
    
    del train_loader, val_loader, sampler, train_ds, val_ds
    
    # 先获取 raw_model 权重
    if world_size > 1:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
        
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    if world_size > 1: dist.barrier()
    
    # ==========================
    # Stage 2
    # ==========================
    if global_rank == 0: print("\n=== Stage 2: Fine-tuning (Airport Data) ===", flush=True)
    
    suffix_s2 = "_9h_airport_pmst_v2"
    
    # 加载 Stage 2 数据 (同样会拷贝到本地 /tmp)
    ft_train_ds, ft_val_ds, ft_cnts, _ = load_data_and_scale(
        os.path.join(base_path, 'data/airport_pmst_processed'), suffix_s2, 
        scaler=scaler, rank=global_rank, device=device
    )
    
    ft_sampler = FastDistributedWeightedSampler(ft_train_ds, num_replicas=world_size, rank=local_rank, alpha=0.9)
    
    ft_loader = DataLoader(ft_train_ds, batch_size=128, sampler=ft_sampler, num_workers=4, pin_memory=True, persistent_workers=True)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    # 重建模型
    new_model = PMSTNet(dyn_vars_count=24, window_size=9, static_cont_dim=5, 
                   veg_num_classes=21, hidden_dim=256).to(device)
    
    # 加载权重
    # 注意：这里直接用刚才提取的 state_dict，避免再次磁盘 IO 读取
    new_model.load_state_dict(state_dict)
    
    # 冻结参数
    for name, param in new_model.tcn.named_parameters():
        param.requires_grad = False
    
    if world_size > 1:
        # [CRITICAL FIX] 同样开启 find_unused_parameters=True
        model = DDP(new_model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = new_model
            
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
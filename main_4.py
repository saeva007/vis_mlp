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
    
    # [Fix 1] 立即绑定设备
    torch.cuda.set_device(local_rank)
    
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl", 
                init_method='env://', 
                timeout=datetime.timedelta(minutes=30) 
            )
    
    return local_rank, global_rank, world_size


def copy_to_local(src_path, local_rank):
    """
    将数据从 NFS 拷贝到本地 /tmp 目录，解决 IO 瓶颈。
    """
    filename = os.path.basename(src_path)
    target_dir = "/tmp" 
    local_path = os.path.join(target_dir, filename)
    lock_path = local_path + ".lock"

    if local_rank == 0:
        need_copy = True
        if os.path.exists(local_path):
            # 简单检查大小是否一致
            if os.path.getsize(local_path) == os.path.getsize(src_path):
                need_copy = False
                print(f"[Node Local-0] Found valid cache: {local_path}, skipping copy.", flush=True)

        if need_copy:
            # 随机睡眠避免瞬间冲击 NFS
            time.sleep(random.randint(0, 10))
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
# 2. Optimized Sampler (支持动态 Alpha)
# ==========================================

class FastDistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, alpha=0.7):
        """
        alpha: 重采样强度。
        - Stage 1 (1:100) 推荐 0.6
        - Stage 2 (1:500) 推荐 1.0 (完全平衡)
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
        
        # 计算每个类别的采样数
        self.samples_per_class = (prob_per_class * total_samples).astype(int)
        self.num_samples = int(math.ceil(total_samples / self.num_replicas))

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.rank * 1000)
        
        indices_local = []
        for c, indices in self.cls_indices.items():
            local_count_per_cls = int(self.samples_per_class[c] / self.num_replicas)
            
            if local_count_per_cls > 0:
                # 如果该类样本少于需要的采样数，允许重复 (Replacement)
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
        
        # 填充或截断以匹配 num_samples
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Advanced Modules (KAN & Prototype)
# ==========================================

class ChebyKANLayer(nn.Module):
    """
    基于切比雪夫多项式的 KAN 层 (Chebyshev KAN)。
    相比 B-Spline KAN，它不需要维护 Grid，数值更稳定，且适合物理公式拟合。
    """
    def __init__(self, input_dim, output_dim, degree=4):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outputdim = output_dim
        self.degree = degree

        # 切比雪夫多项式参数
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.xavier_normal_(self.cheby_coeffs)
        
        # 归一化和激活，保证输入在 [-1, 1] 附近，这对多项式很重要
        self.layer_norm = nn.LayerNorm(input_dim) 
        self.base_activation = nn.SiLU()

    def forward(self, x):
        # x: [Batch, Input_Dim]
        x = self.layer_norm(x)
        x = torch.tanh(x) # 强制将输入压缩到 (-1, 1)，防止多项式爆炸
        
        # 递归计算切比雪夫多项式
        # T_0(x) = 1, T_1(x) = x, T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
        cheby_values = []
        cheby_values.append(torch.ones_like(x))      # T_0
        cheby_values.append(x)                       # T_1
        
        for i in range(2, self.degree + 1):
            next_t = 2 * x * cheby_values[-1] - cheby_values[-2]
            cheby_values.append(next_t)
            
        # Stack: [Batch, Input, Degree+1]
        stacked_cheby = torch.stack(cheby_values, dim=-1)
        
        # Einsum 计算:
        # B: Batch, I: Input, O: Output, D: Degree
        # y = sum_{i, d} (cheby_val_{b,i,d} * coeff_{i,o,d})
        y = torch.einsum("bid,iod->bo", stacked_cheby, self.cheby_coeffs)
        
        return self.base_activation(y)

class CosinePrototypeHead(nn.Module):
    """
    原型余弦分类头。
    解决 Long-tail 和 NaN 问题的核心组件。
    输出 Logits 范围被限制在 [-scale, scale] 之间。
    """
    def __init__(self, in_dim, num_classes, init_scale=16.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        
        # 学习类别的中心向量 (Prototypes)
        self.prototypes = nn.Parameter(torch.FloatTensor(num_classes, in_dim))
        nn.init.normal_(self.prototypes, mean=0, std=0.01)
        
        # 可学习的缩放因子 (Temperature)
        self.scale = nn.Parameter(torch.tensor(init_scale))
        
    def forward(self, x):
        # x: [Batch, Dim]
        # 1. 对特征归一化 (防止 inf)
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
        
        # 2. 对原型权重归一化
        w_norm = F.normalize(self.prototypes, p=2, dim=1, eps=1e-8)
        
        # 3. 计算余弦相似度 [-1, 1]
        cosine_sim = F.linear(x_norm, w_norm)
        
        # 4. 缩放 (限制 scale 上限，防止再次梯度爆炸)
        # 30.0 是经验上限，exp(30) 很大但不会溢出 FP32
        scale_clamped = torch.clamp(self.scale, min=1.0, max=30.0) 
        
        return cosine_sim * scale_clamped

# ==========================================
# 2. Updated TCN & Fusion Components
# ==========================================

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
        # 融合时保持简单线性即可，TCN 内部无需 KAN
        self.proj = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, 1)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        return self.proj(torch.cat(outs, dim=1))

class StaticDynamicFusion(nn.Module):
    """
    使用 Cross-Attention 让动态序列查询静态上下文。
    """
    def __init__(self, dyn_dim, static_dim, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(dyn_dim, hidden_dim)
        
        # 使用 KAN 增强静态特征的 Value 映射
        self.kv_kan = ChebyKANLayer(static_dim, hidden_dim, degree=3)
        
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_dyn, x_static):
        # x_dyn: [B, L, D_dyn]
        # x_static: [B, D_stat]
        
        Q = self.q_proj(x_dyn) # [B, L, H]
        
        # 静态特征变为 Key/Value
        kv = self.kv_kan(x_static).unsqueeze(1) # [B, 1, H]
        
        # Attention: Dynamic looking at Static
        attn_out, _ = self.attn(query=Q, key=kv, value=kv)
        
        return self.norm(Q + self.dropout(attn_out))

# ==========================================
# 3. Optimized PMSTNet Architecture
# ==========================================

class PMSTNet(nn.Module):
    def __init__(self, dyn_vars_count=24, window_size=9, 
                 static_cont_dim=5, veg_num_classes=21, veg_emb_dim=8,
                 hidden_dim=256, num_classes=3):
        super().__init__()
        self.dyn_vars = dyn_vars_count
        self.window = window_size
        self.static_cont_dim = static_cont_dim
        
        # 1. 静态特征处理
        self.veg_embedding = nn.Embedding(veg_num_classes, veg_emb_dim)
        total_static_dim = static_cont_dim + veg_emb_dim
        
        # [NEW] 使用 KAN 替代 MLP 编码静态特征
        # KAN 擅长捕捉静态地理变量与目标之间的非线性关系
        self.static_encoder = nn.Sequential(
            ChebyKANLayer(total_static_dim, 128, degree=4),
            nn.LayerNorm(128), # KAN 后加 Norm 很重要
            nn.Linear(128, hidden_dim) # 降维对齐
        )
        
        # 2. 动态特征处理 (TCN)
        self.dyn_embed = nn.Conv1d(dyn_vars_count, hidden_dim, 1)
        self.tcn = nn.Sequential(
            MultiScaleTCNBlock(hidden_dim, hidden_dim),
            MultiScaleTCNBlock(hidden_dim, hidden_dim)
        )
        
        # 3. 特征融合 (Cross Attention + KAN)
        self.fusion_module = StaticDynamicFusion(hidden_dim, hidden_dim, hidden_dim)
        
        # 4. 全局上下文 (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, 
                                                 dim_feedforward=512, 
                                                 batch_first=True, dropout=0.2,
                                                 norm_first=True) # Pre-Norm 更加稳定
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 5. [NEW] KAN 融合层
        # 在进入分类头之前，再次利用 KAN 进行高阶特征变换
        self.post_fusion_kan = ChebyKANLayer(hidden_dim * 2, hidden_dim, degree=3)
        
        # 6. [NEW] Heads
        # 原型头解决分类不平衡和梯度爆炸
        self.cls_head = CosinePrototypeHead(hidden_dim, num_classes, init_scale=16.0)
        
        # 回归头保持线性，但独立出来
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # 投影头用于 Contrastive Loss
        self.proj_head = nn.Linear(hidden_dim, 64)

    def forward(self, x, return_proj=True):
        # --- Data Parsing ---
        split_dyn = self.dyn_vars * self.window 
        split_static = split_dyn + self.static_cont_dim 
        
        x_dyn_flat = x[:, :split_dyn]
        x_stat_cont = x[:, split_dyn:split_static]
        x_veg_id = x[:, -1].long()
        
        # --- Static Encoding (KAN) ---
        veg_vec = self.veg_embedding(x_veg_id) 
        x_static_full = torch.cat([x_stat_cont, veg_vec], dim=1) 
        static_feat = self.static_encoder(x_static_full) # [B, H]
        
        # --- Dynamic Encoding (TCN) ---
        x_dyn_seq = x_dyn_flat.view(-1, self.window, self.dyn_vars).permute(0, 2, 1) # [B, C, L]
        dyn_feat = self.dyn_embed(x_dyn_seq) 
        tcn_out = self.tcn(dyn_feat) # [B, H, L]
        tcn_pool = F.adaptive_max_pool1d(tcn_out, 1).squeeze(-1) # [B, H]
        
        # --- Fusion (Cross Attention) ---
        # TCN output needs permutation for Attention: [B, C, L] -> [B, L, C]
        tcn_seq = tcn_out.permute(0, 2, 1)
        fused_seq = self.fusion_module(tcn_seq, static_feat) # [B, L, H]
        
        # --- Global Context (Transformer) ---
        trans_out = self.transformer(fused_seq)
        trans_pool = trans_out[:, -1, :] # 取最后一个时间步 [B, H]
        
        # --- Final Combination (KAN) ---
        # 结合 TCN 的瞬时特征和 Transformer 的上下文特征
        combined = torch.cat([tcn_pool, trans_pool], dim=1)
        embedding = self.post_fusion_kan(combined)
        
        # --- Heads ---
        # Logits 范围受控，不会 NaN
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
        # 使用 mmap_mode 读取
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
    
    # 拷贝到本地 SSD
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
# 5. Training Loop (Supports AMP)
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
            
            # Stage 2 时外部会传入较小的回归权重，这里保持通用公式
            # 外部可以通过 loss = l_cls + reg_weight * l_reg 来控制
            # 但为了简单，这里我们在 Stage 2 逻辑中直接通过 criterions 传递权重可能会比较复杂
            # 所以这里保持一个折中方案，或者让 loss 逻辑由主函数控制？
            # 保持一致性：Stage 1 (0.5), Stage 2 (0.1) 
            # 这里我们根据 epoch 动态调整，或者简单地在 Stage 2 减少
            # 简化起见，我们假设 loss 组成固定，但在 Stage 2 数据分布下，
            # 回归 loss 本身会因为 mask_fog 变少而变小。
            
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
# 6. Evaluation with Threshold Search
# ==========================================

def evaluate_multi_metrics(model, loader, device):
    """
    计算指标并自动搜索最佳阈值
    """
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for bx, by_cls, _, _ in loader:
            bx = bx.to(device, non_blocking=True)
            with autocast():
                logits, _, _ = model(bx, return_proj=False)
            # 在移到 CPU 之前或之后，立即转为 float (float32)
            all_logits.append(logits.float().cpu()) 
            all_targets.append(by_cls)
            
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets).numpy()
    
    # 1. 常规指标 (Argmax)
    preds_argmax = torch.argmax(all_logits, dim=1).numpy()
    acc = accuracy_score(all_targets, preds_argmax)
    f1_macro = f1_score(all_targets, preds_argmax, average='macro', zero_division=0)
    
    # 2. 阈值搜索 (Class 0: Fog)
    probs = F.softmax(all_logits, dim=1).numpy()
    prob_c0 = probs[:, 0]
    y_true_c0 = (all_targets == 0).astype(int)
    
    best_rec = 0.0
    best_thresh = 0.5
    best_prec = 0.0
    max_score = -1.0
    
    # 遍历阈值
    thresholds = np.arange(0.05, 0.95, 0.05)
    for t in thresholds:
        y_pred = (prob_c0 > t).astype(int)
        
        rec = recall_score(y_true_c0, y_pred, zero_division=0)
        prec = precision_score(y_true_c0, y_pred, zero_division=0)
        f1 = f1_score(y_true_c0, y_pred, zero_division=0)
        
        # 定义一个综合得分：Recall 优先，但 Precision 不能太烂
        # 如果 Precision < 10%，得分直接惩罚
        if prec > 0.10:
            score = 0.6 * rec + 0.4 * prec
        else:
            score = rec * 0.1 # 惩罚
            
        if score > max_score:
            max_score = score
            best_rec = rec
            best_thresh = t
            best_prec = prec
            
    return {
        'score': max_score,
        'f1_macro': f1_macro,
        'acc': acc,
        'recall_c0': best_rec,
        'prec_c0': best_prec,
        'best_thresh': best_thresh
    }

def save_checkpoint(model, path, metrics, threshold):
    to_save = {
        'state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'metrics': metrics,
        'threshold': threshold
    }
    torch.save(to_save, path)
    print(f"  [Checkpoint] Saved to {path} (Thresh={threshold:.2f})")

# ==========================================
# 7. Main Execution
# ==========================================

def main():
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
    
    suffix_s1 = "_9h_pmst_v2" 
    train_ds, val_ds, cls_cnts, scaler = load_data_and_scale(
        os.path.join(base_path, 'data/pmst_preprocessed'), suffix_s1, 
        rank=global_rank, device=device
    )
    
    if global_rank == 0:
        joblib.dump(scaler, os.path.join(base_path, f"scalers/pmst_scaler_v4.pkl"))

    if world_size > 1: dist.barrier()

    # Stage 1: Alpha 0.6 适中采样
    sampler = FastDistributedWeightedSampler(train_ds, num_replicas=world_size, rank=local_rank, alpha=0.6)
    
    train_loader = DataLoader(train_ds, batch_size=256, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=3)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=3)
    
    model = PMSTNet(dyn_vars_count=24, window_size=9, static_cont_dim=5, 
                   veg_num_classes=21, hidden_dim=256).to(device)
    
    if world_size > 1:
        model_s1 = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        model_s1 = model
        
    optimizer = optim.AdamW(model_s1.parameters(), lr=5e-4, weight_decay=1e-3)
    
    cls_weights = get_cb_weights(cls_cnts).to(device)
    crit_cls = LDAMLoss(cls_cnts, weight=cls_weights, device=device)
    crit_reg = nn.HuberLoss()
    crit_con = SupervisedContrastiveLoss()
    
    # 检查是否存在 Stage 1 检查点
    s1_ckpt_path = os.path.join(base_path, "model/pmst_stage1_best_v4.pth")
    run_stage1 = True
    
    if os.path.exists(s1_ckpt_path):
        if global_rank == 0: print(f"Found Stage 1 Checkpoint, Loading...", flush=True)
        ckpt = torch.load(s1_ckpt_path, map_location=device)
        if 'state_dict' in ckpt: model_s1.module.load_state_dict(ckpt['state_dict']) if world_size > 1 else model_s1.load_state_dict(ckpt['state_dict'])
        else: model_s1.module.load_state_dict(ckpt) if world_size > 1 else model_s1.load_state_dict(ckpt)
        run_stage1 = False # 跳过训练
        
    if run_stage1:
        best_score_s1 = -1
        for epoch in range(200): # Stage 1 Epochs
            sampler.set_epoch(epoch)
            loss, _ = train_one_epoch(model_s1, train_loader, optimizer, (crit_cls, crit_reg, crit_con), 
                                      device, epoch, scaler_amp, rank=global_rank)
            
            if global_rank == 0 and epoch % 1 == 0:
                met = evaluate_multi_metrics(model_s1, val_loader, device)
                print(f"[S1 Ep{epoch}] Recall(C0): {met['recall_c0']:.1%} | Score: {met['score']:.4f}", flush=True)
                
                if met['score'] > best_score_s1:
                    best_score_s1 = met['score']
                    save_checkpoint(model_s1, s1_ckpt_path, met, threshold=met['best_thresh'])

    if world_size > 1: dist.barrier()
    
    # 清理 Stage 1
    if global_rank == 0: print("Transitioning to Stage 2...", flush=True)
    del train_loader, val_loader, sampler, train_ds, val_ds
    
    # 获取权重用于 Stage 2 初始化
    if world_size > 1:
        state_dict_s1 = model_s1.module.state_dict()
    else:
        state_dict_s1 = model_s1.state_dict()
        
    del model_s1, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # ==========================
    # Stage 2: Fine-tuning
    # ==========================
    if global_rank == 0: print("\n=== Stage 2: Fine-tuning (Airport) ===", flush=True)
    
    suffix_s2 = "_9h_airport_pmst_v2"
    ft_train_ds, ft_val_ds, ft_cnts, _ = load_data_and_scale(
        os.path.join(base_path, 'data/airport_pmst_processed'), suffix_s2, 
        scaler=scaler, rank=global_rank, device=device
    )
    
    # [优化点 1] Stage 2 使用 Alpha=1.0 进行激进平衡采样
    ft_sampler = FastDistributedWeightedSampler(ft_train_ds, num_replicas=world_size, rank=local_rank, alpha=1.0)
    
    ft_loader = DataLoader(ft_train_ds, batch_size=128, sampler=ft_sampler, num_workers=4, pin_memory=True, prefetch_factor=3)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=3)
    
    model = PMSTNet(dyn_vars_count=24, window_size=9, static_cont_dim=5, 
                   veg_num_classes=21, hidden_dim=256).to(device)
    
    model.load_state_dict(state_dict_s1)
    
    # [优化点 2] Phase 1: 冻结 Backbone，只训练 Heads
    if global_rank == 0: print("  [Phase 1] Training Heads Only...", flush=True)
    
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    if world_size > 1:
        model_ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        model_ddp = model
        
    ft_optim = optim.AdamW(filter(lambda p: p.requires_grad, model_ddp.parameters()), lr=1e-3, weight_decay=1e-2)
    ft_cls_w = get_cb_weights(ft_cnts).to(device)
    ft_crit = LDAMLoss(ft_cnts, weight=ft_cls_w, device=device)
    
    # Phase 1 训练 (Warmup)
    for epoch in range(200):
        ft_sampler.set_epoch(epoch)
        # 注意: 此时 l_reg 系数为 0.1
        train_one_epoch(model_ddp, ft_loader, ft_optim, (ft_crit, crit_reg, crit_con), device, epoch, scaler_amp, rank=global_rank)
        
    # [优化点 3] Phase 2: 解冻部分层 (Transformer, Fusion)，保持底层 TCN 冻结
    if global_rank == 0: print("  [Phase 2] Unfreezing Transformers...", flush=True)
    
    raw_model = model_ddp.module if world_size > 1 else model_ddp
    for name, param in raw_model.named_parameters():
        # 保持底层特征提取器冻结
        if "tcn" in name or "static" in name or "dyn_embed" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    # 重新初始化优化器，使用更小的 LR
    ft_optim = optim.AdamW(filter(lambda p: p.requires_grad, model_ddp.parameters()), lr=5e-5, weight_decay=1e-2)
    
    best_s2_score = -1
    
    for epoch in range(200):
        ft_sampler.set_epoch(epoch + 10)
        train_one_epoch(model_ddp, ft_loader, ft_optim, (ft_crit, crit_reg, crit_con), device, epoch+10, scaler_amp, rank=global_rank)
        
        if epoch % 1 == 0 and global_rank == 0:
            # [优化点 4] 使用带阈值搜索的评估
            met = evaluate_multi_metrics(model_ddp, ft_val_loader, device)
            
            print(f"[S2 Ep{epoch+10}] R0: {met['recall_c0']:.1%} P0: {met['prec_c0']:.1%} | Thresh: {met['best_thresh']:.2f} | Score: {met['score']:.4f}", flush=True)
            
            if met['score'] > best_s2_score:
                best_s2_score = met['score']
                save_checkpoint(model_ddp, os.path.join(base_path, "model/pmst_airport_best_v4.pth"), met, threshold=met['best_thresh'])

    if world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    main()
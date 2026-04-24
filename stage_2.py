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
        if not os.path.exists(local_path):
            print(f"[Node Local-0] Copying {src_path} to {local_path}...", flush=True)
            try:
                with open(lock_path, 'w') as f: f.write("copying")
                shutil.copyfile(src_path, local_path)
                print(f"[Node Local-0] Copy finished.", flush=True)
            except Exception as e:
                print(f"[Node Local-0] Copy FAILED: {e}. Fallback to NFS.", flush=True)
                local_path = src_path
            finally:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
        else:
            print(f"[Node Local-0] Found local cache: {local_path}", flush=True)
    
    if dist.is_initialized():
        dist.barrier()
        
    if not os.path.exists(local_path):
        return src_path
        
    return local_path

# ==========================================
# 2. Optimized Sampler & Losses
# ==========================================

class FastDistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, alpha=0.8):
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
    
    if rank == 0:
        print(f"Loading Metadata: {data_dir} (Suffix: {suffix})", flush=True)
    
    raw_train_path = os.path.join(data_dir, f'X_train{suffix}.npy')
    raw_val_path = os.path.join(data_dir, f'X_val{suffix}.npy')
    
    # Copy to Local
    train_path = copy_to_local(raw_train_path, local_rank)
    val_path = copy_to_local(raw_val_path, local_rank)
    
    # Load Labels
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
    
    # Scaler (Must be provided for Stage 2)
    if scaler is None:
        raise ValueError("Stage 2 requires a pre-fitted scaler!")
            
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
    
    for i, (bx, by_cls, by_log, by_raw) in enumerate(loader):
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
            # 乘以 0.0 保持计算图连通
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
# 6. Main Execution (Directly Stage 2)
# ==========================================

def main():
    local_rank, global_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    base_path = "/public/home/putianshu/vis_mlp"
    
    if global_rank == 0:
        print(f"Distributed Init: Rank {global_rank}, World {world_size}", flush=True)
        print("\n=== STARTING DIRECTLY FROM STAGE 2 (FINE-TUNING) ===", flush=True)

    if world_size > 1: dist.barrier()

    # 1. Load Scaler from Stage 1 (Critical for Consistency)
    scaler_path = os.path.join(base_path, "scalers/pmst_scaler_s1_15node.pkl")
    if global_rank == 0:
        print(f"Loading Scaler from: {scaler_path}", flush=True)
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please run Stage 1 first.")
        scaler = joblib.load(scaler_path)
    else:
        scaler = None 
        
    # Broadcast scaler to all nodes (simple way: just load it on all nodes since it's small, 
    # but strictly speaking only rank 0 loading is better. Here we assume all nodes can read the pkl)
    # 既然是 shared filesystem，所有节点直接读最简单
    if scaler is None:
        time.sleep(1) # wait a bit
        scaler = joblib.load(scaler_path)

    # 2. Load Airport Data
    suffix_s2 = "_9h_airport_pmst_v2"
    ft_train_ds, ft_val_ds, ft_cnts, _ = load_data_and_scale(
        os.path.join(base_path, 'data/airport_pmst_processed'), suffix_s2, 
        scaler=scaler, rank=global_rank, device=device
    )
    
    ft_sampler = FastDistributedWeightedSampler(ft_train_ds, num_replicas=world_size, rank=local_rank, alpha=0.9)
    
    ft_loader = DataLoader(ft_train_ds, batch_size=128, sampler=ft_sampler, num_workers=4, pin_memory=True, persistent_workers=True)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    # 3. Create Model
    model = PMSTNet(dyn_vars_count=24, window_size=9, static_cont_dim=5, 
                   veg_num_classes=21, hidden_dim=256).to(device)
    
    # 4. Load Pre-trained Weights
    weight_path = os.path.join(base_path, "model/pmst_stage1_best_15node.pth")
    if global_rank == 0: print(f"Loading Stage 1 Weights from: {weight_path}", flush=True)
    
    # Load to CPU first to avoid OOM, then load_state_dict handles moving to GPU
    state_dict = torch.load(weight_path, map_location='cpu')
    # 如果保存时是 model.module.state_dict()，则 key 没有 module. 前缀
    # 如果保存时是 model.state_dict()，也没有
    # 但如果之前代码有 bug 保存了 module. 前缀，这里最好处理一下
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    
    if global_rank == 0: print("  Weights loaded successfully.", flush=True)
    
    # 5. Freeze TCN Layers
    if global_rank == 0: print("  Freezing TCN layers...", flush=True)
    for name, param in model.tcn.named_parameters():
        param.requires_grad = False
    
    # 6. Wrap DDP (Important: find_unused_parameters=False)
    if world_size > 1:
        # 这里的 find_unused_parameters=False 是关键，因为我们手动处理了条件 loss
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
            
    # 7. Optimizer & Scheduler
    ft_optim = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-2)
    
    ft_cls_w = get_cb_weights(ft_cnts).to(device)
    crit_reg = nn.HuberLoss()
    crit_con = SupervisedContrastiveLoss()
    ft_crit = LDAMLoss(ft_cnts, weight=ft_cls_w, device=device)
    
    best_ft_score = -1
    epochs_s2 = 300
    
    if global_rank == 0: print("  [Rank 0] Start Stage 2 Training Loop...", flush=True)
    
    if world_size > 1: dist.barrier()

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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import shutil
import hashlib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
import gc
import sys
import datetime
import time
import math
import joblib

# 过滤不必要的警告
warnings.filterwarnings('ignore')
def force_cleanup_shm():
    import glob
    # 你的用户名，用于识别你的垃圾文件
    user = os.environ.get('USER', 'jarvis226') 
    
    # 清理 PyTorch 共享内存文件
    try:
        # 删除 PyTorch 在 /dev/shm 下的临时文件 (通常以 torch_ 开头)
        for p in glob.glob("/dev/shm/torch_*"):
            try: os.remove(p)
            except: pass
            
        # 删除你之前拷贝的数据集文件 (特征明显的 .npy)
        for p in glob.glob("/dev/shm/*X_train*"):
            try: os.remove(p)
            except: pass
        for p in glob.glob("/dev/shm/*y_train*"):
            try: os.remove(p)
            except: pass
            
        # 删除之前的实验缓存
        for p in glob.glob("/dev/shm/*exp_*"):
            try: os.remove(p)
            except: pass
            
    except Exception as e:
        print(f"Cleanup warning: {e}")

force_cleanup_shm() # <--- 立即执行
print(f"[{os.environ.get('HOSTNAME')}] /dev/shm Cleaned.", flush=True)
# ==========================================
# 0. 全局配置
# ==========================================
TARGET_WINDOW_SIZE = 12
BASE_PATH = "/public/home/putianshu/vis_mlp"
S1_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_{TARGET_WINDOW_SIZE}h"
S2_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_fe_{TARGET_WINDOW_SIZE}h"

CONFIG = {
    # ========== 实验控制 ==========
    'EXPERIMENT_ID':          os.environ.get('EXPERIMENT_JOB_ID', f'exp_{int(time.time())}'),
    'SKIP_STAGE1':            False,
    'S1_PRETRAINED_PATH':     None,
    'BASE_PATH':              BASE_PATH,
    'WINDOW_SIZE':            TARGET_WINDOW_SIZE,
    'S1_DATA_DIR':            S1_DIR,
    'S2_DATA_DIR':            S2_DIR,
    'NUM_WORKERS':            2,
    'SAVE_CKPT_DIR':          os.path.join(BASE_PATH, 'checkpoints'),

    # ========== Stage 1 训练配置 (高稳定性) ==========
    'S1_TOTAL_STEPS':         30000,
    'S1_VAL_INTERVAL':        1000,  # 稍微频繁一点以便观察
    'S1_BATCH_SIZE':          512,
    'S1_GRAD_ACCUM':          2,
    'S1_FOG_RATIO':           0.20,  # 稍微提高雾样本比例
    'S1_MIST_RATIO':          0.20,
    'S1_LR_BACKBONE':         2e-4,  # 降低 LR 以防爆炸
    'S1_WEIGHT_DECAY':        1e-2,  # 增加正则化

    # ========== Stage 2 训练配置 ==========
    'S2_TOTAL_STEPS':         10000,
    'S2_VAL_INTERVAL':        500,
    'S2_BATCH_SIZE':          512,
    'S2_GRAD_ACCUM':          1,
    'S2_FOG_RATIO':           0.15,
    'S2_MIST_RATIO':          0.15,
    'S2_LR_BACKBONE':         1e-5,
    'S2_LR_HEAD':             5e-5,
    'S2_WEIGHT_DECAY':        1e-2,

    # ========== 评估约束条件 ==========
    'MIN_FOG_PRECISION':      0.15,
    'MIN_FOG_RECALL':         0.50,
    'MIN_MIST_PRECISION':     0.10,
    'MIN_MIST_RECALL':        0.15,
    'MIN_CLEAR_ACC':          0.90,

    # ========== 损失函数配置 ==========
    'LOSS_TYPE':              'asymmetric',
    'LOSS_ALPHA_BINARY':      1.0,
    'LOSS_ALPHA_FINE':        1.0,
    'LOSS_ALPHA_CONSISTENCY': 0.5,
    'LOSS_ALPHA_FP':          1.0,
    'LOSS_ALPHA_FOG_BOOST':   0.5, # 增加对Fog Recall的惩罚
    'LOSS_ALPHA_MIST_BOOST':  0.2,
    'LOSS_FP_THRESHOLD':      0.5,
    
    # Asymmetric 参数 (防止过度自信)
    'ASYM_GAMMA_NEG':         2.0, # 降低负样本gamma，防止NaN
    'ASYM_GAMMA_POS':         0,
    'ASYM_CLIP':              0.1, # 增加截断保护

    # 权重
    'BINARY_POS_WEIGHT':      2.0,
    'FINE_CLASS_WEIGHT_FOG':  2.0,
    'FINE_CLASS_WEIGHT_MIST': 1.5,
    'FINE_CLASS_WEIGHT_CLEAR':0.8,

    # ========== 阈值搜索配置 ==========
    'THRESHOLD_FOG_MIN':      0.20,
    'THRESHOLD_FOG_MAX':      0.90,
    'THRESHOLD_FOG_STEP':     0.02,
    'THRESHOLD_MIST_MIN':     0.20,
    'THRESHOLD_MIST_MAX':     0.90,
    'THRESHOLD_MIST_STEP':    0.02,
    
    'SCORE_PHASE1_FOG':       0.50,
    'SCORE_PHASE1_MIST':      0.30,
    'SCORE_PHASE1_CLEAR':     0.20,

    # ========== 模型配置 ==========
    'MODEL_HIDDEN_DIM':       512,
    'MODEL_DROPOUT':          0.3, # 增加 Dropout
    'MODEL_NUM_CLASSES':      3,
    'USE_FEATURE_ENGINEERING':True,
    'FE_EXTRA_DIMS':          24,
    'GRAD_CLIP_NORM':         0.5, # 更严格的裁剪
    'REG_LOSS_ALPHA':         0.1, # 降低回归损失权重，防止干扰分类
    'VAL_SPLIT_RATIO':        0.1,
}

# ==========================================
# 1. 基础工具与分布式设置
# ==========================================

def init_distributed():
    # 优先读取 torchrun 设置的变量，其次读取 Slurm 变量
    local_rank  = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size  = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    
    # DCU 环境下设置设备
    try:
        torch.cuda.set_device(local_rank)
    except Exception as e:
        if global_rank == 0: print(f"Warning setting device: {e}")

    if world_size > 1 and not dist.is_initialized():
        try:
            dist.init_process_group(
                backend="nccl",
                init_method='env://',
                timeout=datetime.timedelta(minutes=120)
            )
            if global_rank == 0: print("[Dist] Group initialized successfully.", flush=True)
        except Exception as e:
            if global_rank == 0: print(f"[Dist] Init failed: {e}", flush=True)
            raise
    return local_rank, global_rank, world_size

def get_available_space(path):
    try:
        stat = os.statvfs(path)
        return stat.f_bavail * stat.f_frsize
    except Exception:
        return 0

def copy_to_local(src_path: str, local_rank: int, world_size: int, exp_id: str = None) -> str:
    """
    脚本1的核心IO处理函数：哈希命名、空间检查、原子移动。
    """
    target_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    target_dir = "/tmp" 
    if exp_id is None:
        exp_id = "default_exp"

    # 生成唯一的缓存文件名 (Script 1 Logic)
    file_hash = hashlib.md5(f"{exp_id}_{os.path.abspath(src_path)}".encode()).hexdigest()[:8]
    basename  = os.path.basename(src_path)
    local_path = os.path.join(target_dir, f"{os.path.splitext(basename)[0]}_{file_hash}{os.path.splitext(basename)[1]}")

    if local_rank == 0:
        try:
            if not os.path.exists(src_path):
                print(f"[Data-Copy] Warning: Source {src_path} not found.", flush=True)
                final_path = src_path
            else:
                src_size = os.path.getsize(src_path)
                cache_valid = os.path.exists(local_path) and os.path.getsize(local_path) == src_size
                
                if cache_valid:
                    print(f"[Data-Copy] Cache hit: {local_path}", flush=True)
                    final_path = local_path
                else:
                    avail = get_available_space(target_dir)
                    if avail < src_size + 500 * 1024 * 1024: # 500MB buffer
                        print(f"[Data-Copy] Insufficient space on {target_dir}, using NFS.", flush=True)
                        final_path = src_path
                    else:
                        print(f"[Data-Copy] Copying {basename} to {target_dir}...", flush=True)
                        tmp_path = local_path + ".tmp"
                        shutil.copyfile(src_path, tmp_path)
                        os.rename(tmp_path, local_path)
                        final_path = local_path
                        print(f"[Data-Copy] Done: {local_path}", flush=True)
        except Exception as e:
            print(f"[Data-Copy] Error during copy: {e}, falling back to NFS.", flush=True)
            final_path = src_path
    else:
        final_path = src_path 

    if world_size > 1:
        dist.barrier()

    # barrier 结束后，local_path 已经由 local_rank=0 写入完成
    # 所有 rank 统一检查本地路径是否可用
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    else:
        return src_path  # 回退到NFS

def cleanup_temp_files(exp_id: str):
    target_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    if not os.path.exists(target_dir): return
    for fname in os.listdir(target_dir):
        if exp_id in fname:
            try: os.remove(os.path.join(target_dir, fname))
            except: pass

# ==========================================
# 2. 数据采样器
# ==========================================
class StratifiedBalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, fog_ratio=0.2, mist_ratio=0.2,
                 rank=0, world_size=1, seed=42, epoch_length=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        
        y = np.array(dataset.y_cls)
        self.n_fog = max(1, int(batch_size * fog_ratio))
        self.n_mist = max(1, int(batch_size * mist_ratio))
        self.n_clear = batch_size - self.n_fog - self.n_mist
        
        self.indices = {
            0: np.where(y == 0)[0],
            1: np.where(y == 1)[0],
            2: np.where(y == 2)[0]
        }
        
        # 分布式分片
        for k in self.indices:
            self.indices[k] = np.array_split(self.indices[k], world_size)[rank]
            if len(self.indices[k]) == 0: # 防止空分片
                 self.indices[k] = np.where(y == k)[0][:1]

        # 替换原来带 dist.all_reduce 的 epoch_length 计算：
        if epoch_length is None:
            # 不做跨进程同步，各 rank 独立计算，差异极小且 DataLoader 会自动对齐
            min_len = max(1, len(self.indices[0])) // max(1, self.n_fog)
            self.epoch_length = min(5000, max(500, min_len))
        else:
            self.epoch_length = epoch_length

    def __iter__(self):
        rng = np.random.default_rng(seed=self.seed + self.rank + int(time.time()))
        for _ in range(self.epoch_length):
            f = rng.choice(self.indices[0], size=self.n_fog, replace=True)
            m = rng.choice(self.indices[1], size=self.n_mist, replace=True)
            c = rng.choice(self.indices[2], size=self.n_clear, replace=True)
            batch = np.concatenate([f, m, c])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.epoch_length

# ==========================================
# 3. Loss 函数 (增强稳定性)
# ==========================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-7, class_weights=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # 关键修改：增加数值稳定性
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
        
        pos_loss = -targets_one_hot * torch.log(probs) * ((1 - probs) ** self.gamma_pos)
        
        # 负样本抑制
        neg_probs = probs
        neg_loss = -(1 - targets_one_hot) * torch.log(1 - neg_probs) * (neg_probs ** self.gamma_neg)
        
        loss = pos_loss + neg_loss
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets].unsqueeze(1)
            
        return loss.sum(dim=1).mean()

class DualBranchLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = kwargs
        self.register_buffer('pos_weight', torch.tensor([kwargs.get('binary_pos_weight', 1.0)]))
        weights = kwargs.get('fine_class_weight', [1.0, 1.0, 1.0])
        self.register_buffer('fine_class_weight', torch.tensor(weights, dtype=torch.float32))
        
        self.fine_loss = AsymmetricLoss(
            gamma_neg=kwargs.get('asym_gamma_neg', 2),
            gamma_pos=kwargs.get('asym_gamma_pos', 0),
            clip=kwargs.get('asym_clip', 0.1),
            class_weights=self.fine_class_weight
        )

    def forward(self, fine_logits, low_vis_logit, targets):
        # 1. 二分类 Loss (带LogitsClip)
        low_vis_logit = torch.clamp(low_vis_logit, -20, 20) # 防止Sigmoid过饱和
        l_bin = F.binary_cross_entropy_with_logits(
            low_vis_logit, 
            (targets <= 1).float().unsqueeze(1),
            pos_weight=self.pos_weight
        )

        # 2. 多分类 Loss
        l_fine = self.fine_loss(fine_logits, targets)

        # 3. 辅助 Loss
        probs = F.softmax(fine_logits, dim=1)
        # 惩罚 Fog 被预测错的情况
        is_fog = (targets == 0).float()
        l_fb = torch.mean((1 - probs[:, 0]) ** 2 * is_fog)
        
        # 简单的FP惩罚
        is_clear = (targets == 2).float()
        l_fp = torch.mean((probs[:, 0] + probs[:, 1]) ** 2 * is_clear)

        total = (
            self.cfg.get('alpha_binary', 1.0) * l_bin +
            self.cfg.get('alpha_fine', 1.0) * l_fine +
            self.cfg.get('alpha_fp', 0.5) * l_fp +
            self.cfg.get('alpha_fog_boost', 0.2) * l_fb
        )
        return total, {'bin': l_bin.item(), 'fine': l_fine.item()}

# ==========================================
# 4. 模型定义 (高稳定性修改版)
# ==========================================
class StableChebyKANLayer(nn.Module):
    """
    修改后的 ChebyKAN，增加了输入钳制和更小的初始化，防止梯度爆炸。
    """
    def __init__(self, input_dim, output_dim, degree=2): # 降阶到2
        super().__init__()
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        # 关键：极小的初始化方差
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=0.02)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        # 关键：Pre-norm 和 强制范围约束
        x = self.layer_norm(x)
        x = torch.tanh(x) # 强行限制在 (-1, 1)
        
        # 数值保护：防止刚好是 1.0 或 -1.0 导致多项式边界问题
        x = torch.clamp(x, -0.99, 0.99)
        
        vals = [torch.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            # Tn = 2xTn-1 - Tn-2
            next_val = 2 * x * vals[-1] - vals[-2]
            # 关键：每一级递归都截断，防止数值溢出
            next_val = torch.clamp(next_val, -5.0, 5.0)
            vals.append(next_val)
            
        cheby_stack = torch.stack(vals, dim=-1)
        # (batch, in, degree) @ (in, out, degree) -> (batch, out)
        out = torch.einsum("bid,iod->bo", cheby_stack, self.cheby_coeffs)
        return self.act(out)

class SEBlock(nn.Module):
    """ Squeeze-and-Excitation block for temporal features """
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [b, c, t]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ImprovedDualStreamPMSTNet(nn.Module):
    def __init__(self, dyn_vars_count=25, window_size=12, static_cont_dim=5, veg_num_classes=21,
                 hidden_dim=512, num_classes=3, extra_feat_dim=0, dropout=0.3):
        super().__init__()
        self.dyn_vars = dyn_vars_count
        self.window = window_size
        
        # 1. 静态分支
        self.veg_embedding = nn.Embedding(veg_num_classes, 16)
        # 用 MLP 替代 KAN 处理静态变量，更稳
        self.static_encoder = nn.Sequential(
            nn.Linear(static_cont_dim + 16, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, hidden_dim // 4)
        )

        # 2. 物理分支 (特征工程)
        self.physics_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim // 4)
        )

        # 3. 时序分支 (GRU + SE)
        self.temporal_input_proj = nn.Linear(6, hidden_dim) # 投影
        self.temporal_stream = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.se_block = SEBlock(hidden_dim * 2)
        self.temporal_norm = nn.LayerNorm(hidden_dim * 2)

        # 4. 融合层 (使用 StableChebyKAN 或 MLP)
        fusion_dim = (hidden_dim * 2) + (hidden_dim // 4) + (hidden_dim // 4) # Temporal + Static + Physics
        if extra_feat_dim > 0:
            self.extra_encoder = nn.Sequential(nn.Linear(extra_feat_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.GELU())
            fusion_dim += hidden_dim//2
        else:
            self.extra_encoder = None

        # 使用修正后的稳定 KAN
        self.fusion_kan = StableChebyKANLayer(fusion_dim, hidden_dim, degree=2)
        self.dropout = nn.Dropout(dropout)

        # Heads
        self.fine_classifier = nn.Linear(hidden_dim, num_classes)
        self.low_vis_detector = nn.Linear(hidden_dim, 1)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def _physics_features(self, x):
        # 简化的物理特征提取，避免除零
        rh2m = x[:, :, 0]
        t2m = x[:, :, 1]
        wspd = torch.clamp(x[:, :, 6], min=0.1)
        dpd = x[:, :, 22]
        inv = x[:, :, 23]
        
        f1 = rh2m / 100.0 * torch.sigmoid(-2 * dpd) # 相对湿度 * 露点抑制
        f2 = 1.0 / (wspd + 1.0) # 低风速指标
        f3 = inv * f2 # 逆温 + 低风
        f4 = (rh2m - x[:, :, 12]) / 100.0 # 湿度层结
        f5 = x[:, :, 10] # 云量
        
        return torch.stack([f1, f2, f3, f4, f5], dim=2).mean(dim=1) # 取时间平均

    def forward(self, x):
        split_dyn = self.dyn_vars * self.window
        split_static = split_dyn + 5
        
        x_dyn = x[:, :split_dyn].view(-1, self.window, self.dyn_vars)
        x_stat = x[:, split_dyn:split_static]
        x_veg = x[:, split_static].long()
        x_extra = x[:, split_static+1:] if self.extra_encoder else None

        # Physics
        phy_feat = self.physics_encoder(self._physics_features(x_dyn))
        
        # Static
        veg_emb = self.veg_embedding(torch.clamp(x_veg, 0, 20))
        stat_feat = self.static_encoder(torch.cat([x_stat, veg_emb], dim=1))
        
        imp_vars = x_dyn[:, :, [0, 1, 2, 4, 6, 10]] 
        t_in = self.temporal_input_proj(imp_vars)
        t_out, _ = self.temporal_stream(t_in)  # t_out shape: [B, window_size, hidden_dim * 2]
        
        # 1. 接入 SEBlock (通道重标定)
        # SEBlock 需要的输入形状是 [Batch, Channel, Time]
        t_out_se = t_out.permute(0, 2, 1)      # 转换形状: [B, hidden_dim * 2, window_size]
        t_out_se = self.se_block(t_out_se)     # 应用注意力机制
        t_out_se = t_out_se.permute(0, 2, 1)   # 还原形状: [B, window_size, hidden_dim * 2]
        
        # 2. 改进的时序特征聚合 (解决 BiGRU 反向特征丢失问题)
        # 既然 SEBlock 已经统筹了整个时间窗口的信息，最稳健的做法是使用时序维度的平均池化 (Mean Pooling)，
        # 这样既保留了正/反向 GRU 的全部历史，又能有效抵御长尾噪声。
        t_feat = self.temporal_norm(t_out_se.mean(dim=1))
        
        parts = [t_feat, stat_feat, phy_feat]
        if x_extra is not None and self.extra_encoder:
            parts.append(self.extra_encoder(x_extra))
            
        emb = torch.cat(parts, dim=1)
        emb = self.fusion_kan(emb)
        emb = self.dropout(emb)
        
        return self.fine_classifier(emb), self.reg_head(emb), self.low_vis_detector(emb)

# ==========================================
# 5. 数据集 (保持原样，增加鲁棒性)
# ==========================================
class PMSTDataset(Dataset):
    def __init__(self, X_path, y_cls, y_reg, y_raw, scaler=None, window_size=12, use_fe=True, indices=None):
        self.X_path = X_path # 保存路径，而不是在 init 时 mmap 打开
        self.indices = np.asarray(indices) if indices is not None else np.arange(len(y_cls)) # 注意长度基准
        self.y_cls = torch.as_tensor(y_cls[self.indices], dtype=torch.long)
        
        clean_raw = np.maximum(y_raw[self.indices], 0.0)
        self.y_reg = torch.as_tensor(np.log1p(clean_raw), dtype=torch.float32)
        self.y_raw = torch.as_tensor(clean_raw, dtype=torch.float32)
        
        self.split_dyn = window_size * 25
        self.scaler = scaler
        self.use_fe = use_fe
        
        self.log_mask = np.zeros(self.split_dyn, dtype=bool)
        for t in range(window_size):
            for i in [2, 4, 9]: self.log_mask[t * 25 + i] = True

    def __len__(self):
        return len(self.indices)
      
      
    def __getitem__(self, idx):
        # 【修复】：多进程 Worker 启动后，各自独立打开 mmap，完美避开文件描述符锁死！
        if getattr(self, 'X', None) is None:
            self.X = np.load(self.X_path, mmap_mode='r')
            
        real_idx = self.indices[idx]
        row = self.X[real_idx]
        # 下面保留你原来的逻辑不变...
        feats = row[:self.split_dyn+5].astype(np.float32)
        feats[:self.split_dyn] = np.where(self.log_mask, np.log1p(np.maximum(feats[:self.split_dyn], 0)), feats[:self.split_dyn])
        
        if self.scaler:
            feats = (feats - self.scaler.center_) / (self.scaler.scale_ + 1e-6) # Epsilon防止除零
            
        # 2. 植被
        veg = np.array([row[self.split_dyn+5]], dtype=np.float32)
        
        # 3. 额外特征
        final = np.concatenate([np.clip(feats, -10, 10), veg])
        if self.use_fe:
            extra = row[self.split_dyn+6:].astype(np.float32)
            # RobustScaler 可能会产生很大的值，再次 clip 保证数值安全
            final = np.concatenate([final, np.clip(extra, -10, 10)])
            
        final = np.nan_to_num(final, nan=0.0)
        return torch.from_numpy(final).float(), self.y_cls[idx], self.y_reg[idx], self.y_raw[idx]

def load_data(data_dir, scaler=None, rank=0, local_rank=0, device=None, 
              reuse_scaler=False, win_size=12, world_size=1, exp_id=None):
    path_X = copy_to_local(os.path.join(data_dir, 'X_train.npy'), local_rank, world_size, exp_id)
    path_y = copy_to_local(os.path.join(data_dir, 'y_train.npy'), local_rank, world_size, exp_id)

    y_raw = np.load(path_y)
    y_cls = np.zeros(len(y_raw), dtype=np.int64)
    if np.max(y_raw) < 100:
        y_raw = y_raw * 1000
    y_cls[y_raw >= 500] = 1
    y_cls[y_raw >= 1000] = 2

    scaler_path = os.path.join(CONFIG['SAVE_CKPT_DIR'], f'robust_scaler_w{win_size}.pkl')
    
    if scaler is None and not reuse_scaler:
        # ★ 修复 1：全员强制同步，防止快慢节点产生不同的行为分支
        if world_size > 1:
            dist.barrier()
            
        if rank == 0:
            if not os.path.exists(scaler_path):
                # ★ 修复 2：限制最大采样数为 20 万
                print("[Scaler] Fitting (first time, will be cached)...", flush=True)
                X_m = np.load(path_X, mmap_mode='r')
                
                n_total = len(X_m)
                max_samples = 200000
                
                # 随机生成索引（使用固定的 seed 保证可重复性）
                rng_scaler = np.random.default_rng(seed=42)
                if n_total > max_samples:
                    sample_indices = rng_scaler.choice(n_total, size=max_samples, replace=False)
                    sample_indices.sort()  # 关键：排序后的索引读取 mmap 速度会快百倍以上，避免磁盘随机寻道跳跃
                else:
                    sample_indices = np.arange(n_total)
                
                print(f"[Scaler] Extracting {len(sample_indices)} random samples for fitting...", flush=True)
                sub = X_m[sample_indices, :win_size * 25 + 5].astype(np.float32)
                
                log_mask = np.zeros(win_size * 25, dtype=bool)
                for t in range(win_size):
                    for i in [2, 4, 9]:
                        log_mask[t * 25 + i] = True
                        
                sub[:, :win_size * 25] = np.where(
                    log_mask,
                    np.log1p(np.maximum(sub[:, :win_size * 25], 0)),
                    sub[:, :win_size * 25]
                )
                
                print("[Scaler] Running RobustScaler...", flush=True)
                scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(sub)
                joblib.dump(scaler, scaler_path)
                print(f"[Scaler] Saved to {scaler_path}", flush=True)
            else:
                print(f"[Scaler] Cached file already exists, skipping fit.", flush=True)

        # ★ 修复 3：全员再次等待，必须确保 Rank 0 彻底将文件写入磁盘
        if world_size > 1:
            dist.barrier()

        # ★ 修复 4：所有进程（包括 Rank 0）统一从文件加载，给 NFS 预留文件可见的延迟宽限
        wait_time = 0
        while not os.path.exists(scaler_path) and wait_time < 60:
            time.sleep(1)
            wait_time += 1
            
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Rank {rank} waited 60s but still cannot see {scaler_path} on NFS.")
            
        scaler = joblib.load(scaler_path)

    # 后续的数据集划分逻辑保持不变...
    n_total = len(y_cls)
    n_val = int(n_total * CONFIG['VAL_SPLIT_RATIO'])
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(n_total)

    tr_ds = PMSTDataset(path_X, y_cls, np.zeros_like(y_raw), y_raw, scaler, win_size, True, indices[n_val:])
    val_ds = PMSTDataset(path_X, y_cls, np.zeros_like(y_raw), y_raw, scaler, win_size, True, indices[:n_val])

    return tr_ds, val_ds, scaler

# ==========================================
# 6. 评估与训练 (带熔断机制)
# ==========================================
class ComprehensiveMetrics:
    def __init__(self, config):
        self.cfg = config
        # 初始化默认阈值
        self.best_th = {'fog': 0.5, 'mist': 0.5}
        # 定义约束
        self.min_prec_threshold = 0.18  # 命中率至少 18%
        self.min_clear_recall = 0.90    # 晴天召回率至少 90%

    def _calc_metrics_per_class(self, targets, preds, class_id):
        """计算指定类别的 Precision 和 Recall"""
        # TP: 预测是该类，真实也是该类
        tp = ((preds == class_id) & (targets == class_id)).sum()
        # FP: 预测是该类，真实不是该类
        fp = ((preds == class_id) & (targets != class_id)).sum()
        # FN: 预测不是该类，真实是该类
        fn = ((preds != class_id) & (targets == class_id)).sum()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        return precision, recall

    def evaluate(self, model, loader, device, rank=0, world_size=1, actual_val_size=None):
        model.eval()
        probs_l, targets_l = [], []
        
        # 1. 各个 GPU 只推理自己分配到的那 1/40 数据
        with torch.no_grad():
            for bx, by, _, _ in loader:
                bx = bx.to(device, non_blocking=True)
                fine, _, _ = model(bx)
                probs_l.append(F.softmax(fine, dim=1))
                targets_l.append(by.to(device))
                
        # 拼接当前 GPU 的结果
        local_probs = torch.cat(probs_l, dim=0)
        local_targets = torch.cat(targets_l, dim=0)

        if world_size > 1:
            # 动态获取最大长度并 padding (非常防御性的编程)
            local_size = torch.tensor([local_probs.size(0)], dtype=torch.long, device=device)
            max_size = local_size.clone()
            dist.all_reduce(max_size, op=dist.ReduceOp.MAX)

            if local_size < max_size:
                pad_size = max_size.item() - local_size.item()
                pad_probs = torch.zeros((pad_size, local_probs.size(1)), dtype=local_probs.dtype, device=device)
                PADDING_LABEL = -1  # 使用 -1 作为 padding 标签，不会被当作真实类别
                pad_targets = torch.full((pad_size,), PADDING_LABEL, dtype=local_targets.dtype, device=device)
                local_probs = torch.cat([local_probs, pad_probs], dim=0)
                local_targets = torch.cat([local_targets, pad_targets], dim=0)

            gathered_probs = [torch.zeros_like(local_probs) for _ in range(world_size)]
            gathered_targets = [torch.zeros_like(local_targets) for _ in range(world_size)]

            dist.all_gather(gathered_probs, local_probs)
            dist.all_gather(gathered_targets, local_targets)
            
            # 将收集到的结果拼接成完整的验证集结果
            all_probs = torch.cat(gathered_probs, dim=0).cpu().numpy()
            all_targets = torch.cat(gathered_targets, dim=0).cpu().numpy()
        else:
            all_probs = local_probs.cpu().numpy()
            all_targets = local_targets.cpu().numpy()
        
        best_score = -1.0
        best_stats = None
        
        # 3. 只有主进程(rank==0)进行阈值搜索和全局指标计算
        if rank == 0:
            print(f"  [Eval] Searching thresholds... (Constraints: Prec>={self.min_prec_threshold}, ClearRecall>={self.min_clear_recall})")
            
            # 确保对齐原始验证集大小（因为 DistributedSampler 为了整除可能会 pad 几个样本）
            actual_val_len = len(loader.dataset)
            n = actual_val_size if actual_val_size is not None else len(loader.dataset)
            probs = all_probs[:n]
            targets = all_targets[:n]

            valid_mask = targets >= 0
            probs = probs[valid_mask]
            targets = targets[valid_mask]
            
            search_space = np.arange(0.10, 0.65, 0.05)
            
            for f_th in search_space:
                for m_th in search_space:
                    preds = np.full(len(targets), 2, dtype=int)
                    is_fog_prob = probs[:, 0] > f_th
                    preds[is_fog_prob] = 0
                    is_mist_prob = (probs[:, 1] > m_th) & (~is_fog_prob)
                    preds[is_mist_prob] = 1
                    
                    p0, r0 = self._calc_metrics_per_class(targets, preds, 0)
                    p1, r1 = self._calc_metrics_per_class(targets, preds, 1)
                    p2, r2 = self._calc_metrics_per_class(targets, preds, 2)
                    
                    if (p0 < self.min_prec_threshold or 
                        p1 < self.min_prec_threshold or 
                        r2 < self.min_clear_recall):
                        continue
                    
                    score = (r0 + r1) / 2.0
                    
                    if score > best_score:
                        best_score = score
                        self.best_th = {'fog': f_th, 'mist': m_th}
                        best_stats = {
                            'Fog_R': r0, 'Fog_P': p0,
                            'Mist_R': r1, 'Mist_P': p1,
                            'Clear_R': r2, 'Clear_P': p2
                        }

            if best_stats is not None:
                print(f"  [Eval] Found Optimal Th -> Fog:{self.best_th['fog']:.2f}, Mist:{self.best_th['mist']:.2f}")
                print(f"  [Eval] Stats: Fog R={best_stats['Fog_R']:.3f}/P={best_stats['Fog_P']:.3f} | "
                      f"Mist R={best_stats['Mist_R']:.3f}/P={best_stats['Mist_P']:.3f} | "
                      f"Clear R={best_stats['Clear_R']:.3f} (Score={best_score:.4f})")
            else:
                preds = np.argmax(probs, axis=1)
                p0, r0 = self._calc_metrics_per_class(targets, preds, 0)
                p1, r1 = self._calc_metrics_per_class(targets, preds, 1)
                r2 = self._calc_metrics_per_class(targets, preds, 2)[1]
                best_score = (r0+r1)/2
                print("  [Eval] WARN: No threshold met constraints!")
                print(f"  [Eval] Fallback Stats: Fog R={r0:.3f}/P={p0:.3f} | Mist R={r1:.3f}/P={p1:.3f} | Clear R={r2:.3f}")

        # 将评估同步，防止节点间失控
        if world_size > 1:
            dist.barrier()

        return {'score': best_score, 'thresholds': self.best_th}

def train_stage(tag, model, tr_ds, val_ds, optimizer, loss_fn, device, rank, world_size,
                total_steps, val_int, batch_size, grad_accum, exp_id):
    loader = DataLoader(
        tr_ds,
        batch_sampler=StratifiedBalancedBatchSampler(
            tr_ds, batch_size, rank=rank, world_size=world_size
        ),
        num_workers=4,
        pin_memory=True
    )
    
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) \
                  if world_size > 1 else None
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, sampler=val_sampler, pin_memory=True
    )
    
    # 保存完整验证集大小，用于后续评估时截断padding
    actual_val_size = len(val_ds)  # ← 这里记录的是真实大小
    
    metrics = ComprehensiveMetrics(CONFIG)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    step = 0
    model.train()
    
    # 无限循环直到达到 step
    iterator = iter(loader)
    
    while step < total_steps:
        try:
            bx, by, blog, braw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            bx, by, blog, braw = next(iterator)
            
        bx, by = bx.to(device), by.to(device)
        blog = blog.to(device) # Reg target
        
        # Forward
        fine, reg, bin_out = model(bx)
        
        # Loss
        l_dual, _ = loss_fn(fine, bin_out, by)
        l_reg = F.mse_loss(reg.view(-1), blog) # 使用简单的MSE更稳定
        loss = l_dual + CONFIG['REG_LOSS_ALPHA'] * l_reg
        loss = loss / grad_accum
        
        # 1. 检查 NaN Loss (关键保护!)
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0: print(f"\n[WARNING] Step {step}: Loss is NaN/Inf! Skipping batch.", flush=True)
            optimizer.zero_grad()
            continue # 跳过此 Batch
            
        loss.backward()
        
        if (step + 1) % grad_accum == 0:
            # 2. 梯度裁剪 (防止爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['GRAD_CLIP_NORM'])
            
            # 3. 检查梯度是否正常
            grad_good = True
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    grad_good = False
                    break
            
            if grad_good:
                optimizer.step()
                scheduler.step()
            else:
                if rank == 0: print(f"\n[WARNING] Step {step}: Gradients contain NaN! Skipper step.", flush=True)
            
            optimizer.zero_grad()
            step += 1
            
            if rank == 0 and step % 50 == 0:
                print(f"\r[{tag}] Step {step}/{total_steps} Loss={loss.item()*grad_accum:.4f} LR={scheduler.get_last_lr()[0]:.6f}", end="", flush=True)

            if step % val_int == 0:
                if rank == 0: print("")
                # 加入 world_size 变量传入
                res = metrics.evaluate(model, val_loader, device, rank, world_size,
                            actual_val_size=actual_val_size)
                model.train()
                # 保存 Checkpoint
                if rank == 0:
                    torch.save(model.state_dict(), os.path.join(CONFIG['SAVE_CKPT_DIR'], f"{exp_id}_{tag}_latest.pt"))

# ==========================================
# Main
# ==========================================
def main():
    l_rank, g_rank, w_size = init_distributed()
    device = torch.device(f"cuda:{l_rank}")
    
    if g_rank == 0:
        os.makedirs(CONFIG['SAVE_CKPT_DIR'], exist_ok=True)
        print(f"Start Exp: {CONFIG['EXPERIMENT_ID']}", flush=True)

    # ★ 新增：DDP 初始化前的连通性测试
    # 如果这里卡死，说明 NCCL 多节点通信本身有问题（不是代码问题）
    if w_size > 1:
        if g_rank == 0:
            print("[Connectivity Test] Starting all-reduce test...", flush=True)
        test_tensor = torch.ones(1, device=device) * g_rank
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(w_size))
        assert test_tensor.item() == expected, \
            f"Rank {g_rank}: all-reduce result {test_tensor.item()} != expected {expected}"
        if g_rank == 0:
            print("[Connectivity Test] PASSED. Proceeding to model init.", flush=True)

    model = ImprovedDualStreamPMSTNet(
        window_size=CONFIG['WINDOW_SIZE'],
        hidden_dim=CONFIG['MODEL_HIDDEN_DIM'],
        num_classes=3,
        extra_feat_dim=CONFIG['FE_EXTRA_DIMS']
    ).to(device)
    
    if g_rank == 0:
        print("[Model] Created. Wrapping with DDP...", flush=True)
    
    if w_size > 1:
        model = DDP(model, device_ids=[l_rank], find_unused_parameters=False)
    
    if g_rank == 0:
        print("[Model] DDP wrap complete.", flush=True)
    
        
    loss_fn = DualBranchLoss(
        binary_pos_weight=CONFIG['BINARY_POS_WEIGHT'],
        fine_class_weight=[CONFIG['FINE_CLASS_WEIGHT_FOG'], CONFIG['FINE_CLASS_WEIGHT_MIST'], CONFIG['FINE_CLASS_WEIGHT_CLEAR']],
        asym_gamma_neg=CONFIG['ASYM_GAMMA_NEG'],
        asym_clip=CONFIG['ASYM_CLIP']
    ).to(device)
    
    # Load Data
    tr_ds, val_ds, scaler = load_data(CONFIG['S1_DATA_DIR'], None, g_rank, l_rank, device, False, CONFIG['WINDOW_SIZE'], w_size, CONFIG['EXPERIMENT_ID'])
    
    # Train Stage 1
    if not CONFIG['SKIP_STAGE1']:
        opt = optim.AdamW(model.parameters(), lr=CONFIG['S1_LR_BACKBONE'], weight_decay=CONFIG['S1_WEIGHT_DECAY'])
        train_stage('S1', model, tr_ds, val_ds, opt, loss_fn, device, g_rank, w_size, CONFIG['S1_TOTAL_STEPS'], CONFIG['S1_VAL_INTERVAL'], CONFIG['S1_BATCH_SIZE'], CONFIG['S1_GRAD_ACCUM'], CONFIG['EXPERIMENT_ID'])
    
    cleanup_temp_files(CONFIG['EXPERIMENT_ID'])
    if w_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    main()
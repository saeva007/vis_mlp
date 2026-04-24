import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import os
import shutil
import hashlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import warnings
import gc
import sys
import datetime
import time
import glob
import joblib  # 恢复：用于保存scaler和thresholds

# 过滤不必要的警告
warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局配置 (恢复脚本2的详细配置)
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
    'NUM_WORKERS':            4,
    'SAVE_CKPT_DIR':          os.path.join(BASE_PATH, 'checkpoints'), # 保持脚本1的路径逻辑

    # ========== Stage 1 训练配置 ==========
    'S1_TOTAL_STEPS':         30000,
    'S1_VAL_INTERVAL':        2000,
    'S1_BATCH_SIZE':          512,
    'S1_GRAD_ACCUM':          2,
    'S1_FOG_RATIO':           0.15,
    'S1_MIST_RATIO':          0.15,
    'S1_LR_BACKBONE':         3e-4,
    'S1_WEIGHT_DECAY':        1e-3,

    # ========== Stage 2 训练配置 ==========
    'S2_TOTAL_STEPS':         5000,
    'S2_VAL_INTERVAL':        500,
    'S2_BATCH_SIZE':          512,
    'S2_GRAD_ACCUM':          1,
    'S2_FOG_RATIO':           0.12,
    'S2_MIST_RATIO':          0.12,
    'S2_LR_BACKBONE':         1e-5,
    'S2_LR_HEAD':             1e-4,
    'S2_WEIGHT_DECAY':        1e-2,

    # ========== 评估约束条件 (恢复) ==========
    'MIN_FOG_PRECISION':      0.15,
    'MIN_FOG_RECALL':         0.50,
    'MIN_MIST_PRECISION':     0.12,
    'MIN_MIST_RECALL':        0.15,
    'MIN_CLEAR_ACC':          0.95,

    # ========== 损失函数配置 (恢复灵活性) ==========
    'LOSS_TYPE':              'asymmetric',   # 'asymmetric', 'focal', etc.
    'LOSS_ALPHA_BINARY':      0,
    'LOSS_ALPHA_FINE':        1.0,
    'LOSS_ALPHA_CONSISTENCY': 0,
    'LOSS_ALPHA_FP':          0,
    'LOSS_ALPHA_FOG_BOOST':   0,
    'LOSS_ALPHA_MIST_BOOST':  0,
    'LOSS_FP_THRESHOLD':      0.5,
    
    # Asymmetric 参数
    'ASYM_GAMMA_NEG':         4.0,
    'ASYM_GAMMA_POS':         0,
    'ASYM_CLIP':              0.05,
    
    # Focal 参数
    'FOCAL_GAMMA':            2.0,
    'FOCAL_ALPHA':            None,

    # 权重
    'BINARY_POS_WEIGHT':      1.0,
    'FINE_CLASS_WEIGHT_FOG':  1.2,
    'FINE_CLASS_WEIGHT_MIST': 1.1,
    'FINE_CLASS_WEIGHT_CLEAR':1.0,

    # ========== 阈值搜索配置 (恢复) ==========
    'THRESHOLD_FOG_MIN':      0.10,
    'THRESHOLD_FOG_MAX':      0.90,
    'THRESHOLD_FOG_STEP':     0.05,
    'THRESHOLD_MIST_MIN':     0.10,
    'THRESHOLD_MIST_MAX':     0.90,
    'THRESHOLD_MIST_STEP':    0.05,
    
    # 阈值搜索阶段约束
    'THRESHOLD_PHASE2_CLEAR_RECALL':  0.90,
    'THRESHOLD_PHASE2_FOG_PRECISION': 0.18,
    'THRESHOLD_PHASE2_FOG_RECALL':    0.45,
    'THRESHOLD_PHASE3_CLEAR_RECALL':  0.90,

    # 综合得分权重 (恢复)
    'SCORE_PHASE1_FOG':       0.45,
    'SCORE_PHASE1_MIST':      0.40,
    'SCORE_PHASE1_CLEAR':     0.15,
    'SCORE_PHASE2_FOG':       0.60,
    'SCORE_PHASE2_MIST':      0.30,
    'SCORE_PHASE2_CLEAR':     0.10,
    'SCORE_PHASE3_FOG':       0.50,
    'SCORE_PHASE3_MIST':      0.35,
    'SCORE_PHASE3_CLEAR':     0.15,

    # ========== 模型配置 ==========
    'MODEL_HIDDEN_DIM':       512,
    'MODEL_DROPOUT':          0.2,
    'MODEL_NUM_CLASSES':      3,
    'USE_FEATURE_ENGINEERING':True,
    'FE_EXTRA_DIMS':          24,
    'GRAD_CLIP_NORM':         1.0,
    'REG_LOSS_ALPHA':         1.0,
    'VAL_SPLIT_RATIO':        0.1,
}

# ==========================================
# 1. 基础工具与分布式设置 (严格保持脚本1)
# ==========================================

def init_distributed():
    """初始化分布式环境，保持脚本1的120分钟超时设置"""
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    
    torch.cuda.set_device(local_rank)
    
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method='env://',
            timeout=datetime.timedelta(minutes=120) # 关键修正
        )
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

    if local_rank != 0:
        if os.path.exists(local_path):
            return local_path
        else:
            return src_path
    else:
        if os.path.exists(local_path) and os.path.getsize(local_path) == os.path.getsize(src_path):
            return local_path
        return src_path

def cleanup_temp_files(exp_id: str):
    target_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    if not os.path.exists(target_dir): return
    for fname in os.listdir(target_dir):
        if exp_id in fname and (fname.endswith('.npy') or fname.endswith('.tmp')):
            try:
                os.remove(os.path.join(target_dir, fname))
            except:
                pass

# ==========================================
# 2. 数据采样器 (采用脚本2的逻辑以支持精确比例控制)
# ==========================================
class StratifiedBalancedBatchSampler(Sampler):
    """
    结合脚本1的Batch机制和脚本2的比例控制。
    """
    def __init__(self, dataset, batch_size, fog_ratio=0.15, mist_ratio=0.15,
                 rank=0, world_size=1, seed=42, epoch_length=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        
        # 只需要读取标签
        y = np.array(dataset.y_cls)
        
        self.n_fog = max(1, int(batch_size * fog_ratio))
        self.n_mist = max(1, int(batch_size * mist_ratio))
        self.n_clear = batch_size - self.n_fog - self.n_mist
        
        fog_idx = np.where(y == 0)[0]
        mist_idx = np.where(y == 1)[0]
        clear_idx = np.where(y == 2)[0]
        
        # 分布式分片
        self.fog_indices = np.array_split(fog_idx, world_size)[rank]
        self.mist_indices = np.array_split(mist_idx, world_size)[rank]
        self.clear_indices = np.array_split(clear_idx, world_size)[rank]
        
        # Fallback handling
        if len(self.fog_indices) == 0: self.fog_indices = fog_idx[:1] 
        if len(self.mist_indices) == 0: self.mist_indices = mist_idx[:1]
        if len(self.clear_indices) == 0: self.clear_indices = clear_idx[:1]

        if epoch_length is None:
            min_len = min(len(self.fog_indices), len(self.mist_indices), len(self.clear_indices))
            # 保证epoch不要太短，也不要无限长
            self.epoch_length = min(10000, max(1000, min_len * 3 // batch_size))
        else:
            self.epoch_length = epoch_length

    def __iter__(self):
        rng = np.random.default_rng(seed=self.seed + self.rank + int(time.time()))
        for _ in range(self.epoch_length):
            f = rng.choice(self.fog_indices, size=self.n_fog, replace=True)
            m = rng.choice(self.mist_indices, size=self.n_mist, replace=True)
            c = rng.choice(self.clear_indices, size=self.n_clear, replace=True)
            indices = np.concatenate([f, m, c])
            rng.shuffle(indices)
            yield indices.tolist()

    def __len__(self):
        return self.epoch_length

# ==========================================
# 3. Loss 函数 (恢复脚本2的灵活工厂模式)
# ==========================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, class_weights=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        probs = F.softmax(logits, dim=1).clamp(self.eps, 1 - self.eps)
        
        pos_loss = -targets_one_hot * torch.log(probs) * ((1 - probs) ** self.gamma_pos)
        neg_probs = torch.clamp(probs, max=1 - self.clip)
        neg_loss = -(1 - targets_one_hot) * torch.log(1 - probs) * (neg_probs ** self.gamma_neg)
        
        loss = pos_loss + neg_loss
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets].unsqueeze(1)
        return loss.sum(dim=1).mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
             self.register_buffer('alpha', torch.tensor([alpha]))
        else:
             self.alpha = None
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return loss.mean()

class DualBranchLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = kwargs
        self.register_buffer('pos_weight', torch.tensor([kwargs.get('binary_pos_weight', 1.0)]))
        weights = kwargs.get('fine_class_weight', [1.0, 1.0, 1.0])
        self.register_buffer('fine_class_weight', torch.tensor(weights, dtype=torch.float32))
        
        # 根据配置选择Loss类型
        loss_type = kwargs.get('loss_type', 'asymmetric')
        if loss_type == 'asymmetric':
            self.fine_loss = AsymmetricLoss(
                gamma_neg=kwargs.get('asym_gamma_neg', 2),
                gamma_pos=kwargs.get('asym_gamma_pos', 1),
                clip=kwargs.get('asym_clip', 0.05),
                class_weights=self.fine_class_weight
            )
        elif loss_type == 'focal':
            self.fine_loss = FocalLoss(
                gamma=kwargs.get('focal_gamma', 2.0),
                alpha=kwargs.get('focal_alpha', None),
                class_weights=self.fine_class_weight
            )
        else:
            # Default to Asymmetric
             self.fine_loss = AsymmetricLoss(class_weights=self.fine_class_weight)

    def forward(self, fine_logits, low_vis_logit, targets):
        # 1. 二分类 Loss
        binary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        l_bin = binary_loss_fn(low_vis_logit, (targets <= 1).float().unsqueeze(1))

        # 2. 多分类 Loss
        l_fine = self.fine_loss(fine_logits, targets)

        # 3. 一致性 Loss
        probs = F.softmax(fine_logits, dim=1)
        l_con = (torch.sigmoid(low_vis_logit) * probs[:, 2:3]).mean()

        # 4. FP 抑制
        is_clear = (targets == 2).float()
        fm_prob = probs[:, 0] + probs[:, 1]
        l_fp = torch.mean((fm_prob * is_clear * (fm_prob > self.cfg.get('fp_threshold', 0.5)).float()) ** 2)

        # 5. Recall Boost
        is_fog = (targets == 0).float()
        is_mist = (targets == 1).float()
        l_fb = torch.mean((1 - probs[:, 0]) ** 2 * is_fog)
        l_mb = torch.mean((1 - probs[:, 1]) ** 2 * is_mist)

        total = (
            self.cfg.get('alpha_binary', 1.0) * l_bin +
            self.cfg.get('alpha_fine', 1.0) * l_fine +
            self.cfg.get('alpha_consistency', 0.5) * l_con +
            self.cfg.get('alpha_fp', 3.0) * l_fp +
            self.cfg.get('alpha_fog_boost', 0.1) * l_fb +
            self.cfg.get('alpha_mist_boost', 0.1) * l_mb
        )
        # 返回 total 和 dict 方便打印
        return total, {'bin': l_bin.item(), 'fine': l_fine.item(), 'fb': l_fb.item()}

class PhysicsConstrainedRegLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, reg_pred, reg_target_log, raw_vis):
        mask = (raw_vis < 2000)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=reg_pred.device)
        return self.alpha * F.huber_loss(reg_pred.view(-1)[mask], reg_target_log[mask], delta=1.0)

# ==========================================
# 4. 模型定义 (保持脚本1的定义)
# ==========================================
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=3):
        super().__init__()
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.xavier_normal_(self.cheby_coeffs)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.base_activation = nn.SiLU()

    def forward(self, x):
        x = torch.tanh(self.layer_norm(x))
        vals = [torch.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            vals.append(2 * x * vals[-1] - vals[-2])
        cheby_stack = torch.stack(vals, dim=-1)
        out = torch.einsum("bid,iod->bo", cheby_stack, self.cheby_coeffs)
        return self.base_activation(out)

class _FogDiagnosticFeatures(nn.Module):
    def __init__(self, window_size=12, dyn_vars_count=25):
        super().__init__()
        self.idx = {'rh2m': 0, 't2m': 1, 'precip': 2, 'sw_rad': 4, 'wspd10': 6, 'lcc': 10, 'rh925': 12, 'dpd': 22, 'inversion': 23, 'zenith': 24}

    def forward(self, x):
        g = lambda n: x[:, :, self.idx[n]]
        f1 = torch.clamp(g('rh2m') / 100 * torch.sigmoid(-3 * (g('dpd') - 1)), 0, 1)
        f2 = torch.exp(-0.5 * ((torch.clamp(g('wspd10'), min=0) - 3.5) / 2.5) ** 2)
        f3 = torch.tanh(g('inversion') / (torch.clamp(g('wspd10'), min=0.5) ** 2 + 0.1) / 2)
        f4 = (g('zenith') > 90).float() * torch.clamp(1 - g('lcc') / 0.3, 0, 1) * (1 - torch.clamp(torch.expm1(g('sw_rad')) / 800, 0, 1))
        f5 = torch.clamp((g('rh2m') - g('rh925')) / 100, -1, 1)
        return torch.nan_to_num(torch.stack([f1, f2, f3, f4, f5], dim=2), nan=0.0)

class ImprovedDualStreamPMSTNet(nn.Module):
    def __init__(self, dyn_vars_count=25, window_size=12, static_cont_dim=5, veg_num_classes=21,
                 hidden_dim=512, num_classes=3, extra_feat_dim=0, dropout=0.2):
        super().__init__()
        self.dyn_vars = dyn_vars_count
        self.window = window_size
        self.static_cont_dim = static_cont_dim
        self.extra_feat_dim = extra_feat_dim

        self.veg_embedding = nn.Embedding(veg_num_classes, 16)
        self.static_encoder = nn.Sequential(ChebyKANLayer(static_cont_dim + 16, 256, degree=3), nn.LayerNorm(256), nn.Linear(256, hidden_dim // 2))

        self.fog_diagnostics = _FogDiagnosticFeatures(window_size, dyn_vars_count)
        self.physics_encoder = nn.Sequential(nn.Conv1d(5, 64, 1), nn.GELU(), nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, hidden_dim // 4))

        self.physical_stream = nn.Sequential(nn.Linear(8, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1))
        self.temporal_stream = nn.GRU(6, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.temporal_proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim))

        self.extra_encoder = nn.Sequential(nn.Linear(extra_feat_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU()) if extra_feat_dim > 0 else None

        fusion_dim = hidden_dim + hidden_dim + hidden_dim // 2 + hidden_dim // 4 + (hidden_dim // 2 if extra_feat_dim > 0 else 0)
        self.fusion_layer = nn.Sequential(nn.Linear(fusion_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), ChebyKANLayer(hidden_dim, hidden_dim, degree=3))

        self.fog_specific = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU())
        self.low_vis_detector = nn.Sequential(nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.15), nn.Linear(hidden_dim, 1))
        self.fine_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.reg_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.low_vis_detector[-1].bias.data.fill_(-0.5)

    def forward(self, x):
        split_dyn = self.dyn_vars * self.window
        split_static = split_dyn + self.static_cont_dim
        
        x_dyn = x[:, :split_dyn].view(-1, self.window, self.dyn_vars)
        x_stat = x[:, split_dyn:split_static]
        x_veg = x[:, split_static].long()
        x_extra = x[:, split_static + 1:] if self.extra_feat_dim > 0 else None

        phy_feats = self.fog_diagnostics(x_dyn).permute(0, 2, 1)
        phy_feat = self.physics_encoder(phy_feats)
        
        veg_emb = self.veg_embedding(torch.clamp(x_veg, 0, 20))
        stat_feat = self.static_encoder(torch.cat([x_stat, veg_emb], dim=1))
        
        phys_vars = x_dyn[:, -1, [0, 1, 10, 12, 19, 20, 22, 23]]
        phys_stream = self.physical_stream(phys_vars)
        
        temp_out, _ = self.temporal_stream(x_dyn[:, :, [2, 3, 4, 5, 6, 7]])
        temp_feat = self.temporal_proj(temp_out[:, -1, :])
        
        parts = [phys_stream, temp_feat, stat_feat, phy_feat]
        if x_extra is not None and self.extra_encoder is not None:
            parts.append(self.extra_encoder(x_extra))
        
        emb = torch.cat(parts, dim=1)
        emb = torch.tanh(emb) # 强行压缩到 [-1, 1] 以适应 KAN
        emb = self.fusion_layer(emb)

        emb = self.fusion_layer(torch.cat(parts, dim=1))
        fog_emb = self.fog_specific(emb)
        
        fine_logits = self.fine_classifier(emb)
        reg_out = self.reg_head(emb)
        low_vis_logit = self.low_vis_detector(torch.cat([emb, fog_emb], dim=1))
        
        return fine_logits, reg_out, low_vis_logit

# ==========================================
# 5. 数据集与加载 (严格保持脚本1的mmap和path逻辑)
# ==========================================
class PMSTDataset(Dataset):
    def __init__(self, X_path, y_cls, y_reg, y_raw, scaler=None, window_size=12, use_fe=True, indices=None):
        self.X = np.load(X_path, mmap_mode='r') # 保持 mmap，减少内存占用
        if indices is not None:
            self.indices = np.asarray(indices)
        else:
            self.indices = np.arange(len(self.X))
            
        self.y_cls = torch.as_tensor(y_cls[self.indices], dtype=torch.long)
        self.y_reg = torch.as_tensor(y_reg[self.indices], dtype=torch.float32)
        self.y_raw = torch.as_tensor(y_raw[self.indices], dtype=torch.float32)
        
        self.split_dyn = window_size * 25
        self.split_static = self.split_dyn + 5
        self.use_fe = use_fe
        self.scaler = scaler
        
        self.log_mask = np.zeros(self.split_dyn, dtype=bool)
        for t in range(window_size):
            for i in [2, 4, 9]:
                self.log_mask[t * 25 + i] = True

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.X[real_idx] # mmap 读取
        feats = row[:self.split_static].astype(np.float32)
        
        feats[:self.split_dyn] = np.where(self.log_mask, np.log1p(np.maximum(feats[:self.split_dyn], 0)), feats[:self.split_dyn])
        if self.scaler is not None:
            feats = (feats - self.scaler.center_) / (self.scaler.scale_ + 1e-8)
            
        veg_cat = np.array([row[self.split_static]], dtype=np.float32)
        final = np.concatenate([np.clip(feats, -10, 10), veg_cat])
        if self.use_fe:
            extra = row[self.split_static + 1:].astype(np.float32)
            final = np.concatenate([final, np.clip(extra, -10, 10)])
        
        final = np.nan_to_num(final, nan=0.0, posinf=10.0, neginf=-10.0)
            
        return torch.from_numpy(final).float(), self.y_cls[idx], self.y_reg[idx], self.y_raw[idx]

def _labels_from_raw(y_raw):
    y_raw = np.asarray(y_raw, dtype=np.float32)
    if np.max(y_raw) < 100: y_raw = y_raw * 1000
    y_cls = np.zeros(len(y_raw), dtype=np.int64)
    y_cls[y_raw >= 500] = 1
    y_cls[y_raw >= 1000] = 2
    return y_raw, y_cls

def load_data(data_dir, scaler=None, rank=0, local_rank=0, device=None, reuse_scaler=False, win_size=12, world_size=1, exp_id=None):
    # 1. 安全地复制数据 (Script 1)
    path_X = copy_to_local(os.path.join(data_dir, 'X_train.npy'), local_rank, world_size, exp_id)
    path_y = copy_to_local(os.path.join(data_dir, 'y_train.npy'), local_rank, world_size, exp_id)
    
    # 2. 加载标签
    y_raw_all = np.load(path_y)
    y_raw_all, y_cls_all = _labels_from_raw(y_raw_all)
    y_reg_all = np.log1p(y_raw_all)

    # 3. Scaler 拟合 (Script 1 Optimized)
    feat_dim = win_size * 25 + 5
    if scaler is None and not reuse_scaler:
        if rank == 0:
            print("[Scaler] Fitting RobustScaler on rank 0 (optimized)...", flush=True)
            X_mmap = np.load(path_X, mmap_mode='r')
            subset_size = min(50000, len(X_mmap))
            start_idx = len(X_mmap) // 2
            sub = X_mmap[start_idx : start_idx + subset_size, :feat_dim].astype(np.float32)
            
            log_mask = np.zeros(win_size * 25, dtype=bool)
            for t in range(win_size):
                for i in [2, 4, 9]: log_mask[t * 25 + i] = True
            
            sub[:, :win_size * 25] = np.where(log_mask, np.log1p(np.maximum(sub[:, :win_size * 25], 0)), sub[:, :win_size * 25])
            
            scaler_obj = RobustScaler().fit(sub)
            center_t = torch.tensor(scaler_obj.center_, dtype=torch.float64, device=device)
            scale_t = torch.tensor(scaler_obj.scale_, dtype=torch.float64, device=device)
            print("[Scaler] Done fitting.", flush=True)
        else:
            center_t = torch.zeros(feat_dim, dtype=torch.float64, device=device)
            scale_t = torch.zeros(feat_dim, dtype=torch.float64, device=device)

        if world_size > 1:
            dist.broadcast(center_t, src=0)
            dist.broadcast(scale_t, src=0)

        if rank != 0:
            scaler = RobustScaler()
            scaler.center_ = center_t.cpu().numpy()
            scaler.scale_ = scale_t.cpu().numpy()
        else:
            scaler = scaler_obj

    # 4. 数据集切分
    val_X_path = os.path.join(data_dir, 'X_val.npy')
    val_y_path = os.path.join(data_dir, 'y_val.npy')
    
    use_separate_val = False
    if local_rank == 0 and os.path.exists(val_X_path):
        use_separate_val = True
    flag_tensor = torch.tensor([float(use_separate_val)], device=device)
    if world_size > 1: dist.broadcast(flag_tensor, src=0)
    use_separate_val = bool(flag_tensor.item())

    if use_separate_val:
        path_Xv = copy_to_local(val_X_path, local_rank, world_size, exp_id)
        path_yv = copy_to_local(val_y_path, local_rank, world_size, exp_id)
        y_raw_v, y_cls_v = _labels_from_raw(np.load(path_yv))
        y_reg_v = np.log1p(y_raw_v)
        
        tr_ds = PMSTDataset(path_X, y_cls_all, y_reg_all, y_raw_all, scaler, win_size, CONFIG['USE_FEATURE_ENGINEERING'])
        val_ds = PMSTDataset(path_Xv, y_cls_v, y_reg_v, y_raw_v, scaler, win_size, CONFIG['USE_FEATURE_ENGINEERING'])
    else:
        n_total = len(y_cls_all)
        n_val = max(1, int(n_total * CONFIG['VAL_SPLIT_RATIO']))
        rng = np.random.default_rng(seed=42)
        all_idx = rng.permutation(n_total)
        val_idx = all_idx[:n_val]
        tr_idx = all_idx[n_val:]
        
        if rank == 0: print(f"[Data] Split train={len(tr_idx)}, val={len(val_idx)}", flush=True)
        tr_ds = PMSTDataset(path_X, y_cls_all, y_reg_all, y_raw_all, scaler, win_size, CONFIG['USE_FEATURE_ENGINEERING'], indices=tr_idx)
        val_ds = PMSTDataset(path_X, y_cls_all, y_reg_all, y_raw_all, scaler, win_size, CONFIG['USE_FEATURE_ENGINEERING'], indices=val_idx)

    return tr_ds, val_ds, scaler

# ==========================================
# 6. 综合评估指标 (恢复脚本2的动态阈值搜索)
# ==========================================
class ComprehensiveMetrics:
    def __init__(self, config):
        self.cfg = config
        self.best_thresholds = {'fog': 0.5, 'mist': 0.5}

    def calculate_f2(self, targets, preds, cls_id):
        y_true = (targets == cls_id)
        y_pred = (preds == cls_id)
        tp = (y_true & y_pred).sum()
        fp = (~y_true & y_pred).sum()
        fn = (y_true & ~y_pred).sum()
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f2 = 5 * p * r / (4 * p + r + 1e-6)
        return p, r, f2

    def search_thresholds(self, probs, targets, rank=0):
        # 简化版三阶段搜索
        fog_ths = np.arange(self.cfg['THRESHOLD_FOG_MIN'], self.cfg['THRESHOLD_FOG_MAX'], self.cfg['THRESHOLD_FOG_STEP'])
        mist_ths = np.arange(self.cfg['THRESHOLD_MIST_MIN'], self.cfg['THRESHOLD_MIST_MAX'], self.cfg['THRESHOLD_MIST_STEP'])
        
        best_score = -1
        best_f, best_m = 0.5, 0.5
        
        # 仅在rank0计算，避免重复开销 (虽然数据是all_gather后的)
        if rank == 0:
            print("[Metrics] Searching optimal thresholds...", flush=True)
            for f_th in fog_ths:
                for m_th in mist_ths:
                    # 预测逻辑
                    preds = np.full(len(targets), 2)
                    # Fog priority
                    is_fog = (probs[:, 0] > f_th) & (probs[:, 0] > probs[:, 1])
                    preds[is_fog] = 0
                    # Mist
                    is_mist = (probs[:, 1] > m_th) & (~is_fog)
                    preds[is_mist] = 1
                    
                    # 快速计算Recall Check
                    p_c, r_c, f2_c = self.calculate_f2(targets, preds, 2)
                    p_f, r_f, f2_f = self.calculate_f2(targets, preds, 0)
                    p_m, r_m, f2_m = self.calculate_f2(targets, preds, 1)

                    # Phase 1 Checks
                    if (r_c >= self.cfg['MIN_CLEAR_ACC'] and 
                        p_f >= self.cfg['MIN_FOG_PRECISION'] and r_f >= self.cfg['MIN_FOG_RECALL'] and
                        p_m >= self.cfg['MIN_MIST_PRECISION'] and r_m >= self.cfg['MIN_MIST_RECALL']):
                        
                        score = (self.cfg['SCORE_PHASE1_FOG']*f2_f + 
                                 self.cfg['SCORE_PHASE1_MIST']*f2_m + 
                                 self.cfg['SCORE_PHASE1_CLEAR']*f2_c)
                        if score > best_score:
                            best_score, best_f, best_m = score, f_th, m_th
        
        # 其他rank等待，实际应用中可以在eval函数外广播，这里简化处理直接返回默认
        return best_f, best_m, best_score

    @torch.no_grad()
    def evaluate(self, model, loader, device, rank=0):
        model.eval()
        all_probs, all_targets, all_low_vis = [], [], []
        
        for bx, by, _, _ in loader:
            bx = bx.to(device, non_blocking=True)
            fine_logits, _, low_vis_logit = model(bx)
            probs = F.softmax(fine_logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(by.numpy())
            all_low_vis.append(torch.sigmoid(low_vis_logit).cpu().numpy())
            
        probs = np.vstack(all_probs)
        targets = np.concatenate(all_targets)
        low_vis_probs = np.concatenate(all_low_vis)
        
        # 搜索阈值 (简单起见，仅使用 Phase 1 逻辑，或者使用之前保存的最佳阈值)
        f_th, m_th, score = self.search_thresholds(probs, targets, rank)
        self.best_thresholds = {'fog': f_th, 'mist': m_th}
        
        # 生成最终预测
        preds = np.full(len(targets), 2)
        is_fog = (probs[:, 0] > f_th) & (probs[:, 0] > probs[:, 1])
        preds[is_fog] = 0
        is_mist = (probs[:, 1] > m_th) & (~is_fog)
        preds[is_mist] = 1
        
        # 计算详细指标
        p0, r0, f2_0 = self.calculate_f2(targets, preds, 0) # Fog
        p1, r1, f2_1 = self.calculate_f2(targets, preds, 1) # Mist
        p2, r2, f2_2 = self.calculate_f2(targets, preds, 2) # Clear
        acc = accuracy_score(targets, preds)
        
        # Low Vis TS
        lv_true = (targets <= 1).astype(int)
        lv_pred = (preds <= 1).astype(int)
        tp = (lv_true & lv_pred).sum()
        den = (lv_true | lv_pred).sum()
        ts = tp / (den + 1e-6)
        
        if rank == 0:
            print(f"\n  [Eval] Th=({f_th:.2f}, {m_th:.2f}) | CompScore={score:.4f}")
            print(f"  Fog : P={p0:.3f} R={r0:.3f} F2={f2_0:.3f}")
            print(f"  Mist: P={p1:.3f} R={r1:.3f} F2={f2_1:.3f}")
            print(f"  LowVis TS={ts:.3f} | Acc={acc:.4f}", flush=True)
            
        return {
            'composite': score,
            'fog_f2': f2_0,
            'mist_f2': f2_1,
            'ts': ts,
            'thresholds': self.best_thresholds
        }

# ==========================================
# 7. 训练与评估流程 (合并版)
# ==========================================
def save_checkpoint(model, optimizer, scaler, metrics, step, tag, rank, exp_id, subtype="best"):
    if rank != 0: return
    ckpt_dir = CONFIG['SAVE_CKPT_DIR']
    os.makedirs(ckpt_dir, exist_ok=True)
    raw_model = model.module if hasattr(model, 'module') else model
    
    path = os.path.join(ckpt_dir, f"{exp_id}_{tag}_{subtype}.pt")
    
    # 保存模型
    torch.save({
        'step': step,
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metrics': metrics,
    }, path)
    
    # 按照脚本2的逻辑保存阈值和scaler
    if 'thresholds' in metrics:
        th_path = os.path.join(ckpt_dir, f"{exp_id}_{tag}_{subtype}_thresholds.pkl")
        joblib.dump(metrics['thresholds'], th_path)
    
    print(f"  [Ckpt] Saved {subtype} to {os.path.basename(path)}", flush=True)

def train_stage(tag, model, tr_ds, val_ds, optimizer, loss_fn, reg_loss_fn, grad_scaler, device, rank, world_size, total_steps, val_interval, batch_size, fog_ratio, mist_ratio, grad_accum, exp_id):
    metrics_engine = ComprehensiveMetrics(CONFIG)
    
    # 记录最佳分数 (Script 2 Logic)
    best_scores = {
        'composite': -1.0,
        'fog_f2': -1.0,
        'mist_f2': -1.0,
        'ts': -1.0
    }
    
    step = 0
    total_opt_steps = total_steps // grad_accum
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[p['lr'] for p in optimizer.param_groups], total_steps=total_opt_steps, pct_start=0.1)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)

    while step < total_steps:
        # 使用 StratifiedBalancedBatchSampler
        sampler = StratifiedBalancedBatchSampler(tr_ds, batch_size, fog_ratio, mist_ratio, rank, world_size, seed=42+step)
        loader = DataLoader(tr_ds, batch_sampler=sampler, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)
        
        model.train()
        optimizer.zero_grad()
        
        for bx, by, blog, braw in loader:
            step += 1
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            blog, braw = blog.to(device, non_blocking=True), braw.to(device, non_blocking=True)
            
            #with autocast():
                # 注意模型返回顺序: fine, reg, bin
            fine, reg, bin_out = model(bx)
            l_dual, l_dict = loss_fn(fine, bin_out, by)
            l_reg = reg_loss_fn(reg, blog, braw)
            loss = (l_dual + CONFIG['REG_LOSS_ALPHA'] * l_reg) / grad_accum
            
            #grad_scaler.scale(loss).backward()
            loss.backward()
            if step % grad_accum == 0:
                #grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['GRAD_CLIP_NORM'])
                #grad_scaler.step(optimizer)
                #grad_scaler.update()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
            if rank == 0 and step % 100 == 0:
                print(f"\r[{tag}] Step {step}/{total_steps} Loss={loss.item()*grad_accum:.4f} FB={l_dict.get('fb',0):.4f}", end="", flush=True)
                
            if step % val_interval == 0:
                if rank == 0: print("")
                res = metrics_engine.evaluate(model, val_loader, device, rank)
                model.train()
                
                # 多策略保存
                if res['composite'] > best_scores['composite']:
                    best_scores['composite'] = res['composite']
                    save_checkpoint(model, optimizer, None, res, step, tag, rank, exp_id, "best_composite")
                
                if res['fog_f2'] > best_scores['fog_f2']:
                    best_scores['fog_f2'] = res['fog_f2']
                    save_checkpoint(model, optimizer, None, res, step, tag, rank, exp_id, "best_fog")
                    
                if res['mist_f2'] > best_scores['mist_f2']:
                    best_scores['mist_f2'] = res['mist_f2']
                    save_checkpoint(model, optimizer, None, res, step, tag, rank, exp_id, "best_mist")

                if res['ts'] > best_scores['ts']:
                    best_scores['ts'] = res['ts']
                    save_checkpoint(model, optimizer, None, res, step, tag, rank, exp_id, "best_ts")
                
            if step >= total_steps: break
            
    return best_scores['composite']

def force_cleanup_shm():
    target_dir = "/dev/shm"
    if not os.path.exists(target_dir): return
    try:
        uid = os.getuid()
        for fname in os.listdir(target_dir):
            fpath = os.path.join(target_dir, fname)
            if os.path.isfile(fpath) and os.stat(fpath).st_uid == uid:
                if fname.endswith('.npy') or fname.endswith('.tmp'):
                    try:
                        os.remove(fpath)
                    except:
                        pass
    except Exception:
        pass

# ==========================================
# 8. Main
# ==========================================
def main():
    l_rank, g_rank, w_size = init_distributed()
    if l_rank == 0: force_cleanup_shm()
    if w_size > 1: dist.barrier()
    
    device = torch.device(f"cuda:{l_rank}")
    exp_id = CONFIG['EXPERIMENT_ID']
    
    if g_rank == 0:
        os.makedirs(CONFIG['SAVE_CKPT_DIR'], exist_ok=True)
        # 保存scaler的目录
        os.makedirs(os.path.join(BASE_PATH, 'scalers'), exist_ok=True)
        print(f"[Main] Start Exp: {exp_id}, World: {w_size}", flush=True)

    model = ImprovedDualStreamPMSTNet(
        window_size=CONFIG['WINDOW_SIZE'],
        hidden_dim=CONFIG['MODEL_HIDDEN_DIM'],
        num_classes=CONFIG['MODEL_NUM_CLASSES'],
        extra_feat_dim=CONFIG['FE_EXTRA_DIMS'] if CONFIG['USE_FEATURE_ENGINEERING'] else 0,
        dropout=CONFIG['MODEL_DROPOUT']
    ).to(device)

    if CONFIG['S1_PRETRAINED_PATH'] and os.path.exists(CONFIG['S1_PRETRAINED_PATH']):
        ckpt = torch.load(CONFIG['S1_PRETRAINED_PATH'], map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        if g_rank == 0: print(f"[Main] Loaded pretrained model.", flush=True)

    if w_size > 1:
        model = DDP(model, device_ids=[l_rank], find_unused_parameters=False)

    # 使用恢复后的灵活 Loss
    loss_fn = DualBranchLoss(
        loss_type=CONFIG['LOSS_TYPE'],
        binary_pos_weight=CONFIG['BINARY_POS_WEIGHT'],
        fine_class_weight=[CONFIG['FINE_CLASS_WEIGHT_FOG'], CONFIG['FINE_CLASS_WEIGHT_MIST'], CONFIG['FINE_CLASS_WEIGHT_CLEAR']],
        asym_gamma_neg=CONFIG['ASYM_GAMMA_NEG'],
        asym_gamma_pos=CONFIG['ASYM_GAMMA_POS'],
        asym_clip=CONFIG['ASYM_CLIP'],
        focal_gamma=CONFIG['FOCAL_GAMMA'],
        focal_alpha=CONFIG['FOCAL_ALPHA'],
        alpha_binary=CONFIG['LOSS_ALPHA_BINARY'],
        alpha_fine=CONFIG['LOSS_ALPHA_FINE'],
        alpha_consistency=CONFIG['LOSS_ALPHA_CONSISTENCY'],
        alpha_fp=CONFIG['LOSS_ALPHA_FP'],
        alpha_fog_boost=CONFIG['LOSS_ALPHA_FOG_BOOST'],
        alpha_mist_boost=CONFIG['LOSS_ALPHA_MIST_BOOST'],
        fp_threshold=CONFIG['LOSS_FP_THRESHOLD']
    ).to(device)
    
    reg_loss_fn = PhysicsConstrainedRegLoss(alpha=CONFIG['REG_LOSS_ALPHA']).to(device)

    # --- Stage 1 ---
    scaler = None
    if not CONFIG['SKIP_STAGE1']:
        if g_rank == 0: print("=== Stage 1 Training ===", flush=True)
        tr_ds, val_ds, scaler = load_data(CONFIG['S1_DATA_DIR'], None, g_rank, l_rank, device, False, CONFIG['WINDOW_SIZE'], w_size, exp_id)
        
        # 保存scaler
        if g_rank == 0:
            joblib.dump(scaler, os.path.join(BASE_PATH, f'scalers/scaler_{exp_id}_s1.pkl'))
        
        opt = optim.AdamW(model.parameters(), lr=CONFIG['S1_LR_BACKBONE'], weight_decay=CONFIG['S1_WEIGHT_DECAY'])
        train_stage('S1', model, tr_ds, val_ds, opt, loss_fn, reg_loss_fn, None, device, g_rank, w_size,
                   CONFIG['S1_TOTAL_STEPS'], CONFIG['S1_VAL_INTERVAL'], CONFIG['S1_BATCH_SIZE'],
                   CONFIG['S1_FOG_RATIO'], CONFIG['S1_MIST_RATIO'], CONFIG['S1_GRAD_ACCUM'], exp_id)
        del tr_ds, val_ds, opt
        gc.collect()

    # --- Stage 2 ---
    if g_rank == 0: print("=== Stage 2 Fine-tuning ===", flush=True)
    tr_ds_s2, val_ds_s2, scaler = load_data(CONFIG['S2_DATA_DIR'], scaler, g_rank, l_rank, device, (scaler is not None), CONFIG['WINDOW_SIZE'], w_size, exp_id)
    
    # 再次保存scaler (如果是Skip S1的情况，这里是新建的)
    if g_rank == 0 and CONFIG['SKIP_STAGE1']:
        joblib.dump(scaler, os.path.join(BASE_PATH, f'scalers/scaler_{exp_id}_s2.pkl'))

    raw_model = model.module if hasattr(model, 'module') else model
    head_ids = list(map(id, list(raw_model.fine_classifier.parameters()) + list(raw_model.low_vis_detector.parameters()) + list(raw_model.reg_head.parameters())))
    backbone_params = [p for p in raw_model.parameters() if id(p) not in head_ids]
    head_params = [p for p in raw_model.parameters() if id(p) in head_ids]
    
    opt_s2 = optim.AdamW([
        {'params': backbone_params, 'lr': CONFIG['S2_LR_BACKBONE']},
        {'params': head_params, 'lr': CONFIG['S2_LR_HEAD']}
    ], weight_decay=CONFIG['S2_WEIGHT_DECAY'])

    train_stage('S2', model, tr_ds_s2, val_ds_s2, opt_s2, loss_fn, reg_loss_fn, None, device, g_rank, w_size,
               CONFIG['S2_TOTAL_STEPS'], CONFIG['S2_VAL_INTERVAL'], CONFIG['S2_BATCH_SIZE'],
               CONFIG['S2_FOG_RATIO'], CONFIG['S2_MIST_RATIO'], CONFIG['S2_GRAD_ACCUM'], exp_id)

    cleanup_temp_files(exp_id)
    if w_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    main()
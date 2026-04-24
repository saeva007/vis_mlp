import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import shutil
import hashlib
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
import contextlib
import sys
import datetime
import time
import joblib

warnings.filterwarnings('ignore')

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

    # ========== Stage 1 训练配置 ==========
    'S1_TOTAL_STEPS':         30000,
    'S1_VAL_INTERVAL':        1000,
    'S1_BATCH_SIZE':          512,
    'S1_GRAD_ACCUM':          2,
    'S1_FOG_RATIO':           0.20,
    'S1_MIST_RATIO':          0.20,
    'S1_LR_BACKBONE':         2e-4,
    'S1_WEIGHT_DECAY':        1e-2,

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
    'LOSS_ALPHA_FOG_BOOST':   0.5,
    'LOSS_ALPHA_MIST_BOOST':  0.2,
    'LOSS_FP_THRESHOLD':      0.5,

    'ASYM_GAMMA_NEG':         2.0,
    'ASYM_GAMMA_POS':         0,
    'ASYM_CLIP':              0.1,

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
    'MODEL_DROPOUT':          0.3,
    'MODEL_NUM_CLASSES':      3,
    'USE_FEATURE_ENGINEERING':True,
    'FE_EXTRA_DIMS':          24,
    'GRAD_CLIP_NORM':         0.5,
    'REG_LOSS_ALPHA':         0.1,
    'VAL_SPLIT_RATIO':        0.1,

    # ========== target_achievement 权重 ==========
    'TARGET_RECALL_500_GOAL':    0.65,
    'TARGET_RECALL_1000_GOAL':   0.75,
    'TARGET_ACCURACY_GOAL':      0.95,
    'TARGET_LOW_VIS_PREC_GOAL':  0.20,
    'TARGET_FPR_GOAL':           0.40,
    'TARGET_W_RECALL_500':       0.30,
    'TARGET_W_RECALL_1000':      0.30,
    'TARGET_W_ACCURACY':         0.25,
    'TARGET_W_LOW_VIS_PREC':     0.10,
    'TARGET_W_FPR':              0.05,
}

# ==========================================
# 1. 基础工具与分布式设置
# ==========================================

def init_distributed():
    local_rank  = int(os.environ.get("LOCAL_RANK",  os.environ.get("SLURM_LOCALID",  0)))
    global_rank = int(os.environ.get("RANK",         os.environ.get("SLURM_PROCID",   0)))
    world_size  = int(os.environ.get("WORLD_SIZE",   os.environ.get("SLURM_NTASKS",   1)))

    torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method='env://',
            # 30 分钟超时：足够长以完成 NCCL 初始化，但不会无限挂死
            timeout=datetime.timedelta(minutes=30)
        )
        if global_rank == 0:
            print("[Dist] Process group initialized.", flush=True)

    return local_rank, global_rank, world_size


def get_available_space(path):
    try:
        stat = os.statvfs(path)
        return stat.f_bavail * stat.f_frsize
    except Exception:
        return 0


def copy_to_local(src_path: str, global_rank: int, local_rank: int,
                  world_size: int, exp_id: str = None) -> str:
    """
    将 NFS 上的数据文件拷贝到本节点 /tmp，加速 IO。
    每个节点的 local_rank==0 负责拷贝，完成后全局 barrier 同步。
    """
    target_dir = "/tmp"
    if exp_id is None:
        exp_id = "default_exp"

    file_hash  = hashlib.md5(f"{exp_id}_{os.path.abspath(src_path)}".encode()).hexdigest()[:8]
    basename   = os.path.basename(src_path)
    local_path = os.path.join(
        target_dir,
        f"{os.path.splitext(basename)[0]}_{file_hash}{os.path.splitext(basename)[1]}"
    )

    # 每个节点的 local_rank==0 负责拷贝到本节点 /tmp
    if local_rank == 0:
        try:
            if not os.path.exists(src_path):
                print(f"[Data-Copy] Warning: Source {src_path} not found.", flush=True)
            else:
                src_size    = os.path.getsize(src_path)
                cache_valid = (os.path.exists(local_path) and
                               os.path.getsize(local_path) == src_size)
                if cache_valid:
                    print(f"[Data-Copy] Cache hit: {local_path}", flush=True)
                else:
                    avail = get_available_space(target_dir)
                    if avail < src_size + 500 * 1024 * 1024:
                        print(f"[Data-Copy] Insufficient space on {target_dir}, using NFS.", flush=True)
                    else:
                        print(f"[Data-Copy] Copying {basename} to {target_dir}...", flush=True)
                        tmp_path = local_path + ".tmp"
                        shutil.copyfile(src_path, tmp_path)
                        os.rename(tmp_path, local_path)
                        print(f"[Data-Copy] Done: {local_path}", flush=True)
        except Exception as e:
            print(f"[Data-Copy] Error: {e}, falling back to NFS.", flush=True)

    # 等待所有节点的 local_rank==0 完成拷贝
    if world_size > 1:
        dist.barrier()

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    else:
        return src_path


def cleanup_temp_files(exp_id: str):
    target_dir = "/tmp"
    if not os.path.exists(target_dir):
        return
    for fname in os.listdir(target_dir):
        if exp_id in fname:
            try:
                os.remove(os.path.join(target_dir, fname))
            except Exception:
                pass

# ==========================================
# 2. 数据采样器
# ==========================================
class StratifiedBalancedBatchSampler(Sampler):
    """
    分层均衡批采样器。
    - 固定 epoch_length=2000，保证各 rank 批次数量完全一致（DDP 要求）
    - 种子基于 seed + rank + epoch，保证确定性且各进程不重复
    - 每个 rank 只从其分片中采样，保证数据不重叠
    """
    def __init__(self, dataset, batch_size, fog_ratio=0.2, mist_ratio=0.2,
                 rank=0, world_size=1, seed=42, epoch_length=2000):
        self.dataset      = dataset
        self.batch_size   = batch_size
        self.rank         = rank
        self.world_size   = world_size
        self.seed         = seed
        self.epoch_length = epoch_length
        self._epoch       = 0

        y = np.array(dataset.y_cls)
        self.n_fog   = max(1, int(batch_size * fog_ratio))
        self.n_mist  = max(1, int(batch_size * mist_ratio))
        self.n_clear = batch_size - self.n_fog - self.n_mist

        self.indices = {
            0: np.where(y == 0)[0],
            1: np.where(y == 1)[0],
            2: np.where(y == 2)[0]
        }

        # 按 rank 分片，保证各进程数据不重叠
        for k in self.indices:
            splits = np.array_split(self.indices[k], world_size)
            self.indices[k] = splits[rank % len(splits)]
            if len(self.indices[k]) == 0:
                # fallback：类别样本极少时直接用全量
                self.indices[k] = np.where(y == k)[0][:1]

    def set_epoch(self, epoch: int):
        """在 epoch 开始时调用，改变随机性但保持确定性"""
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(seed=self.seed + self.rank + self._epoch * 997)
        for _ in range(self.epoch_length):
            f = rng.choice(self.indices[0], size=self.n_fog,   replace=True)
            m = rng.choice(self.indices[1], size=self.n_mist,  replace=True)
            c = rng.choice(self.indices[2], size=self.n_clear, replace=True)
            batch = np.concatenate([f, m, c])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.epoch_length

# ==========================================
# 3. Loss 函数
# ==========================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-7, class_weights=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, targets):
        num_classes     = logits.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()

        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        pos_loss = -targets_one_hot * torch.log(probs) * ((1 - probs) ** self.gamma_pos)
        neg_probs = probs
        neg_loss  = -(1 - targets_one_hot) * torch.log(1 - neg_probs) * (neg_probs ** self.gamma_neg)

        loss = pos_loss + neg_loss
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets].unsqueeze(1)

        return loss.sum(dim=1).mean()


class DualBranchLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = kwargs
        self.register_buffer('pos_weight',       torch.tensor([kwargs.get('binary_pos_weight', 1.0)]))
        weights = kwargs.get('fine_class_weight', [1.0, 1.0, 1.0])
        self.register_buffer('fine_class_weight', torch.tensor(weights, dtype=torch.float32))

        self.fine_loss = AsymmetricLoss(
            gamma_neg=kwargs.get('asym_gamma_neg', 2),
            gamma_pos=kwargs.get('asym_gamma_pos', 0),
            clip=kwargs.get('asym_clip', 0.1),
            class_weights=self.fine_class_weight
        )

    def forward(self, fine_logits, low_vis_logit, targets):
        low_vis_logit = torch.clamp(low_vis_logit, -20, 20)
        l_bin = F.binary_cross_entropy_with_logits(
            low_vis_logit,
            (targets <= 1).float().unsqueeze(1),
            pos_weight=self.pos_weight
        )

        l_fine = self.fine_loss(fine_logits, targets)

        probs    = F.softmax(fine_logits, dim=1)
        is_fog   = (targets == 0).float()
        l_fb     = torch.mean((1 - probs[:, 0]) ** 2 * is_fog)
        is_clear = (targets == 2).float()
        l_fp     = torch.mean((probs[:, 0] + probs[:, 1]) ** 2 * is_clear)

        total = (
            self.cfg.get('alpha_binary',    1.0) * l_bin  +
            self.cfg.get('alpha_fine',      1.0) * l_fine +
            self.cfg.get('alpha_fp',        0.5) * l_fp   +
            self.cfg.get('alpha_fog_boost', 0.2) * l_fb
        )
        return total, {'bin': l_bin.item(), 'fine': l_fine.item()}

# ==========================================
# 4. 模型定义
# ==========================================
class StableChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=2):
        super().__init__()
        self.degree       = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=0.02)
        self.layer_norm   = nn.LayerNorm(input_dim)
        self.act          = nn.SiLU()

    def forward(self, x):
        x = self.layer_norm(x)
        x = torch.tanh(x)
        x = torch.clamp(x, -0.99, 0.99)

        vals = [torch.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            next_val = 2 * x * vals[-1] - vals[-2]
            next_val = torch.clamp(next_val, -5.0, 5.0)
            vals.append(next_val)

        cheby_stack = torch.stack(vals, dim=-1)
        out = torch.einsum("bid,iod->bo", cheby_stack, self.cheby_coeffs)
        return self.act(out)


class SEBlock(nn.Module):
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
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ImprovedDualStreamPMSTNet(nn.Module):
    def __init__(self, dyn_vars_count=25, window_size=12, static_cont_dim=5,
                 veg_num_classes=21, hidden_dim=512, num_classes=3,
                 extra_feat_dim=0, dropout=0.3):
        super().__init__()
        self.dyn_vars = dyn_vars_count
        self.window   = window_size

        # 静态分支
        self.veg_embedding = nn.Embedding(veg_num_classes, 16)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_cont_dim + 16, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, hidden_dim // 4)
        )

        # 物理分支
        self.physics_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim // 4)
        )

        # 时序分支
        self.temporal_input_proj  = nn.Linear(6, hidden_dim)
        self.temporal_stream      = nn.GRU(hidden_dim, hidden_dim, num_layers=2,
                                           batch_first=True, bidirectional=True, dropout=dropout)
        self.se_block             = SEBlock(hidden_dim * 2)
        self.temporal_norm        = nn.LayerNorm(hidden_dim * 2)

        # 融合层
        fusion_dim = (hidden_dim * 2) + (hidden_dim // 4) + (hidden_dim // 4)
        if extra_feat_dim > 0:
            self.extra_encoder = nn.Sequential(
                nn.Linear(extra_feat_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU()
            )
            fusion_dim += hidden_dim // 2
        else:
            self.extra_encoder = None

        self.fusion_kan = StableChebyKANLayer(fusion_dim, hidden_dim, degree=2)
        self.dropout    = nn.Dropout(dropout)

        # 输出头
        self.fine_classifier  = nn.Linear(hidden_dim, num_classes)
        self.low_vis_detector = nn.Linear(hidden_dim, 1)
        self.reg_head         = nn.Linear(hidden_dim, 1)

    def _physics_features(self, x):
        rh2m  = x[:, :, 0]
        t2m   = x[:, :, 1]
        wspd  = torch.clamp(x[:, :, 6], min=0.1)
        dpd   = x[:, :, 22]
        inv   = x[:, :, 23]

        f1 = rh2m / 100.0 * torch.sigmoid(-2 * dpd)
        f2 = 1.0 / (wspd + 1.0)
        f3 = inv * f2
        f4 = (rh2m - x[:, :, 12]) / 100.0
        f5 = x[:, :, 10]

        return torch.stack([f1, f2, f3, f4, f5], dim=2).mean(dim=1)

    def forward(self, x):
        split_dyn    = self.dyn_vars * self.window
        split_static = split_dyn + 5

        x_dyn   = x[:, :split_dyn].view(-1, self.window, self.dyn_vars)
        x_stat  = x[:, split_dyn:split_static]
        x_veg   = x[:, split_static].long()
        x_extra = x[:, split_static + 1:] if self.extra_encoder else None

        phy_feat  = self.physics_encoder(self._physics_features(x_dyn))
        veg_emb   = self.veg_embedding(torch.clamp(x_veg, 0, 20))
        stat_feat = self.static_encoder(torch.cat([x_stat, veg_emb], dim=1))

        imp_vars = x_dyn[:, :, [0, 1, 2, 4, 6, 10]]
        t_in     = self.temporal_input_proj(imp_vars)
        t_out, _ = self.temporal_stream(t_in)

        t_out_se = t_out.permute(0, 2, 1)
        t_out_se = self.se_block(t_out_se)
        t_out_se = t_out_se.permute(0, 2, 1)
        t_feat   = self.temporal_norm(t_out_se.mean(dim=1))

        parts = [t_feat, stat_feat, phy_feat]
        if x_extra is not None and self.extra_encoder:
            parts.append(self.extra_encoder(x_extra))

        emb = torch.cat(parts, dim=1)
        emb = self.fusion_kan(emb)
        emb = self.dropout(emb)

        return self.fine_classifier(emb), self.reg_head(emb), self.low_vis_detector(emb)

# ==========================================
# 5. 数据集
# ==========================================
class PMSTDataset(Dataset):
    def __init__(self, X_path, y_cls, y_raw, scaler=None,
                 window_size=12, use_fe=True, indices=None):
        self.X_path  = X_path
        self.indices = np.asarray(indices) if indices is not None else np.arange(len(y_cls))
        self.y_cls   = torch.as_tensor(y_cls[self.indices], dtype=torch.long)

        clean_raw  = np.maximum(y_raw[self.indices], 0.0)
        self.y_reg = torch.as_tensor(np.log1p(clean_raw), dtype=torch.float32)
        self.y_raw = torch.as_tensor(clean_raw,           dtype=torch.float32)

        self.split_dyn = window_size * 25
        self.scaler    = scaler
        self.use_fe    = use_fe

        self.log_mask = np.zeros(self.split_dyn, dtype=bool)
        for t in range(window_size):
            for i in [2, 4, 9]:
                self.log_mask[t * 25 + i] = True

        # mmap 句柄由 worker 各自懒初始化，避免跨进程共享文件描述符
        self.X = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.X is None:
            self.X = np.load(self.X_path, mmap_mode='r')

        real_idx = self.indices[idx]
        row      = self.X[real_idx]

        feats = row[:self.split_dyn + 5].astype(np.float32)
        feats[:self.split_dyn] = np.where(
            self.log_mask,
            np.log1p(np.maximum(feats[:self.split_dyn], 0)),
            feats[:self.split_dyn]
        )

        if self.scaler:
            feats = (feats - self.scaler.center_) / (self.scaler.scale_ + 1e-6)

        veg   = np.array([row[self.split_dyn + 5]], dtype=np.float32)
        final = np.concatenate([np.clip(feats, -10, 10), veg])

        if self.use_fe:
            extra = row[self.split_dyn + 6:].astype(np.float32)
            final = np.concatenate([final, np.clip(extra, -10, 10)])

        final = np.nan_to_num(final, nan=0.0)
        return torch.from_numpy(final).float(), self.y_cls[idx], self.y_reg[idx], self.y_raw[idx]


def load_data(data_dir, scaler=None, rank=0, local_rank=0, device=None,
              reuse_scaler=False, win_size=12, world_size=1, exp_id=None):
    path_X = copy_to_local(os.path.join(data_dir, 'X_train.npy'), rank, local_rank, world_size, exp_id)
    path_y = copy_to_local(os.path.join(data_dir, 'y_train.npy'), rank, local_rank, world_size, exp_id)

    y_raw = np.load(path_y)
    y_cls = np.zeros(len(y_raw), dtype=np.int64)
    if np.max(y_raw) < 100:
        y_raw = y_raw * 1000
    y_cls[y_raw >= 500]  = 1
    y_cls[y_raw >= 1000] = 2

    scaler_path = os.path.join(CONFIG['SAVE_CKPT_DIR'], f'robust_scaler_w{win_size}.pkl')

    if scaler is None and not reuse_scaler:
        # rank 0 负责拟合并保存 scaler，其他 rank 等待后加载
        if world_size > 1:
            dist.barrier()

        if rank == 0:
            if not os.path.exists(scaler_path):
                print("[Scaler] Fitting (first time, will be cached)...", flush=True)
                X_m = np.load(path_X, mmap_mode='r')
                n_total     = len(X_m)
                max_samples = 200000
                rng_scaler  = np.random.default_rng(seed=42)
                if n_total > max_samples:
                    sample_indices = rng_scaler.choice(n_total, size=max_samples, replace=False)
                    sample_indices.sort()
                else:
                    sample_indices = np.arange(n_total)

                print(f"[Scaler] Sampling {len(sample_indices)} rows...", flush=True)
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
                print("[Scaler] Cached file found, loading.", flush=True)

        # 等待 rank 0 完成 scaler 保存
        if world_size > 1:
            dist.barrier()

        # 等待 NFS 文件可见（最多 60 秒）
        wait_time = 0
        while not os.path.exists(scaler_path) and wait_time < 60:
            time.sleep(1)
            wait_time += 1

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Rank {rank} waited 60s but still cannot see {scaler_path}."
            )

        scaler = joblib.load(scaler_path)

    n_total = len(y_cls)
    n_val   = int(n_total * CONFIG['VAL_SPLIT_RATIO'])
    rng     = np.random.default_rng(seed=42)
    indices = rng.permutation(n_total)

    tr_ds  = PMSTDataset(path_X, y_cls, y_raw, scaler, win_size, True, indices[n_val:])
    val_ds = PMSTDataset(path_X, y_cls, y_raw, scaler, win_size, True, indices[:n_val])

    return tr_ds, val_ds, scaler

# ==========================================
# 6. 评估
# ==========================================

def compute_target_achievement(metrics: dict, cfg: dict) -> float:
    ta = (
        min(metrics['recall_500']          / cfg['TARGET_RECALL_500_GOAL'],    1.0) * cfg['TARGET_W_RECALL_500']  +
        min(metrics['recall_1000']         / cfg['TARGET_RECALL_1000_GOAL'],   1.0) * cfg['TARGET_W_RECALL_1000'] +
        min(metrics['accuracy']            / cfg['TARGET_ACCURACY_GOAL'],      1.0) * cfg['TARGET_W_ACCURACY']    +
        min(metrics['low_vis_precision']   / cfg['TARGET_LOW_VIS_PREC_GOAL'],  1.0) * cfg['TARGET_W_LOW_VIS_PREC']+
        min((1 - metrics['false_positive_rate']) / (1.0 - cfg['TARGET_FPR_GOAL']), 1.0) * cfg['TARGET_W_FPR']
    )
    return float(ta)


class ComprehensiveMetrics:
    def __init__(self, config):
        self.cfg = config
        self.best_th = {'fog': 0.5, 'mist': 0.5}
        self.min_prec_threshold = 0.18
        self.min_clear_recall   = 0.90

    @staticmethod
    def _calc_metrics_per_class(targets, preds, class_id):
        tp = ((preds == class_id) & (targets == class_id)).sum()
        fp = ((preds == class_id) & (targets != class_id)).sum()
        fn = ((preds != class_id) & (targets == class_id)).sum()
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        return float(precision), float(recall)

    def _build_full_metrics(self, probs, targets, f_th, m_th):
        preds = np.full(len(targets), 2, dtype=int)
        is_fog_prob  = probs[:, 0] > f_th
        preds[is_fog_prob] = 0
        is_mist_prob = (probs[:, 1] > m_th) & (~is_fog_prob)
        preds[is_mist_prob] = 1

        p0, r0 = self._calc_metrics_per_class(targets, preds, 0)
        p1, r1 = self._calc_metrics_per_class(targets, preds, 1)
        p2, r2 = self._calc_metrics_per_class(targets, preds, 2)

        accuracy = float((preds == targets).mean())

        pred_low = (preds <= 1)
        true_low = (targets <= 1)
        is_clear = (targets == 2)
        lv_tp    = (pred_low & true_low).sum()
        lv_fp    = (pred_low & ~true_low).sum()
        low_vis_precision = lv_tp / (lv_tp + lv_fp + 1e-6)
        fpr = (pred_low & is_clear).sum() / (is_clear.sum() + 1e-6)

        return {
            'Fog_R':  r0, 'Fog_P':  p0,
            'Mist_R': r1, 'Mist_P': p1,
            'Clear_R': r2, 'Clear_P': p2,
            'recall_500':          r0,
            'recall_1000':         r1,
            'accuracy':            accuracy,
            'low_vis_precision':   float(low_vis_precision),
            'false_positive_rate': float(fpr),
            'preds':               preds,
        }

    def evaluate(self, model, loader, device, rank=0, world_size=1, actual_val_size=None):
        model.eval()
        probs_l, targets_l = [], []

        with torch.no_grad():
            for bx, by, _, _ in loader:
                bx = bx.to(device, non_blocking=True)
                fine, _, _ = model(bx)
                probs_l.append(F.softmax(fine, dim=1))
                targets_l.append(by.to(device))

        local_probs   = torch.cat(probs_l,   dim=0)
        local_targets = torch.cat(targets_l, dim=0)

        if world_size > 1:
            # 对齐各 rank 的数据量（DistributedSampler 最后一个 rank 可能略少）
            local_size = torch.tensor([local_probs.size(0)], dtype=torch.long, device=device)
            max_size   = local_size.clone()
            dist.all_reduce(max_size, op=dist.ReduceOp.MAX)

            if local_size < max_size:
                pad_size    = max_size.item() - local_size.item()
                pad_probs   = torch.zeros((pad_size, local_probs.size(1)),
                                          dtype=local_probs.dtype, device=device)
                pad_targets = torch.full((pad_size,), -1,
                                         dtype=local_targets.dtype, device=device)
                local_probs   = torch.cat([local_probs,   pad_probs],   dim=0)
                local_targets = torch.cat([local_targets, pad_targets], dim=0)

            gathered_probs   = [torch.zeros_like(local_probs)   for _ in range(world_size)]
            gathered_targets = [torch.zeros_like(local_targets) for _ in range(world_size)]
            dist.all_gather(gathered_probs,   local_probs)
            dist.all_gather(gathered_targets, local_targets)

            all_probs   = torch.cat(gathered_probs,   dim=0).cpu().numpy()
            all_targets = torch.cat(gathered_targets, dim=0).cpu().numpy()
        else:
            all_probs   = local_probs.cpu().numpy()
            all_targets = local_targets.cpu().numpy()

        best_ta    = -1.0
        best_stats = None

        if rank == 0:
            n       = actual_val_size if actual_val_size is not None else len(loader.dataset)
            probs   = all_probs[:n]
            targets = all_targets[:n]

            valid_mask = targets >= 0
            probs   = probs[valid_mask]
            targets = targets[valid_mask]

            search_space = np.arange(0.10, 0.65, 0.05)
            print(f"  [Eval] Searching {len(search_space)**2} threshold combinations...", flush=True)

            for f_th in search_space:
                for m_th in search_space:
                    stats = self._build_full_metrics(probs, targets, f_th, m_th)
                    if (stats['Fog_P']   < self.min_prec_threshold or
                        stats['Mist_P']  < self.min_prec_threshold or
                        stats['Clear_R'] < self.min_clear_recall):
                        continue
                    ta = compute_target_achievement(stats, self.cfg)
                    if ta > best_ta:
                        best_ta    = ta
                        best_stats = stats
                        self.best_th = {'fog': float(f_th), 'mist': float(m_th)}

            if best_stats is None:
                # 无约束 fallback
                preds     = np.argmax(probs, axis=1)
                p0, r0    = self._calc_metrics_per_class(targets, preds, 0)
                p1, r1    = self._calc_metrics_per_class(targets, preds, 1)
                p2, r2    = self._calc_metrics_per_class(targets, preds, 2)
                accuracy  = float((preds == targets).mean())
                pred_low  = (preds <= 1)
                true_low  = (targets <= 1)
                is_clear  = (targets == 2)
                lv_prec   = (pred_low & true_low).sum() / (pred_low.sum() + 1e-6)
                fpr       = (pred_low & is_clear).sum() / (is_clear.sum() + 1e-6)
                best_stats = {
                    'Fog_R': r0, 'Fog_P': p0,
                    'Mist_R': r1, 'Mist_P': p1,
                    'Clear_R': r2, 'Clear_P': p2,
                    'recall_500': r0, 'recall_1000': r1,
                    'accuracy': accuracy,
                    'low_vis_precision': float(lv_prec),
                    'false_positive_rate': float(fpr),
                    'preds': preds,
                }
                best_ta = compute_target_achievement(best_stats, self.cfg)
                print("  [Eval] WARN: No threshold met constraints, using argmax fallback.", flush=True)
            else:
                print(f"  [Eval] Best Th -> Fog:{self.best_th['fog']:.2f}, Mist:{self.best_th['mist']:.2f}", flush=True)

            print(
                f"  [Eval] Fog  R={best_stats['Fog_R']:.3f} P={best_stats['Fog_P']:.3f} | "
                f"Mist R={best_stats['Mist_R']:.3f} P={best_stats['Mist_P']:.3f} | "
                f"Clear R={best_stats['Clear_R']:.3f} | "
                f"Acc={best_stats['accuracy']:.3f} | "
                f"LVPrec={best_stats['low_vis_precision']:.3f} | "
                f"FPR={best_stats['false_positive_rate']:.3f}",
                flush=True
            )
            print(f"  [Eval] target_achievement = {best_ta:.4f}", flush=True)

        # 将 best_ta 广播给所有进程，保证一致性
        if world_size > 1:
            ta_tensor = torch.tensor([best_ta], dtype=torch.float32, device=device)
            dist.broadcast(ta_tensor, src=0)
            best_ta = ta_tensor.item()
            dist.barrier()

        return {
            'score':      best_ta,
            'stats':      best_stats,
            'thresholds': self.best_th
        }

# ==========================================
# 7. 训练
# ==========================================

def save_checkpoint(model, path, rank, world_size):
    if rank != 0:
        return
    state = model.module.state_dict() if world_size > 1 else model.state_dict()
    torch.save(state, path)
    print(f"  [Ckpt] Saved -> {os.path.basename(path)}", flush=True)


def load_checkpoint(model, path, rank, world_size, device):
    """从 path 加载权重到 model（自动处理 DDP wrapper）。所有 rank 都执行。"""
    if rank == 0:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Ckpt] Checkpoint not found: {path}")
        print(f"[Ckpt] Loading checkpoint: {os.path.basename(path)}", flush=True)

    if world_size > 1:
        dist.barrier()

    state_dict = torch.load(path, map_location=device)
    target = model.module if world_size > 1 else model
    missing, unexpected = target.load_state_dict(state_dict, strict=False)

    if rank == 0:
        if missing:
            print(f"  [Ckpt] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
        if unexpected:
            print(f"  [Ckpt] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}", flush=True)
        print(f"  [Ckpt] Load complete.", flush=True)

    if world_size > 1:
        dist.barrier()


def train_stage(tag, model, tr_ds, val_ds, optimizer, loss_fn, device,
                rank, world_size, total_steps, val_int, batch_size,
                grad_accum, fog_ratio, mist_ratio, exp_id):

    sampler = StratifiedBalancedBatchSampler(
        tr_ds, batch_size,
        fog_ratio=fog_ratio, mist_ratio=mist_ratio,
        rank=rank, world_size=world_size,
        epoch_length=2000,   # 固定值，保证各 rank 批次数量完全一致
    )
    loader = DataLoader(
        tr_ds,
        batch_sampler=sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_sampler = (DistributedSampler(val_ds, num_replicas=world_size,
                                       rank=rank, shuffle=False)
                   if world_size > 1 else None)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, sampler=val_sampler, pin_memory=True
    )
    actual_val_size = len(val_ds)

    metrics_evaluator = ComprehensiveMetrics(CONFIG)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_score       = -1.0
    best_fog_recall  = -1.0
    best_mist_recall = -1.0

    ckpt_dir = CONFIG['SAVE_CKPT_DIR']
    path_best_score       = os.path.join(ckpt_dir, f"{exp_id}_{tag}_best_score.pt")
    path_best_fog_recall  = os.path.join(ckpt_dir, f"{exp_id}_{tag}_best_fog_recall.pt")
    path_best_mist_recall = os.path.join(ckpt_dir, f"{exp_id}_{tag}_best_mist_recall.pt")
    path_latest           = os.path.join(ckpt_dir, f"{exp_id}_{tag}_latest.pt")

    if rank == 0:
        print(f"\n[{tag}] Training started. total_steps={total_steps}, "
              f"grad_accum={grad_accum}, batch_size={batch_size}", flush=True)

    step         = 0
    batch_count  = 0
    pseudo_epoch = 0
    iterator     = iter(loader)
    model.train()

    while step < total_steps:
        # -------- 获取 batch --------
        try:
            bx, by, blog, braw = next(iterator)
        except StopIteration:
            pseudo_epoch += 1
            sampler.set_epoch(pseudo_epoch)
            iterator = iter(loader)
            bx, by, blog, braw = next(iterator)

        bx, by, blog = bx.to(device), by.to(device), blog.to(device)
        batch_count += 1

        # -------- 前向 --------
        fine, reg, bin_out = model(bx)
        l_dual, loss_dict  = loss_fn(fine, bin_out, by)
        l_reg = F.mse_loss(reg.view(-1), blog)
        loss  = l_dual + CONFIG['REG_LOSS_ALPHA'] * l_reg
        loss  = loss / grad_accum

        # -------- 梯度累积：使用 no_sync() 避免中间步骤触发 all_reduce --------
        # DDP 要求所有进程对称调用 backward()，绝不能在 backward() 前 continue
        is_last_accum_step = (batch_count % grad_accum == 0)

        if world_size > 1 and not is_last_accum_step:
            ctx = model.no_sync()
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            loss.backward()

        # -------- 只在累积完成后更新参数 --------
        if is_last_accum_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), CONFIG['GRAD_CLIP_NORM']
            )

            if torch.isfinite(grad_norm):
                optimizer.step()
                scheduler.step()
            else:
                if rank == 0:
                    print(f"\n[WARNING] Step {step}: NaN/Inf grad norm ({grad_norm:.4f}), "
                          f"skipping optimizer step.", flush=True)

            optimizer.zero_grad()
            step += 1

            # -------- 进度打印 --------
            if rank == 0 and step % 50 == 0:
                print(
                    f"\r[{tag}] Step {step:>6}/{total_steps} | "
                    f"Loss={loss.item() * grad_accum:.4f} | "
                    f"bin={loss_dict['bin']:.4f} fine={loss_dict['fine']:.4f} | "
                    f"LR={scheduler.get_last_lr()[0]:.2e}",
                    end="", flush=True
                )

            # -------- 验证 --------
            if step % val_int == 0:
                if rank == 0:
                    print(f"\n[{tag}] === Validation at step {step} ===", flush=True)

                res   = metrics_evaluator.evaluate(
                    model, val_loader, device, rank, world_size,
                    actual_val_size=actual_val_size
                )
                model.train()

                ta    = res['score']
                stats = res['stats']

                save_checkpoint(model, path_latest, rank, world_size)

                if rank == 0 and stats is not None:
                    fog_r  = stats.get('Fog_R',  -1.0)
                    mist_r = stats.get('Mist_R', -1.0)

                    if ta > best_score:
                        best_score = ta
                        save_checkpoint(model, path_best_score, rank, world_size)
                        print(f"  [Ckpt] ★ New best target_achievement = {best_score:.4f}", flush=True)

                    if fog_r > best_fog_recall:
                        best_fog_recall = fog_r
                        save_checkpoint(model, path_best_fog_recall, rank, world_size)
                        print(f"  [Ckpt] ★ New best Fog Recall = {best_fog_recall:.4f}", flush=True)

                    if mist_r > best_mist_recall:
                        best_mist_recall = mist_r
                        save_checkpoint(model, path_best_mist_recall, rank, world_size)
                        print(f"  [Ckpt] ★ New best Mist Recall = {best_mist_recall:.4f}", flush=True)

                    print(
                        f"  [Best so far] Score={best_score:.4f} | "
                        f"FogR={best_fog_recall:.4f} | MistR={best_mist_recall:.4f}",
                        flush=True
                    )

    if rank == 0:
        print(f"\n[{tag}] Training complete.", flush=True)
        print(
            f"  Final Best -> Score={best_score:.4f} | "
            f"FogR={best_fog_recall:.4f} | MistR={best_mist_recall:.4f}",
            flush=True
        )

# ==========================================
# Main
# ==========================================
def main():
    l_rank, g_rank, w_size = init_distributed()
    device = torch.device(f"cuda:{l_rank}")

    if g_rank == 0:
        os.makedirs(CONFIG['SAVE_CKPT_DIR'], exist_ok=True)
        print(f"Start Exp: {CONFIG['EXPERIMENT_ID']}", flush=True)
        print(f"World size: {w_size}", flush=True)

    # ==========================================
    # 模型构建（Stage 1 和 Stage 2 共用同一结构）
    # ==========================================
    def build_model():
        m = ImprovedDualStreamPMSTNet(
            window_size=CONFIG['WINDOW_SIZE'],
            hidden_dim=CONFIG['MODEL_HIDDEN_DIM'],
            num_classes=3,
            extra_feat_dim=CONFIG['FE_EXTRA_DIMS']
        ).to(device)
        if w_size > 1:
            m = DDP(m, device_ids=[l_rank], find_unused_parameters=False)
        return m

    def build_loss():
        return DualBranchLoss(
            binary_pos_weight=CONFIG['BINARY_POS_WEIGHT'],
            fine_class_weight=[
                CONFIG['FINE_CLASS_WEIGHT_FOG'],
                CONFIG['FINE_CLASS_WEIGHT_MIST'],
                CONFIG['FINE_CLASS_WEIGHT_CLEAR']
            ],
            asym_gamma_neg=CONFIG['ASYM_GAMMA_NEG'],
            asym_clip=CONFIG['ASYM_CLIP']
        ).to(device)

    # Stage 2 输出头的参数名（用于差异化学习率）
    HEAD_PARAM_NAMES = {'fine_classifier', 'low_vis_detector', 'reg_head'}

    def build_s2_optimizer(model):
        """
        Stage 2 差异化学习率：
          - 分类头（fine_classifier / low_vis_detector / reg_head）用较大 LR
          - backbone 其余部分用较小 LR，做微调
        """
        target = model.module if w_size > 1 else model
        head_params     = []
        backbone_params = []
        for name, param in target.named_parameters():
            module_name = name.split('.')[0]
            if module_name in HEAD_PARAM_NAMES:
                head_params.append(param)
            else:
                backbone_params.append(param)

        if g_rank == 0:
            print(f"[S2-Optim] backbone params: {len(backbone_params)}, "
                  f"head params: {len(head_params)}", flush=True)
            print(f"[S2-Optim] LR backbone={CONFIG['S2_LR_BACKBONE']}, "
                  f"LR head={CONFIG['S2_LR_HEAD']}", flush=True)

        return optim.AdamW(
            [
                {'params': backbone_params, 'lr': CONFIG['S2_LR_BACKBONE']},
                {'params': head_params,     'lr': CONFIG['S2_LR_HEAD']},
            ],
            weight_decay=CONFIG['S2_WEIGHT_DECAY']
        )

    # ==========================================
    # Stage 1
    # ==========================================
    exp_id = CONFIG['EXPERIMENT_ID']
    s1_best_score_path = os.path.join(CONFIG['SAVE_CKPT_DIR'], f"{exp_id}_S1_best_score.pt")

    if not CONFIG['SKIP_STAGE1']:
        if g_rank == 0:
            print("\n" + "="*50, flush=True)
            print("[Stage 1] Loading data...", flush=True)

        tr_ds_s1, val_ds_s1, scaler_s1 = load_data(
            CONFIG['S1_DATA_DIR'], None,
            g_rank, l_rank, device, False,
            CONFIG['WINDOW_SIZE'], w_size, exp_id
        )

        if g_rank == 0:
            print(f"[Stage 1] Train={len(tr_ds_s1)}, Val={len(val_ds_s1)}", flush=True)

        model   = build_model()
        loss_fn = build_loss()

        if g_rank == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[Model] Params={n_params/1e6:.2f}M", flush=True)

        # 如果指定了预训练权重，直接加载跳过 S1 训练
        if CONFIG['S1_PRETRAINED_PATH'] is not None:
            load_checkpoint(model, CONFIG['S1_PRETRAINED_PATH'], g_rank, w_size, device)
            if g_rank == 0:
                print("[Stage 1] Skipped training, loaded pretrained weights.", flush=True)
        else:
            opt_s1 = optim.AdamW(
                model.parameters(),
                lr=CONFIG['S1_LR_BACKBONE'],
                weight_decay=CONFIG['S1_WEIGHT_DECAY']
            )
            train_stage(
                tag='S1', model=model, tr_ds=tr_ds_s1, val_ds=val_ds_s1,
                optimizer=opt_s1, loss_fn=loss_fn, device=device,
                rank=g_rank, world_size=w_size,
                total_steps=CONFIG['S1_TOTAL_STEPS'],
                val_int=CONFIG['S1_VAL_INTERVAL'],
                batch_size=CONFIG['S1_BATCH_SIZE'],
                grad_accum=CONFIG['S1_GRAD_ACCUM'],
                fog_ratio=CONFIG['S1_FOG_RATIO'],
                mist_ratio=CONFIG['S1_MIST_RATIO'],
                exp_id=exp_id
            )

        # 释放 S1 数据集内存
        del tr_ds_s1, val_ds_s1
        if w_size > 1:
            dist.barrier()
    else:
        if g_rank == 0:
            print("[Stage 1] SKIPPED.", flush=True)
        model   = build_model()
        loss_fn = build_loss()
        scaler_s1 = None

    # ==========================================
    # Stage 2
    # ==========================================
    if g_rank == 0:
        print("\n" + "="*50, flush=True)
        print("[Stage 2] Starting fine-tune on FE dataset...", flush=True)

    # 加载 Stage 1 最佳权重作为 Stage 2 起点
    if os.path.exists(s1_best_score_path):
        load_checkpoint(model, s1_best_score_path, g_rank, w_size, device)
        if g_rank == 0:
            print(f"[Stage 2] Loaded S1 best_score checkpoint: "
                  f"{os.path.basename(s1_best_score_path)}", flush=True)
    else:
        if g_rank == 0:
            print(f"[Stage 2] WARNING: S1 best_score checkpoint not found at "
                  f"{s1_best_score_path}. Continuing with current model weights.", flush=True)

    if g_rank == 0:
        print("[Stage 2] Loading FE data...", flush=True)

    tr_ds_s2, val_ds_s2, scaler_s2 = load_data(
        CONFIG['S2_DATA_DIR'], None,
        g_rank, l_rank, device, False,
        CONFIG['WINDOW_SIZE'], w_size, exp_id
    )

    if g_rank == 0:
        print(f"[Stage 2] Train={len(tr_ds_s2)}, Val={len(val_ds_s2)}", flush=True)

    opt_s2     = build_s2_optimizer(model)
    loss_fn_s2 = build_loss()

    train_stage(
        tag='S2', model=model, tr_ds=tr_ds_s2, val_ds=val_ds_s2,
        optimizer=opt_s2, loss_fn=loss_fn_s2, device=device,
        rank=g_rank, world_size=w_size,
        total_steps=CONFIG['S2_TOTAL_STEPS'],
        val_int=CONFIG['S2_VAL_INTERVAL'],
        batch_size=CONFIG['S2_BATCH_SIZE'],
        grad_accum=CONFIG['S2_GRAD_ACCUM'],
        fog_ratio=CONFIG['S2_FOG_RATIO'],
        mist_ratio=CONFIG['S2_MIST_RATIO'],
        exp_id=exp_id
    )

    # ==========================================
    # 收尾
    # ==========================================
    cleanup_temp_files(exp_id)
    if w_size > 1:
        dist.destroy_process_group()

    if g_rank == 0:
        print("\nAll stages complete.", flush=True)


if __name__ == "__main__":
    main()
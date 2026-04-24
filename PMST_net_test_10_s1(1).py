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

CONFIG = {
    # ========== 实验控制 ==========
    'EXPERIMENT_ID':          os.environ.get('EXPERIMENT_JOB_ID', f'exp_{int(time.time())}'),
    'S1_PRETRAINED_PATH':     None,
    'BASE_PATH':              BASE_PATH,
    'WINDOW_SIZE':            TARGET_WINDOW_SIZE,
    'S1_DATA_DIR':            S1_DIR,
    'NUM_WORKERS':            2,
    'SAVE_CKPT_DIR':          os.path.join(BASE_PATH, 'checkpoints'),

    # ========== Stage 1 训练配置 ==========
    # S1_TOTAL_STEPS: 15000，配合 Early Stopping 节省算力
    # S1_VAL_INTERVAL: 500，更密集的验证以捕捉最优点
    'S1_TOTAL_STEPS':         15000,
    'S1_VAL_INTERVAL':        500,
    'S1_BATCH_SIZE':          512,
    'S1_GRAD_ACCUM':          2,
    'S1_FOG_RATIO':           0.20,
    'S1_MIST_RATIO':          0.20,
    'S1_LR_BACKBONE':         2e-4,
    'S1_WEIGHT_DECAY':        1e-2,

    # ========== 评估约束条件 ==========
    'MIN_FOG_PRECISION':      0.15,
    'MIN_FOG_RECALL':         0.50,
    'MIN_MIST_PRECISION':     0.10,
    'MIN_MIST_RECALL':        0.15,
    'MIN_CLEAR_ACC':          0.90,

    # ========== 损失函数配置 ==========
    # NOTE: build_loss() now explicitly passes all alpha values to DualBranchLoss,
    # fixing the silent bug in the original Script A where LOSS_ALPHA_FP and
    # LOSS_ALPHA_FOG_BOOST were defined in CONFIG but never forwarded to the loss
    # constructor (so DualBranchLoss silently used its internal defaults instead).
    'LOSS_TYPE':              'asymmetric',
    'LOSS_ALPHA_BINARY':      1.0,
    'LOSS_ALPHA_FINE':        1.0,
    'LOSS_ALPHA_FP':          1.0,
    'LOSS_ALPHA_FOG_BOOST':   0.5,
    'LOSS_ALPHA_MIST_BOOST':  0.2,
    # alpha_clear_margin is intentionally 0.0 for Stage 1 (ERA5 pre-training).
    # The clear-margin calibration term is a Stage 2 technique; enabling it here
    # would interfere with backbone convergence on the reanalysis distribution.
    'LOSS_ALPHA_CLEAR_MARGIN': 0.0,

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

    # ========== Early Stopping ==========
    # S1 允许连续多少个 val_interval 不改善则停止
    'S1_ES_PATIENCE':         8,
}

# ==========================================
# 1. 基础工具与分布式设置
# ==========================================

def _enforce_nccl_shm_disable():
    """
    Force NCCL_SHM_DISABLE=1 before any RCCL/NCCL communicator is created.
    This prevents the SHM mutex crash observed on AMD DCU clusters.
    Must be called at the very start of init_distributed().
    """
    if os.environ.get('NCCL_SHM_DISABLE') != '1':
        os.environ['NCCL_SHM_DISABLE'] = '1'
        rank_str = os.environ.get('RANK', os.environ.get('SLURM_PROCID', None))
        if rank_str is None or str(rank_str) == '0':
            print("[Init] NCCL_SHM_DISABLE was not set; forcing to 1 "
                  "(prevents RCCL SHM mutex crash on AMD DCU cluster).", flush=True)


def safe_barrier(world_size: int, device: torch.device = None):
    """
    Drop-in replacement for dist.barrier().
    Uses a dummy all_reduce on the specific GPU to force communicator
    initialization on the correct device, avoiding AMD DCU barrier hangs.
    torch.cuda.synchronize() is called after all_reduce to guarantee
    the GPU has fully committed before any rank proceeds.
    """
    if world_size > 1:
        if device is not None:
            dummy = torch.zeros(1, device=device)
            dist.all_reduce(dummy, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)
        else:
            dist.barrier()


def get_available_space(path):
    try:
        stat = os.statvfs(path)
        return stat.f_bavail * stat.f_frsize
    except Exception:
        return 0


def copy_to_local(src_path: str, global_rank: int, local_rank: int,
                  world_size: int, exp_id: str = None) -> str:
    """
    Copy NFS data file to local /tmp to accelerate I/O.
    Each node's local_rank==0 performs the copy; safe_barrier synchronizes
    all ranks before and after so no rank proceeds with a stale file.
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

    device = torch.device(f"cuda:{local_rank}")

    # Synchronize all ranks before any I/O begins
    safe_barrier(world_size, device)

    if local_rank == 0:
        try:
            if not os.path.exists(src_path):
                if global_rank == 0:
                    print(f"[Data-Copy] Warning: Source {src_path} not found.", flush=True)
            else:
                src_size    = os.path.getsize(src_path)
                cache_valid = (os.path.exists(local_path) and
                               os.path.getsize(local_path) == src_size)
                if cache_valid:
                    if global_rank == 0:
                        print(f"[Data-Copy] Cache hit: {local_path}", flush=True)
                else:
                    avail = get_available_space(target_dir)
                    if avail < src_size + 500 * 1024 * 1024:
                        if global_rank == 0:
                            print(f"[Data-Copy] Insufficient space on {target_dir}, using NFS.",
                                  flush=True)
                    else:
                        if global_rank == 0:
                            print(f"[Data-Copy] Copying {basename} to {target_dir}...", flush=True)
                        tmp_path = local_path + ".tmp"
                        shutil.copyfile(src_path, tmp_path)
                        os.rename(tmp_path, local_path)
                        if global_rank == 0:
                            print(f"[Data-Copy] Done: {local_path}", flush=True)
        except Exception as e:
            if global_rank == 0:
                print(f"[Data-Copy] Error: {e}, falling back to NFS.", flush=True)

    # Synchronize all ranks after copy is complete
    safe_barrier(world_size, device)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    else:
        return src_path


def init_distributed():
    """
    Initialize the distributed process group using an explicit TCPStore.
    Reads DIST_STORE_PORT (not MASTER_PORT) to avoid the port conflict
    caused by torchrun already binding MASTER_PORT for its own rendezvous.
    """
    _enforce_nccl_shm_disable()

    local_rank  = int(os.environ.get("LOCAL_RANK",  os.environ.get("SLURM_LOCALID",  0)))
    global_rank = int(os.environ.get("RANK",         os.environ.get("SLURM_PROCID",   0)))
    world_size  = int(os.environ.get("WORLD_SIZE",   os.environ.get("SLURM_NTASKS",   1)))

    torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

        torchrun_port   = int(os.environ.get("MASTER_PORT", 29500))
        dist_store_port = int(os.environ.get("DIST_STORE_PORT", torchrun_port + 1))

        if global_rank == 0:
            print(
                f"[Dist] Initializing TCPStore for dist.init_process_group\n"
                f"       master_addr    = {master_addr}\n"
                f"       RDZV_PORT      = {torchrun_port}   (owned by torchrun, NOT used here)\n"
                f"       DIST_STORE_PORT= {dist_store_port} (used for dist.TCPStore)\n"
                f"       world_size     = {world_size}",
                flush=True
            )

        store = dist.TCPStore(
            host_name=master_addr,
            port=dist_store_port,
            world_size=world_size,
            is_master=(global_rank == 0),
            timeout=datetime.timedelta(minutes=30),
            wait_for_workers=True,
        )

        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=global_rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=30),
        )

        if global_rank == 0:
            print("[Dist] Process group initialized successfully.", flush=True)
            print(
                f"[Dist] world_size={world_size}, "
                f"global_rank={global_rank}, local_rank={local_rank}",
                flush=True
            )
            print(
                f"[Dist] NCCL_SHM_DISABLE={os.environ.get('NCCL_SHM_DISABLE', 'NOT SET')}",
                flush=True
            )

    return local_rank, global_rank, world_size


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
    Stratified balanced batch sampler.
    Stores and yields indices into the SUBSET (positions within PMSTDataset),
    not into the original full array.
    - Fixed epoch_length=2000 ensures all ranks produce identical batch counts (DDP requirement).
    - Seed based on seed + rank + epoch guarantees determinism and per-process uniqueness.
    - Each rank only samples from its own shard to prevent data overlap.
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

        y = dataset.y_cls.numpy()

        self.n_fog   = max(1, int(batch_size * fog_ratio))
        self.n_mist  = max(1, int(batch_size * mist_ratio))
        self.n_clear = batch_size - self.n_fog - self.n_mist

        all_positions = np.arange(len(y))
        self.pos_indices = {
            0: all_positions[y == 0],
            1: all_positions[y == 1],
            2: all_positions[y == 2],
        }

        for k in self.pos_indices:
            if len(self.pos_indices[k]) == 0:
                self.pos_indices[k] = all_positions[:1]
                continue
            splits = np.array_split(self.pos_indices[k], world_size)
            shard  = splits[rank % len(splits)]
            self.pos_indices[k] = shard if len(shard) > 0 else self.pos_indices[k]

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(seed=self.seed + self.rank + self._epoch * 997)
        for _ in range(self.epoch_length):
            f = rng.choice(self.pos_indices[0], size=self.n_fog,   replace=True)
            m = rng.choice(self.pos_indices[1], size=self.n_mist,  replace=True)
            c = rng.choice(self.pos_indices[2], size=self.n_clear, replace=True)
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
    """
    Dual-branch loss with an optional clear-margin calibration term.

    New term `l_clear_margin` (from Script B):
        For every clear sample (target==2) where p(fog)+p(mist) exceeds
        `clear_margin`, apply a squared-hinge penalty proportional to the excess.
        This directly drives the model to output low low-visibility probability
        for clear samples, fixing the root cause of precision collapse in Stage 2.

    Backward-compatible: `alpha_clear_margin` defaults to 0.0, so Stage 1
    training is completely unaffected by this term.

    Bug-fix vs original Script A: all alpha coefficients are now explicit
    constructor parameters. In Script A, LOSS_ALPHA_FP and LOSS_ALPHA_FOG_BOOST
    were defined in CONFIG but never passed to DualBranchLoss, so the loss
    silently used its hardcoded internal defaults (0.5 and 0.2 respectively).
    All alpha values are now forwarded from CONFIG via build_loss().
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = kwargs
        self.register_buffer(
            'pos_weight',
            torch.tensor([kwargs.get('binary_pos_weight', 1.0)])
        )
        weights = kwargs.get('fine_class_weight', [1.0, 1.0, 1.0])
        self.register_buffer(
            'fine_class_weight',
            torch.tensor(weights, dtype=torch.float32)
        )

        self.fine_loss = AsymmetricLoss(
            gamma_neg=kwargs.get('asym_gamma_neg', 2),
            gamma_pos=kwargs.get('asym_gamma_pos', 0),
            clip=kwargs.get('asym_clip', 0.1),
            class_weights=self.fine_class_weight
        )

    def forward(self, fine_logits, low_vis_logit, targets):
        # ── Binary branch ────────────────────────────────────────────────────
        low_vis_logit = torch.clamp(low_vis_logit, -20, 20)
        l_bin = F.binary_cross_entropy_with_logits(
            low_vis_logit,
            (targets <= 1).float().unsqueeze(1),
            pos_weight=self.pos_weight
        )

        # ── Fine-grained classification ───────────────────────────────────────
        l_fine = self.fine_loss(fine_logits, targets)

        probs = F.softmax(fine_logits, dim=1)

        # ── Fog-recall boost: penalise missing actual fog events ──────────────
        is_fog = (targets == 0).float()
        l_fb   = torch.mean((1.0 - probs[:, 0]) ** 2 * is_fog)

        # ── False-positive penalty: quadratic penalty on low-vis prob
        #    for clear samples (global, no threshold) ─────────────────────────
        is_clear     = (targets == 2).float()
        low_vis_prob = torch.clamp(probs[:, 0] + probs[:, 1], 0.0, 1.0)
        l_fp         = torch.mean(low_vis_prob ** 2 * is_clear)

        # ── Mist-recall boost: penalise missing actual mist events ────────────
        is_mist = (targets == 1).float()
        l_mb    = torch.mean((1.0 - probs[:, 1]) ** 2 * is_mist)

        # ── Clear-margin calibration (disabled in S1 via alpha=0.0) ──────────
        # Squared-hinge loss that fires only when a clear sample's
        # p(fog)+p(mist) exceeds `clear_margin`. Activated for S2 fine-tuning;
        # kept here so the same class can be reused in both stages.
        clear_margin   = self.cfg.get('clear_margin', 0.30)
        margin_excess  = torch.relu(low_vis_prob - clear_margin)
        l_clear_margin = torch.mean(margin_excess ** 2 * is_clear)

        # ── Combine ───────────────────────────────────────────────────────────
        total = (
            self.cfg.get('alpha_binary',       1.0) * l_bin          +
            self.cfg.get('alpha_fine',         1.0) * l_fine         +
            self.cfg.get('alpha_fp',           0.5) * l_fp           +
            self.cfg.get('alpha_fog_boost',    0.2) * l_fb           +
            self.cfg.get('alpha_mist_boost',   0.0) * l_mb           +
            self.cfg.get('alpha_clear_margin', 0.0) * l_clear_margin
        )

        return total, {
            'bin':  l_bin.item(),
            'fine': l_fine.item(),
            'cm':   l_clear_margin.item(),   # logged for diagnostics (always 0 in S1)
        }

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
    """
    __getitem__ receives a SUBSET-LOCAL position idx (0..len(self)-1).
    It maps that to the original-array row via self.orig_indices[idx].
    The mmap handle (self.X) is intentionally left as None at construction
    time and opened lazily per worker, avoiding the multi-process fd conflict
    that occurs when a single mmap is inherited across forked workers.
    """
    def __init__(self, X_path, y_cls, y_raw, scaler=None,
                 window_size=12, use_fe=True, indices=None):
        self.X_path       = X_path
        self.orig_indices = np.asarray(indices) if indices is not None else np.arange(len(y_cls))

        self.y_cls = torch.as_tensor(y_cls[self.orig_indices], dtype=torch.long)

        clean_raw  = np.maximum(y_raw[self.orig_indices], 0.0)
        self.y_reg = torch.as_tensor(np.log1p(clean_raw), dtype=torch.float32)
        self.y_raw = torch.as_tensor(clean_raw,           dtype=torch.float32)

        self.split_dyn = window_size * 25
        self.scaler    = scaler
        self.use_fe    = use_fe

        self.log_mask = np.zeros(self.split_dyn, dtype=bool)
        for t in range(window_size):
            for i in [2, 4, 9]:
                self.log_mask[t * 25 + i] = True

        self.X = None   # opened lazily in __getitem__

    def __len__(self):
        return len(self.orig_indices)

    def __getitem__(self, idx):
        if self.X is None:
            self.X = np.load(self.X_path, mmap_mode='r')

        real_idx = self.orig_indices[idx]
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
        safe_barrier(world_size, device)

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

        safe_barrier(world_size, device)

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
    """
    Evaluator with a three-tier threshold search strategy (from Script B).

    Tier 1 — Strict:
        Search the *extended* grid (0.10 → 0.95) for thresholds satisfying
        Fog_P≥0.10, Mist_P≥0.10, and Clear_R≥0.90.
        Rank survivors by target_achievement.

    Tier 2 — Relaxed:
        If Tier 1 yields nothing, relax to Fog_P≥0.05, Mist_P≥0.05, Clear_R≥0.88.
        The score is penalised proportionally to the precision shortfall so the
        checkpoint system still gravitates toward higher-precision solutions.

    Tier 3 — Argmax fallback:
        Used only when both Tier 1 and Tier 2 fail. Identical to original Script A behaviour.

    Why extending the search range matters:
        The original ceiling of 0.65 is too low for a miscalibrated model that
        outputs p(fog+mist)>0.65 for many clear samples.  Valid thresholds often
        exist in the 0.70–0.90 range as calibration improves during training.
    """

    def __init__(self, config):
        self.cfg = config
        self.best_th = {'fog': 0.5, 'mist': 0.5}

        # Tier 1 hard constraints
        self.min_prec_threshold = 0.10
        self.min_clear_recall   = 0.90

        # Tier 2 relaxed constraints
        self.relaxed_prec_threshold = 0.05
        self.relaxed_clear_recall   = 0.88

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

        pred_low  = (preds <= 1)
        true_low  = (targets <= 1)
        is_clear  = (targets == 2)
        lv_tp     = (pred_low & true_low).sum()
        lv_fp     = (pred_low & ~true_low).sum()
        low_vis_precision = lv_tp / (lv_tp + lv_fp + 1e-6)
        fpr = (pred_low & is_clear).sum() / (is_clear.sum() + 1e-6)

        return {
            'Fog_R':   r0, 'Fog_P':   p0,
            'Mist_R':  r1, 'Mist_P':  p1,
            'Clear_R': r2, 'Clear_P': p2,
            'recall_500':          r0,
            'recall_1000':         r1,
            'accuracy':            accuracy,
            'low_vis_precision':   float(low_vis_precision),
            'false_positive_rate': float(fpr),
            'preds':               preds,
        }

    @staticmethod
    def _build_search_grid():
        """
        Extended threshold grid:
          • 0.10 – 0.48 in steps of 0.04  (coarse low range)
          • 0.50 – 0.94 in steps of 0.03  (fine high range — critical new region)
        Total: ~10 + ~15 = ~25 values per axis → ~625 combinations.
        Still fast (<0.1 s) on large val sets via numpy vectorisation.
        """
        low_part  = np.arange(0.10, 0.50, 0.04)
        high_part = np.arange(0.50, 0.96, 0.03)
        return np.unique(np.concatenate([low_part, high_part]))

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

        # ── DDP all-gather ────────────────────────────────────────────────────
        if world_size > 1:
            local_size = torch.tensor([local_probs.size(0)], dtype=torch.long, device=device)
            max_size   = local_size.clone()
            dist.all_reduce(max_size, op=dist.ReduceOp.MAX)

            if local_size < max_size:
                pad_size    = max_size.item() - local_size.item()
                pad_probs   = torch.zeros(
                    (pad_size, local_probs.size(1)),
                    dtype=local_probs.dtype, device=device
                )
                pad_targets = torch.full(
                    (pad_size,), -1,
                    dtype=local_targets.dtype, device=device
                )
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
        tier_used  = 0

        if rank == 0:
            n       = actual_val_size if actual_val_size is not None else len(loader.dataset)
            probs   = all_probs[:n]
            targets = all_targets[:n]

            valid_mask = targets >= 0
            probs   = probs[valid_mask]
            targets = targets[valid_mask]

            search_space = self._build_search_grid()
            n_combos     = len(search_space) ** 2
            print(
                f"  [Eval] Searching {n_combos} threshold combinations "
                f"(grid: {search_space[0]:.2f}–{search_space[-1]:.2f})...",
                flush=True
            )

            # ── Tier 1: strict constraints ────────────────────────────────────
            for f_th in search_space:
                for m_th in search_space:
                    stats = self._build_full_metrics(probs, targets, f_th, m_th)
                    if (stats['Fog_P']   >= self.min_prec_threshold  and
                        stats['Mist_P']  >= self.min_prec_threshold  and
                        stats['Clear_R'] >= self.min_clear_recall):
                        ta = compute_target_achievement(stats, self.cfg)
                        if ta > best_ta:
                            best_ta    = ta
                            best_stats = stats
                            self.best_th = {'fog': float(f_th), 'mist': float(m_th)}
                            tier_used    = 1

            # ── Tier 2: relaxed constraints ───────────────────────────────────
            if best_stats is None:
                for f_th in search_space:
                    for m_th in search_space:
                        stats = self._build_full_metrics(probs, targets, f_th, m_th)
                        if (stats['Fog_P']   >= self.relaxed_prec_threshold  and
                            stats['Mist_P']  >= self.relaxed_prec_threshold  and
                            stats['Clear_R'] >= self.relaxed_clear_recall):
                            fog_shortfall  = max(0.0, self.min_prec_threshold - stats['Fog_P'])
                            mist_shortfall = max(0.0, self.min_prec_threshold - stats['Mist_P'])
                            prec_penalty   = (fog_shortfall + mist_shortfall) * 1.0
                            ta = compute_target_achievement(stats, self.cfg) - prec_penalty
                            if ta > best_ta:
                                best_ta    = ta
                                best_stats = stats
                                self.best_th = {'fog': float(f_th), 'mist': float(m_th)}
                                tier_used    = 2

            # ── Tier 3: argmax fallback ───────────────────────────────────────
            if best_stats is None:
                tier_used = 3
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
                    'Fog_R':   r0, 'Fog_P':   p0,
                    'Mist_R':  r1, 'Mist_P':  p1,
                    'Clear_R': r2, 'Clear_P': p2,
                    'recall_500':          r0,
                    'recall_1000':         r1,
                    'accuracy':            accuracy,
                    'low_vis_precision':   float(lv_prec),
                    'false_positive_rate': float(fpr),
                    'preds':               preds,
                }
                best_ta = compute_target_achievement(best_stats, self.cfg)
                print(
                    "  [Eval] WARN: Tier 1+2 all failed — using argmax fallback (Tier 3).",
                    flush=True
                )
            else:
                tier_label = "Strict" if tier_used == 1 else "Relaxed"
                print(
                    f"  [Eval] Tier {tier_used} ({tier_label}): "
                    f"Best Th -> Fog:{self.best_th['fog']:.2f}, "
                    f"Mist:{self.best_th['mist']:.2f}",
                    flush=True
                )

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

        # ── Broadcast best_ta to all ranks ────────────────────────────────────
        if world_size > 1:
            ta_tensor = torch.tensor([best_ta], dtype=torch.float32, device=device)
            dist.broadcast(ta_tensor, src=0)
            best_ta = ta_tensor.item()
            safe_barrier(world_size, device)

        return {
            'score':      best_ta,
            'stats':      best_stats,
            'thresholds': self.best_th
        }

# ==========================================
# 7. 训练工具函数
# ==========================================

def save_checkpoint(model, path, rank, world_size):
    if rank != 0:
        return
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state, path)
    print(f"  [Ckpt] Saved -> {os.path.basename(path)}", flush=True)


def load_checkpoint(model, path, rank, world_size, device):
    """
    Check path existence on ALL ranks before the barrier, so any rank
    can detect a missing file without leaving others stuck at the barrier.
    Broadcasts existence flag from rank 0 for consistency.
    """
    path_exists = os.path.exists(path)

    if world_size > 1:
        exists_tensor = torch.tensor([int(path_exists)], dtype=torch.long, device=device)
        dist.broadcast(exists_tensor, src=0)
        path_exists = bool(exists_tensor.item())

    if not path_exists:
        raise FileNotFoundError(
            f"[Ckpt] Checkpoint not found on all ranks: {path}"
        )

    if rank == 0:
        print(f"[Ckpt] Loading checkpoint: {os.path.basename(path)}", flush=True)

    safe_barrier(world_size, device)

    state_dict = torch.load(path, map_location=device)

    target = model.module if hasattr(model, 'module') else model
    missing, unexpected = target.load_state_dict(state_dict, strict=False)

    if rank == 0:
        if missing:
            print(f"  [Ckpt] Missing keys ({len(missing)}): "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
        if unexpected:
            print(f"  [Ckpt] Unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}", flush=True)
        print(f"  [Ckpt] Load complete.", flush=True)

    safe_barrier(world_size, device)


def rewrap_ddp(model, world_size):
    """Extract the raw model from a DDP wrapper."""
    if world_size > 1 and hasattr(model, 'module'):
        return model.module
    return model


def wrap_ddp(raw_model, local_rank, world_size, find_unused=False):
    """
    Wrap raw model in DDP.
    safe_barrier is called first to ensure all ranks have consistent
    parameter state before the DDP reducer is constructed.
    """
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        safe_barrier(world_size, device)
        return DDP(raw_model, device_ids=[local_rank],
                   find_unused_parameters=find_unused)
    return raw_model


def train_stage(tag, model, tr_ds, val_ds, optimizer, loss_fn, device,
                rank, world_size, total_steps, val_int, batch_size,
                grad_accum, fog_ratio, mist_ratio, exp_id,
                patience=10):
    """
    Train one stage with Early Stopping.
    All DDP ranks synchronize no_improve_count via broadcast for symmetric exit.
    worker_init_fn resets the mmap handle in each DataLoader worker so every
    worker opens its own file descriptor (avoids the inherited-fd bug).
    """
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_info.dataset.X = None

    sampler = StratifiedBalancedBatchSampler(
        tr_ds, batch_size,
        fog_ratio=fog_ratio, mist_ratio=mist_ratio,
        rank=rank, world_size=world_size,
        epoch_length=2000,
    )
    loader = DataLoader(
        tr_ds,
        batch_sampler=sampler,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )

    val_sampler = (DistributedSampler(val_ds, num_replicas=world_size,
                                       rank=rank, shuffle=False)
                   if world_size > 1 else None)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, sampler=val_sampler, pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )
    actual_val_size = len(val_ds)

    metrics_evaluator = ComprehensiveMetrics(CONFIG)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_score       = -1.0
    best_fog_recall  = -1.0
    best_mist_recall = -1.0
    no_improve_count = 0

    ckpt_dir = CONFIG['SAVE_CKPT_DIR']
    path_best_score       = os.path.join(ckpt_dir, f"{exp_id}_{tag}_best_score.pt")
    path_best_fog_recall  = os.path.join(ckpt_dir, f"{exp_id}_{tag}_best_fog_recall.pt")
    path_best_mist_recall = os.path.join(ckpt_dir, f"{exp_id}_{tag}_best_mist_recall.pt")
    path_latest           = os.path.join(ckpt_dir, f"{exp_id}_{tag}_latest.pt")

    if rank == 0:
        print(f"\n[{tag}] Training started. total_steps={total_steps}, "
              f"grad_accum={grad_accum}, batch_size={batch_size}, "
              f"patience={patience}", flush=True)

    step         = 0
    batch_count  = 0
    pseudo_epoch = 0
    iterator     = iter(loader)
    model.train()

    while step < total_steps:
        try:
            bx, by, blog, braw = next(iterator)
        except StopIteration:
            pseudo_epoch += 1
            sampler.set_epoch(pseudo_epoch)
            iterator = iter(loader)
            bx, by, blog, braw = next(iterator)

        bx, by, blog = bx.to(device), by.to(device), blog.to(device)
        batch_count += 1

        fine, reg, bin_out = model(bx)
        l_dual, loss_dict  = loss_fn(fine, bin_out, by)
        l_reg = F.mse_loss(reg.view(-1), blog)
        loss  = l_dual + CONFIG['REG_LOSS_ALPHA'] * l_reg
        loss  = loss / grad_accum

        is_last_accum_step = (batch_count % grad_accum == 0)

        if world_size > 1 and not is_last_accum_step:
            ctx = model.no_sync()
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            loss.backward()

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

            if rank == 0 and step % 50 == 0:
                print(
                    f"\r[{tag}] Step {step:>6}/{total_steps} | "
                    f"Loss={loss.item() * grad_accum:.4f} | "
                    f"bin={loss_dict['bin']:.4f} "
                    f"fine={loss_dict['fine']:.4f} "
                    f"cm={loss_dict.get('cm', 0.0):.4f} | "
                    f"LR={scheduler.get_last_lr()[0]:.2e} | "
                    f"NoImprove={no_improve_count}/{patience}",
                    end="", flush=True
                )

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
                        no_improve_count = 0
                        save_checkpoint(model, path_best_score, rank, world_size)
                        print(f"  [Ckpt] ★ New best target_achievement = {best_score:.4f}",
                              flush=True)
                    else:
                        no_improve_count += 1

                    if fog_r > best_fog_recall:
                        best_fog_recall = fog_r
                        save_checkpoint(model, path_best_fog_recall, rank, world_size)
                        print(f"  [Ckpt] ★ New best Fog Recall = {best_fog_recall:.4f}",
                              flush=True)

                    if mist_r > best_mist_recall:
                        best_mist_recall = mist_r
                        save_checkpoint(model, path_best_mist_recall, rank, world_size)
                        print(f"  [Ckpt] ★ New best Mist Recall = {best_mist_recall:.4f}",
                              flush=True)

                    print(
                        f"  [Best so far] Score={best_score:.4f} | "
                        f"FogR={best_fog_recall:.4f} | MistR={best_mist_recall:.4f} | "
                        f"NoImprove={no_improve_count}/{patience}",
                        flush=True
                    )

                # Broadcast no_improve_count to all ranks for symmetric early stop
                if world_size > 1:
                    stop_tensor = torch.tensor([no_improve_count], dtype=torch.long, device=device)
                    dist.broadcast(stop_tensor, src=0)
                    no_improve_count = stop_tensor.item()

                if patience > 0 and no_improve_count >= patience:
                    if rank == 0:
                        print(
                            f"\n[{tag}] *** Early Stopping triggered at step {step}. "
                            f"Best score={best_score:.4f} ***",
                            flush=True
                        )
                    # safe_barrier before break ensures all ranks exit symmetrically
                    safe_barrier(world_size, device)
                    break

    if rank == 0:
        print(f"\n[{tag}] Training complete.", flush=True)
        print(
            f"  Final Best -> Score={best_score:.4f} | "
            f"FogR={best_fog_recall:.4f} | MistR={best_mist_recall:.4f}",
            flush=True
        )

# ==========================================
# Main — Stage 1 only
# ==========================================
def main():
    l_rank, g_rank, w_size = init_distributed()
    device = torch.device(f"cuda:{l_rank}")

    if g_rank == 0:
        os.makedirs(CONFIG['SAVE_CKPT_DIR'], exist_ok=True)
        print(f"[Stage1-Only] Start Exp: {CONFIG['EXPERIMENT_ID']}", flush=True)
        print(f"World size: {w_size}", flush=True)

    # ==========================================
    # [核心修复] 全局无负载同步。
    # 在进行任何磁盘 I/O 之前强制所有 DCU/GPU 将 NCCL/RCCL 彻底初始化完成，
    # 消除 rank 0 拷数据时非 0 rank 瞬间进 barrier 导致的硬件时序挂死。
    # ==========================================
    safe_barrier(w_size, device)

    if g_rank == 0:
        print("[Main] Initial barrier passed — all ranks synchronized.", flush=True)

    exp_id = CONFIG['EXPERIMENT_ID']

    # ==========================================
    # 构建函数
    # ==========================================
    def build_model():
        m = ImprovedDualStreamPMSTNet(
            window_size=CONFIG['WINDOW_SIZE'],
            hidden_dim=CONFIG['MODEL_HIDDEN_DIM'],
            num_classes=3,
            extra_feat_dim=CONFIG['FE_EXTRA_DIMS']
        ).to(device)
        return wrap_ddp(m, l_rank, w_size, find_unused=False)

    def build_loss():
        """
        Stage 1 loss function.
        All alpha values are explicitly forwarded from CONFIG, fixing the
        silent bug in the original Script A where LOSS_ALPHA_FP and
        LOSS_ALPHA_FOG_BOOST were defined but never passed to DualBranchLoss.
        alpha_clear_margin=0.0 keeps the clear-margin term disabled for S1.
        """
        return DualBranchLoss(
            binary_pos_weight=CONFIG['BINARY_POS_WEIGHT'],
            fine_class_weight=[
                CONFIG['FINE_CLASS_WEIGHT_FOG'],
                CONFIG['FINE_CLASS_WEIGHT_MIST'],
                CONFIG['FINE_CLASS_WEIGHT_CLEAR'],
            ],
            asym_gamma_neg=CONFIG['ASYM_GAMMA_NEG'],
            asym_gamma_pos=CONFIG['ASYM_GAMMA_POS'],
            asym_clip=CONFIG['ASYM_CLIP'],
            alpha_binary=CONFIG['LOSS_ALPHA_BINARY'],
            alpha_fine=CONFIG['LOSS_ALPHA_FINE'],
            alpha_fp=CONFIG['LOSS_ALPHA_FP'],
            alpha_fog_boost=CONFIG['LOSS_ALPHA_FOG_BOOST'],
            alpha_mist_boost=CONFIG['LOSS_ALPHA_MIST_BOOST'],
            alpha_clear_margin=CONFIG['LOSS_ALPHA_CLEAR_MARGIN'],  # 0.0 → disabled
        ).to(device)

    # ==========================================
    # Stage 1 数据加载 (ERA5 reanalysis)
    # ==========================================
    if g_rank == 0:
        print("\n" + "="*60, flush=True)
        print("[Stage 1] Loading ERA5 data...", flush=True)

    tr_ds_s1, val_ds_s1, scaler_s1 = load_data(
        CONFIG['S1_DATA_DIR'], None,
        g_rank, l_rank, device, False,
        CONFIG['WINDOW_SIZE'], w_size, exp_id
    )

    if g_rank == 0:
        print(f"[Stage 1] Train={len(tr_ds_s1)}, Val={len(val_ds_s1)}", flush=True)

    # ==========================================
    # 构建模型 + 损失函数
    # ==========================================
    model   = build_model()
    loss_fn = build_loss()

    if g_rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] Trainable params={n_params/1e6:.2f}M", flush=True)

    # ==========================================
    # 可选：从预训练权重热启动
    # ==========================================
    if CONFIG['S1_PRETRAINED_PATH'] is not None:
        load_checkpoint(model, CONFIG['S1_PRETRAINED_PATH'], g_rank, w_size, device)
        if g_rank == 0:
            print(f"[Stage 1] Loaded pretrained weights from: "
                  f"{CONFIG['S1_PRETRAINED_PATH']}", flush=True)

    # ==========================================
    # Stage 1 训练
    # ==========================================
    if g_rank == 0:
        print("\n" + "-"*60, flush=True)
        print(
            f"[Stage 1] Starting training. "
            f"total_steps={CONFIG['S1_TOTAL_STEPS']}, "
            f"LR={CONFIG['S1_LR_BACKBONE']:.1e}, "
            f"early_stop_patience={CONFIG['S1_ES_PATIENCE']}",
            flush=True
        )

    opt_s1 = optim.AdamW(
        model.parameters(),
        lr=CONFIG['S1_LR_BACKBONE'],
        weight_decay=CONFIG['S1_WEIGHT_DECAY']
    )

    train_stage(
        tag='S1',
        model=model,
        tr_ds=tr_ds_s1,
        val_ds=val_ds_s1,
        optimizer=opt_s1,
        loss_fn=loss_fn,
        device=device,
        rank=g_rank,
        world_size=w_size,
        total_steps=CONFIG['S1_TOTAL_STEPS'],
        val_int=CONFIG['S1_VAL_INTERVAL'],
        batch_size=CONFIG['S1_BATCH_SIZE'],
        grad_accum=CONFIG['S1_GRAD_ACCUM'],
        fog_ratio=CONFIG['S1_FOG_RATIO'],
        mist_ratio=CONFIG['S1_MIST_RATIO'],
        exp_id=exp_id,
        patience=CONFIG['S1_ES_PATIENCE'],
    )

    # ==========================================
    # 收尾
    # ==========================================
    if g_rank == 0:
        print("\n" + "="*60, flush=True)
        print("[Summary] Stage 1 complete.", flush=True)
        print(f"  S1 best checkpoint     : {exp_id}_S1_best_score.pt", flush=True)
        print(f"  S1 best Fog-Recall     : {exp_id}_S1_best_fog_recall.pt", flush=True)
        print(f"  S1 best Mist-Recall    : {exp_id}_S1_best_mist_recall.pt", flush=True)
        print(f"  S1 latest              : {exp_id}_S1_latest.pt", flush=True)
        print(f"  All checkpoints in     : {CONFIG['SAVE_CKPT_DIR']}", flush=True)

    del tr_ds_s1, val_ds_s1
    cleanup_temp_files(exp_id)

    if w_size > 1:
        safe_barrier(w_size, device)
        dist.destroy_process_group()

    if g_rank == 0:
        print("\nJob finished.", flush=True)


if __name__ == "__main__":
    main()
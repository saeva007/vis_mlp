import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import shutil
import hashlib
import math
from collections import Counter
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
import contextlib
import datetime
import time
import joblib
import gc
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (f1_score, recall_score, precision_score,
                              confusion_matrix)

warnings.filterwarnings('ignore')

# ==========================================
# 0. Global Configuration
# ==========================================
BASE_PATH    = "/public/home/putianshu/vis_mlp"
S1_DATA_PATH = "/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_12h"
S2_DATA_PATH = "/public/home/putianshu/vis_mlp/ml_dataset_fe_12h"

CONFIG = {
    # ── Experiment ────────────────────────────────────────────────────────
    'EXPERIMENT_ID':      os.environ.get('EXPERIMENT_JOB_ID',
                                         f'exp_{int(time.time())}'),
    'BASE_PATH':          BASE_PATH,
    'S1_DATA_PATH':       S1_DATA_PATH,
    'S2_DATA_PATH':       S2_DATA_PATH,
    'SAVE_CKPT_DIR':      os.path.join(BASE_PATH, 'checkpoints_simple'),
    'MODEL_SAVE_DIR':     os.path.join(BASE_PATH, 'model'),
    'NUM_WORKERS':        2,

    # ── Model ─────────────────────────────────────────────────────────────
    'INPUT_DIM':          330,
    'HIDDEN_DIM':         1320,
    'NUM_CLASSES':        3,
    'DROPOUT':            0.25,

    # ── Optimiser / gradient ──────────────────────────────────────────────
    'BASE_LR':            1e-4,
    'WEIGHT_DECAY':       2e-5,
    'GRAD_CLIP':          1.0,
    'GRAD_ACCUM':         1,          # no gradient accumulation by default

    # ── Batch / sampling ─────────────────────────────────────────────────
    'BATCH_SIZE':         512,
    'POS_RATIO':          0.25,       # fraction of low-vis samples per batch
    'STEPS_PER_EPOCH':    1000,       # steps that constitute one pseudo-epoch

    # ── Stage 1 – pre-training on ERA5 reanalysis ─────────────────────────
    # Total steps = 300 pseudo-epochs × 1000 steps
    'S1_TOTAL_STEPS':     300_000,
    'S1_VAL_INTERVAL':    15_000,     # validate every 15 pseudo-epochs
    'S1_ES_PATIENCE':     5,
    'SKIP_PRETRAIN':      False,      # set True to jump straight to Stage 2
    'S1_BEST_CKPT_PATH':  None,       # optional: load pre-trained S1 weights
                                      # when SKIP_PRETRAIN=True

    # ── Stage 2 – fine-tuning on FE data ─────────────────────────────────
    # Total steps = 1 500 pseudo-epochs × 1000 steps (early-stop will fire first)
    'S2_TOTAL_STEPS':     1_500_000,
    'S2_VAL_INTERVAL':    15_000,
    'S2_ES_PATIENCE':     8,

    # ── Visibility thresholds (units: km, matching y_*.npy) ───────────────
    'THRESHOLD1':         0.5,        # 500 m
    'THRESHOLD2':         1.0,        # 1 000 m

    # ── Target metrics used in target_achievement score ───────────────────
    'TARGET_RECALL_500':    0.65,
    'TARGET_RECALL_1000':   0.75,
    'TARGET_ACCURACY':      0.95,
    'TARGET_LOW_VIS_PREC':  0.20,
    'TARGET_FPR_GOAL':      0.40,
}

# ==========================================
# 1. Distributed Utilities  (ported verbatim from the working Complex Model)
# ==========================================

def _enforce_nccl_shm_disable():
    """
    Force NCCL_SHM_DISABLE=1 before any RCCL/NCCL communicator is created.
    Prevents the SHM mutex crash on AMD DCU clusters.
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
    Uses a dummy all_reduce on the specific device to force communicator
    initialisation on the correct device, avoiding AMD DCU barrier hangs.
    torch.cuda.synchronize() is called after all_reduce to guarantee the GPU
    has fully committed before any rank proceeds.
    """
    if world_size > 1:
        if device is not None:
            dummy = torch.zeros(1, device=device)
            dist.all_reduce(dummy, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)
        else:
            dist.barrier()


def get_available_space(path: str) -> int:
    try:
        stat = os.statvfs(path)
        return stat.f_bavail * stat.f_frsize
    except Exception:
        return 0


def copy_to_local(src_path: str, global_rank: int, local_rank: int,
                  world_size: int, exp_id: str = None) -> str:
    """
    Copy NFS data file to local /tmp to accelerate I/O.
    Each node's local_rank==0 performs the copy; safe_barrier synchronises
    all ranks before and after so no rank proceeds with a stale file.
    """
    target_dir = "/tmp"
    if exp_id is None:
        exp_id = "default_exp"

    file_hash  = hashlib.md5(
        f"{exp_id}_{os.path.abspath(src_path)}".encode()
    ).hexdigest()[:8]
    basename   = os.path.basename(src_path)
    local_path = os.path.join(
        target_dir,
        f"{os.path.splitext(basename)[0]}_{file_hash}"
        f"{os.path.splitext(basename)[1]}"
    )

    device = torch.device(f"cuda:{local_rank}")

    # Synchronise all ranks before any I/O
    safe_barrier(world_size, device)

    if local_rank == 0:
        try:
            if not os.path.exists(src_path):
                if global_rank == 0:
                    print(f"[Data-Copy] Warning: Source {src_path} not found.",
                          flush=True)
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
                            print(f"[Data-Copy] Insufficient space on "
                                  f"{target_dir}, using NFS.", flush=True)
                    else:
                        if global_rank == 0:
                            print(f"[Data-Copy] Copying {basename} to "
                                  f"{target_dir}...", flush=True)
                        tmp_path = local_path + ".tmp"
                        shutil.copyfile(src_path, tmp_path)
                        os.rename(tmp_path, local_path)
                        if global_rank == 0:
                            print(f"[Data-Copy] Done: {local_path}", flush=True)
        except Exception as e:
            if global_rank == 0:
                print(f"[Data-Copy] Error: {e}, falling back to NFS.", flush=True)

    # Synchronise all ranks after copy
    safe_barrier(world_size, device)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    return src_path


def init_distributed():
    """
    Initialise the distributed process group using an explicit TCPStore.

    ROOT-CAUSE FIX vs. original Simple Model:
        The original called dist.init_process_group(backend="nccl") without
        specifying a store.  torchrun had already bound a TCPStore on
        MASTER_PORT for its own rendezvous, so init_process_group tried to
        bind a *second* TCPStore on the same port → "Address already in use"
        on rank 0, followed by "Connection reset by peer" on every other rank.

    FIX: read DIST_STORE_PORT (set in the SLURM script as MASTER_PORT+1) and
    create an explicit dist.TCPStore on that separate port.  torchrun does NOT
    override custom env-vars, so DIST_STORE_PORT passes through untouched.
    """
    _enforce_nccl_shm_disable()

    local_rank  = int(os.environ.get("LOCAL_RANK",  os.environ.get("SLURM_LOCALID",  0)))
    global_rank = int(os.environ.get("RANK",         os.environ.get("SLURM_PROCID",   0)))
    world_size  = int(os.environ.get("WORLD_SIZE",   os.environ.get("SLURM_NTASKS",   1)))

    # Must come BEFORE init_process_group on AMD DCU clusters
    torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        master_addr     = os.environ.get("MASTER_ADDR", "127.0.0.1")
        torchrun_port   = int(os.environ.get("MASTER_PORT",     29500))
        dist_store_port = int(os.environ.get("DIST_STORE_PORT", torchrun_port + 1))

        if global_rank == 0:
            print(
                f"[Dist] Initializing TCPStore for dist.init_process_group\n"
                f"       master_addr     = {master_addr}\n"
                f"       RDZV_PORT       = {torchrun_port}   "
                f"(owned by torchrun, NOT used here)\n"
                f"       DIST_STORE_PORT = {dist_store_port} "
                f"(used for dist.TCPStore)\n"
                f"       world_size      = {world_size}",
                flush=True,
            )

        store = dist.TCPStore(
            host_name        = master_addr,
            port             = dist_store_port,
            world_size       = world_size,
            is_master        = (global_rank == 0),
            timeout          = datetime.timedelta(minutes=30),
            wait_for_workers = True,
        )

        dist.init_process_group(
            backend    = "nccl",
            store      = store,
            rank       = global_rank,
            world_size = world_size,
            timeout    = datetime.timedelta(minutes=30),
        )

        if global_rank == 0:
            print("[Dist] Process group initialized successfully.", flush=True)
            print(
                f"[Dist] world_size={world_size}, "
                f"global_rank={global_rank}, local_rank={local_rank}",
                flush=True,
            )
            print(
                f"[Dist] NCCL_SHM_DISABLE="
                f"{os.environ.get('NCCL_SHM_DISABLE', 'NOT SET')}",
                flush=True,
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
# 2. Balanced Batch Sampler
# ==========================================

class BalancedBatchSampler(Sampler):
    """
    Deterministic, DDP-safe balanced batch sampler.

    Fixes vs. original InfiniteBalancedSampler:
      • Uses np.random.default_rng(seed + rank + epoch*997) — same pattern as
        the Complex Model — so every rank is deterministic and reproducible,
        and DDP ranks do not diverge.
      • set_epoch() advances the seed between pseudo-epochs (required by DDP
        to prevent all ranks seeing identical batches across epochs).
      • epoch_length is finite; __len__ returns epoch_length so DataLoader
        can reason about dataset size normally.
      • Class-index sharding is done per-rank to prevent inter-rank data overlap.
    """
    def __init__(self, dataset, batch_size: int,
                 pos_ratio: float = 0.25,
                 rank: int = 0, world_size: int = 1,
                 seed: int = 42, epoch_length: int = 1000):
        self.dataset      = dataset
        self.batch_size   = batch_size
        self.rank         = rank
        self.world_size   = world_size
        self.seed         = seed
        self.epoch_length = epoch_length
        self._epoch       = 0

        y = dataset.y_cls.numpy()
        all_positions = np.arange(len(y))

        pos_idx = all_positions[y <= 1]   # fog (0) + mist (1)
        neg_idx = all_positions[y == 2]   # clear

        # Shard across ranks to avoid inter-rank data overlap
        pos_splits = np.array_split(pos_idx, world_size)
        neg_splits = np.array_split(neg_idx, world_size)
        self.pos_indices = pos_splits[rank % len(pos_splits)]
        self.neg_indices = neg_splits[rank % len(neg_splits)]

        if len(self.pos_indices) == 0:
            self.pos_indices = pos_idx[:1]
        if len(self.neg_indices) == 0:
            self.neg_indices = neg_idx[:1]

        self.n_pos = max(1, int(batch_size * pos_ratio))
        self.n_neg = batch_size - self.n_pos

        if rank == 0:
            print(
                f"[Sampler] Pos (low-vis): {len(pos_idx):,}, "
                f"Neg (clear): {len(neg_idx):,} | "
                f"Batch: {self.n_pos} pos + {self.n_neg} neg",
                flush=True,
            )

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(
            seed=self.seed + self.rank + self._epoch * 997
        )
        for _ in range(self.epoch_length):
            pos   = rng.choice(self.pos_indices, size=self.n_pos, replace=True)
            neg   = rng.choice(self.neg_indices, size=self.n_neg, replace=True)
            batch = np.concatenate([pos, neg])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.epoch_length

# ==========================================
# 3. Loss Function  (preserved from Simple Model; step-index replaces epoch)
# ==========================================

class ProgressiveBalancedLoss(nn.Module):
    """
    Progressive balanced focal loss (unchanged from Simple Model).

    The 'epoch' parameter is now treated as a *step* index to work
    with the step-based training loop.  Set total_epochs = total_steps
    when constructing this loss.
    """
    def __init__(self, class_weights=None,
                 initial_recall_weight: float = 6.0,
                 target_recall_weight: float = 3.5,
                 initial_precision_weight: float = 1.2,
                 target_precision_weight: float = 1.8,
                 total_epochs: int = 300_000,
                 gamma: float = 2.0):
        super().__init__()
        self.class_weights            = class_weights
        self.initial_recall_weight    = initial_recall_weight
        self.target_recall_weight     = target_recall_weight
        self.initial_precision_weight = initial_precision_weight
        self.target_precision_weight  = target_precision_weight
        self.total_epochs             = total_epochs
        self.gamma                    = gamma

    def get_dynamic_weights(self, epoch: int):
        progress      = min(epoch / (self.total_epochs * 0.7), 1.0)
        smooth_factor = 1.0 / (1.0 + math.exp(-10.0 * (progress - 0.5)))
        recall_weight = (
            self.initial_recall_weight
            + (self.target_recall_weight - self.initial_recall_weight)
            * smooth_factor
        )
        precision_weight = (
            self.initial_precision_weight
            + (self.target_precision_weight - self.initial_precision_weight)
            * smooth_factor
        )
        return recall_weight, precision_weight

    def forward(self, inputs, targets, step: int = 0):
        recall_weight, precision_weight = self.get_dynamic_weights(step)

        ce_loss    = F.cross_entropy(inputs, targets, reduction='none')
        pt         = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss

        if self.class_weights is not None:
            alpha_t    = self.class_weights.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        probs        = F.softmax(inputs, dim=1)
        _, predicted = torch.max(probs, 1)

        # False-negative boosting (missed low-vis events)
        fn_mask = (targets <= 1) & (predicted == 2)
        if fn_mask.any():
            focal_loss[fn_mask] = focal_loss[fn_mask] * recall_weight

        # False-positive boosting (clear predicted as low-vis)
        fp_mask = (targets == 2) & (predicted <= 1)
        if fp_mask.any():
            focal_loss[fp_mask] = focal_loss[fp_mask] * precision_weight

        return focal_loss.mean()

# ==========================================
# 4. Model Architecture  (preserved unchanged from Simple Model)
# ==========================================

class EnhancedAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)                          # (B, 1, D)
        attn_out, attn_weights = self.self_attention(x, x, x)
        x      = self.norm1(x + attn_out)
        x_flat = x.squeeze(1)                           # (B, D)
        gate   = self.feature_gate(x_flat)
        x_gated = x_flat * gate
        ffn_out = self.ffn(x_gated)
        x_final = self.norm2(x_gated + ffn_out)
        return x_final, attn_weights


class BalancedLowVisibilityClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 num_classes: int = 3, dropout: float = 0.25):
        super().__init__()
        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.num_classes = num_classes

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),

            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim, dropout) for _ in range(4)
        ])

        # num_heads=12 requires hidden_dim % 12 == 0
        # With hidden_dim=1320: 1320 % 12 == 0 ✓
        self.attention = EnhancedAttentionBlock(
            hidden_dim, num_heads=12, dropout=dropout
        )

        self.low_vis_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.fine_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(hidden_dim // 4, num_classes),
        )

        # Input: (num_classes + 1) = 4
        self.fusion_net = nn.Sequential(
            nn.Linear(num_classes + 1, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_dim // 4, num_classes),
        )

        self.apply(self._init_weights)

    def _make_residual_block(self, hidden_dim: int, dropout: float):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        features = self.feature_extractor(x)

        for block in self.residual_blocks:
            residual = features
            features = F.relu(block(features) + residual)

        features, attention_weights = self.attention(features)

        low_vis_confidence = self.low_vis_detector(features)    # (B, 1)
        fine_logits        = self.fine_classifier(features)     # (B, C)

        combined     = torch.cat([fine_logits, low_vis_confidence], dim=1)  # (B, C+1)
        final_logits = self.fusion_net(combined)                # (B, C)

        return final_logits, low_vis_confidence, attention_weights

# ==========================================
# 5. Dataset  (lazy mmap — fixes inherited-fd bug)
# ==========================================

class BalancedVisDataset(Dataset):
    """
    Lazy-mmap dataset.

    BUG FIX vs. original:
        The original opened self.X = np.load(X_path, mmap_mode='r') in
        __init__.  When DataLoader forks worker processes, every worker
        inherits the same file descriptor.  Concurrent reads from different
        workers on the same fd cause data corruption and occasional crashes.

    FIX: self.X is set to None in __init__ and opened on first __getitem__
    call inside each worker (same pattern as the Complex Model).  The
    worker_init_fn in train_stage() also resets self.X = None on worker
    startup to ensure a fresh handle even with persistent_workers=True.
    """
    def __init__(self, X_path: str, y_cls: np.ndarray,
                 center: np.ndarray = None,
                 scale: np.ndarray = None):
        self.X_path = X_path
        self.y_cls  = torch.as_tensor(y_cls, dtype=torch.long)
        # Store scaler parameters as plain numpy arrays (pickle-safe)
        self.center = center
        self.scale  = scale if scale is not None else np.ones(1, dtype=np.float32)
        self.X      = None   # opened lazily per worker

    def __len__(self):
        return len(self.y_cls)

    def __getitem__(self, idx):
        if self.X is None:
            self.X = np.load(self.X_path, mmap_mode='r')

        features = self.X[idx].astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if self.center is not None:
            features = (features - self.center) / (self.scale + 1e-8)

        features = np.clip(features, -10.0, 10.0)
        return torch.from_numpy(features).float(), self.y_cls[idx]

# ==========================================
# 6. Data Loading
# ==========================================

def convert_visibility_to_class(visibility: np.ndarray,
                                 threshold1: float = 0.5,
                                 threshold2: float = 1.0) -> np.ndarray:
    """
    0: < 500 m (vis < threshold1 km)
    1: 500 – 1 000 m (threshold1 ≤ vis < threshold2 km)
    2: > 1 000 m (vis ≥ threshold2 km)
    """
    classes = np.zeros_like(visibility, dtype=np.int64)
    classes[visibility >= threshold1] = 1
    classes[visibility >= threshold2] = 2
    return classes


def load_data(data_dir: str, tag: str,
              global_rank: int, local_rank: int,
              device: torch.device,
              world_size: int, exp_id: str):
    """
    Load (X_train, y_train, X_val, y_val), fit / load a cached RobustScaler,
    and return (train_dataset, val_dataset, class_weights_cpu).
    """
    path_X_tr = copy_to_local(os.path.join(data_dir, 'X_train.npy'),
                               global_rank, local_rank, world_size, exp_id)
    path_y_tr = copy_to_local(os.path.join(data_dir, 'y_train.npy'),
                               global_rank, local_rank, world_size, exp_id)
    path_X_va = copy_to_local(os.path.join(data_dir, 'X_val.npy'),
                               global_rank, local_rank, world_size, exp_id)
    path_y_va = copy_to_local(os.path.join(data_dir, 'y_val.npy'),
                               global_rank, local_rank, world_size, exp_id)

    y_tr = np.load(path_y_tr).astype(np.float32)
    y_va = np.load(path_y_va).astype(np.float32)

    t1, t2 = CONFIG['THRESHOLD1'], CONFIG['THRESHOLD2']
    y_tr_cls = convert_visibility_to_class(y_tr, t1, t2)
    y_va_cls = convert_visibility_to_class(y_va, t1, t2)

    # ── Fit / load scaler (rank-0 fits once, then cached via joblib) ──────
    scaler_path = os.path.join(
        CONFIG['SAVE_CKPT_DIR'], f'robust_scaler_simple_{tag}.pkl'
    )

    safe_barrier(world_size, device)

    if global_rank == 0:
        if not os.path.exists(scaler_path):
            print(f"[Scaler-{tag}] Fitting (first time, will be cached)...",
                  flush=True)
            X_m = np.load(path_X_tr, mmap_mode='r')
            n   = len(X_m)
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=min(n, 200_000), replace=False)
            idx.sort()
            sub = X_m[idx].astype(np.float32)
            sub = np.nan_to_num(sub, nan=0.0)
            sc  = RobustScaler(quantile_range=(5.0, 95.0)).fit(sub)
            joblib.dump(sc, scaler_path)
            del sub, X_m
            gc.collect()
            print(f"[Scaler-{tag}] Saved to {scaler_path}", flush=True)
        else:
            print(f"[Scaler-{tag}] Cache found, loading.", flush=True)

    safe_barrier(world_size, device)

    # All ranks wait until the file appears (NFS can lag)
    wait = 0
    while not os.path.exists(scaler_path) and wait < 60:
        time.sleep(1)
        wait += 1
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Rank {global_rank} timed out waiting for {scaler_path}"
        )

    sc     = joblib.load(scaler_path)
    center = sc.center_.astype(np.float32)
    scale  = np.where(sc.scale_ == 0, 1.0, sc.scale_).astype(np.float32)

    # ── Class weights for ProgressiveBalancedLoss ─────────────────────────
    counts = Counter(y_tr_cls.tolist())
    total  = len(y_tr_cls)
    class_weights = torch.FloatTensor([
        float(np.sqrt(total / (3.0 * max(counts[k], 1)))) for k in range(3)
    ])

    if global_rank == 0:
        print(f"[Data-{tag}] Train={len(y_tr_cls):,}, Val={len(y_va_cls):,}",
              flush=True)
        for k in range(3):
            print(f"  class {k}: {counts[k]:,} ({counts[k]/total*100:.2f}%)",
                  flush=True)
        print(f"[Data-{tag}] Class weights: {class_weights.tolist()}", flush=True)

    tr_ds = BalancedVisDataset(path_X_tr, y_tr_cls, center, scale)
    va_ds = BalancedVisDataset(path_X_va, y_va_cls, center, scale)

    return tr_ds, va_ds, class_weights

# ==========================================
# 7. Evaluation
# ==========================================

def calculate_comprehensive_metrics(y_true: np.ndarray,
                                     y_pred: np.ndarray) -> dict:
    accuracy      = float(np.mean(y_true == y_pred))
    precision_cls = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_cls    = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_cls        = f1_score(y_true, y_pred, average=None, zero_division=0)

    recall_500 = float(recall_cls[0]) if len(recall_cls) > 0 else 0.0

    low_vis_true      = y_true <= 1
    low_vis_pred      = y_pred <= 1
    recall_1000       = float(
        np.sum(low_vis_true & low_vis_pred) / max(np.sum(low_vis_true), 1)
    )
    low_vis_precision = float(
        np.sum(low_vis_true & low_vis_pred) / max(np.sum(low_vis_pred), 1)
    )
    is_clear            = y_true == 2
    false_positive_rate = float(
        np.sum(is_clear & low_vis_pred) / max(np.sum(is_clear), 1)
    )

    target_achievement = (
        min(recall_500        / CONFIG['TARGET_RECALL_500'],   1.0) * 0.30 +
        min(recall_1000       / CONFIG['TARGET_RECALL_1000'],  1.0) * 0.30 +
        min(accuracy          / CONFIG['TARGET_ACCURACY'],     1.0) * 0.25 +
        min(low_vis_precision / CONFIG['TARGET_LOW_VIS_PREC'], 1.0) * 0.10 +
        min((1.0 - false_positive_rate) /
            (1.0 - CONFIG['TARGET_FPR_GOAL']), 1.0) * 0.05
    )

    return {
        'accuracy':            accuracy,
        'recall_500':          recall_500,
        'recall_1000':         recall_1000,
        'low_vis_precision':   low_vis_precision,
        'false_positive_rate': false_positive_rate,
        'precision_per_class': precision_cls,
        'recall_per_class':    recall_cls,
        'f1_per_class':        f1_cls,
        'f1_macro':    f1_score(y_true, y_pred, average='macro',    zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix':    confusion_matrix(y_true, y_pred),
        'target_achievement':  float(target_achievement),
    }


def evaluate(model, val_loader, device,
             rank: int, world_size: int,
             actual_val_size: int = None) -> dict:
    """
    Multi-GPU evaluation with DDP all-gather.

    BUG FIX vs. original:
        The original used a plain DataLoader (no DistributedSampler) so every
        rank ran over the full validation set independently.  On 40 GPUs this
        is both wasteful and — more importantly — means rank-0's print was not
        representative of the DDP model (DDP averages gradients across ranks
        but the validation metrics were computed per-rank and only rank-0's
        were printed, giving false confidence).

    FIX: Use DistributedSampler for val + all-gather results → rank-0 computes
    metrics over the complete dataset, matching the Complex Model pattern.
    """
    model.eval()
    preds_l, targets_l = [], []

    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device, non_blocking=True)
            logits, _, _ = model(bx)
            preds_l.append(torch.argmax(logits, dim=1))
            targets_l.append(by.to(device))

    local_preds   = torch.cat(preds_l,   dim=0)
    local_targets = torch.cat(targets_l, dim=0)

    # ── DDP all-gather ────────────────────────────────────────────────────
    if world_size > 1:
        local_size = torch.tensor([local_preds.size(0)],
                                   dtype=torch.long, device=device)
        max_size = local_size.clone()
        dist.all_reduce(max_size, op=dist.ReduceOp.MAX)

        if local_size < max_size:
            pad = max_size.item() - local_size.item()
            local_preds   = torch.cat([
                local_preds,
                torch.full((pad,), -1,
                           dtype=local_preds.dtype, device=device)
            ])
            local_targets = torch.cat([
                local_targets,
                torch.full((pad,), -1,
                           dtype=local_targets.dtype, device=device)
            ])

        all_preds_list   = [torch.zeros_like(local_preds)   for _ in range(world_size)]
        all_targets_list = [torch.zeros_like(local_targets) for _ in range(world_size)]
        dist.all_gather(all_preds_list,   local_preds)
        dist.all_gather(all_targets_list, local_targets)

        all_preds   = torch.cat(all_preds_list,   dim=0).cpu().numpy()
        all_targets = torch.cat(all_targets_list, dim=0).cpu().numpy()
    else:
        all_preds   = local_preds.cpu().numpy()
        all_targets = local_targets.cpu().numpy()

    metrics = None
    ta      = -1.0

    if rank == 0:
        n = actual_val_size if actual_val_size is not None else len(all_preds)
        preds   = all_preds[:n]
        targets = all_targets[:n]

        valid   = targets >= 0
        preds   = preds[valid]
        targets = targets[valid]

        metrics = calculate_comprehensive_metrics(targets, preds)
        ta      = metrics['target_achievement']

        print(
            f"  [Eval] Acc={metrics['accuracy']:.3f} | "
            f"FogR={metrics['recall_500']:.3f} | "
            f"MistR={metrics['recall_1000']:.3f} | "
            f"LVPrec={metrics['low_vis_precision']:.3f} | "
            f"FPR={metrics['false_positive_rate']:.3f} | "
            f"TA={ta:.4f}",
            flush=True,
        )
        cm = metrics['confusion_matrix']
        print(f"  [Eval] Confusion matrix:", flush=True)
        print(f"         Pred: <500m  500-1km  >1km", flush=True)
        for i, label in enumerate(['<500m (true)', '500-1km    ', '>1km  (true)']):
            print(f"  {label}: {cm[i, 0]:6d}  {cm[i, 1]:6d}  {cm[i, 2]:6d}",
                  flush=True)

    # Broadcast ta to all ranks for symmetric early-stop
    if world_size > 1:
        ta_t = torch.tensor([ta], dtype=torch.float32, device=device)
        dist.broadcast(ta_t, src=0)
        ta = ta_t.item()
        safe_barrier(world_size, device)

    return {'score': ta, 'metrics': metrics}

# ==========================================
# 8. Checkpoint Utilities
# ==========================================

def save_checkpoint(model, path: str, rank: int):
    if rank != 0:
        return
    state = (model.module.state_dict()
             if hasattr(model, 'module')
             else model.state_dict())
    torch.save(state, path)
    print(f"  [Ckpt] Saved -> {os.path.basename(path)}", flush=True)


def load_checkpoint(model, path: str, rank: int,
                    world_size: int, device: torch.device):
    """
    Broadcasts path-existence flag from rank 0 so a missing file is detected
    symmetrically on all ranks without leaving any rank stuck at a barrier.
    """
    path_exists = os.path.exists(path)
    if world_size > 1:
        et = torch.tensor([int(path_exists)], dtype=torch.long, device=device)
        dist.broadcast(et, src=0)
        path_exists = bool(et.item())

    if not path_exists:
        raise FileNotFoundError(f"[Ckpt] Not found on all ranks: {path}")

    if rank == 0:
        print(f"[Ckpt] Loading: {os.path.basename(path)}", flush=True)

    safe_barrier(world_size, device)

    sd     = torch.load(path, map_location=device)
    target = model.module if hasattr(model, 'module') else model
    missing, unexpected = target.load_state_dict(sd, strict=False)

    if rank == 0:
        if missing:
            print(f"  [Ckpt] Missing  ({len(missing)}): {missing[:3]}", flush=True)
        if unexpected:
            print(f"  [Ckpt] Unexpected ({len(unexpected)}): {unexpected[:3]}",
                  flush=True)
        print("  [Ckpt] Load complete.", flush=True)

    safe_barrier(world_size, device)


def wrap_ddp(raw_model, local_rank: int, world_size: int,
             find_unused: bool = False):
    """
    Wrap raw model in DDP.
    safe_barrier is called first so all ranks have consistent parameter state
    before the DDP reducer is constructed.
    """
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        safe_barrier(world_size, device)
        return DDP(raw_model, device_ids=[local_rank],
                   find_unused_parameters=find_unused)
    return raw_model


def rewrap_ddp(model, world_size: int):
    """Extract the raw nn.Module from a DDP wrapper."""
    if world_size > 1 and hasattr(model, 'module'):
        return model.module
    return model

# ==========================================
# 9. Training Function
# ==========================================

def train_stage(tag: str, model, tr_ds, val_ds,
                optimizer, loss_fn, device,
                rank: int, world_size: int,
                total_steps: int, val_int: int,
                batch_size: int, grad_accum: int,
                pos_ratio: float, exp_id: str,
                patience: int = 8) -> None:
    """
    Step-based training loop with early stopping and DDP-safe synchronisation.

    Key improvements vs. the original Simple Model training loop:
      • worker_init_fn resets the mmap handle in each DataLoader worker so
        every worker opens its own file descriptor (avoids the inherited-fd bug).
      • Gradient accumulation is handled with model.no_sync() on non-update
        steps, matching the Complex Model pattern and avoiding unnecessary
        all-reduce calls.
      • no_improve_count is broadcast from rank 0 to all ranks so early
        stopping is symmetric (all ranks exit together).
      • AMP (GradScaler / autocast) removed — not used by the Complex Model
        and can cause AMD-specific failures.
    """
    def worker_init_fn(worker_id):
        wi = torch.utils.data.get_worker_info()
        if wi is not None:
            wi.dataset.X = None   # each worker opens its own mmap fd

    sampler = BalancedBatchSampler(
        tr_ds, batch_size,
        pos_ratio=pos_ratio,
        rank=rank, world_size=world_size,
        epoch_length=CONFIG['STEPS_PER_EPOCH'],
    )
    loader = DataLoader(
        tr_ds,
        batch_sampler=sampler,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )

    val_sampler = (DistributedSampler(val_ds, num_replicas=world_size,
                                       rank=rank, shuffle=False)
                   if world_size > 1 else None)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )
    actual_val_size = len(val_ds)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=5e-6
    )

    ckpt_dir             = CONFIG['SAVE_CKPT_DIR']
    path_best            = os.path.join(ckpt_dir, f"{exp_id}_{tag}_best_score.pt")
    path_best_fog_recall = os.path.join(ckpt_dir, f"{exp_id}_{tag}_best_fog_recall.pt")
    path_best_fog_prec   = os.path.join(ckpt_dir, f"{exp_id}_{tag}_best_fog_prec.pt")
    path_latest          = os.path.join(ckpt_dir, f"{exp_id}_{tag}_latest.pt")

    best_score       = -1.0
    best_fog_recall  = -1.0
    best_fog_prec    = -1.0
    no_improve_count = 0

    if rank == 0:
        print(
            f"\n[{tag}] Training started. total_steps={total_steps}, "
            f"val_interval={val_int}, batch_size={batch_size}, "
            f"grad_accum={grad_accum}, patience={patience}",
            flush=True,
        )

    step         = 0
    batch_count  = 0
    pseudo_epoch = 0
    iterator     = iter(loader)
    model.train()

    while step < total_steps:
        try:
            bx, by = next(iterator)
        except StopIteration:
            pseudo_epoch += 1
            sampler.set_epoch(pseudo_epoch)
            iterator = iter(loader)
            bx, by = next(iterator)

        bx, by    = bx.to(device), by.to(device)
        batch_count += 1

        logits, _, _ = model(bx)
        loss = loss_fn(logits, by, step) / grad_accum

        is_last_accum = (batch_count % grad_accum == 0)
        if world_size > 1 and not is_last_accum:
            ctx = model.no_sync()
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            loss.backward()

        if is_last_accum:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), CONFIG['GRAD_CLIP']
            )
            if torch.isfinite(grad_norm):
                optimizer.step()
                scheduler.step()
            else:
                if rank == 0:
                    print(
                        f"\n[WARNING] Step {step}: NaN/Inf grad norm "
                        f"({grad_norm:.4f}), skipping optimizer step.",
                        flush=True,
                    )
            optimizer.zero_grad()
            step += 1

            if rank == 0 and step % 500 == 0:
                print(
                    f"\r[{tag}] Step {step:>8}/{total_steps} | "
                    f"Loss={loss.item()*grad_accum:.4f} | "
                    f"LR={scheduler.get_last_lr()[0]:.2e} | "
                    f"NoImprove={no_improve_count}/{patience}",
                    end="", flush=True,
                )

            if step % val_int == 0:
                if rank == 0:
                    print(f"\n[{tag}] === Validation at step {step} ===",
                          flush=True)

                res   = evaluate(model, val_loader, device,
                                  rank, world_size, actual_val_size)
                model.train()

                ta      = res['score']
                metrics = res['metrics']

                save_checkpoint(model, path_latest, rank)

                if rank == 0 and metrics is not None:
                    fog_r = (float(metrics['recall_per_class'][0])
                             if len(metrics['recall_per_class']) > 0 else 0.0)
                    fog_p = (float(metrics['precision_per_class'][0])
                             if len(metrics['precision_per_class']) > 0 else 0.0)

                    if ta > best_score:
                        best_score       = ta
                        no_improve_count = 0
                        save_checkpoint(model, path_best, rank)
                        print(f"  [Ckpt] ★ New best TA = {best_score:.4f}",
                              flush=True)
                        all_targets_met = (
                            metrics['recall_500']  >= CONFIG['TARGET_RECALL_500']  and
                            metrics['recall_1000'] >= CONFIG['TARGET_RECALL_1000'] and
                            metrics['accuracy']    >= CONFIG['TARGET_ACCURACY']
                        )
                        if all_targets_met:
                            print("  [Eval] 🎉 All targets achieved!", flush=True)
                    else:
                        no_improve_count += 1

                    if fog_r > best_fog_recall:
                        best_fog_recall = fog_r
                        save_checkpoint(model, path_best_fog_recall, rank)
                        print(f"  [Ckpt] ★ New best Fog Recall = "
                              f"{best_fog_recall:.4f}", flush=True)

                    if fog_p > best_fog_prec:
                        best_fog_prec = fog_p
                        save_checkpoint(model, path_best_fog_prec, rank)
                        print(f"  [Ckpt] ★ New best Fog Precision = "
                              f"{best_fog_prec:.4f}", flush=True)

                    print(
                        f"  [Best] Score={best_score:.4f} | "
                        f"FogR={best_fog_recall:.4f} | "
                        f"FogP={best_fog_prec:.4f} | "
                        f"NoImprove={no_improve_count}/{patience}",
                        flush=True,
                    )

                # Broadcast no_improve_count for symmetric early stop
                if world_size > 1:
                    st = torch.tensor([no_improve_count],
                                       dtype=torch.long, device=device)
                    dist.broadcast(st, src=0)
                    no_improve_count = st.item()

                if patience > 0 and no_improve_count >= patience:
                    if rank == 0:
                        print(
                            f"\n[{tag}] *** Early Stopping at step {step}. "
                            f"Best TA={best_score:.4f} ***",
                            flush=True,
                        )
                    safe_barrier(world_size, device)
                    break

    if rank == 0:
        print(
            f"\n[{tag}] Done.  Best TA={best_score:.4f} | "
            f"FogR={best_fog_recall:.4f} | FogP={best_fog_prec:.4f}",
            flush=True,
        )

# ==========================================
# 10. Main
# ==========================================

def main():
    l_rank, g_rank, w_size = init_distributed()
    device = torch.device(f"cuda:{l_rank}")

    exp_id = CONFIG['EXPERIMENT_ID']

    if g_rank == 0:
        os.makedirs(CONFIG['SAVE_CKPT_DIR'], exist_ok=True)
        os.makedirs(CONFIG['MODEL_SAVE_DIR'], exist_ok=True)
        print(f"[Simple-MLP] Exp: {exp_id}", flush=True)
        print(f"World size:  {w_size}", flush=True)

    # Global no-load synchronisation: force all DCUs to fully initialise the
    # NCCL/RCCL communicator before any disk I/O begins.  This eliminates the
    # hardware timing hang observed when rank-0 starts copying data while
    # non-zero ranks immediately hit a barrier.
    safe_barrier(w_size, device)

    if g_rank == 0:
        print("[Main] Initial barrier passed — all ranks synchronized.", flush=True)

    def build_model():
        m = BalancedLowVisibilityClassifier(
            input_dim   = CONFIG['INPUT_DIM'],
            hidden_dim  = CONFIG['HIDDEN_DIM'],
            num_classes = CONFIG['NUM_CLASSES'],
            dropout     = CONFIG['DROPOUT'],
        ).to(device)
        return m

    # ==========================================
    # Stage 1 — Pre-training on ERA5 reanalysis
    # ==========================================
    if not CONFIG['SKIP_PRETRAIN']:
        if g_rank == 0:
            print("\n" + "=" * 60, flush=True)
            print("[Stage 1] Pre-training on ERA5 data...", flush=True)

        tr_ds_s1, val_ds_s1, cw_s1 = load_data(
            CONFIG['S1_DATA_PATH'], 's1',
            g_rank, l_rank, device, w_size, exp_id,
        )

        raw_model_s1 = build_model()
        if g_rank == 0:
            n_params = sum(p.numel() for p in raw_model_s1.parameters())
            print(f"[Model] Total params = {n_params / 1e6:.2f}M", flush=True)

        model_s1   = wrap_ddp(raw_model_s1, l_rank, w_size, find_unused=False)
        loss_fn_s1 = ProgressiveBalancedLoss(
            class_weights=cw_s1.to(device),
            total_epochs=CONFIG['S1_TOTAL_STEPS'],
        )
        opt_s1 = optim.AdamW(
            model_s1.parameters(),
            lr=CONFIG['BASE_LR'],
            weight_decay=CONFIG['WEIGHT_DECAY'],
        )

        train_stage(
            tag='S1',
            model=model_s1, tr_ds=tr_ds_s1, val_ds=val_ds_s1,
            optimizer=opt_s1, loss_fn=loss_fn_s1, device=device,
            rank=g_rank, world_size=w_size,
            total_steps=CONFIG['S1_TOTAL_STEPS'],
            val_int=CONFIG['S1_VAL_INTERVAL'],
            batch_size=CONFIG['BATCH_SIZE'],
            grad_accum=CONFIG['GRAD_ACCUM'],
            pos_ratio=CONFIG['POS_RATIO'],
            exp_id=exp_id,
            patience=CONFIG['S1_ES_PATIENCE'],
        )

        # Extract raw model, release Stage-1 resources
        raw_model = rewrap_ddp(model_s1, w_size)
        del model_s1, opt_s1, loss_fn_s1, tr_ds_s1, val_ds_s1, cw_s1
        torch.cuda.empty_cache()
        gc.collect()
        safe_barrier(w_size, device)

        # Reload the best S1 checkpoint into the raw model
        s1_best = os.path.join(CONFIG['SAVE_CKPT_DIR'],
                               f"{exp_id}_S1_best_score.pt")
        if os.path.exists(s1_best):
            load_checkpoint(raw_model, s1_best, g_rank, w_size, device)
            if g_rank == 0:
                print(f"[Stage 2] Loaded S1 best weights: "
                      f"{os.path.basename(s1_best)}", flush=True)
        else:
            if g_rank == 0:
                print("[Stage 2] S1 best ckpt not found, "
                      "continuing with last-step weights.", flush=True)

    else:
        # Stage 1 skipped — optionally load a pre-trained checkpoint
        raw_model = build_model()
        if g_rank == 0:
            n_params = sum(p.numel() for p in raw_model.parameters())
            print(f"[Model] Total params = {n_params / 1e6:.2f}M", flush=True)
            print("[Stage 1] Skipped (SKIP_PRETRAIN=True).", flush=True)

        if CONFIG['S1_BEST_CKPT_PATH'] is not None:
            load_checkpoint(
                raw_model, CONFIG['S1_BEST_CKPT_PATH'], g_rank, w_size, device
            )
            if g_rank == 0:
                print(f"[Stage 2] Loaded pre-trained weights from: "
                      f"{CONFIG['S1_BEST_CKPT_PATH']}", flush=True)

    # ==========================================
    # Stage 2 — Fine-tuning on FE data
    # ==========================================
    if g_rank == 0:
        print("\n" + "=" * 60, flush=True)
        print("[Stage 2] Fine-tuning on FE data...", flush=True)

    tr_ds_s2, val_ds_s2, cw_s2 = load_data(
        CONFIG['S2_DATA_PATH'], 's2',
        g_rank, l_rank, device, w_size, exp_id,
    )

    model_s2   = wrap_ddp(raw_model, l_rank, w_size, find_unused=False)
    loss_fn_s2 = ProgressiveBalancedLoss(
        class_weights=cw_s2.to(device),
        total_epochs=CONFIG['S2_TOTAL_STEPS'],
    )
    opt_s2 = optim.AdamW(
        model_s2.parameters(),
        lr=CONFIG['BASE_LR'],
        weight_decay=CONFIG['WEIGHT_DECAY'],
    )

    train_stage(
        tag='S2',
        model=model_s2, tr_ds=tr_ds_s2, val_ds=val_ds_s2,
        optimizer=opt_s2, loss_fn=loss_fn_s2, device=device,
        rank=g_rank, world_size=w_size,
        total_steps=CONFIG['S2_TOTAL_STEPS'],
        val_int=CONFIG['S2_VAL_INTERVAL'],
        batch_size=CONFIG['BATCH_SIZE'],
        grad_accum=CONFIG['GRAD_ACCUM'],
        pos_ratio=CONFIG['POS_RATIO'],
        exp_id=exp_id,
        patience=CONFIG['S2_ES_PATIENCE'],
    )

    # ==========================================
    # Wrap-up
    # ==========================================
    if g_rank == 0:
        print("\n" + "=" * 60, flush=True)
        print("[Summary] Training complete.", flush=True)
        print(f"  S1 checkpoints : {exp_id}_S1_*.pt", flush=True)
        print(f"  S2 checkpoints : {exp_id}_S2_*.pt", flush=True)
        print(f"  All in         : {CONFIG['SAVE_CKPT_DIR']}", flush=True)

    cleanup_temp_files(exp_id)

    if w_size > 1:
        safe_barrier(w_size, device)
        dist.destroy_process_group()

    if g_rank == 0:
        print("\nJob finished.", flush=True)


if __name__ == "__main__":
    main()
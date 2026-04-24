import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import shutil
import hashlib
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
import contextlib
import sys
import datetime
import time
import json
import joblib

warnings.filterwarnings('ignore')

_STORE_BARRIER_SEQ = 0

# ==========================================
# 0. 全局配置
# ==========================================
TARGET_WINDOW_SIZE = 12
BASE_PATH = "/public/home/putianshu/vis_mlp"
S1_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_{TARGET_WINDOW_SIZE}h"
S2_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_cursor_2_{TARGET_WINDOW_SIZE}h"

CONFIG = {
    # ========== 实验控制 ==========
    #'EXPERIMENT_ID':           'exp_1772106326',
    #'EXPERIMENT_ID':           'exp_1772692533',
    'EXPERIMENT_ID':           'exp_1773291129',
    #'S2_RUN_SUFFIX':           os.environ.get('S2_RUN_SUFFIX', ''),
    'S2_RUN_SUFFIX':           '04',
    'BASE_PATH':              BASE_PATH,
    'WINDOW_SIZE':            TARGET_WINDOW_SIZE,
    'S1_DATA_DIR':            S1_DIR,
    'S2_DATA_DIR':            S2_DIR,
    'NUM_WORKERS':            2,
    'SAVE_CKPT_DIR':          os.path.join(BASE_PATH, 'checkpoints'),

    # ========== Stage 1 已完成，指定要加载的 S1 最优权重路径 ==========
    'S1_BEST_CKPT_PATH': '/public/home/putianshu/vis_mlp/checkpoints/exp_1773291129_S1_best_score.pt',

    # ==========================================
    # Stage 2 Phase A1 配置 (Exp4: 3-phase unfreezing)
    # ==========================================
    'S2_PHASE_A1_STEPS':       15000,
    'S2_LR_HEAD_A':            1e-4,
    'S2_LR_FUSION_A':          2e-5,

    # ==========================================
    # Stage 2 Phase A2 配置 (Exp4: unfreeze GRU+SE)
    # ==========================================
    'S2_PHASE_A2_STEPS':       15000,
    'S2_LR_HEAD_A2':           5e-5,
    'S2_LR_GRU_SE_A2':        5e-6,
    'S2_VAL_INTERVAL':         500,
    'S2_BATCH_SIZE':           512,
    'S2_GRAD_ACCUM':           2,

    # ==========================================
    # [修改] 采样比例：降低雾/薄雾过采样比例，使批次中晴天样本占比上升
    # 根因：原25/20%的过采样配合高类别权重，造成模型极度偏向低能见度预测
    # 修复：调回18/18%，批次中晴天样本占~64%，与现实分布更接近
    # ==========================================
    'S2_FOG_RATIO':            0.18,    # ← was 0.25
    'S2_MIST_RATIO':           0.18,    # ← was 0.20

    # ==========================================
    # Stage 2 Phase B 配置
    # ==========================================
    'S2_PHASE_B_STEPS':        30000,
    'S2_LR_BACKBONE_B':        3e-6,
    'S2_LR_HEAD_B':            1e-5,
    'S2_WEIGHT_DECAY':         1e-2,

    # ==========================================
    # [修改] S2损失函数参数 — 三项关键调整：
    #
    # 1. 类别权重再平衡：
    #    原(3.0/2.0/0.6)造成5×不对称压力。修复为(1.5/1.2/0.9)≈1.7×，
    #    晴天损失从降权0.6x提高到0.9x，让模型更认真对待晴天样本。
    #
    # 2. alpha_fp大幅提升(1.5→5.0)：
    #    原FP惩罚远弱于fog_boost激励，无法阻止假阳性。需要更强对抗力。
    #
    # 3. 新增 alpha_clear_margin + clear_margin（最关键修复）：
    #    显式强制晴天样本的 p(fog)+p(mist) < clear_margin。
    #    原损失函数没有任何直接约束晴天样本低能见度概率上界的项，
    #    这是精确率崩塌的根本原因。
    # ==========================================

    # S2专用二分类权重
    'S2_BINARY_POS_WEIGHT':     1.5,

    # S2专用细粒度类别权重
    'S2_FINE_CLASS_WEIGHT_FOG':   1.8,  # Exp1: was 1.5
    'S2_FINE_CLASS_WEIGHT_MIST':  1.6,  # Slightly reduced to curb clear->mist false alarms
    'S2_FINE_CLASS_WEIGHT_CLEAR': 0.9,  # Increase clear weight to stabilise mist precision

    # S2专用损失系数
    'S2_LOSS_ALPHA_BINARY':    0.7,
    'S2_LOSS_ALPHA_FINE':      1.0,
    'S2_LOSS_ALPHA_FP':        5.0,
    'S2_LOSS_ALPHA_FOG_BOOST': 0.4,     # Exp1: was 0.3
    'S2_LOSS_ALPHA_MIST_BOOST':0.2,     # Keep mist recall support, but avoid over-pushing mist positives

    # 晴天边界校准损失
    'S2_CLEAR_MARGIN':              0.20,   # Exp1: was 0.25 (更紧的边界)
    'S2_LOSS_ALPHA_CLEAR_MARGIN':   3.0,

    # ========== 评估约束条件 ==========
    'MIN_FOG_PRECISION':      0.20,
    'MIN_FOG_RECALL':         0.50,
    'MIN_MIST_PRECISION':     0.15,
    'MIN_MIST_RECALL':        0.20,
    'MIN_CLEAR_ACC':          0.90,
    'SEARCH_FOG_PRECISION_STRICT':   0.10,
    'SEARCH_MIST_PRECISION_STRICT':  0.10,
    'SEARCH_CLEAR_RECALL_STRICT':    0.90,
    'SEARCH_FOG_PRECISION_RELAXED':  0.08,
    'SEARCH_MIST_PRECISION_RELAXED': 0.08,
    'SEARCH_CLEAR_RECALL_RELAXED':   0.88,
    'SEASON_FOG_PRECISION_STRICT':   0.10,
    'SEASON_MIST_PRECISION_STRICT':  0.10,
    'SEASON_CLEAR_RECALL_STRICT':    0.85,
    'SEASON_FOG_PRECISION_RELAXED':  0.08,
    'SEASON_MIST_PRECISION_RELAXED': 0.08,
    'SEASON_CLEAR_RECALL_RELAXED':   0.83,
    'POSTPROC_MIN_MIST_PRECISION':   0.12,
    'POSTPROC_MIN_LOW_VIS_PRECISION':0.22,
    'POSTPROC_MAX_MIST_PREC_DROP':   0.02,
    'POSTPROC_MAX_LV_PREC_DROP':     0.02,
    'POSTPROC_MAX_CLEAR_RECALL_DROP':0.01,
    'POSTPROC_RELAX_MIST_PRECISION': 0.02,
    'POSTPROC_RELAX_LOW_VIS_PRECISION': 0.02,
    'POSTPROC_RELAX_CLEAR_RECALL':   0.02,
    'SEASON_FOG_MAX_DROP_FROM_GLOBAL': 0.12,
    'SEASON_MIST_MAX_DROP_FROM_GLOBAL': 0.08,
    'SEASON_MIST_MIN_THRESHOLD':     0.25,

    # ========== S1损失函数配置（保留，供build_loss()引用）==========
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
    'FE_EXTRA_DIMS':          36,
    'REGIME_FEAT_DIM':        6,
    'GRAD_CLIP_NORM':         0.5,
    'REG_LOSS_ALPHA':         0.1,
    'VAL_SPLIT_RATIO':        0.1,

    # ========== TODO #2 / #3 / #4 / #5: postprocess + hard examples ==========
    'S2_HARD_EXAMPLE_RATIO':        0.30,
    'S2_AUG_TEMP_MAX_DELTA':        0.6,
    'S2_AUG_RH_MAX_DELTA':          2.0,
    'S2_AUG_WIND_SCALE':            0.08,
    'S2_AUG_EXTRA_DROPOUT':         0.05,
    'S2_JJA_WEIGHT':                0.35,
    'S2_BOUNDARY_WEIGHT':           0.25,
    'S2_CLEAR_HARD_NEG_WEIGHT':     0.35,
    'S2_PAIR_MARGIN':               0.35,
    'S2_LOSS_ALPHA_PAIR_MARGIN':    0.30,
    'S2_CLEAR_MIST_PENALTY_TH':     0.30,
    'S2_LOSS_ALPHA_CLEAR_MIST':     1.00,
    'CLEAR_MIST_MANIFEST_PATH':     None,
    'CLEAR_MIST_MANIFEST_MIN_ROWS': 128,
    'CLEAR_MIST_MANIFEST_MIN_PMIST':0.35,
    'POSTPROC_OFFSET_GRID':         [-0.06, -0.03, 0.0, 0.03, 0.06],
    'POSTPROC_MIN_GROUP_SIZE':      128,
    'POSTPROC_PRIOR_GRIDS': {
        'mixing_low_vis': [1.0, 0.90, 0.80],
        'convective_fog': [1.0, 0.85, 0.70],
        'ventilation_low_vis': [1.0, 0.92, 0.84],
    },

    # ========== target_achievement 权重 ==========
    'TARGET_RECALL_500_GOAL':    0.65,
    'TARGET_RECALL_1000_GOAL':   0.75,
    'TARGET_FOG_PREC_GOAL':      0.10,
    'TARGET_MIST_PREC_GOAL':     0.10,
    'TARGET_ACCURACY_GOAL':      0.95,
    'TARGET_LOW_VIS_PREC_GOAL':  0.20,
    'TARGET_FPR_GOAL':           0.40,
    'TARGET_W_RECALL_500':       0.30,
    'TARGET_W_RECALL_1000':      0.20,
    'TARGET_W_FOG_PREC':         0.10,
    'TARGET_W_MIST_PREC':        0.20,
    'TARGET_W_ACCURACY':         0.10,
    'TARGET_W_LOW_VIS_PREC':     0.05,
    'TARGET_W_FPR':              0.05,

    # ========== Early Stopping ==========
    'S2_ES_PATIENCE':         12,

    # ========== Exp 4: LR Warmup ==========
    'S2_WARMUP_STEPS':         500,

    # ========== Exp 5: L2-SP Regularization ==========
    'L2SP_ALPHA_A':            1e-4,
    'L2SP_ALPHA_B':            5e-5,

    # ========== Exp 8: Temperature Scaling ==========
    'TEMP_SCALING_LR':         0.01,
    'TEMP_SCALING_MAX_ITER':   50,

    # ========== Exp 9: Label Smoothing ==========
    'USE_LABEL_SMOOTHING':     True,

    # ========== Exp 10: Physics Features ==========
    'PHYSICS_FEAT_DIM':        7,
}


def build_s2_run_exp_id(base_exp_id: str, run_suffix: str) -> str:
    """
    Build the S2 run identifier used for *all* S2 save artifacts.
    - base_exp_id: shared experiment id, typically aligned with the pretrained S1 lineage
    - run_suffix: independent serial suffix for parallel / repeated S2 runs
    """
    suffix = str(run_suffix or "").strip()
    if not suffix:
        return base_exp_id
    return f"{base_exp_id}_{suffix}"

# ==========================================
# 1. 基础工具与分布式设置
# ==========================================

def _enforce_nccl_shm_disable():
    if os.environ.get('NCCL_SHM_DISABLE') != '1':
        os.environ['NCCL_SHM_DISABLE'] = '1'
        rank_str = os.environ.get('RANK', os.environ.get('SLURM_PROCID', None))
        if rank_str is None or str(rank_str) == '0':
            print("[Init] NCCL_SHM_DISABLE was not set; forcing to 1 "
                  "(prevents RCCL SHM mutex crash on AMD DCU cluster).", flush=True)


def safe_barrier(world_size: int, device: torch.device = None):
    """
    Drop-in replacement for dist.barrier().
    Prefer TCPStore-backed synchronization because the first RCCL/NCCL
    collective can hang on this AMD DCU cluster even after
    init_process_group() succeeds. Fall back to dist.barrier() only if the
    default store is unavailable.
    """
    global _STORE_BARRIER_SEQ

    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return

    default_store_getter = getattr(dist.distributed_c10d, "_get_default_store", None)
    if default_store_getter is not None:
        store = default_store_getter()
        barrier_id = _STORE_BARRIER_SEQ
        _STORE_BARRIER_SEQ += 1
        prefix = f"safe_barrier/{barrier_id}"
        rank = dist.get_rank()
        store.set(f"{prefix}/{rank}", "1")
        keys = [f"{prefix}/{r}" for r in range(world_size)]
        store.wait(keys, datetime.timedelta(minutes=30))
        return

    if device is not None:
        try:
            dist.barrier(device_ids=[device.index])
        except TypeError:
            dist.barrier()
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

    safe_barrier(world_size, device)

    if local_rank == 0:
        if world_size > 8:
            time.sleep(global_rank * 3)
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
                            print(f"[Data-Copy] Insufficient space on {target_dir}, using NFS.", flush=True)
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

    safe_barrier(world_size, device)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    else:
        return src_path


def init_distributed():
    """
    Initialize the distributed process group using an explicit TCPStore.
    Reads DIST_STORE_PORT (not MASTER_PORT) to avoid the port conflict
    caused by torchrun already binding MASTER_PORT.
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


def month_to_season_name(month: int) -> str:
    month = int(month)
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def is_coastal_location(latitudes, longitudes):
    latitudes = np.asarray(latitudes, dtype=np.float32)
    longitudes = np.asarray(longitudes, dtype=np.float32)
    return (longitudes >= 118.0) & (latitudes >= 18.0) & (latitudes <= 42.0)


def safe_average_precision(y_true, scores):
    y_true = np.asarray(y_true).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.0
    return float(average_precision_score(y_true, scores))


def extract_last_step_features(rows, window_size=12):
    split_dyn = window_size * 25
    last = rows[:, split_dyn - 25:split_dyn].astype(np.float32)
    u10 = last[:, 5]
    v10 = last[:, 7]
    u925 = last[:, 13]
    v925 = last[:, 15]
    wspd10 = np.maximum(last[:, 6], 0.0)
    sw_rad = np.maximum(last[:, 4], 0.0)
    precip = np.maximum(last[:, 2], 0.0)
    cape = np.maximum(last[:, 9], 0.0)
    inversion = last[:, 23]
    shear_mag = np.sqrt((u925 - u10) ** 2 + (v925 - v10) ** 2)
    mixing = (
        1.0 / (1.0 + np.exp(-(sw_rad - 150.0) / 75.0))
    ) * (
        1.0 / (1.0 + np.exp(-(wspd10 - 4.0) / 1.5))
    ) * (
        1.0 / (1.0 + np.exp(-(-inversion + 0.5) / 1.2))
    )
    convective = (
        1.0 / (1.0 + np.exp(-(np.log1p(cape) - np.log(200.0)) * 1.6))
    ) * (
        1.0 / (1.0 + np.exp(-(np.log1p(precip) - np.log(0.1)) * 2.5))
    )
    ventilation = np.tanh((wspd10 * (1.0 + shear_mag)) / 12.0)
    clear_low_vis_score = (
        1.0 / (1.0 + np.exp(-(last[:, 0] - 92.0) / 4.0))
    ) * (
        1.0 / (1.0 + np.exp(-(2.2 - last[:, 22]) / 0.9))
    ) * (
        1.0 / (1.0 + np.exp(-(4.0 - wspd10) / 1.5))
    ) * (1.0 - 0.35 * convective)
    return {
        'mixing': mixing,
        'convective': convective,
        'ventilation': ventilation,
        'clear_low_vis_score': clear_low_vis_score,
        'zenith': last[:, 24],
    }

# ==========================================
# 2. 数据采样器
# ==========================================

class StratifiedBalancedBatchSampler(Sampler):
    """
    The sampler stores and yields indices into the SUBSET (positions
    within the PMSTDataset), not into the original full array.
    """
    def __init__(self, dataset, batch_size, fog_ratio=0.2, mist_ratio=0.2,
                 rank=0, world_size=1, seed=42, epoch_length=2000,
                 hard_example_ratio=0.0):
        self.dataset      = dataset
        self.batch_size   = batch_size
        self.rank         = rank
        self.world_size   = world_size
        self.seed         = seed
        self.epoch_length = epoch_length
        self.hard_example_ratio = float(max(0.0, min(hard_example_ratio, 0.8)))
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

        hard_pools = getattr(dataset, "hard_pool_by_class", {})
        self.hard_indices = {}
        for k in self.pos_indices:
            pool = np.asarray(hard_pools.get(k, []), dtype=np.int64)
            if pool.size == 0:
                self.hard_indices[k] = self.pos_indices[k]
                continue
            pool = np.intersect1d(pool, self.pos_indices[k], assume_unique=False)
            self.hard_indices[k] = pool if pool.size > 0 else self.pos_indices[k]

    def _sample_class(self, rng, class_id, n_total):
        if n_total <= 0:
            return np.empty(0, dtype=np.int64)
        hard_pool = self.hard_indices.get(class_id, self.pos_indices[class_id])
        base_pool = self.pos_indices[class_id]
        n_hard = int(round(n_total * self.hard_example_ratio)) if len(hard_pool) > 0 else 0
        n_hard = min(n_total, n_hard)
        n_base = n_total - n_hard
        hard_part = rng.choice(hard_pool, size=n_hard, replace=True) if n_hard > 0 else np.empty(0, dtype=np.int64)
        base_part = rng.choice(base_pool, size=n_base, replace=True) if n_base > 0 else np.empty(0, dtype=np.int64)
        return np.concatenate([hard_part, base_part])

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(seed=self.seed + self.rank + self._epoch * 997)
        for _ in range(self.epoch_length):
            f = self._sample_class(rng, 0, self.n_fog)
            m = self._sample_class(rng, 1, self.n_mist)
            c = self._sample_class(rng, 2, self.n_clear)
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


class OrdinalAwareFocalLoss(nn.Module):
    """
    Exp2: Focal loss with class-conditional gamma and ordinal cost penalty.
    Higher gamma for mist concentrates gradient on hard mist samples.
    Ordinal cost penalizes far-class misclassification more heavily.
    """
    def __init__(self, ordinal_cost=None, gamma_per_class=None,
                 eps=1e-7, class_weights=None):
        super().__init__()
        if ordinal_cost is None:
            ordinal_cost = [[0, 1, 3], [1, 0, 2], [3, 2, 0]]
        if gamma_per_class is None:
            gamma_per_class = [2.5, 3.0, 0.5]
        self.register_buffer('ordinal_cost',
                             torch.tensor(ordinal_cost, dtype=torch.float32))
        self.register_buffer('gamma',
                             torch.tensor(gamma_per_class, dtype=torch.float32))
        self.eps = eps
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, targets, soft_targets=None):
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        if soft_targets is not None:
            gamma_expanded = self.gamma.unsqueeze(0)
            focal_weight = (1 - probs) ** gamma_expanded
            pos_loss = -(soft_targets * focal_weight * torch.log(probs)).sum(dim=1)
            expected_cost = torch.matmul(soft_targets, self.ordinal_cost)
            ordinal_penalty = (expected_cost * probs).sum(dim=1)
        else:
            targets_one_hot = F.one_hot(targets, num_classes).float()
            gamma_per_sample = self.gamma[targets]
            p_true = (probs * targets_one_hot).sum(dim=1)
            focal_weight = (1 - p_true) ** gamma_per_sample
            pos_loss = -focal_weight * torch.log(p_true)
            cost_per_sample = self.ordinal_cost[targets]
            ordinal_penalty = (cost_per_sample * probs).sum(dim=1)

        loss = pos_loss + ordinal_penalty
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return loss.mean()


class DualBranchLoss(nn.Module):
    """
    Dual-branch loss with clear-margin calibration and ordinal focal loss support.
    loss_type='ordinal_focal' uses OrdinalAwareFocalLoss (Exp2),
    loss_type='asymmetric' uses original AsymmetricLoss (S1 compat).
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

        loss_type = kwargs.get('loss_type', 'asymmetric')
        if loss_type == 'ordinal_focal':
            self.fine_loss = OrdinalAwareFocalLoss(
                ordinal_cost=kwargs.get('ordinal_cost', [[0, 1, 3], [1, 0, 2], [3, 2, 0]]),
                gamma_per_class=kwargs.get('gamma_per_class', [2.5, 3.0, 0.5]),
                class_weights=self.fine_class_weight,
            )
        else:
            self.fine_loss = AsymmetricLoss(
                gamma_neg=kwargs.get('asym_gamma_neg', 2),
                gamma_pos=kwargs.get('asym_gamma_pos', 0),
                clip=kwargs.get('asym_clip', 0.1),
                class_weights=self.fine_class_weight,
            )

    def forward(self, fine_logits, low_vis_logit, targets, soft_targets=None, sample_flags=None):
        # ── Binary branch ────────────────────────────────────────────────────
        low_vis_logit = torch.clamp(low_vis_logit, -20, 20)
        l_bin = F.binary_cross_entropy_with_logits(
            low_vis_logit,
            (targets <= 1).float().unsqueeze(1),
            pos_weight=self.pos_weight
        )

        # ── Fine-grained classification (Exp2/Exp9: ordinal focal + soft targets) ──
        if soft_targets is not None and isinstance(self.fine_loss, OrdinalAwareFocalLoss):
            l_fine = self.fine_loss(fine_logits, targets, soft_targets=soft_targets)
        else:
            l_fine = self.fine_loss(fine_logits, targets)

        probs    = F.softmax(fine_logits, dim=1)

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

        # ── Clear-margin calibration (KEY NEW TERM) ───────────────────────────
        # Squared-hinge loss that fires only when a clear sample's
        # p(fog)+p(mist) exceeds `clear_margin`.
        #
        # Why this works where l_fp alone fails:
        #   l_fp penalises ALL clear samples proportionally to (p_low)².
        #   For a well-trained model that already outputs p_low≈0.15, the
        #   gradient of l_fp is 2*0.15*alpha_fp — small and diminishing.
        #   l_clear_margin uses a hinge: zero gradient below the margin,
        #   strong gradient only for violators (p_low > margin).
        #   This concentrates gradient pressure exactly on the hard false
        #   positives that destroy precision, while leaving correctly
        #   calibrated clear samples untouched.
        clear_margin    = self.cfg.get('clear_margin', 0.30)
        margin_excess   = torch.relu(low_vis_prob - clear_margin)   # 0 for non-violators
        l_clear_margin  = torch.mean(margin_excess ** 2 * is_clear)

        is_jja = torch.zeros_like(is_clear)
        is_boundary = torch.zeros_like(is_clear)
        is_clear_hard = torch.zeros_like(is_clear)
        if sample_flags is not None:
            is_jja = sample_flags[:, 0].float()
            is_boundary = sample_flags[:, 1].float()
            is_clear_hard = sample_flags[:, 2].float()

        sample_weight = (
            1.0
            + self.cfg.get('jja_weight', 0.0) * is_jja * (is_fog + is_mist)
            + self.cfg.get('boundary_weight', 0.0) * is_boundary * (is_fog + is_mist)
            + self.cfg.get('clear_hard_neg_weight', 0.0) * is_clear_hard * is_clear
        )

        logit_gap = fine_logits[:, 0] - fine_logits[:, 1]
        signed_gap = torch.where(targets == 0, logit_gap, -logit_gap)
        pair_mask = ((targets == 0) | (targets == 1)).float()
        pair_margin = self.cfg.get('pair_margin', 0.35)
        l_pair_margin = torch.mean(F.softplus(pair_margin - signed_gap) * pair_mask * sample_weight)

        clear_mist_threshold = self.cfg.get('clear_mist_penalty_th', 0.45)
        clear_mist_excess = torch.relu(probs[:, 1] - clear_mist_threshold)
        l_clear_mist = torch.mean(clear_mist_excess ** 2 * is_clear * torch.maximum(sample_weight, torch.ones_like(sample_weight)))

        # ── Combine ───────────────────────────────────────────────────────────
        total = (
            self.cfg.get('alpha_binary',       1.0) * l_bin          +
            self.cfg.get('alpha_fine',         1.0) * l_fine         +
            self.cfg.get('alpha_fp',           0.5) * l_fp           +
            self.cfg.get('alpha_fog_boost',    0.2) * l_fb           +
            self.cfg.get('alpha_mist_boost',   0.0) * l_mb           +
            self.cfg.get('alpha_clear_margin', 0.0) * l_clear_margin  +
            self.cfg.get('alpha_pair_margin',  0.0) * l_pair_margin   +
            self.cfg.get('alpha_clear_mist',   0.0) * l_clear_mist
        )

        return total, {
            'bin':   l_bin.item(),
            'fine':  l_fine.item(),
            'cm':    l_clear_margin.item(),   # logged for diagnostics
            'pair':  l_pair_margin.item(),
            'cmt':   l_clear_mist.item(),
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
                 extra_feat_dim=0, regime_feat_dim=0, dropout=0.3):
        super().__init__()
        self.dyn_vars = dyn_vars_count
        self.window   = window_size
        self.extra_feat_dim = extra_feat_dim
        self.regime_feat_dim = regime_feat_dim
        self.temporal_var_indices = [0, 1, 2, 4, 6, 10, 22, 23, 12, 11, 13, 15, 14]

        self.veg_embedding = nn.Embedding(veg_num_classes, 16)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_cont_dim + 16, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, hidden_dim // 4)
        )

        physics_input_dim = 7  # Exp10: was 5, added saturation indicator + temp tendency
        self.physics_encoder = nn.Sequential(
            nn.Linear(physics_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim // 4)
        )

        self.temporal_input_proj  = nn.Linear(len(self.temporal_var_indices), hidden_dim)
        self.temporal_stream      = nn.GRU(hidden_dim, hidden_dim, num_layers=2,
                                           batch_first=True, bidirectional=True, dropout=dropout)
        self.se_block             = SEBlock(hidden_dim * 2)
        self.temporal_norm        = nn.LayerNorm(hidden_dim * 2)

        fusion_dim = (hidden_dim * 2) + (hidden_dim // 4) + (hidden_dim // 4)
        self.extra_encoder = None
        if extra_feat_dim > 0:
            self.extra_encoder = nn.Sequential(
                nn.Linear(extra_feat_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU()
            )
            fusion_dim += hidden_dim // 2
        self.regime_encoder = None
        if regime_feat_dim > 0:
            self.regime_encoder = nn.Sequential(
                nn.Linear(regime_feat_dim, hidden_dim // 8),
                nn.LayerNorm(hidden_dim // 8),
                nn.GELU()
            )
            fusion_dim += hidden_dim // 8

        self.fusion_kan = StableChebyKANLayer(fusion_dim, hidden_dim, degree=2)
        self.dropout    = nn.Dropout(dropout)

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
        f6 = torch.sigmoid((rh2m - 90) / 5.0)
        f7 = t2m[:, -1:].expand_as(t2m) - t2m[:, :1].expand_as(t2m)

        return torch.stack([f1, f2, f3, f4, f5, f6, f7], dim=2).mean(dim=1)

    def forward(self, x):
        split_dyn    = self.dyn_vars * self.window
        split_static = split_dyn + 5

        x_dyn   = x[:, :split_dyn].view(-1, self.window, self.dyn_vars)
        x_stat  = x[:, split_dyn:split_static]
        x_veg   = x[:, split_static].long()
        x_tail = x[:, split_static + 1:]
        x_extra = None
        x_regime = None
        if self.regime_encoder and self.regime_feat_dim > 0:
            x_regime = x_tail[:, -self.regime_feat_dim:]
            x_tail = x_tail[:, :-self.regime_feat_dim]
        if self.extra_encoder and self.extra_feat_dim > 0:
            x_extra = x_tail[:, :self.extra_feat_dim]

        phy_feat  = self.physics_encoder(self._physics_features(x_dyn))
        veg_emb   = self.veg_embedding(torch.clamp(x_veg, 0, 20))
        stat_feat = self.static_encoder(torch.cat([x_stat, veg_emb], dim=1))

        imp_vars = x_dyn[:, :, self.temporal_var_indices]
        t_in     = self.temporal_input_proj(imp_vars)
        t_out, _ = self.temporal_stream(t_in)

        t_out_se = t_out.permute(0, 2, 1)
        t_out_se = self.se_block(t_out_se)
        t_out_se = t_out_se.permute(0, 2, 1)
        t_feat   = self.temporal_norm(t_out_se.mean(dim=1))

        parts = [t_feat, stat_feat, phy_feat]
        if x_extra is not None and self.extra_encoder:
            parts.append(self.extra_encoder(x_extra))
        if x_regime is not None and self.regime_encoder:
            parts.append(self.regime_encoder(x_regime))

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
    """
    def __init__(self, X_path, y_cls, y_raw, meta_path=None, scaler=None,
                 window_size=12, use_fe=True, indices=None, is_training=False,
                 aug_cfg=None, clear_mist_manifest_path=None):
        self.X_path  = X_path
        self.orig_indices = np.asarray(indices) if indices is not None else np.arange(len(y_cls))
        self.window_size = window_size
        self.is_training = is_training
        self.aug_cfg = aug_cfg or {}
        self.clear_mist_manifest_path = clear_mist_manifest_path

        self.y_cls   = torch.as_tensor(y_cls[self.orig_indices], dtype=torch.long)

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

        self.X = None
        self.meta = self._load_meta(meta_path, len(y_cls))
        self.month = self.meta['month'][self.orig_indices]
        self.hour = self.meta['hour'][self.orig_indices]
        self.station_code = self.meta['station_code'][self.orig_indices]
        self.is_coastal = self.meta['is_coastal'][self.orig_indices]
        self.hard_flags = self._build_hard_flags()
        self.hard_pool_by_class = {
            0: np.where(self.hard_flags['fog_pool'])[0],
            1: np.where(self.hard_flags['mist_pool'])[0],
            2: np.where(self.hard_flags['clear_pool'])[0],
        }

    @staticmethod
    def _load_meta(meta_path, n_rows):
        if meta_path is None or not os.path.exists(meta_path):
            station_code = np.arange(n_rows, dtype=np.int64)
            return {
                'month': np.zeros(n_rows, dtype=np.int64),
                'hour': np.zeros(n_rows, dtype=np.int64),
                'station_code': station_code,
                'lat': np.zeros(n_rows, dtype=np.float32),
                'lon': np.zeros(n_rows, dtype=np.float32),
                'is_coastal': np.zeros(n_rows, dtype=np.int64),
            }
        meta_df = pd.read_csv(meta_path, parse_dates=['time'])
        if len(meta_df) != n_rows:
            raise ValueError(f"Meta length mismatch: {meta_path} has {len(meta_df)} rows, expected {n_rows}.")
        station_code, _ = pd.factorize(meta_df['station_id'].astype(str), sort=True)
        lat = meta_df['lat'].values.astype(np.float32) if 'lat' in meta_df.columns else np.zeros(n_rows, dtype=np.float32)
        lon = meta_df['lon'].values.astype(np.float32) if 'lon' in meta_df.columns else np.zeros(n_rows, dtype=np.float32)
        return {
            'month': meta_df['time'].dt.month.values.astype(np.int64),
            'hour': meta_df['time'].dt.hour.values.astype(np.int64),
            'station_code': station_code.astype(np.int64),
            'lat': lat,
            'lon': lon,
            'is_coastal': is_coastal_location(lat, lon).astype(np.int64),
        }

    def _load_manifest_clear_pool(self, y_cls_np):
        path = self.clear_mist_manifest_path
        if not path or not os.path.exists(path):
            return None
        try:
            manifest_df = pd.read_csv(path)
        except Exception:
            return None
        required = {'row_index', 'error_type'}
        if not required.issubset(set(manifest_df.columns)):
            return None
        mask = manifest_df['error_type'].astype(str) == 'clear_to_mist'
        if 'p_mist' in manifest_df.columns:
            mask &= manifest_df['p_mist'].astype(np.float32) >= float(self.aug_cfg.get('manifest_min_pmist', 0.35))
        row_indices = manifest_df.loc[mask, 'row_index'].dropna().astype(np.int64).values
        if row_indices.size == 0:
            return None
        _, local_idx, _ = np.intersect1d(self.orig_indices, row_indices, return_indices=True)
        min_rows = int(self.aug_cfg.get('manifest_min_rows', 128))
        if local_idx.size < min_rows:
            return None
        clear_pool = np.zeros(len(self.orig_indices), dtype=bool)
        clear_pool[local_idx] = True
        clear_pool &= (y_cls_np == 2)
        return clear_pool

    def _build_hard_flags(self):
        rows = np.load(self.X_path, mmap_mode='r')
        batch = 65536
        mixing_parts = []
        conv_parts = []
        score_parts = []
        zenith_parts = []
        for start in range(0, len(self.orig_indices), batch):
            subset = self.orig_indices[start:start + batch]
            feats = extract_last_step_features(rows[subset], self.window_size)
            mixing_parts.append(feats['mixing'])
            conv_parts.append(feats['convective'])
            score_parts.append(feats['clear_low_vis_score'])
            zenith_parts.append(feats['zenith'])

        mixing = np.concatenate(mixing_parts)
        convective = np.concatenate(conv_parts)
        clear_score = np.concatenate(score_parts)
        zenith = np.concatenate(zenith_parts)

        y_cls_np = self.y_cls.numpy()
        y_raw_np = self.y_raw.numpy()
        month = self.month
        station_code = self.station_code
        is_jja = np.isin(month, [6, 7, 8])
        boundary_fm = (y_raw_np >= 400.0) & (y_raw_np <= 650.0)
        boundary_mc = (y_raw_np >= 800.0) & (y_raw_np <= 1200.0)
        clear_threshold = np.quantile(clear_score[y_cls_np == 2], 0.88) if np.any(y_cls_np == 2) else 1.0
        clear_pool = (y_cls_np == 2) & (clear_score >= clear_threshold)
        manifest_clear_pool = self._load_manifest_clear_pool(y_cls_np)
        if manifest_clear_pool is not None:
            clear_pool = manifest_clear_pool

        fog_counts = pd.Series((y_cls_np == 0).astype(np.int64)).groupby(station_code).sum()
        sparse_station_codes = fog_counts[fog_counts <= max(2, int(np.quantile(fog_counts.values, 0.25)))].index.values
        sparse_fog_station = np.isin(station_code, sparse_station_codes)

        return {
            'is_jja': is_jja,
            'is_boundary': boundary_fm | boundary_mc,
            'clear_hard_neg': clear_pool,
            'fog_pool': ((y_cls_np == 0) & (boundary_fm | sparse_fog_station)),
            'mist_pool': ((y_cls_np == 1) & (is_jja | boundary_fm | boundary_mc)),
            'clear_pool': clear_pool,
            'is_day': zenith <= 90.0,
            'mixing': mixing,
            'convective': convective,
        }

    def _augment_row(self, row, target_cls):
        if not self.is_training or target_cls > 1:
            return row
        out = row.copy()
        dyn = out[:self.split_dyn].reshape(self.window_size, 25)
        rng = np.random.default_rng()

        if rng.random() < 0.35:
            shift = int(rng.choice([-1, 1]))
            dyn = np.roll(dyn, shift, axis=0)
            if shift > 0:
                dyn[0] = dyn[1]
            else:
                dyn[-1] = dyn[-2]

        if rng.random() < 0.50:
            rh_delta = rng.uniform(-self.aug_cfg.get('rh_delta', 2.0), self.aug_cfg.get('rh_delta', 2.0))
            dyn[:, 0] = np.clip(dyn[:, 0] + rh_delta, 0.0, 100.0)
            dyn[:, 22] = np.clip(dyn[:, 22] - rh_delta * 0.03, -10.0, 40.0)

        if rng.random() < 0.50:
            temp_delta = rng.uniform(-self.aug_cfg.get('temp_delta', 0.6), self.aug_cfg.get('temp_delta', 0.6))
            dyn[:, 1] = dyn[:, 1] + temp_delta
            dyn[:, 11] = dyn[:, 11] + temp_delta * 0.5
            dyn[:, 23] = dyn[:, 11] - dyn[:, 1]

        if rng.random() < 0.50:
            wind_scale = 1.0 + rng.uniform(-self.aug_cfg.get('wind_scale', 0.08), self.aug_cfg.get('wind_scale', 0.08))
            dyn[:, 5] = dyn[:, 5] * wind_scale
            dyn[:, 6] = np.clip(dyn[:, 6] * wind_scale, 0.0, None)
            dyn[:, 7] = dyn[:, 7] * wind_scale

        out[:self.split_dyn] = dyn.reshape(-1)
        if self.use_fe:
            extra = out[self.split_dyn + 6:].astype(np.float32, copy=True)
            if extra.size > 0 and rng.random() < 0.30:
                drop_mask = rng.random(extra.shape[0]) < self.aug_cfg.get('extra_dropout', 0.05)
                extra[drop_mask] = 0.0
            out[self.split_dyn + 6:] = extra
        return out

    def __len__(self):
        return len(self.orig_indices)

    def __getitem__(self, idx):
        if self.X is None:
            self.X = np.load(self.X_path, mmap_mode='r')

        real_idx = self.orig_indices[idx]
        row      = self._augment_row(self.X[real_idx], int(self.y_cls[idx].item()))

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
        season_name = month_to_season_name(self.month[idx])
        regime_feat = np.array([
            1.0 if season_name == 'DJF' else 0.0,
            1.0 if season_name == 'MAM' else 0.0,
            1.0 if season_name == 'JJA' else 0.0,
            1.0 if season_name == 'SON' else 0.0,
            float(self.hard_flags['is_day'][idx]),
            float(self.is_coastal[idx]),
        ], dtype=np.float32)
        final = np.concatenate([final, regime_feat])

        final = np.nan_to_num(final, nan=0.0)
        ctx = np.array([
            float(self.hard_flags['is_jja'][idx]),
            float(self.hard_flags['is_boundary'][idx]),
            float(self.hard_flags['clear_hard_neg'][idx]),
            float(self.month[idx]),
            float(self.hard_flags['is_day'][idx]),
            float(self.is_coastal[idx]),
        ], dtype=np.float32)
        return torch.from_numpy(final).float(), self.y_cls[idx], self.y_reg[idx], self.y_raw[idx], torch.from_numpy(ctx)


def load_data(data_dir, scaler=None, rank=0, local_rank=0, device=None,
              reuse_scaler=False, win_size=12, world_size=1, exp_id=None):
    path_X = copy_to_local(os.path.join(data_dir, 'X_train.npy'), rank, local_rank, world_size, exp_id)
    path_y = copy_to_local(os.path.join(data_dir, 'y_train.npy'), rank, local_rank, world_size, exp_id)
    meta_train_path = os.path.join(data_dir, 'meta_train.csv')
    path_X_val = copy_to_local(os.path.join(data_dir, 'X_val.npy'), rank, local_rank, world_size, exp_id)
    path_y_val = copy_to_local(os.path.join(data_dir, 'y_val.npy'), rank, local_rank, world_size, exp_id)
    meta_val_path = os.path.join(data_dir, 'meta_val.csv')

    y_raw = np.load(path_y)
    y_raw_val = np.load(path_y_val)
    if np.max(y_raw) < 100:
        y_raw = y_raw * 1000
    if np.max(y_raw_val) < 100:
        y_raw_val = y_raw_val * 1000
    y_cls = np.zeros(len(y_raw), dtype=np.int64)
    y_cls_val = np.zeros(len(y_raw_val), dtype=np.int64)
    y_cls[y_raw >= 500] = 1
    y_cls[y_raw >= 1000] = 2
    y_cls_val[y_raw_val >= 500] = 1
    y_cls_val[y_raw_val >= 1000] = 2

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

    manifest_path = CONFIG.get('CLEAR_MIST_MANIFEST_PATH')
    if manifest_path is None and exp_id:
        candidate_paths = [
            os.path.join(CONFIG['SAVE_CKPT_DIR'], f'{exp_id}_clear_to_mist_manifest_train.csv'),
            os.path.join(CONFIG['SAVE_CKPT_DIR'], f'{exp_id}_hard_example_manifest_train.csv'),
            os.path.join(CONFIG['SAVE_CKPT_DIR'], f'{exp_id}_hard_example_manifest_val.csv'),
        ]
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                manifest_path = candidate
                break
    if rank == 0 and manifest_path:
        print(f"[HardNeg] Using clear->mist manifest: {manifest_path}", flush=True)

    tr_ds = PMSTDataset(
        path_X, y_cls, y_raw, meta_train_path, scaler, win_size, True,
        indices=None, is_training=True,
        aug_cfg={
            'temp_delta': CONFIG['S2_AUG_TEMP_MAX_DELTA'],
            'rh_delta': CONFIG['S2_AUG_RH_MAX_DELTA'],
            'wind_scale': CONFIG['S2_AUG_WIND_SCALE'],
            'extra_dropout': CONFIG['S2_AUG_EXTRA_DROPOUT'],
            'manifest_min_rows': CONFIG['CLEAR_MIST_MANIFEST_MIN_ROWS'],
            'manifest_min_pmist': CONFIG['CLEAR_MIST_MANIFEST_MIN_PMIST'],
        },
        clear_mist_manifest_path=manifest_path,
    )
    val_ds = PMSTDataset(
        path_X_val, y_cls_val, y_raw_val, meta_val_path, scaler, win_size, True,
        indices=None, is_training=False,
    )

    return tr_ds, val_ds, scaler

# ==========================================
# 6. 评估
# ==========================================

def compute_target_achievement(metrics: dict, cfg: dict) -> float:
    ta = (
        min(metrics['recall_500']          / cfg['TARGET_RECALL_500_GOAL'],    1.0) * cfg['TARGET_W_RECALL_500']  +
        min(metrics['recall_1000']         / cfg['TARGET_RECALL_1000_GOAL'],   1.0) * cfg['TARGET_W_RECALL_1000'] +
        min(metrics['Fog_P']               / cfg['TARGET_FOG_PREC_GOAL'],      1.0) * cfg['TARGET_W_FOG_PREC']    +
        min(metrics['Mist_P']              / cfg['TARGET_MIST_PREC_GOAL'],     1.0) * cfg['TARGET_W_MIST_PREC']   +
        min(metrics['accuracy']            / cfg['TARGET_ACCURACY_GOAL'],      1.0) * cfg['TARGET_W_ACCURACY']    +
        min(metrics['low_vis_precision']   / cfg['TARGET_LOW_VIS_PREC_GOAL'],  1.0) * cfg['TARGET_W_LOW_VIS_PREC']+
        min((1 - metrics['false_positive_rate']) / (1.0 - cfg['TARGET_FPR_GOAL']), 1.0) * cfg['TARGET_W_FPR']
    )
    return float(ta)


def candidate_rank_tuple(stats: dict):
    """
    Precision floor is enforced before ranking. Among valid candidates, prefer:
    1) higher mist recall
    2) higher mist CSI
    3) higher mist precision
    4) higher overall low-visibility precision
    5) lower false positive rate
    6) stronger fog recall / CSI as tie-breakers
    7) higher target achievement as final tie-breaker
    """
    return (
        float(stats.get('Mist_R', 0.0)),
        float(stats.get('Mist_CSI', 0.0)),
        float(stats.get('Mist_P', 0.0)),
        float(stats.get('low_vis_precision', 0.0)),
        -float(stats.get('false_positive_rate', 1.0)),
        float(stats.get('Fog_R', 0.0)),
        float(stats.get('Fog_CSI', 0.0)),
        float(stats.get('score', 0.0)),
    )


def update_best_candidate(best_entry, stats, fog_th, mist_th):
    candidate = {
        'stats': stats,
        'fog_th': float(fog_th),
        'mist_th': float(mist_th),
        'rank': candidate_rank_tuple(stats),
    }
    if best_entry is None or candidate['rank'] > best_entry['rank']:
        return candidate
    return best_entry


class ComprehensiveMetrics:
    """
    Evaluator with a three-tier threshold search strategy.

    Tier 1 — Strict:
        Search the *extended* grid (0.10 → 0.95) for thresholds that satisfy
        both Fog_P≥0.10, Mist_P≥0.10 AND Clear_R≥0.90.
        Rank survivors by target_achievement (unchanged metric).

    Tier 2 — Relaxed:
        If Tier 1 yields nothing (expected early in training before calibration
        improves), relax to Fog_P≥0.05, Mist_P≥0.05, Clear_R≥0.88.
        The score is penalised proportionally to the precision shortfall so the
        checkpoint system still gravitates toward higher-precision solutions.

    Tier 3 — Argmax fallback:
        Used only when both tiers fail. Identical to original behaviour.

    Why extending the search range matters:
        The original ceiling of 0.65 is too low for a miscalibrated model that
        outputs p(fog+mist)>0.65 for many clear samples.  Valid thresholds often
        exist in the 0.70–0.90 range once the clear-margin loss takes effect.
    """

    def __init__(self, config):
        self.cfg = config
        self.best_th = {'fog': 0.5, 'mist': 0.5}

        # Tier 1 hard constraints
        self.min_fog_prec_strict = config.get('SEARCH_FOG_PRECISION_STRICT', 0.10)
        self.min_mist_prec_strict = config.get('SEARCH_MIST_PRECISION_STRICT', 0.10)
        self.min_clear_recall_strict = config.get('SEARCH_CLEAR_RECALL_STRICT', 0.90)

        # Tier 2 relaxed constraints
        self.min_fog_prec_relaxed = config.get('SEARCH_FOG_PRECISION_RELAXED', 0.08)
        self.min_mist_prec_relaxed = config.get('SEARCH_MIST_PRECISION_RELAXED', 0.08)
        self.min_clear_recall_relaxed = config.get('SEARCH_CLEAR_RECALL_RELAXED', 0.88)

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
        fog_confident = (probs[:, 0] > f_th) & (probs[:, 0] > probs[:, 1])
        mist_confident = (probs[:, 1] > m_th) & (probs[:, 1] > probs[:, 0])
        preds[fog_confident] = 0
        preds[mist_confident] = 1

        p0, r0 = self._calc_metrics_per_class(targets, preds, 0)
        p1, r1 = self._calc_metrics_per_class(targets, preds, 1)
        p2, r2 = self._calc_metrics_per_class(targets, preds, 2)
        tp0 = ((preds == 0) & (targets == 0)).sum()
        fp0 = ((preds == 0) & (targets != 0)).sum()
        fn0 = ((preds != 0) & (targets == 0)).sum()
        tp1 = ((preds == 1) & (targets == 1)).sum()
        fp1 = ((preds == 1) & (targets != 1)).sum()
        fn1 = ((preds != 1) & (targets == 1)).sum()

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
            'Fog_CSI':             float(tp0 / (tp0 + fp0 + fn0 + 1e-6)),
            'Mist_CSI':            float(tp1 / (tp1 + fp1 + fn1 + 1e-6)),
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
            for bx, by, _, _, _ in loader:
                bx = bx.to(device, non_blocking=True)
                fine, _, _ = model(bx)
                probs_l.append(F.softmax(fine, dim=1))
                targets_l.append(by.to(device))

        local_probs   = torch.cat(probs_l,   dim=0)
        local_targets = torch.cat(targets_l, dim=0)

        # ── DDP all-gather (unchanged) ────────────────────────────────────────
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
        best_entry = None
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
                    if (
                        stats['Fog_P']   >= self.min_fog_prec_strict and
                        stats['Mist_P']  >= self.min_mist_prec_strict and
                        stats['Clear_R'] >= self.min_clear_recall_strict
                    ):
                        stats['score'] = compute_target_achievement(stats, self.cfg)
                        best_entry = update_best_candidate(best_entry, stats, f_th, m_th)
                        tier_used = 1

            # ── Tier 2: relaxed constraints (no argmax jump yet) ──────────────
            # Allows training-time checkpointing to still track precision progress
            # when strict constraints aren't met. The precision shortfall is
            # subtracted from the score so the system still prefers higher precision.
            if best_entry is None:
                for f_th in search_space:
                    for m_th in search_space:
                        stats = self._build_full_metrics(probs, targets, f_th, m_th)
                        if (
                            stats['Fog_P']   >= self.min_fog_prec_relaxed and
                            stats['Mist_P']  >= self.min_mist_prec_relaxed and
                            stats['Clear_R'] >= self.min_clear_recall_relaxed
                        ):
                            stats['score'] = compute_target_achievement(stats, self.cfg)
                            best_entry = update_best_candidate(best_entry, stats, f_th, m_th)
                            tier_used = 2

            if best_entry is not None:
                best_stats = best_entry['stats']
                best_ta = best_stats['score']
                self.best_th = {'fog': best_entry['fog_th'], 'mist': best_entry['mist_th']}

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
                    'Fog_CSI':             float(((preds == 0) & (targets == 0)).sum() / ((((preds == 0) & (targets == 0)).sum()) + (((preds == 0) & (targets != 0)).sum()) + (((preds != 0) & (targets == 0)).sum()) + 1e-6)),
                    'Mist_CSI':            float(((preds == 1) & (targets == 1)).sum() / ((((preds == 1) & (targets == 1)).sum()) + (((preds == 1) & (targets != 1)).sum()) + (((preds != 1) & (targets == 1)).sum()) + 1e-6)),
                    'preds':               preds,
                }
                best_ta = compute_target_achievement(best_stats, self.cfg)
                best_stats['score'] = best_ta
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

            best_stats['Fog_PR_AUC'] = safe_average_precision((targets == 0).astype(np.int64), probs[:, 0])
            best_stats['Mist_PR_AUC'] = safe_average_precision((targets == 1).astype(np.int64), probs[:, 1])
            print(
                f"  [Eval] Fog  R={best_stats['Fog_R']:.3f} P={best_stats['Fog_P']:.3f} | "
                f"Mist R={best_stats['Mist_R']:.3f} P={best_stats['Mist_P']:.3f} | "
                f"FogCSI={best_stats['Fog_CSI']:.3f} MistCSI={best_stats['Mist_CSI']:.3f} | "
                f"FogAP={best_stats['Fog_PR_AUC']:.3f} MistAP={best_stats['Mist_PR_AUC']:.3f} | "
                f"Clear R={best_stats['Clear_R']:.3f} | "
                f"Acc={best_stats['accuracy']:.3f} | "
                f"LVPrec={best_stats['low_vis_precision']:.3f} | "
                f"FPR={best_stats['false_positive_rate']:.3f}",
                flush=True
            )
            print(f"  [Eval] target_achievement = {best_ta:.4f}", flush=True)

        # ── Broadcast best_ta to all ranks (unchanged) ───────────────────────
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
# 6b. Exp7: 逐季节阈值优化
# ==========================================

def evaluate_per_season_from_arrays(probs, targets, months, config, global_thresholds=None):
    """
    Exp7: Run per-season threshold optimization on calibrated probabilities.
    Seasonal thresholds are anchored to the global joint thresholds and are only
    accepted when they keep mist and low-visibility precision near baseline.
    """
    season_map = {
        'DJF': [12, 1, 2], 'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],  'SON': [9, 10, 11],
    }

    evaluator = ComprehensiveMetrics(config)
    search_grid = evaluator._build_search_grid()
    season_thresholds = {}
    if global_thresholds is None:
        global_thresholds = search_global_thresholds(probs, targets, use_joint=True)

    global_fog_th = float(global_thresholds['fog_th'])
    global_mist_th = float(global_thresholds['mist_th'])
    fog_lower_bound = max(0.10, global_fog_th - float(config.get('SEASON_FOG_MAX_DROP_FROM_GLOBAL', 0.12)))
    mist_lower_bound = max(
        float(config.get('SEASON_MIST_MIN_THRESHOLD', 0.25)),
        global_mist_th - float(config.get('SEASON_MIST_MAX_DROP_FROM_GLOBAL', 0.08)),
    )

    for s_name, s_months in season_map.items():
        s_mask = np.isin(months, s_months)
        if s_mask.sum() < 50:
            print(f"  [SeasonEval] {s_name}: too few samples ({s_mask.sum()}), skipping", flush=True)
            continue

        s_probs = probs[s_mask]
        s_targets = targets[s_mask]
        base_preds = apply_joint_threshold_rule(s_probs, global_fog_th, global_mist_th)
        baseline_stats = compute_prediction_metrics(s_probs, s_targets, base_preds)
        best_stats = baseline_stats
        best_fth = global_fog_th
        best_mth = global_mist_th
        best_rank = postprocess_rank_tuple(best_stats)
        source = "global_fallback"

        for relaxed in (False, True):
            found_candidate = False
            for f_th in search_grid:
                if f_th < fog_lower_bound:
                    continue
                for m_th in search_grid:
                    if m_th < mist_lower_bound:
                        continue
                    preds = apply_joint_threshold_rule(s_probs, f_th, m_th)
                    stats = compute_prediction_metrics(s_probs, s_targets, preds)
                    if not is_valid_postprocess_candidate(stats, config, baseline_stats=baseline_stats, relaxed=relaxed):
                        continue
                    rank = postprocess_rank_tuple(stats)
                    if rank > best_rank:
                        best_rank = rank
                        best_stats = stats
                        best_fth = float(f_th)
                        best_mth = float(m_th)
                        source = "seasonal_relaxed" if relaxed else "seasonal_strict"
                    found_candidate = True
            if found_candidate:
                break

        if best_stats is not None:
            best_ta = best_stats['score']
            season_thresholds[s_name] = {
                'fog_th': best_fth, 'mist_th': best_mth,
                'Fog_R': best_stats['Fog_R'], 'Fog_P': best_stats['Fog_P'],
                'Mist_R': best_stats['Mist_R'], 'Mist_P': best_stats['Mist_P'],
                'Fog_CSI': best_stats['Fog_CSI'], 'Mist_CSI': best_stats['Mist_CSI'],
                'score': best_ta,
                'source': source,
            }
            print(
                f"  [SeasonEval] {s_name}: Fog_th={best_fth:.2f} Mist_th={best_mth:.2f} | "
                f"Fog R={best_stats['Fog_R']:.3f} P={best_stats['Fog_P']:.3f} | "
                f"Mist R={best_stats['Mist_R']:.3f} P={best_stats['Mist_P']:.3f} | "
                f"LVPrec={best_stats['low_vis_precision']:.3f} | "
                f"Source={source} | Score={best_ta:.4f}",
                flush=True
            )
        else:
            print(f"  [SeasonEval] {s_name}: no valid thresholds found", flush=True)

    return season_thresholds


# ==========================================
# 6c. Exp8: 温度缩放校准
# ==========================================

def calibrate_temperature(model, loader, device, config, rank=0):
    """
    Exp8: Post-hoc temperature scaling.
    Learns a single scalar T on validation set to calibrate softmax outputs.
    """
    if rank != 0:
        return 1.0

    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for bx, by, _, _, _ in loader:
            bx = bx.to(device, non_blocking=True)
            fine, _, _ = model(bx)
            all_logits.append(fine)
            all_targets.append(by.to(device))

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)
    optimizer_t = optim.LBFGS([temperature],
                              lr=config.get('TEMP_SCALING_LR', 0.01),
                              max_iter=config.get('TEMP_SCALING_MAX_ITER', 50))

    def closure():
        optimizer_t.zero_grad()
        scaled = logits / temperature
        loss = F.cross_entropy(scaled, targets)
        loss.backward()
        return loss

    optimizer_t.step(closure)
    t_val = temperature.item()
    print(f"  [TempScaling] Optimal temperature: {t_val:.4f}", flush=True)
    return t_val


def apply_joint_threshold_rule(probs, fog_th, mist_th):
    probs = np.asarray(probs, dtype=np.float64)
    n = probs.shape[0]
    fog_th = np.asarray(fog_th, dtype=np.float64)
    mist_th = np.asarray(mist_th, dtype=np.float64)
    if fog_th.ndim == 0:
        fog_th = np.full(n, float(fog_th))
    if mist_th.ndim == 0:
        mist_th = np.full(n, float(mist_th))

    fog_excess = (probs[:, 0] - fog_th) / np.maximum(1.0 - fog_th, 1e-6)
    mist_excess = (probs[:, 1] - mist_th) / np.maximum(1.0 - mist_th, 1e-6)
    pred = np.full(n, 2, dtype=np.int64)
    fog_active = fog_excess >= 0.0
    mist_active = mist_excess >= 0.0
    pred[fog_active & ~mist_active] = 0
    pred[mist_active & ~fog_active] = 1
    both = fog_active & mist_active
    pred[both] = np.where(fog_excess[both] >= mist_excess[both], 0, 1)
    return pred


def apply_override_threshold_rule(probs, fog_th, mist_th):
    probs = np.asarray(probs, dtype=np.float64)
    fog_th = np.asarray(fog_th, dtype=np.float64)
    mist_th = np.asarray(mist_th, dtype=np.float64)
    if fog_th.ndim == 0:
        fog_th = np.full(probs.shape[0], float(fog_th))
    if mist_th.ndim == 0:
        mist_th = np.full(probs.shape[0], float(mist_th))
    pred = np.full(probs.shape[0], 2, dtype=np.int64)
    pred[probs[:, 1] >= mist_th] = 1
    pred[probs[:, 0] >= fog_th] = 0
    return pred


def thresholds_from_season_table(months, season_thresholds, default_fog=0.46, default_mist=0.38):
    months = np.asarray(months, dtype=np.int64)
    fog_th = np.full(len(months), float(default_fog), dtype=np.float64)
    mist_th = np.full(len(months), float(default_mist), dtype=np.float64)
    for i, month in enumerate(months):
        season = month_to_season_name(month)
        if season in season_thresholds:
            fog_th[i] = float(season_thresholds[season]['fog_th'])
            mist_th[i] = float(season_thresholds[season]['mist_th'])
    return fog_th, mist_th


def compute_prediction_metrics(probs, targets, preds):
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    def _class_stats(class_id):
        tp = ((preds == class_id) & (targets == class_id)).sum()
        fp = ((preds == class_id) & (targets != class_id)).sum()
        fn = ((preds != class_id) & (targets == class_id)).sum()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        csi = tp / (tp + fp + fn + 1e-6)
        return precision, recall, csi

    p0, r0, c0 = _class_stats(0)
    p1, r1, c1 = _class_stats(1)
    p2, r2, _ = _class_stats(2)
    pred_low = preds <= 1
    true_low = targets <= 1
    clear_true = targets == 2
    lv_tp = (pred_low & true_low).sum()
    lv_fp = (pred_low & ~true_low).sum()
    lv_prec = lv_tp / (lv_tp + lv_fp + 1e-6)
    fpr = (pred_low & clear_true).sum() / (clear_true.sum() + 1e-6)
    stats = {
        'Fog_P': float(p0), 'Fog_R': float(r0), 'Fog_CSI': float(c0),
        'Mist_P': float(p1), 'Mist_R': float(r1), 'Mist_CSI': float(c1),
        'Clear_P': float(p2), 'Clear_R': float(r2),
        'Fog_PR_AUC': safe_average_precision((targets == 0).astype(np.int64), probs[:, 0]),
        'Mist_PR_AUC': safe_average_precision((targets == 1).astype(np.int64), probs[:, 1]),
        'accuracy': float((preds == targets).mean()),
        'low_vis_precision': float(lv_prec),
        'false_positive_rate': float(fpr),
        'recall_500': float(r0),
        'recall_1000': float(r1),
    }
    stats['score'] = compute_target_achievement(stats, CONFIG) + 0.05 * (stats['Fog_CSI'] + stats['Mist_CSI'])
    return stats


def build_postprocess_guardrails(config, baseline_stats=None, relaxed=False):
    mist_floor = float(config.get('POSTPROC_MIN_MIST_PRECISION', config.get('MIN_MIST_PRECISION', 0.15)))
    lv_floor = float(config.get('POSTPROC_MIN_LOW_VIS_PRECISION', config.get('TARGET_LOW_VIS_PREC_GOAL', 0.20)))
    clear_floor = 0.0

    if relaxed:
        mist_floor = max(0.0, mist_floor - float(config.get('POSTPROC_RELAX_MIST_PRECISION', 0.02)))
        lv_floor = max(0.0, lv_floor - float(config.get('POSTPROC_RELAX_LOW_VIS_PRECISION', 0.02)))

    if baseline_stats is not None:
        mist_floor = max(
            mist_floor,
            float(baseline_stats.get('Mist_P', 0.0)) - float(config.get('POSTPROC_MAX_MIST_PREC_DROP', 0.02)),
        )
        lv_floor = max(
            lv_floor,
            float(baseline_stats.get('low_vis_precision', 0.0)) - float(config.get('POSTPROC_MAX_LV_PREC_DROP', 0.02)),
        )
        clear_floor = max(
            clear_floor,
            float(baseline_stats.get('Clear_R', 0.0)) - float(config.get('POSTPROC_MAX_CLEAR_RECALL_DROP', 0.01)),
        )
        if relaxed:
            clear_floor = max(0.0, clear_floor - float(config.get('POSTPROC_RELAX_CLEAR_RECALL', 0.02)))

    return {
        'mist_floor': float(mist_floor),
        'lv_floor': float(lv_floor),
        'clear_floor': float(clear_floor),
    }


def is_valid_postprocess_candidate(stats, config, baseline_stats=None, relaxed=False):
    guards = build_postprocess_guardrails(config, baseline_stats=baseline_stats, relaxed=relaxed)
    return (
        float(stats.get('Mist_P', 0.0)) >= guards['mist_floor'] and
        float(stats.get('low_vis_precision', 0.0)) >= guards['lv_floor'] and
        float(stats.get('Clear_R', 0.0)) >= guards['clear_floor']
    )


def postprocess_rank_tuple(stats):
    return (
        float(stats.get('Mist_P', 0.0)),
        float(stats.get('low_vis_precision', 0.0)),
        float(stats.get('Mist_CSI', 0.0)),
        float(stats.get('Mist_R', 0.0)),
        -float(stats.get('false_positive_rate', 1.0)),
        float(stats.get('Fog_CSI', 0.0)),
        float(stats.get('Fog_R', 0.0)),
        float(stats.get('score', 0.0)),
    )


def collect_validation_outputs(model, dataset, device, batch_size, temperature=1.0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logits_list, targets_list, raw_list, ctx_list = [], [], [], []
    with torch.no_grad():
        for bx, by, _, braw, bctx in loader:
            bx = bx.to(device, non_blocking=True)
            fine, _, _ = model(bx)
            logits_list.append(fine.cpu().numpy())
            targets_list.append(by.numpy())
            raw_list.append(braw.numpy())
            ctx_list.append(bctx.numpy())

    logits = np.concatenate(logits_list, axis=0)
    logits = logits / max(float(temperature), 1e-3)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-6, None)
    targets = np.concatenate(targets_list, axis=0)
    y_raw = np.concatenate(raw_list, axis=0)
    ctx = np.concatenate(ctx_list, axis=0)
    rows = np.load(dataset.X_path, mmap_mode='r')
    regime = extract_last_step_features(rows, dataset.window_size)
    return {
        'probs': probs,
        'targets': targets,
        'y_raw': y_raw,
        'months': np.rint(ctx[:, 3]).astype(np.int64),
        'is_day': regime['zenith'] <= 90.0,
        'regime': regime,
    }


def search_global_thresholds(probs, targets, use_joint=False):
    evaluator = ComprehensiveMetrics(CONFIG)
    best = {'fog_th': 0.5, 'mist_th': 0.5, 'stats': None}
    best_score = -1e9
    for fog_th in evaluator._build_search_grid():
        for mist_th in evaluator._build_search_grid():
            preds = apply_joint_threshold_rule(probs, fog_th, mist_th) if use_joint else apply_override_threshold_rule(probs, fog_th, mist_th)
            stats = compute_prediction_metrics(probs, targets, preds)
            if stats['score'] > best_score:
                best_score = stats['score']
                best = {'fog_th': float(fog_th), 'mist_th': float(mist_th), 'stats': stats}
    return best


def fit_daynight_offsets(probs, targets, months, is_day, season_thresholds):
    grid = CONFIG['POSTPROC_OFFSET_GRID']
    min_group = CONFIG['POSTPROC_MIN_GROUP_SIZE']
    offsets = {}
    base_fog, base_mist = thresholds_from_season_table(months, season_thresholds)
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        for day_flag, day_label in [(True, 'day'), (False, 'night')]:
            mask = np.array([(month_to_season_name(m) == season) and (bool(d) == day_flag) for m, d in zip(months, is_day)])
            if mask.sum() < min_group:
                continue
            base_preds = apply_joint_threshold_rule(probs[mask], base_fog[mask], base_mist[mask])
            baseline_stats = compute_prediction_metrics(probs[mask], targets[mask], base_preds)
            best_stats = baseline_stats
            best_pair = (0.0, 0.0)
            best_rank = postprocess_rank_tuple(best_stats)
            for fog_off in grid:
                for mist_off in grid:
                    fog_th = np.clip(base_fog[mask] + fog_off, 0.10, 0.95)
                    mist_th = np.clip(base_mist[mask] + mist_off, 0.10, 0.95)
                    preds = apply_joint_threshold_rule(probs[mask], fog_th, mist_th)
                    stats = compute_prediction_metrics(probs[mask], targets[mask], preds)
                    if not is_valid_postprocess_candidate(stats, CONFIG, baseline_stats=baseline_stats, relaxed=True):
                        continue
                    rank = postprocess_rank_tuple(stats)
                    if rank > best_rank:
                        best_stats = stats
                        best_pair = (float(fog_off), float(mist_off))
                        best_rank = rank
            offsets[f'{season}_{day_label}'] = {'fog': best_pair[0], 'mist': best_pair[1]}
    return offsets


def apply_daynight_offsets(fog_th, mist_th, months, is_day, offsets):
    fog_adj = fog_th.copy()
    mist_adj = mist_th.copy()
    for i, (month, day_flag) in enumerate(zip(months, is_day)):
        key = f"{month_to_season_name(month)}_{'day' if day_flag else 'night'}"
        if key in offsets:
            fog_adj[i] = np.clip(fog_adj[i] + offsets[key]['fog'], 0.10, 0.95)
            mist_adj[i] = np.clip(mist_adj[i] + offsets[key]['mist'], 0.10, 0.95)
    return fog_adj, mist_adj


def tune_jja_prior_filters(probs, targets, months, regime, fog_th, mist_th):
    best_cfg = {'mixing_low_vis': 1.0, 'convective_fog': 1.0, 'ventilation_low_vis': 1.0}
    base_preds = apply_joint_threshold_rule(probs, fog_th, mist_th)
    baseline_stats = compute_prediction_metrics(probs, targets, base_preds)
    best_stats = baseline_stats
    best_rank = postprocess_rank_tuple(best_stats)
    for mix_factor in CONFIG['POSTPROC_PRIOR_GRIDS']['mixing_low_vis']:
        for conv_factor in CONFIG['POSTPROC_PRIOR_GRIDS']['convective_fog']:
            for vent_factor in CONFIG['POSTPROC_PRIOR_GRIDS']['ventilation_low_vis']:
                adj = probs.copy()
                jja = np.isin(months, [6, 7, 8])
                adj[jja, 0] *= np.where(regime['convective'][jja] > 0.45, conv_factor, 1.0)
                low_vis_factor = np.ones(jja.sum(), dtype=np.float64)
                low_vis_factor *= np.where(regime['mixing'][jja] > 0.55, mix_factor, 1.0)
                low_vis_factor *= np.where(regime['ventilation'][jja] > 0.55, vent_factor, 1.0)
                adj[jja, 0] *= low_vis_factor
                adj[jja, 1] *= low_vis_factor
                adj = adj / np.clip(adj.sum(axis=1, keepdims=True), 1e-6, None)
                preds = apply_joint_threshold_rule(adj, fog_th, mist_th)
                stats = compute_prediction_metrics(adj, targets, preds)
                if not is_valid_postprocess_candidate(stats, CONFIG, baseline_stats=baseline_stats, relaxed=True):
                    continue
                rank = postprocess_rank_tuple(stats)
                if rank > best_rank:
                    best_stats = stats
                    best_rank = rank
                    best_cfg = {
                        'mixing_low_vis': float(mix_factor),
                        'convective_fog': float(conv_factor),
                        'ventilation_low_vis': float(vent_factor),
                    }
    return best_cfg, best_stats


def export_hard_example_manifest(path, payload, preds):
    regime = payload['regime']
    df = pd.DataFrame({
        'row_index': np.arange(len(payload['targets']), dtype=np.int64),
        'month': payload['months'],
        'is_day': payload['is_day'].astype(np.int64),
        'y_true': payload['targets'],
        'y_raw': payload['y_raw'],
        'pred': preds,
        'p_fog': payload['probs'][:, 0],
        'p_mist': payload['probs'][:, 1],
        'mixing_proxy': regime['mixing'],
        'convective_proxy': regime['convective'],
        'ventilation_proxy': regime['ventilation'],
    })
    df['error_type'] = np.where((df['y_true'] == 2) & (df['pred'] == 1), 'clear_to_mist',
                        np.where((df['y_true'] == 1) & (df['pred'] == 0), 'mist_to_fog',
                        np.where((df['y_true'] == 0) & (df['pred'] == 1), 'fog_to_mist',
                        np.where((df['y_true'] == 1) & (df['pred'] == 2), 'mist_to_clear', 'correct'))))
    df['is_boundary_fm'] = ((df['y_raw'] >= 400.0) & (df['y_raw'] <= 650.0)).astype(np.int64)
    df['is_boundary_mc'] = ((df['y_raw'] >= 800.0) & (df['y_raw'] <= 1200.0)).astype(np.int64)
    df['is_jja_mist'] = ((df['month'].isin([6, 7, 8])) & (df['y_true'] == 1)).astype(np.int64)
    df.to_csv(path, index=False)
    return df


# ==========================================
# 6d. Exp9: 标签平滑辅助函数
# ==========================================

def compute_soft_targets(vis_raw, hard_labels, num_classes=3):
    """
    Exp9: Compute soft labels for boundary transition zone samples.
    Fog/Mist boundary: linear interpolation in [400, 600) meter range.
    Mist/Clear boundary: linear interpolation in [800, 1200) meter range.
    """
    soft = F.one_hot(hard_labels, num_classes).float()
    vis = vis_raw.float()

    fm_mask = (vis >= 400) & (vis < 600)
    if fm_mask.any():
        alpha = (vis[fm_mask] - 400) / 200.0
        soft[fm_mask, 0] = 1 - alpha
        soft[fm_mask, 1] = alpha
        soft[fm_mask, 2] = 0

    mc_mask = (vis >= 800) & (vis < 1200)
    if mc_mask.any():
        alpha = (vis[mc_mask] - 800) / 400.0
        soft[mc_mask, 0] = 0
        soft[mc_mask, 1] = 1 - alpha
        soft[mc_mask, 2] = alpha

    return soft


# ==========================================
# 7. 训练
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
    can detect the missing file without leaving others stuck at the barrier.
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
            print(f"  [Ckpt] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
        if unexpected:
            print(f"  [Ckpt] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}", flush=True)
        print(f"  [Ckpt] Load complete.", flush=True)

    safe_barrier(world_size, device)


def rewrap_ddp(model, world_size):
    """Extract the raw model from a DDP wrapper."""
    if world_size > 1 and hasattr(model, 'module'):
        return model.module
    return model


def wrap_ddp(raw_model, local_rank, world_size, find_unused=False):
    """Wrap raw model in DDP with a barrier first to ensure consistent parameter state."""
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        safe_barrier(world_size, device)
        return DDP(raw_model, device_ids=[local_rank],
                   find_unused_parameters=find_unused)
    return raw_model


def train_stage(tag, model, tr_ds, val_ds, optimizer, loss_fn, device,
                rank, world_size, total_steps, val_int, batch_size,
                grad_accum, fog_ratio, mist_ratio, exp_id,
                patience=10, pretrained_state=None, l2sp_alpha=0.0):
    """
    Train one stage with Early Stopping.
    Exp4: LR warmup via SequentialLR.
    Exp5: L2-SP regularization when pretrained_state is provided.
    Exp9: Label smoothing for boundary samples.
    """
    def worker_init_fn(worker_id):
        # Reset the mmap handle in each worker so they open their own fd
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_info.dataset.X = None

    sampler = StratifiedBalancedBatchSampler(
        tr_ds, batch_size,
        fog_ratio=fog_ratio, mist_ratio=mist_ratio,
        rank=rank, world_size=world_size,
        epoch_length=2000,
        hard_example_ratio=CONFIG.get('S2_HARD_EXAMPLE_RATIO', 0.0),
    )
    num_workers = 0 if world_size > 8 else 2
    if rank == 0 and world_size > 8:
        print(f"[DataLoader] world_size={world_size}>8, using num_workers=0 to avoid NFS/fork contention.", flush=True)
    loader_kw = dict(batch_sampler=sampler, pin_memory=True, worker_init_fn=worker_init_fn)
    if num_workers > 0:
        loader_kw.update(num_workers=num_workers, prefetch_factor=2, persistent_workers=True)
    else:
        loader_kw.update(num_workers=0)
    loader = DataLoader(tr_ds, **loader_kw)

    val_sampler = (DistributedSampler(val_ds, num_replicas=world_size,
                                       rank=rank, shuffle=False)
                   if world_size > 1 else None)
    val_loader_kw = dict(batch_size=batch_size, shuffle=False, sampler=val_sampler,
                         pin_memory=True, worker_init_fn=worker_init_fn)
    if num_workers > 0:
        val_loader_kw.update(num_workers=num_workers, persistent_workers=True)
    else:
        val_loader_kw.update(num_workers=0)
    val_loader = DataLoader(val_ds, **val_loader_kw)
    actual_val_size = len(val_ds)

    metrics_evaluator = ComprehensiveMetrics(CONFIG)
    warmup_steps = CONFIG.get('S2_WARMUP_STEPS', 500)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [
        optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps)),
    ], milestones=[warmup_steps])

    best_score       = -1.0
    best_fog_recall  = -1.0
    best_mist_recall = -1.0
    no_improve_count = 0

    # Paper eval: save training history (loss curves, val metrics) for Figure 10
    ckpt_dir = CONFIG['SAVE_CKPT_DIR']
    history_path = os.path.join(ckpt_dir, f"{exp_id}_{tag}_history.json")
    history = {
        "steps": [], "train_loss": [], "val_score": [],
        "val_fog_recall": [], "val_mist_recall": [],
        "val_fog_precision": [], "val_mist_precision": [],
        "val_clear_recall": [], "val_lv_precision": [], "val_fpr": [], "val_accuracy": [],
    }
    train_loss_accum, train_loss_count = 0.0, 0

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
            bx, by, blog, braw, bctx = next(iterator)
        except StopIteration:
            pseudo_epoch += 1
            sampler.set_epoch(pseudo_epoch)
            iterator = iter(loader)
            bx, by, blog, braw, bctx = next(iterator)

        soft_targets = None
        if CONFIG.get('USE_LABEL_SMOOTHING', False):
            soft_targets = compute_soft_targets(braw, by).to(device)

        bx, by, blog, bctx = bx.to(device), by.to(device), blog.to(device), bctx.to(device)
        batch_count += 1

        fine, reg, bin_out = model(bx)
        l_dual, loss_dict  = loss_fn(fine, bin_out, by, soft_targets=soft_targets, sample_flags=bctx)
        l_reg = F.mse_loss(reg.view(-1), blog)
        loss  = l_dual + CONFIG['REG_LOSS_ALPHA'] * l_reg

        if pretrained_state is not None and l2sp_alpha > 0:
            raw_m = model.module if hasattr(model, 'module') else model
            l2_sp = sum(
                ((p - pretrained_state[n]) ** 2).sum()
                for n, p in raw_m.named_parameters()
                if n in pretrained_state and p.requires_grad
                and p.shape == pretrained_state[n].shape
            )
            loss = loss + l2sp_alpha * l2_sp

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

            if rank == 0:
                train_loss_accum += loss.item() * grad_accum
                train_loss_count += 1

            if rank == 0 and step % 50 == 0:
                print(
                    f"\r[{tag}] Step {step:>6}/{total_steps} | "
                    f"Loss={loss.item() * grad_accum:.4f} | "
                    f"bin={loss_dict['bin']:.4f} "
                    f"fine={loss_dict['fine']:.4f} "
                    f"cm={loss_dict.get('cm', 0.0):.4f} "
                    f"pair={loss_dict.get('pair', 0.0):.4f} "
                    f"cmt={loss_dict.get('cmt', 0.0):.4f} | "
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

                if rank == 0:
                    # Append to and save training history for paper Figure 10
                    avg_loss = train_loss_accum / train_loss_count if train_loss_count > 0 else 0.0
                    train_loss_accum, train_loss_count = 0.0, 0
                    history["steps"].append(step)
                    history["train_loss"].append(round(avg_loss, 6))
                    history["val_score"].append(round(ta, 6))
                    if stats is not None:
                        history["val_fog_recall"].append(round(stats.get("Fog_R", -1.0), 6))
                        history["val_mist_recall"].append(round(stats.get("Mist_R", -1.0), 6))
                        history["val_fog_precision"].append(round(stats.get("Fog_P", -1.0), 6))
                        history["val_mist_precision"].append(round(stats.get("Mist_P", -1.0), 6))
                        history["val_clear_recall"].append(round(stats.get("Clear_R", -1.0), 6))
                        history["val_lv_precision"].append(round(stats.get("low_vis_precision", -1.0), 6))
                        history["val_fpr"].append(round(stats.get("false_positive_rate", -1.0), 6))
                        history["val_accuracy"].append(round(stats.get("accuracy", -1.0), 6))
                    else:
                        for k in ["val_fog_recall", "val_mist_recall", "val_fog_precision", "val_mist_precision",
                                 "val_clear_recall", "val_lv_precision", "val_fpr", "val_accuracy"]:
                            history[k].append(-1.0)
                    try:
                        with open(history_path, "w", encoding="utf-8") as f:
                            json.dump(history, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"  [History] WARN: failed to save {history_path}: {e}", flush=True)

                if rank == 0 and stats is not None:
                    fog_r  = stats.get('Fog_R',  -1.0)
                    mist_r = stats.get('Mist_R', -1.0)

                    if ta > best_score:
                        best_score = ta
                        no_improve_count = 0
                        save_checkpoint(model, path_best_score, rank, world_size)
                        print(f"  [Ckpt] ★ New best target_achievement = {best_score:.4f}", flush=True)
                    else:
                        no_improve_count += 1

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
                        f"FogR={best_fog_recall:.4f} | MistR={best_mist_recall:.4f} | "
                        f"NoImprove={no_improve_count}/{patience}",
                        flush=True
                    )

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
# Main（仅 Stage 2）
# ==========================================
def main():
    l_rank, g_rank, w_size = init_distributed()
    device = torch.device(f"cuda:{l_rank}")
    base_exp_id = CONFIG['EXPERIMENT_ID']
    run_exp_id = build_s2_run_exp_id(base_exp_id, CONFIG.get('S2_RUN_SUFFIX', ''))

    if g_rank == 0:
        os.makedirs(CONFIG['SAVE_CKPT_DIR'], exist_ok=True)
        print(f"[Stage2-Only] Base Exp: {base_exp_id}", flush=True)
        print(f"[Stage2-Only] S2 Run ID: {run_exp_id}", flush=True)
        print(f"World size: {w_size}", flush=True)

    safe_barrier(w_size, device)

    if g_rank == 0:
        print("[Main] Initial barrier passed — all ranks synchronized.", flush=True)

    exp_id = base_exp_id

    # ==========================================
    # 公共构建函数
    # ==========================================
    def build_model_raw():
        m = ImprovedDualStreamPMSTNet(
            window_size=CONFIG['WINDOW_SIZE'],
            hidden_dim=CONFIG['MODEL_HIDDEN_DIM'],
            num_classes=3,
            extra_feat_dim=CONFIG['FE_EXTRA_DIMS'],
            regime_feat_dim=CONFIG['REGIME_FEAT_DIM'],
        ).to(device)
        return m

    # [修改4-fix] build_loss() 现在传入 CONFIG 中所有 alpha 值，修复原来的 bug
    # （原 build_loss() 只传入了 binary_pos_weight/fine_class_weight/asym 参数，
    #   alpha_fp/alpha_fog_boost/alpha_mist_boost 均使用 DualBranchLoss 内部默认值，
    #   导致 CONFIG 中的 LOSS_ALPHA_FP=1.0 和 LOSS_ALPHA_FOG_BOOST=0.5 从未生效。）
    # [修复] build_loss(): S1兼容，传入所有CONFIG alpha值
    def build_loss():
        """S1兼容损失函数。alpha_clear_margin=0.0保证不影响原有行为。"""
        return DualBranchLoss(
            binary_pos_weight=CONFIG['BINARY_POS_WEIGHT'],
            fine_class_weight=[
                CONFIG['FINE_CLASS_WEIGHT_FOG'],
                CONFIG['FINE_CLASS_WEIGHT_MIST'],
                CONFIG['FINE_CLASS_WEIGHT_CLEAR']
            ],
            asym_gamma_neg=CONFIG['ASYM_GAMMA_NEG'],
            asym_clip=CONFIG['ASYM_CLIP'],
            alpha_binary=CONFIG['LOSS_ALPHA_BINARY'],
            alpha_fine=CONFIG['LOSS_ALPHA_FINE'],
            alpha_fp=CONFIG['LOSS_ALPHA_FP'],
            alpha_fog_boost=CONFIG['LOSS_ALPHA_FOG_BOOST'],
            alpha_mist_boost=CONFIG['LOSS_ALPHA_MIST_BOOST'],
            alpha_clear_margin=0.0,   # disabled for S1 compatibility
        ).to(device)

    def build_s2_loss():
        """
        S2专用损失函数。
        Exp2: OrdinalAwareFocalLoss替代AsymmetricLoss。
        """
        return DualBranchLoss(
            binary_pos_weight=CONFIG['S2_BINARY_POS_WEIGHT'],
            fine_class_weight=[
                CONFIG['S2_FINE_CLASS_WEIGHT_FOG'],
                CONFIG['S2_FINE_CLASS_WEIGHT_MIST'],
                CONFIG['S2_FINE_CLASS_WEIGHT_CLEAR']
            ],
            loss_type='ordinal_focal',
            gamma_per_class=[2.5, 3.0, 0.5],
            ordinal_cost=[[0, 1, 3], [1, 0, 2], [3, 2, 0]],
            alpha_binary=CONFIG['S2_LOSS_ALPHA_BINARY'],
            alpha_fine=CONFIG['S2_LOSS_ALPHA_FINE'],
            alpha_fp=CONFIG['S2_LOSS_ALPHA_FP'],
            alpha_fog_boost=CONFIG['S2_LOSS_ALPHA_FOG_BOOST'],
            alpha_mist_boost=CONFIG['S2_LOSS_ALPHA_MIST_BOOST'],
            alpha_clear_margin=CONFIG['S2_LOSS_ALPHA_CLEAR_MARGIN'],
            clear_margin=CONFIG['S2_CLEAR_MARGIN'],
            alpha_pair_margin=CONFIG['S2_LOSS_ALPHA_PAIR_MARGIN'],
            pair_margin=CONFIG['S2_PAIR_MARGIN'],
            alpha_clear_mist=CONFIG['S2_LOSS_ALPHA_CLEAR_MIST'],
            clear_mist_penalty_th=CONFIG['S2_CLEAR_MIST_PENALTY_TH'],
            jja_weight=CONFIG['S2_JJA_WEIGHT'],
            boundary_weight=CONFIG['S2_BOUNDARY_WEIGHT'],
            clear_hard_neg_weight=CONFIG['S2_CLEAR_HARD_NEG_WEIGHT'],
        ).to(device)

    # Stage 2 输出头的参数名
    HEAD_PARAM_NAMES = {'fine_classifier', 'low_vis_detector', 'reg_head'}

    # ==========================================
    # [修改1c-impl] Phase A优化器：部分解冻backbone
    # 原策略: 只训练 extra_encoder + heads，backbone完全冻结。
    # 新策略: 在原基础上额外解冻 fusion_kan + temporal_norm。
    #
    # 理由:
    #   - fusion_kan 是所有特征流（时序/静态/物理/extra）的融合汇聚点，
    #     以中等LR解冻可以让模型重新学习NWP特征的最优融合方式，
    #     而不必解冻昂贵的GRU/SE模块。
    #   - temporal_norm (LayerNorm) 的均值/方差参数可以重新校准NWP
    #     时序特征的分布，以极低代价修正ERA5→NWP的分布偏移。
    #   - 这两个模块共约 fusion_dim*hidden_dim*3 + hidden_dim*2 参数，
    #     远小于完整backbone，保持Phase A的"轻量适配"定位。
    #   - find_unused_parameters=True 仍然必需（其余backbone层冻结）。
    # ==========================================
    def build_s2_optimizer_phase_a(raw_model):
        # 第一步：冻结所有参数
        for param in raw_model.parameters():
            param.requires_grad = False

        # [修改] 解冻目标集合
        # 高LR组: extra_encoder（NWP专属特征编码器）+ 所有输出头
        HEAD_EXTRA_NAMES = HEAD_PARAM_NAMES | {'extra_encoder', 'regime_encoder'}
        # 中LR组: fusion_kan（特征融合层）+ temporal_norm（时序归一化）
        FUSION_ADAPT_NAMES = {'fusion_kan', 'temporal_norm'}

        head_extra_params = []
        fusion_adapt_params = []

        for name, param in raw_model.named_parameters():
            top_name = name.split('.')[0]
            if top_name in HEAD_EXTRA_NAMES:
                param.requires_grad = True
                head_extra_params.append(param)
            elif top_name in FUSION_ADAPT_NAMES:
                param.requires_grad = True
                fusion_adapt_params.append(param)

        if g_rank == 0:
            n_head   = sum(p.numel() for p in head_extra_params)
            n_fusion = sum(p.numel() for p in fusion_adapt_params)
            print(
                f"[S2-PhaseA-Optim] Frozen: GRU + SE + static/physics/veg encoders.\n"
                f"  Trainable (high LR={CONFIG['S2_LR_HEAD_A']:.1e}): "
                f"extra/regime encoders + heads  ({n_head/1e6:.3f}M params)\n"
                f"  Trainable (mid  LR={CONFIG['S2_LR_FUSION_A']:.1e}): "
                f"fusion_kan + temporal_norm ({n_fusion/1e6:.3f}M params)",
                flush=True
            )

        # find_unused=True 因为部分backbone仍然冻结
        new_model = wrap_ddp(raw_model, l_rank, w_size, find_unused=True)

        optimizer = optim.AdamW(
            [
                {'params': head_extra_params,  'lr': CONFIG['S2_LR_HEAD_A']},
                {'params': fusion_adapt_params, 'lr': CONFIG['S2_LR_FUSION_A']},
            ],
            weight_decay=CONFIG['S2_WEIGHT_DECAY']
        )
        return new_model, optimizer

    # ==========================================
    # Exp4: Phase A2 优化器 — 解冻 GRU + SE
    # ==========================================
    def build_s2_optimizer_phase_a2(raw_model):
        for param in raw_model.parameters():
            param.requires_grad = False

        HEAD_EXTRA_NAMES = HEAD_PARAM_NAMES | {'extra_encoder', 'regime_encoder'}
        FUSION_ADAPT_NAMES = {'fusion_kan', 'temporal_norm'}
        GRU_SE_NAMES = {'temporal_stream', 'se_block'}

        head_extra_params = []
        fusion_adapt_params = []
        gru_se_params = []

        for name, param in raw_model.named_parameters():
            top_name = name.split('.')[0]
            if top_name in HEAD_EXTRA_NAMES or top_name in FUSION_ADAPT_NAMES:
                param.requires_grad = True
                head_extra_params.append(param)
            elif top_name in GRU_SE_NAMES:
                param.requires_grad = True
                gru_se_params.append(param)

        if g_rank == 0:
            n_head = sum(p.numel() for p in head_extra_params)
            n_gru_se = sum(p.numel() for p in gru_se_params)
            print(
                f"[S2-PhaseA2-Optim] Frozen: static/physics/veg encoders.\n"
                f"  Trainable (LR={CONFIG['S2_LR_HEAD_A2']:.1e}): "
                f"heads+extra+regime+fusion+norm ({n_head/1e6:.3f}M params)\n"
                f"  Trainable (LR={CONFIG['S2_LR_GRU_SE_A2']:.1e}): "
                f"GRU+SE ({n_gru_se/1e6:.3f}M params)",
                flush=True
            )

        new_model = wrap_ddp(raw_model, l_rank, w_size, find_unused=True)

        optimizer = optim.AdamW(
            [
                {'params': head_extra_params,  'lr': CONFIG['S2_LR_HEAD_A2']},
                {'params': gru_se_params,      'lr': CONFIG['S2_LR_GRU_SE_A2']},
            ],
            weight_decay=CONFIG['S2_WEIGHT_DECAY']
        )
        return new_model, optimizer

    # Phase B优化器：全量解冻
    def build_s2_optimizer_phase_b(raw_model):
        for param in raw_model.parameters():
            param.requires_grad = True

        head_params     = []
        backbone_params = []
        for name, param in raw_model.named_parameters():
            module_name = name.split('.')[0]
            if module_name in HEAD_PARAM_NAMES:
                head_params.append(param)
            else:
                backbone_params.append(param)

        if g_rank == 0:
            n_bb   = sum(p.numel() for p in backbone_params)
            n_head = sum(p.numel() for p in head_params)
            print(
                f"[S2-PhaseB-Optim] Unfrozen all. "
                f"backbone params: {n_bb/1e6:.3f}M, "
                f"head params: {n_head/1e6:.3f}M",
                flush=True
            )
            print(
                f"[S2-PhaseB-Optim] LR backbone={CONFIG['S2_LR_BACKBONE_B']:.1e}, "
                f"LR head={CONFIG['S2_LR_HEAD_B']:.1e}",
                flush=True
            )

        new_model = wrap_ddp(raw_model, l_rank, w_size, find_unused=False)

        optimizer = optim.AdamW(
            [
                {'params': backbone_params, 'lr': CONFIG['S2_LR_BACKBONE_B']},
                {'params': head_params,     'lr': CONFIG['S2_LR_HEAD_B']},
            ],
            weight_decay=CONFIG['S2_WEIGHT_DECAY']
        )
        return new_model, optimizer

    # ==========================================
    # 确定 S1 最优权重路径
    # ==========================================
    if CONFIG['S1_BEST_CKPT_PATH'] is not None:
        s1_best_score_path = CONFIG['S1_BEST_CKPT_PATH']
    else:
        s1_best_score_path = os.path.join(
            CONFIG['SAVE_CKPT_DIR'], f"{exp_id}_S1_best_score.pt"
        )

    if g_rank == 0:
        print(f"[Stage 2] Will load S1 weights from: {s1_best_score_path}", flush=True)

    # ==========================================
    # 构建裸模型，加载 S1 最优权重
    # ==========================================
    raw_model = build_model_raw()

    if g_rank == 0:
        n_params = sum(p.numel() for p in raw_model.parameters())
        print(f"[Model] Total params={n_params/1e6:.2f}M", flush=True)

    # ==========================================
    # Exp10: 加载 S1 权重并处理 physics_encoder 尺寸变化
    # ==========================================
    if os.path.exists(s1_best_score_path):
        s1_state = torch.load(s1_best_score_path, map_location=device)
        target_model = raw_model

        compatible_state = {}
        for k, v in s1_state.items():
            if k in target_model.state_dict():
                if v.shape == target_model.state_dict()[k].shape:
                    compatible_state[k] = v
                elif g_rank == 0:
                    print(f"  [Ckpt] Shape mismatch for '{k}': "
                          f"S1={v.shape} vs S2={target_model.state_dict()[k].shape}, "
                          f"using partial load", flush=True)

        missing, unexpected = target_model.load_state_dict(compatible_state, strict=False)

        pe_weight_key = 'physics_encoder.0.weight'
        pe_bias_key = 'physics_encoder.0.bias'
        if pe_weight_key in s1_state:
            s1_w = s1_state[pe_weight_key]
            s1_in_dim = s1_w.shape[1]
            with torch.no_grad():
                target_model.physics_encoder[0].weight[:, :s1_in_dim].copy_(s1_w)
            if pe_bias_key in s1_state:
                target_model.physics_encoder[0].bias.data.copy_(s1_state[pe_bias_key])
            if g_rank == 0:
                print(f"  [Ckpt] Partially loaded physics_encoder: "
                      f"copied {s1_in_dim}/{target_model.physics_encoder[0].weight.shape[1]} input dims",
                      flush=True)

        if g_rank == 0:
            if missing:
                print(f"  [Ckpt] Missing keys ({len(missing)}): "
                      f"{missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
            print(f"  [Ckpt] S1 checkpoint loaded successfully.", flush=True)

        del s1_state
    else:
        if g_rank == 0:
            print(
                f"[Stage 2] WARNING: S1 checkpoint not found at {s1_best_score_path}. "
                f"Proceeding with random initialization.",
                flush=True
            )

    # ==========================================
    # Exp5: 存储预训练权重用于 L2-SP 正则化
    # ==========================================
    pretrained_state = {k: v.clone().detach().to(device)
                        for k, v in raw_model.state_dict().items()}
    if g_rank == 0:
        print(f"[L2-SP] Stored {len(pretrained_state)} pretrained parameter tensors.", flush=True)

    # ==========================================
    # 加载 Stage 2 数据
    # ==========================================
    if g_rank == 0:
        print("\n" + "="*60, flush=True)
        print("[Stage 2] Loading FE data...", flush=True)

    tr_ds_s2, val_ds_s2, scaler_s2 = load_data(
        CONFIG['S2_DATA_DIR'], None,
        g_rank, l_rank, device, False,
        CONFIG['WINDOW_SIZE'], w_size, run_exp_id
    )

    if g_rank == 0:
        print(f"[Stage 2] Train={len(tr_ds_s2)}, Val={len(val_ds_s2)}", flush=True)

    # ==========================================
    # Stage 2 Phase A1 (Exp4: 3-phase unfreezing)
    # ==========================================
    if g_rank == 0:
        print("\n" + "-"*60, flush=True)
        print(
            f"[Stage 2 / Phase A1] Partial freeze: GRU+SE frozen, "
            f"fusion_kan+temporal_norm+extra_encoder+heads trainable. "
            f"Steps={CONFIG['S2_PHASE_A1_STEPS']}",
            flush=True
        )

    model_pa1, opt_pa1 = build_s2_optimizer_phase_a(raw_model)
    loss_fn_pa1 = build_s2_loss()

    train_stage(
        tag='S2_PhaseA1',
        model=model_pa1, tr_ds=tr_ds_s2, val_ds=val_ds_s2,
        optimizer=opt_pa1, loss_fn=loss_fn_pa1, device=device,
        rank=g_rank, world_size=w_size,
        total_steps=CONFIG['S2_PHASE_A1_STEPS'],
        val_int=CONFIG['S2_VAL_INTERVAL'],
        batch_size=CONFIG['S2_BATCH_SIZE'],
        grad_accum=CONFIG['S2_GRAD_ACCUM'],
        fog_ratio=CONFIG['S2_FOG_RATIO'],
        mist_ratio=CONFIG['S2_MIST_RATIO'],
        exp_id=run_exp_id,
        patience=CONFIG['S2_ES_PATIENCE'],
        pretrained_state=pretrained_state,
        l2sp_alpha=CONFIG['L2SP_ALPHA_A'],
    )

    # ==========================================
    # Phase A1 → Phase A2 切换
    # ==========================================
    raw_model_a2 = rewrap_ddp(model_pa1, w_size)
    del model_pa1
    torch.cuda.empty_cache()
    safe_barrier(w_size, device)

    s2_phase_a1_best_path = os.path.join(
        CONFIG['SAVE_CKPT_DIR'], f"{run_exp_id}_S2_PhaseA1_best_score.pt"
    )
    if os.path.exists(s2_phase_a1_best_path):
        load_checkpoint(raw_model_a2, s2_phase_a1_best_path, g_rank, w_size, device)
        if g_rank == 0:
            print(f"[Stage 2 / Phase A2] Loaded A1 best checkpoint.", flush=True)
    else:
        if g_rank == 0:
            print(f"[Stage 2 / Phase A2] WARNING: A1 best checkpoint not found, "
                  f"continuing with current weights.", flush=True)

    # ==========================================
    # Stage 2 Phase A2 (Exp4: unfreeze GRU+SE)
    # ==========================================
    if g_rank == 0:
        print("\n" + "-"*60, flush=True)
        print(
            f"[Stage 2 / Phase A2] GRU+SE unfrozen with low LR. "
            f"Steps={CONFIG['S2_PHASE_A2_STEPS']}",
            flush=True
        )

    model_pa2, opt_pa2 = build_s2_optimizer_phase_a2(raw_model_a2)
    loss_fn_pa2 = build_s2_loss()

    train_stage(
        tag='S2_PhaseA2',
        model=model_pa2, tr_ds=tr_ds_s2, val_ds=val_ds_s2,
        optimizer=opt_pa2, loss_fn=loss_fn_pa2, device=device,
        rank=g_rank, world_size=w_size,
        total_steps=CONFIG['S2_PHASE_A2_STEPS'],
        val_int=CONFIG['S2_VAL_INTERVAL'],
        batch_size=CONFIG['S2_BATCH_SIZE'],
        grad_accum=CONFIG['S2_GRAD_ACCUM'],
        fog_ratio=CONFIG['S2_FOG_RATIO'],
        mist_ratio=CONFIG['S2_MIST_RATIO'],
        exp_id=run_exp_id,
        patience=CONFIG['S2_ES_PATIENCE'],
        pretrained_state=pretrained_state,
        l2sp_alpha=CONFIG['L2SP_ALPHA_A'],
    )

    # ==========================================
    # Phase A2 → Phase B 切换
    # ==========================================
    raw_model_pb = rewrap_ddp(model_pa2, w_size)
    del model_pa2
    torch.cuda.empty_cache()
    safe_barrier(w_size, device)

    s2_phase_a2_best_path = os.path.join(
        CONFIG['SAVE_CKPT_DIR'], f"{run_exp_id}_S2_PhaseA2_best_score.pt"
    )
    if os.path.exists(s2_phase_a2_best_path):
        load_checkpoint(raw_model_pb, s2_phase_a2_best_path, g_rank, w_size, device)
        if g_rank == 0:
            print(f"[Stage 2 / Phase B] Loaded A2 best checkpoint.", flush=True)
    else:
        if g_rank == 0:
            print(f"[Stage 2 / Phase B] WARNING: A2 best checkpoint not found, "
                  f"continuing with current weights.", flush=True)

    # ==========================================
    # Stage 2 Phase B (full unfreeze)
    # ==========================================
    if g_rank == 0:
        print("\n" + "-"*60, flush=True)
        print(
            f"[Stage 2 / Phase B] Backbone FULLY UNFROZEN with calibrated LR. "
            f"Steps={CONFIG['S2_PHASE_B_STEPS']}",
            flush=True
        )

    model_pb, opt_pb = build_s2_optimizer_phase_b(raw_model_pb)
    loss_fn_pb = build_s2_loss()

    train_stage(
        tag='S2_PhaseB',
        model=model_pb, tr_ds=tr_ds_s2, val_ds=val_ds_s2,
        optimizer=opt_pb, loss_fn=loss_fn_pb, device=device,
        rank=g_rank, world_size=w_size,
        total_steps=CONFIG['S2_PHASE_B_STEPS'],
        val_int=CONFIG['S2_VAL_INTERVAL'],
        batch_size=CONFIG['S2_BATCH_SIZE'],
        grad_accum=CONFIG['S2_GRAD_ACCUM'],
        fog_ratio=CONFIG['S2_FOG_RATIO'],
        mist_ratio=CONFIG['S2_MIST_RATIO'],
        exp_id=run_exp_id,
        patience=CONFIG['S2_ES_PATIENCE'],
        pretrained_state=pretrained_state,
        l2sp_alpha=CONFIG['L2SP_ALPHA_B'],
    )

    # ==========================================
    # Exp8: 温度缩放校准
    # ==========================================
    if g_rank == 0:
        print("\n" + "-"*60, flush=True)
        print("[Post-Training] Temperature Scaling Calibration...", flush=True)

    raw_model_final = rewrap_ddp(model_pb, w_size)
    s2_phase_b_best_path = os.path.join(
        CONFIG['SAVE_CKPT_DIR'], f"{run_exp_id}_S2_PhaseB_best_score.pt"
    )
    if os.path.exists(s2_phase_b_best_path):
        load_checkpoint(raw_model_final, s2_phase_b_best_path, g_rank, w_size, device)

    def _worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_info.dataset.X = None

    val_loader_final = DataLoader(
        val_ds_s2, batch_size=CONFIG['S2_BATCH_SIZE'], shuffle=False,
        num_workers=2, pin_memory=True, worker_init_fn=_worker_init_fn,
    )

    optimal_temp = calibrate_temperature(
        raw_model_final, val_loader_final, device, CONFIG, rank=g_rank
    )

    payload = None
    global_best = None
    global_joint = None
    if g_rank == 0:
        payload = collect_validation_outputs(
            raw_model_final,
            val_ds_s2,
            device,
            CONFIG['S2_BATCH_SIZE'],
            temperature=optimal_temp,
        )
        global_best = search_global_thresholds(payload['probs'], payload['targets'], use_joint=False)
        global_joint = search_global_thresholds(payload['probs'], payload['targets'], use_joint=True)

    # ==========================================
    # Exp7: 逐季节阈值优化
    # ==========================================
    if g_rank == 0:
        print("\n[Post-Training] Per-Season Threshold Optimization...", flush=True)

    season_thresholds = None
    if g_rank == 0:
        season_thresholds = evaluate_per_season_from_arrays(
            payload['probs'], payload['targets'], payload['months'], CONFIG,
            global_thresholds=global_joint,
        )

    if g_rank == 0 and season_thresholds:
        season_th_path = os.path.join(
            CONFIG['SAVE_CKPT_DIR'], f"{run_exp_id}_season_thresholds.pt"
        )
        season_fog_th, season_mist_th = thresholds_from_season_table(payload['months'], season_thresholds)
        season_override_preds = apply_override_threshold_rule(payload['probs'], season_fog_th, season_mist_th)
        season_override_stats = compute_prediction_metrics(payload['probs'], payload['targets'], season_override_preds)
        season_preds = apply_joint_threshold_rule(payload['probs'], season_fog_th, season_mist_th)
        season_stats = compute_prediction_metrics(payload['probs'], payload['targets'], season_preds)
        offsets = fit_daynight_offsets(
            payload['probs'], payload['targets'], payload['months'],
            payload['is_day'], season_thresholds,
        )
        offset_fog_th, offset_mist_th = apply_daynight_offsets(
            season_fog_th, season_mist_th, payload['months'], payload['is_day'], offsets
        )
        offset_preds = apply_joint_threshold_rule(payload['probs'], offset_fog_th, offset_mist_th)
        offset_stats = compute_prediction_metrics(payload['probs'], payload['targets'], offset_preds)
        prior_cfg, prior_stats = tune_jja_prior_filters(
            payload['probs'], payload['targets'], payload['months'],
            payload['regime'], offset_fog_th, offset_mist_th,
        )

        adj_probs = payload['probs'].copy()
        jja_mask = np.isin(payload['months'], [6, 7, 8])
        low_vis_factor = np.ones(jja_mask.sum(), dtype=np.float64)
        low_vis_factor *= np.where(payload['regime']['mixing'][jja_mask] > 0.55, prior_cfg['mixing_low_vis'], 1.0)
        low_vis_factor *= np.where(payload['regime']['ventilation'][jja_mask] > 0.55, prior_cfg['ventilation_low_vis'], 1.0)
        adj_probs[jja_mask, 0] *= np.where(payload['regime']['convective'][jja_mask] > 0.45, prior_cfg['convective_fog'], 1.0)
        adj_probs[jja_mask, 0] *= low_vis_factor
        adj_probs[jja_mask, 1] *= low_vis_factor
        adj_probs = adj_probs / np.clip(adj_probs.sum(axis=1, keepdims=True), 1e-6, None)
        best_preds = apply_joint_threshold_rule(adj_probs, offset_fog_th, offset_mist_th)
        best_stats = compute_prediction_metrics(adj_probs, payload['targets'], best_preds)

        manifest_path = os.path.join(CONFIG['SAVE_CKPT_DIR'], f"{run_exp_id}_hard_example_manifest_val.csv")
        train_manifest_path = os.path.join(CONFIG['SAVE_CKPT_DIR'], f"{run_exp_id}_clear_to_mist_manifest_train.csv")
        mode_candidates = {
            'global_thresholds': (payload['probs'], apply_override_threshold_rule(payload['probs'], global_best['fog_th'], global_best['mist_th']), global_best['stats']),
            'season_thresholds_temp': (payload['probs'], season_override_preds, season_override_stats),
            'season_thresholds_temp_joint': (payload['probs'], season_preds, season_stats),
            'season_daynight_joint': (payload['probs'], offset_preds, offset_stats),
            'season_daynight_joint_with_jja_priors': (adj_probs, best_preds, best_stats),
        }
        mode_baseline_stats = global_best['stats']
        best_mode_name = None
        best_mode_payload = None
        best_mode_rank = None
        for mode_name, candidate_payload in mode_candidates.items():
            stats = candidate_payload[2]
            if not is_valid_postprocess_candidate(stats, CONFIG, baseline_stats=mode_baseline_stats, relaxed=True):
                continue
            rank = postprocess_rank_tuple(stats)
            if best_mode_rank is None or rank > best_mode_rank:
                best_mode_rank = rank
                best_mode_name = mode_name
                best_mode_payload = candidate_payload
        if best_mode_payload is None:
            best_mode_name = 'global_thresholds'
            best_mode_payload = mode_candidates[best_mode_name]
        best_mode_probs, best_mode_preds, best_mode_stats = best_mode_payload
        export_payload = dict(payload)
        export_payload['probs'] = best_mode_probs
        export_hard_example_manifest(manifest_path, export_payload, best_mode_preds)

        train_payload = collect_validation_outputs(
            raw_model_final,
            tr_ds_s2,
            device,
            CONFIG['S2_BATCH_SIZE'],
            temperature=optimal_temp,
        )
        train_season_fog_th, train_season_mist_th = thresholds_from_season_table(train_payload['months'], season_thresholds)
        train_offset_fog_th, train_offset_mist_th = apply_daynight_offsets(
            train_season_fog_th, train_season_mist_th, train_payload['months'], train_payload['is_day'], offsets
        )
        train_jja_mask = np.isin(train_payload['months'], [6, 7, 8])
        train_adj_probs = train_payload['probs'].copy()
        train_low_vis_factor = np.ones(train_jja_mask.sum(), dtype=np.float64)
        train_low_vis_factor *= np.where(train_payload['regime']['mixing'][train_jja_mask] > 0.55, prior_cfg['mixing_low_vis'], 1.0)
        train_low_vis_factor *= np.where(train_payload['regime']['ventilation'][train_jja_mask] > 0.55, prior_cfg['ventilation_low_vis'], 1.0)
        train_adj_probs[train_jja_mask, 0] *= np.where(train_payload['regime']['convective'][train_jja_mask] > 0.45, prior_cfg['convective_fog'], 1.0)
        train_adj_probs[train_jja_mask, 0] *= train_low_vis_factor
        train_adj_probs[train_jja_mask, 1] *= train_low_vis_factor
        train_adj_probs = train_adj_probs / np.clip(train_adj_probs.sum(axis=1, keepdims=True), 1e-6, None)

        if best_mode_name == 'global_thresholds':
            train_mode_probs = train_payload['probs']
            train_mode_preds = apply_override_threshold_rule(train_payload['probs'], global_best['fog_th'], global_best['mist_th'])
        elif best_mode_name == 'season_thresholds_temp':
            train_mode_probs = train_payload['probs']
            train_mode_preds = apply_override_threshold_rule(train_payload['probs'], train_season_fog_th, train_season_mist_th)
        elif best_mode_name == 'season_thresholds_temp_joint':
            train_mode_probs = train_payload['probs']
            train_mode_preds = apply_joint_threshold_rule(train_payload['probs'], train_season_fog_th, train_season_mist_th)
        elif best_mode_name == 'season_daynight_joint':
            train_mode_probs = train_payload['probs']
            train_mode_preds = apply_joint_threshold_rule(train_payload['probs'], train_offset_fog_th, train_offset_mist_th)
        else:
            train_mode_probs = train_adj_probs
            train_mode_preds = apply_joint_threshold_rule(train_adj_probs, train_offset_fog_th, train_offset_mist_th)

        train_export_payload = dict(train_payload)
        train_export_payload['probs'] = train_mode_probs
        export_hard_example_manifest(train_manifest_path, train_export_payload, train_mode_preds)

        ablation_summary = {
            'global_thresholds': {
                'fog_th': global_best['fog_th'],
                'mist_th': global_best['mist_th'],
                **global_best['stats'],
            },
            'season_thresholds_temp': season_override_stats,
            'season_thresholds_temp_joint': season_stats,
            'season_daynight_joint': offset_stats,
            'season_daynight_joint_with_jja_priors': best_stats,
        }
        with open(os.path.join(CONFIG['SAVE_CKPT_DIR'], f"{run_exp_id}_postprocess_ablation.json"), "w", encoding="utf-8") as f:
            json.dump(ablation_summary, f, indent=2, ensure_ascii=False)

        torch.save({
            'season_thresholds': season_thresholds,
            'temperature': optimal_temp,
            'global_thresholds': {'fog_th': global_best['fog_th'], 'mist_th': global_best['mist_th']},
            'global_joint_thresholds': {'fog_th': global_joint['fog_th'], 'mist_th': global_joint['mist_th']},
            'daynight_offsets': offsets,
            'prior_filters': prior_cfg,
            'best_postprocess_mode': best_mode_name,
        }, season_th_path)
        print(f"  [Save] Season thresholds + temperature -> {season_th_path}", flush=True)
        print(
            f"  [PostProcess] Global score={global_best['stats']['score']:.4f} | "
            f"Season={season_override_stats['score']:.4f} | "
            f"Season+Joint={season_stats['score']:.4f} | "
            f"Offsets={offset_stats['score']:.4f} | "
            f"Offsets+Priors={best_stats['score']:.4f}",
            flush=True
        )
        print(f"  [HardExamples] Validation manifest -> {manifest_path}", flush=True)
        print(f"  [HardExamples] Training manifest -> {train_manifest_path}", flush=True)

    # ==========================================
    # 收尾
    # ==========================================
    if g_rank == 0:
        print("\n" + "="*60, flush=True)
        print("[Summary] Stage 2 complete.", flush=True)
        print(f"  Base Exp ID      : {exp_id}", flush=True)
        print(f"  S2 Run ID        : {run_exp_id}", flush=True)
        print(f"  S2 PhaseA1 best  : {run_exp_id}_S2_PhaseA1_best_score.pt", flush=True)
        print(f"  S2 PhaseA2 best  : {run_exp_id}_S2_PhaseA2_best_score.pt", flush=True)
        print(f"  S2 PhaseB  best  : {run_exp_id}_S2_PhaseB_best_score.pt", flush=True)
        print(f"  Temperature:     {optimal_temp:.4f}", flush=True)

    del pretrained_state
    cleanup_temp_files(run_exp_id)
    if w_size > 1:
        dist.destroy_process_group()

    if g_rank == 0:
        print("\nJob finished.", flush=True)


if __name__ == "__main__":
    main()
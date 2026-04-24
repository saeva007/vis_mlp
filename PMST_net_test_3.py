import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import os
import shutil
import hashlib
import contextlib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import joblib
import warnings
import gc
import sys
import datetime
import time
import glob

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
    'EXPERIMENT_ID': 'exp_integrated_v1',
    'EXPERIMENT_ID': os.environ.get('EXPERIMENT_JOB_ID', 
                                    f'exp_{int(time.time())}'),
    'SKIP_STAGE1': False,
    'S1_PRETRAINED_PATH': None,

    # ========== 基础路径 ==========
    'BASE_PATH': BASE_PATH,
    'WINDOW_SIZE': TARGET_WINDOW_SIZE,
    'S1_DATA_DIR': S1_DIR,
    'S2_DATA_DIR': S2_DIR,
    'NUM_WORKERS': 2,

    # ========== Stage 1 ==========
    'S1_TOTAL_STEPS': 30000,
    'S1_VAL_INTERVAL': 2000,
    'S1_BATCH_SIZE': 512,
    'S1_GRAD_ACCUM': 2,
    'S1_FOG_RATIO': 0.15,
    'S1_MIST_RATIO': 0.15,
    'S1_LR_BACKBONE': 3e-4,
    'S1_WEIGHT_DECAY': 1e-3,

    # ========== Stage 2 ==========
    'S2_TOTAL_STEPS': 5000,
    'S2_VAL_INTERVAL': 500,
    'S2_BATCH_SIZE': 512,
    'S2_GRAD_ACCUM': 1,
    'S2_FOG_RATIO': 0.12,
    'S2_MIST_RATIO': 0.12,
    'S2_LR_BACKBONE': 5e-6,
    'S2_LR_HEAD': 5e-5,
    'S2_WEIGHT_DECAY': 1e-2,

    # ========== 评估约束 ==========
    'MIN_FOG_PRECISION': 0.20,
    'MIN_FOG_RECALL': 0.50,
    'MIN_MIST_PRECISION': 0.12,
    'MIN_MIST_RECALL': 0.15,
    'MIN_CLEAR_ACC': 0.95,

    # ========== 损失函数 - 保持 Script 1 的稳定参数 ==========
    'LOSS_TYPE': 'asymmetric',
    'LOSS_ALPHA_BINARY': 1.0,
    'LOSS_ALPHA_FINE': 1.0,
    'LOSS_ALPHA_CONSISTENCY': 0.5,
    'LOSS_ALPHA_FP': 3.0,
    'LOSS_ALPHA_FOG_BOOST': 0.1,
    'LOSS_ALPHA_MIST_BOOST': 0.1,
    'LOSS_FP_THRESHOLD': 0.5,

    # AsymmetricLoss - 保持 Script 1 的保守参数（避免损失爆炸）
    'ASYM_GAMMA_NEG': 2.0,   # !! 关键：不要用 5.0，会导致损失爆炸
    'ASYM_GAMMA_POS': 1.0,   # !! 关键：不要用 0.05
    'ASYM_CLIP': 0.05,

    # 类别权重 - 保持 Script 1 的保守参数
    'BINARY_POS_WEIGHT': 1.2,
    'FINE_CLASS_WEIGHT_FOG': 1.5,    # !! 关键：不要用 4.0，会导致损失爆炸
    'FINE_CLASS_WEIGHT_MIST': 1.2,
    'FINE_CLASS_WEIGHT_CLEAR': 1.0,  # !! 关键：不要用 0.5

    # Focal Loss
    'FOCAL_GAMMA': 2.0,
    'FOCAL_ALPHA': None,

    # ========== 阈值搜索 ==========
    'THRESHOLD_FOG_MIN': 0.10,
    'THRESHOLD_FOG_MAX': 0.90,
    'THRESHOLD_FOG_STEP': 0.05,
    'THRESHOLD_MIST_MIN': 0.10,
    'THRESHOLD_MIST_MAX': 0.90,
    'THRESHOLD_MIST_STEP': 0.05,
    'THRESHOLD_PHASE2_CLEAR_RECALL': 0.90,
    'THRESHOLD_PHASE2_FOG_PRECISION': 0.18,
    'THRESHOLD_PHASE2_FOG_RECALL': 0.45,
    'THRESHOLD_PHASE3_CLEAR_RECALL': 0.90,

    # 综合得分权重
    'SCORE_PHASE1_FOG': 0.45,
    'SCORE_PHASE1_MIST': 0.40,
    'SCORE_PHASE1_CLEAR': 0.15,
    'SCORE_PHASE2_FOG': 0.60,
    'SCORE_PHASE2_MIST': 0.30,
    'SCORE_PHASE2_CLEAR': 0.10,
    'SCORE_PHASE3_FOG': 0.50,
    'SCORE_PHASE3_MIST': 0.35,
    'SCORE_PHASE3_CLEAR': 0.15,

    # ========== 模型架构 ==========
    'MODEL_HIDDEN_DIM': 512,
    'MODEL_DROPOUT': 0.2,
    'MODEL_NUM_CLASSES': 3,

    # ========== 特征工程（来自 Script 2）==========
    'USE_FEATURE_ENGINEERING': True,   # 是否启用手工特征工程
    'FE_EXTRA_DIMS': 24,               # 工程特征维度

    # ========== 其他 ==========
    'GRAD_CLIP_NORM': 1.0,
    'REG_LOSS_ALPHA': 1.0,
}

# ==========================================
# 1. 基础工具与分布式初始化
# ==========================================

def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method='env://',
            timeout=datetime.timedelta(minutes=120)
        )
    return local_rank, global_rank, world_size



def cleanup_temp_files():
    """[来自 Script 2] 清理临时文件，避免 /dev/shm 堆积"""
    target_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    task_id = f"{os.getpid()}_{CONFIG['EXPERIMENT_ID']}"
    safe_id = hashlib.md5(task_id.encode()).hexdigest()[:8]
    for fname in os.listdir(target_dir):
        if safe_id in fname and (fname.endswith('.npy') or fname.endswith('.tmp')):
            try:
                os.remove(os.path.join(target_dir, fname))
                print(f"[Cleanup] Removed: {fname}")
            except:
                pass
    # 注意：fe_train_*.npy / fe_val_*.npy 是按数据集 hash 命名的，
    # 下次跑同一数据集可复用，不清理（除非手动删除或 /dev/shm 满了）


def get_available_space(path):
    """获取路径的可用空间（字节）"""
    try:
        stat = os.statvfs(path)
        return stat.f_bavail * stat.f_frsize
    except:
        return 0


def cleanup_old_files_before_copy(target_dir, current_job_id, max_age_hours=24):
    """
    在复制前清理旧文件
    
    清理策略：
    1. 删除超过 max_age_hours 的旧文件
    2. 删除当前作业ID的旧文件（避免冲突）
    3. 删除所有 .tmp 临时文件
    """
    if not os.path.exists(target_dir):
        return 0
    
    # 要清理的文件模式
    patterns = [
        'X_train_*.npy',
        'X_val_*.npy',
        'y_train_*.npy',
        'y_val_*.npy',
        'fe_train_*.npy',
        'fe_val_*.npy',
        '*.tmp'
    ]
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0
    freed_space = 0
    
    for pattern in patterns:
        for filepath in glob.glob(os.path.join(target_dir, pattern)):
            try:
                file_stat = os.stat(filepath)
                file_age = current_time - file_stat.st_mtime
                file_size = file_stat.st_size
                
                should_delete = False
                reason = ""
                
                # 条件1：超过24小时
                if file_age > max_age_seconds:
                    should_delete = True
                    reason = f"old ({file_age/3600:.1f}h)"
                
                # 条件2：当前作业ID（避免冲突，重新生成）
                elif current_job_id in os.path.basename(filepath):
                    should_delete = True
                    reason = "current_job_cleanup"
                
                # 条件3：临时文件
                elif filepath.endswith('.tmp'):
                    should_delete = True
                    reason = "temp_file"
                
                if should_delete:
                    os.remove(filepath)
                    cleaned_count += 1
                    freed_space += file_size
                    print(f"  [Cleanup] Removed {os.path.basename(filepath)[:50]} "
                          f"({file_size/1e9:.2f}GB, {reason})", flush=True)
                    
            except Exception as e:
                # 删除失败不影响继续
                pass
    
    if cleaned_count > 0:
        print(f"  [Cleanup] Total: {cleaned_count} files, {freed_space/1e9:.2f}GB freed", 
              flush=True)
    
    return freed_space


def copy_to_local(src_path, local_rank, world_size, exp_id=None, force_nfs=False):
    """
    改进版：自动清理 + 空间检查 + 失败回退
    
    参数:
        src_path: 源文件路径
        local_rank: 本地rank (0-3)
        world_size: 总进程数
        exp_id: 实验ID（默认使用 SLURM_JOB_ID）
        force_nfs: 强制使用NFS，不复制
    
    返回:
        str: 实际使用的文件路径
    """
    
    filename = os.path.basename(src_path)
    
    # 如果强制NFS模式，直接返回
    if force_nfs:
        if local_rank == 0:
            print(f"[Data-Copy] Force NFS mode: {filename}", flush=True)
        return src_path
    
    # 确定目标目录
    target_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    
    # 生成唯一文件名（基于 SLURM_JOB_ID + PID）
    if exp_id is None:
        exp_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    
    task_id = f"{os.getpid()}_{exp_id}"
    safe_id = hashlib.md5(task_id.encode()).hexdigest()[:8]
    base, ext = os.path.splitext(filename)
    local_filename = f"{base}_{safe_id}{ext}"
    local_path = os.path.join(target_dir, local_filename)
    
    # ============ 只有 rank 0 负责复制 ============
    if local_rank == 0:
        print(f"\n[Data-Copy] Processing: {filename}", flush=True)
        
        # === 步骤1：清理旧文件 ===
        print(f"[Data-Copy] Step 1: Cleanup old files in {target_dir}", flush=True)
        available_before = get_available_space(target_dir)
        print(f"  Available space: {available_before/1e9:.2f}GB", flush=True)
        
        freed = cleanup_old_files_before_copy(target_dir, exp_id, max_age_hours=24)
        
        if freed > 0:
            available_after = get_available_space(target_dir)
            print(f"  After cleanup: {available_after/1e9:.2f}GB", flush=True)
        
        # === 步骤2：检查是否已存在有效缓存 ===
        print(f"[Data-Copy] Step 2: Check existing cache", flush=True)
        need_copy = True
        
        if os.path.exists(local_path):
            try:
                local_size = os.path.getsize(local_path)
                src_size = os.path.getsize(src_path)
                
                if local_size == src_size:
                    # 快速验证文件可读性
                    with open(local_path, 'rb') as f:
                        f.read(1024)  # 读取前1KB
                    need_copy = False
                    print(f"  ✓ Valid cache found, skip copy", flush=True)
                else:
                    print(f"  ✗ Size mismatch (local:{local_size} vs src:{src_size}), "
                          f"will recopy", flush=True)
                    os.remove(local_path)
            except Exception as e:
                print(f"  ✗ Cache validation failed: {e}, will recopy", flush=True)
                try:
                    os.remove(local_path)
                except:
                    pass
        
        # === 步骤3：空间检查并复制 ===
        if need_copy:
            print(f"[Data-Copy] Step 3: Copy to RAM", flush=True)
            try:
                src_size = os.path.getsize(src_path)
                available = get_available_space(target_dir)
                required = src_size * 1.3  # 需要130%空间
                
                print(f"  File size: {src_size/1e9:.2f}GB", flush=True)
                print(f"  Required space: {required/1e9:.2f}GB", flush=True)
                print(f"  Available space: {available/1e9:.2f}GB", flush=True)
                
                if available < required:
                    print(f"  ✗ Insufficient space, fallback to NFS", flush=True)
                    local_path = src_path
                else:
                    print(f"  ✓ Copying... ", end='', flush=True)
                    start_time = time.time()
                    
                    # 原子性写入
                    tmp_path = local_path + ".tmp"
                    
                    # 删除可能存在的临时文件
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    
                    # 复制
                    shutil.copyfile(src_path, tmp_path)
                    
                    # 验证
                    if os.path.getsize(tmp_path) != src_size:
                        raise Exception("Copy size mismatch")
                    
                    # 原子性重命名
                    os.rename(tmp_path, local_path)
                    
                    elapsed = time.time() - start_time
                    speed = src_size / elapsed / 1e6  # MB/s
                    print(f"Done in {elapsed:.1f}s ({speed:.1f}MB/s)", flush=True)
                    
            except Exception as e:
                print(f"  ✗ Copy FAILED: {e}", flush=True)
                print(f"  ✓ Fallback to NFS", flush=True)
                
                # 清理失败的文件
                for path in [local_path, local_path + ".tmp"]:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except:
                        pass
                
                local_path = src_path
    
    # ============ 同步所有进程 ============
    if world_size > 1:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
        except:
            pass
    
    # ============ 其他 rank 验证文件 ============
    if local_rank != 0:
        if not os.path.exists(local_path):
            # Rank 0 可能回退到了 NFS
            local_path = src_path
    
    return local_path


def cleanup_on_exit(exp_id=None):
    """
    训练结束时清理当前作业的临时文件（可选）
    在训练脚本的 finally 块中调用
    """
    if exp_id is None:
        exp_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    
    target_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    task_id = f"{os.getpid()}_{exp_id}"
    safe_id = hashlib.md5(task_id.encode()).hexdigest()[:8]
    
    patterns = [f'*{safe_id}*']
    cleaned = 0
    
    for pattern in patterns:
        for filepath in glob.glob(os.path.join(target_dir, pattern)):
            try:
                os.remove(filepath)
                cleaned += 1
            except:
                pass
    
    if cleaned > 0:
        print(f"[Cleanup] Removed {cleaned} temporary files on exit")


# ==========================================
# 2. [来自 Script 2] 手工特征工程
# ==========================================

class FogFeatureEngineer:
    """
    在数据加载时计算额外的雾诊断特征（24维）。
    在 __getitem__ 之外批量预计算，避免影响训练速度。
    """
    def __init__(self, window_size=12, dyn_vars_count=25):
        self.window_size = window_size
        self.dyn_vars = dyn_vars_count
        self.idx = {
            'rh2m': 0, 't2m': 1, 'precip': 2, 'sw_rad': 4,
            'wspd10': 6, 'cape': 9, 'lcc': 10, 't925': 11,
            'rh925': 12, 'dpd': 22, 'inversion': 23, 'zenith': 24
        }
        self.params = {
            'optimal_wspd': 3.5, 'wspd_sigma': 2.5, 'dpd_threshold': 2.0,
            'stability_scale': 2.0, 'lcc_threshold': 0.3, 'rad_threshold': 800.0
        }

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        """
        批量处理，比 per-sample 调用快得多。
        输入: X shape (N, original_dim)，不含veg_id（veg_id单独存储）
        输出: X_new shape (N, 24) 额外特征
        """
        N = X.shape[0]
        split_idx = self.window_size * self.dyn_vars  # 300
        X_dyn_seq = X[:, :split_idx].reshape(N, self.window_size, self.dyn_vars)
        X_current = X_dyn_seq[:, -1, :]

        rh2m = X_current[:, self.idx['rh2m']]
        rh925 = X_current[:, self.idx['rh925']]
        dpd = X_current[:, self.idx['dpd']]
        wspd = X_current[:, self.idx['wspd10']]
        inversion = X_current[:, self.idx['inversion']]
        sw_rad = X_current[:, self.idx['sw_rad']]
        lcc = X_current[:, self.idx['lcc']]
        zenith = X_current[:, self.idx['zenith']]
        t2m = X_current[:, self.idx['t2m']]

        feats = []

        # 1. 饱和接近度
        rh_norm = np.clip(rh2m / 100.0, 0, 1)
        dpd_weight = 1.0 / (1.0 + np.exp(dpd / self.params['dpd_threshold']))
        feats.append((rh_norm * dpd_weight).reshape(-1, 1))

        # 2. 风速适宜性
        wind_fav = np.exp(-0.5 * ((wspd - self.params['optimal_wspd']) / self.params['wspd_sigma']) ** 2)
        feats.append(wind_fav.reshape(-1, 1))

        # 3. 大气稳定度
        ri = inversion / (wspd ** 2 + 0.1)
        stability = np.tanh(ri / self.params['stability_scale'])
        feats.append(stability.reshape(-1, 1))

        # 4. 辐射冷却潜力
        is_night = (zenith > 90.0).astype(float)
        clear_sky = np.clip(1.0 - lcc / self.params['lcc_threshold'], 0, 1)
        rad_intensity = 1.0 - np.clip(np.maximum(sw_rad, 0) / self.params['rad_threshold'], 0, 1)
        feats.append((is_night * clear_sky * rad_intensity).reshape(-1, 1))

        # 5. 垂直水汽梯度
        feats.append(np.tanh((rh2m - rh925) / 50.0).reshape(-1, 1))

        # 6. 综合雾潜力
        fog_pot = (rh_norm * 0.4 + wind_fav * 0.25 + np.clip(stability, 0, 1) * 0.2 +
                   (is_night * clear_sky * rad_intensity) * 0.15)
        feats.append(fog_pot.reshape(-1, 1))

        # 7-18. 关键变量时间差分/统计（3变量 × 4特征）
        for var_idx in [self.idx['rh2m'], self.idx['t2m'], self.idx['wspd10']]:
            var_seq = X_dyn_seq[:, :, var_idx]
            feats.append((var_seq[:, -1] - var_seq[:, -4]).reshape(-1, 1))   # 3h差
            feats.append((var_seq[:, -1] - var_seq[:, -7]).reshape(-1, 1))   # 6h差（window>=7）
            feats.append(np.std(var_seq, axis=1).reshape(-1, 1))             # 标准差
            feats.append((np.max(var_seq, axis=1) - np.min(var_seq, axis=1)).reshape(-1, 1))  # 极差

        # 19. RH 加速度
        rh_seq = X_dyn_seq[:, :, self.idx['rh2m']]
        rh_accel = (rh_seq[:, -1] - rh_seq[:, -4]) - (rh_seq[:, -4] - rh_seq[:, -7])
        feats.append(rh_accel.reshape(-1, 1))

        # 20. RH×温度交互
        feats.append((rh2m * np.exp(-t2m / 10.0)).reshape(-1, 1))

        # 21. 夜间晴空指数
        feats.append((is_night * (1 - lcc)).reshape(-1, 1))

        # 22. 雾条件指示符
        fog_cond = ((rh2m > 90) & (t2m < 10) & (wspd < 4)).astype(float)
        feats.append(fog_cond.reshape(-1, 1))

        # 23. RH/云量比
        feats.append((rh2m / (lcc * 100 + 1)).reshape(-1, 1))

        # 24. RH 平方
        feats.append(((rh2m / 100.0) ** 2).reshape(-1, 1))

        return np.concatenate(feats, axis=1).astype(np.float32)  # (N, 24)


# ==========================================
# 3. 改进的分层采样器（集成 Script 2 的有限 epoch 设计）
# ==========================================

class StratifiedBalancedSampler(Sampler):
    """
    [集成 Script 2] 使用有限 epoch_length，避免无限迭代带来的问题
    """
    def __init__(self, dataset, batch_size,
                 fog_ratio=0.10, mist_ratio=0.10, clear_ratio=0.80,
                 rank=0, world_size=1, seed=42,
                 epoch_length=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

        total = fog_ratio + mist_ratio + clear_ratio
        self.n_fog = max(1, int(batch_size * (fog_ratio / total)))
        self.n_mist = max(1, int(batch_size * (mist_ratio / total)))
        self.n_clear = batch_size - self.n_fog - self.n_mist

        y = np.array(dataset.y_cls)
        all_fog = np.where(y == 0)[0]
        all_mist = np.where(y == 1)[0]
        all_clear = np.where(y == 2)[0]

        if len(all_fog) == 0 or len(all_mist) == 0 or len(all_clear) == 0:
            raise ValueError(f"[Rank {rank}] Empty class! Fog:{len(all_fog)}, "
                             f"Mist:{len(all_mist)}, Clear:{len(all_clear)}")

        np.random.seed(seed + rank)
        np.random.shuffle(all_fog)
        np.random.shuffle(all_mist)
        np.random.shuffle(all_clear)

        self.fog_indices = np.array_split(all_fog, world_size)[rank]
        self.mist_indices = np.array_split(all_mist, world_size)[rank]
        self.clear_indices = np.array_split(all_clear, world_size)[rank]

        # [来自 Script 2] 明确指定 epoch 长度，避免无限循环
        if epoch_length is None:
            min_samples = min(len(self.fog_indices), len(self.mist_indices),
                              len(self.clear_indices))
            self.epoch_length = min(10000, max(1000, min_samples * 3 // batch_size))
        else:
            self.epoch_length = epoch_length

        if rank == 0:
            print(f"\n[StratifiedBalancedSampler]")
            print(f"  Total: Fog={len(all_fog)}, Mist={len(all_mist)}, Clear={len(all_clear)}")
            print(f"  Batch ({batch_size}): Fog={self.n_fog}, Mist={self.n_mist}, Clear={self.n_clear}")
            print(f"  Epoch length: {self.epoch_length} batches", flush=True)

    def __iter__(self):
        epoch_seed = self.seed + self.rank + int(time.time()) % 100000
        g = torch.Generator()
        g.manual_seed(epoch_seed)

        batch_list = []
        for _ in range(self.epoch_length):
            fog_b = torch.randint(0, len(self.fog_indices), (self.n_fog,), generator=g).numpy()
            mist_b = torch.randint(0, len(self.mist_indices), (self.n_mist,), generator=g).numpy()
            clear_b = torch.randint(0, len(self.clear_indices), (self.n_clear,), generator=g).numpy()

            indices = np.concatenate([
                self.fog_indices[fog_b],
                self.mist_indices[mist_b],
                self.clear_indices[clear_b]
            ])
            np.random.shuffle(indices)
            batch_list.append(indices.tolist())

        return iter(batch_list)

    def __len__(self):
        return self.epoch_length


# ==========================================
# 4. Loss 函数（保持 Script 1 的稳定参数范围）
# ==========================================

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=2, gamma_pos=1, clip=0.05, eps=1e-8,
                 class_weights=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        probs = F.softmax(logits, dim=1).clamp(self.eps, 1 - self.eps)

        pos_loss = -targets_one_hot * torch.log(probs) * ((1 - probs) ** self.gamma_pos)
        neg_probs = torch.clamp(probs, max=1 - self.clip)
        neg_loss = -(1 - targets_one_hot) * torch.log(1 - probs) * (neg_probs ** self.gamma_neg)

        loss = pos_loss + neg_loss
        if self.class_weights is not None:
            weight_mask = self.class_weights[targets].unsqueeze(1)
            loss = loss * weight_mask
        return loss.sum(dim=1).mean()


class DualBranchLoss(nn.Module):
    def __init__(self, alpha_binary=1.0, alpha_fine=1.0, alpha_consistency=0.5,
                 alpha_fp=3.0, alpha_fog_boost=0.1, alpha_mist_boost=0.1,
                 fp_threshold=0.5, loss_type='asymmetric',
                 asym_gamma_neg=2.0, asym_gamma_pos=1.0, asym_clip=0.05,
                 binary_pos_weight=1.2, fine_class_weight=None):
        super().__init__()
        self.alpha_binary = alpha_binary
        self.alpha_fine = alpha_fine
        self.alpha_consistency = alpha_consistency
        self.alpha_fp = alpha_fp
        self.alpha_fog_boost = alpha_fog_boost
        self.alpha_mist_boost = alpha_mist_boost
        self.fp_threshold = fp_threshold

        self.register_buffer('binary_pos_weight', torch.tensor([binary_pos_weight]))

        if fine_class_weight is None:
            fine_class_weight = [1.5, 1.2, 1.0]
        self.register_buffer('fine_class_weight', torch.tensor(fine_class_weight))

        self.binary_loss = nn.BCEWithLogitsLoss(pos_weight=self.binary_pos_weight)
        self.fine_loss = AsymmetricLoss(
            gamma_neg=asym_gamma_neg,
            gamma_pos=asym_gamma_pos,
            clip=asym_clip,
            class_weights=self.fine_class_weight
        )

    def forward(self, final_logits, low_vis_logit, fine_logits, targets):
        binary_targets = (targets <= 1).float().unsqueeze(1)
        loss_binary = self.binary_loss(low_vis_logit, binary_targets)
        loss_fine = self.fine_loss(fine_logits, targets)

        fine_probs = F.softmax(fine_logits, dim=1)
        low_vis_probs = torch.sigmoid(low_vis_logit)
        loss_consistency = (low_vis_probs * fine_probs[:, 2:3]).mean()

        is_clear = (targets == 2).float()
        fog_mist_prob = fine_probs[:, 0] + fine_probs[:, 1]
        high_conf = (fog_mist_prob > self.fp_threshold).float()
        loss_fp = torch.mean((fog_mist_prob * is_clear * high_conf) ** 2) * self.alpha_fp

        is_fog = (targets == 0).float()
        fog_boost = (torch.mean((1 - fine_probs[:, 0]) ** 2 * is_fog) +
                     torch.mean(fine_probs[:, 1] * is_fog)) * self.alpha_fog_boost

        is_mist = (targets == 1).float()
        mist_boost = (torch.mean((1 - fine_probs[:, 1]) ** 2 * is_mist) +
                      torch.mean(fine_probs[:, 0] * is_mist)) * self.alpha_mist_boost

        total_loss = (self.alpha_binary * loss_binary +
                      self.alpha_fine * loss_fine +
                      self.alpha_consistency * loss_consistency +
                      loss_fp + fog_boost + mist_boost)

        return total_loss, {
            'binary': loss_binary.item(),
            'fine': loss_fine.item(),
            'consistency': loss_consistency.item(),
            'false_alarm': loss_fp.item(),
            'fog_boost': fog_boost.item(),
            'mist_boost': mist_boost.item(),
        }


class PhysicsConstrainedRegLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, reg_pred, reg_target_log, raw_vis):
        mask = raw_vis < 2000
        if mask.sum() == 0:
            return torch.tensor(0.0, device=reg_pred.device)
        return self.alpha * F.huber_loss(reg_pred.view(-1)[mask], reg_target_log[mask], delta=1.0)


# ==========================================
# 5. 模型架构
# ==========================================

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=4):
        super().__init__()
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.xavier_normal_(self.cheby_coeffs)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.base_activation = nn.SiLU()

    def forward(self, x):
        x = torch.tanh(self.layer_norm(x))
        vals = [torch.ones_like(x), x]
        for i in range(2, self.degree + 1):
            vals.append(2 * x * vals[-1] - vals[-2])
        stacked = torch.stack(vals, dim=-1)
        return self.base_activation(torch.einsum("bid,iod->bo", stacked, self.cheby_coeffs))


class PhysicalStateEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(output_dim, output_dim), nn.LayerNorm(output_dim)
        )
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.encoder(x) + self.shortcut(x)


class GRUWithAttentionEncoder(nn.Module):
    def __init__(self, n_vars, hidden_dim, n_steps=None, dropout=0.2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_vars, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, 2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        gru_out = hidden_dim * 2
        self.attn = nn.Sequential(nn.Linear(gru_out, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.proj = nn.Sequential(nn.Linear(gru_out, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, x):
        out, _ = self.gru(self.embed(x))
        attn_w = F.softmax(self.attn(out), dim=1)
        return self.proj(torch.sum(out * attn_w, dim=1))


class ImprovedDualStreamPMSTNet(nn.Module):
    """
    [集成] 在原始 Script 1 架构基础上，增加可选的手工特征分支（extra_feat_dim）
    """
    def __init__(self, dyn_vars_count=25, window_size=12,
                 static_cont_dim=5, veg_num_classes=21, veg_emb_dim=16,
                 hidden_dim=512, num_classes=3, extra_feat_dim=0):
        super().__init__()
        self.dyn_vars = dyn_vars_count
        self.window = window_size
        self.static_cont_dim = static_cont_dim
        self.extra_feat_dim = extra_feat_dim

        # 物理诊断特征（Script 1 原版，保持稳定）
        from torch import nn as _nn
        self.fog_diagnostics = _FogDiagnosticFeatures(window_size, dyn_vars_count)
        self.veg_embedding = nn.Embedding(veg_num_classes, veg_emb_dim)

        self.physics_encoder = nn.Sequential(
            nn.Conv1d(5, 64, 1), nn.GELU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, hidden_dim // 4)
        )

        total_static = static_cont_dim + veg_emb_dim
        self.static_encoder = nn.Sequential(
            ChebyKANLayer(total_static, 256, degree=3),
            nn.LayerNorm(256), nn.Linear(256, hidden_dim // 2)
        )

        self.physical_vars_indices = [0, 1, 10, 12, 19, 20, 22, 23]
        self.physical_stream = PhysicalStateEncoder(len(self.physical_vars_indices), hidden_dim)

        self.temporal_vars_indices = [2, 3, 4, 5, 6, 7]
        self.temporal_stream = GRUWithAttentionEncoder(
            len(self.temporal_vars_indices), hidden_dim, window_size
        )

        # [集成 Script 2] 手工特征分支（可选）
        if extra_feat_dim > 0:
            self.extra_encoder = nn.Sequential(
                nn.Linear(extra_feat_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2), nn.GELU()
            )
            extra_h = hidden_dim // 2
        else:
            self.extra_encoder = None
            extra_h = 0

        fusion_dim = hidden_dim * 2 + hidden_dim // 2 + hidden_dim // 4 + extra_h
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(0.2), ChebyKANLayer(hidden_dim, hidden_dim, degree=3)
        )

        self.fog_specific = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(0.1)
        )

        self.low_vis_detector = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        self.fine_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, num_classes)
        )

        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self._init_bias()

    def _init_bias(self):
        self.low_vis_detector[-1].bias.data.fill_(-0.5)
        self.fine_classifier[-1].bias.data = torch.tensor([-1.1, -1.1, 0.0])

    def forward(self, x):
        split_dyn = self.dyn_vars * self.window
        split_static = split_dyn + self.static_cont_dim

        x_dyn_seq = x[:, :split_dyn].view(-1, self.window, self.dyn_vars)
        x_stat_cont = x[:, split_dyn:split_static]
        x_veg_id = x[:, split_static].long()
        x_veg_id = torch.clamp(x_veg_id, 0, self.veg_embedding.num_embeddings - 1)

        # 手工特征（如果启用）
        x_extra = x[:, split_static + 1:] if self.extra_feat_dim > 0 else None

        # 物理诊断
        physics_seq = self.fog_diagnostics(x_dyn_seq).permute(0, 2, 1)
        physics_feat = self.physics_encoder(physics_seq)

        # 静态特征
        veg_vec = self.veg_embedding(x_veg_id)
        static_feat = self.static_encoder(torch.cat([x_stat_cont, veg_vec], dim=1))

        # 物理变量流
        physical_feat = self.physical_stream(x_dyn_seq[:, -1, self.physical_vars_indices])

        # 时序流
        temporal_feat = self.temporal_stream(x_dyn_seq[:, :, self.temporal_vars_indices])

        # 融合
        parts = [physical_feat, temporal_feat, static_feat, physics_feat]
        if x_extra is not None and self.extra_encoder is not None:
            parts.append(self.extra_encoder(x_extra))
        embedding = self.fusion_layer(torch.cat(parts, dim=1))

        fog_feat = self.fog_specific(embedding)
        low_vis_logit = self.low_vis_detector(torch.cat([embedding, fog_feat], dim=1))
        fine_logits = self.fine_classifier(embedding)
        reg_out = self.reg_head(embedding)

        return fine_logits, reg_out, low_vis_logit, fine_logits


class _FogDiagnosticFeatures(nn.Module):
    """保持 Script 1 的固定物理特征（不引入可学习参数，稳定）"""
    def __init__(self, window_size=9, dyn_vars_count=25):
        super().__init__()
        self.window_size = window_size
        self.dyn_vars = dyn_vars_count
        self.idx = {
            'rh2m': 0, 't2m': 1, 'precip': 2, 'sw_rad': 4,
            'wspd10': 6, 'cape': 9, 'lcc': 10, 't925': 11,
            'rh925': 12, 'dpd': 22, 'inversion': 23, 'zenith': 24
        }

    def forward(self, x_dyn_seq):
        def g(n): return x_dyn_seq[:, :, self.idx[n]]

        rh2m, dpd, wspd = g('rh2m'), g('dpd'), g('wspd10')
        inv, sw, lcc = g('inversion'), g('sw_rad'), g('lcc')
        zenith, rh925 = g('zenith'), g('rh925')

        f1 = torch.clamp(torch.clamp(rh2m, 0, 100) / 100.0 *
                         torch.sigmoid(-3.0 * (dpd - 1.0)), 0, 1)
        f2 = torch.exp(-0.5 * ((torch.clamp(wspd, min=0) - 3.5) / 2.5) ** 2)
        f3 = torch.tanh(inv / (torch.clamp(wspd, min=0.5) ** 2 + 0.1) / 2.0)
        is_night = (zenith > 90.0).float()
        sw_lin = torch.clamp(torch.expm1(sw), min=0.0)
        f4 = is_night * torch.clamp(1.0 - torch.clamp(lcc, 0, 1) / 0.3, 0, 1) * \
             (1.0 - torch.clamp(sw_lin / 800.0, 0, 1))
        f5 = torch.clamp((rh2m - rh925) / 100.0, -1, 1)

        out = torch.stack([f1, f2, f3, f4, f5], dim=2)
        return torch.nan_to_num(out, nan=0.0)


# ==========================================
# 6. 综合评估
# ==========================================

class ComprehensiveMetrics:
    def __init__(self, config):
        self.config = config
        self.min_fog_precision = config['MIN_FOG_PRECISION']
        self.min_fog_recall = config['MIN_FOG_RECALL']
        self.min_mist_precision = config['MIN_MIST_PRECISION']
        self.min_mist_recall = config['MIN_MIST_RECALL']
        self.min_clear_acc = config['MIN_CLEAR_ACC']
        self.best_thresholds = {'fog': 0.5, 'mist': 0.5}

    def _class_metrics(self, y_true, y_pred, cls):
        y_t = (y_true == cls).astype(int)
        y_p = (y_pred == cls).astype(int)
        tp = ((y_t == 1) & (y_p == 1)).sum()
        fp = ((y_t == 0) & (y_p == 1)).sum()
        fn = ((y_t == 1) & (y_p == 0)).sum()
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f2 = 5 * p * r / (4 * p + r + 1e-6)
        return {'precision': p, 'recall': r, 'f2': f2, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}

    def _predict(self, probs, fog_th, mist_th):
        preds = np.full(len(probs), 2)
        fog_mask = (probs[:, 0] > fog_th) & (probs[:, 0] > probs[:, 1])
        preds[fog_mask] = 0
        mist_mask = (probs[:, 1] > mist_th) & (~fog_mask)
        preds[mist_mask] = 1
        return preds

    def search_optimal_thresholds(self, probs, targets, rank=0):
        fog_ths = np.arange(self.config['THRESHOLD_FOG_MIN'],
                            self.config['THRESHOLD_FOG_MAX'],
                            self.config['THRESHOLD_FOG_STEP'])
        mist_ths = np.arange(self.config['THRESHOLD_MIST_MIN'],
                             self.config['THRESHOLD_MIST_MAX'],
                             self.config['THRESHOLD_MIST_STEP'])

        best_score, best_fog_th, best_mist_th, phase = -1, 0.5, 0.5, None
        cfg = self.config

        # 阶段1：严格约束
        for ft in fog_ths:
            for mt in mist_ths:
                preds = self._predict(probs, ft, mt)
                fm = self._class_metrics(targets, preds, 0)
                mm = self._class_metrics(targets, preds, 1)
                cm = self._class_metrics(targets, preds, 2)
                if (cm['recall'] < self.min_clear_acc or
                        fm['precision'] < self.min_fog_precision or
                        fm['recall'] < self.min_fog_recall or
                        mm['precision'] < self.min_mist_precision or
                        mm['recall'] < self.min_mist_recall):
                    continue
                s = cfg['SCORE_PHASE1_FOG']*fm['f2'] + cfg['SCORE_PHASE1_MIST']*mm['f2'] + cfg['SCORE_PHASE1_CLEAR']*cm['f2']
                if s > best_score:
                    best_score, best_fog_th, best_mist_th, phase = s, ft, mt, "Phase1"

        # 阶段2：Fog优先
        if best_score < 0:
            for ft in fog_ths:
                for mt in mist_ths:
                    preds = self._predict(probs, ft, mt)
                    fm = self._class_metrics(targets, preds, 0)
                    mm = self._class_metrics(targets, preds, 1)
                    cm = self._class_metrics(targets, preds, 2)
                    if (cm['recall'] < cfg['THRESHOLD_PHASE2_CLEAR_RECALL'] or
                            fm['precision'] < cfg['THRESHOLD_PHASE2_FOG_PRECISION'] or
                            fm['recall'] < cfg['THRESHOLD_PHASE2_FOG_RECALL']):
                        continue
                    s = cfg['SCORE_PHASE2_FOG']*fm['f2'] + cfg['SCORE_PHASE2_MIST']*mm['f2'] + cfg['SCORE_PHASE2_CLEAR']*cm['f2']
                    if s > best_score:
                        best_score, best_fog_th, best_mist_th, phase = s, ft, mt, "Phase2"

        # 阶段3：宽松
        if best_score < 0:
            for ft in fog_ths:
                for mt in mist_ths:
                    preds = self._predict(probs, ft, mt)
                    fm = self._class_metrics(targets, preds, 0)
                    mm = self._class_metrics(targets, preds, 1)
                    cm = self._class_metrics(targets, preds, 2)
                    if cm['recall'] < cfg['THRESHOLD_PHASE3_CLEAR_RECALL']:
                        continue
                    s = cfg['SCORE_PHASE3_FOG']*fm['f2'] + cfg['SCORE_PHASE3_MIST']*mm['f2'] + cfg['SCORE_PHASE3_CLEAR']*cm['f2']
                    if s > best_score:
                        best_score, best_fog_th, best_mist_th, phase = s, ft, mt, "Phase3"

        self.best_thresholds = {'fog': best_fog_th, 'mist': best_mist_th}
        if rank == 0:
            print(f"  Best Thresholds: Fog={best_fog_th:.2f}, Mist={best_mist_th:.2f} | "
                  f"Score={best_score:.4f} ({phase})")
        return best_fog_th, best_mist_th

    def evaluate(self, model, val_loader, device, search_threshold=True, rank=0):
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for bx, by, _, _ in val_loader:
                bx = bx.to(device, non_blocking=True)
                logits, _, _, _ = model(bx)
                all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
                all_targets.append(by.numpy())

        probs = np.vstack(all_probs)
        targets = np.concatenate(all_targets)

        if search_threshold:
            ft, mt = self.search_optimal_thresholds(probs, targets, rank)
        else:
            ft, mt = self.best_thresholds['fog'], self.best_thresholds['mist']

        preds = self._predict(probs, ft, mt)
        fm = self._class_metrics(targets, preds, 0)
        mm = self._class_metrics(targets, preds, 1)
        cm = self._class_metrics(targets, preds, 2)

        lv_t = (targets <= 1).astype(int)
        lv_p = (preds <= 1).astype(int)
        tp = ((lv_t == 1) & (lv_p == 1)).sum()
        fp = ((lv_t == 0) & (lv_p == 1)).sum()
        fn = ((lv_t == 1) & (lv_p == 0)).sum()
        lv_ts = tp / (tp + fp + fn + 1e-6)

        cfg = self.config
        composite = (cfg['SCORE_PHASE1_FOG']*fm['f2'] +
                     cfg['SCORE_PHASE1_MIST']*mm['f2'] +
                     cfg['SCORE_PHASE1_CLEAR']*cm['f2'])

        if rank == 0:
            acc = accuracy_score(targets, preds)
            print(f"\n{'='*65}")
            print(f"  Fog:   P={fm['precision']:.4f}  R={fm['recall']:.4f}  F2={fm['f2']:.4f}  "
                  f"(TP={fm['tp']}, FP={fm['fp']}, FN={fm['fn']})")
            print(f"  Mist:  P={mm['precision']:.4f}  R={mm['recall']:.4f}  F2={mm['f2']:.4f}  "
                  f"(TP={mm['tp']}, FP={mm['fp']}, FN={mm['fn']})")
            print(f"  Clear: P={cm['precision']:.4f}  R={cm['recall']:.4f}  F2={cm['f2']:.4f}")
            print(f"  Acc={acc:.4f}  LV-TS={lv_ts:.4f}  Composite={composite:.4f}")

            constraints = {
                'Fog Precision': fm['precision'] >= self.min_fog_precision,
                'Fog Recall': fm['recall'] >= self.min_fog_recall,
                'Mist Precision': mm['precision'] >= self.min_mist_precision,
                'Mist Recall': mm['recall'] >= self.min_mist_recall,
                'Clear Acc': cm['recall'] >= self.min_clear_acc,
            }
            for name, ok in constraints.items():
                print(f"  {'✓' if ok else '✗'} {name}")
            print('='*65)

        return {
            'fog': fm, 'mist': mm, 'clear': cm,
            'composite_score': composite,
            'low_vis_ts': lv_ts,
            'thresholds': {'fog': ft, 'mist': mt}
        }


# ==========================================
# 7. 数据集（支持手工特征工程，但在加载时批量预计算）
# ==========================================

class PMSTDataset(Dataset):
    """
    [集成] 支持预计算的手工特征（extra_feats），避免 __getitem__ 中的重复计算
    """
    def __init__(self, X_path, y_cls, y_reg, y_raw, scaler=None,
                 window_size=12, extra_feats=None):
        """
        extra_feats: 预计算好的手工特征 np.ndarray (N, 24)，可以为 None
        """
        self.X = np.load(X_path, mmap_mode='r')
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.as_tensor(y_reg, dtype=torch.float32)
        self.y_raw = torch.as_tensor(y_raw, dtype=torch.float32)
        self.window_size = window_size

        self.has_scaler = scaler is not None
        if self.has_scaler:
            self.center = scaler.center_.astype(np.float32)
            self.scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)

        # 预计算手工特征（shape: N x 24 或 None）
        self.extra_feats = extra_feats  # 已经 scale 好了
        self.has_extra = extra_feats is not None

        # log 变换 mask（只对原始特征）
        feat_dim = self.X.shape[1] - 1
        self.log_mask = np.zeros(feat_dim, dtype=bool)
        for t in range(self.window_size):
            offset = t * 25
            for idx in [offset + 2, offset + 4, offset + 9]:
                if idx < feat_dim:
                    self.log_mask[idx] = True

        self.clip_min = -10.0
        self.clip_max = 10.0

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        features = row[:-1].astype(np.float32)
        veg_id = np.int64(row[-1])

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(features, 0, out=features, where=self.log_mask)
        np.log1p(features, out=features, where=self.log_mask)

        if self.has_scaler:
            features = (features - self.center) / self.scale

        features = np.clip(features, self.clip_min, self.clip_max)
        features = np.append(features, veg_id)

        # 拼接预计算的手工特征
        if self.has_extra:
            extra = self.extra_feats[idx]
            features = np.append(features, extra)

        return (torch.from_numpy(features).float(),
                self.y_cls[idx], self.y_reg[idx], self.y_raw[idx])


# 修改 copy_to_local 调用
def load_data_and_scale(data_dir, scaler=None, rank=0, device=None,
                        reuse_scaler=False, window_size=12, world_size=1,
                        use_feature_engineering=True):
    """数据加载函数"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if rank == 0:
        print(f"[Data] Loading from: {data_dir}", flush=True)
        # 显示 /dev/shm 状态
        import subprocess
        result = subprocess.run(['df', '-h', '/dev/shm'], 
                              capture_output=True, text=True)
        print(f"[Data] /dev/shm status:\n{result.stdout}", flush=True)

    raw_train_X = os.path.join(data_dir, 'X_train.npy')
    raw_val_X = os.path.join(data_dir, 'X_val.npy')
    raw_train_y = os.path.join(data_dir, 'y_train.npy')
    raw_val_y = os.path.join(data_dir, 'y_val.npy')

    # 使用改进的 copy_to_local（传入实验ID）
    exp_id = CONFIG['EXPERIMENT_ID']
    train_X = copy_to_local(raw_train_X, local_rank, world_size, exp_id=exp_id)
    val_X = copy_to_local(raw_val_X, local_rank, world_size, exp_id=exp_id)
    train_y_path = copy_to_local(raw_train_y, local_rank, world_size, exp_id=exp_id)
    val_y_path = copy_to_local(raw_val_y, local_rank, world_size, exp_id=exp_id)
  

    y_tr_raw = np.load(train_y_path)
    y_val_raw = np.load(val_y_path)
    y_tr_raw = y_tr_raw * 1000.0 if np.max(y_tr_raw) < 100 else y_tr_raw
    y_val_raw = y_val_raw * 1000.0 if np.max(y_val_raw) < 100 else y_val_raw

    def to_cls(y):
        c = np.zeros_like(y, dtype=np.int64)
        c[y >= 500] = 1
        c[y >= 1000] = 2
        return c

    y_tr_cls = to_cls(y_tr_raw)
    y_val_cls = to_cls(y_val_raw)
    y_tr_log = np.log1p(y_tr_raw).astype(np.float32)
    y_val_log = np.log1p(y_val_raw).astype(np.float32)

    # ===== Scaler（保持 Script 1 的广播逻辑）=====
    if scaler is None or not reuse_scaler:
        if rank == 0:
            print(f"[Data] Fitting Scaler...", flush=True)
            scaler = RobustScaler()
            X_temp = np.load(train_X, mmap_mode='r')
            subset_size = min(300000, len(X_temp))
            idx = np.sort(np.random.choice(len(X_temp), subset_size, replace=False))

            X_sub = X_temp[idx, :-1].copy().astype(np.float32)
            X_sub = np.nan_to_num(X_sub, nan=0.0)
            log_mask = np.zeros(X_sub.shape[1], dtype=bool)
            for t in range(window_size):
                off = t * 25
                for ii in [off + 2, off + 4, off + 9]:
                    if ii < X_sub.shape[1]:
                        log_mask[ii] = True
            np.maximum(X_sub, 0, out=X_sub, where=log_mask)
            np.log1p(X_sub, out=X_sub, where=log_mask)
            scaler.fit(X_sub)
            del X_sub, X_temp

            center_t = torch.from_numpy(scaler.center_).float().to(device)
            scale_t = torch.from_numpy(scaler.scale_).float().to(device)
            dim_t = torch.tensor([len(scaler.center_)], device=device)
        else:
            dim_t = torch.tensor([0], device=device)
            center_t = scale_t = None

        if world_size > 1 and dist.is_initialized():
            dist.broadcast(dim_t, src=0)

        feat_dim = dim_t.item()
        if rank != 0:
            center_t = torch.zeros(feat_dim, device=device)
            scale_t = torch.zeros(feat_dim, device=device)

        if world_size > 1 and dist.is_initialized():
            dist.broadcast(center_t, src=0)
            dist.broadcast(scale_t, src=0)

        if rank != 0:
            scaler = RobustScaler()
            scaler.center_ = center_t.cpu().numpy()
            scaler.scale_ = scale_t.cpu().numpy()

    # ===== [修复 Straggler] 批量预计算手工特征 =====
    # 只让 global rank=0 做计算并写入 /dev/shm，其他 rank barrier 后直接 mmap 读取
    # 避免 40 个进程同时读 NFS 大文件导致 I/O 竞争
    train_extra_feats = None
    val_extra_feats = None

    if use_feature_engineering:
        shm_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"

        # 用数据目录名做哈希，确保同一数据集共用缓存，不同数据集不冲突
        data_tag = hashlib.md5(data_dir.encode()).hexdigest()[:12]
        fe_train_path = os.path.join(shm_dir, f"fe_train_{data_tag}.npy")
        fe_val_path   = os.path.join(shm_dir, f"fe_val_{data_tag}.npy")

        # ---- 只有 global rank 0 计算并写入缓存 ----
        if rank == 0:
            fe = FogFeatureEngineer(window_size=window_size)
            chunk_size = 100000

            def _compute_and_save(X_path, out_path, label):
                if os.path.exists(out_path):
                    print(f"[Data] FE cache hit: {out_path}", flush=True)
                    return
                print(f"[Data] Computing FE ({label})...", flush=True)
                X_full = np.load(X_path, mmap_mode='r')
                parts = []
                for start in range(0, len(X_full), chunk_size):
                    chunk = X_full[start:start+chunk_size, :-1].astype(np.float32)
                    chunk = np.nan_to_num(chunk, nan=0.0)
                    fe_c = fe.transform_batch(chunk)
                    fe_c = np.clip(fe_c, -10.0, 10.0)
                    parts.append(fe_c)
                result = np.concatenate(parts, axis=0)
                np.save(out_path, result)
                print(f"[Data] FE saved -> {out_path}  shape={result.shape}", flush=True)
                del X_full, parts, result

            _compute_and_save(train_X, fe_train_path, "train")
            _compute_and_save(val_X,   fe_val_path,   "val")

        # ---- 所有 rank 等 rank 0 写完，再各自 mmap 读取 ----
        if world_size > 1 and dist.is_initialized():
            dist.barrier()

        # mmap_mode='r' 使各进程共享操作系统页缓存，几乎不增加实际内存消耗
        train_extra_feats = np.load(fe_train_path, mmap_mode='r')
        val_extra_feats   = np.load(fe_val_path,   mmap_mode='r')

        if rank == 0:
            print(f"[Data] FE loaded. Extra dims: {train_extra_feats.shape[1]}", flush=True)

    train_ds = PMSTDataset(train_X, y_tr_cls, y_tr_log, y_tr_raw,
                           scaler=scaler, window_size=window_size,
                           extra_feats=train_extra_feats)
    val_ds = PMSTDataset(val_X, y_val_cls, y_val_log, y_val_raw,
                         scaler=scaler, window_size=window_size,
                         extra_feats=val_extra_feats)

    return train_ds, val_ds, scaler


# ==========================================
# 8. 训练流程（集成 Script 2 的 model.no_sync() 修复）
# ==========================================

def train_stage(stage_name, model, train_ds, val_loader, optimizer,
                criterions, scaler_amp, device, rank, world_size,
                total_steps, val_interval, batch_size,
                fog_ratio, mist_ratio, grad_accum, exp_id):

    sampler = StratifiedBalancedSampler(
        train_ds, batch_size=batch_size,
        fog_ratio=fog_ratio, mist_ratio=mist_ratio,
        clear_ratio=1.0 - fog_ratio - mist_ratio,
        rank=rank, world_size=world_size,
        epoch_length=total_steps // world_size + 200  # 确保足够长度
    )

    train_loader = DataLoader(
        train_ds, batch_sampler=sampler,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True, persistent_workers=True
    )

    dual_loss_fn, reg_loss_fn = criterions
    evaluator = ComprehensiveMetrics(CONFIG)

    best_scores = {'composite': -1, 'fog_f2': -1, 'mist_f2': -1, 'lv_ts': -1}
    win = f"_{CONFIG['WINDOW_SIZE']}h_"

    model.train()
    optimizer.zero_grad()
    step = 0

    while step < total_steps:
        for batch_data in train_loader:
            step += 1
            if step > total_steps:
                break

            bx, by, blog, braw = batch_data
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            blog = blog.to(device, non_blocking=True)
            braw = braw.to(device, non_blocking=True)

            # [来自 Script 2] 正确的 DDP + GradAccum
            is_accum = (step % grad_accum != 0)
            ctx = model.no_sync() if (is_accum and world_size > 1) else contextlib.nullcontext()

            with ctx:
                with autocast():
                    final_logits, reg_out, low_vis_logit, fine_logits = model(bx)
                    l_dual, breakdown = dual_loss_fn(final_logits, low_vis_logit, fine_logits, by)
                    l_reg = reg_loss_fn(reg_out, blog, braw)
                    loss = (l_dual + CONFIG['REG_LOSS_ALPHA'] * l_reg) / grad_accum

                if torch.isnan(loss) or torch.isinf(loss):
                    if rank == 0:
                        print(f"\n[WARN] Step {step}: NaN/Inf, skip")
                    optimizer.zero_grad()
                    continue

                scaler_amp.scale(loss).backward()

            if not is_accum:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['GRAD_CLIP_NORM'])
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad()

            if rank == 0 and step % 100 == 0:
                print(f"\r[{stage_name}] Step {step}/{total_steps} | "
                      f"L_dual: {l_dual.item():.4f} "
                      f"(Fog+: {breakdown['fog_boost']:.4f}, Mist+: {breakdown['mist_boost']:.4f}) | "
                      f"L_reg: {l_reg.item():.4f}",
                      end="", flush=True)

            if step % val_interval == 0:
                # [来自 Script 2] 所有rank一起进入验证，避免死锁
                if world_size > 1 and dist.is_initialized():
                    dist.barrier()

                if rank == 0:
                    print(f"\n[{stage_name}] Validation @ step {step}")

                metrics = evaluator.evaluate(
                    model, val_loader, device, search_threshold=True, rank=rank
                )

                if rank == 0:
                    def _save(tag):
                        path = os.path.join(CONFIG['BASE_PATH'],
                                            f"model/pmst_{stage_name.lower()}{win}{exp_id}_{tag}.pth")
                        torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), path)
                        th_path = path.replace('.pth', '_thresholds.pkl')
                        joblib.dump(evaluator.best_thresholds, th_path)

                    if metrics['composite_score'] > best_scores['composite']:
                        best_scores['composite'] = metrics['composite_score']
                        _save('best_composite')
                        print(f"✓ Best Composite: {best_scores['composite']:.4f}")

                    if metrics['fog']['f2'] > best_scores['fog_f2']:
                        best_scores['fog_f2'] = metrics['fog']['f2']
                        _save('best_fog_f2')
                        print(f"✓ Best Fog F2: {best_scores['fog_f2']:.4f}")

                    if metrics['mist']['f2'] > best_scores['mist_f2']:
                        best_scores['mist_f2'] = metrics['mist']['f2']
                        _save('best_mist_f2')
                        print(f"✓ Best Mist F2: {best_scores['mist_f2']:.4f}")

                    if metrics['low_vis_ts'] > best_scores['lv_ts']:
                        best_scores['lv_ts'] = metrics['low_vis_ts']
                        _save('best_lv_ts')
                        print(f"✓ Best LV-TS: {best_scores['lv_ts']:.4f}")

                model.train()

                if world_size > 1 and dist.is_initialized():
                    dist.barrier()

    if rank == 0:
        print(f"\n{'='*65}")
        print(f"[{stage_name}] Done! Composite={best_scores['composite']:.4f}  "
              f"FogF2={best_scores['fog_f2']:.4f}  MistF2={best_scores['mist_f2']:.4f}  "
              f"LV-TS={best_scores['lv_ts']:.4f}")
        print('='*65)


        
        
# ==========================================
# 9. Main
# ==========================================

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.makedirs(f"/tmp/miopen_cache_{local_rank}", exist_ok=True)
    os.environ["MIOPEN_USER_DB_PATH"] = f"/tmp/miopen_cache_{local_rank}"

    local_rank, global_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    scaler_amp = GradScaler()

    use_fe = CONFIG['USE_FEATURE_ENGINEERING']
    extra_dims = CONFIG['FE_EXTRA_DIMS'] if use_fe else 0

    try:
        if global_rank == 0:
            os.makedirs(os.path.join(CONFIG['BASE_PATH'], "model"), exist_ok=True)
            os.makedirs(os.path.join(CONFIG['BASE_PATH'], "scalers"), exist_ok=True)
            print(f"[Config] Experiment: {CONFIG['EXPERIMENT_ID']}")
            print(f"[Config] Feature Engineering: {use_fe} (extra_dims={extra_dims})")

        exp_id = CONFIG['EXPERIMENT_ID']

        model = ImprovedDualStreamPMSTNet(
            dyn_vars_count=25,
            window_size=CONFIG['WINDOW_SIZE'],
            hidden_dim=CONFIG['MODEL_HIDDEN_DIM'],
            num_classes=CONFIG['MODEL_NUM_CLASSES'],
            extra_feat_dim=extra_dims
        ).to(device)

        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

        dual_loss = DualBranchLoss(
            alpha_binary=CONFIG['LOSS_ALPHA_BINARY'],
            alpha_fine=CONFIG['LOSS_ALPHA_FINE'],
            alpha_consistency=CONFIG['LOSS_ALPHA_CONSISTENCY'],
            alpha_fp=CONFIG['LOSS_ALPHA_FP'],
            alpha_fog_boost=CONFIG['LOSS_ALPHA_FOG_BOOST'],
            alpha_mist_boost=CONFIG['LOSS_ALPHA_MIST_BOOST'],
            fp_threshold=CONFIG['LOSS_FP_THRESHOLD'],
            asym_gamma_neg=CONFIG['ASYM_GAMMA_NEG'],
            asym_gamma_pos=CONFIG['ASYM_GAMMA_POS'],
            asym_clip=CONFIG['ASYM_CLIP'],
            binary_pos_weight=CONFIG['BINARY_POS_WEIGHT'],
            fine_class_weight=[CONFIG['FINE_CLASS_WEIGHT_FOG'],
                               CONFIG['FINE_CLASS_WEIGHT_MIST'],
                               CONFIG['FINE_CLASS_WEIGHT_CLEAR']]
        ).to(device)

        reg_loss = PhysicsConstrainedRegLoss(alpha=CONFIG['REG_LOSS_ALPHA']).to(device)

        scaler_s1 = None

        # ===== Stage 1 =====
        if not CONFIG['SKIP_STAGE1']:
            if global_rank == 0:
                print("\n[STAGE 1] ERA5 Pre-training")

            s1_tr, s1_val, scaler_s1 = load_data_and_scale(
                CONFIG['S1_DATA_DIR'],
                rank=global_rank, device=device,
                reuse_scaler=False,
                window_size=CONFIG['WINDOW_SIZE'],
                world_size=world_size,
                use_feature_engineering=use_fe
            )

            if global_rank == 0:
                joblib.dump(scaler_s1, os.path.join(
                    CONFIG['BASE_PATH'],
                    f"scalers/scaler_{exp_id}_{CONFIG['WINDOW_SIZE']}h.pkl"
                ))

            s1_val_loader = DataLoader(
                s1_val, batch_size=CONFIG['S1_BATCH_SIZE'],
                shuffle=False, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True
            )

            optimizer_s1 = optim.AdamW(
                model.parameters(),
                lr=CONFIG['S1_LR_BACKBONE'],
                weight_decay=CONFIG['S1_WEIGHT_DECAY']
            )

            train_stage(
                'S1', model, s1_tr, s1_val_loader, optimizer_s1,
                (dual_loss, reg_loss), scaler_amp, device, global_rank, world_size,
                CONFIG['S1_TOTAL_STEPS'], CONFIG['S1_VAL_INTERVAL'],
                CONFIG['S1_BATCH_SIZE'], CONFIG['S1_FOG_RATIO'], CONFIG['S1_MIST_RATIO'],
                CONFIG['S1_GRAD_ACCUM'], exp_id
            )

            del s1_tr, s1_val, s1_val_loader
            gc.collect()
            torch.cuda.empty_cache()

        # ===== Stage 2 =====
        if global_rank == 0:
            print("\n[STAGE 2] Forecast Fine-tuning")

        s2_tr, s2_val, scaler_s2 = load_data_and_scale(
            CONFIG['S2_DATA_DIR'],
            scaler=scaler_s1,
            rank=global_rank, device=device,
            reuse_scaler=(scaler_s1 is not None),
            window_size=CONFIG['WINDOW_SIZE'],
            world_size=world_size,
            use_feature_engineering=use_fe
        )

        if CONFIG['SKIP_STAGE1'] and global_rank == 0:
            joblib.dump(scaler_s2, os.path.join(
                CONFIG['BASE_PATH'],
                f"scalers/scaler_{exp_id}_{CONFIG['WINDOW_SIZE']}h.pkl"
            ))

        if CONFIG['SKIP_STAGE1'] and CONFIG['S1_PRETRAINED_PATH']:
            if global_rank == 0:
                print(f"Loading pretrained: {CONFIG['S1_PRETRAINED_PATH']}")
            sd = torch.load(CONFIG['S1_PRETRAINED_PATH'], map_location=device)
            (model.module if world_size > 1 else model).load_state_dict(sd)

        s2_val_loader = DataLoader(
            s2_val, batch_size=CONFIG['S2_BATCH_SIZE'],
            shuffle=False, num_workers=CONFIG['NUM_WORKERS']
        )

        params_backbone = [p for n, p in model.named_parameters()
                           if not any(k in n for k in ['detector', 'classifier', 'reg_head'])]
        params_head = [p for n, p in model.named_parameters()
                       if any(k in n for k in ['detector', 'classifier', 'reg_head'])]

        optimizer_s2 = optim.AdamW([
            {'params': params_backbone, 'lr': CONFIG['S2_LR_BACKBONE']},
            {'params': params_head, 'lr': CONFIG['S2_LR_HEAD']},
        ], weight_decay=CONFIG['S2_WEIGHT_DECAY'])

        train_stage(
            'S2', model, s2_tr, s2_val_loader, optimizer_s2,
            (dual_loss, reg_loss), scaler_amp, device, global_rank, world_size,
            CONFIG['S2_TOTAL_STEPS'], CONFIG['S2_VAL_INTERVAL'],
            CONFIG['S2_BATCH_SIZE'], CONFIG['S2_FOG_RATIO'], CONFIG['S2_MIST_RATIO'],
            CONFIG['S2_GRAD_ACCUM'], exp_id
        )

        if global_rank == 0:
            print("\n[Done] Training Completed!")

    except Exception as e:
        print(f"\n[FATAL] Rank {global_rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_temp_files()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
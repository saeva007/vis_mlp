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
import hashlib
import contextlib

warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局配置 - 集中管理所有超参数
# ==========================================

TARGET_WINDOW_SIZE = 12
BASE_PATH = "/public/home/putianshu/vis_mlp"

S1_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_{TARGET_WINDOW_SIZE}h"
S2_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_finetune_v1_{TARGET_WINDOW_SIZE}h"

CONFIG = {
    'EXPERIMENT_ID': 'exp005_NoLA_FeatEng_FreezeBackbone', 
    'SKIP_STAGE1': False,             
    'S1_PRETRAINED_PATH': None,       
    'BASE_PATH': BASE_PATH,
    'WINDOW_SIZE': TARGET_WINDOW_SIZE,
    'S1_DATA_DIR': S1_DIR,
    'S2_DATA_DIR': S2_DIR,
    'NUM_WORKERS': 6,
    'S1_TOTAL_STEPS': 30000,
    'S1_VAL_INTERVAL': 2000,
    'S1_BATCH_SIZE': 512,
    'S1_GRAD_ACCUM': 2,
    'S1_FOG_RATIO': 0.15,     
    'S1_MIST_RATIO': 0.15,
    'S1_LR_BACKBONE': 3e-4,
    'S1_WEIGHT_DECAY': 1e-3,
    'S2_TOTAL_STEPS': 6000,
    'S2_VAL_INTERVAL': 500,
    'S2_BATCH_SIZE': 512,
    'S2_GRAD_ACCUM': 1,
    'S2_FOG_RATIO': 0.15,      
    'S2_MIST_RATIO': 0.15,
    'S2_FREEZE_STEPS': 1000, 
    'S2_MAX_LR_BACKBONE': 1e-5,
    'S2_MAX_LR_HEAD': 2e-4,
    'S2_PCT_START': 0.3,
    'S2_WEIGHT_DECAY': 1e-2,
    'MIN_FOG_PRECISION': 0.18,
    'MIN_FOG_RECALL': 0.45,
    'MIN_CLEAR_ACC': 0.93,
    'LOSS_TYPE': 'asymmetric',
    'USE_LOGIT_ADJUSTMENT': False,
    'LOGIT_ADJ_TAU': 1.0, 
    'FINE_CLASS_WEIGHT_FOG': 4.0,
    'FINE_CLASS_WEIGHT_MIST': 2.0,  
    'FINE_CLASS_WEIGHT_CLEAR': 0.5,
    'ASYM_GAMMA_NEG': 5.0,          
    'ASYM_GAMMA_POS': 0.05,          
    'ASYM_CLIP': 0.05,
    'LOSS_ALPHA_BINARY': 1.0,
    'LOSS_ALPHA_FINE': 1.0,
    'LOSS_ALPHA_CONSISTENCY': 0.5,
    'LOSS_ALPHA_FP': 2.0,
    'LOSS_ALPHA_FOG_BOOST': 0.8,
    'LOSS_ALPHA_MIST_BOOST': 0.5,
    'LOSS_FP_THRESHOLD': 0.6,
    'THRESHOLD_FOG_MIN': 0.05,
    'THRESHOLD_FOG_MAX': 0.80,
    'THRESHOLD_FOG_STEP': 0.02,
    'MODEL_HIDDEN_DIM': 512,
    'MODEL_DROPOUT': 0.3,
    'MODEL_NUM_CLASSES': 3,
    'USE_FEATURE_ENGINEERING': True,
    'FE_EXTRA_DIMS': 24,
    'GRAD_CLIP_NORM': 1.0,
    'REG_LOSS_ALPHA': 0.5,
}

# ==========================================
# 1. 基础工具与分布式初始化
# ==========================================

def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    torch.cuda.set_device(local_rank)
    
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl", 
                init_method='env://', 
                timeout=datetime.timedelta(minutes=120)
            )
    return local_rank, global_rank, world_size


def copy_to_local(src_path, local_rank, world_size):
    filename = os.path.basename(src_path)
    target_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    
    task_id = f"{os.getpid()}_{CONFIG['EXPERIMENT_ID']}"
    safe_id = hashlib.md5(task_id.encode()).hexdigest()[:8]
    
    base, ext = os.path.splitext(filename)
    local_filename = f"{base}_{safe_id}{ext}"
    local_path = os.path.join(target_dir, local_filename)
    
    if local_rank == 0:
        need_copy = True
        if os.path.exists(local_path):
            try:
                if os.path.getsize(local_path) == os.path.getsize(src_path):
                    need_copy = False
                    print(f"[Rank-0] Reusing cached: {local_filename}", flush=True)
            except:
                need_copy = True
        
        if need_copy:
            print(f"[Rank-0] Copying {filename} -> RAM ({local_filename})...", flush=True)
            try:
                tmp_path = local_path + ".tmp"
                shutil.copyfile(src_path, tmp_path)
                os.rename(tmp_path, local_path)
                print(f"[Rank-0] Copy SUCCESS: {local_filename}", flush=True)
            except Exception as e:
                print(f"[Rank-0] Copy FAILED: {e}. Using NFS.", flush=True)
                local_path = src_path

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        
    if not os.path.exists(local_path):
        return src_path
    return local_path


def cleanup_temp_files():
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

# ==========================================
# 2. 特征工程类
# ==========================================

class FogFeatureEngineer:
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
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        split_idx = self.window_size * self.dyn_vars
        X_dyn_flat = X[:, :split_idx]
        X_dyn_seq = X_dyn_flat.reshape(N, self.window_size, self.dyn_vars)
        
        new_features = []
        X_current = X_dyn_seq[:, -1, :]
        
        rh2m = X_current[:, self.idx['rh2m']]
        rh925 = X_current[:, self.idx['rh925']]
        dpd = X_current[:, self.idx['dpd']]
        wspd = X_current[:, self.idx['wspd10']]
        inversion = X_current[:, self.idx['inversion']]
        sw_rad = X_current[:, self.idx['sw_rad']]
        lcc = X_current[:, self.idx['lcc']]
        zenith = X_current[:, self.idx['zenith']]
        
        rh_norm = np.clip(rh2m / 100.0, 0, 1)
        dpd_weight = 1.0 / (1.0 + np.exp(dpd / self.params['dpd_threshold']))
        saturation_index = rh_norm * dpd_weight
        new_features.append(saturation_index.reshape(-1, 1))
        
        wind_fav = np.exp(-0.5 * ((wspd - self.params['optimal_wspd']) / self.params['wspd_sigma']) ** 2)
        new_features.append(wind_fav.reshape(-1, 1))
        
        ri = inversion / (wspd ** 2 + 0.1)
        stability = np.tanh(ri / self.params['stability_scale'])
        new_features.append(stability.reshape(-1, 1))
        
        is_night = (zenith > 90.0).astype(float)
        clear_sky = np.clip(1.0 - lcc / self.params['lcc_threshold'], 0, 1)
        rad_intensity = 1.0 - np.clip(np.maximum(sw_rad, 0) / self.params['rad_threshold'], 0, 1)
        cooling_pot = is_night * clear_sky * rad_intensity
        new_features.append(cooling_pot.reshape(-1, 1))
        
        moisture_grad = np.tanh((rh2m - rh925) / 50.0)
        new_features.append(moisture_grad.reshape(-1, 1))
        
        fog_potential = (saturation_index * 0.4 + wind_fav * 0.25 + 
                         np.clip(stability, 0, 1) * 0.2 + cooling_pot * 0.15)
        new_features.append(fog_potential.reshape(-1, 1))
        
        key_vars = {'rh2m': self.idx['rh2m'], 't2m': self.idx['t2m'], 'wspd10': self.idx['wspd10']}
        for var_name, var_idx in key_vars.items():
            var_seq = X_dyn_seq[:, :, var_idx]
            if self.window_size >= 3:
                new_features.append((var_seq[:, -1] - var_seq[:, -4]).reshape(-1, 1))
            if self.window_size >= 6:
                new_features.append((var_seq[:, -1] - var_seq[:, -7]).reshape(-1, 1))
            new_features.append(np.std(var_seq, axis=1).reshape(-1, 1))
            new_features.append((np.max(var_seq, axis=1) - np.min(var_seq, axis=1)).reshape(-1, 1))
            
        if self.window_size >= 6:
            rh_seq = X_dyn_seq[:, :, self.idx['rh2m']]
            rh_accel = (rh_seq[:, -1] - rh_seq[:, -4]) - (rh_seq[:, -4] - rh_seq[:, -7])
            new_features.append(rh_accel.reshape(-1, 1))
            
        rh_t_inter = rh2m * np.exp(-X_current[:, self.idx['t2m']] / 10.0)
        new_features.append(rh_t_inter.reshape(-1, 1))
        new_features.append((is_night * (1 - lcc)).reshape(-1, 1))
        fog_cond = ((rh2m > 90) & (X_current[:, self.idx['t2m']] < 10) & (wspd < 4)).astype(float)
        new_features.append(fog_cond.reshape(-1, 1))
        new_features.append((rh2m / (lcc * 100 + 1)).reshape(-1, 1))
        new_features.append(((rh2m / 100.0) ** 2).reshape(-1, 1))
        
        X_new = np.concatenate(new_features, axis=1)
        X_enhanced = np.concatenate([X, X_new], axis=1)
        return X_enhanced

# ==========================================
# 3. ====== [关键修复] 改进的分层采样器 ======
# ==========================================

class StratifiedBalancedSampler(Sampler):
    """
    修复要点：
    1. 返回固定长度的epoch，而不是无限迭代
    2. 确保所有rank产生相同数量的batch
    3. 添加健壮的错误处理
    """
    def __init__(self, dataset, batch_size, 
                 fog_ratio=0.15, mist_ratio=0.15, clear_ratio=0.70,
                 rank=0, world_size=1, seed=42, 
                 epoch_length=None):  # 新增：明确指定epoch长度
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        
        # 归一化比例
        total = fog_ratio + mist_ratio + clear_ratio
        self.n_fog = int(batch_size * (fog_ratio / total))
        self.n_mist = int(batch_size * (mist_ratio / total))
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
        
        # ====== [关键修复] 计算合理的epoch长度 ======
        if epoch_length is None:
            # 基于最小类别的样本数计算
            min_samples = min(len(self.fog_indices), len(self.mist_indices), len(self.clear_indices))
            # 每个epoch大约遍历3遍最小类别
            self.epoch_length = min(10000, max(1000, min_samples * 3 // batch_size))
        else:
            self.epoch_length = epoch_length
            
        if rank == 0:
            print(f"[Sampler] Epoch length: {self.epoch_length} batches", flush=True)
    
    def __iter__(self):
        """
        返回固定长度的batch索引列表
        """
        epoch_seed = self.seed + self.rank + int(time.time())
        g = torch.Generator()
        g.manual_seed(epoch_seed)
        
        batch_list = []
        for _ in range(self.epoch_length):
            fog_batch = torch.randint(0, len(self.fog_indices), (self.n_fog,), generator=g).numpy()
            mist_batch = torch.randint(0, len(self.mist_indices), (self.n_mist,), generator=g).numpy()
            clear_batch = torch.randint(0, len(self.clear_indices), (self.n_clear,), generator=g).numpy()

            indices = np.concatenate([
                self.fog_indices[fog_batch],
                self.mist_indices[mist_batch],
                self.clear_indices[clear_batch]
            ])
            np.random.shuffle(indices)
            batch_list.append(indices.tolist())
            
        return iter(batch_list)
    
    def __len__(self):
        return self.epoch_length

# ==========================================
# 4. Loss 函数
# ==========================================

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, 
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
        
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, self.eps, 1 - self.eps)
        
        pos_loss = -targets_one_hot * torch.log(probs)
        pos_loss = pos_loss * ((1 - probs) ** self.gamma_pos)
        
        neg_probs = torch.clamp(probs, max=1 - self.clip)
        neg_loss = -(1 - targets_one_hot) * torch.log(1 - probs)
        neg_loss = neg_loss * (neg_probs ** self.gamma_neg)
        
        loss = pos_loss + neg_loss
        
        if self.class_weights is not None:
            weight_mask = self.class_weights[targets].unsqueeze(1)
            loss = loss * weight_mask
        
        return loss.sum(dim=1).mean()

class DualBranchLoss(nn.Module):
    def __init__(self, 
                 alpha_binary=1.0, 
                 alpha_fine=1.0, 
                 alpha_consistency=0.5,
                 alpha_fp=1.5, 
                 alpha_fog_boost=0.6,
                 alpha_mist_boost=0.6,
                 fp_threshold=0.6,
                 fine_class_weight=None,
                 class_counts=None,
                 use_logit_adj=False,
                 logit_adj_tau=1.0,
                 **kwargs):
        super().__init__()
        self.alpha_binary = alpha_binary
        self.alpha_fine = alpha_fine
        self.alpha_consistency = alpha_consistency
        self.alpha_fp = alpha_fp
        self.alpha_fog_boost = alpha_fog_boost
        self.alpha_mist_boost = alpha_mist_boost
        self.fp_threshold = fp_threshold
        
        self.use_logit_adj = use_logit_adj
        self.logit_adj_tau = logit_adj_tau
        if class_counts is not None and use_logit_adj:
            total = sum(class_counts)
            priors = torch.tensor([c/total for c in class_counts], dtype=torch.float32)
            self.register_buffer('priors', priors)
        else:
            self.register_buffer('priors', None)

        self.binary_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]))
        
        if fine_class_weight is None:
            fine_class_weight = torch.tensor([4.0, 2.0, 0.5])
        elif isinstance(fine_class_weight, list):
            fine_class_weight = torch.tensor(fine_class_weight)
        self.register_buffer('fine_class_weight', fine_class_weight)
        
        self.fine_loss = AsymmetricLoss(
            gamma_neg=kwargs.get('asym_gamma_neg', 5),
            gamma_pos=kwargs.get('asym_gamma_pos', 0.05),
            clip=kwargs.get('asym_clip', 0.05),
            class_weights=self.fine_class_weight
        )
    
    def forward(self, final_logits, low_vis_logit, fine_logits, targets):
        binary_targets = (targets <= 1).float().unsqueeze(1)
        loss_binary = self.binary_loss(low_vis_logit, binary_targets)
        
        logits_for_loss = fine_logits
        if self.use_logit_adj and self.priors is not None and self.training:
            logits_for_loss = fine_logits + self.logit_adj_tau * torch.log(self.priors + 1e-8)
            
        loss_fine = self.fine_loss(logits_for_loss, targets)
        
        fine_probs = F.softmax(fine_logits, dim=1)
        low_vis_probs = torch.sigmoid(low_vis_logit)
        inconsistency = low_vis_probs * fine_probs[:, 2:3]
        loss_consistency = inconsistency.mean()
        
        is_clear = (targets == 2).float()
        fog_mist_prob = fine_probs[:, 0] + fine_probs[:, 1]
        high_confidence_mask = (fog_mist_prob > self.fp_threshold).float()
        false_alarm_prob = fog_mist_prob * is_clear * high_confidence_mask
        loss_fp = torch.mean(false_alarm_prob ** 2) * self.alpha_fp
        
        is_fog = (targets == 0).float()
        fog_prob = fine_probs[:, 0]
        fog_boost = torch.mean((1 - fog_prob) ** 2 * is_fog) * self.alpha_fog_boost
        
        is_mist = (targets == 1).float()
        mist_prob = fine_probs[:, 1]
        mist_boost = torch.mean((1 - mist_prob) ** 2 * is_mist) * self.alpha_mist_boost
        
        total_loss = (
            self.alpha_binary * loss_binary +
            self.alpha_fine * loss_fine +
            self.alpha_consistency * loss_consistency +
            loss_fp +
            fog_boost +
            mist_boost
        )
        
        return total_loss, {
            'binary': loss_binary.item(),
            'fine': loss_fine.item(),
            'consistency': loss_consistency.item(),
            'false_alarm': loss_fp.item(),
            'fog_boost': fog_boost.item(),
            'mist_boost': mist_boost.item()
        }

class PhysicsConstrainedRegLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, reg_pred, reg_target_log, raw_vis):
        mask_low_vis = raw_vis < 2000
        if mask_low_vis.sum() == 0:
            return torch.tensor(0.0, device=reg_pred.device)
        
        reg_flat = reg_pred.view(-1)
        loss = F.huber_loss(reg_flat[mask_low_vis], reg_target_log[mask_low_vis], delta=1.0)
        return self.alpha * loss

# ==========================================
# 5. 模型架构
# ==========================================

class LearnableFogDiagnosticFeatures(nn.Module):
    def __init__(self, window_size=9, dyn_vars_count=25):
        super().__init__()
        self.window_size = window_size
        self.dyn_vars = dyn_vars_count
        self.idx = {
            'rh2m': 0, 't2m': 1, 'precip': 2, 'sw_rad': 4,
            'wspd10': 6, 'cape': 9, 'lcc': 10, 't925': 11,
            'rh925': 12, 'dpd': 22, 'inversion': 23, 'zenith': 24
        }
        self.dpd_scale = nn.Parameter(torch.tensor(3.0))
        self.dpd_shift = nn.Parameter(torch.tensor(1.0))
        self.optimal_wspd = nn.Parameter(torch.tensor(3.5))
        self.wspd_sigma = nn.Parameter(torch.tensor(2.5))
        self.stab_scale = nn.Parameter(torch.tensor(2.0))
        self.rad_threshold = nn.Parameter(torch.tensor(800.0))
        self.lcc_threshold = nn.Parameter(torch.tensor(0.3))
    
    def _extract_feature(self, x_seq, feature_name):
        idx = self.idx[feature_name]
        return x_seq[:, :, idx]
    
    def forward(self, x_dyn_seq):
        rh2m = self._extract_feature(x_dyn_seq, 'rh2m')
        dpd = self._extract_feature(x_dyn_seq, 'dpd')
        wspd = self._extract_feature(x_dyn_seq, 'wspd10')
        inversion = self._extract_feature(x_dyn_seq, 'inversion')
        sw_rad = self._extract_feature(x_dyn_seq, 'sw_rad')
        lcc = self._extract_feature(x_dyn_seq, 'lcc')
        zenith = self._extract_feature(x_dyn_seq, 'zenith')
        rh925 = self._extract_feature(x_dyn_seq, 'rh925')
        
        rh_clamp = torch.clamp(rh2m, 0, 100) / 100.0
        dpd_weight = torch.sigmoid(-self.dpd_scale * (dpd - self.dpd_shift))
        f1 = torch.clamp(rh_clamp * dpd_weight, 0, 1)
        
        wspd_clamp = F.softplus(wspd)
        sigma = F.softplus(self.wspd_sigma) + 0.1
        f2 = torch.exp(-0.5 * ((wspd_clamp - self.optimal_wspd) / sigma) ** 2)
        
        wspd_clamp_stab = F.softplus(wspd) + 0.1
        ri = inversion / (wspd_clamp_stab ** 2 + 0.1)
        f3 = torch.tanh(ri / self.stab_scale)
        
        is_night = torch.sigmoid((zenith - 90.0) * 10.0)
        lcc_clamp = torch.clamp(lcc, 0, 1)
        lcc_denom = F.softplus(self.lcc_threshold) + 1e-4
        clear_sky = torch.clamp(1.0 - lcc_clamp / lcc_denom, 0, 1)
        sw_linear = torch.clamp(torch.expm1(sw_rad), min=0.0)
        rad_denom = F.softplus(self.rad_threshold) + 1.0
        rad_intensity = 1.0 - torch.clamp(sw_linear / rad_denom, 0, 1)
        f4 = is_night * clear_sky * rad_intensity
        
        f5 = torch.tanh((rh2m - rh925) / 50.0)
        
        physics_seq = torch.stack([f1, f2, f3, f4, f5], dim=2)
        if torch.isnan(physics_seq).any():
            physics_seq = torch.nan_to_num(physics_seq, nan=0.0)
        return physics_seq

class SimpleLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    def forward(self, x):
        return self.net(x)

class GRUWithAttentionEncoder(nn.Module):
    def __init__(self, n_vars, hidden_dim, n_steps=None, dropout=0.2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_vars, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, 2, batch_first=True, bidirectional=True, dropout=dropout)
        gru_out_dim = hidden_dim * 2
        self.attention_net = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.out_proj = nn.Sequential(nn.Linear(gru_out_dim, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, x):
        x_emb = self.embed(x)
        output, _ = self.gru(x_emb)
        attn_scores = self.attention_net(output)
        attn_weights = F.softmax(attn_scores, dim=1)
        context_vector = torch.sum(output * attn_weights, dim=1)
        return self.out_proj(context_vector)

class ImprovedDualStreamPMSTNet(nn.Module):
    def __init__(self, dyn_vars_count=25, window_size=12, 
                 static_cont_dim=5, veg_num_classes=21, veg_emb_dim=16,
                 hidden_dim=512, num_classes=3, extra_feat_dim=0):
        super().__init__()
        self.dyn_vars = dyn_vars_count
        self.window = window_size
        self.static_cont_dim = static_cont_dim
        self.extra_feat_dim = extra_feat_dim
        
        self.fog_diagnostics = LearnableFogDiagnosticFeatures(window_size, dyn_vars_count)
        self.veg_embedding = nn.Embedding(veg_num_classes, veg_emb_dim)
        
        self.physics_encoder = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=1), 
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),
            nn.Linear(128, hidden_dim // 4)
        )
        
        total_static_dim = static_cont_dim + veg_emb_dim
        self.static_encoder = SimpleLinearLayer(total_static_dim, hidden_dim // 2)
        
        if extra_feat_dim > 0:
            self.extra_feat_encoder = nn.Sequential(
                nn.Linear(extra_feat_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU()
            )
            extra_h_dim = hidden_dim // 2
        else:
            self.extra_feat_encoder = nn.Identity()
            extra_h_dim = 0

        self.temporal_vars_indices = [2, 3, 4, 5, 6, 7]
        self.temporal_stream = GRUWithAttentionEncoder(
            n_vars=len(self.temporal_vars_indices), 
            hidden_dim=hidden_dim,
            n_steps=window_size
        )
        
        fusion_input_dim = hidden_dim + (hidden_dim // 2) + (hidden_dim // 4) + extra_h_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            SimpleLinearLayer(hidden_dim, hidden_dim)
        )
        
        self.fog_specific_features = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )
        
        self.low_vis_detector = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.fine_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self._init_bias()
    
    def _init_bias(self):
        self.low_vis_detector[-1].bias.data.fill_(-0.5)
        self.fine_classifier[-1].bias.data = torch.tensor([-0.5, -0.5, 1.0])
        
    def forward(self, x):
        split_dyn = self.dyn_vars * self.window
        split_static = split_dyn + self.static_cont_dim
        split_extra = split_static + 1
        
        x_dyn_flat = x[:, :split_dyn]
        x_stat_cont = x[:, split_dyn:split_static]
        
        raw_veg = x[:, split_static]
        raw_veg = torch.nan_to_num(raw_veg, nan=0.0)
        x_veg_id = raw_veg.long()
        x_veg_id = torch.clamp(x_veg_id, 0, self.veg_embedding.num_embeddings - 1)
        
        x_extra = x[:, split_extra:] if self.extra_feat_dim > 0 else None
        
        x_dyn_seq = x_dyn_flat.view(-1, self.window, self.dyn_vars)
        
        physics_seq = self.fog_diagnostics(x_dyn_seq).permute(0, 2, 1)
        physics_feat = self.physics_encoder(physics_seq)
        
        x_temporal = x_dyn_seq[:, :, self.temporal_vars_indices]
        temporal_feat = self.temporal_stream(x_temporal)
        
        veg_vec = self.veg_embedding(x_veg_id)
        x_static_full = torch.cat([x_stat_cont, veg_vec], dim=1)
        static_feat = self.static_encoder(x_static_full)
        
        if x_extra is not None:
            extra_feat = self.extra_feat_encoder(x_extra)
            combined_feat = torch.cat([temporal_feat, static_feat, physics_feat, extra_feat], dim=1)
        else:
            combined_feat = torch.cat([temporal_feat, static_feat, physics_feat], dim=1)
            
        embedding = self.fusion_layer(combined_feat)
        
        fog_feat = self.fog_specific_features(embedding)
        low_vis_input = torch.cat([embedding, fog_feat], dim=1)
        low_vis_logit = self.low_vis_detector(low_vis_input)
        
        fine_logits = self.fine_classifier(embedding)
        reg_out = self.reg_head(embedding)
        
        return fine_logits, reg_out, low_vis_logit, fine_logits

# ==========================================
# 6. 综合评估指标
# ==========================================
class ComprehensiveMetrics:
    def __init__(self, config):
        self.config = config
        self.min_fog_precision = config['MIN_FOG_PRECISION']
        self.min_fog_recall = config['MIN_FOG_RECALL']
        self.min_clear_acc = config['MIN_CLEAR_ACC']
        
    def calculate_ts_score(self, y_true, y_pred, target_class):
        y_t = (y_true == target_class).astype(int)
        y_p = (y_pred == target_class).astype(int)
        tp = ((y_t == 1) & (y_p == 1)).sum()
        den = (y_t == 1).sum() + (y_p == 1).sum() - tp
        return tp / (den + 1e-6)

    def calculate_metrics(self, y_true, y_pred, cls):
        y_t = (y_true == cls).astype(int)
        y_p = (y_pred == cls).astype(int)
        tp = ((y_t == 1) & (y_p == 1)).sum()
        fp = ((y_t == 0) & (y_p == 1)).sum()
        fn = ((y_t == 1) & (y_p == 0)).sum()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f2 = 5 * precision * recall / (4 * precision + recall + 1e-6)
        ts = tp / (tp + fp + fn + 1e-6)
        
        return {'p': precision, 'r': recall, 'f2': f2, 'ts': ts}

    def search_optimal_thresholds(self, probs, targets):
        fog_ths = np.arange(self.config['THRESHOLD_FOG_MIN'], self.config['THRESHOLD_FOG_MAX'], self.config['THRESHOLD_FOG_STEP'])
        
        best_th = 0.5
        best_score = -1
        best_metrics = None
        
        mist_th = 0.5 
        
        for f_th in fog_ths:
            preds = np.full(len(targets), 2)
            mask_fog = probs[:, 0] > f_th
            preds[mask_fog] = 0
            mask_mist = (probs[:, 1] > mist_th) & (~mask_fog)
            preds[mask_mist] = 1
            
            m_clear = self.calculate_metrics(targets, preds, 2)
            m_fog = self.calculate_metrics(targets, preds, 0)
            
            if m_clear['r'] < self.min_clear_acc: continue
            if m_fog['p'] < self.min_fog_precision: continue
            
            score = m_fog['r']
            
            if score > best_score:
                best_score = score
                best_th = f_th
                best_metrics = {'fog': m_fog, 'clear': m_clear}
        
        if best_metrics is None:
            best_th = 0.3
        
        return best_th, 0.5

    def evaluate(self, model, val_loader, device, rank=0):
        model.eval()
        probs_list = []
        targets_list = []
        
        with torch.no_grad():
            for bx, by, _, _ in val_loader:
                bx = bx.to(device)
                logits, _, _, _ = model(bx)
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs.cpu().numpy())
                targets_list.append(by.numpy())
        
        probs = np.vstack(probs_list)
        targets = np.concatenate(targets_list)
        
        if rank == 0:
            f_th, m_th = self.search_optimal_thresholds(probs, targets)
            
            preds = np.full(len(targets), 2)
            mask_fog = probs[:, 0] > f_th
            preds[mask_fog] = 0
            mask_mist = (probs[:, 1] > m_th) & (~mask_fog)
            preds[mask_mist] = 1
            
            m0 = self.calculate_metrics(targets, preds, 0)
            m1 = self.calculate_metrics(targets, preds, 1)
            m2 = self.calculate_metrics(targets, preds, 2)
            
            acc = accuracy_score(targets, preds)
            
            composite = (m0['r'] * 0.5) + (m0['ts'] * 0.3) + (m1['ts'] * 0.1) + (m2['r'] * 0.1)
            
            print(f"\n[Eval] Fog Th: {f_th:.2f} | Acc: {acc:.4f}")
            print(f"  FOG : R={m0['r']:.4f}, P={m0['p']:.4f}, TS={m0['ts']:.4f}")
            print(f"  MIST: R={m1['r']:.4f}, P={m1['p']:.4f}, TS={m1['ts']:.4f}")
            print(f"  CLR : R={m2['r']:.4f}, P={m2['p']:.4f}")
            
            return composite, {'fog': m0, 'mist': m1, 'clear': m2}
        
        return 0.0, {}

# ==========================================
# 7. 数据集
# ==========================================

class PMSTDatasetWithFE(Dataset):
    def __init__(self, X_path, y_cls, y_reg, y_raw, scaler=None, apply_fe=True, window_size=12):
        self.X = np.load(X_path, mmap_mode='r')
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.as_tensor(y_reg, dtype=torch.float32)
        self.y_raw = torch.as_tensor(y_raw, dtype=torch.float32)
        
        self.apply_fe = apply_fe
        if apply_fe:
            self.fe_engineer = FogFeatureEngineer(window_size=window_size)
            
        self.scaler = scaler

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        x = row.astype(np.float32)
        
        if self.apply_fe:
            x_enhanced = self.fe_engineer.transform(x.reshape(1, -1))[0]
            x = x_enhanced
            
        if self.scaler is not None:
            x = (x - self.scaler.center_) / self.scaler.scale_
            
        return torch.from_numpy(x).float(), self.y_cls[idx], self.y_reg[idx], self.y_raw[idx]

def load_data(data_dir, rank=0, device=None, reuse_scaler=None):
    train_path = os.path.join(data_dir, 'X_train.npy')
    val_path = os.path.join(data_dir, 'X_val.npy')
    y_train_path = os.path.join(data_dir, 'y_train.npy')
    y_val_path = os.path.join(data_dir, 'y_val.npy')
    
    local_train_X = copy_to_local(train_path, rank, 1)
    local_val_X = copy_to_local(val_path, rank, 1)
    local_train_y = copy_to_local(y_train_path, rank, 1)
    local_val_y = copy_to_local(y_val_path, rank, 1)
    
    y_tr_raw = np.load(local_train_y)
    y_val_raw = np.load(local_val_y)
    
    if np.max(y_tr_raw) < 100: y_tr_raw *= 1000.0
    if np.max(y_val_raw) < 100: y_val_raw *= 1000.0
    
    def to_cls(y):
        c = np.zeros_like(y, dtype=int)
        c[y >= 500] = 1
        c[y >= 1000] = 2
        return c
    
    y_tr_cls = to_cls(y_tr_raw)
    y_val_cls = to_cls(y_val_raw)
    y_tr_log = np.log1p(y_tr_raw)
    y_val_log = np.log1p(y_val_raw)
    
    scaler = reuse_scaler
    if scaler is None and rank == 0:
        print("[Data] Fitting Scaler on Enhanced Features...", flush=True)
        X_temp = np.load(local_train_X, mmap_mode='r')
        indices = np.random.choice(len(X_temp), min(20000, len(X_temp)), replace=False)
        indices.sort()
        X_sub = X_temp[indices].astype(np.float32)
        
        fe = FogFeatureEngineer(window_size=CONFIG['WINDOW_SIZE'])
        X_sub_enhanced = fe.transform(X_sub)
        
        scaler = RobustScaler()
        scaler.fit(X_sub_enhanced)
        print(f"[Data] Feature dim: {X_sub.shape[1]} -> {X_sub_enhanced.shape[1]}")
        del X_temp, X_sub, X_sub_enhanced

    train_ds = PMSTDatasetWithFE(local_train_X, y_tr_cls, y_tr_log, y_tr_raw, scaler, True, CONFIG['WINDOW_SIZE'])
    val_ds = PMSTDatasetWithFE(local_val_X, y_val_cls, y_val_log, y_val_raw, scaler, True, CONFIG['WINDOW_SIZE'])
    
    return train_ds, val_ds, scaler

# ==========================================
# 8. ====== [关键修复] 训练流程 ======
# ==========================================

def train_one_stage(stage_name, model, train_loader, val_loader, optimizer, 
                    criterions, scaler_amp, device, rank, world_size, total_steps, config):
    """
    修复要点：
    1. 移除训练循环内的所有 barrier（死锁根源）
    2. 正确处理 DDP + GradAccum
    3. 添加异常处理和超时保护
    4. 确保所有rank同步验证
    """
    
    dual_loss_fn, reg_loss_fn = criterions
    evaluator = ComprehensiveMetrics(config)
    best_score = -1.0
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[g['lr'] for g in optimizer.param_groups],
        total_steps=total_steps,
        pct_start=config['S2_PCT_START'] if stage_name=='S2' else 0.3
    )
    
    # ====== [修复1] 冻结逻辑在训练开始前执行，不在循环中 ======
    freeze_steps = config.get('S2_FREEZE_STEPS', 0) if stage_name == 'S2' else 0
    if freeze_steps > 0 and stage_name == 'S2':
        if rank == 0: 
            print(f"[{stage_name}] Freezing Backbone for first {freeze_steps} steps...", flush=True)
        
        # 冻结backbone参数
        for name, param in model.module.named_parameters():
            if 'detector' not in name and 'classifier' not in name and 'reg_head' not in name:
                param.requires_grad = False
    
    model.train()
    step = 0
    
    # ====== [修复2] 使用常规迭代器，避免无限循环 ======
    while step < total_steps:
        for batch_data in train_loader:
            step += 1
            if step > total_steps:
                break
            
            # ====== [修复3] 解冻逻辑 - 不使用barrier ======
            if freeze_steps > 0 and step == freeze_steps + 1 and stage_name == 'S2':
                if rank == 0:
                    print(f"\n[{stage_name}] Unfreezing Backbone at step {step}!", flush=True)
                
                for param in model.module.parameters():
                    param.requires_grad = True
                
                # 无需barrier，各rank独立执行
            
            # ====== [修复4] 数据获取异常处理 ======
            try:
                bx, by, blog, braw = batch_data
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                blog = blog.to(device, non_blocking=True)
                braw = braw.to(device, non_blocking=True)
            except Exception as e:
                if rank == 0:
                    print(f"\n[ERROR] Data loading failed at step {step}: {e}", flush=True)
                continue
            
            # ====== [修复5] 正确的 DDP + GradAccum ======
            grad_accum = config[f'{stage_name}_GRAD_ACCUM']
            is_accumulating = (step % grad_accum != 0)
            
            # 在accumulation期间禁用DDP同步
            if is_accumulating and world_size > 1:
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()
            
            with context:
                with autocast():
                    fine_logits, reg_out, low_vis_logit, _ = model(bx)
                    
                    loss_cls, loss_dict = dual_loss_fn(fine_logits, low_vis_logit, fine_logits, by)
                    loss_reg = reg_loss_fn(reg_out, blog, braw)
                    
                    loss = loss_cls + config['REG_LOSS_ALPHA'] * loss_reg
                    loss = loss / grad_accum
                
                scaler_amp.scale(loss).backward()
            
            # 只在真正要更新参数时才step
            if not is_accumulating:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['GRAD_CLIP_NORM'])
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad()
                scheduler.step()
            
            # ====== [修复6] 日志输出（无需同步）======
            if rank == 0 and step % 100 == 0:
                lr = optimizer.param_groups[-1]['lr']
                print(f"\r[{stage_name}] Step {step}/{total_steps} | LR: {lr:.2e} | "
                      f"Loss: {loss.item()*grad_accum:.4f} | Fog+: {loss_dict['fog_boost']:.4f}", 
                      end="", flush=True)
            
            # ====== [修复7] 验证逻辑 - 先同步再验证 ======
            if step % config[f'{stage_name}_VAL_INTERVAL'] == 0:
                # 所有rank都进入验证，避免死锁
                if world_size > 1 and dist.is_initialized():
                    dist.barrier()  # 唯一的barrier：确保所有rank同时开始验证
                
                if rank == 0:
                    print(f"\n[{stage_name}] Running validation at step {step}...", flush=True)
                
                try:
                    score, metrics = evaluator.evaluate(model, val_loader, device, rank)
                    if rank == 0:
                        if score > best_score:
                            best_score = score
                            save_path = os.path.join(config['BASE_PATH'], 
                                                    f"model/best_{stage_name}_{config['EXPERIMENT_ID']}.pth")
                            torch.save(model.module.state_dict(), save_path)
                            print(f" -> Best Model Saved! (Composite: {score:.4f})", flush=True)
                except Exception as e:
                    if rank == 0:
                        print(f"\n[WARNING] Validation failed: {e}", flush=True)
                
                model.train()
                
                # 验证后同步
                if world_size > 1 and dist.is_initialized():
                    dist.barrier()

# ==========================================
# 9. Main
# ==========================================

def main():
    local_rank, global_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    scaler_amp = GradScaler()
    
    try:
        if global_rank == 0:
            os.makedirs(os.path.join(CONFIG['BASE_PATH'], "model"), exist_ok=True)
            print(f"Start Exp: {CONFIG['EXPERIMENT_ID']}", flush=True)

        # --- Stage 1: Pretraining (ERA5) ---
        scaler_s1 = None
        if not CONFIG['SKIP_STAGE1']:
            if global_rank == 0: print("\n[STAGE 1] Loading ERA5 Data...", flush=True)
            ds_tr, ds_val, scaler_s1 = load_data(CONFIG['S1_DATA_DIR'], global_rank)

            sampler = StratifiedBalancedSampler(
                ds_tr, CONFIG['S1_BATCH_SIZE'], 
                fog_ratio=CONFIG['S1_FOG_RATIO'], mist_ratio=CONFIG['S1_MIST_RATIO'],
                rank=global_rank, world_size=world_size,
                epoch_length=CONFIG['S1_TOTAL_STEPS'] // world_size + 100  # 确保足够长度
            )
            loader_tr = DataLoader(ds_tr, batch_sampler=sampler, num_workers=CONFIG['NUM_WORKERS'], 
                                  pin_memory=True, persistent_workers=True)
            loader_val = DataLoader(ds_val, batch_size=CONFIG['S1_BATCH_SIZE'], shuffle=False, 
                                   num_workers=CONFIG['NUM_WORKERS'])

            model = ImprovedDualStreamPMSTNet(
                window_size=CONFIG['WINDOW_SIZE'],
                extra_feat_dim=CONFIG['FE_EXTRA_DIMS']
            ).to(device)
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

            criterion_cls = DualBranchLoss(
                use_logit_adj=False,
                fine_class_weight=[
                    CONFIG['FINE_CLASS_WEIGHT_FOG'], 
                    CONFIG['FINE_CLASS_WEIGHT_MIST'], 
                    CONFIG['FINE_CLASS_WEIGHT_CLEAR']
                ],
                asym_gamma_neg=CONFIG['ASYM_GAMMA_NEG'],
                asym_gamma_pos=CONFIG['ASYM_GAMMA_POS']
            ).to(device)
            criterion_reg = PhysicsConstrainedRegLoss(alpha=1.0).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=CONFIG['S1_LR_BACKBONE'], 
                                   weight_decay=CONFIG['S1_WEIGHT_DECAY'])

            if global_rank == 0: print("[STAGE 1] Start Training...", flush=True)
            train_one_stage('S1', model, loader_tr, loader_val, optimizer, (criterion_cls, criterion_reg),
                            scaler_amp, device, global_rank, world_size, CONFIG['S1_TOTAL_STEPS'], CONFIG)

            del ds_tr, ds_val, loader_tr, loader_val
            gc.collect()

        # --- Stage 2: Fine-tuning (Forecast) ---
        if global_rank == 0: 
            print("\n[STAGE 2] Loading Forecast Data...", flush=True)

        ds_tr_s2, ds_val_s2, _ = load_data(CONFIG['S2_DATA_DIR'], global_rank, reuse_scaler=scaler_s1)

        sampler_s2 = StratifiedBalancedSampler(
            ds_tr_s2, CONFIG['S2_BATCH_SIZE'], 
            fog_ratio=CONFIG['S2_FOG_RATIO'], mist_ratio=CONFIG['S2_MIST_RATIO'],
            rank=global_rank, world_size=world_size,
            epoch_length=CONFIG['S2_TOTAL_STEPS'] // world_size + 100
        )
        loader_tr_s2 = DataLoader(ds_tr_s2, batch_sampler=sampler_s2, num_workers=CONFIG['NUM_WORKERS'], 
                                 pin_memory=True, persistent_workers=True)
        loader_val_s2 = DataLoader(ds_val_s2, batch_size=CONFIG['S2_BATCH_SIZE'], shuffle=False, 
                                   num_workers=CONFIG['NUM_WORKERS'])

        # 正确处理模型状态转换
        if CONFIG['SKIP_STAGE1']:
            if global_rank == 0: print("[STAGE 2] Initializing new model...", flush=True)
            model_s2 = ImprovedDualStreamPMSTNet(
                window_size=CONFIG['WINDOW_SIZE'],
                extra_feat_dim=CONFIG['FE_EXTRA_DIMS']
            ).to(device)
        else:
            if global_rank == 0: print("[STAGE 2] Loading weights from Stage 1...", flush=True)
            state_dict_s1 = model.module.state_dict()
            del model
            torch.cuda.empty_cache()
            if world_size > 1 and dist.is_initialized():
                dist.barrier()

            model_s2 = ImprovedDualStreamPMSTNet(
                window_size=CONFIG['WINDOW_SIZE'],
                extra_feat_dim=CONFIG['FE_EXTRA_DIMS']
            ).to(device)
            model_s2.load_state_dict(state_dict_s1)
            if global_rank == 0: print("[STAGE 2] Stage 1 weights loaded successfully!", flush=True)

        model_s2 = DDP(model_s2, device_ids=[local_rank], find_unused_parameters=False)

        # 参数分组
        head_params = []
        backbone_params = []
        for name, param in model_s2.module.named_parameters():
            if 'detector' in name or 'classifier' in name or 'reg_head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        if global_rank == 0:
            print(f"[STAGE 2] Param groups: Backbone={len(backbone_params)}, Head={len(head_params)}", flush=True)

        optimizer_s2 = optim.AdamW([
            {'params': backbone_params, 'lr': CONFIG['S2_MAX_LR_BACKBONE']},
            {'params': head_params, 'lr': CONFIG['S2_MAX_LR_HEAD']}
        ], weight_decay=CONFIG['S2_WEIGHT_DECAY'])

        # Loss Functions
        criterion_cls_s2 = DualBranchLoss(
            use_logit_adj=False,
            fine_class_weight=[
                CONFIG['FINE_CLASS_WEIGHT_FOG'], 
                CONFIG['FINE_CLASS_WEIGHT_MIST'], 
                CONFIG['FINE_CLASS_WEIGHT_CLEAR']
            ],
            asym_gamma_neg=CONFIG['ASYM_GAMMA_NEG'],
            asym_gamma_pos=CONFIG['ASYM_GAMMA_POS']
        ).to(device)
        criterion_reg_s2 = PhysicsConstrainedRegLoss(alpha=CONFIG['REG_LOSS_ALPHA']).to(device)

        if global_rank == 0: print("[STAGE 2] Start Fine-tuning...", flush=True)
        train_one_stage('S2', model_s2, loader_tr_s2, loader_val_s2, optimizer_s2, 
                        (criterion_cls_s2, criterion_reg_s2),
                        scaler_amp, device, global_rank, world_size, CONFIG['S2_TOTAL_STEPS'], CONFIG)

        if world_size > 1:
            dist.destroy_process_group()
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Rank {global_rank}: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        cleanup_temp_files()
        if world_size > 1 and dist.is_initialized():
            dist.barrier()

if __name__ == "__main__":
    main()
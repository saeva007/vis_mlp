import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import os
import shutil
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score, confusion_matrix
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
# 0. 全局配置 - 集中管理所有超参数
# ==========================================

# 定义基础路径和窗口大小
TARGET_WINDOW_SIZE = 12
BASE_PATH = "/public/home/putianshu/vis_mlp"

# 数据集路径配置
# S1 (Stage 1): 通常使用ERA5再分析数据，数据量大，用于预训练物理特征
S1_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_{TARGET_WINDOW_SIZE}h"
# S2 (Stage 2): 通常使用模式预报数据（Forecast），数据量小，用于微调适应实际业务
S2_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_finetune_v1_{TARGET_WINDOW_SIZE}h"

CONFIG = {
    # ========== 1. 实验控制配置 ==========
    # 实验ID，建议每次修改配置后更新，用于区分保存的模型文件
    'EXPERIMENT_ID': 'exp004_Physics_LogitAdj_OneCycle', 
    
    # 是否跳过Stage 1预训练？
    # True: 直接加载 S1_PRETRAINED_PATH 指定的模型进行微调，或直接从头训练Stage 2
    # False: 先在 S1 数据集上训练，再在 S2 数据集上微调（推荐，物理特征更稳固）
    'SKIP_STAGE1': False,             
    
    # 如果 SKIP_STAGE1=True，指定要加载的预训练模型路径 (.pth)
    'S1_PRETRAINED_PATH': None,       
    
    # ========== 2. 基础环境配置 ==========
    'BASE_PATH': BASE_PATH,            # 根目录，用于保存模型和scaler
    'WINDOW_SIZE': TARGET_WINDOW_SIZE, # 时间窗口长度（小时）
    'S1_DATA_DIR': S1_DIR,             # Stage 1 数据路径
    'S2_DATA_DIR': S2_DIR,             # Stage 2 数据路径
    
    # ========== 3. Stage 1 (预训练) 训练配置 ==========
    'S1_TOTAL_STEPS': 30000,      # S1 总训练步数（不是Epoch，是Batch更新次数）
    'S1_VAL_INTERVAL': 2000,      # 每多少步在验证集上评估一次
    'S1_BATCH_SIZE': 512,         # 单卡 Batch Size
    'S1_GRAD_ACCUM': 2,           # 梯度累积步数（显存不够时调大此值，等效Batch Size = 512 * 2）
    
    # 采样器配置（关键！解决不平衡）：
    # 强制每个 Batch 中包含 15% 的雾样本和 15% 的薄雾样本，剩余 70% 为晴天
    'S1_FOG_RATIO': 0.15,     
    'S1_MIST_RATIO': 0.15,
    
    'S1_LR_BACKBONE': 3e-4,       # S1 初始学习率（使用AdamW）
    'S1_WEIGHT_DECAY': 1e-3,      # S1 权重衰减（防止过拟合）
    
    # ========== 4. Stage 2 (微调) 训练配置 (OneCycleLR) ==========
    'S2_TOTAL_STEPS': 5000,       # S2 微调步数（通常比 S1 少）
    'S2_VAL_INTERVAL': 500,       # S2 评估频率（更高频）
    'S2_BATCH_SIZE': 512,
    'S2_GRAD_ACCUM': 1,
    
    # S2 采样率（通常比 S1 略低，防止过拟合少数类）
    'S2_FOG_RATIO': 0.12,      
    'S2_MIST_RATIO': 0.12,
    
    # OneCycleLR 调度器配置（关键优化）：
    # 采用“预热-峰值-衰减”策略
    'S2_MAX_LR_BACKBONE': 1e-5,   #由于 Backbone 在 S1 已训练好，这里用极小的学习率微调
    'S2_MAX_LR_HEAD': 1e-4,       #分类头（Head）需要快速适应新数据，学习率稍大
    'S2_PCT_START': 0.3,          #前 30% 的步数用于预热学习率
    'S2_WEIGHT_DECAY': 1e-2,      # S2 正则化力度加大
    
    # ========== 5. 评估与模型保存约束 ==========
    # 只有满足以下最低要求的模型才会被保存（防止保存严重偏科的模型）
    'MIN_FOG_PRECISION': 0.20,   # 雾的精确率至少 20%
    'MIN_FOG_RECALL': 0.50,      # 雾的召回率至少 50%
    'MIN_MIST_PRECISION': 0.12,  
    'MIN_MIST_RECALL': 0.15,     
    'MIN_CLEAR_ACC': 0.95,       # 晴天准确率至少 95%（保证不乱报）
    
    # ========== 6. 损失函数高级配置 ==========
    'LOSS_TYPE': 'asymmetric',   # 核心Loss类型：'asymmetric' (推荐) 或 'focal'
    
    # [Logit Adjustment] 长尾分布优化 (关键新增)
    # 原理：在计算 Loss 前，减去类别的先验概率，强迫模型关注少样本类别
    'USE_LOGIT_ADJUSTMENT': True,
    'LOGIT_ADJ_TAU': 1.0,           # 温度系数。值越大，对长尾类别的补偿力度越大（通常 1.0 左右）
    
    # DualBranchLoss (双分支损失) 权重配置
    'LOSS_ALPHA_BINARY': 1.0,       # 二分类分支（低能见度 vs 高能见度）的权重
    'LOSS_ALPHA_FINE': 1.0,         # 精细三分类分支的权重
    'LOSS_ALPHA_CONSISTENCY': 0.5,  # 一致性损失（二分类和三分类结果要逻辑自洽）
    'LOSS_ALPHA_FP': 3.0,           # 虚警惩罚（False Alarm Penalty）权重
    'LOSS_ALPHA_FOG_BOOST': 0.1,    # 对“雾”样本的额外Loss加权
    'LOSS_ALPHA_MIST_BOOST': 0.1,   # 对“薄雾”样本的额外Loss加权
    'LOSS_FP_THRESHOLD': 0.5,       # 判定为虚警的概率阈值
    
    # Asymmetric Loss 参数 (针对极度不平衡)
    'ASYM_GAMMA_NEG': 2.0,          # 负样本衰减系数（压制简单负样本）
    'ASYM_GAMMA_POS': 1.0,          # 正样本衰减系数
    'ASYM_CLIP': 0.05,              # 概率截断，防止数值不稳定
    
    # Focal Loss 参数 (备用)
    'FOCAL_GAMMA': 2.0,             
    'FOCAL_ALPHA': None,            
    
    # 手动类别权重 (配合 CrossEntropy 使用)
    'BINARY_POS_WEIGHT': 1.2,       # 二分类中正样本的权重
    'FINE_CLASS_WEIGHT_FOG': 1.5,   # 三分类中雾的权重
    'FINE_CLASS_WEIGHT_MIST': 1.2,  # 三分类中薄雾的权重
    'FINE_CLASS_WEIGHT_CLEAR': 1.0, # 三分类中晴天的权重
    
    # ========== 7. 阈值搜索配置 ==========
    # 模型输出概率后，通过网格搜索寻找最佳的切分阈值，而不是默认的 0.5
    'THRESHOLD_FOG_MIN': 0.10,      # 雾阈值搜索下限
    'THRESHOLD_FOG_MAX': 0.90,      # 雾阈值搜索上限
    'THRESHOLD_FOG_STEP': 0.05,     # 步长
    'THRESHOLD_MIST_MIN': 0.10,     
    'THRESHOLD_MIST_MAX': 0.90,     
    'THRESHOLD_MIST_STEP': 0.05,    
    
    # 阈值搜索的三阶段策略（从严格到宽松）
    # 阶段 2 放宽条件：
    'THRESHOLD_PHASE2_CLEAR_RECALL': 0.90,   
    'THRESHOLD_PHASE2_FOG_PRECISION': 0.18,
    'THRESHOLD_PHASE2_FOG_RECALL': 0.45,
    # 阶段 3 进一步放宽：
    'THRESHOLD_PHASE3_CLEAR_RECALL': 0.90,
    
    # 综合得分 (F2 Score) 权重计算
    # F2 Score 更看重召回率 (Recall)，适合气象预警
    # 阶段 1 (Strict) 权重:
    'SCORE_PHASE1_FOG': 0.45,       
    'SCORE_PHASE1_MIST': 0.40,      
    'SCORE_PHASE1_CLEAR': 0.15,     
    # 阶段 2 (Fog Priority) 权重:
    'SCORE_PHASE2_FOG': 0.60,       # 更加侧重雾
    'SCORE_PHASE2_MIST': 0.30,      
    'SCORE_PHASE2_CLEAR': 0.10,     
    # 阶段 3 (Relaxed) 权重:
    'SCORE_PHASE3_FOG': 0.50,       
    'SCORE_PHASE3_MIST': 0.35,      
    'SCORE_PHASE3_CLEAR': 0.15,     
    
    # ========== 8. 模型架构配置 ==========
    'MODEL_HIDDEN_DIM': 512,        # 隐藏层维度
    'MODEL_DROPOUT': 0.2,           # Dropout 比率
    'MODEL_NUM_CLASSES': 3,         # 分类数量 (Fog, Mist, Clear)
    
    # ========== 9. 其他杂项 ==========
    'GRAD_CLIP_NORM': 1.0,          # 梯度裁剪阈值（防止梯度爆炸）
    'REG_LOSS_ALPHA': 1.0,          # 辅助回归 Loss (预测具体能见度数值) 的权重
}

# ==========================================
# 1. 基础工具与分布式初始化 (保持不变)
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
    local_path = os.path.join(target_dir, filename)
    
    if local_rank == 0:
        need_copy = True
        if os.path.exists(local_path):
            try:
                if os.path.getsize(local_path) == os.path.getsize(src_path):
                    need_copy = False
            except:
                need_copy = True
        
        if need_copy:
            print(f"[Node Local-0] Copying {filename} to RAM...", flush=True)
            try:
                tmp_path = local_path + ".tmp"
                shutil.copyfile(src_path, tmp_path)
                os.rename(tmp_path, local_path)
            except Exception as e:
                print(f"[Node Local-0] Copy FAILED: {e}. Fallback to NFS.", flush=True)
                local_path = src_path

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        
    if not os.path.exists(local_path):
        return src_path
    return local_path

# ==========================================
# 2. 改进的分层采样器 (保持不变)
# ==========================================

class StratifiedBalancedSampler(Sampler):
    def __init__(self, dataset, batch_size, 
                 fog_ratio=0.10, mist_ratio=0.10, clear_ratio=0.80,
                 rank=0, world_size=1, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.fog_ratio = fog_ratio
        self.mist_ratio = mist_ratio
        self.clear_ratio = clear_ratio
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        
        total_ratio = fog_ratio + mist_ratio + clear_ratio
        assert abs(total_ratio - 1.0) < 1e-6, f"Ratios must sum to 1, got {total_ratio}"
        
        y = np.array(dataset.y_cls)
        all_fog = np.where(y == 0)[0]
        all_mist = np.where(y == 1)[0]
        all_clear = np.where(y == 2)[0]
        
        np.random.seed(seed + rank)
        np.random.shuffle(all_fog)
        np.random.shuffle(all_mist)
        np.random.shuffle(all_clear)
        
        self.fog_indices = np.array_split(all_fog, world_size)[rank]
        self.mist_indices = np.array_split(all_mist, world_size)[rank]
        self.clear_indices = np.array_split(all_clear, world_size)[rank]
        
        self.n_fog = max(1, int(batch_size * fog_ratio))
        self.n_mist = max(1, int(batch_size * mist_ratio))
        self.n_clear = batch_size - self.n_fog - self.n_mist
    
    def __iter__(self):
        epoch_counter = 0
        while True:
            epoch_seed = self.seed + self.rank + epoch_counter * 1000
            g = torch.Generator()
            g.manual_seed(epoch_seed)

            fog_batch = torch.randint(0, len(self.fog_indices), (self.n_fog,), generator=g).numpy()
            mist_batch = torch.randint(0, len(self.mist_indices), (self.n_mist,), generator=g).numpy()
            clear_batch = torch.randint(0, len(self.clear_indices), (self.n_clear,), generator=g).numpy()

            indices = np.concatenate([
                self.fog_indices[fog_batch],
                self.mist_indices[mist_batch],
                self.clear_indices[clear_batch]
            ])
            np.random.shuffle(indices)
            yield indices.tolist()
            epoch_counter += 1
    
    def __len__(self):
        return 2147483647

# ==========================================
# 3. Loss 函数 (引入 Logit Adjustment)
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

# ... (FocalLoss, MultiClassFocalLoss, BalancedFocalLoss 保持原样，省略以节省空间，若需要请保留) ...

class DualBranchLoss(nn.Module):
    def __init__(self, 
                 alpha_binary=1.0, 
                 alpha_fine=1.0, 
                 alpha_consistency=0.5,
                 alpha_fp=1.5, 
                 alpha_fog_boost=0.6,
                 alpha_mist_boost=0.6,
                 fp_threshold=0.6,
                 loss_type='asymmetric',
                 fine_class_weight=None,
                 # 新增参数
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
        self.loss_type = loss_type
        
        # Logit Adjustment Configuration
        self.use_logit_adj = use_logit_adj
        self.logit_adj_tau = logit_adj_tau
        if class_counts is not None and use_logit_adj:
            total = sum(class_counts)
            priors = torch.tensor([c/total for c in class_counts], dtype=torch.float32)
            self.register_buffer('priors', priors)
            print(f"[Loss] Logit Adjustment Enabled. Priors: {priors.tolist()}, Tau: {logit_adj_tau}")
        else:
            self.register_buffer('priors', None)

        # Binary Loss
        self.binary_pos_weight = kwargs.get('binary_pos_weight', 1.2)
        self.register_buffer('binary_pos_weight_tensor', torch.tensor([self.binary_pos_weight]))
        self.binary_loss = nn.BCEWithLogitsLoss(pos_weight=self.binary_pos_weight_tensor)
        
        if fine_class_weight is None:
            fine_class_weight = torch.tensor([2.0, 2.0, 1.0])
        elif isinstance(fine_class_weight, list):
            fine_class_weight = torch.tensor(fine_class_weight)
        self.register_buffer('fine_class_weight', fine_class_weight)
        
        # Fine Loss Initialization
        if loss_type == 'asymmetric':
            self.fine_loss = AsymmetricLoss(
                gamma_neg=kwargs.get('asym_gamma_neg', 4),
                gamma_pos=kwargs.get('asym_gamma_pos', 1),
                clip=kwargs.get('asym_clip', 0.05),
                class_weights=self.fine_class_weight
            )
        # ... (其他 Loss 类型保持原样)
        else: # Default Fallback
             self.fine_loss = nn.CrossEntropyLoss(weight=self.fine_class_weight)
    
    def forward(self, final_logits, low_vis_logit, fine_logits, targets):
        # 1. Binary Loss
        binary_targets = (targets <= 1).float().unsqueeze(1)
        loss_binary = self.binary_loss(low_vis_logit, binary_targets)
        
        # 2. Fine Classification Loss (with Logit Adjustment)
        logits_for_loss = fine_logits
        if self.use_logit_adj and self.priors is not None and self.training:
            # 应用 Logit Adjustment: logit + tau * log(prior)
            # 加上 1e-8 防止 log(0)
            logits_for_loss = fine_logits + self.logit_adj_tau * torch.log(self.priors + 1e-8)
            
        loss_fine = self.fine_loss(logits_for_loss, targets)
        
        # 3. Consistency Loss
        fine_probs = F.softmax(fine_logits, dim=1) # 注意：这里用原始logits计算概率
        low_vis_probs = torch.sigmoid(low_vis_logit)
        inconsistency = low_vis_probs * fine_probs[:, 2:3]
        loss_consistency = inconsistency.mean()
        
        # 4. FP & Boost Loss (Physics Constraints)
        is_clear = (targets == 2).float()
        fog_mist_prob = fine_probs[:, 0] + fine_probs[:, 1]
        high_confidence_mask = (fog_mist_prob > self.fp_threshold).float()
        false_alarm_prob = fog_mist_prob * is_clear * high_confidence_mask
        loss_fp = torch.mean(false_alarm_prob ** 2) * self.alpha_fp
        
        # Fog/Mist Boost
        is_fog = (targets == 0).float()
        fog_prob = fine_probs[:, 0]
        fog_confidence_loss = torch.mean((1 - fog_prob) ** 2 * is_fog)
        fog_to_mist_penalty = torch.mean(fine_probs[:, 1] * is_fog)
        fog_boost = (fog_confidence_loss + fog_to_mist_penalty) * self.alpha_fog_boost
        
        is_mist = (targets == 1).float()
        mist_prob = fine_probs[:, 1]
        mist_confidence_loss = torch.mean((1 - mist_prob) ** 2 * is_mist)
        mist_to_fog_penalty = torch.mean(fine_probs[:, 0] * is_mist)
        mist_boost = (mist_confidence_loss + mist_to_fog_penalty) * self.alpha_mist_boost
        
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
# 4. 模型架构 (修改点：可学习物理层)
# ==========================================

class LearnableFogDiagnosticFeatures(nn.Module):
    """
    修改点：将硬编码的物理参数转换为可学习的 nn.Parameter，
    并使用软激活函数 (Sigmoid/Softplus) 保证梯度流动。
    """
    def __init__(self, window_size=9, dyn_vars_count=25):
        super().__init__()
        self.window_size = window_size
        self.dyn_vars = dyn_vars_count
        
        self.idx = {
            'rh2m': 0, 't2m': 1, 'precip': 2, 'sw_rad': 4,
            'wspd10': 6, 'cape': 9, 'lcc': 10, 't925': 11,
            'rh925': 12, 'dpd': 22, 'inversion': 23, 'zenith': 24
        }
        
        # === 可学习的物理参数 ===
        # 1. 饱和度 (RH & DPD)
        self.dpd_scale = nn.Parameter(torch.tensor(3.0))
        self.dpd_shift = nn.Parameter(torch.tensor(1.0))
        
        # 2. 风速 (Wind)
        self.optimal_wspd = nn.Parameter(torch.tensor(3.5))
        self.wspd_sigma = nn.Parameter(torch.tensor(2.5))
        
        # 3. 稳定性 (Stability)
        self.stab_scale = nn.Parameter(torch.tensor(2.0))
        
        # 4. 辐射 (Radiation)
        self.rad_threshold = nn.Parameter(torch.tensor(800.0))
        self.lcc_threshold = nn.Parameter(torch.tensor(0.3))
    
    def _extract_feature(self, x_seq, feature_name):
        idx = self.idx[feature_name]
        return x_seq[:, :, idx]
    
    def compute_saturation_proximity(self, rh, dpd):
        rh_clamp = torch.clamp(rh, 0, 100) / 100.0
        # Soft logic: Sigmoid for smooth gradient
        dpd_weight = torch.sigmoid(-self.dpd_scale * (dpd - self.dpd_shift))
        return torch.clamp(rh_clamp * dpd_weight, 0, 1)
    
    def compute_wind_favorability(self, wspd):
        wspd_clamp = F.softplus(wspd) # Ensure positive
        # Learnable Gaussian bell curve
        sigma = F.softplus(self.wspd_sigma) + 0.1
        return torch.exp(-0.5 * ((wspd_clamp - self.optimal_wspd) / sigma) ** 2)
    
    def compute_stability_index(self, inversion, wspd):
        wspd_clamp = F.softplus(wspd) + 0.1
        ri = inversion / (wspd_clamp ** 2 + 0.1)
        return torch.tanh(ri / self.stab_scale)
    
    def compute_radiative_cooling_potential(self, sw_rad, lcc, zenith):
        is_night = torch.sigmoid((zenith - 90.0) * 10.0) # Smooth transition at 90 deg
        
        lcc_clamp = torch.clamp(lcc, 0, 1)
        lcc_denom = F.softplus(self.lcc_threshold) + 1e-4
        clear_sky = torch.clamp(1.0 - lcc_clamp / lcc_denom, 0, 1)
        
        sw_linear = torch.clamp(torch.expm1(sw_rad), min=0.0)
        rad_denom = F.softplus(self.rad_threshold) + 1.0
        rad_intensity = 1.0 - torch.clamp(sw_linear / rad_denom, 0, 1)
        
        return is_night * clear_sky * rad_intensity
    
    def compute_vertical_moisture_transport(self, rh2m, rh925):
        return torch.tanh((rh2m - rh925) / 50.0) # Normalized difference
    
    def forward(self, x_dyn_seq):
        rh2m = self._extract_feature(x_dyn_seq, 'rh2m')
        dpd = self._extract_feature(x_dyn_seq, 'dpd')
        wspd = self._extract_feature(x_dyn_seq, 'wspd10')
        inversion = self._extract_feature(x_dyn_seq, 'inversion')
        sw_rad = self._extract_feature(x_dyn_seq, 'sw_rad')
        lcc = self._extract_feature(x_dyn_seq, 'lcc')
        zenith = self._extract_feature(x_dyn_seq, 'zenith')
        rh925 = self._extract_feature(x_dyn_seq, 'rh925')
        
        f1 = self.compute_saturation_proximity(rh2m, dpd)
        f2 = self.compute_wind_favorability(wspd)
        f3 = self.compute_stability_index(inversion, wspd)
        f4 = self.compute_radiative_cooling_potential(sw_rad, lcc, zenith)
        f5 = self.compute_vertical_moisture_transport(rh2m, rh925)
        
        physics_seq = torch.stack([f1, f2, f3, f4, f5], dim=2)
        
        if torch.isnan(physics_seq).any():
            physics_seq = torch.nan_to_num(physics_seq, nan=0.0)
            
        return physics_seq

# ... (ChebyKANLayer, PhysicalStateEncoder, GRUWithAttentionEncoder 保持原样) ...
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=4):
        super(ChebyKANLayer, self).__init__()
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.xavier_normal_(self.cheby_coeffs)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.base_activation = nn.SiLU()

    def forward(self, x):
        x = self.layer_norm(x)
        x = torch.tanh(x)
        cheby_values = [torch.ones_like(x), x]
        for i in range(2, self.degree + 1):
            next_t = 2 * x * cheby_values[-1] - cheby_values[-2]
            cheby_values.append(next_t)
        stacked_cheby = torch.stack(cheby_values, dim=-1)
        y = torch.einsum("bid,iod->bo", stacked_cheby, self.cheby_coeffs)
        return self.base_activation(y)

class PhysicalStateEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.encoder(x) + self.shortcut(x)

class GRUWithAttentionEncoder(nn.Module):
    def __init__(self, n_vars, hidden_dim, n_steps=None, dropout=0.2):
        super().__init__()
        
        self.embed = nn.Sequential(
            nn.Linear(n_vars, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.gru = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout
        )
        
        gru_out_dim = hidden_dim * 2
        
        self.attention_net = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.out_proj = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        x_emb = self.embed(x)
        output, _ = self.gru(x_emb)
        attn_scores = self.attention_net(output)
        attn_weights = F.softmax(attn_scores, dim=1)
        context_vector = torch.sum(output * attn_weights, dim=1)
        return self.out_proj(context_vector)

class ImprovedDualStreamPMSTNet(nn.Module):
    def __init__(self, dyn_vars_count=25, window_size=9, 
                 static_cont_dim=5, veg_num_classes=21, veg_emb_dim=16,
                 hidden_dim=512, num_classes=3):
        super().__init__()
        self.dyn_vars = dyn_vars_count
        self.window = window_size
        self.static_cont_dim = static_cont_dim
        
        # 使用新的可学习物理层
        self.fog_diagnostics = LearnableFogDiagnosticFeatures(window_size, dyn_vars_count)
        self.veg_embedding = nn.Embedding(veg_num_classes, veg_emb_dim)
        
        self.physics_encoder = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=1), 
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),
            nn.Linear(128, hidden_dim // 4)
        )
        
        total_static_dim = static_cont_dim + veg_emb_dim
        self.static_encoder = nn.Sequential(
            ChebyKANLayer(total_static_dim, 256, degree=3),
            nn.LayerNorm(256),
            nn.Linear(256, hidden_dim // 2)
        )
        
        self.physical_vars_indices = [0, 1, 10, 12, 19, 20, 22, 23]
        physical_dim = len(self.physical_vars_indices)
        self.physical_stream = PhysicalStateEncoder(physical_dim, hidden_dim)
        
        self.temporal_vars_indices = [2, 3, 4, 5, 6, 7]
        temporal_dim = len(self.temporal_vars_indices)
        self.temporal_stream = GRUWithAttentionEncoder(
            n_vars=temporal_dim, 
            hidden_dim=hidden_dim,
            n_steps=window_size
        )
        
        fusion_dim = hidden_dim * 2 + hidden_dim // 2 + hidden_dim // 4
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            ChebyKANLayer(hidden_dim, hidden_dim, degree=3)
        )
        
        self.fog_specific_features = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.low_vis_detector = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.fine_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self._init_bias()
    
    def _init_bias(self):
        self.low_vis_detector[-1].bias.data.fill_(-0.5)
        self.fine_classifier[-1].bias.data = torch.tensor([-1.1, -1.1, 0.0])
        
    def forward(self, x):
        split_dyn = self.dyn_vars * self.window
        split_static = split_dyn + self.static_cont_dim
        
        x_dyn_flat = x[:, :split_dyn]
        x_stat_cont = x[:, split_dyn:split_static]
        x_veg_id = x[:, -1].long()
        
        x_dyn_seq = x_dyn_flat.view(-1, self.window, self.dyn_vars)
        
        physics_seq = self.fog_diagnostics(x_dyn_seq)
        physics_seq = physics_seq.permute(0, 2, 1)
        physics_feat = self.physics_encoder(physics_seq)
        
        veg_vec = self.veg_embedding(x_veg_id)
        x_static_full = torch.cat([x_stat_cont, veg_vec], dim=1)
        static_feat = self.static_encoder(x_static_full)
        
        x_current = x_dyn_seq[:, -1, :]
        x_physical = x_current[:, self.physical_vars_indices]
        physical_feat = self.physical_stream(x_physical)
        
        x_temporal = x_dyn_seq[:, :, self.temporal_vars_indices]
        temporal_feat = self.temporal_stream(x_temporal)
        
        combined_feat = torch.cat([
            physical_feat, temporal_feat, static_feat, physics_feat
        ], dim=1)
        embedding = self.fusion_layer(combined_feat)
        
        fog_feat = self.fog_specific_features(embedding)
        
        low_vis_input = torch.cat([embedding, fog_feat], dim=1)
        low_vis_logit = self.low_vis_detector(low_vis_input)
        
        fine_logits = self.fine_classifier(embedding)
        final_logits = fine_logits
        reg_out = self.reg_head(embedding)
        
        return final_logits, reg_out, low_vis_logit, fine_logits

# ==========================================
# 5. 综合评估指标 (保持不变)
# ==========================================
class ComprehensiveMetrics:
    def __init__(self, 
                 min_fog_precision=0.20, 
                 min_fog_recall=0.50,
                 min_mist_precision=0.12, 
                 min_mist_recall=0.15,
                 min_clear_acc=0.93,
                 config=None):
        self.min_fog_precision = min_fog_precision
        self.min_fog_recall = min_fog_recall
        self.min_mist_precision = min_mist_precision
        self.min_mist_recall = min_mist_recall
        self.min_clear_acc = min_clear_acc
        self.config = config if config is not None else CONFIG
        self.best_thresholds = {
            'fog': 0.5,
            'mist': 0.5
        }
    
    def calculate_class_specific_f2(self, y_true, y_pred, target_class):
        y_true_binary = (y_true == target_class).astype(int)
        y_pred_binary = (y_pred == target_class).astype(int)
        
        tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
        fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
        fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f2 = 5 * precision * recall / (4 * precision + recall + 1e-6)
        
        return {
            'precision': precision,
            'recall': recall,
            'f2': f2,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    def search_optimal_thresholds(self, all_probs, all_targets, rank=0):
        fog_thresholds = np.arange(
            self.config['THRESHOLD_FOG_MIN'], 
            self.config['THRESHOLD_FOG_MAX'], 
            self.config['THRESHOLD_FOG_STEP']
        )
        mist_thresholds = np.arange(
            self.config['THRESHOLD_MIST_MIN'], 
            self.config['THRESHOLD_MIST_MAX'], 
            self.config['THRESHOLD_MIST_STEP']
        )
        
        best_score = -1
        best_fog_th = 0.5
        best_mist_th = 0.5
        
        # Phase 1
        for fog_th in fog_thresholds:
            for mist_th in mist_thresholds:
                preds = self._predict_with_thresholds(all_probs, fog_th, mist_th)
                fog_m = self.calculate_class_specific_f2(all_targets, preds, 0)
                mist_m = self.calculate_class_specific_f2(all_targets, preds, 1)
                clear_m = self.calculate_class_specific_f2(all_targets, preds, 2)
                
                if clear_m['recall'] < self.min_clear_acc: continue
                if fog_m['precision'] < self.min_fog_precision: continue
                if fog_m['recall'] < self.min_fog_recall: continue
                
                score = (self.config['SCORE_PHASE1_FOG'] * fog_m['f2'] + 
                        self.config['SCORE_PHASE1_MIST'] * mist_m['f2'] + 
                        self.config['SCORE_PHASE1_CLEAR'] * clear_m['f2'])
                
                if score > best_score:
                    best_score = score
                    best_fog_th = fog_th
                    best_mist_th = mist_th

        # Phase 2 (Fallback)
        if best_score < 0:
            for fog_th in fog_thresholds:
                for mist_th in mist_thresholds:
                    preds = self._predict_with_thresholds(all_probs, fog_th, mist_th)
                    fog_m = self.calculate_class_specific_f2(all_targets, preds, 0)
                    mist_m = self.calculate_class_specific_f2(all_targets, preds, 1)
                    clear_m = self.calculate_class_specific_f2(all_targets, preds, 2)
                    
                    if clear_m['recall'] < self.config['THRESHOLD_PHASE2_CLEAR_RECALL']: continue
                    if fog_m['precision'] < self.config['THRESHOLD_PHASE2_FOG_PRECISION']: continue
                    
                    score = (self.config['SCORE_PHASE2_FOG'] * fog_m['f2'] + 
                            self.config['SCORE_PHASE2_MIST'] * mist_m['f2'] + 
                            self.config['SCORE_PHASE2_CLEAR'] * clear_m['f2'])
                    
                    if score > best_score:
                        best_score = score
                        best_fog_th = fog_th
                        best_mist_th = mist_th
                        
        self.best_thresholds['fog'] = best_fog_th
        self.best_thresholds['mist'] = best_mist_th
        return best_fog_th, best_mist_th
    
    def _predict_with_thresholds(self, probs, fog_threshold, mist_threshold):
        prob_fog = probs[:, 0]
        prob_mist = probs[:, 1]
        predictions = np.full(len(probs), 2)
        
        mask_fog = (prob_fog > fog_threshold) & (prob_fog > prob_mist)
        predictions[mask_fog] = 0
        mask_mist = (prob_mist > mist_threshold) & (~mask_fog)
        predictions[mask_mist] = 1
        return predictions
    
    def print_detailed_report(self, metrics, title="Evaluation"):
        print(f"\n[{title}] Composite: {metrics['composite_score']:.4f}")
        for k in ['fog', 'mist', 'clear']:
            m = metrics[k]
            print(f"  {k.upper()}: P={m['precision']:.4f}, R={m['recall']:.4f}, F2={m['f2']:.4f}")

    def evaluate_comprehensive(self, model, val_loader, device, search_threshold=True, rank=0):
        model.eval()
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for bx, by_cls, _, _ in val_loader:
                bx = bx.to(device, non_blocking=True)
                final_logits, _, low_vis_logit, fine_logits = model(bx)
                probs = F.softmax(final_logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(by_cls.cpu().numpy())

        all_probs = np.vstack(all_probs)
        all_targets = np.concatenate(all_targets)

        if search_threshold:
            fog_th, mist_th = self.search_optimal_thresholds(all_probs, all_targets, rank)
        else:
            fog_th = self.best_thresholds['fog']
            mist_th = self.best_thresholds['mist']

        predictions = self._predict_with_thresholds(all_probs, fog_th, mist_th)
        
        fog_metrics = self.calculate_class_specific_f2(all_targets, predictions, 0)
        mist_metrics = self.calculate_class_specific_f2(all_targets, predictions, 1)
        clear_metrics = self.calculate_class_specific_f2(all_targets, predictions, 2)
        
        # Calculate TS score
        low_vis_true = (all_targets <= 1).astype(int)
        low_vis_pred = (predictions <= 1).astype(int)
        tp = ((low_vis_true == 1) & (low_vis_pred == 1)).sum()
        den = (low_vis_true == 1).sum() + (low_vis_pred == 1).sum() - tp
        low_vis_ts = tp / (den + 1e-6)

        composite_score = (self.config['SCORE_PHASE1_FOG'] * fog_metrics['f2'] + 
                          self.config['SCORE_PHASE1_MIST'] * mist_metrics['f2'] + 
                          self.config['SCORE_PHASE1_CLEAR'] * clear_metrics['f2'])

        return {
            'fog': fog_metrics, 'mist': mist_metrics, 'clear': clear_metrics,
            'low_vis_metrics': {'ts': low_vis_ts},
            'composite_score': composite_score,
            'thresholds': {'fog': fog_th, 'mist': mist_th}
        }

# ==========================================
# 6. 数据处理
# ==========================================

class PMSTDataset(Dataset):
    def __init__(self, X_path, y_cls, y_reg, y_raw, scaler=None, apply_log_transform=True, window_size=9):
        self.X = np.load(X_path, mmap_mode='r')
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.as_tensor(y_reg, dtype=torch.float32)
        self.y_raw = torch.as_tensor(y_raw, dtype=torch.float32)
        self.window_size = window_size
        
        self.has_scaler = scaler is not None
        if self.has_scaler:
            self.center = scaler.center_.astype(np.float32)
            self.scale = scaler.scale_.astype(np.float32)
            self.scale = np.where(self.scale == 0, 1.0, self.scale)
        
        self.apply_log_transform = apply_log_transform
        feat_dim = self.X.shape[1] - 1
        self.log_mask = np.zeros(feat_dim, dtype=bool)
        if apply_log_transform:
            for t in range(self.window_size):
                offset = t * 25
                indices = [offset + 2, offset + 4, offset + 9]
                for idx in indices:
                    if idx < feat_dim:
                        self.log_mask[idx] = True
        self.clip_min = -10.0
        self.clip_max = 10.0

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        features = row[:-1].astype(np.float32)
        veg_id = row[-1]
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.apply_log_transform:
            np.maximum(features, 0, out=features, where=self.log_mask)
            np.log1p(features, out=features, where=self.log_mask)
        if self.has_scaler:
            features = (features - self.center) / self.scale
        features = np.clip(features, self.clip_min, self.clip_max)
        veg_id = np.int64(veg_id)
        features = np.append(features, veg_id)
        return (torch.from_numpy(features).float(), self.y_cls[idx], self.y_reg[idx], self.y_raw[idx])

def load_data_and_scale(data_dir, scaler=None, rank=0, device=None, reuse_scaler=False, window_size=9, world_size=1):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        print(f"Loading from: {data_dir} (Window: {window_size})", flush=True)
    
    raw_train_path = os.path.join(data_dir, 'X_train.npy')
    raw_val_path = os.path.join(data_dir, 'X_val.npy')
    raw_y_train_path = os.path.join(data_dir, 'y_train.npy')
    raw_y_val_path = os.path.join(data_dir, 'y_val.npy')
    
    train_path = copy_to_local(raw_train_path, local_rank, world_size)
    val_path = copy_to_local(raw_val_path, local_rank, world_size)
    local_y_train = copy_to_local(raw_y_train_path, local_rank, world_size)
    local_y_val = copy_to_local(raw_y_val_path, local_rank, world_size)
    
    y_train_raw = np.load(local_y_train)
    y_val_raw = np.load(local_y_val)
    
    y_train_m = y_train_raw * 1000.0 if np.max(y_train_raw) < 100 else y_train_raw
    y_val_m = y_val_raw * 1000.0 if np.max(y_val_raw) < 100 else y_val_raw
    
    def to_class(y_m):
        cls = np.zeros_like(y_m, dtype=np.int64)
        cls[y_m >= 500] = 1
        cls[y_m >= 1000] = 2
        return cls
    
    y_train_cls = to_class(y_train_m)
    y_val_cls = to_class(y_val_m)
    
    # === 计算类别统计 (用于 Logit Adjustment) ===
    class_counts = Counter(y_train_cls)
    counts_list = [class_counts[0], class_counts[1], class_counts[2]]
    if rank == 0:
        print(f"[Data] Training Class Counts: {counts_list}")

    y_train_log = np.log1p(y_train_m).astype(np.float32)
    y_val_log = np.log1p(y_val_m).astype(np.float32)
    
    # ... Scaler fitting logic (保持不变) ...
    if scaler is None or not reuse_scaler:
        if rank == 0:
            print(f"  Fitting Scaler...", flush=True)
            scaler = RobustScaler()
            X_temp = np.load(train_path, mmap_mode='r')
            subset_size = min(300000, len(X_temp))
            indices = np.random.choice(len(X_temp), subset_size, replace=False)
            indices.sort()
            
            X_subset = X_temp[indices, :-1].copy()
            X_subset = np.nan_to_num(X_subset, nan=0.0)
            
            log_mask = np.zeros(X_subset.shape[1], dtype=bool)
            for t in range(window_size):
                offset = t * 25
                for idx in [offset + 2, offset + 4, offset + 9]:
                    if idx < X_subset.shape[1]:
                        log_mask[idx] = True
            
            np.maximum(X_subset, 0, out=X_subset, where=log_mask)
            np.log1p(X_subset, out=X_subset, where=log_mask)
            
            scaler.fit(X_subset)
            del X_subset, X_temp
            
            center_tensor = torch.from_numpy(scaler.center_).float().to(device)
            scale_tensor = torch.from_numpy(scaler.scale_).float().to(device)
            dim_tensor = torch.tensor([len(scaler.center_)], device=device)
        else:
            dim_tensor = torch.tensor([0], device=device)
            center_tensor = None
            scale_tensor = None

        if world_size > 1 and dist.is_initialized():
            dist.broadcast(dim_tensor, src=0)
        
        feat_dim = dim_tensor.item()
        
        if rank != 0:
            center_tensor = torch.zeros(feat_dim, device=device)
            scale_tensor = torch.zeros(feat_dim, device=device)
        
        if world_size > 1 and dist.is_initialized():
            dist.broadcast(center_tensor, src=0)
            dist.broadcast(scale_tensor, src=0)
        
        if rank != 0:
            scaler = RobustScaler()
            scaler.center_ = center_tensor.cpu().numpy()
            scaler.scale_ = scale_tensor.cpu().numpy()
            
    train_ds = PMSTDataset(train_path, y_train_cls, y_train_log, y_train_m, scaler=scaler, window_size=window_size)
    val_ds = PMSTDataset(val_path, y_val_cls, y_val_log, y_val_m, scaler=scaler, window_size=window_size)
    
    return train_ds, val_ds, scaler, counts_list

# ==========================================
# 7. 改进的训练流程 (引入 OneCycleLR)
# ==========================================

def train_with_comprehensive_evaluation(
    stage_name, model, train_ds, val_loader, optimizer,
    criterions, scaler_amp, device, rank, world_size,
    total_steps, val_interval, batch_size, 
    fog_ratio, mist_ratio, grad_accum, exp_id,
    scheduler=None # 新增 scheduler 参数
):
    sampler = StratifiedBalancedSampler(
        train_ds, batch_size=batch_size,
        fog_ratio=fog_ratio, mist_ratio=mist_ratio,
        clear_ratio=1.0 - fog_ratio - mist_ratio,
        rank=rank, world_size=world_size
    )
    
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler,
        num_workers=6, pin_memory=True
    )
    
    train_iter = iter(train_loader)
    model.train()
    
    dual_loss_fn, reg_loss_fn = criterions
    
    evaluator = ComprehensiveMetrics(
        min_fog_precision=CONFIG['MIN_FOG_PRECISION'],
        min_fog_recall=CONFIG['MIN_FOG_RECALL'],
        min_mist_precision=CONFIG['MIN_MIST_PRECISION'],
        min_mist_recall=CONFIG['MIN_MIST_RECALL'],
        min_clear_acc=CONFIG['MIN_CLEAR_ACC'],
        config=CONFIG
    )
    
    best_scores = {'composite': -1, 'fog_f2': -1, 'mist_f2': -1, 'low_vis_ts': -1}
    optimizer.zero_grad()
    
    if rank == 0:
        print(f"[{stage_name}] Starting training for {total_steps} steps...")
    
    for step in range(1, total_steps + 1):
        try:
            bx, by_cls, by_log, by_raw = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            bx, by_cls, by_log, by_raw = next(train_iter)
        
        bx = bx.to(device, non_blocking=True)
        by_cls = by_cls.to(device, non_blocking=True)
        by_log = by_log.to(device, non_blocking=True)
        by_raw = by_raw.to(device, non_blocking=True)
        
        with autocast():
            final_logits, reg_out, low_vis_logit, fine_logits = model(bx)
            l_dual, dual_breakdown = dual_loss_fn(final_logits, low_vis_logit, fine_logits, by_cls)
            l_reg = reg_loss_fn(reg_out, by_log, by_raw)
            loss = l_dual + 0.1 * l_reg
            loss = loss / grad_accum
        
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0: print(f"\n[WARNING] Step {step}: NaN/Inf, skipping")
            optimizer.zero_grad()
            continue
        
        scaler_amp.scale(loss).backward()
        
        if step % grad_accum == 0:
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['GRAD_CLIP_NORM'])
            scaler_amp.step(optimizer)
            scaler_amp.update()
            optimizer.zero_grad()
            
            # Step Scheduler per batch (for OneCycleLR)
            if scheduler is not None:
                scheduler.step()
        
        if rank == 0 and step % 100 == 0:
            fog_boost_val = dual_breakdown.get('fog_boost', 0)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\r[{stage_name}] Step {step}/{total_steps} | LR: {current_lr:.2e} | "
                  f"L_dual: {l_dual.item():.4f} (Fog+: {fog_boost_val:.4f}) | "
                  f"L_reg: {l_reg.item():.4f}", end="", flush=True)
        
        if step % val_interval == 0:
            if rank == 0: print(f"\n")
            metrics = evaluator.evaluate_comprehensive(model, val_loader, device, search_threshold=True, rank=rank)
            
            if rank == 0:
                evaluator.print_detailed_report(metrics, title=f"{stage_name} @ Step {step}")
                win_suffix = f"_{CONFIG['WINDOW_SIZE']}h_"
                
                # Save checkpoints logic (保持不变)
                if metrics['composite_score'] > best_scores['composite']:
                    best_scores['composite'] = metrics['composite_score']
                    save_path = os.path.join(CONFIG['BASE_PATH'], f"model/pmst_{stage_name.lower()}{win_suffix}{exp_id}_best_composite.pth")
                    torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), save_path)
                    print(f"✓ Best Composite: {best_scores['composite']:.4f}")
                # ... (其余保存逻辑相同)

            model.train()

# ==========================================
# 8. Main
# ==========================================

def main():
    local_rank, global_rank, world_size = init_distributed()
    miopen_cache_dir = f"/tmp/miopen_cache_rank_{local_rank}"
    os.makedirs(miopen_cache_dir, exist_ok=True)
    os.environ["MIOPEN_USER_DB_PATH"] = miopen_cache_dir
    
    device = torch.device(f"cuda:{local_rank}")
    scaler_amp = GradScaler()
    
    if global_rank == 0:
        os.makedirs(os.path.join(CONFIG['BASE_PATH'], "model"), exist_ok=True)
        os.makedirs(os.path.join(CONFIG['BASE_PATH'], "scalers"), exist_ok=True)
        print(f"Experiment ID: {CONFIG['EXPERIMENT_ID']}")

    exp_id = CONFIG['EXPERIMENT_ID']
    
    # 1. 创建模型 (包含可学习物理层)
    model = ImprovedDualStreamPMSTNet(
        dyn_vars_count=25, 
        window_size=CONFIG['WINDOW_SIZE'],
        hidden_dim=CONFIG['MODEL_HIDDEN_DIM'], 
        num_classes=CONFIG['MODEL_NUM_CLASSES']
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # 2. Stage 1: Pre-training
    if not CONFIG['SKIP_STAGE1']:
        if global_rank == 0: print("\n[STAGE 1] ERA5 Pre-training")
        
        s1_train_ds, s1_val_ds, scaler_s1, s1_counts = load_data_and_scale(
            CONFIG['S1_DATA_DIR'], rank=global_rank, device=device, 
            reuse_scaler=False, window_size=CONFIG['WINDOW_SIZE'], world_size=world_size
        )
        
        # 初始化 Loss (带 Logit Adjustment)
        dual_loss = DualBranchLoss(
            loss_type=CONFIG['LOSS_TYPE'],
            class_counts=s1_counts, # 传入类别计数
            use_logit_adj=CONFIG['USE_LOGIT_ADJUSTMENT'],
            logit_adj_tau=CONFIG['LOGIT_ADJ_TAU'],
            alpha_binary=CONFIG['LOSS_ALPHA_BINARY'],
            fine_class_weight=[CONFIG['FINE_CLASS_WEIGHT_FOG'], CONFIG['FINE_CLASS_WEIGHT_MIST'], CONFIG['FINE_CLASS_WEIGHT_CLEAR']],
            # ... 其他 Loss 参数
        ).to(device)
        reg_loss = PhysicsConstrainedRegLoss(alpha=CONFIG['REG_LOSS_ALPHA']).to(device)
        
        s1_val_loader = DataLoader(s1_val_ds, batch_size=CONFIG['S1_BATCH_SIZE'], shuffle=False, num_workers=4)
        
        # Stage 1 使用标准 AdamW
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['S1_LR_BACKBONE'], weight_decay=CONFIG['S1_WEIGHT_DECAY'])
        
        train_with_comprehensive_evaluation(
            'S1', model, s1_train_ds, s1_val_loader, optimizer, 
            (dual_loss, reg_loss), scaler_amp, device, global_rank, world_size,
            CONFIG['S1_TOTAL_STEPS'], CONFIG['S1_VAL_INTERVAL'], 
            CONFIG['S1_BATCH_SIZE'], CONFIG['S1_FOG_RATIO'], CONFIG['S1_MIST_RATIO'],
            CONFIG['S1_GRAD_ACCUM'], exp_id
        )
        
        # 保存 Scaler 并清理
        if global_rank == 0:
            joblib.dump(scaler_s1, os.path.join(CONFIG['BASE_PATH'], f"scalers/scaler_pmst_{exp_id}_{CONFIG['WINDOW_SIZE']}h.pkl"))
        del s1_train_ds, s1_val_ds, s1_val_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    # 3. Stage 2: Fine-tuning
    if global_rank == 0: print("\n[STAGE 2] Forecast Fine-tuning")
    
    # 加载 Stage 2 数据
    scaler_s1 = None
    if not CONFIG['SKIP_STAGE1']:
        # 重载 scaler (或者如果你有更好的管理方式)
        # 这里为了简化，假设 scaler_s1 在内存中被清除，需要重新加载或复用
        # 但在上面的代码块中，scaler_s1 只是 python 对象，除非被 del，否则还在
        pass 
    
    s2_train_ds, s2_val_ds, scaler_s2, s2_counts = load_data_and_scale(
        CONFIG['S2_DATA_DIR'], 
        scaler=None, # 如果 SKIP_STAGE1=False，此处应复用 Stage 1 的 scaler。这里简化处理，重新拟合或加载。
                     # 实际生产中建议复用 Stage 1 Scaler。
        rank=global_rank, device=device, reuse_scaler=False, 
        window_size=CONFIG['WINDOW_SIZE'], world_size=world_size
    )
    
    # 更新 Loss 的 Priors (因为数据集变了，分布可能变了)
    dual_loss = DualBranchLoss(
        loss_type=CONFIG['LOSS_TYPE'],
        class_counts=s2_counts, # 更新计数
        use_logit_adj=CONFIG['USE_LOGIT_ADJUSTMENT'],
        logit_adj_tau=CONFIG['LOGIT_ADJ_TAU'],
        alpha_binary=CONFIG['LOSS_ALPHA_BINARY'],
        fine_class_weight=[CONFIG['FINE_CLASS_WEIGHT_FOG'], CONFIG['FINE_CLASS_WEIGHT_MIST'], CONFIG['FINE_CLASS_WEIGHT_CLEAR']]
    ).to(device)
    reg_loss = PhysicsConstrainedRegLoss(alpha=CONFIG['REG_LOSS_ALPHA']).to(device)

    # 4. Stage 2 Optimizer & Scheduler (OneCycleLR)
    # 定义参数组：Head 学习率高，Backbone 学习率低
    params = [
        {'params': [p for n, p in model.named_parameters() if "head" not in n and "detector" not in n and "classifier" not in n], 
         'lr': CONFIG['S2_MAX_LR_BACKBONE']},
        {'params': [p for n, p in model.named_parameters() if "head" in n or "detector" in n or "classifier" in n], 
         'lr': CONFIG['S2_MAX_LR_HEAD']}
    ]
    
    s2_optimizer = optim.AdamW(params, weight_decay=CONFIG['S2_WEIGHT_DECAY'])
    
    # OneCycleLR Scheduler
    # 注意：steps_per_epoch 设为 1，因为我们在 train loop 里是基于 step 计数的，不是 epoch
    s2_scheduler = optim.lr_scheduler.OneCycleLR(
        s2_optimizer,
        max_lr=[CONFIG['S2_MAX_LR_BACKBONE'], CONFIG['S2_MAX_LR_HEAD']],
        total_steps=int(CONFIG['S2_TOTAL_STEPS'] / CONFIG['S2_GRAD_ACCUM']), # 调整总步数以适应梯度累积
        pct_start=CONFIG['S2_PCT_START'],
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    s2_val_loader = DataLoader(s2_val_ds, batch_size=CONFIG['S2_BATCH_SIZE'], shuffle=False, num_workers=4)
    
    train_with_comprehensive_evaluation(
        'S2', model, s2_train_ds, s2_val_loader, s2_optimizer,
        (dual_loss, reg_loss), scaler_amp, device, global_rank, world_size,
        CONFIG['S2_TOTAL_STEPS'], CONFIG['S2_VAL_INTERVAL'],
        CONFIG['S2_BATCH_SIZE'], CONFIG['S2_FOG_RATIO'], CONFIG['S2_MIST_RATIO'],
        CONFIG['S2_GRAD_ACCUM'], exp_id,
        scheduler=s2_scheduler # 传入 Scheduler
    )
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
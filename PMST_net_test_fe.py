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

TARGET_WINDOW_SIZE = 12
BASE_PATH = "/public/home/putianshu/vis_mlp"

S1_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_{TARGET_WINDOW_SIZE}h"
S2_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_fe_{TARGET_WINDOW_SIZE}h"

CONFIG = {
    # ========== 实验控制配置 ==========
    'EXPERIMENT_ID': 'test1_fe',       # 实验ID，用于区分不同实验的模型文件
    'SKIP_STAGE1': False,             # 是否跳过Stage1，直接在Stage2数据上训练
    'S1_PRETRAINED_PATH': None,       # Stage1预训练模型路径（SKIP_STAGE1=True且需要加载时使用）
    
    # ========== 基础路径配置 ==========
    'BASE_PATH': BASE_PATH,
    'WINDOW_SIZE': TARGET_WINDOW_SIZE,
    
    'S1_DATA_DIR': S1_DIR, 
    'S1_SUFFIX': f"_{TARGET_WINDOW_SIZE}h_pmst_v2",   
    
    'S2_DATA_DIR': S2_DIR, 
    'S2_SUFFIX': f"_{TARGET_WINDOW_SIZE}h_forecast_v1", 

    # ========== 特征工程配置 ==========
    'USE_FEATURE_ENGINEERING': True,  # 是否使用特征工程额外特征
    'FE_EXTRA_DIMS': 24,              # 额外特征工程维度（S1若无FE则设为0，S2使用FE则设为实际维度）
    'S1_USE_FE': False,               # Stage1数据集是否包含FE特征
    'S2_USE_FE': True,                # Stage2数据集是否包含FE特征
    
    # ========== Stage 1 训练配置 ==========
    'S1_TOTAL_STEPS': 30000,      
    'S1_VAL_INTERVAL': 2000,      
    'S1_BATCH_SIZE': 512,         
    'S1_GRAD_ACCUM': 2,
    'S1_FOG_RATIO': 0.15,     
    'S1_MIST_RATIO': 0.15,
    'S1_LR_BACKBONE': 3e-4,       # Stage 1 学习率
    'S1_WEIGHT_DECAY': 1e-3,      # Stage 1 权重衰减
    
    # ========== Stage 2 训练配置 ==========
    'S2_TOTAL_STEPS': 5000,       
    'S2_VAL_INTERVAL': 500,
    'S2_BATCH_SIZE': 512,
    'S2_GRAD_ACCUM': 1,
    'S2_FOG_RATIO': 0.12,      
    'S2_MIST_RATIO': 0.12,
    'S2_LR_BACKBONE': 5e-6,       # Stage 2 backbone学习率
    'S2_LR_HEAD': 5e-5,           # Stage 2 head学习率
    'S2_WEIGHT_DECAY': 1e-2,      # Stage 2 权重衰减
    
    # ========== 评估约束条件 ==========
    'MIN_FOG_PRECISION': 0.20,   # Fog最小精确率
    'MIN_FOG_RECALL': 0.50,      # Fog最小召回率
    'MIN_MIST_PRECISION': 0.12,  # Mist最小精确率
    'MIN_MIST_RECALL': 0.15,     # Mist最小召回率
    'MIN_CLEAR_ACC': 0.95,       # Clear最小准确率
    
    # ========== 损失函数配置 ==========
    'LOSS_TYPE': 'asymmetric',   # 'asymmetric', 'focal', 'multifocal', 'balanced_focal'
    
    # DualBranchLoss 参数
    'LOSS_ALPHA_BINARY': 1.0,       # 二分类损失权重
    'LOSS_ALPHA_FINE': 1.0,         # 精细分类损失权重
    'LOSS_ALPHA_CONSISTENCY': 0.5,  # 一致性损失权重
    'LOSS_ALPHA_FP': 3.0,           # False Alarm惩罚权重
    'LOSS_ALPHA_FOG_BOOST': 0.1,    # Fog增强权重
    'LOSS_ALPHA_MIST_BOOST': 0.1,   # Mist增强权重
    'LOSS_FP_THRESHOLD': 0.5,       # False Alarm阈值
    
    # AsymmetricLoss 参数
    'ASYM_GAMMA_NEG': 2.0,            # 负样本gamma
    'ASYM_GAMMA_POS': 1.0,            # 正样本gamma
    'ASYM_CLIP': 0.05,              # 负样本概率裁剪值
    
    # Focal Loss 参数
    'FOCAL_GAMMA': 2.0,             # Focal Loss gamma
    'FOCAL_ALPHA': None,            # Focal Loss alpha (None则使用默认)
    
    # 类别权重
    'BINARY_POS_WEIGHT': 1.2,       # 二分类正样本权重
    'FINE_CLASS_WEIGHT_FOG': 1.5,   # Fog类别权重
    'FINE_CLASS_WEIGHT_MIST': 1.2,  # Mist类别权重
    'FINE_CLASS_WEIGHT_CLEAR': 1.0, # Clear类别权重
    
    # ========== 阈值搜索配置 ==========
    'THRESHOLD_FOG_MIN': 0.10,      # Fog阈值搜索下限
    'THRESHOLD_FOG_MAX': 0.90,      # Fog阈值搜索上限
    'THRESHOLD_FOG_STEP': 0.05,     # Fog阈值搜索步长
    'THRESHOLD_MIST_MIN': 0.10,     # Mist阈值搜索下限
    'THRESHOLD_MIST_MAX': 0.90,     # Mist阈值搜索上限
    'THRESHOLD_MIST_STEP': 0.05,    # Mist阈值搜索步长
    
    # 阈值搜索阶段2约束（放宽版本）
    'THRESHOLD_PHASE2_CLEAR_RECALL': 0.90,   # 阶段2 Clear召回率要求
    'THRESHOLD_PHASE2_FOG_PRECISION': 0.18,  # 阶段2 Fog精确率要求
    'THRESHOLD_PHASE2_FOG_RECALL': 0.45,     # 阶段2 Fog召回率要求
    
    # 阈值搜索阶段3约束（最宽松）
    'THRESHOLD_PHASE3_CLEAR_RECALL': 0.90,   # 阶段3 Clear召回率要求
    
    # 综合得分权重（阶段1）
    'SCORE_PHASE1_FOG': 0.45,       # 阶段1 Fog F2权重
    'SCORE_PHASE1_MIST': 0.40,      # 阶段1 Mist F2权重
    'SCORE_PHASE1_CLEAR': 0.15,     # 阶段1 Clear F2权重
    
    # 综合得分权重（阶段2 - Fog优先）
    'SCORE_PHASE2_FOG': 0.60,       # 阶段2 Fog F2权重
    'SCORE_PHASE2_MIST': 0.30,      # 阶段2 Mist F2权重
    'SCORE_PHASE2_CLEAR': 0.10,     # 阶段2 Clear F2权重
    
    # 综合得分权重（阶段3 - 平衡）
    'SCORE_PHASE3_FOG': 0.50,       # 阶段3 Fog F2权重
    'SCORE_PHASE3_MIST': 0.35,      # 阶段3 Mist F2权重
    'SCORE_PHASE3_CLEAR': 0.15,     # 阶段3 Clear F2权重
    
    # ========== 模型架构配置 ==========
    'MODEL_HIDDEN_DIM': 512,        # 隐藏层维度
    'MODEL_DROPOUT': 0.2,           # Dropout率
    'MODEL_NUM_CLASSES': 3,         # 分类数
    
    # ========== 其他配置 ==========
    'GRAD_CLIP_NORM': 1.0,          # 梯度裁剪范数
    'REG_LOSS_ALPHA': 1.0,          # 回归损失权重
}

# ==========================================
# 辅助函数：打印超参数配置
# ==========================================

def print_hyperparameters(config, rank=0):
    """打印所有超参数配置"""
    if rank != 0:
        return
    
    print("\n" + "="*80)
    print("  HYPERPARAMETER CONFIGURATION")
    print("="*80)
    
    # 实验控制
    print("\n【实验控制】")
    print(f"  Experiment ID: {config['EXPERIMENT_ID']}")
    print(f"  Skip Stage1: {config['SKIP_STAGE1']}")
    if config['S1_PRETRAINED_PATH']:
        print(f"  S1 Pretrained: {config['S1_PRETRAINED_PATH']}")
    
    # 基础配置
    print("\n【基础配置】")
    print(f"  Window Size: {config['WINDOW_SIZE']}h")
    print(f"  Base Path: {config['BASE_PATH']}")
    if not config['SKIP_STAGE1']:
        print(f"  S1 Data Dir: {config['S1_DATA_DIR']}")
    print(f"  S2 Data Dir: {config['S2_DATA_DIR']}")

    # 特征工程配置
    print("\n【特征工程配置】")
    print(f"  Use Feature Engineering: {config['USE_FEATURE_ENGINEERING']}")
    print(f"  FE Extra Dims: {config['FE_EXTRA_DIMS']}")
    print(f"  S1 Use FE: {config['S1_USE_FE']}")
    print(f"  S2 Use FE: {config['S2_USE_FE']}")
    
    # Stage 1 训练配置
    if not config['SKIP_STAGE1']:
        print("\n【Stage 1 训练配置】")
        print(f"  Total Steps: {config['S1_TOTAL_STEPS']}")
        print(f"  Val Interval: {config['S1_VAL_INTERVAL']}")
        print(f"  Batch Size: {config['S1_BATCH_SIZE']}")
        print(f"  Grad Accum: {config['S1_GRAD_ACCUM']}")
        print(f"  Fog Ratio: {config['S1_FOG_RATIO']:.2%}")
        print(f"  Mist Ratio: {config['S1_MIST_RATIO']:.2%}")
        print(f"  Learning Rate: {config['S1_LR_BACKBONE']:.2e}")
        print(f"  Weight Decay: {config['S1_WEIGHT_DECAY']:.2e}")
    
    # Stage 2 训练配置
    print("\n【Stage 2 训练配置】")
    print(f"  Total Steps: {config['S2_TOTAL_STEPS']}")
    print(f"  Val Interval: {config['S2_VAL_INTERVAL']}")
    print(f"  Batch Size: {config['S2_BATCH_SIZE']}")
    print(f"  Grad Accum: {config['S2_GRAD_ACCUM']}")
    print(f"  Fog Ratio: {config['S2_FOG_RATIO']:.2%}")
    print(f"  Mist Ratio: {config['S2_MIST_RATIO']:.2%}")
    print(f"  LR Backbone: {config['S2_LR_BACKBONE']:.2e}")
    print(f"  LR Head: {config['S2_LR_HEAD']:.2e}")
    print(f"  Weight Decay: {config['S2_WEIGHT_DECAY']:.2e}")
    
    # 评估约束
    print("\n【评估约束条件】")
    print(f"  Min Fog Precision: {config['MIN_FOG_PRECISION']:.2f}")
    print(f"  Min Fog Recall: {config['MIN_FOG_RECALL']:.2f}")
    print(f"  Min Mist Precision: {config['MIN_MIST_PRECISION']:.2f}")
    print(f"  Min Mist Recall: {config['MIN_MIST_RECALL']:.2f}")
    print(f"  Min Clear Accuracy: {config['MIN_CLEAR_ACC']:.2f}")
    
    # 损失函数配置
    print("\n【损失函数配置】")
    print(f"  Loss Type: {config['LOSS_TYPE']}")
    print(f"  Alpha Binary: {config['LOSS_ALPHA_BINARY']:.2f}")
    print(f"  Alpha Fine: {config['LOSS_ALPHA_FINE']:.2f}")
    print(f"  Alpha Consistency: {config['LOSS_ALPHA_CONSISTENCY']:.2f}")
    print(f"  Alpha FP: {config['LOSS_ALPHA_FP']:.2f}")
    print(f"  Alpha Fog Boost: {config['LOSS_ALPHA_FOG_BOOST']:.2f}")
    print(f"  Alpha Mist Boost: {config['LOSS_ALPHA_MIST_BOOST']:.2f}")
    print(f"  FP Threshold: {config['LOSS_FP_THRESHOLD']:.2f}")
    
    if config['LOSS_TYPE'] == 'asymmetric':
        print(f"\n  AsymmetricLoss参数:")
        print(f"    Gamma Neg: {config['ASYM_GAMMA_NEG']}")
        print(f"    Gamma Pos: {config['ASYM_GAMMA_POS']}")
        print(f"    Clip: {config['ASYM_CLIP']:.3f}")
    elif config['LOSS_TYPE'] in ['focal', 'multifocal', 'balanced_focal']:
        print(f"\n  FocalLoss参数:")
        print(f"    Gamma: {config['FOCAL_GAMMA']:.2f}")
        print(f"    Alpha: {config['FOCAL_ALPHA']}")
    
    # 类别权重
    print("\n【类别权重】")
    print(f"  Binary Pos Weight: {config['BINARY_POS_WEIGHT']:.2f}")
    print(f"  Fine Class Weight: [{config['FINE_CLASS_WEIGHT_FOG']:.1f}, "
          f"{config['FINE_CLASS_WEIGHT_MIST']:.1f}, {config['FINE_CLASS_WEIGHT_CLEAR']:.1f}]")
    
    # 模型架构
    print("\n【模型架构配置】")
    print(f"  Hidden Dim: {config['MODEL_HIDDEN_DIM']}")
    print(f"  Dropout: {config['MODEL_DROPOUT']:.2f}")
    print(f"  Num Classes: {config['MODEL_NUM_CLASSES']}")
    
    # 其他配置
    print("\n【其他配置】")
    print(f"  Gradient Clip Norm: {config['GRAD_CLIP_NORM']:.2f}")
    print(f"  Reg Loss Alpha: {config['REG_LOSS_ALPHA']:.2f}")
    
    print("\n" + "="*80 + "\n")

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
    """修复版本：移除device_id参数，使用world_size判断是否需要barrier"""
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

    # 修复：NCCL不支持device_ids参数，直接调用barrier()
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        
    if not os.path.exists(local_path):
        return src_path
    return local_path

# ==========================================
# 2. 改进的分层采样器
# ==========================================

class StratifiedBalancedSampler(Sampler):
    """分层平衡采样器 - 独立控制 Fog/Mist/Clear 比例"""
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
        
        if rank == 0:
            print(f"\n[StratifiedBalancedSampler]")
            print(f"  Total: Fog={len(all_fog)}, Mist={len(all_mist)}, Clear={len(all_clear)}")
            print(f"  Batch ({batch_size}): Fog={self.n_fog} ({fog_ratio*100:.1f}%), "
                  f"Mist={self.n_mist} ({mist_ratio*100:.1f}%), "
                  f"Clear={self.n_clear} ({clear_ratio*100:.1f}%)")
    
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
# 3. Loss 函数
# ==========================================

class AsymmetricLoss(nn.Module):
    """非对称损失 - 针对极度不平衡分类"""
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
        """
        logits: [B, C] 未归一化的logits
        targets: [B] 类别标签
        """
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

class FocalLoss(nn.Module):
    """
    Focal Loss - 解决类别不平衡问题
    论文: Focal Loss for Dense Object Detection (Lin et al., 2017)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (int, float)):
                alpha = torch.tensor([alpha], dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        
        if class_weights is not None:
            if isinstance(class_weights, (list, np.ndarray)):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        probs = F.softmax(logits, dim=1)
        
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - pt) ** self.gamma
        
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            if self.alpha.numel() == 1:
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.class_weights is not None:
            weight_t = self.class_weights.gather(0, targets)
            focal_loss = weight_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiClassFocalLoss(nn.Module):
    """
    多类别 Focal Loss 的另一种实现
    更灵活的参数配置
    """
    def __init__(self, num_classes=3, alpha=None, gamma=2.0, 
                 class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
        if alpha is None:
            alpha = [4.0, 4.0, 0.5]
        
        if isinstance(alpha, (list, np.ndarray)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha)
        
        if class_weights is not None:
            if isinstance(class_weights, (list, np.ndarray)):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            one_hot = F.one_hot(targets, self.num_classes).float()
            smooth_targets = one_hot * (1 - self.label_smoothing) + \
                           self.label_smoothing / self.num_classes
            
            log_probs = F.log_softmax(logits, dim=1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)
            
            probs = F.softmax(logits, dim=1)
            pt = (smooth_targets * probs).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            probs = F.softmax(logits, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_t = self.alpha.gather(0, targets)
        
        loss = alpha_t * focal_weight * ce_loss
        
        if self.class_weights is not None:
            weight_t = self.class_weights.gather(0, targets)
            loss = weight_t * loss
        
        return loss.mean()


class BalancedFocalLoss(nn.Module):
    """
    平衡 Focal Loss - 专门针对极度不平衡数据
    结合了类别平衡和难样本挖掘
    """
    def __init__(self, alpha=0.25, gamma=2.0, beta=0.999, num_classes=3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.num_classes = num_classes
        
        self.register_buffer('class_freq', torch.zeros(num_classes))
        self.register_buffer('total_samples', torch.tensor(0.0))
    
    def _update_class_freq(self, targets):
        """更新类别频率（可选，用于动态权重）"""
        for c in range(self.num_classes):
            count = (targets == c).sum().float()
            self.class_freq[c] = self.beta * self.class_freq[c] + (1 - self.beta) * count
        self.total_samples = self.beta * self.total_samples + (1 - self.beta) * targets.numel()
    
    def forward(self, logits, targets, update_freq=False):
        if update_freq and self.training:
            self._update_class_freq(targets)
        
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.total_samples > 0:
            class_weights = 1.0 / (self.class_freq + 1e-6)
            class_weights = class_weights / class_weights.sum() * self.num_classes
            alpha_t = class_weights.gather(0, targets)
        else:
            alpha_t = self.alpha
        
        loss = alpha_t * focal_weight * ce_loss
        
        return loss.mean()

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
                 focal_gamma=2.0,
                 focal_alpha=None,
                 asym_gamma_neg=5,
                 asym_gamma_pos=1,
                 asym_clip=0.05,
                 binary_pos_weight=1.2,
                 fine_class_weight=None):
        super().__init__()
        self.alpha_binary = alpha_binary
        self.alpha_fine = alpha_fine
        self.alpha_consistency = alpha_consistency
        self.alpha_fp = alpha_fp
        self.alpha_fog_boost = alpha_fog_boost
        self.alpha_mist_boost = alpha_mist_boost
        self.fp_threshold = fp_threshold
        self.loss_type = loss_type
        
        self.register_buffer('binary_pos_weight', torch.tensor([binary_pos_weight]))
        
        if fine_class_weight is None:
            fine_class_weight = torch.tensor([2.0, 2.0, 1.0])
        elif isinstance(fine_class_weight, list):
            fine_class_weight = torch.tensor(fine_class_weight)
        self.register_buffer('fine_class_weight', fine_class_weight)
        
        self.binary_loss = nn.BCEWithLogitsLoss(pos_weight=self.binary_pos_weight)
        
        if loss_type == 'asymmetric':
            self.fine_loss = AsymmetricLoss(
                gamma_neg=asym_gamma_neg,
                gamma_pos=asym_gamma_pos,
                clip=asym_clip,
                class_weights=self.fine_class_weight
            )
        elif loss_type == 'focal':
            self.fine_loss = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                class_weights=self.fine_class_weight
            )
        elif loss_type == 'multifocal':
            self.fine_loss = MultiClassFocalLoss(
                num_classes=3,
                gamma=focal_gamma,
                class_weights=self.fine_class_weight
            )
        elif loss_type == 'balanced_focal':
            self.fine_loss = BalancedFocalLoss(
                gamma=focal_gamma,
                num_classes=3
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    def forward(self, final_logits, low_vis_logit, fine_logits, targets):
        binary_targets = (targets <= 1).float().unsqueeze(1)
        loss_binary = self.binary_loss(low_vis_logit, binary_targets)
        
        if self.loss_type == 'balanced_focal':
            loss_fine = self.fine_loss(fine_logits, targets, update_freq=self.training)
        else:
            loss_fine = self.fine_loss(fine_logits, targets)
        
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
            'mist_boost': mist_boost.item(),
            'fp_count': high_confidence_mask.sum().item()
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
# 4. 模型架构
# ==========================================

class FogDiagnosticFeatures(nn.Module):
    def __init__(self, window_size=9, dyn_vars_count=25):
        super().__init__()
        self.window_size = window_size
        self.dyn_vars = dyn_vars_count
        
        self.idx = {
            'rh2m': 0, 't2m': 1, 'precip': 2, 'sw_rad': 4,
            'wspd10': 6, 'cape': 9, 'lcc': 10, 't925': 11,
            'rh925': 12, 'dpd': 22, 'inversion': 23, 'zenith': 24
        }
    
    def _extract_feature(self, x_seq, feature_name):
        idx = self.idx[feature_name]
        return x_seq[:, :, idx]
    
    def compute_saturation_proximity(self, rh, dpd):
        rh_clamp = torch.clamp(rh, 0, 100) / 100.0
        dpd_weight = torch.sigmoid(-3.0 * (dpd - 1.0))
        return torch.clamp(rh_clamp * dpd_weight, 0, 1)
    
    def compute_wind_favorability(self, wspd):
        wspd_clamp = torch.clamp(wspd, min=0)
        optimal_wspd = 3.5
        return torch.exp(-0.5 * ((wspd_clamp - optimal_wspd) / 2.5) ** 2)
    
    def compute_stability_index(self, inversion, wspd):
        wspd_clamp = torch.clamp(wspd, min=0.5)
        ri = inversion / (wspd_clamp ** 2 + 0.1)
        return torch.tanh(ri / 2.0)
    
    def compute_radiative_cooling_potential(self, sw_rad, lcc, zenith):
        is_night = (zenith > 90.0).float()
        lcc_clamp = torch.clamp(lcc, 0, 1)
        clear_sky = torch.clamp(1.0 - lcc_clamp / 0.3, 0, 1)
        sw_linear = torch.clamp(torch.expm1(sw_rad), min=0.0)
        rad_intensity = 1.0 - torch.clamp(sw_linear / 800.0, 0, 1)
        return is_night * clear_sky * rad_intensity
    
    def compute_vertical_moisture_transport(self, rh2m, rh925):
        return torch.clamp((rh2m - rh925) / 100.0, -1, 1)
    
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

# ==========================================
# 5. 双分支模型（支持额外特征工程维度）
# ==========================================

class ImprovedDualStreamPMSTNet(nn.Module):
    """改进的双分支架构模型，支持额外特征工程输入"""
    def __init__(self, dyn_vars_count=25, window_size=9, 
                 static_cont_dim=5, veg_num_classes=21, veg_emb_dim=16,
                 hidden_dim=512, num_classes=3, extra_feat_dim=0, dropout=0.2):
        super().__init__()
        self.dyn_vars = dyn_vars_count
        self.window = window_size
        self.static_cont_dim = static_cont_dim
        self.extra_feat_dim = extra_feat_dim
        
        self.fog_diagnostics = FogDiagnosticFeatures(window_size, dyn_vars_count)
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

        # 额外特征工程编码器（当 extra_feat_dim > 0 时启用）
        if extra_feat_dim > 0:
            self.extra_encoder = nn.Sequential(
                nn.Linear(extra_feat_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.extra_encoder = None
        
        # fusion_dim 根据是否有额外特征动态调整
        fusion_dim = hidden_dim * 2 + hidden_dim // 2 + hidden_dim // 4
        if extra_feat_dim > 0:
            fusion_dim += hidden_dim // 2

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
        """初始化偏置"""
        self.low_vis_detector[-1].bias.data.fill_(-0.5)
        self.fine_classifier[-1].bias.data = torch.tensor([-1.1, -1.1, 0.0])
        
    def forward(self, x):
        split_dyn = self.dyn_vars * self.window
        split_static = split_dyn + self.static_cont_dim

        x_dyn_flat = x[:, :split_dyn]
        x_stat_cont = x[:, split_dyn:split_static]
        x_veg_id = x[:, split_static].long()

        # 额外FE特征（若有）
        if self.extra_feat_dim > 0:
            x_extra = x[:, split_static + 1: split_static + 1 + self.extra_feat_dim]
        else:
            x_extra = None
        
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
        
        parts = [physical_feat, temporal_feat, static_feat, physics_feat]

        if x_extra is not None and self.extra_encoder is not None:
            extra_feat = self.extra_encoder(x_extra)
            parts.append(extra_feat)

        combined_feat = torch.cat(parts, dim=1)
        embedding = self.fusion_layer(combined_feat)
        
        fog_feat = self.fog_specific_features(embedding)
        
        low_vis_input = torch.cat([embedding, fog_feat], dim=1)
        low_vis_logit = self.low_vis_detector(low_vis_input)
        
        fine_logits = self.fine_classifier(embedding)
        
        final_logits = fine_logits
        
        reg_out = self.reg_head(embedding)
        
        return final_logits, reg_out, low_vis_logit, fine_logits

# ==========================================
# 6. 综合评估指标
# ==========================================

class ComprehensiveMetrics:
    """综合评估系统 - 支持动态阈值搜索"""
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
        """计算特定类别的F2分数"""
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
        """
        平衡的三阶段阈值搜索
        阶段1：严格约束（Fog和Mist都满足）
        阶段2：Fog优先（放宽Mist约束）
        阶段3：宽松模式（只保证Clear）
        """
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
        search_phase = None
        
        if rank == 0:
            print("\n[Balanced Threshold Search]")
        
        # ==== 阶段1：严格约束 ====
        for fog_th in fog_thresholds:
            for mist_th in mist_thresholds:
                preds = self._predict_with_thresholds(all_probs, fog_th, mist_th)
                
                fog_m = self.calculate_class_specific_f2(all_targets, preds, 0)
                mist_m = self.calculate_class_specific_f2(all_targets, preds, 1)
                clear_m = self.calculate_class_specific_f2(all_targets, preds, 2)
                
                if clear_m['recall'] < self.min_clear_acc:
                    continue
                if fog_m['precision'] < self.min_fog_precision:
                    continue
                if fog_m['recall'] < self.min_fog_recall:
                    continue
                if mist_m['precision'] < self.min_mist_precision:
                    continue
                if mist_m['recall'] < self.min_mist_recall:
                    continue
                
                score = (self.config['SCORE_PHASE1_FOG'] * fog_m['f2'] + 
                        self.config['SCORE_PHASE1_MIST'] * mist_m['f2'] + 
                        self.config['SCORE_PHASE1_CLEAR'] * clear_m['f2'])
                
                if score > best_score:
                    best_score = score
                    best_fog_th = fog_th
                    best_mist_th = mist_th
                    search_phase = "Phase 1 (Strict)"
        
        # ==== 阶段2：Fog优先，放宽Mist ====
        if best_score < 0:
            if rank == 0:
                print("  [Phase 2] Prioritizing Fog...")
            
            for fog_th in fog_thresholds:
                for mist_th in mist_thresholds:
                    preds = self._predict_with_thresholds(all_probs, fog_th, mist_th)
                    
                    fog_m = self.calculate_class_specific_f2(all_targets, preds, 0)
                    mist_m = self.calculate_class_specific_f2(all_targets, preds, 1)
                    clear_m = self.calculate_class_specific_f2(all_targets, preds, 2)
                    
                    if clear_m['recall'] < self.config['THRESHOLD_PHASE2_CLEAR_RECALL']:
                        continue
                    if fog_m['precision'] < self.config['THRESHOLD_PHASE2_FOG_PRECISION']:
                        continue
                    if fog_m['recall'] < self.config['THRESHOLD_PHASE2_FOG_RECALL']:
                        continue
                    
                    score = (self.config['SCORE_PHASE2_FOG'] * fog_m['f2'] + 
                            self.config['SCORE_PHASE2_MIST'] * mist_m['f2'] + 
                            self.config['SCORE_PHASE2_CLEAR'] * clear_m['f2'])
                    
                    if score > best_score:
                        best_score = score
                        best_fog_th = fog_th
                        best_mist_th = mist_th
                        search_phase = "Phase 2 (Fog Priority)"
        
        # ==== 阶段3：最宽松 ====
        if best_score < 0:
            if rank == 0:
                print("  [Phase 3] Minimum constraints...")
            
            for fog_th in fog_thresholds:
                for mist_th in mist_thresholds:
                    preds = self._predict_with_thresholds(all_probs, fog_th, mist_th)
                    
                    fog_m = self.calculate_class_specific_f2(all_targets, preds, 0)
                    mist_m = self.calculate_class_specific_f2(all_targets, preds, 1)
                    clear_m = self.calculate_class_specific_f2(all_targets, preds, 2)
                    
                    if clear_m['recall'] < self.config['THRESHOLD_PHASE3_CLEAR_RECALL']:
                        continue
                    
                    score = (self.config['SCORE_PHASE3_FOG'] * fog_m['f2'] + 
                            self.config['SCORE_PHASE3_MIST'] * mist_m['f2'] + 
                            self.config['SCORE_PHASE3_CLEAR'] * clear_m['f2'])
                    
                    if score > best_score:
                        best_score = score
                        best_fog_th = fog_th
                        best_mist_th = mist_th
                        search_phase = "Phase 3 (Relaxed)"
        
        self.best_thresholds['fog'] = best_fog_th
        self.best_thresholds['mist'] = best_mist_th
        
        if rank == 0:
            print(f"  Best Thresholds: Fog={best_fog_th:.2f}, Mist={best_mist_th:.2f}")
            print(f"  Score: {best_score:.4f} ({search_phase})")
        
        return best_fog_th, best_mist_th
    
    def _predict_with_thresholds(self, probs, fog_threshold, mist_threshold):
        """改进的预测逻辑"""
        prob_fog = probs[:, 0]
        prob_mist = probs[:, 1]
        prob_clear = probs[:, 2]
        
        predictions = np.full(len(probs), 2)
        
        mask_fog = (prob_fog > fog_threshold) & (prob_fog > prob_mist)
        predictions[mask_fog] = 0
        
        mask_mist = (prob_mist > mist_threshold) & (~mask_fog)
        predictions[mask_mist] = 1
        
        return predictions
    
    def print_detailed_report(self, metrics, title="Evaluation"):
        """增强的报告输出"""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
        
        if 'thresholds' in metrics:
            print(f"\n【使用阈值】")
            print(f"  Fog: {metrics['thresholds']['fog']:.2f}")
            print(f"  Mist: {metrics['thresholds']['mist']:.2f}")
        
        print("\n【各类别指标】")
        for class_name, class_id in [('Fog', 'fog'), ('Mist', 'mist'), ('Clear', 'clear')]:
            m = metrics[class_id]
            print(f"  {class_name}: P={m['precision']:.4f}, R={m['recall']:.4f}, F2={m['f2']:.4f} "
                  f"(TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
        
        print(f"\n【整体性能】")
        print(f"  Overall Acc: {metrics['overall_acc']:.4f}")
        print(f"  Low-Vis TS:  {metrics['low_vis_metrics']['ts']:.4f}")
        
        print(f"\n【综合得分】: {metrics['composite_score']:.4f}")
        
        print(f"\n【Fog-Mist交叉分析】")
        fog_err = metrics.get('fog_error_analysis', {})
        mist_err = metrics.get('mist_error_analysis', {})
        print(f"  Fog → Mist: {fog_err.get('to_mist_rate', 0):.2%}")
        print(f"  Mist → Fog: {mist_err.get('to_fog_rate', 0):.2%}")
        
        print(f"\n【约束检查】")
        constraints = metrics['constraints_met']
        for name, status in constraints.items():
            print(f"  {'✓' if status else '✗'} {name}")
        
        print("\n" + "="*70)
        
    def evaluate_comprehensive(self, model, val_loader, device, search_threshold=True, rank=0):
        """综合评估函数"""
        model.eval()
        all_probs = []
        all_targets = []
        all_low_vis_probs = []

        with torch.no_grad():
            for bx, by_cls, _, _ in val_loader:
                bx = bx.to(device, non_blocking=True)
                by_cls = by_cls.to(device, non_blocking=True)

                final_logits, _, low_vis_logit, fine_logits = model(bx)

                probs = F.softmax(final_logits, dim=1).cpu().numpy()
                low_vis_prob = torch.sigmoid(low_vis_logit).cpu().numpy()

                all_probs.append(probs)
                all_targets.append(by_cls.cpu().numpy())
                all_low_vis_probs.append(low_vis_prob)

        all_probs = np.vstack(all_probs)
        all_targets = np.concatenate(all_targets)
        all_low_vis_probs = np.concatenate(all_low_vis_probs)

        if search_threshold:
            fog_th, mist_th = self.search_optimal_thresholds(all_probs, all_targets, rank)
        else:
            fog_th = self.best_thresholds['fog']
            mist_th = self.best_thresholds['mist']

        predictions = self._predict_with_thresholds(all_probs, fog_th, mist_th)

        fog_metrics = self.calculate_class_specific_f2(all_targets, predictions, 0)
        mist_metrics = self.calculate_class_specific_f2(all_targets, predictions, 1)
        clear_metrics = self.calculate_class_specific_f2(all_targets, predictions, 2)

        low_vis_true = (all_targets <= 1).astype(int)
        low_vis_pred = (predictions <= 1).astype(int)
        low_vis_tp = ((low_vis_true == 1) & (low_vis_pred == 1)).sum()
        low_vis_fp = ((low_vis_true == 0) & (low_vis_pred == 1)).sum()
        low_vis_fn = ((low_vis_true == 1) & (low_vis_pred == 0)).sum()
        low_vis_ts = low_vis_tp / (low_vis_tp + low_vis_fp + low_vis_fn + 1e-6)

        composite_score = (self.config['SCORE_PHASE1_FOG'] * fog_metrics['f2'] + 
                          self.config['SCORE_PHASE1_MIST'] * mist_metrics['f2'] + 
                          self.config['SCORE_PHASE1_CLEAR'] * clear_metrics['f2'])

        overall_acc = accuracy_score(all_targets, predictions)

        constraints_met = {
            'Fog Precision': fog_metrics['precision'] >= self.min_fog_precision,
            'Fog Recall': fog_metrics['recall'] >= self.min_fog_recall,
            'Mist Precision': mist_metrics['precision'] >= self.min_mist_precision,
            'Mist Recall': mist_metrics['recall'] >= self.min_mist_recall,
            'Clear Accuracy': clear_metrics['recall'] >= self.min_clear_acc
        }

        fog_mask = (all_targets == 0)
        mist_mask = (all_targets == 1)

        fog_to_mist_count = ((predictions == 1) & fog_mask).sum()
        fog_total = fog_mask.sum()
        fog_error_analysis = {
            'to_mist_rate': fog_to_mist_count / (fog_total + 1e-6) if fog_total > 0 else 0.0
        }

        mist_to_fog_count = ((predictions == 0) & mist_mask).sum()
        mist_total = mist_mask.sum()
        mist_error_analysis = {
            'to_fog_rate': mist_to_fog_count / (mist_total + 1e-6) if mist_total > 0 else 0.0
        }

        return {
            'fog': fog_metrics,
            'mist': mist_metrics,
            'clear': clear_metrics,
            'low_vis_metrics': {'ts': low_vis_ts},
            'composite_score': composite_score,
            'overall_acc': overall_acc,
            'thresholds': {'fog': fog_th, 'mist': mist_th},
            'constraints_met': constraints_met,
            'fog_error_analysis': fog_error_analysis,
            'mist_error_analysis': mist_error_analysis
        }


# ==========================================
# 7. 数据处理
# ==========================================

class PMSTDataset(Dataset):
    def __init__(self, X_path, y_cls, y_reg, y_raw, scaler=None, apply_log_transform=True,
                 window_size=9, use_fe=False, fe_extra_dims=0):
        self.X = np.load(X_path, mmap_mode='r')
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.as_tensor(y_reg, dtype=torch.float32)
        self.y_raw = torch.as_tensor(y_raw, dtype=torch.float32)
        self.window_size = window_size
        self.use_fe = use_fe
        self.fe_extra_dims = fe_extra_dims

        # split points
        self.split_dyn = window_size * 25
        self.split_static = self.split_dyn + 5   # 5 static continuous vars
        # veg_id is at index split_static (single int column)
        # if use_fe: extra FE features follow after veg_id
        
        self.has_scaler = scaler is not None
        if self.has_scaler:
            # scaler was fit on dyn + static_cont only (split_static columns)
            self.center = scaler.center_.astype(np.float32)
            self.scale = scaler.scale_.astype(np.float32)
            self.scale = np.where(self.scale == 0, 1.0, self.scale)
        
        self.apply_log_transform = apply_log_transform
        
        self.log_mask = np.zeros(self.split_dyn, dtype=bool)
        if apply_log_transform:
            for t in range(self.window_size):
                offset = t * 25
                for idx in [offset + 2, offset + 4, offset + 9]:
                    if idx < self.split_dyn:
                        self.log_mask[idx] = True
        
        self.clip_min = -10.0
        self.clip_max = 10.0

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]

        # ---- dynamic + static continuous features ----
        features = row[:self.split_static].astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # log-transform selected dynamic features
        if self.apply_log_transform:
            dyn_part = features[:self.split_dyn]
            np.maximum(dyn_part, 0, out=dyn_part, where=self.log_mask[:len(dyn_part)])
            np.log1p(dyn_part, out=dyn_part, where=self.log_mask[:len(dyn_part)])
            features[:self.split_dyn] = dyn_part

        # scale
        if self.has_scaler:
            features = (features - self.center) / self.scale

        features = np.clip(features, self.clip_min, self.clip_max)

        # ---- vegetation id (integer, kept as float for concat) ----
        veg_id = np.array([row[self.split_static]], dtype=np.float32)

        # ---- extra FE features (if present) ----
        if self.use_fe and self.fe_extra_dims > 0:
            fe_start = self.split_static + 1
            fe_end = fe_start + self.fe_extra_dims
            extra = row[fe_start:fe_end].astype(np.float32)
            extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
            extra = np.clip(extra, self.clip_min, self.clip_max)
            final = np.concatenate([features, veg_id, extra])
        else:
            final = np.concatenate([features, veg_id])

        return (
            torch.from_numpy(final).float(), 
            self.y_cls[idx], 
            self.y_reg[idx], 
            self.y_raw[idx]
        )

def load_data_and_scale(data_dir, scaler=None, rank=0, device=None, reuse_scaler=False,
                        window_size=9, world_size=1, use_fe=False, fe_extra_dims=0):
    """加载数据并拟合/应用Scaler，支持特征工程额外维度"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        print(f"Loading from: {data_dir} (Window: {window_size}, use_fe={use_fe}, fe_extra_dims={fe_extra_dims})", flush=True)
    
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
    
    y_train_log = np.log1p(y_train_m).astype(np.float32)
    y_val_log = np.log1p(y_val_m).astype(np.float32)

    # Scaler is fit only on dyn (window_size*25) + static_cont (5) = split_static columns
    split_static = window_size * 25 + 5

    if scaler is None or not reuse_scaler:
        if rank == 0:
            print(f"  Fitting Scaler (on first {split_static} feature columns)...", flush=True)
            scaler = RobustScaler()
            X_temp = np.load(train_path, mmap_mode='r')
            subset_size = min(300000, len(X_temp))
            indices = np.random.choice(len(X_temp), subset_size, replace=False)
            indices.sort()
            
            # Only use dyn + static_cont for scaler fitting (exclude veg_id and FE extras)
            X_subset = X_temp[indices, :split_static].copy()
            X_subset = np.nan_to_num(X_subset, nan=0.0)
            
            # Apply log transform before fitting
            log_mask = np.zeros(window_size * 25, dtype=bool)
            for t in range(window_size):
                offset = t * 25
                for idx in [offset + 2, offset + 4, offset + 9]:
                    if idx < window_size * 25:
                        log_mask[idx] = True
            
            dyn_part = X_subset[:, :window_size * 25]
            np.maximum(dyn_part, 0, out=dyn_part, where=log_mask)
            np.log1p(dyn_part, out=dyn_part, where=log_mask)
            X_subset[:, :window_size * 25] = dyn_part
            
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
    
    train_ds = PMSTDataset(
        train_path, y_train_cls, y_train_log, y_train_m,
        scaler=scaler, window_size=window_size,
        use_fe=use_fe, fe_extra_dims=fe_extra_dims
    )
    val_ds = PMSTDataset(
        val_path, y_val_cls, y_val_log, y_val_m,
        scaler=scaler, window_size=window_size,
        use_fe=use_fe, fe_extra_dims=fe_extra_dims
    )
    
    return train_ds, val_ds, scaler

# ==========================================
# 8. 改进的训练流程
# ==========================================

def train_with_comprehensive_evaluation(
    stage_name, model, train_ds, val_loader, optimizer,
    criterions, scaler_amp, device, rank, world_size,
    total_steps, val_interval, batch_size, 
    fog_ratio, mist_ratio, grad_accum, exp_id
):
    """改进的训练流程"""
    sampler = StratifiedBalancedSampler(
        train_ds,
        batch_size=batch_size,
        fog_ratio=fog_ratio,
        mist_ratio=mist_ratio,
        clear_ratio=1.0 - fog_ratio - mist_ratio,
        rank=rank,
        world_size=world_size
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=6,
        pin_memory=True
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
    best_scores = {
        'composite': -1,
        'fog_f2': -1,
        'mist_f2': -1,
        'low_vis_ts': -1,
    }
    
    optimizer.zero_grad()
    
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
            if rank == 0:
                print(f"\n[WARNING] Step {step}: NaN/Inf, skipping")
            optimizer.zero_grad()
            continue
        
        scaler_amp.scale(loss).backward()
        
        if step % grad_accum == 0:
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['GRAD_CLIP_NORM'])
            scaler_amp.step(optimizer)
            scaler_amp.update()
            optimizer.zero_grad()
        
        if rank == 0 and step % 100 == 0:
            fog_boost_val = dual_breakdown.get('fog_boost', 0)
            mist_boost_val = dual_breakdown.get('mist_boost', 0)
            print(f"\r[{stage_name}] Step {step}/{total_steps} | "
                  f"L_dual: {l_dual.item():.4f} "
                  f"(Fog+: {fog_boost_val:.4f}, Mist+: {mist_boost_val:.4f}) | "
                  f"L_reg: {l_reg.item():.4f}", 
                  end="", flush=True)
        
        if step % val_interval == 0:
            if rank == 0:
                print(f"\n")
            
            metrics = evaluator.evaluate_comprehensive(
                model, val_loader, device, 
                search_threshold=True,
                rank=rank
            )
            
            if rank == 0:
                evaluator.print_detailed_report(metrics, title=f"{stage_name} @ Step {step}")
                
                win_suffix = f"_{CONFIG['WINDOW_SIZE']}h_"
                
                if metrics['composite_score'] > best_scores['composite']:
                    best_scores['composite'] = metrics['composite_score']
                    save_path = os.path.join(CONFIG['BASE_PATH'], 
                        f"model/pmst_{stage_name.lower()}{win_suffix}{exp_id}_best_composite.pth")
                    torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), save_path)
                    threshold_path = os.path.join(CONFIG['BASE_PATH'],
                        f"model/pmst_{stage_name.lower()}{win_suffix}{exp_id}_best_composite_thresholds.pkl")
                    joblib.dump(evaluator.best_thresholds, threshold_path)
                    print(f"✓ Best Composite: {best_scores['composite']:.4f}")
                
                if metrics['fog']['f2'] > best_scores['fog_f2']:
                    best_scores['fog_f2'] = metrics['fog']['f2']
                    save_path = os.path.join(CONFIG['BASE_PATH'],
                        f"model/pmst_{stage_name.lower()}{win_suffix}{exp_id}_best_fog_f2.pth")
                    torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), save_path)
                    threshold_path = os.path.join(CONFIG['BASE_PATH'],
                        f"model/pmst_{stage_name.lower()}{win_suffix}{exp_id}_best_fog_f2_thresholds.pkl")
                    joblib.dump(evaluator.best_thresholds, threshold_path)
                    print(f"✓ Best Fog F2: {best_scores['fog_f2']:.4f}")
                
                if metrics['mist']['f2'] > best_scores['mist_f2']:
                    best_scores['mist_f2'] = metrics['mist']['f2']
                    save_path = os.path.join(CONFIG['BASE_PATH'],
                        f"model/pmst_{stage_name.lower()}{win_suffix}{exp_id}_best_mist_f2.pth")
                    torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), save_path)
                    threshold_path = os.path.join(CONFIG['BASE_PATH'],
                        f"model/pmst_{stage_name.lower()}{win_suffix}{exp_id}_best_mist_f2_thresholds.pkl")
                    joblib.dump(evaluator.best_thresholds, threshold_path)
                    print(f"✓ Best Mist F2: {best_scores['mist_f2']:.4f}")
                
                if metrics['low_vis_metrics']['ts'] > best_scores['low_vis_ts']:
                    best_scores['low_vis_ts'] = metrics['low_vis_metrics']['ts']
                    save_path = os.path.join(CONFIG['BASE_PATH'],
                        f"model/pmst_{stage_name.lower()}{win_suffix}{exp_id}_best_low_vis_ts.pth")
                    torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), save_path)
                    threshold_path = os.path.join(CONFIG['BASE_PATH'],
                        f"model/pmst_{stage_name.lower()}{win_suffix}{exp_id}_best_low_vis_ts_thresholds.pkl")
                    joblib.dump(evaluator.best_thresholds, threshold_path)
                    print(f"✓ Best Low-Vis TS: {best_scores['low_vis_ts']:.4f}\n")
            
            model.train()
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"[{stage_name}] Training Completed!")
        print(f"  Composite: {best_scores['composite']:.4f}")
        print(f"  Fog F2:    {best_scores['fog_f2']:.4f}")
        print(f"  Mist F2:   {best_scores['mist_f2']:.4f}")
        print(f"  Low-Vis TS: {best_scores['low_vis_ts']:.4f}")
        print('='*70)

        
# ==========================================
# 9. Main
# ==========================================

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    miopen_cache_dir = f"/tmp/miopen_cache_rank_{local_rank}"
    os.makedirs(miopen_cache_dir, exist_ok=True)
    os.environ["MIOPEN_USER_DB_PATH"] = miopen_cache_dir
    
    local_rank, global_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    scaler_amp = GradScaler()
    
    if global_rank == 0:
        os.makedirs(os.path.join(CONFIG['BASE_PATH'], "model"), exist_ok=True)
        os.makedirs(os.path.join(CONFIG['BASE_PATH'], "scalers"), exist_ok=True)
        
        print_hyperparameters(CONFIG, rank=global_rank)

    exp_id = CONFIG['EXPERIMENT_ID']

    # 决定 Stage1 和 Stage2 各自的 extra_feat_dim
    s1_fe = CONFIG['S1_USE_FE'] and CONFIG['USE_FEATURE_ENGINEERING']
    s2_fe = CONFIG['S2_USE_FE'] and CONFIG['USE_FEATURE_ENGINEERING']
    s1_extra_dim = CONFIG['FE_EXTRA_DIMS'] if s1_fe else 0
    s2_extra_dim = CONFIG['FE_EXTRA_DIMS'] if s2_fe else 0

    # 创建模型：以 Stage2 的 extra_feat_dim 为准（若 S1 无 FE 而 S2 有 FE，
    # 则 S1 阶段模型中 extra_encoder 存在但不会被 forward 调用到，
    # 因为 S1 数据不包含额外列；S2 阶段加载 S1 权重后直接使用 extra_encoder）
    # 为保证 S1→S2 权重兼容，模型始终以最大 extra_dim 构建。
    model_extra_dim = max(s1_extra_dim, s2_extra_dim)

    model = ImprovedDualStreamPMSTNet(
        dyn_vars_count=25, 
        window_size=CONFIG['WINDOW_SIZE'],
        hidden_dim=CONFIG['MODEL_HIDDEN_DIM'], 
        num_classes=CONFIG['MODEL_NUM_CLASSES'],
        extra_feat_dim=model_extra_dim,
        dropout=CONFIG['MODEL_DROPOUT']
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # 创建损失函数
    dual_loss = DualBranchLoss(
        alpha_binary=CONFIG['LOSS_ALPHA_BINARY'],
        alpha_fine=CONFIG['LOSS_ALPHA_FINE'],
        alpha_consistency=CONFIG['LOSS_ALPHA_CONSISTENCY'],
        alpha_fp=CONFIG['LOSS_ALPHA_FP'],
        alpha_fog_boost=CONFIG['LOSS_ALPHA_FOG_BOOST'],
        alpha_mist_boost=CONFIG['LOSS_ALPHA_MIST_BOOST'],
        fp_threshold=CONFIG['LOSS_FP_THRESHOLD'],
        loss_type=CONFIG['LOSS_TYPE'],
        focal_gamma=CONFIG['FOCAL_GAMMA'],
        focal_alpha=CONFIG['FOCAL_ALPHA'],
        asym_gamma_neg=CONFIG['ASYM_GAMMA_NEG'],
        asym_gamma_pos=CONFIG['ASYM_GAMMA_POS'],
        asym_clip=CONFIG['ASYM_CLIP'],
        binary_pos_weight=CONFIG['BINARY_POS_WEIGHT'],
        fine_class_weight=[CONFIG['FINE_CLASS_WEIGHT_FOG'], 
                          CONFIG['FINE_CLASS_WEIGHT_MIST'], 
                          CONFIG['FINE_CLASS_WEIGHT_CLEAR']]
    ).to(device)
    
    reg_loss = PhysicsConstrainedRegLoss(alpha=CONFIG['REG_LOSS_ALPHA']).to(device)

    # Stage 1 或 Stage 2
    if not CONFIG['SKIP_STAGE1']:
        # Stage 1: ERA5预训练
        if global_rank == 0:
            print("\n[STAGE 1] ERA5 Pre-training")
            print(f"  use_fe={s1_fe}, extra_dim={s1_extra_dim}")
            print("-" * 70)
        
        s1_train_ds, s1_val_ds, scaler_s1 = load_data_and_scale(
            CONFIG['S1_DATA_DIR'], 
            rank=global_rank, 
            device=device, 
            reuse_scaler=False, 
            window_size=CONFIG['WINDOW_SIZE'],
            world_size=world_size,
            use_fe=s1_fe,
            fe_extra_dims=s1_extra_dim
        )
        
        if global_rank == 0:
            scaler_name = f"scaler_pmst_{exp_id}_{CONFIG['WINDOW_SIZE']}h.pkl"
            joblib.dump(scaler_s1, os.path.join(CONFIG['BASE_PATH'], f"scalers/{scaler_name}"))
        
        s1_val_loader = DataLoader(
            s1_val_ds, 
            batch_size=CONFIG['S1_BATCH_SIZE'], 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        optimizer = optim.AdamW(model.parameters(), 
                               lr=CONFIG['S1_LR_BACKBONE'], 
                               weight_decay=CONFIG['S1_WEIGHT_DECAY'])
        
        train_with_comprehensive_evaluation(
            'S1', model, s1_train_ds, s1_val_loader, optimizer, 
            (dual_loss, reg_loss), scaler_amp, device, global_rank, world_size,
            CONFIG['S1_TOTAL_STEPS'], CONFIG['S1_VAL_INTERVAL'], 
            CONFIG['S1_BATCH_SIZE'], CONFIG['S1_FOG_RATIO'], CONFIG['S1_MIST_RATIO'],
            CONFIG['S1_GRAD_ACCUM'], exp_id
        )
        
        del s1_train_ds, s1_val_ds, s1_val_loader
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # 跳过Stage1，直接在Stage2数据上训练
        if global_rank == 0:
            print("\n[SKIP STAGE 1] Training directly on Stage 2 data")
            print("-" * 70)
        
        scaler_s1 = None

    # Stage 2: Forecast Fine-tuning
    if global_rank == 0:
        print("\n[STAGE 2] Forecast Fine-tuning")
        print(f"  use_fe={s2_fe}, extra_dim={s2_extra_dim}")
        print("-" * 70)
    
    s2_train_ds, s2_val_ds, scaler_s2 = load_data_and_scale(
        CONFIG['S2_DATA_DIR'], 
        scaler=scaler_s1,  # 如果SKIP_STAGE1=True，这里是None，会重新拟合
        rank=global_rank, 
        device=device, 
        reuse_scaler=(scaler_s1 is not None),
        window_size=CONFIG['WINDOW_SIZE'],
        world_size=world_size,
        use_fe=s2_fe,
        fe_extra_dims=s2_extra_dim
    )
    
    # 如果跳过了Stage1，保存新拟合的scaler
    if CONFIG['SKIP_STAGE1'] and global_rank == 0:
        scaler_name = f"scaler_pmst_{exp_id}_{CONFIG['WINDOW_SIZE']}h.pkl"
        joblib.dump(scaler_s2, os.path.join(CONFIG['BASE_PATH'], f"scalers/{scaler_name}"))
    
    s2_val_loader = DataLoader(
        s2_val_ds, 
        batch_size=CONFIG['S2_BATCH_SIZE'], 
        shuffle=False, 
        num_workers=4
    )
    
    # 可选：加载Stage1预训练模型
    if CONFIG['SKIP_STAGE1'] and CONFIG['S1_PRETRAINED_PATH']:
        if global_rank == 0:
            print(f"Loading pretrained model from: {CONFIG['S1_PRETRAINED_PATH']}")
        
        state_dict = torch.load(CONFIG['S1_PRETRAINED_PATH'], map_location=device)
        if world_size > 1:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        
        if global_rank == 0:
            print("✓ Pretrained model loaded successfully")
    
    # Stage2优化器：区分backbone和head的学习率
    params = [
        {'params': [p for n, p in model.named_parameters() 
                   if "head" not in n and "detector" not in n and "classifier" not in n], 
         'lr': CONFIG['S2_LR_BACKBONE']},
        {'params': [p for n, p in model.named_parameters() 
                   if "head" in n or "detector" in n or "classifier" in n], 
         'lr': CONFIG['S2_LR_HEAD']}
    ]
    s2_optimizer = optim.AdamW(params, weight_decay=CONFIG['S2_WEIGHT_DECAY'])
    
    train_with_comprehensive_evaluation(
        'S2', model, s2_train_ds, s2_val_loader, s2_optimizer,
        (dual_loss, reg_loss), scaler_amp, device, global_rank, world_size,
        CONFIG['S2_TOTAL_STEPS'], CONFIG['S2_VAL_INTERVAL'],
        CONFIG['S2_BATCH_SIZE'], CONFIG['S2_FOG_RATIO'], CONFIG['S2_MIST_RATIO'],
        CONFIG['S2_GRAD_ACCUM'], exp_id
    )
    
    if global_rank == 0:
        print("\n" + "="*70)
        print("  Training Completed!")
        print("="*70)
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
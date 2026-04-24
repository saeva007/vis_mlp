import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import os
import shutil
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.preprocessing import RobustScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import pickle
import math
from collections import Counter
import gc
import warnings
import time

warnings.filterwarnings('ignore')

# ==========================================
# 分布式初始化
# ==========================================

def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
    
    return local_rank, global_rank, world_size

def copy_to_local(src_path, local_rank, device_id=None):
    """将数据复制到本地高速存储"""
    filename = os.path.basename(src_path)
    target_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    local_path = os.path.join(target_dir, filename)
    
    if local_rank == 0:
        need_copy = True
        if os.path.exists(local_path):
            if os.path.getsize(local_path) == os.path.getsize(src_path):
                need_copy = False
        
        if need_copy:
            print(f"[Rank 0] Copying {filename} to {target_dir}...", flush=True)
            try:
                tmp_path = local_path + ".tmp"
                shutil.copyfile(src_path, tmp_path)
                os.rename(tmp_path, local_path)
            except Exception as e:
                print(f"[Rank 0] Copy failed: {e}. Using NFS.", flush=True)
                local_path = src_path
    
    if dist.is_initialized():
        if device_id is not None:
            dist.barrier(device_ids=[device_id])
        else:
            dist.barrier()
    
    if not os.path.exists(local_path):
        return src_path
    return local_path

# ==========================================
# 数据集类（使用 mmap）
# ==========================================

class BalancedVisDataset(Dataset):
    def __init__(self, X_path, y_cls, y_reg=None, y_raw=None, scaler=None):
        """
        X_path: .npy文件路径
        y_cls: 分类标签 (0: <500m, 1: 500-1000m, 2: >1000m)
        y_reg: 回归目标（log变换后的能见度）
        y_raw: 原始能见度值（米）
        scaler: RobustScaler对象
        """
        # 使用内存映射模式，不立即加载到内存
        self.X = np.load(X_path, mmap_mode='r')
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.long)
        
        if y_reg is not None:
            self.y_reg = torch.as_tensor(y_reg, dtype=torch.float32)
        else:
            self.y_reg = torch.zeros(len(y_cls), dtype=torch.float32)
        
        if y_raw is not None:
            self.y_raw = torch.as_tensor(y_raw, dtype=torch.float32)
        else:
            self.y_raw = torch.zeros(len(y_cls), dtype=torch.float32)
        
        self.has_scaler = scaler is not None
        if self.has_scaler:
            self.center = scaler.center_.astype(np.float32)
            self.scale = scaler.scale_.astype(np.float32)
            self.scale = np.where(self.scale == 0, 1.0, self.scale)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # 只在需要时加载单个样本
        features = self.X[idx].astype(np.float32)
        
        # 处理异常值
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 标准化
        if self.has_scaler:
            features = (features - self.center) / self.scale
        
        # 裁剪极端值
        features = np.clip(features, -10.0, 10.0)
        
        return (torch.from_numpy(features).float(), 
                self.y_cls[idx], 
                self.y_reg[idx], 
                self.y_raw[idx])

# ==========================================
# 平衡采样器（替代SMOTE）
# ==========================================

class InfiniteBalancedSampler(Sampler):
    """无限循环的平衡采样器"""
    def __init__(self, dataset, batch_size, pos_ratio=0.25, rank=0, world_size=1, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        
        y = np.array(dataset.y_cls)
        # 正样本: 雾(0) + 薄雾(1)
        all_pos = np.where(y <= 1)[0]
        # 负样本: 晴朗(2)
        all_neg = np.where(y == 2)[0]
        
        np.random.seed(seed + rank)
        np.random.shuffle(all_pos)
        np.random.shuffle(all_neg)
        
        # 多卡训练时分割数据
        self.pos_indices = np.array_split(all_pos, world_size)[rank]
        self.neg_indices = np.array_split(all_neg, world_size)[rank]
        
        self.n_pos = int(batch_size * pos_ratio)
        self.n_neg = batch_size - self.n_pos
        
        if rank == 0:
            print(f"[Sampler] Pos (Fog/Mist): {len(all_pos):,}, Neg (Clear): {len(all_neg):,}")
            print(f"[Sampler] Batch: {self.n_pos} Pos ({pos_ratio*100:.1f}%) + {self.n_neg} Neg")

    def __iter__(self):
        epoch_seed = self.seed + self.rank + int(time.time() * 1000) % 10000
        g = torch.Generator()
        g.manual_seed(epoch_seed)
        
        while True:
            # 循环采样，防止样本耗尽
            pos_batch = torch.randint(0, len(self.pos_indices), (self.n_pos,), generator=g).numpy()
            neg_batch = torch.randint(0, len(self.neg_indices), (self.n_neg,), generator=g).numpy()
            
            indices = np.concatenate([self.pos_indices[pos_batch], self.neg_indices[neg_batch]])
            np.random.shuffle(indices)
            yield indices.tolist()

    def __len__(self):
        return 2147483647  # 无限长度

# ==========================================
# 分类标签转换
# ==========================================

def convert_visibility_to_class(visibility, threshold1=500, threshold2=1000):
    """
    将连续能见度值转换为分类标签
    visibility: 单位为公里(km)
    0: < 500m (threshold1=0.5km)
    1: 500-1000m 
    2: > 1000m (threshold2=1.0km)
    """
    classes = np.zeros_like(visibility, dtype=np.int64)
    classes[visibility >= threshold1] = 1
    classes[visibility >= threshold2] = 2
    return classes

# ==========================================
# 损失函数
# ==========================================

class ProgressiveBalancedLoss(nn.Module):
    """渐进式平衡损失函数"""
    def __init__(self, class_weights=None, initial_recall_weight=6.0, 
                 target_recall_weight=3.5, initial_precision_weight=1.2, 
                 target_precision_weight=1.8, total_epochs=4000, gamma=2.0):
        super().__init__()
        self.class_weights = class_weights
        self.initial_recall_weight = initial_recall_weight
        self.target_recall_weight = target_recall_weight
        self.initial_precision_weight = initial_precision_weight
        self.target_precision_weight = target_precision_weight
        self.total_epochs = total_epochs
        self.gamma = gamma
        
    def get_dynamic_weights(self, epoch):
        """根据训练进度获取动态权重"""
        progress = min(epoch / (self.total_epochs * 0.7), 1.0)
        smooth_factor = 1 / (1 + math.exp(-10 * (progress - 0.5)))
        
        recall_weight = self.initial_recall_weight + (self.target_recall_weight - self.initial_recall_weight) * smooth_factor
        precision_weight = self.initial_precision_weight + (self.target_precision_weight - self.initial_precision_weight) * smooth_factor
        
        return recall_weight, precision_weight
    
    def forward(self, inputs, targets, epoch=0):
        recall_weight, precision_weight = self.get_dynamic_weights(epoch)
        
        # Focal Loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 类别权重
        if self.class_weights is not None:
            alpha_t = self.class_weights.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # 召回率增强
        probs = F.softmax(inputs, dim=1)
        _, predicted = torch.max(probs, 1)
        
        low_vis_true_mask = targets <= 1
        high_vis_pred_mask = predicted == 2
        false_negative_mask = low_vis_true_mask & high_vis_pred_mask
        
        if false_negative_mask.any():
            focal_loss[false_negative_mask] = focal_loss[false_negative_mask] * recall_weight
        
        # 精确率增强
        high_vis_true_mask = targets == 2
        low_vis_pred_mask = predicted <= 1
        false_positive_mask = high_vis_true_mask & low_vis_pred_mask
        
        if false_positive_mask.any():
            focal_loss[false_positive_mask] = focal_loss[false_positive_mask] * precision_weight
        
        return focal_loss.mean()

# ==========================================
# 模型架构
# ==========================================

class EnhancedAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        attn_out, attn_weights = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        x_flat = x.squeeze(1)
        gate = self.feature_gate(x_flat)
        x_gated = x_flat * gate
        
        ffn_out = self.ffn(x_gated)
        x_final = self.norm2(x_gated + ffn_out)
        
        return x_final, attn_weights

class BalancedLowVisibilityClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=3, dropout=0.25):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 特征提取
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
            nn.Dropout(dropout)
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim, dropout) for _ in range(4)
        ])
        
        # 注意力机制
        self.attention = EnhancedAttentionBlock(hidden_dim, num_heads=12, dropout=dropout)
        
        # 低能见度检测分支
        self.low_vis_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 精细分类分支
        self.fine_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(num_classes + 1, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        self.apply(self._init_weights)
        
    def _make_residual_block(self, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        for block in self.residual_blocks:
            residual = features
            features = block(features)
            features = F.relu(features + residual)
        
        features, attention_weights = self.attention(features)
        
        low_vis_confidence = self.low_vis_detector(features)
        fine_logits = self.fine_classifier(features)
        
        combined_features = torch.cat([fine_logits, low_vis_confidence], dim=1)
        final_logits = self.fusion_net(combined_features)
        
        return final_logits, low_vis_confidence, attention_weights

# ==========================================
# 数据加载（改进版）
# ==========================================

def load_and_preprocess_balanced_data(path, batch_size=64, threshold1=0.5, threshold2=1.0, 
                                     rank=0, device=None):
    """
    改进的数据加载函数 - 使用mmap和采样器
    """
    data_path = "/public/home/putianshu/vis_mlp/ml_dataset_fe_12h"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if rank == 0:
        print("正在加载数据（使用内存映射）...", flush=True)
    
    # 复制到本地高速存储
    raw_X_train_path = os.path.join(data_path, "X_train.npy")
    raw_X_val_path = os.path.join(data_path, "X_val.npy")
    raw_y_train_path = os.path.join(data_path, "y_train.npy")
    raw_y_val_path = os.path.join(data_path, "y_val.npy")
    
    X_train_path = copy_to_local(raw_X_train_path, local_rank, device_id=local_rank)
    X_val_path = copy_to_local(raw_X_val_path, local_rank, device_id=local_rank)
    y_train_path = copy_to_local(raw_y_train_path, local_rank, device_id=local_rank)
    y_val_path = copy_to_local(raw_y_val_path, local_rank, device_id=local_rank)
    
    # 只加载标签到内存
    y_train = np.load(y_train_path).astype(np.float32)
    y_val = np.load(y_val_path).astype(np.float32)
    
    if rank == 0:
        print(f"  ✓ 训练样本数: {len(y_train):,}", flush=True)
        print(f"  ✓ 验证样本数: {len(y_val):,}", flush=True)
    
    # 转换为分类标签
    y_train_class = convert_visibility_to_class(y_train, threshold1, threshold2)
    y_val_class = convert_visibility_to_class(y_val, threshold1, threshold2)
    
    # 统计分布
    if rank == 0:
        train_counts = Counter(y_train_class)
        total = len(y_train_class)
        print(f"\n原始训练集分布:")
        for cls in [0, 1, 2]:
            count = train_counts[cls]
            print(f"  类别{cls}: {count:,} ({count/total*100:.2f}%)")
    
    # 拟合Scaler（使用子集）
    if rank == 0:
        print("\n正在拟合RobustScaler（使用子集）...", flush=True)
        X_temp = np.load(X_train_path, mmap_mode='r')
        subset_size = min(100000, len(X_temp))
        indices = np.random.choice(len(X_temp), subset_size, replace=False)
        indices.sort()
        
        X_subset = X_temp[indices].copy()
        X_subset = np.nan_to_num(X_subset, nan=0.0)
        
        scaler = RobustScaler()
        scaler.fit(X_subset)
        print(f"  ✓ Scaler已拟合", flush=True)
        
        del X_subset, X_temp
        gc.collect()
        
        # 广播scaler参数
        center_tensor = torch.from_numpy(scaler.center_).float().to(device)
        scale_tensor = torch.from_numpy(scaler.scale_).float().to(device)
        dim_tensor = torch.tensor([len(scaler.center_)], device=device)
    else:
        dim_tensor = torch.tensor([0], device=device)
        center_tensor = None
        scale_tensor = None
    
    if dist.is_initialized():
        dist.broadcast(dim_tensor, src=0)
    
    feat_dim = dim_tensor.item()
    
    if rank != 0:
        center_tensor = torch.zeros(feat_dim, device=device)
        scale_tensor = torch.zeros(feat_dim, device=device)
    
    if dist.is_initialized():
        dist.broadcast(center_tensor, src=0)
        dist.broadcast(scale_tensor, src=0)
    
    if rank != 0:
        scaler = RobustScaler()
        scaler.center_ = center_tensor.cpu().numpy()
        scaler.scale_ = scale_tensor.cpu().numpy()
    
    # 计算类别权重（温和版本）
    class_counts = Counter(y_train_class)
    total_samples = len(y_train_class)
    
    class_weights = torch.FloatTensor([
        np.sqrt(total_samples / (3 * class_counts[0])) if class_counts[0] > 0 else 1.0,
        np.sqrt(total_samples / (3 * class_counts[1])) if class_counts[1] > 0 else 1.0,
        np.sqrt(total_samples / (3 * class_counts[2])) if class_counts[2] > 0 else 1.0,
    ])
    
    if rank == 0:
        print(f"\n类别权重: {class_weights.tolist()}")
    
    # 保存预处理器
    if rank == 0:
        model_dir = os.path.join(path, "model")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'old_mlp_fe.pkl'), 'wb') as f:
            pickle.dump({
                'scaler': scaler,
                'class_weights': class_weights,
                'threshold1': threshold1,
                'threshold2': threshold2,
                'class_counts': class_counts
            }, f)
        print("  ✓ 预处理器已保存\n")
    
    # 创建数据集（使用mmap）
    train_ds = BalancedVisDataset(X_train_path, y_train_class, scaler=scaler)
    val_ds = BalancedVisDataset(X_val_path, y_val_class, scaler=scaler)
    
    return train_ds, val_ds, scaler, class_weights

# ==========================================
# 评估指标
# ==========================================

def calculate_comprehensive_metrics(y_true, y_pred, class_names=None):
    if class_names is None:
        class_names = ['<500m', '500-1000m', '>1000m']
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = np.mean(y_true == y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    recall_500 = recall[0] if len(recall) > 0 else 0
    
    low_vis_true = y_true <= 1
    low_vis_pred = y_pred <= 1
    recall_1000 = np.sum(low_vis_true & low_vis_pred) / max(np.sum(low_vis_true), 1)
    low_vis_precision = np.sum(low_vis_true & low_vis_pred) / max(np.sum(low_vis_pred), 1)
    
    high_vis_true = y_true == 2
    false_positives = np.sum(~high_vis_true & low_vis_pred)
    false_positive_rate = false_positives / max(np.sum(high_vis_true), 1)
    
    return {
        'accuracy': accuracy,
        'recall_500': recall_500,
        'recall_1000': recall_1000,
        'low_vis_precision': low_vis_precision,
        'false_positive_rate': false_positive_rate,
        'false_positives': false_positives,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1_per_class,
        'very_low_vis_recall': recall_500,
        'low_vis_recall': recall_1000,
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

# ==========================================
# 主训练函数
# ==========================================

def main():
    # 初始化分布式
    local_rank, global_rank, world_size = init_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # 配置参数
    base_path = "/public/home/putianshu/vis_mlp"
    input_dim = 330
    hidden_dim = 1320
    num_classes = 3
    batch_size = 64
    num_epochs = 1500
    base_learning_rate = 1e-4
    threshold1 = 500
    threshold2 = 1000
    pos_ratio = 0.25  # 每批次25%正样本
    
    if global_rank == 0:
        print("="*70)
        print("  低能见度分类器训练 - 改进版（无SMOTE，使用平衡采样）")
        print("="*70)
        print(f"  输入维度: {input_dim}")
        print(f"  隐藏层维度: {hidden_dim}")
        print(f"  批次大小: {batch_size}")
        print(f"  正样本比例: {pos_ratio*100:.0f}%")
        print("="*70 + "\n")
    
    # 加载数据
    train_ds, val_ds, scaler, class_weights = load_and_preprocess_balanced_data(
        base_path, batch_size, threshold1, threshold2, global_rank, device
    )
    
    # 验证集DataLoader
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 创建模型
    model = BalancedLowVisibilityClassifier(input_dim, hidden_dim, num_classes, dropout=0.25).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # 损失函数
    criterion = ProgressiveBalancedLoss(
        class_weights=class_weights.to(device),
        total_epochs=num_epochs
    )
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=2e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=2, eta_min=5e-6)
    scaler_amp = GradScaler()
    
    # 训练循环
    best_score = 0.0
    patience_counter = 0
    max_patience = 120
    
    if global_rank == 0:
        print("开始训练...\n")
        print(f"🎯 目标: 500m以下命中率≥65%, 1000m以下命中率≥75%, 整体准确率≥95%\n")
    
    # 使用平衡采样器
    sampler = InfiniteBalancedSampler(
        train_ds,
        batch_size=batch_size,
        pos_ratio=pos_ratio,
        rank=global_rank,
        world_size=world_size
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    train_iter = iter(train_loader)
    steps_per_epoch = 1000  # 每个epoch的步数
    
    for epoch in range(num_epochs):
        model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for step in range(steps_per_epoch):
            try:
                batch_x, batch_y, _, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_x, batch_y, _, _ = next(train_iter)
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                main_logits, low_vis_confidence, _ = model(batch_x)
                loss = criterion(main_logits, batch_y, epoch)
            
            if torch.isnan(loss):
                if global_rank == 0:
                    print("警告: 损失为NaN，跳过此批次")
                continue
            
            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(main_logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        scheduler.step()
        
        # 验证
        if (epoch + 1) % 15 == 0:
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y, _, _ in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    with autocast():
                        main_logits, _, _ = model(batch_x)
                    
                    _, predicted = torch.max(main_logits, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            metrics = calculate_comprehensive_metrics(all_targets, all_preds)
            
            if global_rank == 0:
                train_acc = 100 * correct / total
                avg_loss = total_loss / steps_per_epoch
                
                print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
                print(f"损失: {avg_loss:.4f} | 训练准确率: {train_acc:.2f}%")
                
                print(f"\n🎯 目标指标进展:")
                print(f"  整体准确率:     {metrics['accuracy']:.4f} / 0.95 ({'✅' if metrics['accuracy'] >= 0.95 else '❌'})")
                print(f"  500m以下命中率: {metrics['recall_500']:.4f} / 0.65 ({'✅' if metrics['recall_500'] >= 0.65 else '❌'})")
                print(f"  1000m以下命中率: {metrics['recall_1000']:.4f} / 0.75 ({'✅' if metrics['recall_1000'] >= 0.75 else '❌'})")
                
                print(f"\n📊 性能分析:")
                print(f"  F1分数 - 宏平均: {metrics['f1_macro']:.4f} | 加权平均: {metrics['f1_weighted']:.4f}")
                print(f"  低能见度精确率: {metrics['low_vis_precision']:.4f}")
                print(f"  假阳性率: {metrics['false_positive_rate']:.4f}")
                
                # 计算综合得分
                target_achievement = (
                    min(metrics['recall_500'] / 0.65, 1.0) * 0.3 +
                    min(metrics['recall_1000'] / 0.75, 1.0) * 0.3 +
                    min(metrics['accuracy'] / 0.95, 1.0) * 0.25 +
                    min(metrics['low_vis_precision'] / 0.2, 1.0) * 0.1 +
                    min((1 - metrics['false_positive_rate']) / 0.6, 1.0) * 0.05
                )
                
                print(f"\n🏆 目标达成度: {target_achievement:.4f}")
                
                if target_achievement > best_score:
                    best_score = target_achievement
                    patience_counter = 0
                    
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save({
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'metrics': metrics,
                        'target_achievement': target_achievement,
                        'config': {
                            'input_dim': input_dim,
                            'hidden_dim': hidden_dim,
                            'num_classes': num_classes,
                            'threshold1': threshold1,
                            'threshold2': threshold2
                        }
                    }, os.path.join(base_path, "model/old_mlp_fe.pth"))
                    
                    print("✨ 保存最佳模型")
                    
                    all_targets_met = (
                        metrics['recall_500'] >= 0.65 and
                        metrics['recall_1000'] >= 0.75 and 
                        metrics['accuracy'] >= 0.95
                    )
                    
                    if all_targets_met:
                        print("🎉 所有目标达成！提前结束训练")
                        break
                else:
                    patience_counter += 1
                
                cm = metrics['confusion_matrix']
                print(f"\n📈 混淆矩阵:")
                print(f"         预测: <500m  500-1000m  >1000m")
                for i, true_name in enumerate(['<500m真实', '500-1000m ', '>1000m真实']):
                    print(f"{true_name}: {cm[i,0]:5d}    {cm[i,1]:5d}    {cm[i,2]:5d}")
                print("=" * 70)
            
            model.train()
            
            if patience_counter >= max_patience:
                if global_rank == 0:
                    print(f"早停触发 (patience: {patience_counter})")
                break
    
    if world_size > 1:
        dist.destroy_process_group()
    
    if global_rank == 0:
        print(f"\n🎉 训练完成！最佳目标达成度: {best_score:.4f}")

if __name__ == "__main__":
    main()
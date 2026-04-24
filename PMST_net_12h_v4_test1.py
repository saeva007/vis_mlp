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
# 0. 全局配置
# ==========================================

TARGET_WINDOW_SIZE = 12
BASE_PATH = "/public/home/putianshu/vis_mlp"

S1_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_{TARGET_WINDOW_SIZE}h"
S2_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_finetune_v1_{TARGET_WINDOW_SIZE}h"

CONFIG = {
    'BASE_PATH': BASE_PATH,
    'WINDOW_SIZE': TARGET_WINDOW_SIZE,
    
    'S1_DATA_DIR': S1_DIR, 
    'S1_SUFFIX': f"_{TARGET_WINDOW_SIZE}h_pmst_v2",   
    
    'S2_DATA_DIR': S2_DIR, 
    'S2_SUFFIX': f"_{TARGET_WINDOW_SIZE}h_forecast_v1", 
    
    'S1_TOTAL_STEPS': 30000,      
    'S1_VAL_INTERVAL': 2000,      
    'S1_BATCH_SIZE': 512,         
    'S1_GRAD_ACCUM': 2,
    'S1_FOG_RATIO': 0.18,      # 新增
    'S1_MIST_RATIO': 0.15,     # 新增
    
    'S2_TOTAL_STEPS': 15000,       
    'S2_VAL_INTERVAL': 500,
    'S2_BATCH_SIZE': 256,
    'S2_GRAD_ACCUM': 1,
    'S2_FOG_RATIO': 0.15,      # 新增
    'S2_MIST_RATIO': 0.12,     # 新增
    
    'MIN_PRECISION': 0.15,
    'MIN_CLEAR_ACC': 0.95,
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

def copy_to_local(src_path, local_rank, device_id=None):
    filename = os.path.basename(src_path)
    target_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    local_path = os.path.join(target_dir, filename)
    
    if local_rank == 0:
        need_copy = True
        if os.path.exists(local_path):
            if os.path.getsize(local_path) == os.path.getsize(src_path):
                need_copy = False
        
        if need_copy:
            print(f"[Node Local-0] Copying {filename} to RAM...", flush=True)
            try:
                tmp_path = local_path + ".tmp"
                shutil.copyfile(src_path, tmp_path)
                os.rename(tmp_path, local_path)
            except Exception as e:
                print(f"[Node Local-0] Copy FAILED: {e}. Fallback to NFS.", flush=True)
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
# 2. 改进的分层采样器
# ==========================================

class StratifiedBalancedSampler(Sampler):
    """分层平衡采样器 - 独立控制 Fog/Mist/Clear 比例"""
    def __init__(self, dataset, batch_size, 
                 fog_ratio=0.15, mist_ratio=0.15, clear_ratio=0.70,
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
        # 修改：每次迭代使用不同的种子
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

class DualBranchLoss(nn.Module):
    """双分支损失函数"""
    def __init__(self, alpha_binary=1.0, alpha_fine=1.0, alpha_consistency=0.5):
        super().__init__()
        self.alpha_binary = alpha_binary
        self.alpha_fine = alpha_fine
        self.alpha_consistency = alpha_consistency
        
        # 修改：将权重注册为 buffer
        self.register_buffer('binary_pos_weight', torch.tensor([3.0]))
        self.register_buffer('fine_class_weight', torch.tensor([5.0, 3.0, 1.0]))
        
        self.binary_loss = nn.BCEWithLogitsLoss(pos_weight=self.binary_pos_weight)
        self.fine_loss = nn.CrossEntropyLoss(weight=self.fine_class_weight)
    
    def forward(self, final_logits, low_vis_logit, fine_logits, targets):
        # Binary Loss
        binary_targets = (targets <= 1).float().unsqueeze(1)
        loss_binary = self.binary_loss(low_vis_logit, binary_targets)
        
        # Fine Classification Loss
        loss_fine = self.fine_loss(fine_logits, targets)
        
        # Consistency Loss
        fine_probs = F.softmax(fine_logits, dim=1)
        low_vis_probs = torch.sigmoid(low_vis_logit)
        inconsistency = low_vis_probs * fine_probs[:, 2:3]
        loss_consistency = inconsistency.mean()
        
        total_loss = (
            self.alpha_binary * loss_binary +
            self.alpha_fine * loss_fine +
            self.alpha_consistency * loss_consistency
        )
        
        return total_loss, {
            'binary': loss_binary.item(),
            'fine': loss_fine.item(),
            'consistency': loss_consistency.item()
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
# 4. 模型架构 (保留原有的编码器)
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
# 5. 双分支模型 (核心改进)
# ==========================================

class ImprovedDualStreamPMSTNet(nn.Module):
    """改进的双分支架构模型"""
    def __init__(self, dyn_vars_count=25, window_size=9, 
                 static_cont_dim=5, veg_num_classes=21, veg_emb_dim=16,
                 hidden_dim=512, num_classes=3):
        super().__init__()
        self.dyn_vars = dyn_vars_count
        self.window = window_size
        self.static_cont_dim = static_cont_dim
        
        # 特征提取模块
        self.fog_diagnostics = FogDiagnosticFeatures(window_size, dyn_vars_count)
        self.veg_embedding = nn.Embedding(veg_num_classes, veg_emb_dim)
        
        # 物理指标编码器
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
        
        # 静态特征编码器
        total_static_dim = static_cont_dim + veg_emb_dim
        self.static_encoder = nn.Sequential(
            ChebyKANLayer(total_static_dim, 256, degree=3),
            nn.LayerNorm(256),
            nn.Linear(256, hidden_dim // 2)
        )
        
        # 物理状态流
        self.physical_vars_indices = [0, 1, 10, 12, 19, 20, 22, 23]
        physical_dim = len(self.physical_vars_indices)
        self.physical_stream = PhysicalStateEncoder(physical_dim, hidden_dim)
        
        # 时间演变流
        self.temporal_vars_indices = [2, 3, 4, 5, 6, 7]
        temporal_dim = len(self.temporal_vars_indices)
        self.temporal_stream = GRUWithAttentionEncoder(
            n_vars=temporal_dim, 
            hidden_dim=hidden_dim,
            n_steps=window_size
        )
        
        # 融合层
        fusion_dim = hidden_dim * 2 + hidden_dim // 2 + hidden_dim // 4
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            ChebyKANLayer(hidden_dim, hidden_dim, degree=3)
        )
        
        # ============ 双分支设计 ============
        
        # 分支 1: 低能见度检测器
        self.low_vis_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 分支 2: 精细分类器
        self.fine_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # 回归头
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
        # 特征提取
        split_dyn = self.dyn_vars * self.window
        split_static = split_dyn + self.static_cont_dim
        
        x_dyn_flat = x[:, :split_dyn]
        x_stat_cont = x[:, split_dyn:split_static]
        x_veg_id = x[:, -1].long()
        
        x_dyn_seq = x_dyn_flat.view(-1, self.window, self.dyn_vars)
        
        # 物理诊断
        physics_seq = self.fog_diagnostics(x_dyn_seq)
        physics_seq = physics_seq.permute(0, 2, 1)
        physics_feat = self.physics_encoder(physics_seq)
        
        # 静态特征
        veg_vec = self.veg_embedding(x_veg_id)
        x_static_full = torch.cat([x_stat_cont, veg_vec], dim=1)
        static_feat = self.static_encoder(x_static_full)
        
        # 物理状态
        x_current = x_dyn_seq[:, -1, :]
        x_physical = x_current[:, self.physical_vars_indices]
        physical_feat = self.physical_stream(x_physical)
        
        # 时间演变
        x_temporal = x_dyn_seq[:, :, self.temporal_vars_indices]
        temporal_feat = self.temporal_stream(x_temporal)
        
        # 融合
        combined_feat = torch.cat([
            physical_feat, temporal_feat, static_feat, physics_feat
        ], dim=1)
        embedding = self.fusion_layer(combined_feat)
        
        # 双分支预测
        low_vis_logit = self.low_vis_detector(embedding)
        fine_logits = self.fine_classifier(embedding)
        
        # 自适应融合
        low_vis_prob = torch.sigmoid(low_vis_logit)
        final_logits = fine_logits.clone()
        
        clear_suppression = low_vis_prob * 3.0
        final_logits[:, 2] = final_logits[:, 2] - clear_suppression.squeeze()
        
        clear_boost = (1 - low_vis_prob) * 2.0
        final_logits[:, 2] = final_logits[:, 2] + clear_boost.squeeze()
        
        # 回归
        reg_out = self.reg_head(embedding)
        
        return final_logits, reg_out, low_vis_logit, fine_logits

# ==========================================
# 6. 综合评估指标
# ==========================================

class ComprehensiveMetrics:
    """综合评估系统"""
    def __init__(self, min_precision=0.15, min_clear_acc=0.95):
        self.min_precision = min_precision
        self.min_clear_acc = min_clear_acc
    
    def calculate_class_specific_f2(self, y_true, y_pred, target_class):
        """计算特定类别的 F2 分数"""
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
    
    def calculate_meteorological_metrics(self, y_true, y_pred):
        """计算气象学指标"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        ts = tp / (tp + fp + fn + 1e-6)
        num = 2 * (tp * tn - fp * fn)
        den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        hss = num / (den + 1e-6)
        hits_rand = (tp + fn) * (tp + fp) / (tp + fn + fp + tn + 1e-6)
        ets = (tp - hits_rand) / (tp + fn + fp - hits_rand + 1e-6)
        
        return {'ts': ts, 'hss': hss, 'ets': ets}
    
    def calculate_composite_score(self, fog_f2, mist_f2, clear_acc, overall_acc, low_vis_ts):
        """计算综合得分"""
        score = (
            0.40 * fog_f2 +
            0.25 * mist_f2 +
            0.20 * clear_acc +
            0.10 * overall_acc +
            0.05 * low_vis_ts
        )
        return score
    
    def evaluate_comprehensive(self, model, loader, device):
        """综合评估"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for bx, by_cls, _, _ in loader:
                bx = bx.to(device, non_blocking=True)
                
                final_logits, _, _, _ = model(bx)
                probs = F.softmax(final_logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.append(preds.cpu())
                all_targets.append(by_cls)
        
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        # 各类别指标
        fog_metrics = self.calculate_class_specific_f2(all_targets, all_preds, 0)
        mist_metrics = self.calculate_class_specific_f2(all_targets, all_preds, 1)
        clear_metrics = self.calculate_class_specific_f2(all_targets, all_preds, 2)
        
        # 整体准确率
        overall_acc = (all_preds == all_targets).mean()
        
        # 低能见度指标
        low_vis_true = (all_targets <= 1).astype(int)
        low_vis_pred = (all_preds <= 1).astype(int)
        low_vis_met = self.calculate_meteorological_metrics(low_vis_true, low_vis_pred)
        
        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        
        # Mist 误分析
        mist_samples = (all_targets == 1).sum()
        if mist_samples > 0:
            mist_to_fog = cm[1, 0] / mist_samples
            mist_to_clear = cm[1, 2] / mist_samples
            mist_correct = cm[1, 1] / mist_samples
        else:
            mist_to_fog = mist_to_clear = mist_correct = 0.0
        
        # 综合得分
        composite_score = self.calculate_composite_score(
            fog_metrics['f2'], mist_metrics['f2'], clear_metrics['recall'],
            overall_acc, low_vis_met['ts']
        )
        
        # 约束检查
        constraints_met = {
            'fog_precision_ok': fog_metrics['precision'] >= self.min_precision,
            'mist_precision_ok': mist_metrics['precision'] >= self.min_precision,
            'clear_acc_ok': clear_metrics['recall'] >= self.min_clear_acc,
        }
        
        return {
            'fog': fog_metrics,
            'mist': mist_metrics,
            'clear': clear_metrics,
            'overall_acc': overall_acc,
            'low_vis_metrics': low_vis_met,
            'composite_score': composite_score,
            'mist_error_analysis': {
                'to_fog_rate': mist_to_fog,
                'to_clear_rate': mist_to_clear,
                'correct_rate': mist_correct
            },
            'constraints_met': constraints_met,
            'all_constraints_ok': all(constraints_met.values()),
            'confusion_matrix': cm,
        }
    
    def print_detailed_report(self, metrics, title="Evaluation"):
        """打印详细报告"""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
        
        # 各类别性能
        print("\n【各类别指标】")
        for class_name, class_id in [('Fog', 'fog'), ('Mist', 'mist'), ('Clear', 'clear')]:
            m = metrics[class_id]
            print(f"  {class_name}: P={m['precision']:.4f}, R={m['recall']:.4f}, F2={m['f2']:.4f} "
                  f"(TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
        
        # 整体性能
        print(f"\n【整体性能】")
        print(f"  Overall Acc: {metrics['overall_acc']:.4f}")
        print(f"  Low-Vis TS:  {metrics['low_vis_metrics']['ts']:.4f}")
        
        # 综合得分
        print(f"\n【综合得分】: {metrics['composite_score']:.4f}")
        
        # Mist 误分析
        print(f"\n【Mist 分析】")
        mist_err = metrics['mist_error_analysis']
        print(f"  Correct: {mist_err['correct_rate']:.2%}, "
              f"→Fog: {mist_err['to_fog_rate']:.2%}, "
              f"→Clear: {mist_err['to_clear_rate']:.2%}")
        
        # 约束检查
        print(f"\n【约束检查】")
        constraints = metrics['constraints_met']
        for name, status in constraints.items():
            print(f"  {'✓' if status else '✗'} {name}")
        
        print("\n" + "="*70)

# ==========================================
# 7. 数据处理
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
        features = np.append(features, veg_id)
        
        return torch.from_numpy(features).float(), self.y_cls[idx], self.y_reg[idx], self.y_raw[idx]

def load_data_and_scale(data_dir, scaler=None, rank=0, device=None, reuse_scaler=False, window_size=9):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        print(f"Loading from: {data_dir} (Window: {window_size})", flush=True)
    
    raw_train_path = os.path.join(data_dir, 'X_train.npy')
    raw_val_path = os.path.join(data_dir, 'X_val.npy')
    raw_y_train_path = os.path.join(data_dir, 'y_train.npy')
    raw_y_val_path = os.path.join(data_dir, 'y_val.npy')
    
    train_path = copy_to_local(raw_train_path, local_rank, device_id=local_rank)
    val_path = copy_to_local(raw_val_path, local_rank, device_id=local_rank)
    local_y_train = copy_to_local(raw_y_train_path, local_rank, device_id=local_rank)
    local_y_val = copy_to_local(raw_y_val_path, local_rank, device_id=local_rank)
    
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
    
    train_ds = PMSTDataset(train_path, y_train_cls, y_train_log, y_train_m, scaler=scaler, window_size=window_size)
    val_ds = PMSTDataset(val_path, y_val_cls, y_val_log, y_val_m, scaler=scaler, window_size=window_size)
    
    return train_ds, val_ds, scaler

# ==========================================
# 8. 改进的训练流程
# ==========================================

def train_with_comprehensive_evaluation(
    stage_name, model, train_ds, val_loader, optimizer,
    criterions, scaler_amp, device, rank, world_size,
    total_steps, val_interval, batch_size, 
    fog_ratio, mist_ratio, grad_accum
):
    """改进的训练流程"""
    # 创建分层采样器
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
        min_precision=CONFIG['MIN_PRECISION'],
        min_clear_acc=CONFIG['MIN_CLEAR_ACC']
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            optimizer.zero_grad()
        
        if rank == 0 and step % 100 == 0:
            print(f"\r[{stage_name}] Step {step}/{total_steps} | "
                  f"L_dual: {l_dual.item():.4f} | L_reg: {l_reg.item():.4f}", 
                  end="", flush=True)
        
        # 验证
        if step % val_interval == 0:
            if rank == 0:
                print(f"\n")
            
            metrics = evaluator.evaluate_comprehensive(model, val_loader, device)
            
            if rank == 0:
                evaluator.print_detailed_report(metrics, title=f"{stage_name} @ Step {step}")
                
                win_suffix = f"_{CONFIG['WINDOW_SIZE']}h_"
                
                # 保存多个最优模型
                if metrics['composite_score'] > best_scores['composite']:
                    best_scores['composite'] = metrics['composite_score']
                    save_path = os.path.join(CONFIG['BASE_PATH'], 
                        f"model/pmst_{stage_name.lower()}{win_suffix}best_composite_v4.pth")
                    torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), save_path)
                    print(f"✓ Best Composite: {best_scores['composite']:.4f}")
                
                if metrics['fog']['f2'] > best_scores['fog_f2']:
                    best_scores['fog_f2'] = metrics['fog']['f2']
                    save_path = os.path.join(CONFIG['BASE_PATH'],
                        f"model/pmst_{stage_name.lower()}{win_suffix}best_fog_f2_v4.pth")
                    torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), save_path)
                    print(f"✓ Best Fog F2: {best_scores['fog_f2']:.4f}")
                
                if metrics['mist']['f2'] > best_scores['mist_f2']:
                    best_scores['mist_f2'] = metrics['mist']['f2']
                    save_path = os.path.join(CONFIG['BASE_PATH'],
                        f"model/pmst_{stage_name.lower()}{win_suffix}best_mist_f2_v4.pth")
                    torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), save_path)
                    print(f"✓ Best Mist F2: {best_scores['mist_f2']:.4f}")
                
                if metrics['low_vis_metrics']['ts'] > best_scores['low_vis_ts']:
                    best_scores['low_vis_ts'] = metrics['low_vis_metrics']['ts']
                    save_path = os.path.join(CONFIG['BASE_PATH'],
                        f"model/pmst_{stage_name.lower()}{win_suffix}best_low_vis_ts_v4.pth")
                    torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), save_path)
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
        print("="*70)
        print(f"  Improved Dual-Branch Pipeline (V4) - {CONFIG['WINDOW_SIZE']}h Window")
        print("="*70)
        print(f"  S1: {CONFIG['S1_DATA_DIR']}")
        print(f"  S2: {CONFIG['S2_DATA_DIR']}")
        print(f"  Key Features:")
        print(f"    ✓ Dual-Branch Architecture")
        print(f"    ✓ Stratified Sampling (Fog/Mist/Clear)")
        print(f"    ✓ Comprehensive Metrics")
        print("="*70)

    # Stage 1
    if global_rank == 0:
        print("\n[STAGE 1] ERA5 Pre-training")
        print("-" * 70)
    
    s1_train_ds, s1_val_ds, scaler_s1 = load_data_and_scale(
        CONFIG['S1_DATA_DIR'], rank=global_rank, device=device, 
        reuse_scaler=False, window_size=CONFIG['WINDOW_SIZE']
    )
    
    if global_rank == 0:
        scaler_name = f"scaler_pmst_v4_{CONFIG['WINDOW_SIZE']}h.pkl"
        joblib.dump(scaler_s1, os.path.join(CONFIG['BASE_PATH'], f"scalers/{scaler_name}"))
    
    s1_val_loader = DataLoader(
        s1_val_ds, 
        batch_size=CONFIG['S1_BATCH_SIZE'], 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    model = ImprovedDualStreamPMSTNet(
        dyn_vars_count=25, 
        window_size=CONFIG['WINDOW_SIZE'],
        hidden_dim=512, 
        num_classes=3
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    
    dual_loss = DualBranchLoss(
        alpha_binary=1.0, 
        alpha_fine=1.0, 
        alpha_consistency=0.5
    ).to(device)
    
    reg_loss = PhysicsConstrainedRegLoss(alpha=1.0).to(device)
    

    train_with_comprehensive_evaluation(
        'S1', model, s1_train_ds, s1_val_loader, optimizer, 
        (dual_loss, reg_loss), scaler_amp, device, global_rank, world_size,
        CONFIG['S1_TOTAL_STEPS'], CONFIG['S1_VAL_INTERVAL'], 
        CONFIG['S1_BATCH_SIZE'], CONFIG['S1_FOG_RATIO'], CONFIG['S1_MIST_RATIO'],
        CONFIG['S1_GRAD_ACCUM']
    )
    
    del s1_train_ds, s1_val_ds, s1_val_loader
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 2
    if global_rank == 0:
        print("\n[STAGE 2] Forecast Fine-tuning")
        print("-" * 70)
    
    s2_train_ds, s2_val_ds, _ = load_data_and_scale(
        CONFIG['S2_DATA_DIR'], 
        scaler=scaler_s1, 
        rank=global_rank, 
        device=device, 
        reuse_scaler=True,
        window_size=CONFIG['WINDOW_SIZE']
    )
    
    s2_val_loader = DataLoader(
        s2_val_ds, 
        batch_size=CONFIG['S2_BATCH_SIZE'], 
        shuffle=False, 
        num_workers=4
    )
    
    params = [
        {'params': [p for n, p in model.named_parameters() if "head" not in n and "detector" not in n and "classifier" not in n], 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if "head" in n or "detector" in n or "classifier" in n], 'lr': 5e-5}
    ]
    s2_optimizer = optim.AdamW(params, weight_decay=1e-2)
    
    train_with_comprehensive_evaluation(
        'S2', model, s2_train_ds, s2_val_loader, s2_optimizer,
        (dual_loss, reg_loss), scaler_amp, device, global_rank, world_size,
        CONFIG['S2_TOTAL_STEPS'], CONFIG['S2_VAL_INTERVAL'],
        CONFIG['S2_BATCH_SIZE'], CONFIG['S2_FOG_RATIO'], CONFIG['S2_MIST_RATIO'],
        CONFIG['S2_GRAD_ACCUM']
    )
    
    if global_rank == 0:
        print("\n" + "="*70)
        print("  Training Completed!")
        print("="*70)
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
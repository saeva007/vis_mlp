import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import os
import shutil
import argparse
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import joblib
import warnings
import sys
import time
import random
from datetime import timedelta

# 尝试导入 LightGBM 和 XGBoost
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

warnings.filterwarnings('ignore')

# ==========================================
# 1. 参数解析与配置
# ==========================================

def get_args():
    parser = argparse.ArgumentParser(description="Visibility Forecasting Ablation Study")
    
    parser.add_argument('--model', type=str, required=True, 
                        choices=['ours', 'mlp', 'lstm', 'lightgbm', 'xgboost'],
                        help="选择模型架构")
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'window', 'hourly', 'no_physics', 'no_kan', 'tcn', 'ce_loss'],
                        help="消融变体类型")
    parser.add_argument('--window_size', type=int, default=12, help="时间窗口大小")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--base_path', type=str, default="/public/home/putianshu/vis_mlp")
    parser.add_argument('--save_dir', type=str, default="./checkpoints", help="模型和Scaler保存路径")
    parser.add_argument('--local_rank', type=int, default=-1) 
    
    args = parser.parse_args()
    return args

ARGS = get_args()

EXP_NAME = f"{ARGS.model}_{ARGS.ablation}_w{ARGS.window_size}h"
SAVE_PATH = os.path.join(ARGS.save_dir, EXP_NAME)

# 原始数据路径
DATA_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_finetune_v1_{ARGS.window_size}h"
if ARGS.window_size == 12:
    DATA_DIR = "/public/home/putianshu/vis_mlp/ml_dataset_pmst_finetune_v1_12h"

# ==========================================
# 2. 分布式与数据工具
# ==========================================

def init_distributed():
    if ARGS.model in ['lightgbm', 'xgboost']: 
        return 0, 0, 1
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method='env://', timeout=timedelta(minutes=120))
    return local_rank, dist.get_rank(), dist.get_world_size()

def copy_to_local(src_path, local_rank, device_id=None):
    """将文件从网络存储复制到计算节点的本地存储"""
    filename = os.path.basename(src_path)
    target_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    local_path = os.path.join(target_dir, filename)
    
    if local_rank == 0:
        need_copy = True
        if os.path.exists(local_path):
            if os.path.getsize(local_path) == os.path.getsize(src_path):
                need_copy = False
                print(f"[Data] Found cache in {local_path}, skipping copy.")
        
        if need_copy:
            print(f"[Data] Copying {filename} to {target_dir}...", flush=True)
            try:
                t0 = time.time()
                tmp_path = local_path + ".tmp"
                shutil.copyfile(src_path, tmp_path)
                os.rename(tmp_path, local_path)
                print(f"[Data] Copy finished in {time.time()-t0:.1f}s", flush=True)
            except Exception as e:
                print(f"[Data] Copy FAILED: {e}. Fallback to NFS.", flush=True)
                local_path = src_path

    if dist.is_initialized():
        if device_id is not None:
            dist.barrier(device_ids=[device_id])
        else:
            dist.barrier()
        
    if not os.path.exists(local_path):
        return src_path
    return local_path

def calc_f2_score(precision, recall):
    if precision + recall == 0: return 0.0
    beta2 = 4.0 
    f2 = (1 + beta2) * (precision * recall) / ((beta2 * precision) + recall + 1e-8)
    return f2

class AblationDataset(Dataset):
    def __init__(self, X_path, y_data, scaler=None, mode='window', window_size=9, vars_per_step=25):
        self.X = np.load(X_path, mmap_mode='r')
        
        if isinstance(y_data, str):
             self.y_cls = torch.from_numpy(np.load(y_data)).long()
        else:
             self.y_cls = torch.as_tensor(y_data, dtype=torch.long)

        self.mode = mode
        self.window_size = window_size
        self.vars = vars_per_step
        self.scaler = scaler
        self.dyn_len = window_size * vars_per_step
        
        if self.scaler is not None:
            self.center = self.scaler.center_.astype(np.float32)
            self.scale = self.scaler.scale_.astype(np.float32)
            self.scale = np.where(self.scale == 0, 1.0, self.scale)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx].astype(np.float32) 
        
        # 手动执行 Scaler 变换
        if self.scaler is not None:
            feat_dim = row.shape[0] - 1
            limit = min(feat_dim, self.center.shape[0])
            row[:limit] = (row[:limit] - self.center[:limit]) / self.scale[:limit]

        # 处理 NaN 和 Inf
        row = np.nan_to_num(row, nan=0.0, posinf=10.0, neginf=-10.0)
        
        static_start = self.dyn_len
        static_feats = row[static_start:] 
        
        if self.mode == 'hourly':
            dyn_feats = row[self.dyn_len - self.vars : self.dyn_len]
            features = np.concatenate([dyn_feats, static_feats])
        else:
            features = row
            
        return torch.from_numpy(features).float(), self.y_cls[idx]

# ==========================================
# 3. 完整的模型定义
# ==========================================

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.BatchNorm1d(hidden_dim // 2), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        # 初始化偏置以缓解类别不平衡
        self.net[-1].bias.data = torch.tensor([-1.1, -1.1, 0.0])
        
    def forward(self, x): 
        return self.net(x), None

class BaselineLSTM(nn.Module):
    def __init__(self, dyn_vars, static_vars, window_size, hidden_dim=256, num_classes=3):
        super().__init__()
        self.dyn_vars = dyn_vars
        self.window = window_size
        self.lstm = nn.LSTM(dyn_vars, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.static_net = nn.Sequential(
            nn.Linear(static_vars, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        # 初始化偏置
        self.head[-1].bias.data = torch.tensor([-1.1, -1.1, 0.0])
        
    def forward(self, x):
        split = self.dyn_vars * self.window
        x_dyn = x[:, :split].view(-1, self.window, self.dyn_vars)
        x_static = x[:, split:] 
        
        # 处理潜在的NaN
        x_dyn = torch.nan_to_num(x_dyn, nan=0.0)
        x_static = torch.nan_to_num(x_static, nan=0.0)
        
        out, _ = self.lstm(x_dyn)
        combined = torch.cat([out[:, -1, :], self.static_net(x_static)], dim=1)
        return self.head(combined), None

# ==========================================
# ChebyKAN 层（从原始脚本复制）
# ==========================================
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

# ==========================================
# FogDiagnosticFeatures（从原始脚本复制）
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

# ==========================================
# GRU with Attention（从原始脚本复制）
# ==========================================
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
# TCN Block（用于消融实验）
# ==========================================
class MultiScaleTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 7]):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = (k - 1) // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=pad),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(0.25)
            ))
        self.proj = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, 1)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        return self.proj(torch.cat(outs, dim=1))

# ==========================================
# DualStreamOurs（我们的完整架构）
# ==========================================
class DualStreamOurs(nn.Module):
    def __init__(self, ablation_type, dyn_vars=25, window=9, hidden=512):
        super().__init__()
        self.ablation = ablation_type
        self.use_physics = (ablation_type != 'no_physics')
        self.use_kan = (ablation_type != 'no_kan')
        self.window = window
        self.dyn_vars = dyn_vars
        
        # 植被嵌入
        self.veg_emb = nn.Embedding(21, 16)
        
        # 物理诊断模块（可选）
        if self.use_physics:
            self.fog_diagnostics = FogDiagnosticFeatures(
                window_size=window,
                dyn_vars_count=dyn_vars
            )
            self.physics_encoder = nn.Sequential(
                nn.Conv1d(5, 64, kernel_size=1), 
                nn.GELU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1), 
                nn.Flatten(),
                nn.Linear(128, hidden // 4)
            )
        
        # 时序编码（TCN vs GRU）
        temporal_dim = 6  # [2,3,4,5,6,7] - 降水, 辐射, 风速等
        if ablation_type == 'tcn':
            self.temporal_stream = nn.Sequential(
                MultiScaleTCNBlock(temporal_dim, hidden, kernel_sizes=[3, 7]),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
        else:
            self.temporal_stream = GRUWithAttentionEncoder(
                n_vars=temporal_dim, 
                hidden_dim=hidden,
                n_steps=window
            )
        
        # 物理状态编码（当前时刻）
        physical_dim = 8  # [0, 1, 10, 12, 19, 20, 22, 23]
        self.physical_stream = nn.Sequential(
            nn.Linear(physical_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden)
        )
        
        # 静态特征编码（使用或不使用KAN）
        static_dim = 5 + 16  # 5个连续 + 16维嵌入
        if self.use_kan:
            self.static_encoder = nn.Sequential(
                ChebyKANLayer(static_dim, 256, degree=3),
                nn.LayerNorm(256),
                nn.Linear(256, hidden // 2)
            )
        else:
            self.static_encoder = nn.Sequential(
                nn.Linear(static_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, hidden // 2)
            )
        
        # 融合层
        fusion_dim = hidden * 2 + hidden // 2  # physical + temporal + static
        if self.use_physics:
            fusion_dim += hidden // 4  # + physics
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
        self.cls_head[-1].bias.data = torch.tensor([-1.1, -1.1, 0.0])

    def forward(self, x):
        split_dyn = self.dyn_vars * self.window
        split_static = split_dyn + 5  # 5个连续静态特征
        
        # 分割输入
        x_dyn_flat = x[:, :split_dyn]
        x_stat_cont = x[:, split_dyn:split_static]
        x_veg_id = x[:, -1].long()
        
        # 处理 NaN
        x_dyn_flat = torch.nan_to_num(x_dyn_flat, nan=0.0)
        x_stat_cont = torch.nan_to_num(x_stat_cont, nan=0.0)
        
        # 重塑动态序列
        x_dyn_seq = x_dyn_flat.view(-1, self.window, self.dyn_vars)
        
        feats = []
        
        # 1. 物理诊断特征（可选）
        if self.use_physics:
            physics_seq = self.fog_diagnostics(x_dyn_seq)
            physics_seq = physics_seq.permute(0, 2, 1)  # [B, 5, W]
            physics_feat = self.physics_encoder(physics_seq)
            feats.append(physics_feat)
        
        # 2. 当前物理状态
        x_current = x_dyn_seq[:, -1, :]
        physical_indices = [0, 1, 10, 12, 19, 20, 22, 23]
        x_physical = x_current[:, physical_indices]
        physical_feat = self.physical_stream(x_physical)
        feats.append(physical_feat)
        
        # 3. 时序特征
        temporal_indices = [2, 3, 4, 5, 6, 7]
        x_temporal = x_dyn_seq[:, :, temporal_indices]
        
        if self.ablation == 'tcn':
            x_temporal = x_temporal.permute(0, 2, 1)  # [B, C, L]
            temporal_feat = self.temporal_stream(x_temporal)
        else:
            temporal_feat = self.temporal_stream(x_temporal)
        feats.append(temporal_feat)
        
        # 4. 静态特征
        veg_vec = self.veg_emb(x_veg_id)
        x_static_full = torch.cat([x_stat_cont, veg_vec], dim=1)
        static_feat = self.static_encoder(x_static_full)
        feats.append(static_feat)
        
        # 融合
        combined_feat = torch.cat(feats, dim=1)
        embedding = self.fusion_layer(combined_feat)
        
        logits = self.cls_head(embedding)
        
        return logits, None

# ==========================================
# 4. Loss 函数
# ==========================================

class RecallFocusedLoss(nn.Module):
    """改进的Focal Loss用于类别不平衡"""
    def __init__(self, gamma_pos=1.0, gamma_neg=3.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        
        if alpha is None:
            self.alpha = torch.tensor([5.0, 2.0, 1.0])
        else:
            self.alpha = torch.tensor(alpha)
        
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.device != self.alpha.device:
            self.alpha = self.alpha.to(inputs.device)
        
        # Clamp输入防止NaN
        inputs = torch.clamp(inputs, min=-20, max=20)
            
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        
        focal_weight = torch.ones_like(targets, dtype=torch.float)
        
        pos_mask = targets <= 1
        focal_weight[pos_mask] = (1 - pt[pos_mask] + 1e-6) ** self.gamma_pos
        
        neg_mask = targets == 2
        focal_weight[neg_mask] = (1 - pt[neg_mask] + 1e-6) ** self.gamma_neg
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==========================================
# 5. PyTorch 训练流程
# ==========================================

def train_pytorch(local_rank, global_rank, world_size):
    if global_rank == 0:
        if not os.path.exists(SAVE_PATH): 
            os.makedirs(SAVE_PATH)
        print("="*60)
        print(f"[{ARGS.model.upper()}] Experiment: {EXP_NAME}")
        print(f"  Window Size: {ARGS.window_size}h")
        print(f"  Ablation: {ARGS.ablation}")
        print(f"  Data Dir: {DATA_DIR}")
        print("="*60)

    # 1. 数据复制到本地
    raw_X_train = os.path.join(DATA_DIR, 'X_train.npy')
    raw_y_train = os.path.join(DATA_DIR, 'y_train.npy')
    raw_X_val   = os.path.join(DATA_DIR, 'X_val.npy')
    raw_y_val   = os.path.join(DATA_DIR, 'y_val.npy')

    local_X_train = copy_to_local(raw_X_train, local_rank, device_id=local_rank)
    if global_rank == 0: print("✓ X_train cached")
    
    local_y_train = copy_to_local(raw_y_train, local_rank, device_id=local_rank)
    if global_rank == 0: print("✓ y_train cached")
    
    local_X_val = copy_to_local(raw_X_val, local_rank, device_id=local_rank)
    if global_rank == 0: print("✓ X_val cached")
    
    local_y_val = copy_to_local(raw_y_val, local_rank, device_id=local_rank)
    if global_rank == 0: print("✓ y_val cached")

    # 2. Scaler 处理
    scaler_path = os.path.join(SAVE_PATH, f"{EXP_NAME}_scaler.pkl")
    scaler = None

    if global_rank == 0:
        if os.path.exists(scaler_path):
            print("Loading existing scaler...")
            scaler = joblib.load(scaler_path)
        else:
            print("Fitting RobustScaler on LOCAL data...")
            X_temp = np.load(local_X_train, mmap_mode='r')
            limit = min(len(X_temp), 100000)
            scaler = RobustScaler()
            scaler.fit(X_temp[:limit, :-1]) 
            joblib.dump(scaler, scaler_path)
            print("Scaler fitted and saved.")
    
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
    
    if global_rank != 0:
        scaler = joblib.load(scaler_path)

    # 3. 初始化 Datasets
    if global_rank == 0: print("Initializing Datasets...")
    
    ds_mode = 'hourly' if (ARGS.model == 'mlp' and ARGS.ablation == 'hourly') else 'window'
    
    y_train_mem = np.load(local_y_train)
    y_val_mem = np.load(local_y_val)

    def transform_labels(y):
        y_out = np.zeros_like(y, dtype=int)
        y_out[y >= 500] = 1
        y_out[y >= 1000] = 2
        return y_out

    y_train_cls = transform_labels(y_train_mem)
    y_val_cls = transform_labels(y_val_mem)

    train_ds = AblationDataset(local_X_train, y_train_cls, scaler=scaler, mode=ds_mode, window_size=ARGS.window_size)
    val_ds   = AblationDataset(local_X_val,   y_val_cls,   scaler=scaler, mode=ds_mode, window_size=ARGS.window_size)

    sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=ARGS.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=ARGS.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 4. 模型初始化
    device = torch.device(f"cuda:{local_rank}")
    
    if ARGS.model == 'mlp':
        sample_x, _ = train_ds[0]
        model = BaselineMLP(sample_x.shape[0], hidden_dim=512).to(device)
    elif ARGS.model == 'lstm':
        model = BaselineLSTM(dyn_vars=25, static_vars=6, window_size=ARGS.window_size, hidden_dim=256).to(device)
    elif ARGS.model == 'ours':
        model = DualStreamOurs(
            ablation_type=ARGS.ablation, 
            dyn_vars=25,
            window=ARGS.window_size, 
            hidden=512
        ).to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # 5. 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    # 根据消融类型选择损失函数
    if ARGS.ablation == 'ce_loss':
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([5.0, 2.0, 1.0]).to(device))
    else:
        criterion = RecallFocusedLoss(
            gamma_pos=1.0,
            gamma_neg=3.0,
            alpha=[5.0, 2.0, 1.0]
        )
    
    scaler_amp = GradScaler()

    # 6. 训练循环
    best_f2 = 0.0 
    num_epochs = 20
    
    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()
        
        for i, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            
            # 额外的NaN检查
            if torch.isnan(bx).any() or torch.isinf(bx).any():
                print(f"[WARNING] NaN/Inf in input at step {i}, skipping...")
                continue
            
            with autocast():
                logits, _ = model(bx)
                loss = criterion(logits, by)
            
            # 检查loss是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] NaN/Inf loss at step {i}, skipping...")
                optimizer.zero_grad()
                continue
            
            optimizer.zero_grad()
            scaler_amp.scale(loss).backward()
            
            # 梯度裁剪
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if i % 50 == 0 and global_rank == 0:
                elapsed = time.time() - t0
                speed = 50 / elapsed if i > 0 else 0
                avg_loss = epoch_loss / (num_batches + 1e-8)
                print(f"\rEpoch {epoch} Step {i}/{len(train_loader)} "
                      f"Loss {avg_loss:.4f} | Speed: {speed:.1f} it/s", end="", flush=True)
                t0 = time.time()
        
        if dist.is_initialized():
            dist.barrier(device_ids=[local_rank])

        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=ARGS.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

        if global_rank == 0:
            print(f"\n[Epoch {epoch}] Validating (Distributed)...")

        model.eval()
        
        # 存储当前 GPU 的预测结果
        local_preds = []
        local_targets = []
        
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)

                if torch.isnan(bx).any() or torch.isinf(bx).any():
                    continue
                
                # 关键修改：验证时使用 model.module 避免 DDP 的梯度/Buffer同步开销
                logits, _ = model.module(bx)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                local_preds.append(preds)
                local_targets.append(by)
        
        # 拼接当前 GPU 的结果
        if len(local_preds) > 0:
            local_preds = torch.cat(local_preds)
            local_targets = torch.cat(local_targets)
        else:
            # 防止某个卡数据为空的极端情况
            local_preds = torch.tensor([], device=device)
            local_targets = torch.tensor([], device=device)

        # 收集所有 GPU 的结果
        # 1. 准备容器
        gathered_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
        gathered_targets = [torch.zeros_like(local_targets) for _ in range(world_size)]
        
        # 2. 执行 Gather
        dist.all_gather(gathered_preds, local_preds)
        dist.all_gather(gathered_targets, local_targets)
        
        # 3. 只有 Rank 0 计算指标
        if global_rank == 0:
            # 拼接所有卡的数据并转回 CPU
            all_preds = torch.cat(gathered_preds).cpu().numpy()
            all_targets = torch.cat(gathered_targets).cpu().numpy()
            
            # 计算指标（针对Fog类别，class 0）
            y_true_fog = (all_targets == 0).astype(int)
            y_pred_fog = (all_preds == 0).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true_fog, y_pred_fog).ravel()
            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            ts = tp / (tp + fp + fn + 1e-8)
            f2 = calc_f2_score(precision, recall)
            
            print(f"Epoch {epoch} Metrics:")
            print(f"  Fog F2={f2:.4f} | TS={ts:.4f} | Recall={recall:.4f} | Precision={precision:.4f}")
            
            # 保存最佳模型
            if f2 > best_f2:
                best_f2 = f2
                save_file = os.path.join(SAVE_PATH, f"{EXP_NAME}_best_f2.pth")
                torch.save(model.module.state_dict(), save_file)
                print(f"--> Best F2 model saved: {save_file}")

        # 同步，确保下一轮训练开始前所有进程都完成了验证
        if dist.is_initialized():
            dist.barrier(device_ids=[local_rank])
    
    if global_rank == 0:
        print("="*60)
        print(f"Training completed. Best F2: {best_f2:.4f}")
        print("="*60)

# ==========================================
# 6. LightGBM 训练流程
# ==========================================

def run_lightgbm():
    if lgb is None:
        print("[ERROR] LightGBM not installed!")
        return
    
    print("="*60)
    print(f"[LIGHTGBM] Experiment: {EXP_NAME}")
    print("="*60)
    
    # 加载数据
    print("Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    
    # 转换标签
    def transform_labels(y):
        y_out = np.zeros_like(y, dtype=int)
        y_out[y >= 500] = 1
        y_out[y >= 1000] = 2
        return y_out
    
    y_train_cls = transform_labels(y_train)
    y_val_cls = transform_labels(y_val)
    
    # 移除最后一列（植被ID）作为类别特征单独处理
    X_train_feat = X_train[:, :-1]
    X_val_feat = X_val[:, :-1]
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train_feat, label=y_train_cls)
    val_data = lgb.Dataset(X_val_feat, label=y_val_cls, reference=train_data)
    
    # 参数设置
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1,
        'class_weight': {0: 5.0, 1: 2.0, 2: 1.0}
    }
    
    # 训练
    print("Training LightGBM...")
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # 预测
    y_pred_proba = gbm.predict(X_val_feat, num_iteration=gbm.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 评估
    y_true_fog = (y_val_cls == 0).astype(int)
    y_pred_fog = (y_pred == 0).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true_fog, y_pred_fog).ravel()
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    ts = tp / (tp + fp + fn + 1e-8)
    f2 = calc_f2_score(precision, recall)
    
    print("="*60)
    print(f"LightGBM Results:")
    print(f"  Fog F2={f2:.4f} | TS={ts:.4f} | Recall={recall:.4f} | Precision={precision:.4f}")
    print("="*60)
    
    # 保存模型
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    model_path = os.path.join(SAVE_PATH, f"{EXP_NAME}_model.txt")
    gbm.save_model(model_path)
    print(f"Model saved to {model_path}")

# ==========================================
# 7. XGBoost 训练流程
# ==========================================

def run_xgboost():
    if xgb is None:
        print("[ERROR] XGBoost not installed!")
        return
    
    print("="*60)
    print(f"[XGBOOST] Experiment: {EXP_NAME}")
    print("="*60)
    
    # 加载数据
    print("Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    
    # 转换标签
    def transform_labels(y):
        y_out = np.zeros_like(y, dtype=int)
        y_out[y >= 500] = 1
        y_out[y >= 1000] = 2
        return y_out
    
    y_train_cls = transform_labels(y_train)
    y_val_cls = transform_labels(y_val)
    
    # 移除最后一列
    X_train_feat = X_train[:, :-1]
    X_val_feat = X_val[:, :-1]
    
    # 创建样本权重（类别平衡）
    sample_weights = np.ones(len(y_train_cls))
    sample_weights[y_train_cls == 0] = 5.0
    sample_weights[y_train_cls == 1] = 2.0
    sample_weights[y_train_cls == 2] = 1.0
    
    # 创建XGBoost数据集
    dtrain = xgb.DMatrix(X_train_feat, label=y_train_cls, weight=sample_weights)
    dval = xgb.DMatrix(X_val_feat, label=y_val_cls)
    
    # 参数设置
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'min_child_weight': 3,
        'tree_method': 'gpu_hist',  # 使用GPU加速
        'gpu_id': 0
    }
    
    # 训练
    print("Training XGBoost...")
    evals = [(dtrain, 'train'), (dval, 'val')]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=10
    )
    
    # 预测
    y_pred_proba = bst.predict(dval)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 评估
    y_true_fog = (y_val_cls == 0).astype(int)
    y_pred_fog = (y_pred == 0).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true_fog, y_pred_fog).ravel()
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    ts = tp / (tp + fp + fn + 1e-8)
    f2 = calc_f2_score(precision, recall)
    
    print("="*60)
    print(f"XGBoost Results:")
    print(f"  Fog F2={f2:.4f} | TS={ts:.4f} | Recall={recall:.4f} | Precision={precision:.4f}")
    print("="*60)
    
    # 保存模型
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    model_path = os.path.join(SAVE_PATH, f"{EXP_NAME}_model.json")
    bst.save_model(model_path)
    print(f"Model saved to {model_path}")

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    if ARGS.model == 'lightgbm':
        run_lightgbm()
    elif ARGS.model == 'xgboost':
        run_xgboost()
    else:
        local_rank, global_rank, world_size = init_distributed()
        train_pytorch(local_rank, global_rank, world_size)
        
        if world_size > 1:
            dist.destroy_process_group()
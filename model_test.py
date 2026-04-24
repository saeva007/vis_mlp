import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import RobustScaler
import joblib
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 模型架构定义（与训练脚本保持一致）
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


class DualStreamPMSTNet(nn.Module):
    def __init__(self, dyn_vars_count=25, window_size=9, 
                 static_cont_dim=5, veg_num_classes=21, veg_emb_dim=16,
                 hidden_dim=512, num_classes=3):
        super().__init__()
        self.dyn_vars = dyn_vars_count
        self.window = window_size
        self.static_cont_dim = static_cont_dim
        self.fog_diagnostics = FogDiagnosticFeatures(
            window_size=window_size,
            dyn_vars_count=dyn_vars_count
        )
        
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
        
        self.veg_embedding = nn.Embedding(veg_num_classes, veg_emb_dim)
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
        
        fusion_dim = hidden_dim * 2 + hidden_dim // 2+ hidden_dim // 4
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            ChebyKANLayer(hidden_dim, hidden_dim, degree=3)
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.cls_head[-1].bias.data = torch.tensor([-1.1, -1.1, 0.0])
        
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

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
            physical_feat,
            temporal_feat,
            static_feat,
            physics_feat
        ], dim=1)

        embedding = self.fusion_layer(combined_feat)
        
        logits = self.cls_head(embedding)
        reg_out = self.reg_head(embedding)
        
        return logits, reg_out


# ==========================================
# 2. 数据集定义
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


# ==========================================
# 3. 评估指标计算
# ==========================================

def calculate_meteorological_metrics(y_true, y_pred):
    """计算气象学评价指标：TS, HSS, ETS"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Threat Score (TS / CSI)
    ts = tp / (tp + fp + fn + 1e-6)
    
    # Heidke Skill Score (HSS)
    num = 2 * (tp * tn - fp * fn)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = num / (den + 1e-6)
    
    # Equitable Threat Score (ETS)
    hits_rand = (tp + fn) * (tp + fp) / (tp + fn + fp + tn)
    ets = (tp - hits_rand) / (tp + fn + fp - hits_rand + 1e-6)
    
    return ts, hss, ets


def find_optimal_threshold(probs, y_true, min_precision=0.15):
    """寻找最优阈值"""
    thresholds = np.concatenate([
        np.arange(0.01, 0.1, 0.01),      
        np.arange(0.1, 0.5, 0.05),
        np.arange(0.5, 0.9, 0.1)
    ])
    
    candidate_results = []
    
    for t in thresholds:
        y_pred = (probs > t).astype(int)
        
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        
        ts, hss, ets = calculate_meteorological_metrics(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        rec = tp / (tp + fn + 1e-6)
        prec = tp / (tp + fp + 1e-6)
        
        f2 = 5 * (prec * rec) / (4 * prec + rec + 1e-6)
        
        candidate_results.append({
            'thresh': t,
            'ts': ts,
            'ets': ets,
            'hss': hss,
            'recall': rec,
            'precision': prec,
            'f2': f2,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
    
    # 选择策略：优先考虑精确率>=min_precision的结果中F2最高的
    valid_candidates = [c for c in candidate_results if c['precision'] >= min_precision]
    
    if valid_candidates:
        best_res = max(valid_candidates, key=lambda x: x['f2'])
        selection_note = f"Max F2 (Prec>={min_precision:.0%})"
    else:
        if candidate_results:
            best_res = max(candidate_results, key=lambda x: x['ts'])
            selection_note = f"Fallback: Max TS"
        else:
            best_res = None
            selection_note = "No valid threshold"
    
    return best_res, candidate_results, selection_note


def evaluate_model_comprehensive(model, loader, device, min_precision=0.15):
    """全面评估模型性能"""
    model.eval()
    all_logits = []
    all_targets = []
    all_raw_vis = []
    
    with torch.no_grad():
        for bx, by_cls, _, by_raw in loader:
            bx = bx.to(device, non_blocking=True)
            logits, _ = model(bx)
            all_logits.append(logits.float().cpu())
            all_targets.append(by_cls)
            all_raw_vis.append(by_raw)
    
    all_logits = torch.cat(all_logits)
    probs = F.softmax(all_logits, dim=1).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_raw_vis = torch.cat(all_raw_vis).numpy()
    
    # 总体分类性能
    y_pred_argmax = probs.argmax(axis=1)
    
    results = {
        'overall': {
            'accuracy': (y_pred_argmax == all_targets).mean(),
            'confusion_matrix': confusion_matrix(all_targets, y_pred_argmax).tolist()
        },
        'per_class': {}
    }
    
    # 分类别详细评估
    class_names = ['Fog (0-500m)', 'Mist (500-1000m)', 'Clear (≥1000m)']
    
    for cls_id, cls_name in enumerate(class_names):
        y_true_binary = (all_targets == cls_id).astype(int)
        prob_cls = probs[:, cls_id]
        
        # 寻找最优阈值
        best_res, candidates, note = find_optimal_threshold(prob_cls, y_true_binary, min_precision)
        
        if best_res:
            results['per_class'][cls_name] = {
                'best_threshold': best_res['thresh'],
                'f2_score': best_res['f2'],
                'ts_score': best_res['ts'],
                'ets_score': best_res['ets'],
                'hss_score': best_res['hss'],
                'recall': best_res['recall'],
                'precision': best_res['precision'],
                'tp': int(best_res['tp']),
                'fp': int(best_res['fp']),
                'fn': int(best_res['fn']),
                'tn': int(best_res['tn']),
                'selection_note': note,
                'top5_candidates': sorted(candidates, key=lambda x: x['f2'], reverse=True)[:5]
            }
        else:
            results['per_class'][cls_name] = {
                'error': 'No valid threshold found'
            }
    
    # 二分类评估（雾/轻雾 vs 晴）
    y_true_fog = (all_targets <= 1).astype(int)
    prob_fog = probs[:, 0] + probs[:, 1]  # 雾或轻雾的概率
    
    best_fog, candidates_fog, note_fog = find_optimal_threshold(prob_fog, y_true_fog, min_precision)
    
    if best_fog:
        results['binary_fog_vs_clear'] = {
            'best_threshold': best_fog['thresh'],
            'f2_score': best_fog['f2'],
            'ts_score': best_fog['ts'],
            'ets_score': best_fog['ets'],
            'hss_score': best_fog['hss'],
            'recall': best_fog['recall'],
            'precision': best_fog['precision'],
            'tp': int(best_fog['tp']),
            'fp': int(best_fog['fp']),
            'fn': int(best_fog['fn']),
            'tn': int(best_fog['tn']),
            'selection_note': note_fog
        }
    
    return results


# ==========================================
# 4. 主测试函数
# ==========================================

def test_single_model(window_size, model_path, scaler_path, test_data_dir, device, min_precision=0.15):
    """测试单个模型"""
    print(f"\n{'='*80}")
    print(f"Testing Model: Window Size = {window_size}h")
    print(f"{'='*80}")
    print(f"Model Path: {model_path}")
    print(f"Scaler Path: {scaler_path}")
    print(f"Test Data: {test_data_dir}")
    
    # 加载scaler
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"✓ Scaler loaded")
    
    # 加载测试数据
    X_test_path = os.path.join(test_data_dir, 'X_test.npy')
    y_test_path = os.path.join(test_data_dir, 'y_test.npy')
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Test data not found in {test_data_dir}")
    
    y_test_raw = np.load(y_test_path)
    y_test_m = y_test_raw * 1000.0 if np.max(y_test_raw) < 100 else y_test_raw
    
    def to_class(y_m):
        cls = np.zeros_like(y_m, dtype=np.int64)
        cls[y_m >= 500] = 1
        cls[y_m >= 1000] = 2
        return cls
    
    y_test_cls = to_class(y_test_m)
    y_test_log = np.log1p(y_test_m).astype(np.float32)
    
    test_ds = PMSTDataset(
        X_test_path, 
        y_test_cls, 
        y_test_log, 
        y_test_m, 
        scaler=scaler,
        window_size=window_size
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=512, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Test dataset loaded: {len(test_ds)} samples")
    
    # 加载模型
    model = DualStreamPMSTNet(
        dyn_vars_count=25, 
        window_size=window_size,
        hidden_dim=512, 
        num_classes=3
    ).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✓ Model checkpoint loaded")
    
    # 评估
    print(f"\nEvaluating...")
    results = evaluate_model_comprehensive(model, test_loader, device, min_precision)
    
    return results


def print_results(window_size, results):
    """打印单个模型的评估结果"""
    print(f"\n{'='*80}")
    print(f"Results for Window Size = {window_size}h")
    print(f"{'='*80}")
    
    # 总体准确率
    print(f"\n【Overall Performance】")
    print(f"  Accuracy: {results['overall']['accuracy']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = np.array(results['overall']['confusion_matrix'])
    print(f"                Pred Fog  Pred Mist  Pred Clear")
    print(f"  True Fog      {cm[0,0]:8d}  {cm[0,1]:9d}  {cm[0,2]:10d}")
    print(f"  True Mist     {cm[1,0]:8d}  {cm[1,1]:9d}  {cm[1,2]:10d}")
    print(f"  True Clear    {cm[2,0]:8d}  {cm[2,1]:9d}  {cm[2,2]:10d}")
    
    # 分类别性能
    print(f"\n【Per-Class Performance】")
    for cls_name, metrics in results['per_class'].items():
        if 'error' in metrics:
            print(f"\n  {cls_name}: {metrics['error']}")
            continue
        
        print(f"\n  {cls_name}:")
        print(f"    Selection: {metrics['selection_note']}")
        print(f"    Threshold: {metrics['best_threshold']:.4f}")
        print(f"    F2 Score:  {metrics['f2_score']:.4f}")
        print(f"    TS Score:  {metrics['ts_score']:.4f}")
        print(f"    ETS Score: {metrics['ets_score']:.4f}")
        print(f"    HSS Score: {metrics['hss_score']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
    
    # 二分类性能（雾 vs 晴）
    if 'binary_fog_vs_clear' in results:
        print(f"\n【Binary Classification: Fog/Mist vs Clear】")
        binary = results['binary_fog_vs_clear']
        print(f"    Selection: {binary['selection_note']}")
        print(f"    Threshold: {binary['best_threshold']:.4f}")
        print(f"    F2 Score:  {binary['f2_score']:.4f}")
        print(f"    TS Score:  {binary['ts_score']:.4f}")
        print(f"    ETS Score: {binary['ets_score']:.4f}")
        print(f"    HSS Score: {binary['hss_score']:.4f}")
        print(f"    Recall:    {binary['recall']:.4f}")
        print(f"    Precision: {binary['precision']:.4f}")
        print(f"    TP={binary['tp']}, FP={binary['fp']}, FN={binary['fn']}, TN={binary['tn']}")


def create_comparison_table(all_results):
    """创建对比表格"""
    comparison_data = []
    
    # 提取关键指标
    for window_size, results in all_results.items():
        row = {'Window Size': f"{window_size}h"}
        
        # 总体准确率
        row['Overall Acc'] = f"{results['overall']['accuracy']:.4f}"
        
        # 各类别F2和Recall
        for cls_name in ['Fog (0-500m)', 'Mist (500-1000m)', 'Clear (≥1000m)']:
            if cls_name in results['per_class'] and 'error' not in results['per_class'][cls_name]:
                metrics = results['per_class'][cls_name]
                short_name = cls_name.split()[0]
                row[f'{short_name} F2'] = f"{metrics['f2_score']:.4f}"
                row[f'{short_name} Recall'] = f"{metrics['recall']:.4f}"
                row[f'{short_name} Precision'] = f"{metrics['precision']:.4f}"
                row[f'{short_name} TS'] = f"{metrics['ts_score']:.4f}"
            else:
                short_name = cls_name.split()[0]
                row[f'{short_name} F2'] = "N/A"
                row[f'{short_name} Recall'] = "N/A"
                row[f'{short_name} Precision'] = "N/A"
                row[f'{short_name} TS'] = "N/A"
        
        # 二分类指标
        if 'binary_fog_vs_clear' in results:
            binary = results['binary_fog_vs_clear']
            row['Binary F2'] = f"{binary['f2_score']:.4f}"
            row['Binary Recall'] = f"{binary['recall']:.4f}"
            row['Binary Precision'] = f"{binary['precision']:.4f}"
            row['Binary TS'] = f"{binary['ts_score']:.4f}"
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df


def main():
    """主函数"""
    # ==========================================
    # 配置区域 - 根据实际情况修改
    # ==========================================
    BASE_PATH = "/public/home/putianshu/vis_mlp"
    
    # 定义要测试的窗口大小
    WINDOW_SIZES = [3, 9, 12]
    
    # 定义模型类型（best_f2 或 best_ts）
    MODEL_TYPE = "best_f2"  # 或 "best_ts"
    
    # 最小精确率阈值
    MIN_PRECISION = 0.15
    
    # GPU设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==========================================
    # 测试所有模型
    # ==========================================
    all_results = {}
    
    for window_size in WINDOW_SIZES:
        try:
            # 构建路径
            model_name = f"pmst_s2_{window_size}h_{MODEL_TYPE}_v3_test1.pth"
            model_path = os.path.join(BASE_PATH, "model", model_name)
            
            scaler_name = f"scaler_pmst_balanced_v3_{window_size}h_test1.pkl"
            scaler_path = os.path.join(BASE_PATH, "scalers", scaler_name)
            
            test_data_dir = f"{BASE_PATH}/ml_dataset_pmst_finetune_v1_{window_size}h"
            
            # 测试模型
            results = test_single_model(
                window_size=window_size,
                model_path=model_path,
                scaler_path=scaler_path,
                test_data_dir=test_data_dir,
                device=device,
                min_precision=MIN_PRECISION
            )
            
            all_results[window_size] = results
            print_results(window_size, results)
            
        except Exception as e:
            print(f"\n❌ Error testing {window_size}h model: {e}")
            import traceback
            traceback.print_exc()
    
    # ==========================================
    # 生成对比报告
    # ==========================================
    if all_results:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        comparison_df = create_comparison_table(all_results)
        print(comparison_df.to_string(index=False))
        
        # 保存结果
        output_dir = os.path.join(BASE_PATH, "test_results")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存CSV
        csv_path = os.path.join(output_dir, f"comparison_{MODEL_TYPE}_{timestamp}.csv")
        comparison_df.to_csv(csv_path, index=False)
        print(f"\n✓ Comparison table saved to: {csv_path}")
        
        # 保存完整JSON
        json_path = os.path.join(output_dir, f"detailed_results_{MODEL_TYPE}_{timestamp}.json")
        with open(json_path, 'w') as f:
            # 转换numpy类型为Python原生类型
            json_results = {}
            for k, v in all_results.items():
                json_results[str(k)] = v
            json.dump(json_results, f, indent=2)
        print(f"✓ Detailed results saved to: {json_path}")
        
        print(f"\n{'='*80}")
        print("Testing completed!")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
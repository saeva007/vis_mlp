#!/usr/bin/env python3
"""
能见度预报模型 - DCU多节点分布式训练版本
支持 XGBoost-Dask / LightGBM-Dask 分布式训练
"""

import numpy as np
import pandas as pd
import joblib
import os
import argparse
import warnings
from pathlib import Path
from sklearn.metrics import (
    recall_score, precision_score, f1_score, 
    confusion_matrix, classification_report
)
from sklearn.preprocessing import RobustScaler
import time
import socket
import sys

warnings.filterwarnings('ignore')

# ==========================================
# 分布式环境检测
# ==========================================

def setup_distributed_environment():
    """设置分布式环境变量"""
    # SLURM环境变量
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    
    # 获取主节点信息
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    
    env_info = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'master_addr': master_addr,
        'master_port': master_port,
        'hostname': socket.gethostname(),
    }
    
    if rank == 0:
        print("="*70)
        print("  Distributed Training Environment (DCU)")
        print("="*70)
        print(f"  World Size:    {world_size}")
        print(f"  Master Node:   {master_addr}:{master_port}")
        print(f"  Hostname:      {env_info['hostname']}")
        print("="*70)
    
    return env_info


# ==========================================
# 全局配置
# ==========================================

CONFIG = {
    'BASE_PATH': "/public/home/putianshu/vis_mlp",
    'WINDOW_SIZE': 3,
    
    'S1_DIR_TEMPLATE': "/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_{window}h",
    'S2_DIR_TEMPLATE': "/public/home/putianshu/vis_mlp/ml_dataset_pmst_finetune_v1_{window}h",
    
    'MODEL_TYPE': 'xgboost',  # 'xgboost', 'lightgbm'
    'USE_DASK': True,  # 启用Dask分布式
    'N_JOBS': -1,
    
    'STRATEGY': 'mixed',
    'HANDLE_IMBALANCE': 'focal',
    
    'MIN_PRECISION': 0.15,
    'OPTIMIZE_METRIC': 'f2',
}


# ==========================================
# 物理诊断特征（与单机版相同）
# ==========================================

class PhysicsFeaturesExtractor:
    """提取物理诊断特征（保持与单机版一致）"""
    
    def __init__(self, window_size=3, dyn_vars_count=25):
        self.window_size = window_size
        self.dyn_vars = dyn_vars_count
        
        self.idx = {
            'rh2m': 0, 't2m': 1, 'precip': 2, 'sw_rad': 4,
            'wspd10': 6, 'cape': 9, 'lcc': 10, 't925': 11,
            'rh925': 12, 'dpd': 22, 'inversion': 23, 'zenith': 24
        }
    
    def _extract_var(self, X_seq, var_name, timestep=-1):
        N = X_seq.shape[0]
        X_reshaped = X_seq[:, :(self.window_size * self.dyn_vars)].reshape(
            N, self.window_size, self.dyn_vars
        )
        return X_reshaped[:, timestep, self.idx[var_name]]
    
    def compute_saturation_proximity(self, rh, dpd):
        rh_norm = np.clip(rh / 100.0, 0, 1)
        dpd_weight = 1 / (1 + np.exp(3.0 * (dpd - 1.0)))
        return np.clip(rh_norm * dpd_weight, 0, 1)
    
    def compute_wind_favorability(self, wspd):
        optimal_wspd = 3.5
        return np.exp(-0.5 * ((wspd - optimal_wspd) / 2.5) ** 2)
    
    def compute_stability_index(self, inversion, wspd):
        wspd_safe = np.maximum(wspd, 0.5)
        ri = inversion / (wspd_safe ** 2 + 0.1)
        return np.tanh(ri / 2.0)
    
    def compute_radiative_cooling_potential(self, sw_rad, lcc, zenith):
        is_night = (zenith > 90.0).astype(float)
        clear_sky = np.clip(1.0 - lcc / 0.3, 0, 1)
        sw_linear = np.clip(np.expm1(sw_rad), 0, None)
        rad_intensity = 1.0 - np.clip(sw_linear / 800.0, 0, 1)
        return is_night * clear_sky * rad_intensity
    
    def compute_vertical_moisture_transport(self, rh2m, rh925):
        return np.clip((rh2m - rh925) / 100.0, -1, 1)
    
    def extract_all_physics_features(self, X):
        rh2m = self._extract_var(X, 'rh2m', -1)
        dpd = self._extract_var(X, 'dpd', -1)
        wspd = self._extract_var(X, 'wspd10', -1)
        inversion = self._extract_var(X, 'inversion', -1)
        sw_rad = self._extract_var(X, 'sw_rad', -1)
        lcc = self._extract_var(X, 'lcc', -1)
        zenith = self._extract_var(X, 'zenith', -1)
        rh925 = self._extract_var(X, 'rh925', -1)
        
        features = {
            'saturation_proximity': self.compute_saturation_proximity(rh2m, dpd),
            'wind_favorability': self.compute_wind_favorability(wspd),
            'stability_index': self.compute_stability_index(inversion, wspd),
            'radiative_cooling': self.compute_radiative_cooling_potential(
                sw_rad, lcc, zenith
            ),
            'vertical_moisture': self.compute_vertical_moisture_transport(rh2m, rh925),
        }
        
        for var_name in ['rh2m', 't2m', 'wspd10', 'dpd']:
            var_seq = np.array([
                self._extract_var(X, var_name, t) 
                for t in range(self.window_size)
            ]).T
            
            features[f'{var_name}_mean'] = var_seq.mean(axis=1)
            features[f'{var_name}_std'] = var_seq.std(axis=1)
            features[f'{var_name}_trend'] = var_seq[:, -1] - var_seq[:, 0]
        
        return pd.DataFrame(features)


# ==========================================
# 数据加载（与单机版相同）
# ==========================================

def load_and_preprocess_data(data_dir, window_size=3, scaler=None, 
                             add_domain_feature=False, domain_label=0,
                             rank=0):
    """加载数据（仅在rank 0上打印日志）"""
    if rank == 0:
        print(f"Loading data from: {data_dir}")
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_train_raw = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val_raw = np.load(os.path.join(data_dir, 'y_val.npy'))
    
    y_train_m = y_train_raw * 1000 if y_train_raw.max() < 100 else y_train_raw
    y_val_m = y_val_raw * 1000 if y_val_raw.max() < 100 else y_val_raw
    
    def to_class(y_m):
        cls = np.zeros_like(y_m, dtype=np.int64)
        cls[y_m >= 500] = 1
        cls[y_m >= 1000] = 2
        return cls
    
    y_train_cls = to_class(y_train_m)
    y_val_cls = to_class(y_val_m)
    
    dyn_feat_size = window_size * 25
    X_train_dyn = X_train[:, :dyn_feat_size]
    X_val_dyn = X_val[:, :dyn_feat_size]
    
    X_train_static = X_train[:, dyn_feat_size:-1]
    X_val_static = X_val[:, dyn_feat_size:-1]
    
    veg_train = X_train[:, -1].astype(int)
    veg_val = X_val[:, -1].astype(int)
    
    log_mask = np.zeros(dyn_feat_size, dtype=bool)
    for t in range(window_size):
        offset = t * 25
        for idx in [offset + 2, offset + 4, offset + 9]:
            if idx < dyn_feat_size:
                log_mask[idx] = True
    
    X_train_dyn = np.nan_to_num(X_train_dyn, nan=0.0)
    X_val_dyn = np.nan_to_num(X_val_dyn, nan=0.0)
    
    X_train_dyn[:, log_mask] = np.log1p(np.maximum(X_train_dyn[:, log_mask], 0))
    X_val_dyn[:, log_mask] = np.log1p(np.maximum(X_val_dyn[:, log_mask], 0))
    
    if scaler is None:
        scaler = RobustScaler()
        X_train_dyn_scaled = scaler.fit_transform(X_train_dyn)
    else:
        X_train_dyn_scaled = scaler.transform(X_train_dyn)
    
    X_val_dyn_scaled = scaler.transform(X_val_dyn)
    
    X_train_combined = np.concatenate([X_train_dyn_scaled, X_train_static], axis=1)
    X_val_combined = np.concatenate([X_val_dyn_scaled, X_val_static], axis=1)
    
    if add_domain_feature:
        domain_train = np.full((X_train_combined.shape[0], 1), domain_label)
        domain_val = np.full((X_val_combined.shape[0], 1), domain_label)
        X_train_combined = np.concatenate([X_train_combined, domain_train], axis=1)
        X_val_combined = np.concatenate([X_val_combined, domain_val], axis=1)
    
    if rank == 0:
        print(f"  Train: {X_train_combined.shape}, Val: {X_val_combined.shape}")
        print(f"  Class distribution - Train: {np.bincount(y_train_cls)}, Val: {np.bincount(y_val_cls)}")
    
    return X_train_combined, X_val_combined, y_train_cls, y_val_cls, scaler, y_train_m, y_val_m


def add_physics_features(X, y_raw, window_size=3):
    """添加物理特征"""
    extractor = PhysicsFeaturesExtractor(window_size=window_size)
    physics_df = extractor.extract_all_physics_features(X)
    
    if y_raw is not None:
        physics_df['vis_log'] = np.log1p(y_raw)
    
    X_enhanced = np.concatenate([X, physics_df.values], axis=1)
    
    return X_enhanced, list(physics_df.columns)


# ==========================================
# 分布式模型训练
# ==========================================

def get_distributed_model(model_type='xgboost', params=None, use_dask=True, env_info=None):
    """获取分布式梯度提升树模型
    
    Args:
        model_type: 'xgboost' or 'lightgbm'
        params: 自定义参数
        use_dask: 是否使用Dask分布式
        env_info: 分布式环境信息
    
    Returns:
        model: 分布式模型实例
        client: Dask客户端（如果使用Dask）
    """
    client = None
    
    if use_dask:
        from dask.distributed import Client, LocalCluster
        
        # 设置Dask集群
        if env_info and env_info['world_size'] > 1:
            # 多节点模式：连接到scheduler
            scheduler_address = f"{env_info['master_addr']}:{env_info['master_port']}"
            
            if env_info['rank'] == 0:
                print(f"Connecting to Dask scheduler: {scheduler_address}")
            
            try:
                client = Client(scheduler_address, timeout='60s')
            except:
                # Fallback: 本地集群
                if env_info['rank'] == 0:
                    print("Failed to connect to scheduler, using local cluster")
                    cluster = LocalCluster(n_workers=4, threads_per_worker=8)
                    client = Client(cluster)
        else:
            # 单节点模式：本地集群
            cluster = LocalCluster(n_workers=4, threads_per_worker=8)
            client = Client(cluster)
        
        if env_info and env_info['rank'] == 0:
            print(f"Dask cluster started: {client}")
    
    if model_type == 'xgboost':
        if use_dask:
            import xgboost as xgb
            default_params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'hist',  # DCU兼容的方法
                'verbosity': 1,
            }
            if params:
                default_params.update(params)
            
            # 使用DaskXGBClassifier
            model = xgb.dask.DaskXGBClassifier(**default_params, client=client)
        else:
            import xgboost as xgb
            default_params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'tree_method': 'hist',
                'n_jobs': -1,
            }
            if params:
                default_params.update(params)
            model = xgb.XGBClassifier(**default_params)
    
    elif model_type == 'lightgbm':
        if use_dask:
            import lightgbm as lgb
            default_params = {
                'objective': 'multiclass',
                'num_class': 3,
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'device': 'cpu',  # DCU暂不直接支持，用CPU
                'verbose': -1,
            }
            if params:
                default_params.update(params)
            
            # 使用DaskLGBMClassifier
            model = lgb.dask.DaskLGBMClassifier(**default_params, client=client)
        else:
            import lightgbm as lgb
            default_params = {
                'objective': 'multiclass',
                'num_class': 3,
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'n_jobs': -1,
                'verbose': -1,
            }
            if params:
                default_params.update(params)
            model = lgb.LGBMClassifier(**default_params)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, client


def handle_class_imbalance(X_train, y_train, method='focal', rank=0):
    """处理类别不平衡"""
    from collections import Counter
    
    if rank == 0:
        print(f"Handling class imbalance with method: {method}")
        original_dist = Counter(y_train)
        print(f"  Original distribution: {original_dist}")
    
    if method == 'focal':
        weights_map = {0: 5.0, 1: 2.0, 2: 1.0}
        sample_weight = np.array([weights_map[y] for y in y_train])
        X_resampled, y_resampled = X_train, y_train
        if rank == 0:
            print(f"  Using focal weights: {weights_map}")
    
    elif method == 'class_weight':
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight('balanced', y_train)
        X_resampled, y_resampled = X_train, y_train
    
    else:
        X_resampled, y_resampled = X_train, y_train
        sample_weight = None
    
    return X_resampled, y_resampled, sample_weight


# ==========================================
# 评估函数（与单机版相同）
# ==========================================

def calculate_meteorological_metrics(y_true, y_pred):
    """计算气象评估指标"""
    y_true_binary = (y_true <= 1).astype(int)
    y_pred_binary = (y_pred <= 1).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    ts = tp / (tp + fp + fn + 1e-6)
    
    num = 2 * (tp * tn - fp * fn)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = num / (den + 1e-6)
    
    hits_rand = (tp + fn) * (tp + fp) / (tp + fn + fp + tn)
    ets = (tp - hits_rand) / (tp + fn + fp - hits_rand + 1e-6)
    
    return ts, hss, ets


def evaluate_with_threshold_search(model, X_val, y_val, min_precision=0.15, 
                                   optimize_metric='f2', return_details=False):
    """阈值搜索评估"""
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_val)
        
        # Dask返回的可能是Dask Array，需要compute
        if hasattr(probs, 'compute'):
            probs = probs.compute()
        
        prob_fog = probs[:, 0]
    else:
        y_pred = model.predict(X_val)
        if hasattr(y_pred, 'compute'):
            y_pred = y_pred.compute()
        prob_fog = (y_pred == 0).astype(float)
    
    y_true_fog = (y_val == 0).astype(int)
    
    thresholds = np.concatenate([
        np.arange(0.01, 0.1, 0.01),
        np.arange(0.1, 0.5, 0.05),
        np.arange(0.5, 0.9, 0.1)
    ])
    
    candidates = []
    
    for thresh in thresholds:
        y_pred_fog = (prob_fog > thresh).astype(int)
        
        if y_pred_fog.sum() == 0 or y_pred_fog.sum() == len(y_pred_fog):
            continue
        
        ts, hss, ets = calculate_meteorological_metrics(y_true_fog, y_pred_fog)
        
        tn, fp, fn, tp = confusion_matrix(y_true_fog, y_pred_fog).ravel()
        recall = tp / (tp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        f2 = 5 * (precision * recall) / (4 * precision + recall + 1e-6)
        
        candidates.append({
            'threshold': thresh,
            'ts': ts,
            'hss': hss,
            'ets': ets,
            'recall': recall,
            'precision': precision,
            'f2': f2,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        })
    
    valid_candidates = [c for c in candidates if c['precision'] >= min_precision]
    
    if valid_candidates:
        if optimize_metric == 'ts':
            best = max(valid_candidates, key=lambda x: x['ts'])
            note = f"Max TS (Prec>={min_precision:.0%})"
        elif optimize_metric == 'f2':
            best = max(valid_candidates, key=lambda x: x['f2'])
            note = f"Max F2 (Prec>={min_precision:.0%})"
        else:
            best = max(valid_candidates, key=lambda x: x['recall'])
            note = f"Max Recall (Prec>={min_precision:.0%})"
    else:
        if candidates:
            best = max(candidates, key=lambda x: x['ts'])
            note = f"Fallback: Max TS (Precision < {min_precision:.0%})"
        else:
            best = {
                'threshold': 0.5, 'ts': 0.0, 'hss': 0.0, 'ets': 0.0,
                'recall': 0.0, 'precision': 0.0, 'f2': 0.0,
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
            }
            note = "No valid threshold found"
    
    result = {
        'selection_note': note,
        **best
    }
    
    if return_details:
        result['all_candidates'] = sorted(candidates, key=lambda x: x['f2'], reverse=True)[:10]
    
    return result


# ==========================================
# 分布式训练主函数
# ==========================================

def train_distributed_model(model_type='xgboost', strategy='mixed', window_size=3,
                           handle_imbalance='focal', use_dask=True,
                           min_precision=0.15, optimize_metric='f2'):
    """分布式训练主函数"""
    
    # 设置分布式环境
    env_info = setup_distributed_environment()
    rank = env_info['rank']
    
    base_path = CONFIG['BASE_PATH']
    s1_dir = CONFIG['S1_DIR_TEMPLATE'].format(window=window_size)
    s2_dir = CONFIG['S2_DIR_TEMPLATE'].format(window=window_size)
    
    if rank == 0:
        print("="*70)
        print(f"  Distributed Visibility Forecasting - {model_type.upper()}")
        print("="*70)
        print(f"  Strategy: {strategy}")
        print(f"  Window Size: {window_size}h")
        print(f"  Dask Enabled: {use_dask}")
        print("="*70)
    
    # 加载数据（所有节点都加载）
    if strategy == 'mixed':
        if rank == 0:
            print("\n[Strategy] Mixed Training (ERA5 + Forecast)")
        
        X_s1_train, X_s1_val, y_s1_train, y_s1_val, scaler, y_s1_train_raw, _ = \
            load_and_preprocess_data(s1_dir, window_size, add_domain_feature=True, 
                                    domain_label=0, rank=rank)
        
        X_s2_train, X_s2_val, y_s2_train, y_s2_val, _, y_s2_train_raw, _ = \
            load_and_preprocess_data(s2_dir, window_size, scaler=scaler, 
                                    add_domain_feature=True, domain_label=1, rank=rank)
        
        # 添加物理特征
        X_s1_train, phys_cols = add_physics_features(X_s1_train, y_s1_train_raw, window_size)
        X_s1_val, _ = add_physics_features(X_s1_val, None, window_size)
        X_s2_train, _ = add_physics_features(X_s2_train, y_s2_train_raw, window_size)
        X_s2_val, _ = add_physics_features(X_s2_val, None, window_size)
        
        # 合并数据
        X_train_mixed = np.vstack([X_s1_train, X_s2_train])
        y_train_mixed = np.concatenate([y_s1_train, y_s2_train])
        
        if rank == 0:
            print(f"  Mixed training set: {X_train_mixed.shape}")
            print(f"  Class distribution: {np.bincount(y_train_mixed)}")
        
        # 处理不平衡
        X_train_mixed, y_train_mixed, sample_weight = handle_class_imbalance(
            X_train_mixed, y_train_mixed, method=handle_imbalance, rank=rank
        )
        
        # 转换为Dask DataFrame/Array（如果使用Dask）
        if use_dask:
            import dask.array as da
            import dask.dataframe as dd
            
            # 分块大小（根据内存调整）
            chunk_size = len(X_train_mixed) // (env_info['world_size'] * 4)
            
            X_train_dask = da.from_array(X_train_mixed, chunks=(chunk_size, X_train_mixed.shape[1]))
            y_train_dask = da.from_array(y_train_mixed, chunks=chunk_size)
            
            if rank == 0:
                print(f"  Dask chunks: {X_train_dask.chunks}")
        else:
            X_train_dask = X_train_mixed
            y_train_dask = y_train_mixed
        
        # 获取分布式模型
        model, client = get_distributed_model(
            model_type, use_dask=use_dask, env_info=env_info
        )
        
        # 训练
        if rank == 0:
            print("\n[Training] Starting distributed training...")
            start_time = time.time()
        
        fit_params = {}
        if sample_weight is not None and not use_dask:
            fit_params['sample_weight'] = sample_weight
        
        if use_dask:
            # Dask训练不需要eval_set（会自动分布式评估）
            model.fit(X_train_dask, y_train_dask, **fit_params)
        else:
            if model_type == 'xgboost':
                fit_params['eval_set'] = [(X_s2_val, y_s2_val)]
                fit_params['verbose'] = 50
            model.fit(X_train_dask, y_train_dask, **fit_params)
        
        if rank == 0:
            elapsed = time.time() - start_time
            print(f"  Training completed in {elapsed:.1f}s")
        
        final_model = model
        final_val_X, final_val_y = X_s2_val, y_s2_val
        
        # 关闭Dask客户端
        if client is not None and rank == 0:
            client.close()
    
    else:
        raise ValueError(f"Strategy '{strategy}' not implemented for distributed training")
    
    # 评估（仅在rank 0上）
    if rank == 0:
        print("\n" + "="*70)
        print("[Final Evaluation on Forecast Validation Set]")
        
        result_final = evaluate_with_threshold_search(
            final_model, final_val_X, final_val_y,
            min_precision=min_precision,
            optimize_metric=optimize_metric,
            return_details=True
        )
        
        print(f"  {result_final['selection_note']}")
        print(f"  TS Score:     {result_final['ts']:.4f}")
        print(f"  F2 Score:     {result_final['f2']:.4f}")
        print(f"  Recall (Fog): {result_final['recall']:.1%}")
        print(f"  Precision:    {result_final['precision']:.1%}")
        print(f"  Threshold:    {result_final['threshold']:.4f}")
        
        # 保存模型
        model_path = os.path.join(
            base_path,
            f"model/{model_type}_{strategy}_{window_size}h_distributed.pkl"
        )
        
        # 提取底层模型（Dask包装器需要特殊处理）
        if use_dask:
            if hasattr(final_model, '_Booster'):
                joblib.dump(final_model._Booster, model_path)
            else:
                # Fallback: 保存整个模型
                joblib.dump(final_model, model_path)
        else:
            joblib.dump(final_model, model_path)
        
        scaler_path = os.path.join(
            base_path,
            f"scalers/scaler_{model_type}_{window_size}h_distributed.pkl"
        )
        joblib.dump(scaler, scaler_path)
        
        print(f"\n  ✓ Model saved to: {model_path}")
        print(f"  ✓ Scaler saved to: {scaler_path}")
        print("="*70)
    
    return final_model if rank == 0 else None


# ==========================================
# 命令行接口
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description='能见度预报 - DCU分布式训练')
    
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm'],
                       help='模型类型')
    
    parser.add_argument('--strategy', type=str, default='mixed',
                       choices=['mixed'],
                       help='训练策略（分布式仅支持mixed）')
    
    parser.add_argument('--window', type=int, default=3,
                       help='时间窗口大小（小时）')
    
    parser.add_argument('--imbalance', type=str, default='focal',
                       choices=['focal', 'class_weight', 'none'],
                       help='类别不平衡处理方法')
    
    parser.add_argument('--use-dask', action='store_true', default=True,
                       help='使用Dask分布式（推荐）')
    
    parser.add_argument('--no-dask', dest='use_dask', action='store_false',
                       help='不使用Dask（单机模式）')
    
    parser.add_argument('--min-precision', type=float, default=0.15,
                       help='最小精确率约束')
    
    parser.add_argument('--optimize', type=str, default='f2',
                       choices=['ts', 'f2', 'recall'],
                       help='优化目标指标')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 更新配置
    CONFIG['MODEL_TYPE'] = args.model
    CONFIG['STRATEGY'] = args.strategy
    CONFIG['WINDOW_SIZE'] = args.window
    CONFIG['HANDLE_IMBALANCE'] = args.imbalance
    CONFIG['USE_DASK'] = args.use_dask
    CONFIG['MIN_PRECISION'] = args.min_precision
    CONFIG['OPTIMIZE_METRIC'] = args.optimize
    
    # 训练
    model = train_distributed_model(
        model_type=args.model,
        strategy=args.strategy,
        window_size=args.window,
        handle_imbalance=args.imbalance,
        use_dask=args.use_dask,
        min_precision=args.min_precision,
        optimize_metric=args.optimize
    )
    
    # 同步所有节点
    env_info = setup_distributed_environment()
    if env_info['rank'] == 0:
        print("\n✓ Distributed training completed successfully!")
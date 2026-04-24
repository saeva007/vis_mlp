import argparse
import numpy as np
import os
import joblib
import time
import warnings
import json
import gc
from sklearn.metrics import confusion_matrix

# 导入模型库
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
except ImportError:
    cb = None

warnings.filterwarnings('ignore')

# ==========================================
# 0. 配置与参数解析
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="GBDT Visibility Forecasting (Optimized)")
    
    # --- 核心控制 ---
    parser.add_argument('--model_type', type=str, default='lightgbm', 
                        choices=['xgboost', 'lightgbm', 'catboost'])
    parser.add_argument('--only_stage2', action='store_true', 
                        help='If set, skip Stage 1 and train ONLY on Stage 2 data.')
    
    # --- 树模型核心参数 ---
    parser.add_argument('--max_depth', type=int, default=10, 
                        help='Max tree depth. Suggest 6-10 for XGB/Cat, -1 or deep for LGBM.')
    
    parser.add_argument('--window_size', type=int, default=12, help='Time window size')
    parser.add_argument('--base_path', type=str, default="/public/home/putianshu/vis_mlp")
    
    # 训练参数
    parser.add_argument('--n_estimators_s1', type=int, default=10000, help='Iterations for Stage 1')
    parser.add_argument('--n_estimators_s2', type=int, default=10000, help='Iterations for Stage 2')
    parser.add_argument('--learning_rate', type=float, default=0.03)
    parser.add_argument('--num_threads', type=int, default=32)
    parser.add_argument('--pos_weight', type=float, default=3.0)
    
    return parser.parse_args()

args = parse_args()

# 路径配置
S1_DIR = f"{args.base_path}/ml_dataset_pmst_v5_aligned_{args.window_size}h"
S2_DIR = f"{args.base_path}/ml_dataset_pmst_finetune_v1_{args.window_size}h"
MODEL_DIR = os.path.join(args.base_path, "model_gbdt")
os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# 1. 高效数据加载与处理
# ==========================================

def get_sample_weights_vectorized(y_arr, class_weight_dict):
    """
    向量化计算权重，替代低效的分块循环。
    y_arr 已经在内存中，利用 numpy 广播极快。
    """
    print("Computing sample weights (Vectorized)...", flush=True)
    # 创建默认权重数组 (默认为1.0)
    weights = np.ones(y_arr.shape, dtype=np.float32)
    
    # 只处理非 1.0 的权重
    for cls, w in class_weight_dict.items():
        if w != 1.0:
            weights[y_arr == cls] = w
            
    return weights

def load_data(data_dir, stage='S1'):
    print(f"[{stage}] Loading data from {data_dir} (mmap mode)...", flush=True)
    
    # 训练集 X 保持 mmap，防止内存溢出
    X_train_raw = np.load(os.path.join(data_dir, 'X_train.npy'), mmap_mode='r')
    
    # 标签 y 数据量小，直接读入内存以便快速处理 (Label Encode & Weight Calc)
    y_train_raw = np.load(os.path.join(data_dir, 'y_train.npy')) 
    
    # 【重要优化】验证集 X 也开启 mmap，防止 Stage 2 验证集过大导致 OOM
    X_val_raw = np.load(os.path.join(data_dir, 'X_val.npy'), mmap_mode='r')
    y_val_raw = np.load(os.path.join(data_dir, 'y_val.npy'))

    # 标签预处理逻辑
    def process_labels(y):
        # 确保是内存中的 array
        y_mem = np.array(y, copy=False) 
        # 兼容之前的单位问题 (如果标签很小可能是单位不对)
        if np.max(y_mem) < 100: 
            y_mem = y_mem * 1000.0
        
        # 转换为分类: 0:Fog(<500), 1:Mist(500-1000), 2:Clear(>1000)
        cls = np.zeros_like(y_mem, dtype=int)
        cls[y_mem >= 500] = 1
        cls[y_mem >= 1000] = 2
        return cls

    y_train_cls = process_labels(y_train_raw)
    y_val_cls = process_labels(y_val_raw)
    
    return X_train_raw, y_train_cls, X_val_raw, y_val_cls

# ==========================================
# 2. 评估逻辑 (TS/HSS/F2)
# ==========================================

def calculate_scores(y_true, y_pred_prob, threshold=0.5):
    # 预测概率 > 阈值 判定为 1 (但这里我们只关心 Class 0 是否被预测出来)
    # 注意：输入 y_pred_prob 是 Class 0 (Fog) 的概率
    
    # 预测结果：如果 Prob(Fog) > T，则 Pred = Fog(0)，否则非Fog
    # 为了方便计算混淆矩阵，我们将 Fog 设为 Positive(1)，其他设为 Negative(0)
    y_pred_binary = (y_pred_prob > threshold).astype(int)
    
    # 真实标签：Class 0 是 Fog，转为 Binary 1
    y_true_binary = (y_true == 0).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    # TS / CSI
    ts = tp / (tp + fp + fn + 1e-6)
    
    # HSS
    num = 2 * (tp * tn - fp * fn)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = num / (den + 1e-6)
    
    # Precision / Recall
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    # F2 Score (Recall focused)
    f2 = 5 * (precision * recall) / (4 * precision + recall + 1e-6)
    
    return {'ts': ts, 'hss': hss, 'f2': f2, 'prec': precision, 'recall': recall}

def evaluate_model(model, X_val, y_val, model_type):
    print("Evaluating...", flush=True)
    
    # 为了防止验证集过大，LightGBM 和 XGBoost 可以分批预测，但通常 predict 内存占用较小
    if model_type == 'xgboost':
        # 这里的 X_val 是 mmap，XGBoost 会按需读取
        dval = xgb.DMatrix(X_val) 
        probs = model.predict(dval) # Returns (N, 3)
    elif model_type == 'lightgbm':
        probs = model.predict(X_val)
    elif model_type == 'catboost':
        probs = model.predict_proba(X_val)
        
    # 取 Class 0 (Fog) 的概率
    prob_fog = probs[:, 0]
    
    best_score = -1
    best_res = {}
    
    # 扫描最佳阈值
    for t in np.arange(0.1, 0.9, 0.05):
        res = calculate_scores(y_val, prob_fog, threshold=t)
        # 只有 Precision 满足一定门槛才考虑 F2，防止全报 Fog 骗 Recall
        if res['prec'] > 0.15 and res['f2'] > best_score:
            best_score = res['f2']
            best_res = res
            best_res['thresh'] = t
            
    return best_res

# ==========================================
# 3. 训练函数实现
# ==========================================

def train_xgboost(X_train, y_train, X_val, y_val, params, sample_weights=None, init_model=None):
    print("Converting data to QuantileDMatrix (Low Memory)...", flush=True)
    
    # QuantileDMatrix 必须配合 tree_method='hist'，非常适合大内存 mmap 数据
    # 传入 weight (如果有)
    dtrain = xgb.QuantileDMatrix(X_train, label=y_train, weight=sample_weights)
    
    # ref=dtrain 确保验证集的分桶策略与训练集一致
    dval = xgb.QuantileDMatrix(X_val, label=y_val, ref=dtrain)
    
    train_params = params.copy()
    train_params['tree_method'] = 'hist' # 强制 hist
    
    bst = xgb.train(
        train_params, 
        dtrain, 
        num_boost_round=args.n_estimators_s1 if init_model is None else args.n_estimators_s2,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        xgb_model=init_model,
        verbose_eval=100
    )
    return bst

def train_lightgbm(X_train, y_train, X_val, y_val, params, sample_weights=None, init_model=None, save_bin_path=None):
    print("Preparing LightGBM Training...", flush=True)
    
    lgb_train = None
    
    ds_params = {
        'max_bin': params.get('max_bin', 63), 
        'two_round': True, 
        'verbose': -1
    }
    
    # 【核心修改】：当 init_model 存在时，必须从内存构建 Dataset，
    # 因为 LightGBM 无法直接对 .bin 文件路径进行 init_score 预测。
    if save_bin_path and os.path.exists(save_bin_path) and init_model is None:
        print(f"Loading cached binary dataset from {save_bin_path}...", flush=True)
        lgb_train = lgb.Dataset(save_bin_path, params=ds_params)
    else:
        print("Constructing Dataset from mmap (Force memory load due to init_model or no cache)...", flush=True)
        # 构建时传入 ds_params
        lgb_train = lgb.Dataset(X_train, y_train, weight=sample_weights, params=ds_params, free_raw_data=True)
        
        # 即使这次是从内存构建，也可以顺便保存 .bin 供下次"只跑Stage2"时使用
        if save_bin_path and init_model is None:
            print(f"Saving binary dataset to {save_bin_path}...", flush=True)
            lgb_train.save_binary(save_bin_path)

    print("Constructing Validation Dataset...", flush=True)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=True)
    
    print("Starting Training...", flush=True)
    
    # 清理参数
    train_params = params.copy()
    keys_to_remove = ['max_bin', 'class_weight']
    for k in keys_to_remove:
        if k in train_params: del train_params[k]
    
    # 【修改点 2】：移除这里强行设置的 two_round，或者保持一致
    # 建议直接移除下面这行，因为上面 ds_params 已经生效了
    # train_params['two_round'] = True  <--- 删除或注释掉这一行
    
    bst = lgb.train(
        train_params,
        lgb_train,
        num_boost_round=args.n_estimators_s1 if init_model is None else args.n_estimators_s2,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)],
        init_model=init_model
    )
    return bst

def train_catboost(X_train, y_train, X_val, y_val, params, init_model=None):
    if cb is None: raise ImportError("Catboost not installed")
    
    print("Starting CatBoost Training...", flush=True)
    model = cb.CatBoostClassifier(
        iterations=args.n_estimators_s1 if init_model is None else args.n_estimators_s2,
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        loss_function='MultiClass',
        thread_count=args.num_threads,
        class_weights=params['class_weights'],
        verbose=100,
        task_type="CPU", # 如果有 GPU 可改为 GPU
        #used_ram_limit='30gb' # 限制内存
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        init_model=init_model
    )
    return model

# ==========================================
# 4. 主流程
# ==========================================

def main():
    print(f"=== Starting GBDT Training ({args.model_type}) ===")
    print(f"Mode: {'Stage 2 Only' if args.only_stage2 else 'Stage 1 -> Stage 2'}")
    print(f"Threads: {args.num_threads}, Window: {args.window_size}h, Max Depth: {args.max_depth}")
    
    model_s1 = None
    
    # 定义类别权重字典
    class_weight_dict = {0: args.pos_weight, 1: 2.0, 2: 1.0}

    # ==========================================
    # Stage 1: ERA5 Pre-training (可选)
    # ==========================================
    if not args.only_stage2:
        print("\n" + "="*40)
        print("--- STAGE 1: ERA5 Pre-training ---")
        print("="*40)
        
        X_s1, y_s1, X_val_s1, y_val_s1 = load_data(S1_DIR, stage='S1')
        
        # 准备样本权重
        weights_s1 = get_sample_weights_vectorized(y_s1, class_weight_dict)
        
        if args.model_type == 'xgboost':
            params = {
                'objective': 'multi:softprob', 'num_class': 3,
                'eta': args.learning_rate, 'max_depth': args.max_depth,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'nthread': args.num_threads, 'tree_method': 'hist',
            }
            model_s1 = train_xgboost(X_s1, y_s1, X_val_s1, y_val_s1, params, sample_weights=weights_s1)

        elif args.model_type == 'lightgbm':
            params = {
                'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
                'learning_rate': args.learning_rate,
                'num_leaves': 63, 'max_depth': args.max_depth,
                'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
                'n_jobs': 16, # 限制物理核数，避免超线程导致缓存抖动
                'verbose': -1,
                'max_bin': 63 # 降低内存占用
            }
            # S1 不缓存 bin 文件，因为只跑一次
            model_s1 = train_lightgbm(X_s1, y_s1, X_val_s1, y_val_s1, params, sample_weights=weights_s1)

        elif args.model_type == 'catboost':
            params = {
                'learning_rate': args.learning_rate,
                'depth': args.max_depth,
                'class_weights': [args.pos_weight, 2.0, 1.0]
            }
            model_s1 = train_catboost(X_s1, y_s1, X_val_s1, y_val_s1, params)

        # 评估 Stage 1
        res_s1 = evaluate_model(model_s1, X_val_s1, y_val_s1, args.model_type)
        print(f"Stage 1 Best Result: {json.dumps(res_s1, indent=2)}")
        
        # 释放 Stage 1 内存
        print("Cleaning up Stage 1 memory...", flush=True)
        del X_s1, y_s1, X_val_s1, y_val_s1, weights_s1
        gc.collect()

    else:
        print("\n--- Skipping STAGE 1 (User Request) ---")

    # ==========================================
    # Stage 2: Forecast Fine-tuning
    # ==========================================
    print("\n" + "="*40)
    print("--- STAGE 2: Forecast Data Training ---")
    print("="*40)
    
    X_s2, y_s2, X_val_s2, y_val_s2 = load_data(S2_DIR, stage='S2')
    
    # 动态调整学习率：如果有预训练模型，LR 降低 10 倍
    current_lr = args.learning_rate * 0.1 if model_s1 is not None else args.learning_rate
    if model_s1: print(f">> Fine-tuning mode enabled. LR reduced to {current_lr}")
    
    if args.model_type == 'xgboost':
        weights_s2 = get_sample_weights_vectorized(y_s2, class_weight_dict)
        params = {
            'objective': 'multi:softprob', 'num_class': 3,
            'eta': current_lr, 
            'max_depth': args.max_depth,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'nthread': args.num_threads, 'tree_method': 'hist'
        }
        model_s2 = train_xgboost(X_s2, y_s2, X_val_s2, y_val_s2, params, sample_weights=weights_s2, init_model=model_s1)
        model_s2.save_model(os.path.join(MODEL_DIR, f"xgb_stage2_{args.window_size}h.json"))
        
    elif args.model_type == 'lightgbm':
        params = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'learning_rate': current_lr,
            'num_leaves': 63, 'max_depth': args.max_depth,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'n_jobs': 16,
            'verbose': -1,
            'max_bin': 63,      # 【关键】显著降低内存
            'two_round': True   # 【关键】优化 mmap 读取
        }
        
        # 缓存路径
        bin_cache_path = os.path.join(MODEL_DIR, f"train_s2_{args.window_size}h.bin")
        
        # 仅当没有缓存时，才计算权重
        weights_s2 = None
        if not os.path.exists(bin_cache_path):
            weights_s2 = get_sample_weights_vectorized(y_s2, class_weight_dict)
        
        model_s2 = train_lightgbm(X_s2, y_s2, X_val_s2, y_val_s2, params, 
                                  sample_weights=weights_s2, 
                                  init_model=model_s1,
                                  save_bin_path=bin_cache_path)
        model_s2.save_model(os.path.join(MODEL_DIR, f"lgb_stage2_{args.window_size}h.txt"))
        
    elif args.model_type == 'catboost':
        params = {
            'learning_rate': current_lr,
            'depth': args.max_depth,
            'class_weights': [args.pos_weight, 2.0, 1.0]
        }
        model_s2 = train_catboost(X_s2, y_s2, X_val_s2, y_val_s2, params, init_model=model_s1)
        model_s2.save_model(os.path.join(MODEL_DIR, f"cb_stage2_{args.window_size}h.cbm"))

    # 评估 Stage 2
    res_s2 = evaluate_model(model_s2, X_val_s2, y_val_s2, args.model_type)
    print(f"Stage 2 Best Result: {json.dumps(res_s2, indent=2)}")
    print("Training Completed Successfully.")

if __name__ == "__main__":
    main()
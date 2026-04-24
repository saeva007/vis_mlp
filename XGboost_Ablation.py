import numpy as np
import os
import argparse
import joblib
import warnings
import time
import shutil
import sys
from sklearn.metrics import confusion_matrix

# 尝试导入库
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
# 1. 参数配置
# ==========================================

def get_args():
    parser = argparse.ArgumentParser(description="Visibility Forecasting - ML Baselines (XGB/LGBM)")
    
    parser.add_argument('--model', type=str, required=True, 
                        choices=['lightgbm', 'xgboost'],
                        help="选择模型: lightgbm 或 xgboost")
    parser.add_argument('--window_size', type=int, default=12, help="时间窗口大小 (默认12h)")
    parser.add_argument('--save_dir', type=str, default="./checkpoints_ml", help="模型保存路径")
    parser.add_argument('--base_path', type=str, default="/public/home/putianshu/vis_mlp/ml_dataset_pmst_finetune_v1",
                        help="数据基础路径前缀")
    
    args = parser.parse_args()
    return args

ARGS = get_args()

# 构造实验名称和路径
EXP_NAME = f"{ARGS.model}_w{ARGS.window_size}h"
SAVE_PATH = os.path.join(ARGS.save_dir, EXP_NAME)

# 自动推断数据目录
if ARGS.window_size == 12:
    DATA_DIR = "/public/home/putianshu/vis_mlp/ml_dataset_pmst_finetune_v1_12h"
else:
    DATA_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_finetune_v1_{ARGS.window_size}h"

# ==========================================
# 2. 数据搬运工具 (copy_to_local)
# ==========================================

def copy_to_local(src_path):
    """
    将文件从网络存储复制到计算节点的本地存储 (/dev/shm 或 /tmp)。
    针对单进程/单节点任务优化，移除了 DDP 同步逻辑。
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    filename = os.path.basename(src_path)
    # 优先使用内存盘 /dev/shm，速度最快
    target_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    local_path = os.path.join(target_dir, filename)
    
    need_copy = True
    if os.path.exists(local_path):
        # 简单检查文件大小是否一致
        if os.path.getsize(local_path) == os.path.getsize(src_path):
            need_copy = False
            print(f"[Data] Found cache in {local_path}, skipping copy.")
        else:
            print(f"[Data] Cache size mismatch, re-copying...")
    
    if need_copy:
        print(f"[Data] Copying {filename} to {target_dir}...", flush=True)
        try:
            t0 = time.time()
            tmp_path = local_path + ".tmp"
            shutil.copyfile(src_path, tmp_path)
            os.rename(tmp_path, local_path) # 原子操作，防止中断导致文件损坏
            print(f"[Data] Copy finished in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[Data] Copy FAILED: {e}. Fallback to NFS source.", flush=True)
            return src_path # 如果复制失败，回退到原始路径

    return local_path

# ==========================================
# 3. 数据加载与预处理
# ==========================================

def calc_f2_score(precision, recall):
    if precision + recall == 0: return 0.0
    beta2 = 4.0 
    f2 = (1 + beta2) * (precision * recall) / ((beta2 * precision) + recall + 1e-8)
    return f2

def transform_labels(y):
    y_out = np.zeros_like(y, dtype=int)
    y_out[y >= 500] = 1
    y_out[y >= 1000] = 2
    return y_out

def load_data():
    """加载数据，包含复制到本地的流程"""
    print(f"[Data] Source Directory: {DATA_DIR}")
    
    raw_paths = {
        'X_train': os.path.join(DATA_DIR, 'X_train.npy'),
        'y_train': os.path.join(DATA_DIR, 'y_train.npy'),
        'X_val':   os.path.join(DATA_DIR, 'X_val.npy'),
        'y_val':   os.path.join(DATA_DIR, 'y_val.npy')
    }

    # 执行复制并获取本地路径
    local_paths = {}
    for key, path in raw_paths.items():
        local_paths[key] = copy_to_local(path)

    print("[Data] Loading into memory...")
    t0 = time.time()
    
    # 使用 mmap_mode='r' 可以在内存不足时进行部分加载，但对于 LGBM/XGB 通常建议读入内存
    # 这里直接 load 读入内存，因为已经在 /dev/shm 了，速度极快
    X_train = np.load(local_paths['X_train'])
    y_train = np.load(local_paths['y_train'])
    X_val = np.load(local_paths['X_val'])
    y_val = np.load(local_paths['y_val'])
    
    print(f"[Data] Loaded in {time.time()-t0:.2f}s")
    print(f"       Train: {X_train.shape}, Val: {X_val.shape}")
    
    return X_train, y_train, X_val, y_val

# ==========================================
# 4. LightGBM 训练流程 (修正版)
# ==========================================

def run_lightgbm(X_train, y_train, X_val, y_val):
    if lgb is None:
        print("[ERROR] LightGBM not installed!")
        sys.exit(1)
    
    print("="*60)
    print(f"[LIGHTGBM] Experiment: {EXP_NAME}")
    print("="*60)
    
    y_train_cls = transform_labels(y_train)
    y_val_cls = transform_labels(y_val)
    
    # 移除最后一列（植被ID/静态ID）
    X_train_feat = X_train[:, :-1]
    X_val_feat = X_val[:, :-1]

    # -------------------------------------------------------
    # [修复] 1. 手动生成样本权重 (替代 class_weight 参数)
    # -------------------------------------------------------
    print("Generating sample weights for LightGBM...")
    # 默认权重为 1.0 (对应 class 2 / Clear)
    sample_weights = np.ones(len(y_train_cls), dtype=np.float32)
    # Class 0 (Fog): 权重 5.0
    sample_weights[y_train_cls == 0] = 5.0
    # Class 1 (Mist): 权重 2.0
    sample_weights[y_train_cls == 1] = 2.0
    
    # -------------------------------------------------------
    # [修复] 2. 将权重传入 Dataset
    # -------------------------------------------------------
    # free_raw_data=False 保证原始 numpy 数组还能被访问
    train_data = lgb.Dataset(
        X_train_feat, 
        label=y_train_cls, 
        weight=sample_weights,  # <--- 在这里传入权重
        free_raw_data=False
    )
    
    val_data = lgb.Dataset(
        X_val_feat, 
        label=y_val_cls, 
        reference=train_data, 
        free_raw_data=False
    )
    
    # -------------------------------------------------------
    # [修复] 3. 从 params 中移除 class_weight
    # -------------------------------------------------------
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
        'verbose': -1,
        'n_jobs': 32,  # CPU 核心数
        # 'class_weight': {0: 5.0, 1: 2.0, 2: 1.0}  <--- 删除这一行，原生API不支持
    }
    
    print("Training LightGBM...")
    t_start = time.time()
    
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    print(f"Training finished in {time.time() - t_start:.2f}s")
    
    print("Predicting...")
    y_pred_proba = gbm.predict(X_val_feat, num_iteration=gbm.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    return y_val_cls, y_pred, gbm

# ==========================================
# 5. XGBoost 训练流程
# ==========================================

def run_xgboost(X_train, y_train, X_val, y_val):
    if xgb is None:
        print("[ERROR] XGBoost not installed!")
        sys.exit(1)
    
    print("="*60)
    print(f"[XGBOOST] Experiment: {EXP_NAME}")
    print("="*60)
    
    y_train_cls = transform_labels(y_train)
    y_val_cls = transform_labels(y_val)
    
    X_train_feat = X_train[:, :-1]
    X_val_feat = X_val[:, :-1]
    
    # 样本权重
    print("Generating sample weights...")
    sample_weights = np.ones(len(y_train_cls))
    sample_weights[y_train_cls == 0] = 5.0
    sample_weights[y_train_cls == 1] = 2.0
    sample_weights[y_train_cls == 2] = 1.0
    
    print("Creating DMatrix...")
    # nthread 设置为 CPU 核心数，加快数据加载
    dtrain = xgb.DMatrix(X_train_feat, label=y_train_cls, weight=sample_weights, nthread=32)
    dval = xgb.DMatrix(X_val_feat, label=y_val_cls, nthread=32)
    
    # GPU 检测
    use_gpu = False
    try:
        # 简单通过 torch 检测 cuda，或者直接 try xgboost gpu 参数
        import torch
        if torch.cuda.is_available():
            use_gpu = True
    except:
        pass
        
    print(f"XGBoost Config: Use GPU = {use_gpu}")

    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'min_child_weight': 3,
        'nthread': 32, 
    }

    if use_gpu:
        params['tree_method'] = 'gpu_hist'
        params['gpu_id'] = 0
    else:
        params['tree_method'] = 'hist'

    print("Training XGBoost...")
    t_start = time.time()
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50
    )
    print(f"Training finished in {time.time() - t_start:.2f}s")
    
    print("Predicting...")
    y_pred_proba = bst.predict(dval)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    return y_val_cls, y_pred, bst

# ==========================================
# 6. 主程序
# ==========================================

if __name__ == "__main__":
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    # 1. 加载数据 (自动执行 copy_to_local)
    X_train, y_train, X_val, y_val = load_data()
    
    y_true_cls = None
    y_pred_cls = None
    model_obj = None

    # 2. 运行
    if ARGS.model == 'lightgbm':
        y_true_cls, y_pred_cls, model_obj = run_lightgbm(X_train, y_train, X_val, y_val)
        model_file = os.path.join(SAVE_PATH, f"{EXP_NAME}_model.txt")
        model_obj.save_model(model_file)
        print(f"Model saved to {model_file}")

    elif ARGS.model == 'xgboost':
        y_true_cls, y_pred_cls, model_obj = run_xgboost(X_train, y_train, X_val, y_val)
        model_file = os.path.join(SAVE_PATH, f"{EXP_NAME}_model.json")
        model_obj.save_model(model_file)
        print(f"Model saved to {model_file}")

    # 3. 评估
    if y_true_cls is not None:
        print("\n" + "="*60)
        print("Final Evaluation (Focus on Fog/Class 0):")
        
        y_true_fog = (y_true_cls == 0).astype(int)
        y_pred_fog = (y_pred_cls == 0).astype(int)
        
        cm = confusion_matrix(y_true_fog, y_pred_fog)
        tn, fp, fn, tp = cm.ravel()
        
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        ts = tp / (tp + fp + fn + 1e-8)
        f2 = calc_f2_score(precision, recall)
        
        print(f"  Fog F2 Score : {f2:.4f}")
        print(f"  Fog TS (CSI) : {ts:.4f}")
        print(f"  Fog Recall   : {recall:.4f}")
        print(f"  Fog Precision: {precision:.4f}")
        print("="*60 + "\n")
        
        metrics_file = os.path.join(SAVE_PATH, "metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write(f"Model: {ARGS.model}\nWindow: {ARGS.window_size}h\nF2: {f2:.4f}\nTS: {ts:.4f}\nRecall: {recall:.4f}\n")
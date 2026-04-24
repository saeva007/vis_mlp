import numpy as np
import os
import argparse
from sklearn.metrics import confusion_matrix
import joblib
import warnings

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("LightGBM not installed! Please install: pip install lightgbm")

warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="LightGBM Visibility Forecasting")
    parser.add_argument('--window_size', type=int, default=12, help="时间窗口大小")
    parser.add_argument('--base_path', type=str, default="/public/home/putianshu/vis_mlp")
    parser.add_argument('--save_dir', type=str, default="./checkpoints", help="模型保存路径")
    parser.add_argument('--num_boost_round', type=int, default=500, help="最大迭代次数")
    parser.add_argument('--early_stopping', type=int, default=50, help="早停轮数")
    parser.add_argument('--learning_rate', type=float, default=0.05, help="学习率")
    parser.add_argument('--num_leaves', type=int, default=31, help="叶子节点数")
    args = parser.parse_args()
    return args

def calc_f2_score(precision, recall):
    if precision + recall == 0: 
        return 0.0
    beta2 = 4.0 
    f2 = (1 + beta2) * (precision * recall) / ((beta2 * precision) + recall + 1e-8)
    return f2

def transform_labels(y):
    y_out = np.zeros_like(y, dtype=int)
    y_out[y >= 500] = 1
    y_out[y >= 1000] = 2
    return y_out

def main():
    args = get_args()
    
    # 实验名称
    EXP_NAME = f"lightgbm_w{args.window_size}h"
    SAVE_PATH = os.path.join(args.save_dir, EXP_NAME)
    
    # 数据路径
    if args.window_size == 12:
        DATA_DIR = "/public/home/putianshu/vis_mlp/ml_dataset_pmst_finetune_v1_12h"
    else:
        DATA_DIR = f"/public/home/putianshu/vis_mlp/ml_dataset_pmst_finetune_v1_{args.window_size}h"
    
    print("="*60)
    print(f"[LIGHTGBM] Experiment: {EXP_NAME}")
    print(f"  Window Size: {args.window_size}h")
    print(f"  Data Dir: {DATA_DIR}")
    print(f"  Save Path: {SAVE_PATH}")
    print("="*60)
    
    # 创建保存目录
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    # 加载数据
    print("Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shape: X={X_val.shape}, y={y_val.shape}")
    
    # 转换标签
    y_train_cls = transform_labels(y_train)
    y_val_cls = transform_labels(y_val)
    
    print(f"Label distribution (train): {np.bincount(y_train_cls)}")
    print(f"Label distribution (val): {np.bincount(y_val_cls)}")
    
    # 移除最后一列（植被ID）- 如果需要作为类别特征可以单独处理
    X_train_feat = X_train[:, :-1]
    X_val_feat = X_val[:, :-1]
    
    # 处理 NaN 和 Inf
    X_train_feat = np.nan_to_num(X_train_feat, nan=0.0, posinf=10.0, neginf=-10.0)
    X_val_feat = np.nan_to_num(X_val_feat, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # 创建LightGBM数据集
    print("Creating LightGBM datasets...")
    train_data = lgb.Dataset(X_train_feat, label=y_train_cls)
    val_data = lgb.Dataset(X_val_feat, label=y_val_cls, reference=train_data)
    
    # 参数设置
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1,
        'class_weight': {0: 5.0, 1: 2.0, 2: 1.0},
        'max_depth': -1,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    
    print("Training parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # 训练
    print("\nTraining LightGBM...")
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping),
            lgb.log_evaluation(period=10)
        ]
    )
    
    print(f"\nBest iteration: {gbm.best_iteration}")
    
    # 预测
    print("Predicting on validation set...")
    y_pred_proba = gbm.predict(X_val_feat, num_iteration=gbm.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 评估 - 针对 Fog 类别 (class 0)
    y_true_fog = (y_val_cls == 0).astype(int)
    y_pred_fog = (y_pred == 0).astype(int)
    
    cm = confusion_matrix(y_true_fog, y_pred_fog)
    print(f"\nConfusion Matrix (Fog vs Non-Fog):")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    ts = tp / (tp + fp + fn + 1e-8)
    f2 = calc_f2_score(precision, recall)
    
    print("="*60)
    print(f"LightGBM Results (Fog Detection):")
    print(f"  F2 Score: {f2:.4f}")
    print(f"  Threat Score (TS): {ts:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print("="*60)
    
    # 整体准确率
    overall_acc = np.mean(y_pred == y_val_cls)
    print(f"Overall Accuracy: {overall_acc:.4f}")
    
    # 保存模型
    model_path = os.path.join(SAVE_PATH, f"{EXP_NAME}_model.txt")
    gbm.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # 保存特征重要性
    importance = gbm.feature_importance(importance_type='gain')
    importance_path = os.path.join(SAVE_PATH, f"{EXP_NAME}_feature_importance.npy")
    np.save(importance_path, importance)
    print(f"Feature importance saved to: {importance_path}")
    
    # 保存结果摘要
    results = {
        'f2_score': f2,
        'threat_score': ts,
        'recall': recall,
        'precision': precision,
        'accuracy': overall_acc,
        'best_iteration': gbm.best_iteration,
        'confusion_matrix': cm.tolist()
    }
    
    import json
    results_path = os.path.join(SAVE_PATH, f"{EXP_NAME}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
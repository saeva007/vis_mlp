# -*- coding: utf-8 -*-
import os
import numpy as np

BASE = "/public/home/putianshu/vis_mlp"

NEW_DIR = os.path.join(BASE, "ml_dataset_s2_tianji_12h_pm10_monthtail")
OLD_DIR = os.path.join(BASE, "ml_dataset_cursor_2_12h")

WIN_SIZE = 12
SPLIT_DYN = WIN_SIZE * 25
BASE_DIM = SPLIT_DYN + 6   # 与 PMSTDataset 一致：dyn_flat + static_cont(5) + veg(1)
CLIP_MIN = -10.0
CLIP_MAX = 10.0

def audit_one(name, d, k=120000, seed=0):
    x_path = os.path.join(d, "X_train.npy")
    y_path = os.path.join(d, "y_train.npy")

    if not os.path.exists(x_path):
        print("\n[%s] missing X_train.npy: %s" % (name, x_path))
        return
    if not os.path.exists(y_path):
        print("\n[%s] missing y_train.npy: %s" % (name, y_path))
        return

    X = np.load(x_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')

    N, D = X.shape
    fe_dim = D - BASE_DIM

    rs = np.random.RandomState(seed)
    kk = min(k, N)
    idx = rs.randint(0, N, size=kk)

    # ---- pm10 column (assume last extra column) ----
    last_col = np.asarray(X[idx, D - 1], dtype=np.float32)
    nan_frac = float(np.isnan(last_col).mean())

    last0 = np.nan_to_num(last_col, nan=0.0)
    last_clip = np.clip(last0, CLIP_MIN, CLIP_MAX)

    sat_pos = float((last_clip >= (CLIP_MAX - 1e-4)).mean())
    sat_neg = float((last_clip <= (CLIP_MIN + 1e-4)).mean())

    # ---- basic stats on raw (finite only) ----
    finite = np.isfinite(last_col)
    if finite.any():
        raw_finite = last_col[finite]
        raw_min = float(raw_finite.min())
        raw_max = float(raw_finite.max())
        raw_mean = float(raw_finite.mean())
        raw_std = float(raw_finite.std())
    else:
        raw_min = raw_max = raw_mean = raw_std = float("nan")

    # ---- y distribution on the same sampled indices (fast approximate) ----
    y_s = np.asarray(y[idx], dtype=np.float32)
    y_max = float(np.max(y_s))
    y_scaled = y_s
    # mirrors train rule
    if y_max < 100.0:
        y_scaled = y_scaled * 1000.0

    y_cls = np.zeros(len(y_scaled), dtype=np.int64)
    y_cls[y_scaled >= 500.0] = 1
    y_cls[y_scaled >= 1000.0] = 2

    cnt = np.bincount(y_cls, minlength=3)
    fog_frac = float(cnt[0]) / float(kk)
    mist_frac = float(cnt[1]) / float(kk)
    clear_frac = float(cnt[2]) / float(kk)

    # boundary band around 500 (approx)
    boundary_band = float(((y_scaled >= 480.0) & (y_scaled < 520.0)).mean())

    print("\n== %s ==" % name)
    print("X_train shape=%s  fe_dim(D-BASE_DIM)=%d  (BASE_DIM=%d)" % (str(X.shape), fe_dim, BASE_DIM))
    print("last_col(pm10?, inferred) sample: NaN_frac=%.6f" % nan_frac)
    print("raw finite stats: min=%.3f max=%.3f mean=%.3f std=%.3f" % (raw_min, raw_max, raw_mean, raw_std))
    print("after loader clip(-10,10): sat>=10=%.4f sat<=-10=%.4f" % (sat_pos, sat_neg))
    print("y_sample class fractions (approx): fog=%.4f mist=%.4f clear=%.4f" % (fog_frac, mist_frac, clear_frac))
    print("boundary(480-520)m fraction (approx)=%.6f" % boundary_band)

    # sanity: if fe_dim==37, your pm10 is “supposed” to be last column
    print("Note: if fe_dim==37 then last extra col is expected to be pm10; if not, column interpretation may differ.")

if __name__ == "__main__":
    audit_one("NEW_pm10", NEW_DIR, k=120000, seed=0)
    audit_one("OLD_no_pm10", OLD_DIR, k=120000, seed=0)
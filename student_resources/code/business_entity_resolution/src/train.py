"""
Amazon ML Challenge 2026 - Business Entity Resolution
train.py

Binary pair classifier: given a (S1, S23) candidate pair and its 24-dimensional
feature vector, predict match probability.

Label convention
----------------
  1  =  true match  (pair appears in train_ground_truth.tsv)
  0  =  hard negative (pair is in the blocked set but NOT in ground truth)

Entity-level train / validation split
--------------------------------------
Split is performed on *S1 entity IDs* so that no S1 entity appears in both
train and validation.  This prevents entity leakage: the model cannot
memorise a business and exploit that memorisation at validation time.

Class imbalance
---------------
EDA: ~0.14 % of blocked pairs are positives (ratio ≈ 1:700).
We handle this with:
  1. scale_pos_weight in LightGBM  (≈ ratio of neg:pos in *training* fold)
  2. No under/over-sampling — all hard negatives are used; hard negatives are
     informative (they passed blocking for a reason).

Outputs saved to models/
-------------------------
  lgbm_model.txt          LightGBM booster (text format, portable)
  feature_meta.json       Feature names, sentinel value, training config
  train_report.json       Full numeric training report
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import ALL_FEATURES, N_FEATURES, MISSING_VAL

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_SRC    = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.normpath(os.path.join(_SRC, '..', '..', '..'))
_OUTPUT = os.path.join(_ROOT, 'output')
_MODELS = os.path.join(_SRC, '..', 'models')
_MODELS = os.path.normpath(_MODELS)

SEP = '=' * 66


# ─────────────────────────────────────────────────────────────────────────────
# Entity-level train / validation split
# ─────────────────────────────────────────────────────────────────────────────

def entity_split(
    meta: pd.DataFrame,
    val_frac: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split pair indices into train / validation sets by S1 entity ID.

    All pairs belonging to a given S1 entity go entirely into train OR
    entirely into validation — never split across both.  This prevents
    entity leakage.

    Returns train_idx, val_idx  (integer index arrays into meta).
    """
    rng = np.random.default_rng(seed)
    s1_ids = meta['s1_id'].unique()
    rng.shuffle(s1_ids)

    n_val = max(1, int(len(s1_ids) * val_frac))
    val_s1  = set(s1_ids[:n_val])
    train_s1 = set(s1_ids[n_val:])

    train_idx = np.where(meta['s1_id'].isin(train_s1))[0]
    val_idx   = np.where(meta['s1_id'].isin(val_s1))[0]
    return train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM configuration
# ─────────────────────────────────────────────────────────────────────────────

def build_lgbm_params(
    scale_pos_weight: float,
    seed: int = 42,
) -> Dict:
    """
    Return LightGBM parameters for binary pair classification.

    Key choices:
    - binary objective; primary metric is auc (stable for imbalanced data)
    - scale_pos_weight corrects for class imbalance
    - moderate tree depth (max_depth=8) avoids overfitting on 24 features
    - feature_fraction + bagging add regularisation
    - min_child_samples=20 allows small positive-class nodes to split
    - Missing values handled natively via NaN (sentinel converted before fit)
    """
    return {
        # Task
        'objective':          'binary',
        'metric':             ['binary_logloss', 'auc'],
        'verbosity':          -1,

        # Class imbalance
        'scale_pos_weight':   scale_pos_weight,

        # Tree structure
        'num_leaves':         63,
        'max_depth':          8,
        'min_child_samples':  20,
        'min_child_weight':   1e-3,

        # Regularisation
        'learning_rate':      0.05,
        'feature_fraction':   0.80,
        'bagging_fraction':   0.80,
        'bagging_freq':       5,
        'lambda_l1':          0.1,
        'lambda_l2':          0.1,

        # Reproducibility
        'seed':               seed,
        'data_random_seed':   seed,
        'feature_fraction_seed': seed,
        'bagging_seed':       seed,
        'drop_seed':          seed,

        # Missing-value handling
        'use_missing':        True,
        'zero_as_missing':    False,

        # Devices
        'num_threads':        0,   # use all available cores
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    X:           np.ndarray,
    meta:        pd.DataFrame,
    val_frac:    float = 0.20,
    num_rounds:  int   = 1500,
    early_stop:  int   = 80,
    seed:        int   = 42,
    verbose_eval: int  = 100,
) -> Tuple[lgb.Booster, Dict]:
    """
    Train a LightGBM binary classifier on the pair feature matrix.

    Parameters
    ----------
    X            : (n_pairs, 24) float32 feature matrix
    meta         : DataFrame with columns [s1_id, s23_id, label]
    val_frac     : fraction of S1 entities held out for validation
    num_rounds   : maximum boosting rounds
    early_stop   : stop if val AUC doesn't improve for this many rounds
    seed         : random seed
    verbose_eval : print eval metrics every N rounds

    Returns
    -------
    booster      : trained LightGBM Booster
    report       : dict of training statistics
    """
    assert 'label' in meta.columns, "meta must contain a 'label' column"
    assert len(X) == len(meta), "X and meta row counts must match"

    y = meta['label'].values.astype(np.int32)

    # ── Split ──────────────────────────────────────────────────────────────
    train_idx, val_idx = entity_split(meta, val_frac=val_frac, seed=seed)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx],   y[val_idx]

    n_pos_tr = int(y_tr.sum())
    n_neg_tr = int((y_tr == 0).sum())
    n_pos_va = int(y_va.sum())
    n_neg_va = int((y_va == 0).sum())

    # Guard against degenerate splits
    if n_pos_tr == 0:
        raise ValueError(
            "No positive pairs in training fold. "
            "Increase sample size or reduce val_frac.")

    spw = n_neg_tr / max(n_pos_tr, 1)
    params = build_lgbm_params(scale_pos_weight=spw, seed=seed)

    print(f"\n  Train fold : {len(train_idx):>10,} pairs  "
          f"pos={n_pos_tr:,}  neg={n_neg_tr:,}  "
          f"ratio={spw:.1f}:1", flush=True)
    print(f"  Val fold   : {len(val_idx):>10,} pairs  "
          f"pos={n_pos_va:,}  neg={n_neg_va:,}", flush=True)

    # ── Sentinel → NaN so LightGBM treats address-missing as missing ───────
    # LightGBM handles NaN natively; our sentinel -1.0 is converted here
    # so the model learns the missingness pattern rather than treating -1
    # as a real feature value.
    X_tr_lgb = X_tr.copy().astype(np.float32)
    X_va_lgb = X_va.copy().astype(np.float32)
    X_tr_lgb[X_tr_lgb == MISSING_VAL] = np.nan
    X_va_lgb[X_va_lgb == MISSING_VAL] = np.nan

    # ── LightGBM datasets ──────────────────────────────────────────────────
    dtrain = lgb.Dataset(
        X_tr_lgb, label=y_tr,
        feature_name=ALL_FEATURES,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_va_lgb, label=y_va,
        feature_name=ALL_FEATURES,
        reference=dtrain,
        free_raw_data=False,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    # Early stopping monitors val auc (second metric, index 1).
    # With scale_pos_weight >> 1, auc converges quickly — we allow up to
    # early_stop rounds without AUC improvement before stopping.
    callbacks = [
        lgb.early_stopping(early_stop, verbose=False),
        lgb.log_evaluation(verbose_eval),
    ]

    print(f"\n  Training LightGBM "
          f"(max_rounds={num_rounds}  early_stop={early_stop})...",
          flush=True)
    t0 = time.time()

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=callbacks,
    )

    train_time = time.time() - t0
    best_round = booster.best_iteration

    # ── Validation metrics ─────────────────────────────────────────────────
    val_probs = booster.predict(X_va_lgb, num_iteration=best_round)
    val_preds = (val_probs >= 0.5).astype(int)

    # Manual precision / recall / F1 at 0.5 threshold
    tp = int(((val_preds == 1) & (y_va == 1)).sum())
    fp = int(((val_preds == 1) & (y_va == 0)).sum())
    fn = int(((val_preds == 0) & (y_va == 1)).sum())
    tn = int(((val_preds == 0) & (y_va == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    # AUC (manual, sorted scores)
    from sklearn.metrics import roc_auc_score
    try:
        auc = float(roc_auc_score(y_va, val_probs))
    except Exception:
        auc = float('nan')

    # Best threshold by F1
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1_t, best_thr = 0.0, 0.5
    for thr in thresholds:
        p_ = (val_probs >= thr).astype(int)
        tp_ = int(((p_ == 1) & (y_va == 1)).sum())
        fp_ = int(((p_ == 1) & (y_va == 0)).sum())
        fn_ = int(((p_ == 0) & (y_va == 1)).sum())
        pr  = tp_ / max(tp_ + fp_, 1)
        rc  = tp_ / max(tp_ + fn_, 1)
        f1_ = 2 * pr * rc / max(pr + rc, 1e-9)
        if f1_ > best_f1_t:
            best_f1_t, best_thr = f1_, thr

    # Feature importances
    fi_gain  = booster.feature_importance(importance_type='gain')
    fi_split = booster.feature_importance(importance_type='split')
    fi_names = booster.feature_name()

    report = {
        # Data
        'n_pairs_total':   int(len(X)),
        'n_pos_total':     int(y.sum()),
        'n_neg_total':     int((y == 0).sum()),
        'class_ratio':     float(n_neg_tr / max(n_pos_tr, 1)),
        'n_features':      N_FEATURES,
        'feature_names':   ALL_FEATURES,

        # Split
        'val_frac':        val_frac,
        'n_train_pairs':   int(len(train_idx)),
        'n_val_pairs':     int(len(val_idx)),
        'n_train_pos':     n_pos_tr,
        'n_train_neg':     n_neg_tr,
        'n_val_pos':       n_pos_va,
        'n_val_neg':       n_neg_va,

        # Model config
        'model_params':    {k: v for k, v in params.items()
                            if k not in ('metric',)},
        'num_rounds_max':  num_rounds,
        'best_round':      best_round,
        'early_stop':      early_stop,
        'train_time_s':    round(train_time, 1),

        # Val metrics
        'val_auc':         round(auc, 6),
        'val_f1_at_05':    round(f1, 6),
        'val_precision_at_05': round(precision, 6),
        'val_recall_at_05':    round(recall, 6),
        'val_tp': tp, 'val_fp': fp, 'val_fn': fn, 'val_tn': tn,
        'best_threshold_by_f1': round(best_thr, 4),
        'best_f1_by_threshold': round(best_f1_t, 6),

        # Feature importance (top 10 by gain)
        'feature_importance_gain': {
            n: int(g) for n, g in zip(fi_names, fi_gain)
        },
    }

    return booster, report


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_model(
    booster: lgb.Booster,
    report:  Dict,
    models_dir: str = _MODELS,
) -> None:
    """Save booster, feature metadata, and training report to models_dir."""
    os.makedirs(models_dir, exist_ok=True)

    model_path  = os.path.join(models_dir, 'lgbm_model.txt')
    meta_path   = os.path.join(models_dir, 'feature_meta.json')
    report_path = os.path.join(models_dir, 'train_report.json')

    booster.save_model(model_path)

    feat_meta = {
        'feature_names':     ALL_FEATURES,
        'n_features':        N_FEATURES,
        'missing_sentinel':  MISSING_VAL,
        'sentinel_to_nan':   True,   # convert MISSING_VAL -> NaN before predict
        'best_round':        report['best_round'],
        'best_threshold':    report['best_threshold_by_f1'],
    }
    with open(meta_path, 'w') as f:
        json.dump(feat_meta, f, indent=2)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  Saved: {model_path}")
    print(f"  Saved: {meta_path}")
    print(f"  Saved: {report_path}")


def load_model(models_dir: str = _MODELS) -> Tuple[lgb.Booster, Dict]:
    """Load the booster and feature metadata from models_dir."""
    model_path = os.path.join(models_dir, 'lgbm_model.txt')
    meta_path  = os.path.join(models_dir, 'feature_meta.json')
    booster = lgb.Booster(model_file=model_path)
    with open(meta_path) as f:
        feat_meta = json.load(f)
    return booster, feat_meta


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    overall_t0 = time.time()

    print(SEP)
    print("  AMAZON ML CHALLENGE 2026 -- PAIR CLASSIFIER TRAINING")
    print(SEP)

    # ── Load feature matrix + meta ─────────────────────────────────────────
    print("\n[1] Loading features...", flush=True)
    X_path    = os.path.join(_OUTPUT, 'X_train.npy')
    meta_path = os.path.join(_OUTPUT, 'meta_train.csv')

    if not os.path.exists(X_path):
        print(f"  ERROR: {X_path} not found. Run run_features.py first.")
        return
    if not os.path.exists(meta_path):
        print(f"  ERROR: {meta_path} not found. Run regen_meta.py first.")
        return

    X    = np.load(X_path)
    meta = pd.read_csv(meta_path)

    assert len(X) == len(meta), (
        f"Row count mismatch: X={len(X):,}  meta={len(meta):,}. "
        "Re-run regen_meta.py to regenerate meta_train.csv.")

    n_pairs = len(X)
    n_pos   = int(meta['label'].sum())
    n_neg   = n_pairs - n_pos
    ratio   = n_neg / max(n_pos, 1)

    print(f"  Feature matrix : {X.shape}  dtype={X.dtype}")
    print(f"  Positive pairs : {n_pos:>10,}  ({100*n_pos/n_pairs:.3f}%)")
    print(f"  Negative pairs : {n_neg:>10,}  ({100*n_neg/n_pairs:.3f}%)")
    print(f"  Class ratio    : {ratio:.1f} : 1  (neg:pos)")
    print(f"  Features       : {N_FEATURES}")
    print(f"  Feature names  : {ALL_FEATURES}")

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n[2] Training...", flush=True)
    booster, report = train(
        X, meta,
        val_frac    = 0.20,
        num_rounds  = 1500,
        early_stop  = 80,
        seed        = 42,
        verbose_eval= 100,
    )

    # ── Save ───────────────────────────────────────────────────────────────
    print("\n[3] Saving model...", flush=True)
    save_model(booster, report, _MODELS)

    # ── Report ─────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TRAINING REPORT")
    print(SEP)
    print(f"  Total pairs              : {report['n_pairs_total']:>10,}")
    print(f"  Positive pairs           : {report['n_pos_total']:>10,}")
    print(f"  Negative pairs           : {report['n_neg_total']:>10,}")
    print(f"  Class ratio (neg:pos)    : {report['class_ratio']:>10.1f}")
    print(f"  Feature count            : {report['n_features']:>10,}")
    print(f"  Training pairs           : {report['n_train_pairs']:>10,}")
    print(f"    Train pos / neg        :  {report['n_train_pos']:,} / {report['n_train_neg']:,}")
    print(f"  Validation pairs         : {report['n_val_pairs']:>10,}")
    print(f"    Val pos / neg          :  {report['n_val_pos']:,} / {report['n_val_neg']:,}")
    print(f"  scale_pos_weight         : {report['class_ratio']:>10.1f}")
    print(f"  Best boosting round      : {report['best_round']:>10,}")
    print(f"  Train time               : {report['train_time_s']:>10.1f}s")
    print(f"\n  --- Validation metrics (threshold = 0.50) ---")
    print(f"  AUC                      : {report['val_auc']:>10.4f}")
    print(f"  F1                       : {report['val_f1_at_05']:>10.4f}")
    print(f"  Precision                : {report['val_precision_at_05']:>10.4f}")
    print(f"  Recall                   : {report['val_recall_at_05']:>10.4f}")
    print(f"  TP / FP / FN / TN        :  "
          f"{report['val_tp']} / {report['val_fp']} / "
          f"{report['val_fn']} / {report['val_tn']}")
    print(f"\n  --- Best threshold by F1 ---")
    print(f"  Best threshold           : {report['best_threshold_by_f1']:>10.4f}")
    print(f"  Best F1                  : {report['best_f1_by_threshold']:>10.4f}")

    print(f"\n  --- Feature importance (gain, top 10) ---")
    fi = report['feature_importance_gain']
    top10 = sorted(fi.items(), key=lambda x: -x[1])[:10]
    for rank, (fname, gain) in enumerate(top10, 1):
        print(f"  {rank:>2}. {fname:<26}  gain={gain:>10,}")

    print(f"\n  Total wall time          : {time.time()-overall_t0:.1f}s")
    print(SEP)


if __name__ == '__main__':
    main()

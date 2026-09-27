"""
Amazon ML Challenge 2026 - Business Entity Resolution
evaluate.py

Official macro-averaged F0.5 evaluation.

F0.5 weights precision twice as heavily as recall, reflecting that
false positives (spurious matches) are penalised more than false negatives.

Per-entity scoring
------------------
For each Source 1 entity:
  - precision = |pred ∩ true| / |pred|    (0 if pred is empty)
  - recall    = |pred ∩ true| / |true|    (0 if true is non-empty and pred empty)
  - F0.5      = 1.25 * P * R / (0.25 * P + R)
  - Special case: both pred AND true are empty  -> F0.5 = 1.0

Macro average: mean of all per-entity F0.5 scores.

Threshold sweep
---------------
Sweeps thresholds over [0.01, 0.99] (99 steps) on the validation fold,
computes macro-F0.5 at each threshold, selects the best, and saves it to
models/feature_meta.json under key 'best_threshold'.

No test labels are used at any point.

Outputs
-------
models/eval_report.json   full numeric evaluation report
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import entity_split          # reuse exact same split logic
from features import MISSING_VAL, ALL_FEATURES
from data_ingestion import load_source, load_ground_truth, parse_ground_truth

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_SRC    = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.normpath(os.path.join(_SRC, '..', '..', '..'))
_OUTPUT = os.path.join(_ROOT, 'output')
_MODELS = os.path.normpath(os.path.join(_SRC, '..', 'models'))

SEP = '=' * 66


# ─────────────────────────────────────────────────────────────────────────────
# Core F0.5 calculation  (pure, deterministic, no external data)
# ─────────────────────────────────────────────────────────────────────────────

def f05(precision: float, recall: float) -> float:
    """
    F0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R)
         = 1.25 * P * R / (0.25 * P + R)

    Returns 0.0 when denominator is 0 (P=R=0 except the empty-set corner case
    which is handled before this function is called).
    """
    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return 1.25 * precision * recall / denom


def score_one_entity(
    pred: Set[str],
    true: Set[str],
) -> Tuple[float, float, float]:
    """
    Compute (precision, recall, F0.5) for a single Source 1 entity.

    Corner cases
    ------------
    both empty                 : (1.0, 1.0, 1.0)
    pred empty, true non-empty : (0.0, 0.0, 0.0)   recall = 0
    pred non-empty, true empty : (0.0, 1.0, 0.0)   precision = 0, recall = 1
    """
    if not pred and not true:
        return 1.0, 1.0, 1.0

    if not pred:
        # predicted nothing, true has matches → recall = 0
        return 0.0, 0.0, 0.0

    tp = len(pred & true)

    if not true:
        # predicted something, true is empty → precision = 0
        p = 0.0
        r = 1.0          # convention: can't miss something that doesn't exist
        return p, r, f05(p, r)

    p = tp / len(pred)
    r = tp / len(true)
    return p, r, f05(p, r)


def macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Compute macro-averaged F0.5 over all S1 entities in ground_truth.

    Parameters
    ----------
    predictions  : {s1_id -> set(predicted_s23_ids)}
                   Missing S1 IDs are treated as empty predictions.
    ground_truth : {s1_id -> set(true_s23_ids)}
                   Must contain ALL S1 entities (including unmatched ones).

    Returns
    -------
    macro_f05    : float
    macro_prec   : float
    macro_recall : float
    per_entity   : {s1_id -> f05_score}
    """
    per_entity: Dict[str, float] = {}
    prec_sum = rec_sum = f_sum = 0.0
    n = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        p, r, f = score_one_entity(pred_set, true_set)
        per_entity[s1_id] = f
        prec_sum += p
        rec_sum  += r
        f_sum    += f
        n += 1

    if n == 0:
        return 0.0, 0.0, 0.0, {}

    return f_sum / n, prec_sum / n, rec_sum / n, per_entity


# ─────────────────────────────────────────────────────────────────────────────
# Threshold application
# ─────────────────────────────────────────────────────────────────────────────

def apply_threshold(
    probs:   np.ndarray,
    meta:    pd.DataFrame,
    threshold: float,
) -> Dict[str, Set[str]]:
    """
    Convert pair-level probabilities into per-S1 predicted match sets.

    Returns {s1_id -> set(s23_ids)} for all S1 entities in meta,
    including those with empty prediction sets (prob < threshold for all).
    """
    preds: Dict[str, Set[str]] = defaultdict(set)
    # pre-register all S1 IDs so entities with 0 predictions are included
    for s1_id in meta['s1_id'].unique():
        preds[s1_id]  # creates empty set

    mask = probs >= threshold
    for s1_id, s23_id in zip(
        meta.loc[mask, 's1_id'].values,
        meta.loc[mask, 's23_id'].values,
    ):
        preds[s1_id].add(s23_id)

    return dict(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def threshold_sweep(
    probs:        np.ndarray,
    meta:         pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    thresholds:   Optional[np.ndarray] = None,
) -> Tuple[float, List[dict]]:
    """
    Sweep thresholds, compute macro-F0.5 at each, return results and best threshold.

    Parameters
    ----------
    probs        : probability array aligned with meta rows
    meta         : DataFrame with s1_id / s23_id columns (val fold only)
    ground_truth : {s1_id -> set(true_s23_ids)}  for val S1 entities only
    thresholds   : array of thresholds to test (default: 0.01 – 0.99, 99 steps)

    Returns
    -------
    best_threshold : float
    sweep_results  : list of {threshold, macro_f05, macro_prec, macro_recall}
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 4)

    sweep_results = []
    best_f05 = -1.0
    best_thr = 0.5

    for thr in thresholds:
        preds = apply_threshold(probs, meta, float(thr))
        mf, mp, mr, _ = macro_f05(preds, ground_truth)
        sweep_results.append({
            'threshold':   float(thr),
            'macro_f05':   round(float(mf), 6),
            'macro_prec':  round(float(mp), 6),
            'macro_recall': round(float(mr), 6),
        })
        if mf > best_f05:
            best_f05 = mf
            best_thr = float(thr)

    return best_thr, sweep_results


# ─────────────────────────────────────────────────────────────────────────────
# Segment analysis
# ─────────────────────────────────────────────────────────────────────────────

def segment_scores(
    per_entity:  Dict[str, float],
    ground_truth: Dict[str, Set[str]],
    s1_country:  Dict[str, str],
) -> dict:
    """
    Break down per-entity F0.5 scores by:
      - match-count group: zero / singleton / multi-match
      - country of the S1 entity
    """
    zero_scores, single_scores, multi_scores = [], [], []
    country_scores: Dict[str, List[float]] = defaultdict(list)

    for s1_id, score in per_entity.items():
        n_true = len(ground_truth.get(s1_id, set()))
        if n_true == 0:
            zero_scores.append(score)
        elif n_true == 1:
            single_scores.append(score)
        else:
            multi_scores.append(score)

        country = s1_country.get(s1_id, 'Unknown')
        country_scores[country].append(score)

    def _mean(lst):
        return float(np.mean(lst)) if lst else float('nan')

    return {
        'zero_match':   {'n': len(zero_scores),   'macro_f05': _mean(zero_scores)},
        'singleton':    {'n': len(single_scores),  'macro_f05': _mean(single_scores)},
        'multi_match':  {'n': len(multi_scores),   'macro_f05': _mean(multi_scores)},
        'by_country': {
            c: {'n': len(scores), 'macro_f05': _mean(scores)}
            for c, scores in sorted(country_scores.items())
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    overall_t0 = time.time()

    print(SEP)
    print("  AMAZON ML CHALLENGE 2026 -- F0.5 EVALUATION")
    print(SEP)

    # ── 1. Load artefacts ─────────────────────────────────────────────────────
    print("\n[1] Loading artefacts...", flush=True)

    X_path    = os.path.join(_OUTPUT, 'X_train.npy')
    meta_path = os.path.join(_OUTPUT, 'meta_train.csv')
    model_path = os.path.join(_MODELS, 'lgbm_model.txt')
    feat_meta_path = os.path.join(_MODELS, 'feature_meta.json')

    for p in [X_path, meta_path, model_path, feat_meta_path]:
        if not os.path.exists(p):
            print(f"  ERROR: missing {p}")
            return

    X    = np.load(X_path)
    meta = pd.read_csv(meta_path)

    assert len(X) == len(meta), f"X/meta mismatch: {len(X)} vs {len(meta)}"

    with open(feat_meta_path) as f:
        feat_meta = json.load(f)

    booster = lgb.Booster(model_file=model_path)
    best_round = feat_meta['best_round']
    sentinel   = feat_meta['missing_sentinel']

    print(f"    X={X.shape}  meta={len(meta):,}  "
          f"pos={int(meta['label'].sum()):,}")

    # ── 2. Reproduce val split (same seed / logic as train.py) ───────────────
    print("\n[2] Reproducing validation split (seed=42, val_frac=0.20)...", flush=True)
    _, val_idx = entity_split(meta, val_frac=0.20, seed=42)

    meta_val = meta.iloc[val_idx].reset_index(drop=True)
    X_val    = X[val_idx].astype(np.float32).copy()
    X_val[X_val == sentinel] = np.nan

    print(f"    Val pairs : {len(meta_val):,}  "
          f"pos={int(meta_val['label'].sum()):,}  "
          f"neg={(meta_val['label']==0).sum():,}")

    # ── 3. Get model probabilities ────────────────────────────────────────────
    print("\n[3] Computing model probabilities on validation set...", flush=True)
    t0 = time.time()
    probs_val = booster.predict(X_val, num_iteration=best_round)
    print(f"    done in {time.time()-t0:.1f}s  "
          f"prob range=[{probs_val.min():.4f}, {probs_val.max():.4f}]")

    # ── 4. Build ground-truth for val S1 entities ─────────────────────────────
    print("\n[4] Building per-entity ground truth for val entities...", flush=True)

    val_s1_set = set(meta_val['s1_id'].unique())
    print(f"    Val S1 entities: {len(val_s1_set):,}", flush=True)

    # Ground truth from meta labels (pair-level)
    # This gives the true matched IDs that were captured by blocking.
    # We build it from meta_val so we only score on blocked candidates.
    gt_from_meta: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_set}
    for s1_id, s23_id, lbl in zip(
        meta_val['s1_id'], meta_val['s23_id'], meta_val['label']
    ):
        if lbl == 1:
            gt_from_meta[s1_id].add(s23_id)

    # Also load the complete ground truth so we can score against ALL true matches
    # (including those missed by blocking) for the full picture.
    # We load the training data in the same sample used by gen_dataset.py.
    print("    Loading full training ground truth...", flush=True)
    t0 = time.time()

    # Rebuild the same sample (seed=42 matches gen_dataset.py / run_features.py)
    S1_SAMPLE  = 100_000
    S23_SAMPLE = 200_000

    ts1_full = load_source('train', 1)
    ts2_full = load_source('train', 2)
    ts3_full = load_source('train', 3)
    gt_df    = load_ground_truth()
    gt_dict, _ = parse_ground_truth(gt_df)

    def _stratified_sample(df, n, seed=42):
        if len(df) <= n:
            return df.copy()
        total = len(df)
        parts = []
        for _, grp in df.groupby('country', sort=True):
            k = min(len(grp), max(1, round(n * len(grp) / total)))
            parts.append(grp.sample(k, random_state=seed))
        result = pd.concat(parts, ignore_index=True)
        if len(result) < n:
            extra = df[~df['entity_id'].isin(result['entity_id'])].sample(
                n - len(result), random_state=seed + 1)
            result = pd.concat([result, extra], ignore_index=True)
        return result.head(n).reset_index(drop=True)

    ts1 = _stratified_sample(ts1_full, S1_SAMPLE)
    ts2 = _stratified_sample(ts2_full, S23_SAMPLE)
    ts3 = _stratified_sample(ts3_full, S23_SAMPLE)

    s2_sample = set(ts2['entity_id'].astype(str))
    s3_sample = set(ts3['entity_id'].astype(str))
    s1_country: Dict[str, str] = dict(zip(
        ts1['entity_id'].astype(str), ts1['country'].astype(str)
    ))

    # Full ground truth restricted to sampled S2/S3 (what the model could predict)
    gt_full: Dict[str, Set[str]] = {}
    for s1_id in val_s1_set:
        true_ids = gt_dict.get(s1_id, [])
        gt_full[s1_id] = {m for m in true_ids if m in s2_sample or m in s3_sample}

    print(f"    done in {time.time()-t0:.1f}s  "
          f"total true pairs in val: {sum(len(v) for v in gt_full.values()):,}")

    # ── 5. Threshold sweep ────────────────────────────────────────────────────
    print("\n[5] Threshold sweep...", flush=True)
    t0 = time.time()
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 4)
    best_thr, sweep = threshold_sweep(probs_val, meta_val, gt_full, thresholds)
    print(f"    Sweep over {len(thresholds)} thresholds in {time.time()-t0:.1f}s")

    # Print sweep table (key thresholds + neighbourhood of best)
    print(f"\n  {'Threshold':>10}  {'macro_F0.5':>12}  {'Precision':>10}  {'Recall':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}")
    show_thrs = set([0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                     0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    best_row = next(r for r in sweep if r['threshold'] == best_thr)
    show_thrs.add(best_thr)
    for row in sweep:
        if row['threshold'] in show_thrs:
            marker = ' <-- BEST' if row['threshold'] == best_thr else ''
            print(f"  {row['threshold']:>10.2f}  "
                  f"{row['macro_f05']:>12.4f}  "
                  f"{row['macro_prec']:>10.4f}  "
                  f"{row['macro_recall']:>10.4f}"
                  f"{marker}")

    # ── 6. Score at best threshold ────────────────────────────────────────────
    print(f"\n[6] Scoring at best threshold ({best_thr})...", flush=True)
    best_preds = apply_threshold(probs_val, meta_val, best_thr)
    mf, mp, mr, per_entity = macro_f05(best_preds, gt_full)

    # Segment analysis
    segs = segment_scores(per_entity, gt_full, s1_country)

    # ── 7. Update feature_meta.json with best threshold ───────────────────────
    print("\n[7] Saving best threshold to feature_meta.json...", flush=True)
    feat_meta['best_threshold'] = best_thr
    feat_meta['val_macro_f05_at_best_threshold'] = round(float(mf), 6)
    with open(feat_meta_path, 'w') as f:
        json.dump(feat_meta, f, indent=2)
    print(f"    best_threshold updated to {best_thr}")

    # ── 8. Save eval report ───────────────────────────────────────────────────
    report = {
        'best_threshold':       best_thr,
        'val_macro_f05':        round(float(mf), 6),
        'val_macro_precision':  round(float(mp), 6),
        'val_macro_recall':     round(float(mr), 6),
        'n_val_s1_entities':    len(val_s1_set),
        'n_val_pairs':          len(meta_val),
        'n_val_positives':      int(meta_val['label'].sum()),
        'n_predicted_positive': int(sum(len(v) for v in best_preds.values())),
        'segments':             segs,
        'sweep': sweep,
    }
    report_path = os.path.join(_MODELS, 'eval_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"    Saved: {report_path}")

    # ── 9. Full report ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  EVALUATION REPORT")
    print(SEP)
    print(f"  Best threshold           : {best_thr:.4f}")
    print(f"  Macro F0.5               : {mf:.4f}")
    print(f"  Macro Precision          : {mp:.4f}")
    print(f"  Macro Recall             : {mr:.4f}")
    print(f"  Val S1 entities          : {len(val_s1_set):,}")
    print(f"  Predicted positive pairs : {sum(len(v) for v in best_preds.values()):,}")

    print(f"\n  --- F0.5 by match-count group ---")
    for grp in ('zero_match', 'singleton', 'multi_match'):
        g = segs[grp]
        print(f"  {grp:<16} : n={g['n']:>6,}  F0.5={g['macro_f05']:.4f}")

    print(f"\n  --- F0.5 by country ---")
    for country, g in segs['by_country'].items():
        print(f"  {country:<12} : n={g['n']:>6,}  F0.5={g['macro_f05']:.4f}")

    print(f"\n  Total wall time : {time.time()-overall_t0:.1f}s")
    print(SEP)


if __name__ == '__main__':
    main()

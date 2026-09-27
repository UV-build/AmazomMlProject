"""
Amazon ML Challenge 2026 - Business Entity Resolution
run_features.py

Runs feature generation on blocked training pairs and saves both
X_train.npy and meta_train.csv atomically in a single run.

REPRODUCIBILITY: launch with PYTHONHASHSEED=0 to ensure identical
blocking output across runs:
    PYTHONHASHSEED=0 python run_features.py   (Linux/Mac)
    $env:PYTHONHASHSEED='0'; python run_features.py  (PowerShell)

Usage:
    python run_features.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import load_source, load_ground_truth, parse_ground_truth
from blocking     import BlockingConfig, BlockingEngine
from features     import FeatureBuilder, ALL_FEATURES, N_FEATURES, feature_report

SEP = '=' * 66

# ── sample sizes matching run_blocking.py ──────────────────────────────────
S1_SAMPLE  = 100_000
S23_SAMPLE = 200_000

_OUTPUT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', '..', '..', 'output'))


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    total = len(df)
    parts = []
    for _country, grp in df.groupby('country', sort=True):
        k = min(len(grp), max(1, round(n * len(grp) / total)))
        parts.append(grp.sample(k, random_state=seed))
    result = pd.concat(parts, ignore_index=True)
    if len(result) < n:
        extra = df[~df['entity_id'].isin(result['entity_id'])].sample(
            n - len(result), random_state=seed + 1)
        result = pd.concat([result, extra], ignore_index=True)
    return result.head(n).reset_index(drop=True)


def main() -> None:
    overall_t0 = time.time()

    print(SEP)
    print("  AMAZON ML CHALLENGE 2026 -- FEATURE GENERATION")
    print(SEP)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("\n[1] Loading training data...", flush=True)
    t0 = time.time()
    ts1_full = load_source('train', 1)
    ts2_full = load_source('train', 2)
    ts3_full = load_source('train', 3)
    gt_df    = load_ground_truth()
    gt_dict, _ = parse_ground_truth(gt_df)
    print(f"    Loaded in {time.time()-t0:.1f}s  "
          f"S1={len(ts1_full):,}  S2={len(ts2_full):,}  S3={len(ts3_full):,}")

    # ── 2. Sample ─────────────────────────────────────────────────────────────
    print(f"\n[2] Stratified sample  "
          f"(S1={S1_SAMPLE:,}  S2/S3={S23_SAMPLE:,} each)...", flush=True)
    ts1 = stratified_sample(ts1_full, S1_SAMPLE)
    ts2 = stratified_sample(ts2_full, S23_SAMPLE)
    ts3 = stratified_sample(ts3_full, S23_SAMPLE)
    print(f"    S1={len(ts1):,}  S2={len(ts2):,}  S3={len(ts3):,}")

    s1_ids_set = set(ts1['entity_id'].astype(str))
    s2_ids_set = set(ts2['entity_id'].astype(str))
    s3_ids_set = set(ts3['entity_id'].astype(str))
    gt_sample: dict = {
        k: [m for m in v if m in s2_ids_set or m in s3_ids_set]
        for k, v in gt_dict.items()
        if k in s1_ids_set
    }
    gt_sample = {k: v for k, v in gt_sample.items() if v}
    gt_pairs = sum(len(v) for v in gt_sample.values())
    print(f"    GT pairs in sample: {gt_pairs:,}")

    # ── 3. Blocking ───────────────────────────────────────────────────────────
    print("\n[3] Running blocking...", flush=True)
    cfg = BlockingConfig(
        min_name_token_overlap=2,
        min_addr_token_overlap=2,
        min_trigram_overlap=4,
        max_postings_pct=0.005,
        per_key_cap=500,
        candidate_cap=200,
        cross_country_k1k2=True,
    )
    t0 = time.time()
    engine     = BlockingEngine(ts1, ts2, ts3, cfg)
    candidates = engine.generate_candidates(verbose=False)
    n_cand_pairs = sum(len(v) for v in candidates.values())
    print(f"    Candidates: {n_cand_pairs:,} pairs  ({time.time()-t0:.1f}s)")

    # ── 4. Feature builder ────────────────────────────────────────────────────
    print("\n[4] Building FeatureBuilder (IDF estimation)...", flush=True)
    fb = FeatureBuilder(ts1, ts2, ts3, verbose=True)

    # ── 5. Generate features ──────────────────────────────────────────────────
    print("\n[5] Generating features...", flush=True)
    t0 = time.time()
    X, meta = fb.build_pair_features(candidates, label_dict=gt_sample, verbose=True)
    gen_time = time.time() - t0

    # ── 6. Report ─────────────────────────────────────────────────────────────
    print("\n[6] Feature report...", flush=True)
    feature_report(X, meta)

    # Feature names
    print(f"\n  Feature names ({N_FEATURES} total):")
    for i, name in enumerate(ALL_FEATURES):
        print(f"    {i+1:>2}. {name}")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    print("\n[7] Saving outputs...", flush=True)
    os.makedirs(_OUTPUT, exist_ok=True)
    np.save(os.path.join(_OUTPUT, 'X_train.npy'), X)
    meta.to_csv(os.path.join(_OUTPUT, 'meta_train.csv'), index=False)
    print(f"    Saved X_train.npy  shape={X.shape}")
    print(f"    Saved meta_train.csv  rows={len(meta):,}")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    total_time = time.time() - overall_t0
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    print(f"  Candidate pairs          : {n_cand_pairs:,}")
    print(f"  Feature pairs generated  : {len(meta):,}")
    print(f"  Number of features       : {N_FEATURES}")
    print(f"  Feature generation time  : {gen_time:.1f}s")
    print(f"  Total wall time          : {total_time:.1f}s")
    if 'label' in meta.columns:
        pos = int(meta['label'].sum())
        print(f"  Positive pairs (matches) : {pos:,}  "
              f"({100*pos/max(len(meta),1):.3f}%)")
        print(f"  Negative pairs           : {len(meta)-pos:,}")
    print(SEP)


if __name__ == '__main__':
    main()

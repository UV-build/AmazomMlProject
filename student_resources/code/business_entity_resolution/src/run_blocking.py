"""
Amazon ML Challenge 2026 - Business Entity Resolution
run_blocking.py

Runs the blocking pipeline on a stratified sample of training data,
reports full diagnostics, per-key contribution, and missed-pair analysis.

Sample sizes are chosen so the total run completes in ~15 minutes:
  S1_SAMPLE  = 100,000  (~4.5 % of full S1)
  S23_SAMPLE = 200,000  (~1.9 % of full S23 per source)

All metrics are reported on the sample and extrapolated to full-data scale.

Usage:
    python run_blocking.py
"""

from __future__ import annotations

import os
import sys
import time
import random

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import load_source, load_ground_truth, parse_ground_truth
from blocking import (
    BlockingConfig,
    BlockingEngine,
    compute_diagnostics,
    print_diagnostics,
    _country_matches,
)

SEP = '=' * 66

# ---- sample sizes ----
S1_SAMPLE  = 100_000
S23_SAMPLE = 200_000


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Proportional stratified sample by country, reproducible."""
    if len(df) <= n:
        return df.copy()
    total = len(df)
    parts = []
    for _country, grp in df.groupby('country', sort=True):
        k = min(len(grp), max(1, round(n * len(grp) / total)))
        parts.append(grp.sample(k, random_state=seed))
    result = pd.concat(parts, ignore_index=True)
    # top up / trim to exactly n
    if len(result) < n:
        extra = df[~df['entity_id'].isin(result['entity_id'])].sample(
            n - len(result), random_state=seed + 1)
        result = pd.concat([result, extra], ignore_index=True)
    return result.head(n).reset_index(drop=True)


def main() -> None:
    overall_t0 = time.time()

    print(SEP)
    print("  AMAZON ML CHALLENGE 2026 -- BLOCKING PIPELINE")
    print(SEP)

    # ------------------------------------------------------------------ #
    # 1. Load full training data                                          #
    # ------------------------------------------------------------------ #
    print("\n[1] Loading training data...", flush=True)
    t0 = time.time()
    ts1_full = load_source('train', 1)
    ts2_full = load_source('train', 2)
    ts3_full = load_source('train', 3)
    gt_df    = load_ground_truth()
    gt_dict, _ = parse_ground_truth(gt_df)
    print(f"    Loaded in {time.time()-t0:.1f}s")
    print(f"    S1={len(ts1_full):,}  S2={len(ts2_full):,}  S3={len(ts3_full):,}")

    # ------------------------------------------------------------------ #
    # 2. Stratified sample                                                #
    # ------------------------------------------------------------------ #
    print(f"\n[2] Stratified sample  "
          f"(S1={S1_SAMPLE:,}  S2/S3={S23_SAMPLE:,} each)...", flush=True)
    ts1 = stratified_sample(ts1_full, S1_SAMPLE)
    ts2 = stratified_sample(ts2_full, S23_SAMPLE)
    ts3 = stratified_sample(ts3_full, S23_SAMPLE)
    print(f"    S1={len(ts1):,}  S2={len(ts2):,}  S3={len(ts3):,}")

    s1_ids_set = set(ts1['entity_id'].astype(str))
    s2_ids_set = set(ts2['entity_id'].astype(str))
    s3_ids_set = set(ts3['entity_id'].astype(str))

    gt_sample: dict = {}
    for s1id, mids in gt_dict.items():
        if s1id not in s1_ids_set:
            continue
        filtered = [m for m in mids if m in s2_ids_set or m in s3_ids_set]
        if filtered:
            gt_sample[s1id] = filtered

    gt_sample_pairs = sum(len(v) for v in gt_sample.values())
    print(f"    GT pairs in sample: {gt_sample_pairs:,}")

    # country distribution of sample
    for label, df in [('S1', ts1), ('S2', ts2), ('S3', ts3)]:
        dist = df['country'].value_counts().to_dict()
        dist_str = '  '.join(f"{c}:{n:,}" for c, n in
                             sorted(dist.items(), key=lambda x: -x[1]))
        print(f"    {label} countries: {dist_str}")

    # ------------------------------------------------------------------ #
    # 3. Build blocking engine                                            #
    # ------------------------------------------------------------------ #
    print("\n[3] Building blocking engine...", flush=True)
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
    engine = BlockingEngine(ts1, ts2, ts3, cfg)
    build_time = time.time() - t0
    print(f"    Engine ready in {build_time:.1f}s")

    # ------------------------------------------------------------------ #
    # 4. Generate candidates                                              #
    # ------------------------------------------------------------------ #
    print("\n[4] Generating candidates...", flush=True)
    t0 = time.time()
    candidates = engine.generate_candidates(verbose=True)
    gen_time = time.time() - t0
    print(f"    Done in {gen_time:.1f}s")

    # ------------------------------------------------------------------ #
    # 5. Diagnostics                                                      #
    # ------------------------------------------------------------------ #
    print("\n[5] Diagnostics...", flush=True)
    n_s23 = len(ts2) + len(ts3)
    diag  = compute_diagnostics(candidates, len(ts1), n_s23, gt_sample)
    print_diagnostics(diag, label=f"SAMPLE ({len(ts1):,} S1 / {n_s23:,} S23)")

    # Extrapolate to full dataset
    full_s1  = len(ts1_full)
    full_s23 = len(ts2_full) + len(ts3_full)
    s1_scale  = full_s1  / max(len(ts1),  1)
    s23_scale = full_s23 / max(n_s23, 1)
    est_pairs = diag.candidate_pairs * s1_scale * s23_scale
    est_rr    = 1.0 - est_pairs / (full_s1 * full_s23)
    print(f"\n  Full-data extrapolation  "
          f"(S1={full_s1:,}  S23={full_s23:,}):")
    print(f"    Est. candidate pairs  : {est_pairs:,.0f}")
    print(f"    Est. reduction ratio  : {est_rr:.4%}")

    # ------------------------------------------------------------------ #
    # 6. Per-key contribution on 5 k sub-sample                          #
    # ------------------------------------------------------------------ #
    print("\n[6] Per-key contribution (5,000-entity sub-sample)...", flush=True)
    random.seed(cfg.seed)
    probe_ids = random.sample(sorted(engine._s1.keys()),
                              min(5_000, len(engine._s1)))
    gt_set: set = {(s1id, mid)
                   for s1id, mids in gt_sample.items()
                   for mid in mids}

    KEY_LABELS   = ['K1 exact name ', 'K2 sorted toks',
                    'K3 phonetic   ', 'K4 name tok   ',
                    'K5 addr tok   ', 'K6 trigrams   ']
    KEY_METHODS  = [engine._k1, engine._k2, engine._k3,
                    engine._k4, engine._k5, engine._k6]

    stats = {lbl: [0, 0] for lbl in KEY_LABELS}   # [total_cands, gt_hits]
    for s1id in probe_ids:
        rec     = engine._s1[s1id]
        country = rec['country']
        for lbl, meth in zip(KEY_LABELS, KEY_METHODS):
            raw = meth(rec)
            if country:
                raw = [h for h in raw
                       if _country_matches(country, engine._s23[h]['country'])]
            stats[lbl][0] += len(raw)
            stats[lbl][1] += sum(1 for h in raw if (s1id, h) in gt_set)

    n_probe = len(probe_ids)
    print(f"\n  {'Key':<16}  {'Cands/S1':>10}  {'GT hits/S1':>12}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*12}")
    for lbl in KEY_LABELS:
        tc, gh = stats[lbl]
        print(f"  {lbl}  {tc/n_probe:>10.1f}  {gh/n_probe:>12.3f}")

    # ------------------------------------------------------------------ #
    # 7. Missed-pair analysis                                             #
    # ------------------------------------------------------------------ #
    print("\n[7] Missed ground-truth pairs...", flush=True)
    missed = [(s1id, mid)
              for s1id, mids in gt_sample.items()
              for mid in mids
              if mid not in candidates.get(s1id, set())]
    print(f"    Missed pairs: {len(missed):,} "
          f"/ {gt_sample_pairs:,} "
          f"({100*len(missed)/max(gt_sample_pairs,1):.2f}%)")

    if missed:
        s1_map  = ts1.set_index('entity_id').to_dict('index')
        s23_map = (pd.concat([ts2, ts3], ignore_index=True)
                   .set_index('entity_id').to_dict('index'))
        random.seed(cfg.seed)
        show = random.sample(missed, min(15, len(missed)))
        print(f"\n  Sample missed pairs (up to 15):\n")
        for s1id, mid in show:
            r1  = s1_map.get(s1id, {})
            r23 = s23_map.get(mid, {})
            print(f"  S1  [{r1.get('country','')}] "
                  f"{str(r1.get('business_name',''))[:55]}")
            print(f"  {mid[:2]}  [{r23.get('country','')}] "
                  f"{str(r23.get('business_name',''))[:55]}")
            print(f"  S1  addr: {str(r1.get('business_address',''))[:55]}")
            print(f"  S?  addr: {str(r23.get('business_address',''))[:55]}")
            print()

    # ------------------------------------------------------------------ #
    # 8. Summary                                                          #
    # ------------------------------------------------------------------ #
    total_time = time.time() - overall_t0
    print(SEP)
    print("  SUMMARY")
    print(SEP)
    print(f"  Sample size               : {len(ts1):,} S1 / {n_s23:,} S23")
    print(f"  Sample blocking recall    : {diag.recall:.4%}")
    print(f"  Sample reduction ratio    : {diag.reduction_ratio:.4%}")
    print(f"  Sample candidate pairs    : {diag.candidate_pairs:,}")
    print(f"  Sample missed GT pairs    : "
          f"{diag.gt_pairs_total - diag.gt_pairs_recalled:,}")
    print(f"  Est. full candidate pairs : {est_pairs:,.0f}")
    print(f"  Est. full reduction ratio : {est_rr:.4%}")
    print(f"  Build time                : {build_time:.1f}s")
    print(f"  Generation time           : {gen_time:.1f}s")
    print(f"  Total wall time           : {total_time:.1f}s")
    print(SEP)


if __name__ == '__main__':
    main()

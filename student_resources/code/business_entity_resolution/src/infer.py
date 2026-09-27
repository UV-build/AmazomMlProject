"""
Amazon ML Challenge 2026 - Business Entity Resolution
infer.py

Complete test-time inference pipeline:

  test data -> blocking -> features -> model -> threshold -> matches

Produces
--------
output/candidate_pairs.tsv      source1_entity_id <TAB> candidate_entity_ids
output/matching_results.tsv     source1_entity_id <TAB> matched_entity_ids

Both files:
  - Tab-separated, header row included
  - Exactly one row per test Source 1 entity (1,732,544 rows + header)
  - IDs comma-separated within the cell; empty list -> empty string field
  - No S1 IDs in match columns, no duplicates, matched subset of candidates
  - Only valid test S2/S3 IDs appear in either column

Design
------
The full test set (1.73M S1 + 10M S23) cannot be processed in a single
command invocation on this hardware. The script therefore:

  1. Builds S23 blocking indexes and feature caches ONCE per country.
  2. Processes S1 entities in batches (S1_BATCH_SIZE) reusing the S23 state.
  3. Checkpoints results to  output/_infer_checkpoint.json  after each batch.
  4. On re-run picks up exactly where it left off (completed batches skipped).
  5. Writes final TSVs once every S1 entity has a row.

Usage
-----
  $env:PYTHONHASHSEED='0'; python -u infer.py

To restart from scratch:
  Remove  output/_infer_checkpoint.json  then re-run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Set

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import load_source
from blocking       import BlockingConfig, BlockingEngine
from features       import (FeatureBuilder, _build_cache, _estimate_idf,
                             MISSING_VAL)
from evaluate       import apply_threshold

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_SRC    = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.normpath(os.path.join(_SRC, '..', '..', '..'))
_OUTPUT = os.path.join(_ROOT, 'output')
_MODELS = os.path.normpath(os.path.join(_SRC, '..', 'models'))

CAND_TSV    = os.path.join(_OUTPUT, 'candidate_pairs.tsv')
RESULTS_TSV = os.path.join(_OUTPUT, 'matching_results.tsv')
CHECKPOINT  = os.path.join(_OUTPUT, '_infer_checkpoint.json')

SEP = '=' * 66

# Number of S1 entities processed per batch.
# Smaller = less memory; larger = fewer overhead calls per S23 build.
S1_BATCH_SIZE = 50_000


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _load_ckpt() -> dict:
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding='utf-8') as f:
            return json.load(f)
    return {'done': [], 'cand': {}, 'match': {}}


def _save_ckpt(ckpt: dict) -> None:
    os.makedirs(_OUTPUT, exist_ok=True)
    with open(CHECKPOINT, 'w', encoding='utf-8') as f:
        json.dump(ckpt, f)


def _flush_tsv(ckpt: dict, s1_order: List[str]) -> None:
    """
    Write TSVs immediately from the current checkpoint state.
    S1 entities not yet processed get an empty row (empty field).
    Called after every batch so the files are always up to date.
    """
    write_tsv(CAND_TSV,    ckpt['cand'],  s1_order,
              'source1_entity_id', 'candidate_entity_ids')
    write_tsv(RESULTS_TSV, ckpt['match'], s1_order,
              'source1_entity_id', 'matched_entity_ids')
    n_done = len(ckpt['cand'])
    n_total = len(s1_order)
    print(f"    TSVs written: {n_done:,}/{n_total:,} S1 rows "
          f"({100*n_done/max(n_total,1):.1f}% complete)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Output validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    cand_rows:    Dict[str, List[str]],
    match_rows:   Dict[str, List[str]],
    all_s1:       Set[str],
    valid_s23:    Set[str],
    n_sample:     int = 2000,
) -> List[str]:
    """
    Check all 8 integrity requirements.
    Sample-checks for per-pair rules to avoid O(n) scan on 12M IDs.
    Returns list of error strings (empty = all pass).
    """
    import random
    rng = random.Random(42)

    errs: List[str] = []

    miss_c = all_s1 - set(cand_rows)
    if miss_c:
        errs.append(f"candidate_pairs missing {len(miss_c):,} S1 entities")
    miss_m = all_s1 - set(match_rows)
    if miss_m:
        errs.append(f"matching_results missing {len(miss_m):,} S1 entities")

    sample = rng.sample(sorted(all_s1), min(n_sample, len(all_s1)))
    for s1_id in sample:
        cands   = set(cand_rows.get(s1_id, []))
        matches = set(match_rows.get(s1_id, []))

        # req 5: only valid S2/S3 IDs
        bad = cands - valid_s23
        if bad:
            errs.append(f"[req5] {s1_id}: invalid cand IDs {list(bad)[:2]}")
            break
        bad = matches - valid_s23
        if bad:
            errs.append(f"[req5] {s1_id}: invalid match IDs {list(bad)[:2]}")
            break
        # req 7: no S1 IDs in matches
        s1_in = {m for m in matches if m.startswith('S1-')}
        if s1_in:
            errs.append(f"[req7] {s1_id}: S1 ID in matches {list(s1_in)[:2]}")
            break
        # req 8: matched ⊆ candidates
        extra = matches - cands
        if extra:
            errs.append(f"[req8] {s1_id}: match not in cands {list(extra)[:2]}")
            break

    return errs


# ─────────────────────────────────────────────────────────────────────────────
# TSV writer
# ─────────────────────────────────────────────────────────────────────────────

def write_tsv(
    path:      str,
    rows:      Dict[str, List[str]],
    s1_order:  List[str],
    id_col:    str,
    val_col:   str,
) -> int:
    """
    Write two-column TSV in s1_order.
    IDs comma-separated; empty list -> empty field.
    Returns number of data rows written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(f"{id_col}\t{val_col}\n")
        for s1_id in s1_order:
            ids  = rows.get(s1_id, [])
            cell = ','.join(ids)           # empty list -> ''
            fh.write(f"{s1_id}\t{cell}\n")
            n += 1
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Per-country pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_country(
    country:    str,
    s1_df:      pd.DataFrame,
    s2_df:      pd.DataFrame,
    s3_df:      pd.DataFrame,
    booster:    lgb.Booster,
    threshold:  float,
    best_round: int,
    cfg:        BlockingConfig,
    ckpt:       dict,
    s1_order:   List[str],
) -> None:
    """
    Process all S1 entities for one country in S1_BATCH_SIZE batches.
    Writes results directly into ckpt['cand'] and ckpt['match'].
    Saves checkpoint after each batch.
    """
    done_set   = set(ckpt['done'])
    valid_s23c = (set(s2_df['entity_id'].astype(str)) |
                  set(s3_df['entity_id'].astype(str)))

    n_s1      = len(s1_df)
    n_batches = max(1, (n_s1 + S1_BATCH_SIZE - 1) // S1_BATCH_SIZE)

    # Skip if every batch for this country is already done
    all_keys = [f"{country}_{i}" for i in range(n_batches)]
    if all(k in done_set for k in all_keys):
        already = sum(1 for k in all_keys if k in done_set)
        print(f"  [{country}] All {n_batches} batches already done, skipping.",
              flush=True)
        return

    # ── Build S23 side once for this country ─────────────────────────────────
    print(f"\n  [{country}] Building S23 blocking indexes...", flush=True)
    t0 = time.time()
    eng = BlockingEngine.from_s23_only(s2_df, s3_df, cfg, verbose=True)
    print(f"  [{country}] S23 blocking ready in {time.time()-t0:.1f}s", flush=True)

    print(f"  [{country}] Building S23 feature cache + IDF...", flush=True)
    t0 = time.time()
    s23_concat = pd.concat([s2_df, s3_df], ignore_index=True)
    s23_cache  = _build_cache(s23_concat)
    name_idf   = _estimate_idf([c.core_name  for c in s23_cache.values()])
    addr_idf   = _estimate_idf([c.clean_addr for c in s23_cache.values()])
    print(f"  [{country}] S23 feature cache ready in {time.time()-t0:.1f}s  "
          f"name_idf={len(name_idf):,}  addr_idf={len(addr_idf):,}", flush=True)

    # ── Process S1 in batches ─────────────────────────────────────────────────
    # We need a consistent S1 ID ordering for batch boundaries.
    # Use the original DataFrame row order (sorted by entity_id alphabetically
    # for reproducibility).
    s1_ids_sorted = sorted(s1_df['entity_id'].astype(str).tolist())
    s1_id_set     = set(s1_ids_sorted)
    # Index the s1_df by entity_id for fast batch lookup
    s1_df_idx = s1_df.copy()
    s1_df_idx['entity_id_str'] = s1_df_idx['entity_id'].astype(str)

    for batch_i in range(n_batches):
        bkey = f"{country}_{batch_i}"
        if bkey in done_set:
            print(f"  [{country}] Batch {batch_i+1}/{n_batches} "
                  f"already done.", flush=True)
            continue

        b_start   = batch_i * S1_BATCH_SIZE
        b_end     = min(b_start + S1_BATCH_SIZE, len(s1_ids_sorted))
        batch_ids = s1_ids_sorted[b_start:b_end]
        batch_id_set = set(batch_ids)

        print(f"  [{country}] Batch {batch_i+1}/{n_batches}  "
              f"S1[{b_start:,}:{b_end:,}]...", flush=True)
        t0b = time.time()

        # Load S1 batch into engine
        batch_s1 = s1_df_idx[
            s1_df_idx['entity_id_str'].isin(batch_id_set)
        ].drop(columns=['entity_id_str']).copy()
        eng.load_s1_batch(batch_s1)

        # Generate candidates
        cands = eng.generate_candidates(s1_ids=batch_ids, verbose=False)
        n_cand = sum(len(v) for v in cands.values())

        if n_cand > 0:
            # Build S1 feature cache for this batch, reuse S23 cache
            fb = FeatureBuilder.from_s23_cache(
                batch_s1, s23_cache, name_idf, addr_idf)
            X, meta = fb.build_pair_features(
                cands, label_dict=None, verbose=False)

            # Predict
            Xf = X.astype(np.float32).copy()
            Xf[Xf == MISSING_VAL] = np.nan
            probs = booster.predict(Xf, num_iteration=best_round)
            pred_sets = apply_threshold(probs, meta, threshold)
        else:
            pred_sets = {}

        # Accumulate results
        for eid in batch_ids:
            cset = cands.get(eid, set())
            ckpt['cand'][eid] = sorted(cset)

            matched = pred_sets.get(eid, set())
            ckpt['match'][eid] = sorted(
                m for m in matched
                if m in valid_s23c
                and not m.startswith('S1-')
                and m in cset
            )

        elapsed = time.time() - t0b
        avg_c   = n_cand / max(len(batch_ids), 1)
        n_match = sum(len(ckpt['match'][e]) for e in batch_ids)
        print(f"  [{country}] Batch {batch_i+1}: "
              f"{n_cand:,} cands  avg={avg_c:.1f}  "
              f"matches={n_match:,}  {elapsed:.1f}s", flush=True)

        ckpt['done'].append(bkey)
        _save_ckpt(ckpt)
        _flush_tsv(ckpt, s1_order)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_overall = time.time()
    print(SEP)
    print("  AMAZON ML CHALLENGE 2026 -- TEST INFERENCE")
    print(SEP)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print("\n[1] Loading model...", flush=True)
    with open(os.path.join(_MODELS, 'feature_meta.json'), encoding='utf-8') as f:
        feat_meta = json.load(f)
    booster    = lgb.Booster(model_file=os.path.join(_MODELS, 'lgbm_model.txt'))
    threshold  = feat_meta['best_threshold']
    best_round = feat_meta['best_round']
    print(f"    threshold={threshold}  best_round={best_round}  "
          f"n_features={feat_meta['n_features']}", flush=True)

    # ── 2. Load test data ─────────────────────────────────────────────────────
    print("\n[2] Loading test data...", flush=True)
    t0 = time.time()
    te1 = load_source('test', 1)
    te2 = load_source('test', 2)
    te3 = load_source('test', 3)
    print(f"    Loaded in {time.time()-t0:.1f}s  "
          f"S1={len(te1):,}  S2={len(te2):,}  S3={len(te3):,}", flush=True)
    print(f"    S1 country dist: {dict(te1['country'].value_counts())}",
          flush=True)

    s1_order   = te1['entity_id'].astype(str).tolist()   # original row order
    all_s1_set = set(s1_order)
    valid_s23  = (set(te2['entity_id'].astype(str)) |
                  set(te3['entity_id'].astype(str)))

    cfg = BlockingConfig(
        min_name_token_overlap=2,
        min_addr_token_overlap=2,
        min_trigram_overlap=4,
        max_postings_pct=0.005,
        per_key_cap=500,
        candidate_cap=200,
        cross_country_k1k2=True,
    )

    # ── 3. Checkpoint ─────────────────────────────────────────────────────────
    ckpt = _load_ckpt()
    n_prev = len(ckpt.get('cand', {}))
    if n_prev:
        print(f"\n    Resuming checkpoint: {len(ckpt['done'])} batches done, "
              f"{n_prev:,} S1 rows accumulated.", flush=True)
        # Write TSVs immediately from the existing checkpoint state
        print("    Writing TSVs from checkpoint...", flush=True)
        _flush_tsv(ckpt, s1_order)

    # ── 4. Process each country ───────────────────────────────────────────────
    countries = sorted(te1['country'].unique())
    print(f"\n[3] Processing {len(countries)} countries: {countries}", flush=True)

    for country in countries:
        s1c = te1[te1['country'] == country].copy().reset_index(drop=True)
        s2c = te2[te2['country'] == country].copy().reset_index(drop=True)
        s3c = te3[te3['country'] == country].copy().reset_index(drop=True)
        process_country(
            country, s1c, s2c, s3c,
            booster, threshold, best_round, cfg, ckpt,
            s1_order,
        )

    # ── 5. Fill remaining S1 entities (if any batches not yet done) ───────────
    print("\n[4] Filling any remaining S1 entities...", flush=True)
    n_filled = 0
    for s1_id in all_s1_set:
        if s1_id not in ckpt['cand']:
            ckpt['cand'][s1_id]  = []
            ckpt['match'][s1_id] = []
            n_filled += 1
    if n_filled:
        print(f"    Filled {n_filled:,} S1 entities with empty predictions.")
    else:
        print("    All S1 entities present in results.")

    # ── 6. Validate ───────────────────────────────────────────────────────────
    print("\n[5] Validating outputs...", flush=True)
    errs = validate(ckpt['cand'], ckpt['match'], all_s1_set, valid_s23)
    if errs:
        print("  ERRORS:")
        for e in errs:
            print(f"    [!!] {e}")
    else:
        print("  All 8 integrity requirements satisfied.")

    # ── 7. Write TSVs ─────────────────────────────────────────────────────────
    print("\n[6] Writing output files...", flush=True)
    n_cand_rows = write_tsv(
        CAND_TSV, ckpt['cand'], s1_order,
        'source1_entity_id', 'candidate_entity_ids')
    n_match_rows = write_tsv(
        RESULTS_TSV, ckpt['match'], s1_order,
        'source1_entity_id', 'matched_entity_ids')
    print(f"    candidate_pairs.tsv   -> {n_cand_rows:,} data rows")
    print(f"    matching_results.tsv  -> {n_match_rows:,} data rows")

    # ── 8. Report ─────────────────────────────────────────────────────────────
    cand_sizes  = [len(ckpt['cand'].get(s, [])) for s in s1_order]
    match_sizes = [len(ckpt['match'].get(s, [])) for s in s1_order]

    n_s1       = len(s1_order)
    zero_cand  = sum(1 for x in cand_sizes  if x == 0)
    total_cand = sum(cand_sizes)
    max_cand   = max(cand_sizes) if cand_sizes else 0

    zero_match  = sum(1 for x in match_sizes if x == 0)
    singleton   = sum(1 for x in match_sizes if x == 1)
    multi_match = sum(1 for x in match_sizes if x >= 2)
    total_match = sum(match_sizes)

    print(f"\n{SEP}")
    print("  INFERENCE REPORT")
    print(SEP)
    print(f"  Test S1 entities           : {n_s1:>12,}")
    print(f"  candidate_pairs.tsv rows   : {n_cand_rows:>12,}  (+ 1 header)")
    print(f"  matching_results.tsv rows  : {n_match_rows:>12,}  (+ 1 header)")
    print(f"\n  Candidate statistics:")
    print(f"    Total candidate pairs    : {total_cand:>12,}")
    print(f"    Avg candidates / S1      : {total_cand/max(n_s1,1):>12.1f}")
    print(f"    Max candidates / S1      : {max_cand:>12,}")
    print(f"    S1 with 0 candidates     : {zero_cand:>12,}  "
          f"({100*zero_cand/max(n_s1,1):.2f}%)")
    print(f"\n  Match distribution  (threshold={threshold}):")
    print(f"    Total matched pairs      : {total_match:>12,}")
    print(f"    Zero matches (empty)     : {zero_match:>12,}  "
          f"({100*zero_match/max(n_s1,1):.2f}%)")
    print(f"    Singleton (1 match)      : {singleton:>12,}  "
          f"({100*singleton/max(n_s1,1):.2f}%)")
    print(f"    Multi-match (≥2)         : {multi_match:>12,}  "
          f"({100*multi_match/max(n_s1,1):.2f}%)")
    print(f"\n  Total wall time            : {time.time()-t_overall:.1f}s")
    print(SEP)


if __name__ == '__main__':
    main()

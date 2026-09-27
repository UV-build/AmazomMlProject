"""
Amazon ML Challenge 2026 - Business Entity Resolution
run_sanity_checks.py

Driver script: loads every dataset file via data_ingestion.py,
runs all sanity checks, and prints a structured report.

Usage:
    python run_sanity_checks.py
"""

from __future__ import annotations

import sys
from data_ingestion import (
    load_and_validate_source,
    load_and_validate_ground_truth,
    SourceValidation,
    GroundTruthValidation,
)

DIVIDER = '=' * 72
SECTION = '-' * 72

# ──────────────────────────────────────────────────────────────────────────────
# Printer helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ok(msg: str) -> str:
    return f"  [OK]  {msg}"

def _warn(msg: str) -> str:
    return f"  [!!]  {msg}"

def _info(msg: str) -> str:
    return f"  [--]  {msg}"


def print_source_report(v: SourceValidation) -> bool:
    """Print the validation report for one source file. Returns True if clean."""
    clean = True

    print(f"\n{SECTION}")
    print(f"  FILE : {v.label}")
    print(f"  PATH : {v.path}")
    print(SECTION)

    # Columns
    if v.column_ok:
        print(_ok(f"Columns: {v.columns}"))
    else:
        print(_warn(f"COLUMN MISMATCH — got {v.columns}"))
        clean = False

    # Row count
    print(_info(f"Row count (excl. header): {v.row_count:,}"))

    # Nulls
    any_nulls = any(n > 0 for n in v.null_counts.values())
    if not any_nulls:
        print(_ok("No null/empty values in any column"))
    else:
        for col, cnt in v.null_counts.items():
            if cnt > 0:
                pct = 100.0 * cnt / v.row_count if v.row_count else 0
                print(_warn(f"Null '{col}': {cnt:,} rows  ({pct:.2f}%)"))
            else:
                print(_ok(f"No nulls in '{col}'"))

    # ID prefix
    if v.prefix_ok:
        print(_ok(f"All {v.row_count:,} entity_ids have correct prefix"))
    else:
        print(_warn(f"Bad-prefix IDs: {len(v.bad_prefix_ids):,} — first 5: {v.bad_prefix_ids[:5]}"))
        clean = False

    # Duplicates
    if not v.duplicate_ids:
        print(_ok("No duplicate entity_ids"))
    else:
        print(_warn(f"Duplicate entity_ids: {len(v.duplicate_ids):,} — first 5: {v.duplicate_ids[:5]}"))
        clean = False

    # Country distribution
    total = v.row_count or 1
    dist_str = ', '.join(
        f"{country}: {cnt:,} ({100*cnt/total:.1f}%)"
        for country, cnt in sorted(v.country_dist.items(), key=lambda x: -x[1])
    )
    print(_info(f"Country distribution: {dist_str}"))

    return clean


def print_ground_truth_report(v: GroundTruthValidation) -> bool:
    """Print the ground-truth validation report. Returns True if clean."""
    clean = True

    print(f"\n{SECTION}")
    print(f"  FILE : train_ground_truth")
    print(f"  PATH : {v.path}")
    print(SECTION)

    # Columns
    if v.column_ok:
        print(_ok(f"Columns: {v.columns}"))
    else:
        print(_warn(f"COLUMN MISMATCH — got {v.columns}"))
        clean = False

    # Row count
    print(_info(f"Row count (excl. header): {v.row_count:,}"))

    # Nulls
    if v.null_source1_id == 0:
        print(_ok("No null source1_entity_id values"))
    else:
        print(_warn(f"Null source1_entity_id: {v.null_source1_id:,}"))
        clean = False

    if v.null_matched_ids == 0:
        print(_ok("No null matched_entity_ids values"))
    else:
        print(_info(
            f"Empty matched_entity_ids: {v.null_matched_ids:,} rows "
            f"({100*v.null_matched_ids/v.row_count:.2f}%) — "
            f"these are legitimately unmatched S1 entities"
        ))
        # Not a data error — unmatched entities are valid

    # source1_entity_id prefix
    if not v.bad_s1_prefix_ids:
        print(_ok("All source1_entity_id values have S1- prefix"))
    else:
        print(_warn(f"Non-S1- source1_entity_id: {len(v.bad_s1_prefix_ids):,} — first 5: {v.bad_s1_prefix_ids[:5]}"))
        clean = False

    # source1_entity_id existence in train_source1
    if not v.unknown_s1_ids:
        print(_ok(f"All {v.row_count:,} source1_entity_ids found in train_source1"))
    else:
        print(_warn(f"source1_entity_ids missing from train_source1: {len(v.unknown_s1_ids):,} — first 5: {v.unknown_s1_ids[:5]}"))
        clean = False

    # Duplicate rows
    if not v.duplicate_s1_ids:
        print(_ok("No duplicate source1_entity_id rows"))
    else:
        print(_warn(f"Duplicate source1_entity_id rows: {len(v.duplicate_s1_ids):,} — first 5: {v.duplicate_s1_ids[:5]}"))
        clean = False

    # Matched ID counts
    print(_info(f"Total matched ID references: {v.total_matched_refs:,}"))
    print(_info(f"  S2 references: {v.s2_ref_count:,}"))
    print(_info(f"  S3 references: {v.s3_ref_count:,}"))

    # No S1 IDs in matched column — critical check
    if not v.s1_in_matched:
        print(_ok("No S1 IDs found in matched_entity_ids (correct)"))
    else:
        print(_warn(f"CRITICAL: S1 IDs in matched_entity_ids: {len(v.s1_in_matched):,} — first 5: {v.s1_in_matched[:5]}"))
        clean = False

    # No unknown-prefix IDs
    if not v.unknown_prefix_matched:
        print(_ok("No unknown-prefix IDs in matched_entity_ids"))
    else:
        print(_warn(f"Unknown-prefix matched IDs: {len(v.unknown_prefix_matched):,} — first 5: {v.unknown_prefix_matched[:5]}"))
        clean = False

    # Referential integrity
    if not v.s2_not_found:
        print(_ok(f"All {v.s2_ref_count:,} S2 matched IDs exist in train_source2"))
    else:
        print(_warn(f"S2 matched IDs missing from train_source2: {len(v.s2_not_found):,} — first 5: {v.s2_not_found[:5]}"))
        clean = False

    if not v.s3_not_found:
        print(_ok(f"All {v.s3_ref_count:,} S3 matched IDs exist in train_source3"))
    else:
        print(_warn(f"S3 matched IDs missing from train_source3: {len(v.s3_not_found):,} — first 5: {v.s3_not_found[:5]}"))
        clean = False

    # Match-count distribution
    print(_info(
        f"Matches per S1 entity — "
        f"min: {v.match_min}, max: {v.match_max}, avg: {v.match_avg:.2f}, "
        f"unmatched: {v.unmatched_s1_count:,}"
    ))
    dist_sorted = sorted(v.match_count_dist.items())
    dist_str = '  '.join(f"{k}→{cnt:,}" for k, cnt in dist_sorted)
    print(_info(f"Distribution (matches→count): {dist_str}"))

    return clean


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    problems: list[str] = []

    print(DIVIDER)
    print("  AMAZON ML CHALLENGE 2026 — DATA INGESTION & SANITY CHECK REPORT")
    print(DIVIDER)

    # ── TRAIN SOURCE FILES ────────────────────────────────────────────────────
    print("\n\n>>> TRAIN SOURCE FILES")

    print("\n  Loading train_source1 ...", flush=True)
    ts1 = load_and_validate_source('train', 1)
    if not print_source_report(ts1):
        problems.append("train_source1: validation failures")

    print("\n  Loading train_source2 ...", flush=True)
    ts2 = load_and_validate_source('train', 2)
    if not print_source_report(ts2):
        problems.append("train_source2: validation failures")

    print("\n  Loading train_source3 ...", flush=True)
    ts3 = load_and_validate_source('train', 3)
    if not print_source_report(ts3):
        problems.append("train_source3: validation failures")

    # Build ID sets for ground-truth integrity checks
    s1_ids = set(ts1.df['entity_id'].dropna())
    s2_ids = set(ts2.df['entity_id'].dropna())
    s3_ids = set(ts3.df['entity_id'].dropna())

    # ── TEST SOURCE FILES ─────────────────────────────────────────────────────
    print("\n\n>>> TEST SOURCE FILES")

    print("\n  Loading test_source1 ...", flush=True)
    es1 = load_and_validate_source('test', 1)
    if not print_source_report(es1):
        problems.append("test_source1: validation failures")

    print("\n  Loading test_source2 ...", flush=True)
    es2 = load_and_validate_source('test', 2)
    if not print_source_report(es2):
        problems.append("test_source2: validation failures")

    print("\n  Loading test_source3 ...", flush=True)
    es3 = load_and_validate_source('test', 3)
    if not print_source_report(es3):
        problems.append("test_source3: validation failures")

    # ── GROUND TRUTH ──────────────────────────────────────────────────────────
    print("\n\n>>> GROUND TRUTH")

    print("\n  Loading train_ground_truth ...", flush=True)
    gt = load_and_validate_ground_truth(s1_ids, s2_ids, s3_ids)
    if not print_ground_truth_report(gt):
        problems.append("train_ground_truth: validation failures")

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    print(f"\n\n{DIVIDER}")
    print("  SUMMARY")
    print(DIVIDER)
    rows = [
        ("train_source1",      ts1.row_count, ts1.null_counts.get('business_address', 0), ts1.country_dist),
        ("train_source2",      ts2.row_count, ts2.null_counts.get('business_address', 0), ts2.country_dist),
        ("train_source3",      ts3.row_count, ts3.null_counts.get('business_address', 0), ts3.country_dist),
        ("test_source1",       es1.row_count, es1.null_counts.get('business_address', 0), es1.country_dist),
        ("test_source2",       es2.row_count, es2.null_counts.get('business_address', 0), es2.country_dist),
        ("test_source3",       es3.row_count, es3.null_counts.get('business_address', 0), es3.country_dist),
        ("train_ground_truth", gt.row_count,  gt.null_matched_ids,                        {}),
    ]
    print(f"\n  {'File':<22}  {'Rows':>10}  {'Null addr':>10}  Countries")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*28}")
    for name, rc, nulls, cdist in rows:
        countries = ', '.join(sorted(cdist.keys())) if cdist else 'N/A'
        print(f"  {name:<22}  {rc:>10,}  {nulls:>10,}  {countries}")

    print(f"\n  Ground-truth matched refs : {gt.total_matched_refs:,}")
    print(f"  S2 refs                   : {gt.s2_ref_count:,}")
    print(f"  S3 refs                   : {gt.s3_ref_count:,}")
    print(f"  Unmatched S1 entities     : {gt.unmatched_s1_count:,}")
    print(f"  Unique matched ID set size: {len(gt.matched_id_set):,}")

    # ── PROBLEMS ──────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    if problems:
        print("  PROBLEMS FOUND:")
        for p in problems:
            print(f"    [!!] {p}")
        print(DIVIDER)
        return 1
    else:
        print("  ALL CHECKS PASSED — dataset is clean and ready.")
        print(DIVIDER)
        return 0


if __name__ == '__main__':
    sys.exit(main())

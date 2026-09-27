"""
Amazon ML Challenge 2026 - Business Entity Resolution
Dataset Inspection Script
Checks: columns, row counts, nulls, country values, ID prefixes, ground-truth integrity
"""

import os
import csv
from collections import defaultdict, Counter

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data_set')
TRAIN = os.path.join(BASE, 'train')
TEST  = os.path.join(BASE, 'test')

TRAIN_FILES = {
    'train_source1': os.path.join(TRAIN, 'train_source1.tsv'),
    'train_source2': os.path.join(TRAIN, 'train_source2.tsv'),
    'train_source3': os.path.join(TRAIN, 'train_source3.tsv'),
    'train_ground_truth': os.path.join(TRAIN, 'train_ground_truth.tsv'),
}
TEST_FILES = {
    'test_source1': os.path.join(TEST, 'test_source1.tsv'),
    'test_source2': os.path.join(TEST, 'test_source2.tsv'),
    'test_source3': os.path.join(TEST, 'test_source3.tsv'),
}

EXPECTED_SOURCE_COLS = ['entity_id', 'business_name', 'business_address', 'country']
EXPECTED_GT_COLS     = ['source1_entity_id', 'matched_entity_ids']

SEP = '=' * 70

def read_tsv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    return rows

def get_columns(path):
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        return next(reader)

def inspect_source_file(label, path):
    print(f"\n{SEP}")
    print(f"FILE: {label}  ({os.path.basename(path)})")
    print(SEP)

    cols = get_columns(path)
    print(f"  Columns ({len(cols)}): {cols}")

    col_mismatch = cols != EXPECTED_SOURCE_COLS
    if col_mismatch:
        print(f"  *** COLUMN MISMATCH! Expected: {EXPECTED_SOURCE_COLS}")
    else:
        print(f"  Column check: OK")

    rows = read_tsv(path)
    print(f"  Row count (excl. header): {len(rows)}")

    # Null / empty counts per column
    null_counts = {c: 0 for c in cols}
    for row in rows:
        for c in cols:
            val = row.get(c, None)
            if val is None or val.strip() == '':
                null_counts[c] += 1
    print(f"  Null/empty counts: {null_counts}")

    # Country values
    country_counter = Counter(row.get('country', '').strip() for row in rows)
    print(f"  Distinct country values ({len(country_counter)}): {dict(country_counter.most_common())}")

    # ID prefix check
    prefix_map = {'train_source1': 'S1-', 'train_source2': 'S2-', 'train_source3': 'S3-',
                  'test_source1':  'S1-', 'test_source2':  'S2-', 'test_source3':  'S3-'}
    expected_prefix = prefix_map.get(label, None)
    bad_prefix = [row['entity_id'] for row in rows if not row.get('entity_id','').startswith(expected_prefix)] if expected_prefix else []
    if bad_prefix:
        print(f"  *** BAD ID PREFIX (expected '{expected_prefix}'): {len(bad_prefix)} rows — first 5: {bad_prefix[:5]}")
    else:
        print(f"  ID prefix check ('{expected_prefix}'): OK — all {len(rows)} IDs have correct prefix")

    # Duplicate IDs
    id_counter = Counter(row.get('entity_id','') for row in rows)
    dupes = {k: v for k, v in id_counter.items() if v > 1}
    if dupes:
        print(f"  *** DUPLICATE entity_ids: {len(dupes)} — first 5: {list(dupes.items())[:5]}")
    else:
        print(f"  Duplicate ID check: OK — all entity_ids unique")

    return rows

def inspect_ground_truth(path, s2_ids, s3_ids, s1_ids):
    label = 'train_ground_truth'
    print(f"\n{SEP}")
    print(f"FILE: {label}  ({os.path.basename(path)})")
    print(SEP)

    cols = get_columns(path)
    print(f"  Columns ({len(cols)}): {cols}")
    col_mismatch = cols != EXPECTED_GT_COLS
    if col_mismatch:
        print(f"  *** COLUMN MISMATCH! Expected: {EXPECTED_GT_COLS}")
    else:
        print(f"  Column check: OK")

    rows = read_tsv(path)
    print(f"  Row count (excl. header): {len(rows)}")

    # Null checks
    null_s1 = sum(1 for r in rows if not r.get('source1_entity_id','').strip())
    null_matched = sum(1 for r in rows if not r.get('matched_entity_ids','').strip())
    print(f"  Null source1_entity_id: {null_s1}")
    print(f"  Null matched_entity_ids: {null_matched}")

    # All source1_entity_id must start with S1-
    bad_s1_prefix = [r['source1_entity_id'] for r in rows if not r['source1_entity_id'].startswith('S1-')]
    if bad_s1_prefix:
        print(f"  *** source1_entity_id non-S1- prefix: {len(bad_s1_prefix)} — first 5: {bad_s1_prefix[:5]}")
    else:
        print(f"  source1_entity_id prefix (S1-): OK")

    # Verify source1_entity_ids exist in train_source1
    unknown_s1 = [r['source1_entity_id'] for r in rows if r['source1_entity_id'] not in s1_ids]
    if unknown_s1:
        print(f"  *** source1_entity_id not in train_source1: {len(unknown_s1)} — first 5: {unknown_s1[:5]}")
    else:
        print(f"  source1_entity_id existence in train_source1: OK — all {len(rows)} found")

    # Parse matched IDs
    all_matched = []
    for r in rows:
        ids = [x.strip() for x in r['matched_entity_ids'].split(',') if x.strip()]
        all_matched.extend(ids)

    total_matched = len(all_matched)
    print(f"  Total matched IDs (across all rows): {total_matched}")

    # Count per source
    s2_matched = [i for i in all_matched if i.startswith('S2-')]
    s3_matched = [i for i in all_matched if i.startswith('S3-')]
    s1_in_matched = [i for i in all_matched if i.startswith('S1-')]
    other_matched = [i for i in all_matched if not (i.startswith('S1-') or i.startswith('S2-') or i.startswith('S3-'))]

    print(f"  Matched IDs from S2: {len(s2_matched)}")
    print(f"  Matched IDs from S3: {len(s3_matched)}")
    print(f"  Matched IDs from S1 (SHOULD BE 0): {len(s1_in_matched)}")
    print(f"  Matched IDs with unknown prefix (SHOULD BE 0): {len(other_matched)}")

    if s1_in_matched:
        print(f"  *** PROBLEM: S1 IDs in matched_entity_ids: {s1_in_matched[:5]}")
    else:
        print(f"  No S1 IDs in matched_entity_ids: OK")

    if other_matched:
        print(f"  *** PROBLEM: Unknown-prefix IDs in matched: {other_matched[:5]}")

    # Verify matched IDs exist in train_source2 or train_source3
    s2_not_found = [i for i in s2_matched if i not in s2_ids]
    s3_not_found = [i for i in s3_matched if i not in s3_ids]

    if s2_not_found:
        print(f"  *** S2 matched IDs not found in train_source2: {len(s2_not_found)} — first 5: {s2_not_found[:5]}")
    else:
        print(f"  All S2 matched IDs exist in train_source2: OK ({len(s2_matched)} checked)")

    if s3_not_found:
        print(f"  *** S3 matched IDs not found in train_source3: {len(s3_not_found)} — first 5: {s3_not_found[:5]}")
    else:
        print(f"  All S3 matched IDs exist in train_source3: OK ({len(s3_matched)} checked)")

    # Distribution: how many matches per S1 entity
    match_counts = [len([x.strip() for x in r['matched_entity_ids'].split(',') if x.strip()]) for r in rows]
    print(f"\n  Matches-per-S1-entity stats:")
    print(f"    Min: {min(match_counts)}")
    print(f"    Max: {max(match_counts)}")
    print(f"    Avg: {sum(match_counts)/len(match_counts):.2f}")
    dist = Counter(match_counts)
    print(f"    Distribution: { {k: dist[k] for k in sorted(dist)} }")

    # Duplicate source1_entity_id rows
    gt_id_counter = Counter(r['source1_entity_id'] for r in rows)
    gt_dupes = {k: v for k, v in gt_id_counter.items() if v > 1}
    if gt_dupes:
        print(f"  *** Duplicate source1_entity_id rows: {len(gt_dupes)} — {list(gt_dupes.items())[:5]}")
    else:
        print(f"  Duplicate source1_entity_id check: OK")

    return rows


def main():
    print(SEP)
    print("AMAZON ML CHALLENGE 2026 — DATASET INSPECTION REPORT")
    print(SEP)

    # ── TRAIN SOURCE FILES ──────────────────────────────────────────────
    print("\n\n>>> TRAIN SOURCE FILES")
    s1_rows = inspect_source_file('train_source1', TRAIN_FILES['train_source1'])
    s2_rows = inspect_source_file('train_source2', TRAIN_FILES['train_source2'])
    s3_rows = inspect_source_file('train_source3', TRAIN_FILES['train_source3'])

    s1_ids = set(r['entity_id'] for r in s1_rows)
    s2_ids = set(r['entity_id'] for r in s2_rows)
    s3_ids = set(r['entity_id'] for r in s3_rows)

    # ── TEST SOURCE FILES ───────────────────────────────────────────────
    print("\n\n>>> TEST SOURCE FILES")
    inspect_source_file('test_source1', TEST_FILES['test_source1'])
    inspect_source_file('test_source2', TEST_FILES['test_source2'])
    inspect_source_file('test_source3', TEST_FILES['test_source3'])

    # ── GROUND TRUTH ────────────────────────────────────────────────────
    print("\n\n>>> GROUND TRUTH")
    inspect_ground_truth(TRAIN_FILES['train_ground_truth'], s2_ids, s3_ids, s1_ids)

    # ── SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n\n{SEP}")
    print("SUMMARY")
    print(SEP)
    print(f"  train_source1 rows : {len(s1_rows)}")
    print(f"  train_source2 rows : {len(s2_rows)}")
    print(f"  train_source3 rows : {len(s3_rows)}")
    print(f"  Unique S1 IDs      : {len(s1_ids)}")
    print(f"  Unique S2 IDs      : {len(s2_ids)}")
    print(f"  Unique S3 IDs      : {len(s3_ids)}")
    print(f"\nInspection complete.")


if __name__ == '__main__':
    main()

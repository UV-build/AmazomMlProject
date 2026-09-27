"""
Amazon ML Challenge 2026 - Business Entity Resolution
data_ingestion.py

Reusable functions for:
  - Loading every TSV file into a pandas DataFrame
  - Validating columns, ID prefixes, null counts, country distributions
  - Parsing train_ground_truth.tsv into usable structures
  - Verifying ground-truth referential integrity

No external data, blocking, or ML logic here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_SRC_DIR, '..', '..', '..', 'data_set'))
_TRAIN    = os.path.join(_DATA_DIR, 'train')
_TEST     = os.path.join(_DATA_DIR, 'test')


def _path(split: str, filename: str) -> str:
    """Return absolute path for a dataset file."""
    base = _TRAIN if split == 'train' else _TEST
    return os.path.join(base, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Schema constants
# ──────────────────────────────────────────────────────────────────────────────

SOURCE_COLS: List[str] = ['entity_id', 'business_name', 'business_address', 'country']
GT_COLS:     List[str] = ['source1_entity_id', 'matched_entity_ids']

ID_PREFIX: Dict[str, str] = {
    'source1': 'S1-',
    'source2': 'S2-',
    'source3': 'S3-',
}


# ──────────────────────────────────────────────────────────────────────────────
# Return-type dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceValidation:
    """Results of validating a single source TSV."""
    label:           str
    path:            str
    row_count:       int
    columns:         List[str]
    column_ok:       bool
    null_counts:     Dict[str, int]
    country_dist:    Dict[str, int]
    prefix_ok:       bool
    bad_prefix_ids:  List[str]
    duplicate_ids:   List[str]
    df:              pd.DataFrame   # the loaded DataFrame


@dataclass
class GroundTruthValidation:
    """Results of validating train_ground_truth.tsv."""
    path:                   str
    row_count:              int
    columns:                List[str]
    column_ok:              bool
    null_source1_id:        int
    null_matched_ids:       int          # rows with empty matched_entity_ids
    unmatched_s1_count:     int          # S1 entities with zero matches
    bad_s1_prefix_ids:      List[str]
    unknown_s1_ids:         List[str]    # source1_entity_id not in train_source1
    total_matched_refs:     int          # total comma-expanded matched IDs
    s2_ref_count:           int
    s3_ref_count:           int
    s1_in_matched:          List[str]    # MUST be empty
    unknown_prefix_matched: List[str]    # MUST be empty
    s2_not_found:           List[str]    # S2 refs missing from train_source2
    s3_not_found:           List[str]    # S3 refs missing from train_source3
    duplicate_s1_ids:       List[str]
    match_count_dist:       Dict[int, int]
    match_min:              int
    match_max:              int
    match_avg:              float
    # Usable structures
    gt_df:                  pd.DataFrame          # raw DataFrame
    s1_to_matches:          Dict[str, List[str]]  # {S1-id: [S2/S3-ids, ...]}
    matched_id_set:         Set[str]              # flat set of all matched IDs


# ──────────────────────────────────────────────────────────────────────────────
# Core loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_source(split: str, source_num: int) -> pd.DataFrame:
    """
    Load a source TSV (source1 / source2 / source3) for the given split
    ('train' or 'test') into a DataFrame.

    Reads with sep='\\t', utf-8 encoding, no type inference on entity_id.
    """
    filename = f'{split}_source{source_num}.tsv'
    path = _path(split, filename)
    df = pd.read_csv(
        path,
        sep='\t',
        encoding='utf-8',
        dtype={'entity_id': str, 'business_name': str,
               'business_address': str, 'country': str},
        keep_default_na=False,   # treat empty strings as empty, not NaN
        na_values=[''],          # but still flag empty strings as NaN for null checks
    )
    return df


def load_ground_truth() -> pd.DataFrame:
    """
    Load train_ground_truth.tsv.

    Returns a DataFrame with columns:
        source1_entity_id  (str)
        matched_entity_ids (str)  — raw comma-separated string; may be empty
    """
    path = _path('train', 'train_ground_truth.tsv')
    df = pd.read_csv(
        path,
        sep='\t',
        encoding='utf-8',
        dtype={'source1_entity_id': str, 'matched_entity_ids': str},
        keep_default_na=False,
        na_values=[''],
    )
    return df


def parse_ground_truth(gt_df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Set[str]]:
    """
    Parse a ground-truth DataFrame into two usable structures.

    Returns
    -------
    s1_to_matches : dict  {source1_entity_id -> [matched_id, ...]}
                    Unmatched S1 entities map to an empty list.
    matched_id_set : set  flat set of every referenced S2/S3 ID
    """
    s1_to_matches: Dict[str, List[str]] = {}
    matched_id_set: Set[str] = set()

    for _, row in gt_df.iterrows():
        s1_id = str(row['source1_entity_id']).strip()
        raw   = row['matched_entity_ids']

        if pd.isna(raw) or str(raw).strip() == '':
            s1_to_matches[s1_id] = []
        else:
            ids = [x.strip() for x in str(raw).split(',') if x.strip()]
            s1_to_matches[s1_id] = ids
            matched_id_set.update(ids)

    return s1_to_matches, matched_id_set


# ──────────────────────────────────────────────────────────────────────────────
# Validators
# ──────────────────────────────────────────────────────────────────────────────

def validate_source(
    df: pd.DataFrame,
    label: str,
    path: str,
    source_key: str,          # 'source1' | 'source2' | 'source3'
) -> SourceValidation:
    """
    Run all sanity checks on a source DataFrame.

    Parameters
    ----------
    df         : DataFrame returned by load_source()
    label      : human-readable name, e.g. 'train_source2'
    path       : file path (for reporting)
    source_key : 'source1', 'source2', or 'source3'
    """
    cols      = list(df.columns)
    col_ok    = (cols == SOURCE_COLS)
    row_count = len(df)

    # Null counts (NaN + strings that are literally blank after strip)
    null_counts: Dict[str, int] = {}
    for c in cols:
        if c in df.columns:
            null_counts[c] = int(df[c].isna().sum())

    # Country distribution
    country_dist = (
        df['country'].fillna('').str.strip().value_counts().to_dict()
        if 'country' in df.columns else {}
    )

    # ID prefix check
    expected_prefix = ID_PREFIX.get(source_key, '')
    if 'entity_id' in df.columns:
        bad_mask        = ~df['entity_id'].str.startswith(expected_prefix, na=False)
        bad_prefix_ids  = df.loc[bad_mask, 'entity_id'].tolist()
    else:
        bad_prefix_ids = []

    prefix_ok = len(bad_prefix_ids) == 0

    # Duplicate entity_ids
    dup_mask      = df['entity_id'].duplicated(keep=False) if 'entity_id' in df.columns else pd.Series([], dtype=bool)
    duplicate_ids = df.loc[dup_mask, 'entity_id'].unique().tolist() if 'entity_id' in df.columns else []

    return SourceValidation(
        label=label,
        path=path,
        row_count=row_count,
        columns=cols,
        column_ok=col_ok,
        null_counts=null_counts,
        country_dist=country_dist,
        prefix_ok=prefix_ok,
        bad_prefix_ids=bad_prefix_ids,
        duplicate_ids=duplicate_ids,
        df=df,
    )


def validate_ground_truth(
    gt_df: pd.DataFrame,
    s1_id_set: Set[str],
    s2_id_set: Set[str],
    s3_id_set: Set[str],
    path: str,
) -> GroundTruthValidation:
    """
    Run all sanity checks on the ground-truth DataFrame.

    Parameters
    ----------
    gt_df     : DataFrame from load_ground_truth()
    s1_id_set : set of all entity_ids in train_source1
    s2_id_set : set of all entity_ids in train_source2
    s3_id_set : set of all entity_ids in train_source3
    path      : file path (for reporting)
    """
    cols      = list(gt_df.columns)
    col_ok    = (cols == GT_COLS)
    row_count = len(gt_df)

    null_source1_id  = int(gt_df['source1_entity_id'].isna().sum())  if 'source1_entity_id'  in gt_df.columns else -1
    null_matched_ids = int(gt_df['matched_entity_ids'].isna().sum())  if 'matched_entity_ids' in gt_df.columns else -1

    # source1_entity_id prefix
    bad_s1_prefix_ids = (
        gt_df.loc[~gt_df['source1_entity_id'].str.startswith('S1-', na=False),
                  'source1_entity_id'].tolist()
        if 'source1_entity_id' in gt_df.columns else []
    )

    # source1_entity_id existence in train_source1
    unknown_s1_ids = (
        gt_df.loc[~gt_df['source1_entity_id'].isin(s1_id_set),
                  'source1_entity_id'].tolist()
        if 'source1_entity_id' in gt_df.columns else []
    )

    # Parse all matched IDs
    s1_to_matches, matched_id_set = parse_ground_truth(gt_df)

    all_matched: List[str] = [mid for ids in s1_to_matches.values() for mid in ids]

    s2_matched             = [i for i in all_matched if i.startswith('S2-')]
    s3_matched             = [i for i in all_matched if i.startswith('S3-')]
    s1_in_matched          = [i for i in all_matched if i.startswith('S1-')]
    unknown_prefix_matched = [i for i in all_matched
                               if not (i.startswith('S1-') or i.startswith('S2-') or i.startswith('S3-'))]

    # Referential integrity: matched IDs must exist in S2 or S3
    s2_not_found = [i for i in s2_matched if i not in s2_id_set]
    s3_not_found = [i for i in s3_matched if i not in s3_id_set]

    # Match-count distribution
    match_counts        = [len(v) for v in s1_to_matches.values()]
    unmatched_s1_count  = sum(1 for c in match_counts if c == 0)
    match_count_dist    = {}
    for c in match_counts:
        match_count_dist[c] = match_count_dist.get(c, 0) + 1

    match_min = min(match_counts) if match_counts else 0
    match_max = max(match_counts) if match_counts else 0
    match_avg = sum(match_counts) / len(match_counts) if match_counts else 0.0

    # Duplicate source1_entity_id rows
    dup_s1 = (
        gt_df.loc[gt_df['source1_entity_id'].duplicated(keep=False), 'source1_entity_id']
             .unique().tolist()
        if 'source1_entity_id' in gt_df.columns else []
    )

    return GroundTruthValidation(
        path=path,
        row_count=row_count,
        columns=cols,
        column_ok=col_ok,
        null_source1_id=null_source1_id,
        null_matched_ids=null_matched_ids,
        unmatched_s1_count=unmatched_s1_count,
        bad_s1_prefix_ids=bad_s1_prefix_ids,
        unknown_s1_ids=unknown_s1_ids,
        total_matched_refs=len(all_matched),
        s2_ref_count=len(s2_matched),
        s3_ref_count=len(s3_matched),
        s1_in_matched=s1_in_matched,
        unknown_prefix_matched=unknown_prefix_matched,
        s2_not_found=s2_not_found,
        s3_not_found=s3_not_found,
        duplicate_s1_ids=dup_s1,
        match_count_dist=match_count_dist,
        match_min=match_min,
        match_max=match_max,
        match_avg=match_avg,
        gt_df=gt_df,
        s1_to_matches=s1_to_matches,
        matched_id_set=matched_id_set,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: load + validate in one call
# ──────────────────────────────────────────────────────────────────────────────

def load_and_validate_source(split: str, source_num: int) -> SourceValidation:
    """Load and validate a source file in one call."""
    filename   = f'{split}_source{source_num}.tsv'
    path       = _path(split, filename)
    label      = f'{split}_source{source_num}'
    source_key = f'source{source_num}'
    df         = load_source(split, source_num)
    return validate_source(df, label, path, source_key)


def load_and_validate_ground_truth(
    s1_id_set: Set[str],
    s2_id_set: Set[str],
    s3_id_set: Set[str],
) -> GroundTruthValidation:
    """Load and validate the ground-truth file in one call."""
    path  = _path('train', 'train_ground_truth.tsv')
    gt_df = load_ground_truth()
    return validate_ground_truth(gt_df, s1_id_set, s2_id_set, s3_id_set, path)

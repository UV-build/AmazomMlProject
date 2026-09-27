"""
validate_outputs.py
-------------------
INTERNAL output-validation script for Amazon ML Challenge 2026 —
Business Entity Resolution.

THIS IS NOT AN OFFICIAL HACKATHON VALIDATOR.
It is an independent internal check to catch formatting or content
errors before submission.

Checks performed
----------------
 1.  matching_results.tsv   exists
 2.  candidate_pairs.tsv    exists
 3.  Both are valid TSV files (UTF-8, tab-separated, parseable)
 4.  Exact required headers are present
         candidate_pairs.tsv   -> source1_entity_id  candidate_entity_ids
         matching_results.tsv  -> source1_entity_id  matched_entity_ids
 5.  Every test Source 1 entity appears exactly once in matching_results.tsv
 6.  Every test Source 1 entity appears exactly once in candidate_pairs.tsv
 7.  Every candidate ID is a valid test S2 or S3 entity ID
 8.  Every matched ID is a valid test S2 or S3 entity ID
 9.  No Source 1 IDs appear in candidate or match lists
10.  No duplicate IDs within a single entity's list
11.  matched_entity_ids is a subset of candidate_entity_ids (per entity)
12.  No unexpected columns exist in either file
13.  No unexpected Source 1 IDs exist (IDs not in test_source1.tsv)
14.  Empty lists are handled correctly (empty field, not 'nan'/None/whitespace)

Usage
-----
    python validate_outputs.py [--cand PATH] [--match PATH] [--max-errors N]

Options
-------
    --cand PATH       Path to candidate_pairs.tsv
                      (default: <project_root>/output/candidate_pairs.tsv)
    --match PATH      Path to matching_results.tsv
                      (default: <project_root>/output/matching_results.tsv)
    --max-errors N    Max per-entity errors shown per check
                      (default: 20;  0 = show all)

Exit codes
----------
    0  All checks passed
    1  One or more checks failed
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_SRC  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_SRC, '..', '..', '..'))

DEFAULT_CAND    = os.path.join(_ROOT, 'output', 'candidate_pairs.tsv')
DEFAULT_MATCH   = os.path.join(_ROOT, 'output', 'matching_results.tsv')
DEFAULT_MAX_ERR = 20

CAND_COLS  = ('source1_entity_id', 'candidate_entity_ids')
MATCH_COLS = ('source1_entity_id', 'matched_entity_ids')

SEP = '=' * 70
BAR = '-' * 70


# ─────────────────────────────────────────────────────────────────────────────
# Report accumulator
# ─────────────────────────────────────────────────────────────────────────────

class Report:
    """Collects pass/fail results and prints a structured summary."""

    def __init__(self, max_errors: int) -> None:
        self._checks: List[Tuple[str, bool, str]] = []
        self._n_fail = 0
        self._n_pass = 0
        self.max_errors = max_errors

    def ok(self, label: str, detail: str = '') -> None:
        self._checks.append((label, True, detail))
        self._n_pass += 1

    def fail(self, label: str, detail: str = '') -> None:
        self._checks.append((label, False, detail))
        self._n_fail += 1

    def passed(self) -> bool:
        return self._n_fail == 0

    def print_checks(self) -> None:
        for label, ok, detail in self._checks:
            mark = '[PASS]' if ok else '[FAIL]'
            print(f"  {mark}  {label}")
            if detail:
                # Indent continuation lines neatly
                for part in detail.split('\n'):
                    print(f"         {part}")


# ─────────────────────────────────────────────────────────────────────────────
# Reference ID loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_ref_ids() -> Tuple[Set[str], Set[str]]:
    """
    Return (s1_ids, s23_ids) from test_source*.tsv.
    Uses data_ingestion.load_source when available; falls back to pandas.
    """
    sys.path.insert(0, _SRC)
    try:
        from data_ingestion import load_source  # type: ignore
        te1 = load_source('test', 1)
        te2 = load_source('test', 2)
        te3 = load_source('test', 3)
        s1  = set(te1['entity_id'].astype(str))
        s23 = set(te2['entity_id'].astype(str)) | set(te3['entity_id'].astype(str))
        return s1, s23
    except Exception as exc_primary:
        pass

    # Fallback: read TSVs directly with pandas
    import pandas as pd
    data_dir = os.path.join(_ROOT, 'data_set', 'test')
    s1: Set[str] = set()
    s23: Set[str] = set()
    for num in [1, 2, 3]:
        path = os.path.join(data_dir, f'test_source{num}.tsv')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cannot find reference file: {path}\n"
                "Make sure data_set/test/ is present next to the output/ directory."
            )
        df = pd.read_csv(
            path, sep='\t', usecols=['entity_id'],
            dtype=str, encoding='utf-8', keep_default_na=False)
        ids = set(df['entity_id'].str.strip())
        if num == 1:
            s1 = ids
        else:
            s23 |= ids
    return s1, s23


# ─────────────────────────────────────────────────────────────────────────────
# TSV reader — returns raw cells, tracks duplicate source1_entity_id rows
# ─────────────────────────────────────────────────────────────────────────────

def _read_tsv(
    path:          str,
    expected_cols: Tuple[str, str],
    rpt:           Report,
    label:         str,
    max_err:       int,
) -> Optional[Tuple[Dict[str, str], Set[str]]]:
    """
    Read a two-column TSV.

    Returns (rows_dict, duplicate_s1_ids) where:
      rows_dict          : {source1_entity_id -> raw_value_cell}
                           When a S1 ID appears more than once the last row wins.
      duplicate_s1_ids   : set of S1 IDs that appeared more than once.

    Returns None on critical parse failure.
    Adds Check-3, Check-4, Check-12 results to rpt.
    """
    try:
        rows: Dict[str, str] = {}
        id_counts: Counter = Counter()
        bad_col_lines: List[int] = []

        with open(path, encoding='utf-8', newline='') as fh:
            reader = csv.reader(fh, delimiter='\t')

            # Header row
            try:
                header = next(reader)
            except StopIteration:
                rpt.fail(f"Check 3 [{label}]: file is empty")
                return None

            # Check 3: valid UTF-8 TSV (already reading with utf-8; if we get
            # here without UnicodeDecodeError the file is parseable)
            # Check 4: exact header
            if tuple(h.strip() for h in header) != expected_cols:
                rpt.fail(
                    f"Check 4 [{label}]: wrong header",
                    f"got {header!r}\n"
                    f"expected {list(expected_cols)!r}")
                return None

            # Check 12: exactly 2 columns
            if len(header) != 2:
                rpt.fail(
                    f"Check 12 [{label}]: {len(header)} columns found; "
                    f"expected exactly 2")
                return None

            # Data rows
            for lineno, row in enumerate(reader, start=2):
                if len(row) != 2:
                    bad_col_lines.append(lineno)
                    continue
                s1_id = row[0].strip()
                cell  = row[1]          # preserve raw value (may be empty)
                rows[s1_id]   = cell
                id_counts[s1_id] += 1

        # Check 3: every data row had exactly 2 columns
        if bad_col_lines:
            shown = bad_col_lines[:max_err]
            rpt.fail(
                f"Check 3 [{label}]: {len(bad_col_lines)} line(s) with "
                f"wrong column count",
                f"lines: {shown!r}" +
                (f" ... and {len(bad_col_lines)-max_err} more"
                 if len(bad_col_lines) > max_err else ''))
        else:
            rpt.ok(f"Check 3 [{label}]: valid UTF-8 TSV, "
                   f"all rows have exactly 2 columns")

        # Check 4 / 12 passed (header was correct)
        rpt.ok(f"Check 4 [{label}]: header is correct "
               f"{list(expected_cols)!r}")
        rpt.ok(f"Check 12 [{label}]: exactly 2 columns")

        dupes = {k for k, v in id_counts.items() if v > 1}
        return rows, dupes

    except UnicodeDecodeError as exc:
        rpt.fail(f"Check 3 [{label}]: file is not valid UTF-8", str(exc))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ID-list parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_id_list(cell: str) -> List[str]:
    """
    Parse a comma-separated cell into a list of IDs.
    Truly empty field  -> [].
    Each ID is stripped of surrounding whitespace.
    """
    stripped = cell.strip()
    if not stripped:
        return []
    return [x.strip() for x in stripped.split(',') if x.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Per-entity list checks (7–11, 14)
# ─────────────────────────────────────────────────────────────────────────────

def _check_id_lists(
    cand_raw:  Dict[str, str],
    match_raw: Dict[str, str],
    s1_ids:    Set[str],
    s23_ids:   Set[str],
    rpt:       Report,
    max_err:   int,
) -> None:
    """
    Check every entity's candidate and match lists for violations of
    requirements 7–11 and 14. Accumulates counts and examples, then reports.
    """

    # Counters and example buffers (one per check)
    c7_n, c7_ex   = 0, []   # req 7: invalid cand IDs
    c8_n, c8_ex   = 0, []   # req 8: invalid match IDs
    c9c_n, c9c_ex = 0, []   # req 9a: S1 ID in candidate list
    c9m_n, c9m_ex = 0, []   # req 9b: S1 ID in match list
    c10c_n, c10c_ex = 0, [] # req 10a: duplicate cand IDs
    c10m_n, c10m_ex = 0, [] # req 10b: duplicate match IDs
    c11_n, c11_ex = 0, []   # req 11: match not subset of cands
    c14_n, c14_ex = 0, []   # req 14: empty field contains 'nan'/None/etc.

    for s1_id in sorted(s1_ids):   # sorted → deterministic error ordering
        raw_c = cand_raw.get(s1_id, '')
        raw_m = match_raw.get(s1_id, '')

        # Req 14: empty field must be truly empty, not 'nan' / 'none' / etc.
        for field_name, raw in (('cand', raw_c), ('match', raw_m)):
            if raw != raw.strip() or raw.lower() in ('nan', 'none', 'null', 'na'):
                c14_n += 1
                if len(c14_ex) < max_err:
                    c14_ex.append(f"{s1_id} ({field_name})={raw!r}")

        cands   = _parse_id_list(raw_c)
        matches = _parse_id_list(raw_m)
        cand_set  = set(cands)
        match_set = set(matches)

        # Req 7: every candidate ID is a valid S2/S3 ID
        invalid_c = cand_set - s23_ids
        if invalid_c:
            c7_n += len(invalid_c)
            if len(c7_ex) < max_err:
                c7_ex.append(f"{s1_id}: {sorted(invalid_c)[:3]!r}")

        # Req 8: every matched ID is a valid S2/S3 ID
        invalid_m = match_set - s23_ids
        if invalid_m:
            c8_n += len(invalid_m)
            if len(c8_ex) < max_err:
                c8_ex.append(f"{s1_id}: {sorted(invalid_m)[:3]!r}")

        # Req 9a: no S1 IDs in candidate list
        s1_in_c = {x for x in cands if x.startswith('S1-')}
        if s1_in_c:
            c9c_n += 1
            if len(c9c_ex) < max_err:
                c9c_ex.append(f"{s1_id}: {sorted(s1_in_c)[:3]!r}")

        # Req 9b: no S1 IDs in match list
        s1_in_m = {x for x in matches if x.startswith('S1-')}
        if s1_in_m:
            c9m_n += 1
            if len(c9m_ex) < max_err:
                c9m_ex.append(f"{s1_id}: {sorted(s1_in_m)[:3]!r}")

        # Req 10a: no duplicate IDs in candidate list
        if len(cands) != len(cand_set):
            c10c_n += 1
            dupes = sorted(k for k, v in Counter(cands).items() if v > 1)
            if len(c10c_ex) < max_err:
                c10c_ex.append(f"{s1_id}: {dupes[:3]!r}")

        # Req 10b: no duplicate IDs in match list
        if len(matches) != len(match_set):
            c10m_n += 1
            dupes = sorted(k for k, v in Counter(matches).items() if v > 1)
            if len(c10m_ex) < max_err:
                c10m_ex.append(f"{s1_id}: {dupes[:3]!r}")

        # Req 11: matched_entity_ids ⊆ candidate_entity_ids
        not_in_cands = match_set - cand_set
        if not_in_cands:
            c11_n += 1
            if len(c11_ex) < max_err:
                c11_ex.append(f"{s1_id}: {sorted(not_in_cands)[:3]!r}")

    # Helper: add one check result to the report
    def _r(num: str, desc: str, n: int, ex: List[str]) -> None:
        if n == 0:
            rpt.ok(f"Check {num}: {desc}")
        else:
            shown = ex[:max_err]
            rpt.fail(
                f"Check {num}: {desc}",
                f"{n:,} violation(s)\n" +
                '\n'.join(f"  {e}" for e in shown) +
                (f"\n  ... and {n - len(shown)} more" if n > len(shown) else ''))

    _r('7',   'all candidate IDs are valid test S2/S3 IDs',
       c7_n,   c7_ex)
    _r('8',   'all matched IDs are valid test S2/S3 IDs',
       c8_n,   c8_ex)
    _r('9a',  'no Source 1 IDs appear in candidate lists',
       c9c_n,  c9c_ex)
    _r('9b',  'no Source 1 IDs appear in match lists',
       c9m_n,  c9m_ex)
    _r('10a', 'no duplicate IDs within candidate lists',
       c10c_n, c10c_ex)
    _r('10b', 'no duplicate IDs within match lists',
       c10m_n, c10m_ex)
    _r('11',  'matched_entity_ids is a subset of candidate_entity_ids',
       c11_n,  c11_ex)
    _r('14',  'empty lists stored as empty field (not nan/None/null)',
       c14_n,  c14_ex)


# ─────────────────────────────────────────────────────────────────────────────
# Main validation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    cand_path:  str,
    match_path: str,
    max_errors: int,
) -> bool:
    """Run all 14 checks. Print a structured report. Return True iff all pass."""

    rpt = Report(max_errors)
    t0  = time.time()

    print(SEP)
    print("  INTERNAL OUTPUT VALIDATOR")
    print("  Amazon ML Challenge 2026 — Business Entity Resolution")
    print("  (This is NOT an official hackathon validator)")
    print(SEP)
    print(f"  candidate_pairs.tsv  : {cand_path}")
    print(f"  matching_results.tsv : {match_path}")
    print()

    # ── Checks 1–2: files exist ───────────────────────────────────────────────
    cand_exists  = os.path.exists(cand_path)
    match_exists = os.path.exists(match_path)

    if cand_exists:
        rpt.ok(f"Check 1: candidate_pairs.tsv exists "
               f"({os.path.getsize(cand_path):,} bytes)")
    else:
        rpt.fail("Check 1: candidate_pairs.tsv not found",
                 cand_path)

    if match_exists:
        rpt.ok(f"Check 2: matching_results.tsv exists "
               f"({os.path.getsize(match_path):,} bytes)")
    else:
        rpt.fail("Check 2: matching_results.tsv not found",
                 match_path)

    if not (cand_exists and match_exists):
        print(BAR)
        print("  CHECK RESULTS")
        print(BAR)
        rpt.print_checks()
        print()
        print(SEP)
        print("  RESULT: FAIL  (output file(s) missing — cannot continue)")
        print(SEP)
        return False

    # ── Load reference IDs ────────────────────────────────────────────────────
    print("  Loading reference test IDs from data_set/test/ ...", flush=True)
    try:
        s1_ids, s23_ids = _load_ref_ids()
    except FileNotFoundError as exc:
        print(f"\n  ERROR: {exc}")
        return False
    print(f"  Reference IDs loaded: S1={len(s1_ids):,}  "
          f"S23={len(s23_ids):,}")
    print()

    # ── Checks 3, 4, 12: parse both TSV files ────────────────────────────────
    result_c = _read_tsv(cand_path,  CAND_COLS,  rpt, 'candidate_pairs',  max_errors)
    result_m = _read_tsv(match_path, MATCH_COLS, rpt, 'matching_results', max_errors)

    if result_c is None or result_m is None:
        print(BAR)
        print("  CHECK RESULTS (so far)")
        print(BAR)
        rpt.print_checks()
        print()
        print(SEP)
        print("  RESULT: FAIL  (TSV parse error — cannot continue)")
        print(SEP)
        return False

    cand_raw,  cand_dupes  = result_c
    match_raw, match_dupes = result_m
    print(f"  Rows parsed: candidate_pairs={len(cand_raw):,}  "
          f"matching_results={len(match_raw):,}")
    print()

    # ── Checks 5–6, 13: row coverage and no unexpected IDs ───────────────────
    cand_s1  = set(cand_raw)
    match_s1 = set(match_raw)

    # Check 5: every test S1 in matching_results exactly once
    miss_m = s1_ids - match_s1
    xtra_m = match_s1 - s1_ids
    if not miss_m and not xtra_m and not match_dupes:
        rpt.ok(f"Check 5: all {len(s1_ids):,} test S1 entities present in "
               f"matching_results.tsv exactly once")
    else:
        if miss_m:
            rpt.fail(f"Check 5: matching_results.tsv missing "
                     f"{len(miss_m):,} test S1 entities",
                     f"first 5: {sorted(miss_m)[:5]!r}")
        if match_dupes:
            rpt.fail(f"Check 5: {len(match_dupes):,} S1 IDs appear more than "
                     f"once in matching_results.tsv",
                     f"first 5: {sorted(match_dupes)[:5]!r}")
        if not miss_m and not match_dupes:
            rpt.ok(f"Check 5: all {len(s1_ids):,} test S1 entities present "
                   f"in matching_results.tsv exactly once")

    # Check 6: every test S1 in candidate_pairs exactly once
    miss_c = s1_ids - cand_s1
    xtra_c = cand_s1 - s1_ids
    if not miss_c and not xtra_c and not cand_dupes:
        rpt.ok(f"Check 6: all {len(s1_ids):,} test S1 entities present in "
               f"candidate_pairs.tsv exactly once")
    else:
        if miss_c:
            rpt.fail(f"Check 6: candidate_pairs.tsv missing "
                     f"{len(miss_c):,} test S1 entities",
                     f"first 5: {sorted(miss_c)[:5]!r}")
        if cand_dupes:
            rpt.fail(f"Check 6: {len(cand_dupes):,} S1 IDs appear more than "
                     f"once in candidate_pairs.tsv",
                     f"first 5: {sorted(cand_dupes)[:5]!r}")
        if not miss_c and not cand_dupes:
            rpt.ok(f"Check 6: all {len(s1_ids):,} test S1 entities present "
                   f"in candidate_pairs.tsv exactly once")

    # Check 13: no unexpected S1 IDs in either file
    unexpected = xtra_c | xtra_m
    if not unexpected:
        rpt.ok("Check 13: no unexpected Source 1 IDs in either file")
    else:
        if xtra_c:
            rpt.fail(f"Check 13: candidate_pairs.tsv contains "
                     f"{len(xtra_c):,} IDs not in test_source1.tsv",
                     f"first 5: {sorted(xtra_c)[:5]!r}")
        if xtra_m:
            rpt.fail(f"Check 13: matching_results.tsv contains "
                     f"{len(xtra_m):,} IDs not in test_source1.tsv",
                     f"first 5: {sorted(xtra_m)[:5]!r}")

    # ── Checks 7–11, 14: per-entity list checks ───────────────────────────────
    # Only run on entities present in both files to avoid cascading errors.
    common = s1_ids & cand_s1 & match_s1
    skipped = len(s1_ids) - len(common)
    if skipped:
        print(f"  NOTE: {skipped:,} entities skipped for per-list checks "
              f"(not present in both files).")
    _check_id_lists(
        {k: cand_raw[k]  for k in common},
        {k: match_raw[k] for k in common},
        common, s23_ids, rpt, max_errors,
    )

    # ── Statistics ────────────────────────────────────────────────────────────
    n_empty_m  = sum(1 for v in match_raw.values() if not v.strip())
    n_single_m = sum(1 for v in match_raw.values()
                     if v.strip() and ',' not in v.strip())
    n_multi_m  = sum(1 for v in match_raw.values()
                     if v.strip() and ',' in v.strip())
    total_m    = sum(len(_parse_id_list(v)) for v in match_raw.values())
    total_c    = sum(len(_parse_id_list(v)) for v in cand_raw.values())

    elapsed = time.time() - t0

    # ── Print results ─────────────────────────────────────────────────────────
    print(BAR)
    print("  CHECK RESULTS")
    print(BAR)
    rpt.print_checks()

    print()
    print(BAR)
    print("  STATISTICS")
    print(BAR)
    print(f"  Test S1 entities             : {len(s1_ids):>12,}")
    print(f"  Test S2+S3 entities          : {len(s23_ids):>12,}")
    print(f"  Rows in candidate_pairs.tsv  : {len(cand_raw):>12,}")
    print(f"  Rows in matching_results.tsv : {len(match_raw):>12,}")
    print(f"  Total candidate IDs          : {total_c:>12,}")
    print(f"  Total matched IDs            : {total_m:>12,}")
    print(f"  Empty predictions            : {n_empty_m:>12,}  "
          f"({100*n_empty_m/max(len(match_raw),1):.2f}%)")
    print(f"  Singleton predictions        : {n_single_m:>12,}  "
          f"({100*n_single_m/max(len(match_raw),1):.2f}%)")
    print(f"  Multi-match predictions      : {n_multi_m:>12,}  "
          f"({100*n_multi_m/max(len(match_raw),1):.2f}%)")
    print(f"  Validation time              : {elapsed:>12.2f}s")

    n_fail = rpt._n_fail
    n_pass = rpt._n_pass

    print()
    print(SEP)
    if rpt.passed():
        print(f"  RESULT: PASS  ({n_pass} checks passed)")
    else:
        print(f"  RESULT: FAIL  ({n_fail} check(s) FAILED,  "
              f"{n_pass} passed)")
    print(SEP)
    print()

    return rpt.passed()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'INTERNAL output validator — Amazon ML Challenge 2026 '
            'Business Entity Resolution.\n'
            'NOT an official hackathon validator.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--cand', default=DEFAULT_CAND, metavar='PATH',
        help='Path to candidate_pairs.tsv '
             f'(default: output/candidate_pairs.tsv)')
    parser.add_argument(
        '--match', default=DEFAULT_MATCH, metavar='PATH',
        help='Path to matching_results.tsv '
             f'(default: output/matching_results.tsv)')
    parser.add_argument(
        '--max-errors', type=int, default=DEFAULT_MAX_ERR, metavar='N',
        help=f'Max per-check violations to display (default: {DEFAULT_MAX_ERR}; '
             f'0 = all)')
    return parser.parse_args()


if __name__ == '__main__':
    args   = _parse_args()
    max_e  = args.max_errors if args.max_errors > 0 else 10 ** 9
    passed = run_validation(args.cand, args.match, max_e)
    sys.exit(0 if passed else 1)

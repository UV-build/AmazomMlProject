"""
Amazon ML Challenge 2026 - Business Entity Resolution
eda.py

Exploratory Data Analysis over training data only.
Covers:
  1.  Record counts by source
  2.  Match-count distribution (per S1 entity)
  3.  Zero-match / singleton / multi-match entity counts
  4.  Business-name length statistics
  5.  Business-address length statistics
  6.  Token counts (name + address)
  7.  Country distributions (train + test)
  8.  Noisy true-match examples
  9.  Punctuation differences
  10. Legal-suffix differences (Corp/Ltd/Pvt/Inc/LLC/LLP/&/and …)
  11. Word-order changes
  12. Address abbreviation / component reordering / missing postal info

No external data, no ML.
"""

from __future__ import annotations

import os
import re
import sys
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
_SRC  = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.normpath(os.path.join(_SRC, '..', '..', '..', 'data_set'))
_TRAIN = os.path.join(_DATA, 'train')
_TEST  = os.path.join(_DATA, 'test')

DIVIDER = '=' * 74
SECTION = '-' * 74

random.seed(42)

# ── helpers ────────────────────────────────────────────────────────────────────

def load(split: str, num: int) -> pd.DataFrame:
    path = os.path.join(_TRAIN if split == 'train' else _TEST,
                        f'{split}_source{num}.tsv')
    return pd.read_csv(path, sep='\t', encoding='utf-8',
                       dtype=str, keep_default_na=False, na_values=[''])

def load_gt() -> pd.DataFrame:
    path = os.path.join(_TRAIN, 'train_ground_truth.tsv')
    return pd.read_csv(path, sep='\t', encoding='utf-8',
                       dtype=str, keep_default_na=False, na_values=[''])

def tok(text: str) -> List[str]:
    """Simple whitespace tokeniser, lowercased."""
    return str(text).lower().split() if pd.notna(text) else []

def safe_len(x) -> int:
    return len(str(x)) if pd.notna(x) else 0

def describe_series(s: pd.Series, label: str) -> None:
    s = s.dropna()
    print(f"  {label}:")
    print(f"    count={len(s):,}  mean={s.mean():.1f}  median={s.median():.0f}  "
          f"std={s.std():.1f}  min={s.min():.0f}  max={s.max():.0f}")
    pcts = [5, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(s, pcts)
    print("    percentiles: " + "  ".join(f"p{p}={v:.0f}" for p, v in zip(pcts, vals)))

def header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def sub(title: str) -> None:
    print(f"\n  {SECTION[:50]}")
    print(f"  {title}")

# ── legal suffix regex ─────────────────────────────────────────────────────────

LEGAL_TOKENS = [
    r'\bInc\.?\b', r'\bIncorporated\b',
    r'\bLLC\b', r'\bL\.L\.C\.?\b',
    r'\bLLP\b', r'\bL\.L\.P\.?\b',
    r'\bLtd\.?\b', r'\bLimited\b',
    r'\bCorp\.?\b', r'\bCorporation\b',
    r'\bPvt\.?\b', r'\bPrivate\b',
    r'\bCo\.?\b', r'\bCompany\b',
    r'\bPlc\.?\b',
    r'\bGmbH\b', r'\bS\.A\.S\.?\b', r'\bS\.A\.?\b', r'\bSARL\b',
    r'\bPty\.?\b',
]
LEGAL_RE = re.compile('|'.join(LEGAL_TOKENS), re.IGNORECASE)

def extract_legal(name: str) -> List[str]:
    return LEGAL_RE.findall(str(name)) if pd.notna(name) else []

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(DIVIDER)
    print("  AMAZON ML CHALLENGE 2026 — EXPLORATORY DATA ANALYSIS")
    print(DIVIDER)

    # ── load ──────────────────────────────────────────────────────────────────
    print("\nLoading datasets …", flush=True)
    ts1 = load('train', 1);  print("  train_source1 loaded", flush=True)
    ts2 = load('train', 2);  print("  train_source2 loaded", flush=True)
    ts3 = load('train', 3);  print("  train_source3 loaded", flush=True)
    gt  = load_gt();         print("  ground truth  loaded", flush=True)
    es1 = load('test',  1);  print("  test_source1  loaded", flush=True)
    es2 = load('test',  2);  print("  test_source2  loaded", flush=True)
    es3 = load('test',  3);  print("  test_source3  loaded", flush=True)

    # build lookup maps
    s1_map: Dict[str, dict] = ts1.set_index('entity_id').to_dict('index')
    s2_map: Dict[str, dict] = ts2.set_index('entity_id').to_dict('index')
    s3_map: Dict[str, dict] = ts3.set_index('entity_id').to_dict('index')

    # parse ground truth
    s1_to_matches: Dict[str, List[str]] = {}
    for _, row in gt.iterrows():
        s1id = str(row['source1_entity_id']).strip()
        raw  = row['matched_entity_ids']
        if pd.isna(raw) or str(raw).strip() == '':
            s1_to_matches[s1id] = []
        else:
            s1_to_matches[s1id] = [x.strip() for x in str(raw).split(',') if x.strip()]

    # ── 1. Record counts ──────────────────────────────────────────────────────
    header("1. RECORD COUNTS BY SOURCE")
    for label, df in [('train_source1', ts1), ('train_source2', ts2),
                      ('train_source3', ts3), ('test_source1',  es1),
                      ('test_source2',  es2), ('test_source3',  es3)]:
        print(f"  {label:<22}: {len(df):>10,} rows")
    print(f"\n  Total train records : {len(ts1)+len(ts2)+len(ts3):>10,}")
    print(f"  Total test  records : {len(es1)+len(es2)+len(es3):>10,}")

    # ── 2 + 3. Match-count distribution ───────────────────────────────────────
    header("2-3. MATCH-COUNT DISTRIBUTION PER S1 ENTITY")
    match_counts = [len(v) for v in s1_to_matches.values()]
    dist = Counter(match_counts)

    zero      = dist.get(0, 0)
    singleton = dist.get(1, 0)
    multi     = sum(cnt for k, cnt in dist.items() if k >= 2)
    total_s1  = len(match_counts)

    print(f"  Total S1 entities   : {total_s1:>10,}")
    print(f"  Zero-match          : {zero:>10,}  ({100*zero/total_s1:.2f}%)")
    print(f"  Singleton (1 match) : {singleton:>10,}  ({100*singleton/total_s1:.2f}%)")
    print(f"  Multi-match (≥2)    : {multi:>10,}  ({100*multi/total_s1:.2f}%)")
    print(f"\n  Full distribution (match_count → #S1_entities):")
    for k in sorted(dist):
        bar = '█' * min(k * 3, 40) if k > 0 else ''
        print(f"    {k:>3} match(es) : {dist[k]:>8,}  {bar}")

    mc_arr = np.array(match_counts)
    print(f"\n  mean={mc_arr.mean():.2f}  median={np.median(mc_arr):.0f}  "
          f"std={mc_arr.std():.2f}  max={mc_arr.max()}")

    # ── 4. Business-name length stats ─────────────────────────────────────────
    header("4. BUSINESS-NAME LENGTH (characters)")
    for label, df in [('train_source1', ts1), ('train_source2', ts2), ('train_source3', ts3)]:
        lengths = df['business_name'].dropna().apply(safe_len).astype(float)
        describe_series(lengths, label)

    # ── 5. Business-address length stats ──────────────────────────────────────
    header("5. BUSINESS-ADDRESS LENGTH (characters, non-null rows only)")
    for label, df in [('train_source1', ts1), ('train_source2', ts2), ('train_source3', ts3)]:
        lengths = df['business_address'].dropna().apply(safe_len).astype(float)
        lengths = lengths[lengths > 0]
        describe_series(lengths, label)

    # ── 6. Token counts ───────────────────────────────────────────────────────
    header("6. TOKEN COUNTS")
    for label, df in [('train_source1', ts1), ('train_source2', ts2), ('train_source3', ts3)]:
        name_toks = df['business_name'].apply(lambda x: len(tok(x))).astype(float)
        addr_toks = df['business_address'].dropna().apply(lambda x: len(tok(x))).astype(float)
        addr_toks = addr_toks[addr_toks > 0]
        print(f"\n  {label}:")
        print(f"    name tokens  — mean={name_toks.mean():.1f}  median={name_toks.median():.0f}  "
              f"max={name_toks.max():.0f}")
        print(f"    addr tokens  — mean={addr_toks.mean():.1f}  median={addr_toks.median():.0f}  "
              f"max={addr_toks.max():.0f}")

    # ── 7. Country distributions ──────────────────────────────────────────────
    header("7. COUNTRY DISTRIBUTIONS")
    for label, df in [('train_source1', ts1), ('train_source2', ts2), ('train_source3', ts3),
                      ('test_source1',  es1), ('test_source2',  es2), ('test_source3',  es3)]:
        dist_c = df['country'].value_counts()
        total  = len(df)
        parts  = ', '.join(f"{c}: {n:,} ({100*n/total:.1f}%)" for c, n in dist_c.items())
        print(f"  {label:<22}: {parts}")

    # ── 8. Noisy true-match examples ──────────────────────────────────────────
    header("8. NOISY TRUE-MATCH EXAMPLES (sampled matched pairs)")

    # collect all matched pairs (S1, S2/S3)
    all_pairs: List[Tuple[str, str]] = []
    for s1id, mids in s1_to_matches.items():
        for mid in mids:
            all_pairs.append((s1id, mid))

    sample_pairs = random.sample(all_pairs, min(200_000, len(all_pairs)))

    shown = 0
    max_show = 30
    print(f"\n  (Showing up to {max_show} illustrative matched pairs)\n")
    for s1id, mid in sample_pairs:
        if shown >= max_show:
            break
        s1rec = s1_map.get(s1id, {})
        if mid.startswith('S2-'):
            rec2 = s2_map.get(mid, {})
        else:
            rec2 = s3_map.get(mid, {})
        n1 = str(s1rec.get('business_name', '')).strip()
        n2 = str(rec2.get('business_name', '')).strip()
        a1 = str(s1rec.get('business_address', '')).strip()
        a2 = str(rec2.get('business_address', '')).strip()
        if n1 and n2 and n1.lower() != n2.lower():
            print(f"  S1  name : {n1}")
            print(f"  {mid[:2]}  name : {n2}")
            print(f"  S1  addr : {a1[:80]}")
            print(f"  {mid[:2]}  addr : {a2[:80]}")
            print()
            shown += 1

    # ── 9-12. Noise-pattern deep-dives ────────────────────────────────────────
    header("9-12. NOISE PATTERN ANALYSIS (from matched pairs)")

    # Pre-collect all matched pairs with their name strings
    print("\n  Collecting all matched pair names …", flush=True)
    pair_names: List[Tuple[str, str, str, str, str, str]] = []
    # (s1id, mid, s1_name, s2s3_name, s1_addr, s2s3_addr)
    for s1id, mids in s1_to_matches.items():
        s1rec = s1_map.get(s1id, {})
        n1 = str(s1rec.get('business_name', '')).strip()
        a1 = str(s1rec.get('business_address', '')).strip()
        for mid in mids:
            rec2 = (s2_map if mid.startswith('S2-') else s3_map).get(mid, {})
            n2 = str(rec2.get('business_name', '')).strip()
            a2 = str(rec2.get('business_address', '')).strip()
            pair_names.append((s1id, mid, n1, n2, a1, a2))

    print(f"  Total matched pairs: {len(pair_names):,}\n")

    # ── 9a. Punctuation differences ───────────────────────────────────────────
    sub("9a. PUNCTUATION DIFFERENCES")
    PUNCT_RE = re.compile(r"[.\-',&/()!]")

    punct_diff_examples: List[Tuple[str,str]] = []
    ampersand_examples:  List[Tuple[str,str]] = []

    for _, _, n1, n2, _, _ in pair_names:
        if not n1 or not n2:
            continue
        p1 = set(PUNCT_RE.findall(n1))
        p2 = set(PUNCT_RE.findall(n2))
        if p1 != p2:
            punct_diff_examples.append((n1, n2))
        # & vs and
        has_amp1 = '&' in n1
        has_and1 = bool(re.search(r'\band\b', n1, re.I))
        has_amp2 = '&' in n2
        has_and2 = bool(re.search(r'\band\b', n2, re.I))
        if (has_amp1 and has_and2) or (has_and1 and has_amp2):
            ampersand_examples.append((n1, n2))

    print(f"  Pairs with differing punctuation sets : {len(punct_diff_examples):,}")
    print(f"  '&' ↔ 'and' substitutions             : {len(ampersand_examples):,}")
    print("\n  Punctuation-diff examples (up to 10):")
    for n1, n2 in punct_diff_examples[:10]:
        print(f"    S1: {n1}")
        print(f"    S?: {n2}")
        print()
    print("\n  '&' ↔ 'and' examples (up to 5):")
    for n1, n2 in ampersand_examples[:5]:
        print(f"    S1: {n1}")
        print(f"    S?: {n2}")
        print()

    # ── 9b. Legal-suffix differences ──────────────────────────────────────────
    sub("9b. LEGAL-SUFFIX DIFFERENCES")

    # Normalise a name to its suffix tokens and root
    def norm_suffix(name: str):
        suffixes = [s.lower().rstrip('.') for s in extract_legal(name)]
        root = LEGAL_RE.sub('', name).strip().lower()
        return frozenset(suffixes), root

    suffix_diff_examples: List[Tuple[str,str]] = []
    suffix_pair_counter  = Counter()

    for _, _, n1, n2, _, _ in pair_names:
        if not n1 or not n2:
            continue
        suf1, r1 = norm_suffix(n1)
        suf2, r2 = norm_suffix(n2)
        if suf1 != suf2:
            suffix_diff_examples.append((n1, n2))
            key = (tuple(sorted(suf1)), tuple(sorted(suf2)))
            suffix_pair_counter[key] += 1

    print(f"  Pairs with differing legal suffixes: {len(suffix_diff_examples):,}")
    print("\n  Most common suffix substitutions (original → matched):")
    for (suf1, suf2), cnt in suffix_pair_counter.most_common(20):
        s1str = ', '.join(suf1) if suf1 else '(none)'
        s2str = ', '.join(suf2) if suf2 else '(none)'
        print(f"    [{s1str}]  ↔  [{s2str}]  :  {cnt:,}")
    print("\n  Legal-suffix diff examples (up to 10):")
    for n1, n2 in suffix_diff_examples[:10]:
        print(f"    S1: {n1}")
        print(f"    S?: {n2}")
        print()

    # ── 10. Word-order changes ─────────────────────────────────────────────────
    sub("10. WORD-ORDER CHANGES (same token set, different order)")

    word_order_examples: List[Tuple[str,str]] = []
    for _, _, n1, n2, _, _ in pair_names:
        if not n1 or not n2:
            continue
        t1 = sorted(tok(n1))
        t2 = sorted(tok(n2))
        if t1 == t2 and n1.lower() != n2.lower():
            word_order_examples.append((n1, n2))

    print(f"  Pairs with identical token bags but different order: {len(word_order_examples):,}")
    print("\n  Examples (up to 10):")
    for n1, n2 in word_order_examples[:10]:
        print(f"    S1: {n1}")
        print(f"    S?: {n2}")
        print()

    # ── 11. Typos (edit-distance-1 names, same length ±1) ─────────────────────
    sub("11. LIKELY TYPOS (short edit distance after lowercasing)")

    # Use a cheap char-level diff: same token count, Jaccard on chars >= 0.85
    # (avoids importing rapidfuzz in the EDA script itself)
    def char_jaccard(a: str, b: str) -> float:
        sa, sb = set(a.lower()), set(b.lower())
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    def likely_typo(n1: str, n2: str) -> bool:
        # Same character bag but ~1 difference, similar length
        if abs(len(n1) - len(n2)) > 3:
            return False
        if len(n1) < 5:
            return False
        cj = char_jaccard(n1, n2)
        return 0.80 <= cj < 1.0 and n1.lower() != n2.lower()

    typo_examples: List[Tuple[str,str]] = []
    for _, _, n1, n2, _, _ in pair_names:
        if not n1 or not n2:
            continue
        # strip legal suffixes first so "Inc" vs "" doesn't dominate
        r1 = LEGAL_RE.sub('', n1).strip()
        r2 = LEGAL_RE.sub('', n2).strip()
        if likely_typo(r1, r2):
            typo_examples.append((n1, n2))

    print(f"  Likely-typo pairs (char-Jaccard 0.80-1.0, len diff ≤3): {len(typo_examples):,}")
    print("\n  Examples (up to 15):")
    for n1, n2 in typo_examples[:15]:
        print(f"    S1: {n1}")
        print(f"    S?: {n2}")
        print()

    # ── 12. Address patterns ──────────────────────────────────────────────────
    sub("12. ADDRESS PATTERNS")

    # 12a. Pairs where address is empty on one side
    addr_one_missing = [(n1,n2,a1,a2) for _,_,n1,n2,a1,a2 in pair_names
                        if bool(a1) != bool(a2)]
    print(f"  Pairs where address present on one side only : {len(addr_one_missing):,}")
    print("  Examples (up to 5):")
    for n1, n2, a1, a2 in addr_one_missing[:5]:
        print(f"    Name S1: {n1}    Addr: {a1[:60] or '(empty)'}")
        print(f"    Name S?: {n2}    Addr: {a2[:60] or '(empty)'}")
        print()

    # 12b. Common US state abbreviations in S1 vs full name in S2/S3
    US_ABBR = {
        'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
        'IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV',
        'NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN',
        'TX','UT','VT','VA','WA','WV','WI','WY','DC',
    }

    abbr_expand_examples: List[Tuple[str,str,str,str]] = []
    for _, _, n1, n2, a1, a2 in pair_names:
        if not a1 or not a2:
            continue
        toks1 = set(a1.upper().split())
        toks2 = set(a2.upper().split())
        if toks1 & US_ABBR and not toks2 & US_ABBR:
            abbr_expand_examples.append((n1, a1, n2, a2))
        elif toks2 & US_ABBR and not toks1 & US_ABBR:
            abbr_expand_examples.append((n1, a1, n2, a2))

    print(f"\n  Pairs with US-state abbreviation vs full state name : {len(abbr_expand_examples):,}")
    print("  Examples (up to 5):")
    for n1, a1, n2, a2 in abbr_expand_examples[:5]:
        print(f"    S1: {n1} | {a1[:70]}")
        print(f"    S?: {n2} | {a2[:70]}")
        print()

    # 12c. Reordered address components (same tokens, different order)
    addr_reorder_examples: List[Tuple[str,str,str,str]] = []
    for _, _, n1, n2, a1, a2 in pair_names:
        if not a1 or not a2:
            continue
        t1 = sorted(tok(a1))
        t2 = sorted(tok(a2))
        if t1 == t2 and a1.lower() != a2.lower():
            addr_reorder_examples.append((n1, a1, n2, a2))

    print(f"\n  Pairs with identical address tokens but reordered : {len(addr_reorder_examples):,}")
    print("  Examples (up to 5):")
    for n1, a1, n2, a2 in addr_reorder_examples[:5]:
        print(f"    S1: {n1} | {a1[:70]}")
        print(f"    S?: {n2} | {a2[:70]}")
        print()

    # 12d. Missing PIN / postal code
    # Look for pairs where one address has digits resembling a postal code and the other doesn't
    PIN_RE = re.compile(r'\b\d{5,6}\b')  # US ZIP (5) or India PIN (6)
    pin_missing_examples: List[Tuple[str,str,str,str]] = []
    for _, _, n1, n2, a1, a2 in pair_names:
        if not a1 or not a2:
            continue
        has_pin1 = bool(PIN_RE.search(a1))
        has_pin2 = bool(PIN_RE.search(a2))
        if has_pin1 != has_pin2:
            pin_missing_examples.append((n1, a1, n2, a2))

    print(f"\n  Pairs where PIN/ZIP present on one side only : {len(pin_missing_examples):,}")
    print("  Examples (up to 5):")
    for n1, a1, n2, a2 in pin_missing_examples[:5]:
        print(f"    S1: {n1} | {a1[:70]}")
        print(f"    S?: {n2} | {a2[:70]}")
        print()

    # 12e. Common address abbreviations
    ADDR_ABBR_MAP = {
        r'\bSt\.?\b':   'Street',
        r'\bAve\.?\b':  'Avenue',
        r'\bBlvd\.?\b': 'Boulevard',
        r'\bDr\.?\b':   'Drive',
        r'\bRd\.?\b':   'Road',
        r'\bLn\.?\b':   'Lane',
        r'\bCt\.?\b':   'Court',
        r'\bPl\.?\b':   'Place',
        r'\bPkwy\.?\b': 'Parkway',
        r'\bHwy\.?\b':  'Highway',
        r'\bFt\.?\b':   'Fort',
        r'\bMt\.?\b':   'Mount',
        r'\bN\.?\b':    'North',
        r'\bS\.?\b':    'South',
        r'\bE\.?\b':    'East',
        r'\bW\.?\b':    'West',
        r'\bNE\.?\b':   'Northeast', r'\bNW\.?\b': 'Northwest',
        r'\bSE\.?\b':   'Southeast', r'\bSW\.?\b': 'Southwest',
    }
    addr_abbr_counter = Counter()
    addr_abbr_examples: List[Tuple[str,str,str,str]] = []
    for _, _, n1, n2, a1, a2 in pair_names:
        if not a1 or not a2:
            continue
        found_any = False
        for abbr_re, full in ADDR_ABBR_MAP.items():
            has_abbr1 = bool(re.search(abbr_re, a1, re.I))
            has_full1 = bool(re.search(r'\b' + full + r'\b', a1, re.I))
            has_abbr2 = bool(re.search(abbr_re, a2, re.I))
            has_full2 = bool(re.search(r'\b' + full + r'\b', a2, re.I))
            if (has_abbr1 and has_full2) or (has_full1 and has_abbr2):
                addr_abbr_counter[full] += 1
                found_any = True
        if found_any and len(addr_abbr_examples) < 8:
            addr_abbr_examples.append((n1, a1, n2, a2))

    print(f"\n  Address abbreviation ↔ full-word substitutions found:")
    for word, cnt in addr_abbr_counter.most_common(15):
        print(f"    {word:<12}: {cnt:,}")
    print("\n  Examples (up to 8):")
    for n1, a1, n2, a2 in addr_abbr_examples:
        print(f"    S1: {n1} | {a1[:70]}")
        print(f"    S?: {n2} | {a2[:70]}")
        print()

    # ── Summary counts ────────────────────────────────────────────────────────
    header("NOISE PATTERN SUMMARY")
    print(f"  Total matched pairs analysed            : {len(pair_names):,}")
    print(f"  Punctuation-set differences             : {len(punct_diff_examples):,}")
    print(f"  '&' ↔ 'and' substitutions               : {len(ampersand_examples):,}")
    print(f"  Legal-suffix differences                : {len(suffix_diff_examples):,}")
    print(f"  Word-order changes (same token bag)     : {len(word_order_examples):,}")
    print(f"  Likely typos                            : {len(typo_examples):,}")
    print(f"  Address: one side missing               : {len(addr_one_missing):,}")
    print(f"  Address: US-state abbr vs full          : {len(abbr_expand_examples):,}")
    print(f"  Address: component reorder              : {len(addr_reorder_examples):,}")
    print(f"  Address: PIN/ZIP on one side only       : {len(pin_missing_examples):,}")
    total_addr_abbr = sum(addr_abbr_counter.values())
    print(f"  Address: street-type abbr substitutions : {total_addr_abbr:,}")

    print(f"\n{DIVIDER}")
    print("  EDA COMPLETE")
    print(DIVIDER)


if __name__ == '__main__':
    main()

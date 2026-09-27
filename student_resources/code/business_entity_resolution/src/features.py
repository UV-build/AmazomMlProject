"""
Amazon ML Challenge 2026 - Business Entity Resolution
features.py

For each blocked (S1, S23) candidate pair, generate a fixed-width numerical
feature vector.

Feature groups
--------------
NAME (7 features):
  name_lev_ratio          Levenshtein normalised similarity on core names
  name_jaro_winkler       Jaro-Winkler on core names
  name_token_set          rapidfuzz token-set ratio / 100
  name_token_sort         rapidfuzz token-sort ratio / 100
  name_jaccard            Jaccard on core-name token sets
  name_tfidf_cos          Character-trigram TF-IDF cosine on core names
  name_phonetic_eq        1.0 if Metaphone keys are equal, else 0.0

ADDRESS (9 features):
  addr_lev_ratio          Levenshtein on cleaned addresses
  addr_jaro_winkler       Jaro-Winkler on cleaned addresses
  addr_token_set          token-set ratio / 100
  addr_token_sort         token-sort ratio / 100
  addr_jaccard            Jaccard on address token sets
  addr_tfidf_cos          Char-trigram TF-IDF cosine on cleaned addresses
  addr_postal_eq          1 both have postal codes and they match, 0 mismatch,
                          -1 (missing) if either is absent
  addr_state_eq           1 match / 0 mismatch / -1 missing (same logic)
  addr_city_eq            1 match / 0 mismatch / -1 missing

OTHER (8 features):
  country_eq              1.0 if same country, 0.0 if different, 0.5 if either blank
  name_len_diff           |len(s1_core) - len(s23_core)| / max(lens, 1)  [0,1]
  addr_len_diff           |len(s1_addr_clean) - len(s23_addr_clean)| / max(lens, 1)
  digit_pattern_sim       Jaccard on digit-run token sets extracted from addresses
  s1_has_legal_suffix     1 if S1 raw name contains a legal suffix
  s23_has_legal_suffix    1 if S23 raw name contains a legal suffix
  s1_addr_missing         1 if S1 address is empty
  s23_addr_missing        1 if S23 address is empty

Total: 24 features

Design principles
-----------------
- Deterministic: same inputs always produce the same outputs.
- No leakage: uses only information present in the record pair itself.
- No external data / APIs / internet.
- Country remains an open string; France and unseen countries handled
  identically to US and India.
- Missing fields produce sentinel value -1.0 (not NaN) for tree models.
- All string comparisons use pre-normalised forms from normalize.py.
- TF-IDF is computed on-the-fly per pair using character trigrams and IDF
  weights estimated from the corpus passed at FeatureBuilder construction
  time (no sklearn / C-extension DLLs needed).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import jellyfish
import rapidfuzz.distance as rfd
import rapidfuzz.fuzz      as rff

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalize import (
    normalize_name,
    normalize_address,
    strip_legal_suffixes,
    phonetic_name,
    extract_postal_code,
    extract_state_region,
    extract_city,
    _safe_str,
    _LEGAL_STRIP_RE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature names (canonical order)
# ─────────────────────────────────────────────────────────────────────────────

NAME_FEATURES = [
    'name_lev_ratio',
    'name_jaro_winkler',
    'name_token_set',
    'name_token_sort',
    'name_jaccard',
    'name_tfidf_cos',
    'name_phonetic_eq',
]

ADDR_FEATURES = [
    'addr_lev_ratio',
    'addr_jaro_winkler',
    'addr_token_set',
    'addr_token_sort',
    'addr_jaccard',
    'addr_tfidf_cos',
    'addr_postal_eq',
    'addr_state_eq',
    'addr_city_eq',
]

OTHER_FEATURES = [
    'country_eq',
    'name_len_diff',
    'addr_len_diff',
    'digit_pattern_sim',
    's1_has_legal_suffix',
    's23_has_legal_suffix',
    's1_addr_missing',
    's23_addr_missing',
]

ALL_FEATURES: List[str] = NAME_FEATURES + ADDR_FEATURES + OTHER_FEATURES
N_FEATURES = len(ALL_FEATURES)          # 24
FEAT_IDX   = {name: i for i, name in enumerate(ALL_FEATURES)}

MISSING_VAL = -1.0   # sentinel for missing/undefined features


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_DIGIT_RUN_RE = re.compile(r'\d+')
_LEGAL_PRESENCE_RE = re.compile(
    r'\b(?:private\s+limited|pvt\.?\s+ltd\.?|pvtltd|llc|l\.l\.c|llp|'
    r'incorporated|inc\.?|corporation|corp\.?|limited|ltd\.?|'
    r'private|pvt\.?|pty\.?|plc\.?|company|co\.?|gmbh|sarl|sas|sa)\b',
    re.IGNORECASE,
)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def _jaccard(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union else 0.0


def _char_trigrams(text: str) -> List[str]:
    """Character 3-grams with space-padding."""
    padded = f' {text} '
    return [padded[i:i+3] for i in range(len(padded) - 2)]


def _tfidf_cosine(a: str, b: str, idf: Dict[str, float]) -> float:
    """
    Character-trigram TF-IDF cosine similarity between two strings.
    IDF values are pre-computed over the corpus (passed via FeatureBuilder).
    Falls back to raw Jaccard on trigram sets when IDF is empty.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    def _tfidf_vec(text: str) -> Dict[str, float]:
        tf: Counter = Counter(_char_trigrams(text))
        vec = {}
        for ng, cnt in tf.items():
            tfidf = (1.0 + math.log(cnt)) * idf.get(ng, 1.0)
            vec[ng] = tfidf
        # L2 normalise
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {k: v / norm for k, v in vec.items()}
        return vec

    va = _tfidf_vec(a)
    vb = _tfidf_vec(b)
    common = set(va) & set(vb)
    return sum(va[k] * vb[k] for k in common)


def _eq_with_missing(v1: Optional[str], v2: Optional[str]) -> float:
    """
    1.0 if both present and equal (case-insensitive),
    0.0 if both present and different,
    -1.0 (MISSING_VAL) if either is absent.
    """
    if not v1 or not v2:
        return MISSING_VAL
    return 1.0 if v1.lower() == v2.lower() else 0.0


def _country_eq(c1: str, c2: str) -> float:
    """1.0 same, 0.0 different, 0.5 if either blank."""
    if not c1 or not c2:
        return 0.5
    return 1.0 if c1.lower() == c2.lower() else 0.0


def _has_legal_suffix(raw: str) -> float:
    return 1.0 if _LEGAL_PRESENCE_RE.search(raw) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Record cache  (pre-computed normalised representations)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _RecordCache:
    """All normalised fields for one record, computed once."""
    # identifiers
    entity_id:      str
    country:        str

    # name fields
    raw_name:       str
    core_name:      str      # normalize_name + strip_legal_suffixes
    phonetic:       str
    name_tokens:    frozenset
    has_legal:      float    # 1/0

    # address fields
    raw_addr:       str
    clean_addr:     str      # normalize_address(raw, country)
    addr_tokens:    frozenset
    postal_code:    Optional[str]
    state_region:   Optional[str]
    city:           Optional[str]
    addr_missing:   float    # 1 if empty


def _build_cache(
    df: pd.DataFrame,
) -> Dict[str, _RecordCache]:
    """
    Build a {entity_id -> _RecordCache} dict from a source DataFrame.
    Calls normalize_name + strip_legal_suffixes once per record.
    All other fields derived cheaply from that output.
    """
    cache: Dict[str, _RecordCache] = {}

    eids    = df['entity_id'].fillna('').astype(str).tolist()
    rnames  = df['business_name'].fillna('').astype(str).tolist()
    raddrs  = df['business_address'].fillna('').astype(str).tolist()
    ctries  = df['country'].fillna('').astype(str).tolist()

    for eid, rname, raddr, country in zip(eids, rnames, raddrs, ctries):
        rname   = _safe_str(rname)
        raddr   = _safe_str(raddr)
        country = _safe_str(country)

        # name
        canon    = normalize_name(rname)
        core     = strip_legal_suffixes(canon)
        ph_key   = phonetic_name(rname)
        ntoks    = frozenset(t for t in core.split() if t)
        has_leg  = _has_legal_suffix(rname)

        # address
        clean_a  = normalize_address(raddr, country)
        atoks    = frozenset(t for t in clean_a.split() if t)
        postal   = extract_postal_code(raddr, country) if raddr else None
        state    = extract_state_region(raddr, country) if raddr else None
        city     = extract_city(raddr, country) if raddr else None
        addr_mis = 1.0 if not raddr else 0.0

        cache[eid] = _RecordCache(
            entity_id=eid, country=country,
            raw_name=rname, core_name=core,
            phonetic=ph_key, name_tokens=ntoks, has_legal=has_leg,
            raw_addr=raddr, clean_addr=clean_a,
            addr_tokens=atoks, postal_code=postal,
            state_region=state, city=city, addr_missing=addr_mis,
        )

    return cache


# ─────────────────────────────────────────────────────────────────────────────
# IDF estimator
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_idf(
    texts: List[str],
    max_df_frac: float = 0.90,
    min_df: int = 2,
) -> Dict[str, float]:
    """
    Estimate smooth IDF weights for character trigrams over a list of strings.
    Uses sklearn-style smooth IDF:  log((1+N)/(1+df)) + 1

    Parameters
    ----------
    texts       : list of strings (core names or cleaned addresses)
    max_df_frac : trigrams appearing in > this fraction of docs are ignored
    min_df      : trigrams appearing in < min_df docs are treated as rare
                  (use a floor IDF rather than excluding them)
    """
    N = len(texts)
    if N == 0:
        return {}

    df_count: Counter = Counter()
    for t in texts:
        unique_tg = set(_char_trigrams(t))
        for tg in unique_tg:
            df_count[tg] += 1

    max_df_abs = int(max_df_frac * N)
    idf: Dict[str, float] = {}
    for tg, df in df_count.items():
        if df > max_df_abs:
            continue          # too common — skip
        idf[tg] = math.log((1.0 + N) / (1.0 + max(df, min_df))) + 1.0

    return idf


# ─────────────────────────────────────────────────────────────────────────────
# Main feature builder
# ─────────────────────────────────────────────────────────────────────────────

class FeatureBuilder:
    """
    Pre-computes normalised representations and IDF weights, then generates
    the 24-dimensional feature vector for each (S1, S23) candidate pair.

    Usage
    -----
        builder = FeatureBuilder(s1_df, s2_df, s3_df)
        X, meta = builder.build_pair_features(candidates)
        # X is (n_pairs, 24) numpy float32 array
        # meta is a DataFrame with columns [s1_id, s23_id]
    """

    def __init__(
        self,
        s1_df:  pd.DataFrame,
        s2_df:  pd.DataFrame,
        s3_df:  pd.DataFrame,
        max_df_frac: float = 0.90,
        verbose: bool = True,
    ) -> None:
        import time
        t0 = time.time()
        if verbose:
            print("  FeatureBuilder: building record caches...", flush=True)

        self._s1_cache  = _build_cache(s1_df)
        s23_df          = pd.concat([s2_df, s3_df], ignore_index=True)
        self._s23_cache = _build_cache(s23_df)

        if verbose:
            print(f"    S1={len(self._s1_cache):,}  "
                  f"S23={len(self._s23_cache):,}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

        # Estimate IDF over the S23 corpus (avoids leakage: S1 not used)
        if verbose:
            print("  FeatureBuilder: estimating IDF weights...", flush=True)
        t0 = time.time()
        s23_names  = [c.core_name  for c in self._s23_cache.values()]
        s23_addrs  = [c.clean_addr for c in self._s23_cache.values()]
        self._name_idf = _estimate_idf(s23_names, max_df_frac)
        self._addr_idf = _estimate_idf(s23_addrs, max_df_frac)
        if verbose:
            print(f"    name IDF vocab={len(self._name_idf):,}  "
                  f"addr IDF vocab={len(self._addr_idf):,}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # Factory: construct from pre-built S23 cache (avoids rebuilding it per S1 batch)
    # ------------------------------------------------------------------

    @classmethod
    def from_s23_cache(
        cls,
        s1_df:       pd.DataFrame,
        s23_cache:   dict,         # pre-built {entity_id -> _RecordCache}
        name_idf:    dict,         # pre-estimated name IDF
        addr_idf:    dict,         # pre-estimated addr IDF
    ) -> 'FeatureBuilder':
        """
        Construct a FeatureBuilder using a pre-built S23 cache and IDF weights.
        Only the S1 cache is built from scratch.  S23 build cost is amortised
        across all S1 batches for the same country.
        """
        obj = object.__new__(cls)
        obj._s1_cache  = _build_cache(s1_df)
        obj._s23_cache = s23_cache
        obj._name_idf  = name_idf
        obj._addr_idf  = addr_idf
        return obj

    # ------------------------------------------------------------------
    # Core pair-feature computation
    # ------------------------------------------------------------------

    def _pair_features(
        self,
        r1: _RecordCache,
        r2: _RecordCache,
    ) -> List[float]:
        """
        Compute the 24-feature vector for one (S1, S23) pair.
        Returns a list of floats in ALL_FEATURES order.
        """
        fv = [0.0] * N_FEATURES

        n1 = r1.core_name
        n2 = r2.core_name

        # ── NAME FEATURES ───────────────────────────────────────────────

        # 1. Levenshtein ratio
        fv[FEAT_IDX['name_lev_ratio']] = (
            rfd.Levenshtein.normalized_similarity(n1, n2)
            if n1 and n2 else (1.0 if not n1 and not n2 else 0.0)
        )

        # 2. Jaro-Winkler
        fv[FEAT_IDX['name_jaro_winkler']] = (
            jellyfish.jaro_winkler_similarity(n1, n2)
            if n1 and n2 else (1.0 if not n1 and not n2 else 0.0)
        )

        # 3. Token-set ratio
        fv[FEAT_IDX['name_token_set']] = (
            rff.token_set_ratio(n1, n2) / 100.0
            if n1 and n2 else (1.0 if not n1 and not n2 else 0.0)
        )

        # 4. Token-sort ratio
        fv[FEAT_IDX['name_token_sort']] = (
            rff.token_sort_ratio(n1, n2) / 100.0
            if n1 and n2 else (1.0 if not n1 and not n2 else 0.0)
        )

        # 5. Jaccard on name token sets
        fv[FEAT_IDX['name_jaccard']] = _jaccard(
            set(r1.name_tokens), set(r2.name_tokens)
        )

        # 6. Character TF-IDF cosine
        fv[FEAT_IDX['name_tfidf_cos']] = _tfidf_cosine(n1, n2, self._name_idf)

        # 7. Phonetic equality
        ph1, ph2 = r1.phonetic, r2.phonetic
        fv[FEAT_IDX['name_phonetic_eq']] = (
            1.0 if (ph1 and ph2 and ph1 == ph2) else 0.0
        )

        # ── ADDRESS FEATURES ────────────────────────────────────────────

        a1 = r1.clean_addr
        a2 = r2.clean_addr

        # 8. Address Levenshtein
        fv[FEAT_IDX['addr_lev_ratio']] = (
            rfd.Levenshtein.normalized_similarity(a1, a2)
            if a1 and a2 else MISSING_VAL
        )

        # 9. Address Jaro-Winkler
        fv[FEAT_IDX['addr_jaro_winkler']] = (
            jellyfish.jaro_winkler_similarity(a1, a2)
            if a1 and a2 else MISSING_VAL
        )

        # 10. Address token-set ratio
        fv[FEAT_IDX['addr_token_set']] = (
            rff.token_set_ratio(a1, a2) / 100.0
            if a1 and a2 else MISSING_VAL
        )

        # 11. Address token-sort ratio
        fv[FEAT_IDX['addr_token_sort']] = (
            rff.token_sort_ratio(a1, a2) / 100.0
            if a1 and a2 else MISSING_VAL
        )

        # 12. Address token Jaccard
        fv[FEAT_IDX['addr_jaccard']] = (
            _jaccard(set(r1.addr_tokens), set(r2.addr_tokens))
            if r1.addr_tokens or r2.addr_tokens else MISSING_VAL
        )

        # 13. Address TF-IDF cosine
        fv[FEAT_IDX['addr_tfidf_cos']] = (
            _tfidf_cosine(a1, a2, self._addr_idf)
            if a1 and a2 else MISSING_VAL
        )

        # 14. Postal code equality
        fv[FEAT_IDX['addr_postal_eq']] = _eq_with_missing(
            r1.postal_code, r2.postal_code
        )

        # 15. State/region equality
        fv[FEAT_IDX['addr_state_eq']] = _eq_with_missing(
            r1.state_region, r2.state_region
        )

        # 16. City equality
        fv[FEAT_IDX['addr_city_eq']] = _eq_with_missing(r1.city, r2.city)

        # ── OTHER FEATURES ──────────────────────────────────────────────

        # 17. Country equality
        fv[FEAT_IDX['country_eq']] = _country_eq(r1.country, r2.country)

        # 18. Name length difference (normalised)
        l1, l2 = len(n1), len(n2)
        fv[FEAT_IDX['name_len_diff']] = abs(l1 - l2) / max(l1, l2, 1)

        # 19. Address length difference (normalised)
        la1, la2 = len(a1), len(a2)
        fv[FEAT_IDX['addr_len_diff']] = (
            abs(la1 - la2) / max(la1, la2, 1)
            if la1 or la2 else MISSING_VAL
        )

        # 20. Digit-pattern similarity (Jaccard on digit-run token sets)
        d1 = set(_DIGIT_RUN_RE.findall(r1.raw_addr))
        d2 = set(_DIGIT_RUN_RE.findall(r2.raw_addr))
        fv[FEAT_IDX['digit_pattern_sim']] = (
            _jaccard(d1, d2) if d1 or d2 else MISSING_VAL
        )

        # 21–22. Legal suffix indicators
        fv[FEAT_IDX['s1_has_legal_suffix']]  = r1.has_legal
        fv[FEAT_IDX['s23_has_legal_suffix']] = r2.has_legal

        # 23–24. Missing address indicators
        fv[FEAT_IDX['s1_addr_missing']]  = r1.addr_missing
        fv[FEAT_IDX['s23_addr_missing']] = r2.addr_missing

        return fv

    # ------------------------------------------------------------------
    # Batch build
    # ------------------------------------------------------------------

    def build_pair_features(
        self,
        candidates: Dict[str, set],
        label_dict: Optional[Dict[str, List[str]]] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate feature matrix for all blocked candidate pairs.

        Parameters
        ----------
        candidates  : {s1_id -> set(s23_ids)} from BlockingEngine
        label_dict  : optional {s1_id -> [matched_s23_ids]} ground truth;
                      if provided, adds a 'label' column (1=match, 0=non-match)
        verbose     : print progress

        Returns
        -------
        X    : np.ndarray, shape (n_pairs, N_FEATURES), dtype float32
        meta : pd.DataFrame with columns [s1_id, s23_id] and optionally [label]
        """
        import time
        t0 = time.time()

        # Flatten candidate pairs (deterministic: sorted iteration)
        pairs: List[Tuple[str, str]] = []
        for s1_id in sorted(candidates.keys()):
            for s23_id in sorted(candidates[s1_id]):
                pairs.append((s1_id, s23_id))

        n_pairs = len(pairs)
        if verbose:
            print(f"  Building features for {n_pairs:,} pairs...", flush=True)

        X    = np.zeros((n_pairs, N_FEATURES), dtype=np.float32)
        s1s  = []
        s23s = []

        REPORT_EVERY = 100_000
        missing_s1  = 0
        missing_s23 = 0

        for i, (s1_id, s23_id) in enumerate(pairs):
            r1 = self._s1_cache.get(s1_id)
            r2 = self._s23_cache.get(s23_id)

            if r1 is None:
                missing_s1 += 1
                r1 = _make_empty_cache(s1_id)
            if r2 is None:
                missing_s23 += 1
                r2 = _make_empty_cache(s23_id)

            X[i] = self._pair_features(r1, r2)
            s1s.append(s1_id)
            s23s.append(s23_id)

            if verbose and (i + 1) % REPORT_EVERY == 0:
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed
                eta     = (n_pairs - i - 1) / rate
                print(f"    {i+1:>8,}/{n_pairs:,}  "
                      f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
                      f"rate={rate:.0f}/s", flush=True)

        if verbose:
            elapsed = time.time() - t0
            print(f"    Done: {n_pairs:,} pairs in {elapsed:.1f}s  "
                  f"({n_pairs/elapsed:.0f} pairs/s)", flush=True)
            if missing_s1 or missing_s23:
                print(f"    WARNING: {missing_s1} missing S1 records, "
                      f"{missing_s23} missing S23 records", flush=True)

        meta = pd.DataFrame({'s1_id': s1s, 's23_id': s23s})

        if label_dict is not None:
            # Build label: 1 = true match, 0 = non-match
            match_sets = {s1: set(mids) for s1, mids in label_dict.items()}
            labels = [
                1 if s23 in match_sets.get(s1, set()) else 0
                for s1, s23 in zip(s1s, s23s)
            ]
            meta['label'] = labels

        return X, meta


def _make_empty_cache(eid: str) -> _RecordCache:
    """Fallback empty cache for records not found in lookup."""
    return _RecordCache(
        entity_id=eid, country='',
        raw_name='', core_name='',
        phonetic='', name_tokens=frozenset(), has_legal=0.0,
        raw_addr='', clean_addr='',
        addr_tokens=frozenset(), postal_code=None,
        state_region=None, city=None, addr_missing=1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def feature_report(X: np.ndarray, meta: pd.DataFrame) -> None:
    """Print a summary of the feature matrix."""
    SEP = '=' * 66
    print(f"\n{SEP}")
    print("  FEATURE MATRIX REPORT")
    print(SEP)
    n_pairs, n_feat = X.shape
    print(f"  Pairs     : {n_pairs:>10,}")
    print(f"  Features  : {n_feat:>10,}")

    if 'label' in meta.columns:
        pos = int(meta['label'].sum())
        neg = n_pairs - pos
        print(f"  Positives : {pos:>10,}  ({100*pos/n_pairs:.2f}%)")
        print(f"  Negatives : {neg:>10,}  ({100*neg/n_pairs:.2f}%)")

    print(f"\n  {'Feature':<24}  {'Mean':>8}  {'Std':>8}  {'Missing(-1)':>12}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*12}")
    for j, fname in enumerate(ALL_FEATURES):
        col    = X[:, j]
        n_miss = int((col == MISSING_VAL).sum())
        valid  = col[col != MISSING_VAL]
        mean   = float(valid.mean()) if len(valid) else float('nan')
        std    = float(valid.std())  if len(valid) else float('nan')
        print(f"  {fname:<24}  {mean:>8.4f}  {std:>8.4f}  {n_miss:>12,}")

    print(SEP)

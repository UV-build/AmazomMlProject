"""
Amazon ML Challenge 2026 - Business Entity Resolution
blocking.py

Multi-key candidate-generation system.

Blocking keys applied to every (S1, S2/S3) pair:
  K1 - Exact normalised core business name
  K2 - Sorted-token business name  (handles word-order variants)
  K3 - Phonetic (Metaphone) business name
  K4 - Name token overlap  (>= min_name_token_overlap shared tokens)
  K5 - Address token overlap (>= min_addr_token_overlap shared tokens)
  K6 - Character trigram overlap (fast n-gram similarity proxy)

Country scoping:
  Candidates are generated within the same country first.
  A cross-country pass on K1+K2 handles transliteration pairs where country
  tags differ.  Country is an open string -- unseen values (France in test)
  work automatically.

Performance design:
  - _build_record_repr calls normalize_name + strip_legal_suffixes exactly
    once per record; all other representations are derived cheaply.
  - K1/K2/K3 use pandas merge (vectorised) -- O(n log n).
  - K4/K5/K6 use inverted indexes trimmed to max_postings_pct of corpus to
    drop high-frequency tokens before iteration.
  - All token-overlap keys are processed in 10k-row chunks to bound memory.
  - Candidate cap is applied with priority ordering (K1/K2/K3 kept first).
  - Output is reproducible: all key iterations use sorted IDs.

No external data, no ML training, no C-extension DLLs beyond numpy/scipy.
"""

from __future__ import annotations

import os
import sys
import math
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import (
    normalize_name,
    strip_legal_suffixes,
    address_tokens,
    _safe_str,
)

import jellyfish


# ============================================================
# Configuration
# ============================================================

@dataclass
class BlockingConfig:
    # Token-overlap thresholds
    min_name_token_overlap: int = 2
    min_addr_token_overlap: int = 2
    min_trigram_overlap:    int = 4

    # Posting-list cap: tokens in > max_postings_pct of S23 are ignored
    max_postings_pct: float = 0.005   # 0.5 % of S23

    # Per-key result cap before union (prevents token-overlap explosion)
    per_key_cap: int = 500

    # Final cap per S1 entity after union
    candidate_cap: int = 200

    # Cross-country pass for K1+K2 (catches transliterations)
    cross_country_k1k2: bool = True

    seed: int = 42

    name_stop_tokens: FrozenSet[str] = field(default_factory=lambda: frozenset({
        'and', 'the', 'of', 'in', 'a', 'an', 'for', 'at', 'by', 'to',
        'pvt', 'pvtltd', 'ltd', 'llc', 'llp', 'inc', 'corp', 'co',
        'private', 'limited', 'public', 'group', 'services', 'solutions',
    }))

    addr_stop_tokens: FrozenSet[str] = field(default_factory=lambda: frozenset({
        'st', 'ave', 'blvd', 'rd', 'dr', 'ln', 'ct', 'pl', 'pkwy', 'hwy',
        'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw',
        'unit', 'apt', 'ste', 'suite', 'floor', 'fl', 'bldg',
        'no', 'and', 'the', 'of',
    }))


DEFAULT_CONFIG = BlockingConfig()


# ============================================================
# Helpers
# ============================================================

def _country_matches(c1: str, c2: str) -> bool:
    """True when countries match (case-insensitive) or either is empty."""
    if not c1 or not c2:
        return True
    return c1.lower() == c2.lower()


def _build_record_repr(
    df: pd.DataFrame,
    cfg: BlockingConfig,
) -> Dict[str, dict]:
    """
    Pre-compute all blocking representations for every record in df.

    Calls normalize_name + strip_legal_suffixes once per record.
    All other representations (sorted tokens, phonetic, trigrams, address
    tokens) are derived from that single canonical string at negligible cost.

    Returns {entity_id: {...}} dict.
    """
    name_stop = cfg.name_stop_tokens
    addr_stop = cfg.addr_stop_tokens

    eids    = df['entity_id'].fillna('').astype(str).tolist()
    names   = df['business_name'].fillna('').astype(str).tolist()
    addrs   = df['business_address'].fillna('').astype(str).tolist()
    ctries  = df['country'].fillna('').astype(str).tolist()

    records: Dict[str, dict] = {}

    for eid, raw_name, raw_addr, country in zip(eids, names, addrs, ctries):
        # Step 1: canonical name (one call to the full normalize pipeline)
        canon = normalize_name(raw_name)
        core  = strip_legal_suffixes(canon)

        # Step 2: name tokens filtered by stop-list
        toks = [t for t in core.split() if t not in name_stop and len(t) >= 2]

        # Step 3: sorted-token bag (word-order normalisation)
        st = ' '.join(sorted(toks))

        # Step 4: Metaphone phonetic key per ASCII token
        ph_parts = []
        for tok in toks:
            if tok.isascii():
                try:
                    code = jellyfish.metaphone(tok)
                    ph_parts.append(code if code else tok)
                except Exception:
                    ph_parts.append(tok)
            else:
                ph_parts.append(tok)
        ph = ' '.join(ph_parts)

        # Step 5: address tokens
        atok = [t for t in address_tokens(raw_addr, country)
                if t not in addr_stop and len(t) >= 2]

        # Step 6: character 3-grams of core name
        padded = f' {core} '
        trig = list({padded[i:i+3] for i in range(len(padded) - 2)})

        records[eid] = {
            'country':      country,
            'core_name':    core,
            'sorted_tok':   st,
            'phonetic':     ph,
            'name_tokens':  toks,
            'addr_tokens':  atok,
            'trigrams':     trig,
        }

    return records


def _build_inverted_index(
    records: Dict[str, dict],
    key_fn,
) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = defaultdict(list)
    for eid, rec in records.items():
        k = key_fn(rec)
        if k:
            idx[k].append(eid)
    return dict(idx)


def _build_token_index(
    records: Dict[str, dict],
    field: str,
    max_postings: int,
) -> Dict[str, List[str]]:
    """Build token -> [entity_id] index, dropping tokens with > max_postings entries."""
    idx: Dict[str, List[str]] = defaultdict(list)
    for eid, rec in records.items():
        for tok in rec.get(field, []):
            idx[tok].append(eid)
    return {k: v for k, v in idx.items() if len(v) <= max_postings}


# ============================================================
# Blocking engine
# ============================================================

@dataclass
class BlockingDiagnostics:
    n_s1:                int
    n_s23:               int
    all_pairs:           int
    candidate_pairs:     int
    reduction_ratio:     float
    gt_pairs_total:      int
    gt_pairs_recalled:   int
    recall:              float
    avg_cands_per_s1:    float
    median_cands_per_s1: float
    max_cands_per_s1:    int
    zero_cand_s1:        int


class BlockingEngine:
    """
    Multi-key blocking engine.

    Build once, call generate_candidates() for any subset of S1 IDs.
    """

    def __init__(
        self,
        s1_df:  pd.DataFrame,
        s2_df:  pd.DataFrame,
        s3_df:  pd.DataFrame,
        config: BlockingConfig = DEFAULT_CONFIG,
    ) -> None:
        self.config = config

        # ---- record representations ----
        print("  Building record representations...", flush=True)
        t0 = time.time()
        self._s1  = _build_record_repr(s1_df, config)
        self._s2  = _build_record_repr(s2_df, config)
        self._s3  = _build_record_repr(s3_df, config)
        self._s23 = {**self._s2, **self._s3}
        n_s23 = len(self._s23)
        print(f"    done in {time.time()-t0:.1f}s  "
              f"(S1={len(self._s1):,}  S23={n_s23:,})", flush=True)

        max_post = max(50, int(config.max_postings_pct * n_s23))
        self._max_postings = max_post

        # ---- inverted indexes over S23 ----
        print("  Building inverted indexes...", flush=True)
        t0 = time.time()
        self._idx_core     = _build_inverted_index(self._s23, lambda r: r['core_name'])
        self._idx_sorted   = _build_inverted_index(self._s23, lambda r: r['sorted_tok'])
        self._idx_phonetic = _build_inverted_index(self._s23, lambda r: r['phonetic'])
        self._idx_nametok  = _build_token_index(self._s23, 'name_tokens', max_post)
        self._idx_addrtok  = _build_token_index(self._s23, 'addr_tokens', max_post)
        self._idx_trigram  = _build_token_index(self._s23, 'trigrams',    max_post)
        print(f"    done in {time.time()-t0:.1f}s  "
              f"max_postings={max_post:,}  "
              f"nametok={len(self._idx_nametok):,}  "
              f"trigram={len(self._idx_trigram):,}", flush=True)

    # ----------------------------------------------------------
    # Factory: build S23 indexes once; load S1 batches later
    # ----------------------------------------------------------

    @classmethod
    def from_s23_only(
        cls,
        s2_df:  pd.DataFrame,
        s3_df:  pd.DataFrame,
        config: BlockingConfig = DEFAULT_CONFIG,
        verbose: bool = True,
    ) -> 'BlockingEngine':
        """
        Build only the S23 side of the engine (repr + inverted indexes).
        Call load_s1_batch() to set the S1 representations for each batch.
        This amortises the expensive S23 build across many S1 batches.
        """
        obj = object.__new__(cls)
        obj.config = config
        obj._s1    = {}   # filled per batch via load_s1_batch()

        if verbose:
            print("  Building S23 record representations...", flush=True)
        t0 = time.time()
        obj._s2  = _build_record_repr(s2_df, config)
        obj._s3  = _build_record_repr(s3_df, config)
        obj._s23 = {**obj._s2, **obj._s3}
        n_s23 = len(obj._s23)
        if verbose:
            print(f"    done in {time.time()-t0:.1f}s  S23={n_s23:,}", flush=True)

        max_post = max(50, int(config.max_postings_pct * n_s23))
        obj._max_postings = max_post

        if verbose:
            print("  Building inverted indexes...", flush=True)
        t0 = time.time()
        obj._idx_core     = _build_inverted_index(obj._s23, lambda r: r['core_name'])
        obj._idx_sorted   = _build_inverted_index(obj._s23, lambda r: r['sorted_tok'])
        obj._idx_phonetic = _build_inverted_index(obj._s23, lambda r: r['phonetic'])
        obj._idx_nametok  = _build_token_index(obj._s23, 'name_tokens', max_post)
        obj._idx_addrtok  = _build_token_index(obj._s23, 'addr_tokens', max_post)
        obj._idx_trigram  = _build_token_index(obj._s23, 'trigrams',    max_post)
        if verbose:
            print(f"    done in {time.time()-t0:.1f}s  "
                  f"max_postings={max_post:,}  "
                  f"nametok={len(obj._idx_nametok):,}  "
                  f"trigram={len(obj._idx_trigram):,}", flush=True)
        return obj

    def load_s1_batch(self, s1_df: pd.DataFrame) -> None:
        """Replace the S1 record representations with a new batch."""
        self._s1 = _build_record_repr(s1_df, self.config)

    # ----------------------------------------------------------
    # Per-key generators (operate on a single S1 record dict)
    # ----------------------------------------------------------

    def _k1(self, rec: dict) -> List[str]:
        k = rec['core_name']
        return self._idx_core.get(k, []) if k else []

    def _k2(self, rec: dict) -> List[str]:
        k = rec['sorted_tok']
        return self._idx_sorted.get(k, []) if k else []

    def _k3(self, rec: dict) -> List[str]:
        k = rec['phonetic']
        return self._idx_phonetic.get(k, []) if k else []

    def _k4(self, rec: dict) -> List[str]:
        return self._overlap(rec['name_tokens'], self._idx_nametok,
                             self.config.min_name_token_overlap)

    def _k5(self, rec: dict) -> List[str]:
        return self._overlap(rec['addr_tokens'], self._idx_addrtok,
                             self.config.min_addr_token_overlap)

    def _k6(self, rec: dict) -> List[str]:
        return self._overlap(rec['trigrams'], self._idx_trigram,
                             self.config.min_trigram_overlap)

    def _overlap(
        self,
        tokens: List[str],
        idx: Dict[str, List[str]],
        min_ov: int,
    ) -> List[str]:
        if not tokens:
            return []
        ctr: Counter = Counter()
        for tok in tokens:
            for eid in idx.get(tok, []):
                ctr[eid] += 1
        cap = self.config.per_key_cap
        hits = [(eid, cnt) for eid, cnt in ctr.items() if cnt >= min_ov]
        if len(hits) > cap:
            hits.sort(key=lambda x: (-x[1], x[0]))   # count desc, eid asc (stable)
            hits = hits[:cap]
        return [eid for eid, _ in hits]

    # ----------------------------------------------------------
    # Batch candidate generation  (vectorised merge for K1/K2/K3)
    # ----------------------------------------------------------

    def generate_candidates(
        self,
        s1_ids: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Set[str]]:
        """
        Generate candidate S23 IDs for every S1 entity.

        K1/K2/K3  use pandas inner-join (fast, vectorised).
        K4/K5/K6  use chunk-wise inverted-index iteration.

        Returns {s1_id -> set(s23_ids)}  -- reproducible (sorted iteration).
        """
        if s1_ids is None:
            s1_ids = sorted(self._s1.keys())
        s1_id_set = set(s1_ids)
        cfg = self.config
        t0  = time.time()

        result: Dict[str, Set[str]] = {sid: set() for sid in s1_ids}

        # ---- build slim DataFrames for merge-based keys ----
        s1_rows  = [(eid, r['core_name'], r['sorted_tok'],
                     r['phonetic'], r['country'])
                    for eid, r in self._s1.items() if eid in s1_id_set]
        s23_rows = [(eid, r['core_name'], r['sorted_tok'],
                     r['phonetic'], r['country'])
                    for eid, r in self._s23.items()]

        COLS = ['entity_id', 'core_name', 'sorted_tok', 'phonetic', 'country']
        df_s1  = pd.DataFrame(s1_rows,  columns=COLS)
        df_s23 = pd.DataFrame(s23_rows, columns=COLS)

        def _merge_key(col: str, label: str) -> None:
            """Inner-join on col, apply country scope, add to result."""
            left  = df_s1 [df_s1 [col] != ''][['entity_id', col, 'country']]
            right = df_s23[df_s23[col] != ''][['entity_id', col, 'country']] \
                        .rename(columns={'entity_id': 's23_id',
                                         'country':   's23_country'})
            m = left.merge(right, on=col, how='inner')

            # same-country or either blank
            same = (m['country'] == '') | \
                   (m['s23_country'] == '') | \
                   (m['country'].str.lower() == m['s23_country'].str.lower())
            for s1id, s23id in zip(m.loc[same, 'entity_id'],
                                   m.loc[same, 's23_id']):
                result[s1id].add(s23id)

            # cross-country pass for K1/K2
            if cfg.cross_country_k1k2 and col in ('core_name', 'sorted_tok'):
                cross = ~same & (m['country'] != '') & (m['s23_country'] != '')
                for s1id, s23id in zip(m.loc[cross, 'entity_id'],
                                       m.loc[cross, 's23_id']):
                    result[s1id].add(s23id)

            if verbose:
                n = sum(len(v) for v in result.values())
                print(f"    {label}: pairs so far = {n:,}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

        _merge_key('core_name',  'K1 exact name ')
        _merge_key('sorted_tok', 'K2 sorted toks')
        _merge_key('phonetic',   'K3 phonetic   ')

        # ---- K4 / K5 / K6 : chunked inverted-index ----
        key_specs = [
            ('name_tokens', self._idx_nametok, cfg.min_name_token_overlap, 'K4 name tok  '),
            ('addr_tokens', self._idx_addrtok, cfg.min_addr_token_overlap, 'K5 addr tok  '),
            ('trigrams',    self._idx_trigram, cfg.min_trigram_overlap,    'K6 trigrams  '),
        ]
        CHUNK = 10_000

        for field, idx, min_ov, label in key_specs:
            s1_list = [sid for sid in s1_ids if sid in self._s1]
            for chunk_start in range(0, len(s1_list), CHUNK):
                chunk = s1_list[chunk_start: chunk_start + CHUNK]
                for s1id in chunk:
                    rec  = self._s1[s1id]
                    c1   = rec['country']
                    hits = self._overlap(rec[field], idx, min_ov)
                    if c1:
                        hits = [h for h in hits
                                if _country_matches(c1, self._s23[h]['country'])]
                    result[s1id].update(hits)

            if verbose:
                n = sum(len(v) for v in result.values())
                print(f"    {label}: pairs so far = {n:,}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

        # ---- apply final candidate cap ----
        #  Priority: K1/K2/K3 hits are always kept; K4/K5/K6 fill the rest.
        capped = 0
        cap = cfg.candidate_cap
        if cap > 0:
            # Re-fetch K1/K2/K3 hits cheaply from the already-merged DataFrames
            priority_sets: Dict[str, Set[str]] = {sid: set() for sid in s1_ids}
            for col in ('core_name', 'sorted_tok', 'phonetic'):
                left  = df_s1 [df_s1 [col] != ''][['entity_id', col, 'country']]
                right = df_s23[df_s23[col] != ''][['entity_id', col, 'country']] \
                            .rename(columns={'entity_id': 's23_id',
                                             'country':   's23_country'})
                m = left.merge(right, on=col, how='inner')
                same = (m['country'] == '') | \
                       (m['s23_country'] == '') | \
                       (m['country'].str.lower() == m['s23_country'].str.lower())
                for s1id, s23id in zip(m.loc[same, 'entity_id'],
                                       m.loc[same, 's23_id']):
                    if s1id in priority_sets:
                        priority_sets[s1id].add(s23id)

            for s1id in s1_ids:
                cands = result[s1id]
                if len(cands) > cap:
                    pri  = priority_sets[s1id] & cands
                    rest = sorted(cands - pri)   # sorted -> deterministic
                    result[s1id] = pri | set(rest[:max(0, cap - len(pri))])
                    capped += 1

        if verbose:
            total = sum(len(v) for v in result.values())
            print(f"    Cap applied to {capped:,} entities.  "
                  f"Total pairs = {total:,}  ({time.time()-t0:.1f}s)", flush=True)

        return result


# ============================================================
# Diagnostics
# ============================================================

def compute_diagnostics(
    candidates:   Dict[str, Set[str]],
    n_s1:         int,
    n_s23:        int,
    ground_truth: Optional[Dict[str, List[str]]] = None,
) -> BlockingDiagnostics:
    sizes = np.array([len(v) for v in candidates.values()], dtype=np.int64)

    candidate_pairs = int(sizes.sum())
    all_pairs       = n_s1 * n_s23
    rr  = 1.0 - candidate_pairs / all_pairs if all_pairs > 0 else 0.0
    avg = float(sizes.mean())        if len(sizes) else 0.0
    med = float(np.median(sizes))    if len(sizes) else 0.0
    mx  = int(sizes.max())           if len(sizes) else 0
    z   = int((sizes == 0).sum())

    gt_total = gt_hit = 0
    if ground_truth:
        for s1id, mids in ground_truth.items():
            if not mids:
                continue
            gt_total += len(mids)
            cs = candidates.get(s1id, set())
            gt_hit += sum(1 for m in mids if m in cs)

    recall = gt_hit / gt_total if gt_total else 0.0

    return BlockingDiagnostics(
        n_s1=n_s1, n_s23=n_s23,
        all_pairs=all_pairs,
        candidate_pairs=candidate_pairs,
        reduction_ratio=rr,
        gt_pairs_total=gt_total,
        gt_pairs_recalled=gt_hit,
        recall=recall,
        avg_cands_per_s1=avg,
        median_cands_per_s1=med,
        max_cands_per_s1=mx,
        zero_cand_s1=z,
    )


def print_diagnostics(d: BlockingDiagnostics, label: str = '') -> None:
    SEP = '=' * 66
    hdr = f"  BLOCKING DIAGNOSTICS -- {label}" if label else "  BLOCKING DIAGNOSTICS"
    print(f"\n{SEP}")
    print(hdr)
    print(SEP)
    print(f"  S1 entities            : {d.n_s1:>12,}")
    print(f"  S2+S3 entities         : {d.n_s23:>12,}")
    print(f"  All possible pairs     : {d.all_pairs:>12,}")
    print(f"  Candidate pairs        : {d.candidate_pairs:>12,}")
    print(f"  Reduction ratio        : {d.reduction_ratio:>12.4%}")
    print(f"  Avg candidates / S1    : {d.avg_cands_per_s1:>12.1f}")
    print(f"  Median candidates / S1 : {d.median_cands_per_s1:>12.1f}")
    print(f"  Max candidates / S1    : {d.max_cands_per_s1:>12,}")
    print(f"  S1 with 0 candidates   : {d.zero_cand_s1:>12,}")
    if d.gt_pairs_total > 0:
        print(f"  Ground-truth pairs     : {d.gt_pairs_total:>12,}")
        print(f"  GT pairs recalled      : {d.gt_pairs_recalled:>12,}")
        print(f"  GT pairs missed        : {d.gt_pairs_total - d.gt_pairs_recalled:>12,}")
        print(f"  Blocking recall        : {d.recall:>12.4%}")
    print(SEP)

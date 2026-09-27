"""
gen_dataset.py  --  generate X_train.npy and meta_train.csv atomically.

Run as:
    $env:PYTHONHASHSEED='0'; python -u gen_dataset.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from data_ingestion import load_source, load_ground_truth, parse_ground_truth
from blocking import BlockingConfig, BlockingEngine
from features import FeatureBuilder, ALL_FEATURES, N_FEATURES

S1_SAMPLE  = 100_000
S23_SAMPLE = 200_000
SEED       = 42
_OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..', '..', 'output'))

def sample(df, n, seed=SEED):
    if len(df) <= n:
        return df.copy()
    total = len(df)
    parts = []
    for _, grp in df.groupby('country', sort=True):
        k = min(len(grp), max(1, round(n * len(grp) / total)))
        parts.append(grp.sample(k, random_state=seed))
    result = pd.concat(parts, ignore_index=True)
    if len(result) < n:
        extra = df[~df['entity_id'].isin(result['entity_id'])].sample(
            n - len(result), random_state=seed + 1)
        result = pd.concat([result, extra], ignore_index=True)
    return result.head(n).reset_index(drop=True)

print("[1] Loading data...", flush=True)
t0 = time.time()
ts1f = load_source('train', 1)
ts2f = load_source('train', 2)
ts3f = load_source('train', 3)
gt_df = load_ground_truth()
gt_dict, _ = parse_ground_truth(gt_df)
print(f"    done in {time.time()-t0:.1f}s", flush=True)

ts1 = sample(ts1f, S1_SAMPLE)
ts2 = sample(ts2f, S23_SAMPLE)
ts3 = sample(ts3f, S23_SAMPLE)
s1_set = set(ts1['entity_id'].astype(str))
s2_set = set(ts2['entity_id'].astype(str))
s3_set = set(ts3['entity_id'].astype(str))
gt_s = {k: [m for m in v if m in s2_set or m in s3_set]
        for k, v in gt_dict.items() if k in s1_set}
gt_s = {k: v for k, v in gt_s.items() if v}
gt_pairs = sum(len(v) for v in gt_s.values())
print(f"    S1={len(ts1):,}  S2={len(ts2):,}  S3={len(ts3):,}  GT={gt_pairs:,}", flush=True)

print("[2] Blocking...", flush=True)
cfg = BlockingConfig(
    min_name_token_overlap=2, min_addr_token_overlap=2,
    min_trigram_overlap=4, max_postings_pct=0.005,
    per_key_cap=500, candidate_cap=200, cross_country_k1k2=True,
)
eng = BlockingEngine(ts1, ts2, ts3, cfg)
cands = eng.generate_candidates(verbose=False)
n_c = sum(len(v) for v in cands.values())
print(f"    Candidates: {n_c:,}", flush=True)

print("[3] FeatureBuilder...", flush=True)
fb = FeatureBuilder(ts1, ts2, ts3, verbose=True)

print("[4] Generating features...", flush=True)
X, meta = fb.build_pair_features(cands, label_dict=gt_s, verbose=True)

n_pos = int(meta['label'].sum())
n_neg = len(meta) - n_pos
print(f"    pairs={len(meta):,}  pos={n_pos:,}  neg={n_neg:,}", flush=True)

print("[5] Saving...", flush=True)
os.makedirs(_OUT, exist_ok=True)
np.save(os.path.join(_OUT, 'X_train.npy'), X)
meta.to_csv(os.path.join(_OUT, 'meta_train.csv'), index=False)
print(f"    X_train.npy  shape={X.shape}", flush=True)
print(f"    meta_train.csv  rows={len(meta):,}", flush=True)

# Verify alignment
X2   = np.load(os.path.join(_OUT, 'X_train.npy'))
meta2 = pd.read_csv(os.path.join(_OUT, 'meta_train.csv'))
assert len(X2) == len(meta2), f"MISMATCH: X={len(X2)} meta={len(meta2)}"
print(f"    Alignment check PASSED: {len(X2):,} rows match", flush=True)
print("DONE")

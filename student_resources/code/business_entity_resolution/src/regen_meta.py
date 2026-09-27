"""
Regenerate meta_train.csv (s1_id, s23_id, label) using the same
deterministic sample + blocking configuration as run_features.py.
X_train.npy row order == meta_train.csv row order (both sorted by s1_id
then s23_id, which is how FeatureBuilder.build_pair_features iterates).
"""
import os, sys, time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import load_source, load_ground_truth, parse_ground_truth
from blocking import BlockingConfig, BlockingEngine

S1_SAMPLE  = 100_000
S23_SAMPLE = 200_000
SEED       = 42
_OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..','..','..','output'))

def stratified_sample(df, n, seed=42):
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

print("Loading data...", flush=True)
t0 = time.time()
ts1_full = load_source('train', 1)
ts2_full = load_source('train', 2)
ts3_full = load_source('train', 3)
gt_df    = load_ground_truth()
gt_dict, _ = parse_ground_truth(gt_df)
print(f"  loaded in {time.time()-t0:.1f}s")

ts1 = stratified_sample(ts1_full, S1_SAMPLE, SEED)
ts2 = stratified_sample(ts2_full, S23_SAMPLE, SEED)
ts3 = stratified_sample(ts3_full, S23_SAMPLE, SEED)
print(f"  sample S1={len(ts1):,} S2={len(ts2):,} S3={len(ts3):,}")

s1_set = set(ts1['entity_id'].astype(str))
s2_set = set(ts2['entity_id'].astype(str))
s3_set = set(ts3['entity_id'].astype(str))
gt_sample = {k: [m for m in v if m in s2_set or m in s3_set]
             for k, v in gt_dict.items() if k in s1_set}
gt_sample = {k: v for k, v in gt_sample.items() if v}
match_sets = {k: set(v) for k, v in gt_sample.items()}

print("Running blocking...", flush=True)
t0 = time.time()
cfg = BlockingConfig(min_name_token_overlap=2, min_addr_token_overlap=2,
                     min_trigram_overlap=4, max_postings_pct=0.005,
                     per_key_cap=500, candidate_cap=200, cross_country_k1k2=True)
engine = BlockingEngine(ts1, ts2, ts3, cfg)
candidates = engine.generate_candidates(verbose=False)
print(f"  blocking done in {time.time()-t0:.1f}s")

print("Building meta CSV...", flush=True)
rows = []
for s1_id in sorted(candidates.keys()):
    ms = match_sets.get(s1_id, set())
    for s23_id in sorted(candidates[s1_id]):
        rows.append((s1_id, s23_id, 1 if s23_id in ms else 0))

meta = pd.DataFrame(rows, columns=['s1_id', 's23_id', 'label'])
os.makedirs(_OUT, exist_ok=True)
meta.to_csv(os.path.join(_OUT, 'meta_train.csv'), index=False)
print(f"  meta_train.csv: {len(meta):,} rows  "
      f"pos={int(meta['label'].sum()):,}  "
      f"neg={int((meta['label']==0).sum()):,}")
print("Done.")

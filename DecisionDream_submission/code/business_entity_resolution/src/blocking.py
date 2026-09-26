import os
import re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESOURCE_ROOT = PROJECT_ROOT / "student_resource"


def find_dataset_root():
    for path in (RESOURCE_ROOT / "dataset", PROJECT_ROOT / "dataset"):
        if (path / "train" / "train_source1.tsv").is_file():
            return path
    raise FileNotFoundError("Could not locate dataset/train/train_source1.tsv")

# ─── Text Cleaning ─────────────────────────────────────────────────────────────
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # Remove all legal suffixes (English + French)
    text = re.sub(r'\b(pvt|ltd|limited|corp|corporation|inc|llc|the|and|of|co|sas|sarl|sa|sasu|eurl|snc|plc|gmbh|bv|nv)\b', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())

# ─── Load Data ─────────────────────────────────────────────────────────────────
def load_data(base_dir=None):
    print("Loading training datasets...")
    train_dir = Path(base_dir) if base_dir else find_dataset_root() / "train"
    s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", dtype=str).fillna("")
    return s1, s2, s3

def load_test_data(base_dir="student_resource/dataset/test"):
    print("Loading test datasets...")
    s1 = pd.read_csv(f"{base_dir}/test_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(f"{base_dir}/test_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(f"{base_dir}/test_source3.tsv", sep="\t", dtype=str).fillna("")
    return s1, s2, s3

# ─── Pass 1: Deterministic blocking (country + name prefix) ───────────────────
def pass1_deterministic(s1_df, target_pool):
    print("  Pass 1: Deterministic (country + 6-char prefix)...")
    for df in [s1_df, target_pool]:
        df['clean_name'] = df['business_name'].apply(clean_text)
        df['block_key']  = df['country'].astype(str).str.lower() + '_' + df['clean_name'].str[:6]

    target_counts = target_pool['block_key'].value_counts()
    valid_blocks  = target_counts[target_counts < 500].index
    filtered      = target_pool[target_pool['block_key'].isin(valid_blocks)]
    grouped       = filtered.groupby('block_key')['entity_id'].apply(list).to_dict()

    results = {}
    for _, row in s1_df.iterrows():
        sid  = row['entity_id']
        key  = row['block_key']
        results[sid] = set(grouped.get(key, []))
    return results

# ─── Pass 2: TF-IDF cosine similarity blocking ────────────────────────────────
def pass2_tfidf(s1_df, target_pool, top_k=15):
    print("  Pass 2: TF-IDF cosine similarity blocking...")
    s1_names     = s1_df['clean_name'].tolist()
    target_names = target_pool['clean_name'].tolist()

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    all_names  = s1_names + target_names
    vectorizer.fit(all_names)

    s1_vecs     = vectorizer.transform(s1_names)
    target_vecs = vectorizer.transform(target_names)
    target_ids  = target_pool['entity_id'].tolist()
    target_ctry = target_pool['country'].str.lower().str.strip().tolist()
    s1_ctry     = s1_df['country'].str.lower().str.strip().tolist()

    results = {}
    BATCH = 500
    for batch_start in range(0, len(s1_df), BATCH):
        batch_end    = min(batch_start + BATCH, len(s1_df))
        batch_vecs   = s1_vecs[batch_start:batch_end]
        batch_ctry   = s1_ctry[batch_start:batch_end]
        batch_ids    = s1_df['entity_id'].iloc[batch_start:batch_end].tolist()

        sims = cosine_similarity(batch_vecs, target_vecs)

        for i, (sid, ctry) in enumerate(zip(batch_ids, batch_ctry)):
            row_sims = sims[i]
            # filter to same country only
            same_ctry_mask = [1 if tc == ctry else 0 for tc in target_ctry]
            masked = row_sims * np.array(same_ctry_mask)
            top_idx = np.argsort(masked)[::-1][:top_k]
            candidates = {target_ids[j] for j in top_idx if masked[j] > 0.3}
            results[sid] = results.get(sid, set()) | candidates

        if batch_start % 5000 == 0:
            print(f"    TF-IDF pass: {batch_end}/{len(s1_df)} S1 rows...")

    return results

# ─── Pass 3: Sorted bigram token blocking ─────────────────────────────────────
def pass3_bigram(s1_df, target_pool):
    print("  Pass 3: Sorted bigram token blocking...")
    def bigram_key(text):
        tokens = sorted(text.split())
        if len(tokens) < 2:
            return text[:8] if text else ""
        return tokens[0][:4] + tokens[-1][:4]

    s1_df      = s1_df.copy()
    target_pool = target_pool.copy()
    if 'clean_name' not in s1_df.columns:
        s1_df['clean_name'] = s1_df['business_name'].apply(clean_text)
    if 'clean_name' not in target_pool.columns:
        target_pool['clean_name'] = target_pool['business_name'].apply(clean_text)
    s1_df['bigram_key']      = s1_df['country'].str.lower() + '_' + s1_df['clean_name'].apply(bigram_key)
    target_pool['bigram_key'] = target_pool['country'].str.lower() + '_' + target_pool['clean_name'].apply(bigram_key)

    counts  = target_pool['bigram_key'].value_counts()
    valid   = counts[counts < 300].index
    filtered = target_pool[target_pool['bigram_key'].isin(valid)]
    grouped  = filtered.groupby('bigram_key')['entity_id'].apply(list).to_dict()

    results = {}
    for _, row in s1_df.iterrows():
        sid = row['entity_id']
        key = row['bigram_key']
        results[sid] = set(grouped.get(key, []))
    return results

# ─── Merge all passes ──────────────────────────────────────────────────────────
def generate_candidates(s1_df, s2_df, s3_df, include_tfidf=False):
    target_pool = pd.concat([
        s2_df[['entity_id', 'business_name', 'country']],
        s3_df[['entity_id', 'business_name', 'country']]
    ], ignore_index=True)
    target_pool['clean_name'] = target_pool['business_name'].apply(clean_text)

    r1 = pass1_deterministic(s1_df.copy(), target_pool.copy())
    if include_tfidf:
        r2 = pass2_tfidf(s1_df.copy(), target_pool)
    else:
        print("  Pass 2: TF-IDF skipped (enable ER_ENABLE_TFIDF=1 when sufficient RAM is available).")
        r2 = {}
    r3 = pass3_bigram(s1_df.copy(), target_pool)

    print("  Merging all passes...")
    all_ids  = s1_df['entity_id'].tolist()
    merged   = {}
    for sid in all_ids:
        merged[sid] = r1.get(sid, set()) | r2.get(sid, set()) | r3.get(sid, set())
        merged[sid].discard(sid)  # never self-match

    with_cands = sum(1 for v in merged.values() if v)
    print(f"  S1 entities with candidates: {with_cands}/{len(all_ids)}")
    return merged

# ─── Save ──────────────────────────────────────────────────────────────────────
def save_candidates(candidates_dict, path=None):
    path = Path(path) if path else PROJECT_ROOT / "output" / "training_candidate_pairs.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"source1_entity_id": sid, "candidate_entity_ids": ",".join(cids) if cids else ""}
        for sid, cids in candidates_dict.items()
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    print(f"Saved {path} ({len(rows)} rows)")

if __name__ == "__main__":
    s1, s2, s3 = load_data()
    print("Running on TRAIN data to generate training candidates...")
    enable_tfidf = os.environ.get("ER_ENABLE_TFIDF", "0") == "1"
    cands = generate_candidates(s1, s2, s3, include_tfidf=enable_tfidf)
    save_candidates(cands)
    print("Done. Now run features.py")

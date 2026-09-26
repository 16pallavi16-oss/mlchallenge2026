import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import unicodedata

# ─── Text utilities ────────────────────────────────────────────────────────────
LEGAL = re.compile(r'\b(pvt|ltd|limited|corp|corporation|inc|llc|co|sas|sarl|sa|sasu|eurl|snc|plc|gmbh|bv|nv|the|and|of)\b')
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESOURCE_ROOT = PROJECT_ROOT / "student_resource"


def find_dataset_root():
    for path in (RESOURCE_ROOT / "dataset", PROJECT_ROOT / "dataset"):
        if (path / "train" / "train_source1.tsv").is_file():
            return path
    raise FileNotFoundError("Could not locate dataset/train/train_source1.tsv")

def clean(text):
    if pd.isna(text) or str(text).strip() in ("", "nan"):
        return ""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = text.lower()
    text = LEGAL.sub('', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())

def token_set(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def extract_numbers(text):
    return set(re.findall(r'\d+', text))

# ─── Feature computation for one pair ─────────────────────────────────────────
def compute_features(n1, a1, c1, n2, a2, c2):
    cn1, cn2 = clean(n1), clean(n2)
    ca1, ca2 = clean(a1), clean(a2)

    f = {}

    # ── Name features ──────────────────────────────────────────────────────────
    f['name_ratio']        = fuzz.ratio(cn1, cn2) / 100.0
    f['name_partial']      = fuzz.partial_ratio(cn1, cn2) / 100.0
    f['name_token_sort']   = fuzz.token_sort_ratio(cn1, cn2) / 100.0
    f['name_token_set']    = fuzz.token_set_ratio(cn1, cn2) / 100.0
    f['name_jaccard']      = token_set(cn1, cn2)
    f['name_len_diff']     = abs(len(cn1) - len(cn2)) / (max(len(cn1), len(cn2)) + 1)
    f['name_len_s1']       = len(cn1)
    f['name_len_s2']       = len(cn2)
    f['name_word_cnt_s1']  = len(cn1.split())
    f['name_word_cnt_s2']  = len(cn2.split())
    f['name_exact']        = int(cn1 == cn2)

    # First word match
    w1 = cn1.split()
    w2 = cn2.split()
    f['name_first_word_match'] = int(bool(w1 and w2 and w1[0] == w2[0]))
    f['name_shared_tokens']    = len(set(w1) & set(w2))

    # ── Address features ───────────────────────────────────────────────────────
    f['addr_ratio']      = fuzz.ratio(ca1, ca2) / 100.0
    f['addr_partial']    = fuzz.partial_ratio(ca1, ca2) / 100.0
    f['addr_token_sort'] = fuzz.token_sort_ratio(ca1, ca2) / 100.0
    f['addr_token_set']  = fuzz.token_set_ratio(ca1, ca2) / 100.0
    f['addr_jaccard']    = token_set(ca1, ca2)
    f['addr_exact']      = int(ca1 == ca2)

    # Number overlap in address (house numbers, PIN codes)
    nums1 = extract_numbers(str(a1))
    nums2 = extract_numbers(str(a2))
    if nums1 | nums2:
        f['addr_num_overlap'] = len(nums1 & nums2) / len(nums1 | nums2)
    else:
        f['addr_num_overlap'] = 0.0

    # ── Country feature ────────────────────────────────────────────────────────
    f['same_country'] = int(
        str(c1).strip().lower() == str(c2).strip().lower()
        and str(c1).strip().lower() not in ("", "nan")
    )

    return f

# ─── Load data ─────────────────────────────────────────────────────────────────
def load_raw_data(base_dir="student_resource/dataset/train"):
    print("Loading raw datasets...")
    s1 = pd.read_csv(f"{base_dir}/train_source1.tsv", sep="\t", dtype=str).fillna("").set_index("entity_id")
    s2 = pd.read_csv(f"{base_dir}/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(f"{base_dir}/train_source3.tsv", sep="\t", dtype=str).fillna("")
    targets = pd.concat([s2, s3], ignore_index=True).set_index("entity_id")
    return s1, targets

# ─── Build training pairs ──────────────────────────────────────────────────────
def build_pairs(candidate_path="output/candidate_pairs.tsv", sample_size=None):
    print("Loading candidate pairs...")
    cands = pd.read_csv(candidate_path, sep="\t", dtype=str).fillna("")
    cands = cands[cands['candidate_entity_ids'].str.strip() != ""]

    if sample_size:
        cands = cands.sample(n=min(sample_size, len(cands)), random_state=42)

    cands['target_id'] = cands['candidate_entity_ids'].str.split(',')
    pairs = cands.explode('target_id')[['source1_entity_id', 'target_id']].copy()
    pairs['target_id'] = pairs['target_id'].str.strip()
    pairs = pairs[pairs['target_id'] != ""].reset_index(drop=True)
    print(f"Total pairs to featurize: {len(pairs):,}")
    return pairs

# ─── Generate features ─────────────────────────────────────────────────────────
def generate_features(pairs, s1, targets):
    print("Extracting features...")
    rows = []
    for i, row in enumerate(pairs.itertuples(), 1):
        sid = row.source1_entity_id
        tid = row.target_id
        try:
            s1r = s1.loc[sid]
            tgt = targets.loc[tid]
            f = compute_features(
                s1r.get('business_name',''), s1r.get('business_address',''), s1r.get('country',''),
                tgt.get('business_name',''), tgt.get('business_address',''), tgt.get('country','')
            )
            f['name_similarity'] = f['name_token_sort'] * 100.0
            f['address_similarity'] = f['addr_token_sort'] * 100.0
            f['is_same_country'] = f['same_country']
            f['source1_entity_id'] = sid
            f['target_id']         = tid
            rows.append(f)
        except KeyError:
            continue
        if i % 100000 == 0:
            print(f"  {i:,} / {len(pairs):,} pairs featurized...")
    return pd.DataFrame(rows)

# ─── Attach labels ─────────────────────────────────────────────────────────────
def attach_labels(features_df, gt_path="student_resource/dataset/train/train_ground_truth.tsv"):
    print("Attaching ground truth labels...")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    gt = gt[gt['matched_entity_ids'].str.strip() != ""]
    gt['target_id'] = gt['matched_entity_ids'].str.split(',')
    gt_pairs = gt.explode('target_id')[['source1_entity_id', 'target_id']].copy()
    gt_pairs['target_id'] = gt_pairs['target_id'].str.strip()
    gt_pairs['label'] = 1

    final = pd.merge(features_df, gt_pairs, on=['source1_entity_id', 'target_id'], how='left')
    final['label'] = final['label'].fillna(0).astype(int)
    print(f"Labels attached. Positives: {final['label'].sum():,} / {len(final):,}")
    return final

def find_default_candidate_path():
    candidates = [PROJECT_ROOT / "output" / "training_candidate_pairs.tsv"]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return "output/candidate_pairs.tsv"


def find_default_train_dir():
    return find_dataset_root() / "train"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract features for candidate pairs.")
    parser.add_argument("--candidate-path", default=find_default_candidate_path())
    parser.add_argument("--train-dir", default=find_default_train_dir())
    parser.add_argument("--sample-size", type=int, default=50000, help="Sample size of S1 entities to featurize")
    parser.add_argument("--output-file", default=str(PROJECT_ROOT / "output" / "training_features.csv"))
    args = parser.parse_args()

    s1, targets = load_raw_data(base_dir=args.train_dir)
    pairs = build_pairs(candidate_path=args.candidate_path, sample_size=args.sample_size)
    features = generate_features(pairs, s1, targets)
    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")
    labeled = attach_labels(features, gt_path=gt_path)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    labeled.to_csv(args.output_file, index=False)
    print(f"\nSaved {args.output_file}")
    print(f"Shape: {labeled.shape}")
    print(f"Class distribution:\n{labeled['label'].value_counts()}")


if __name__ == "__main__":
    main()

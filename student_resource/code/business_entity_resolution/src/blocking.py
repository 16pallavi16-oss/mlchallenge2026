import os
import re
import pandas as pd

def load_data():
    """Loads train TSV files safely using explicit tab separators."""
    print("Loading training datasets...")
    train_s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t")
    train_s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep="\t")
    train_s3 = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep="\t")
    return train_s1, train_s2, train_s3

def clean_text(text):
    """Normalizes business names and removes stop words."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Added common stop words to prevent massive block grouping
    text = re.sub(r'\b(pvt|ltd|limited|corp|corporation|inc|llc|the|and|of)\b', '', text)
    return " ".join(text.split())

def generate_blocking_candidates(s1_df, s2_df, s3_df):
    """
    RAM-safe candidate generation with frequency capping.
    """
    print("Preprocessing text fields...")
    for df in [s1_df, s2_df, s3_df]:
        df['clean_name'] = df['business_name'].apply(clean_text)
        # Stricter key: Country + first 6 characters of the cleaned name
        df['block_key'] = df['country'].astype(str).str.lower() + '_' + df['clean_name'].str[:6]

    # Pool targets
    target_pool = pd.concat([
        s2_df[['entity_id', 'block_key']], 
        s3_df[['entity_id', 'block_key']]
    ], ignore_index=True)

    print("Filtering explosive blocks to save RAM...")
    target_counts = target_pool['block_key'].value_counts()
    
    # CRITICAL: Drop any block key that matches more than 500 records.
    # This prevents the 123 GB RAM crash caused by empty or generic names.
    valid_blocks = target_counts[target_counts < 500].index
    target_pool = target_pool[target_pool['block_key'].isin(valid_blocks)]

    print("Building candidate dictionary...")
    grouped_targets = target_pool.groupby('block_key')['entity_id'].apply(list).to_dict()

    print("Mapping candidates to Source 1...")
    candidates_dict = {}
    for _, row in s1_df.iterrows():
        s1_id = row['entity_id']
        b_key = row['block_key']
        
        # Look up candidates; default to empty list if key was dropped or missing
        match_list = grouped_targets.get(b_key, [])
        candidates_dict[s1_id] = list(set(match_list))

    return candidates_dict

def save_candidates(candidates_dict):
    """Saves the candidate pairs into output/candidate_pairs.tsv."""
    os.makedirs("output", exist_ok=True)
    
    rows = [
        {
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cand_ids) if cand_ids else ""
        }
        for s1_id, cand_ids in candidates_dict.items()
    ]
    
    df_out = pd.DataFrame(rows)
    output_path = "output/candidate_pairs.tsv"
    df_out.to_csv(output_path, sep="\t", index=False)
    print(f"Success! Generated {output_path}")

if __name__ == "__main__":
    s1, s2, s3 = load_data()
    candidates = generate_blocking_candidates(s1, s2, s3)
    save_candidates(candidates)
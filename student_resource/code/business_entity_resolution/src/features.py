import os
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

def load_raw_data():
    """Loads the raw TSV files for feature lookup."""
    print("Loading raw datasets for text lookup...")
    s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t")
    s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep="\t")
    s3 = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep="\t")
    
    # Combine targets for easy lookup
    targets = pd.concat([s2, s3], ignore_index=True)
    targets.set_index("entity_id", inplace=True)
    s1.set_index("entity_id", inplace=True)
    
    return s1, targets

def build_training_pairs(sample_size=50000):
    """
    Safely parses the 1.5GB candidate_pairs.tsv and creates a manageable 
    dataset of pairs (S1_ID, Target_ID) for local ML training.
    """
    print("Parsing candidate pairs (this may take a minute)...")
    candidates = pd.read_csv("output/candidate_pairs.tsv", sep="\t")
    
    # Drop singletons (empty candidate lists) for training the pairwise classifier
    candidates = candidates.dropna(subset=['candidate_entity_ids'])
    
    # Sample Source 1 entities to prevent RAM crashes
    candidates_sample = candidates.sample(n=min(sample_size, len(candidates)), random_state=42)
    
    # Convert comma-separated strings to lists and explode into individual pairs
    candidates_sample['target_id'] = candidates_sample['candidate_entity_ids'].str.split(',')
    pairs_df = candidates_sample.explode('target_id')[['source1_entity_id', 'target_id']]
    
    return pairs_df

def compute_similarity(row, text_col, s1_df, target_df):
    """Computes RapidFuzz token ratio between two text fields."""
    try:
        val1 = str(s1_df.at[row['source1_entity_id'], text_col]).lower()
        val2 = str(target_df.at[row['target_id'], text_col]).lower()
        if val1 == 'nan' or val2 == 'nan':
            return 0.0
        return fuzz.token_sort_ratio(val1, val2)
    except KeyError:
        return 0.0

def generate_features(pairs_df, s1_df, target_df):
    """Extracts machine learning features for every pair."""
    print(f"Extracting features for {len(pairs_df)} pairs...")
    
    # 1. Name Similarity (Token Sort Ratio handles out-of-order words)
    pairs_df['name_similarity'] = pairs_df.apply(
        lambda x: compute_similarity(x, 'business_name', s1_df, target_df), axis=1
    )
    
    # 2. Address Similarity
    pairs_df['address_similarity'] = pairs_df.apply(
        lambda x: compute_similarity(x, 'business_address', s1_df, target_df), axis=1
    )
    
    # 3. Country Match Flag
    pairs_df['is_same_country'] = pairs_df.apply(
        lambda x: 1 if str(s1_df.at[x['source1_entity_id'], 'country']).lower() == 
                       str(target_df.at[x['target_id'], 'country']).lower() else 0, axis=1
    )
    
    return pairs_df

def attach_labels(features_df):
    """Attaches binary Ground Truth labels (1 for match, 0 for false positive)."""
    print("Attaching ground truth labels...")
    gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t")
    gt = gt.dropna(subset=['matched_entity_ids'])
    
    # Explode ground truth just like we did with candidates
    gt['target_id'] = gt['matched_entity_ids'].str.split(',')
    gt_pairs = gt.explode('target_id')[['source1_entity_id', 'target_id']]
    gt_pairs['label'] = 1  # These are the true matches
    
    # Left join onto our features; anything without a match gets a 0 label
    final_df = pd.merge(features_df, gt_pairs, on=['source1_entity_id', 'target_id'], how='left')
    final_df['label'] = final_df['label'].fillna(0).astype(int)
    
    return final_df

if __name__ == "__main__":
    s1, targets = load_raw_data()
    
    # Build a 50k S1 entity sample (~200k+ pairs) for training
    pairs = build_training_pairs(sample_size=50000)
    
    # Generate Features
    features = generate_features(pairs, s1, targets)
    
    # Attach 1/0 Labels
    labeled_features = attach_labels(features)
    
    # Save the training dataset
    os.makedirs("output", exist_ok=True)
    output_path = "output/training_features.csv"
    labeled_features.to_csv(output_path, index=False)
    
    print(f"Success! Generated {output_path}")
    print("Class Distribution:")
    print(labeled_features['label'].value_counts())
# Business Entity Resolution Pipeline

This repository contains the end-to-end Business Entity Resolution pipeline developed by Team **DecisionDream** for the ML Challenge 2026. The pipeline matches business records across noisy, independent data sources without shared identifiers.

---

## 1. Overview & Architecture

The pipeline consists of a two-stage approach designed to scale to millions of entity pairs efficiently:

1. **Candidate Generation (Blocking):**
   - Normalizes business names (lowercasing, punctuation stripping, stop-word removal for common legal suffixes like `pvt`, `ltd`, `corp`, `inc`, `llc`, etc.).
   - Constructs blocking keys combining country and the first 6 characters of the cleaned name (`<country>_<prefix6>`).
   - Caps block frequency at 500 records to prevent memory explosion.
   - Outputs candidate pairs to `output/candidate_pairs.tsv`.

2. **Feature Engineering:**
   - Computes pairwise similarity features:
     - `name_similarity`: RapidFuzz `token_sort_ratio` on business names.
     - `address_similarity`: RapidFuzz `token_sort_ratio` on business addresses.
     - `is_same_country`: Binary flag indicating country agreement.
   - Merges with ground truth labels (`train_ground_truth.tsv`) to create labeled pairs (`output/training_features.csv`).

3. **Classification & Matching:**
   - Trains an ensemble of 5-fold `StratifiedKFold` LightGBM binary classifiers.
   - Accounts for extreme class imbalance via `scale_pos_weight`.
   - Tunes decision thresholds on out-of-fold predictions to optimize the competition metric ($F_{0.5}$).
   - Scores test candidate pairs and generates `output/matching_results.tsv`.

---

## 2. Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10 – 3.14)
- Recommended: 16 GB+ RAM for full test dataset inference

### Installation
Install the pinned dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Pinned packages:
- `pandas==2.2.2`
- `numpy==1.26.4`
- `lightgbm==4.3.0`
- `rapidfuzz==3.9.3`
- `scikit-learn==1.4.2`
- `joblib==1.4.2`

---

## 3. Directory & Data Structure

The pipeline expects data located in the project root or standard dataset directory:

```text
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── blocking.py
│       │   ├── features.py
│       │   └── model.py
│       ├── requirements.txt
│       └── README.md
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
└── Documentation_template.md
```

---

## 4. End-to-End Reproduction Steps

Execute the following steps in sequence from the project root:

### Step 1: Candidate Generation (Blocking)
Run blocking to generate candidate pairs:
```bash
python code/business_entity_resolution/src/blocking.py
```
- **Input:** `dataset/test/` (or `train/`) source files
- **Output:** `output/candidate_pairs.tsv`

### Step 2: Feature Extraction
Extract pairwise features and create the training dataset:
```bash
python code/business_entity_resolution/src/features.py
```
- **Input:** `output/candidate_pairs.tsv` and `dataset/train/`
- **Output:** `output/training_features.csv`

### Step 3: Model Training & Test Matching
Train the 5-fold LightGBM ensemble and predict matches for test candidates:
```bash
python code/business_entity_resolution/src/model.py
```
- **Input:** `output/training_features.csv`, `output/candidate_pairs.tsv`, and `dataset/test/`
- **Output:** `output/matching_results.tsv` and serialized model checkpoint `output/lgb_model.pkl`

---

## 5. Verification & Validation

Verify the generated output files against official competition rules:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Expected result:
```text
PASS — no blocking issues found. Safe to submit.
```

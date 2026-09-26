import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rapidfuzz import fuzz
from sklearn.model_selection import StratifiedKFold


FEATURE_COLUMNS = ["name_similarity", "address_similarity", "is_same_country"]
RANDOM_STATE = 42
MODEL_VERSION = 3


def find_dataset_root():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    candidates = [
        os.path.join(project_root, "student_resource", "dataset"),
        os.path.join(project_root, "student_resource", "student_resource", "dataset"),
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "train", "train_source1.tsv")):
            return path
    raise FileNotFoundError("Could not find the train/test dataset directory.")


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
DATASET_ROOT = find_dataset_root()
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
MODEL_PATH = os.path.join(OUTPUT_ROOT, "lgb_model.pkl")


def f05_score(y_true, probabilities, threshold):
    predictions = probabilities >= threshold
    true_positives = np.sum((predictions == 1) & (y_true == 1))
    false_positives = np.sum((predictions == 1) & (y_true == 0))
    false_negatives = np.sum((predictions == 0) & (y_true == 1))
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    denominator = 0.25 * precision + recall
    return (1.25 * precision * recall) / denominator if denominator else 0.0


def train_models():
    features_path = os.path.join(OUTPUT_ROOT, "training_features.csv")
    training = pd.read_csv(features_path)
    required_features = set(FEATURE_COLUMNS) | {"label"}
    if not required_features.issubset(training.columns):
        raise ValueError(
            "training_features.csv is missing model features: "
            f"{sorted(required_features - set(training.columns))}"
        )
    x = training[FEATURE_COLUMNS].copy()
    x["name_similarity"] = pd.to_numeric(x["name_similarity"], errors="coerce").fillna(0.0) / 100.0
    x["address_similarity"] = pd.to_numeric(x["address_similarity"], errors="coerce").fillna(0.0) / 100.0
    x["is_same_country"] = pd.to_numeric(x["is_same_country"], errors="coerce").fillna(0.0)
    y = training["label"].astype(int)
    negatives = int((y == 0).sum())
    positives = int((y == 1).sum())
    if positives == 0:
        raise ValueError("training_features.csv contains no positive labels.")

    oof_predictions = np.zeros(len(training), dtype=float)
    models = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_indices, validation_indices) in enumerate(splitter.split(x, y), start=1):
        model = LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE + fold,
            scale_pos_weight=negatives / positives,
            verbosity=-1,
        )
        model.fit(x.iloc[train_indices], y.iloc[train_indices])
        oof_predictions[validation_indices] = model.predict_proba(
            x.iloc[validation_indices]
        )[:, 1]
        models.append(model)
        print(f"Trained fold {fold}/5")

    thresholds = np.arange(0.20, 0.951, 0.01)
    scores = [f05_score(y.to_numpy(), oof_predictions, threshold) for threshold in thresholds]
    best_index = int(np.argmax(scores))
    best_threshold = float(thresholds[best_index])
    print(f"Best threshold: {best_threshold:.2f}  →  OOF F0.5 = {scores[best_index]:.4f}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    joblib.dump(
        {
            "version": MODEL_VERSION,
            "models": models,
            "threshold": best_threshold,
            "features": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    return models, best_threshold


def text_value(value):
    return "" if pd.isna(value) else str(value)


def similarity(left, right):
    if not left.strip() or not right.strip():
        return 0.0
    return fuzz.token_sort_ratio(left, right) / 100.0


def load_test_features():
    test_root = os.path.join(DATASET_ROOT, "test")
    source1 = pd.read_csv(os.path.join(test_root, "test_source1.tsv"), sep="\t")
    source2 = pd.read_csv(os.path.join(test_root, "test_source2.tsv"), sep="\t")
    source3 = pd.read_csv(os.path.join(test_root, "test_source3.tsv"), sep="\t")
    targets = pd.concat([source2, source3], ignore_index=True).set_index("entity_id")

    candidates = pd.read_csv(os.path.join(OUTPUT_ROOT, "candidate_pairs.tsv"), sep="\t")
    candidates = candidates.dropna(subset=["candidate_entity_ids"])
    candidates = candidates[candidates["candidate_entity_ids"].astype(str).str.strip() != ""]
    candidates["target_id"] = candidates["candidate_entity_ids"].str.split(",")
    pairs = candidates.explode("target_id")[["source1_entity_id", "target_id"]]
    pairs["target_id"] = pairs["target_id"].astype(str).str.strip()
    pairs = pairs[pairs["target_id"].str.startswith(("S2-", "S3-"))]
    source1 = source1.set_index("entity_id")

    rows = []
    for pair in pairs.itertuples(index=False):
        if pair.source1_entity_id not in source1.index or pair.target_id not in targets.index:
            continue
        left = source1.loc[pair.source1_entity_id]
        right = targets.loc[pair.target_id]
        left_name = text_value(left["business_name"])
        right_name = text_value(right["business_name"])
        left_address = text_value(left["business_address"])
        right_address = text_value(right["business_address"])
        rows.append(
            {
                "source1_entity_id": pair.source1_entity_id,
                "target_id": pair.target_id,
                "name_similarity": similarity(left_name, right_name),
                "address_similarity": similarity(left_address, right_address),
                "is_same_country": int(
                    text_value(left["country"]).lower() == text_value(right["country"]).lower()
                ),
            }
        )
    return source1, pd.DataFrame(rows)


def write_matches(models, threshold):
    source1, pairs = load_test_features()
    matched = {entity_id: [] for entity_id in source1.index}
    if not pairs.empty:
        probabilities = np.mean(
            [model.predict_proba(pairs[FEATURE_COLUMNS])[:, 1] for model in models],
            axis=0,
        )
        for pair, probability in zip(pairs.itertuples(index=False), probabilities):
            if probability >= threshold:
                matched[pair.source1_entity_id].append(pair.target_id)

    output = pd.DataFrame(
        {
            "source1_entity_id": list(matched),
            "matched_entity_ids": [
                ",".join(dict.fromkeys(target_ids)) for target_ids in matched.values()
            ],
        }
    )
    output.to_csv(
        os.path.join(OUTPUT_ROOT, "matching_results.tsv"), sep="\t", index=False
    )


def main():
    saved = joblib.load(MODEL_PATH) if os.path.isfile(MODEL_PATH) else None
    if isinstance(saved, dict) and saved.get("version") == MODEL_VERSION:
        models = saved["models"]
        threshold = saved["threshold"]
        print(f"Loaded cached model from {MODEL_PATH}")
    else:
        models, threshold = train_models()
    predictor = Path(__file__).resolve().with_name("repredict.py")
    subprocess.run([sys.executable, str(predictor)], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
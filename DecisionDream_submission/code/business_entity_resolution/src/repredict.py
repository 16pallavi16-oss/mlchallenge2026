import csv
import os
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


CHUNK_SIZE = 100_000
THRESHOLD = 0.90
FEATURE_COLUMNS = ["name_similarity", "address_similarity", "is_same_country"]


def find_test_files(search_root):
    for current_root, _, files in os.walk(search_root):
        if "test_source1.tsv" in files:
            source1_path = Path(current_root) / "test_source1.tsv"
            source2_path = Path(current_root) / "test_source2.tsv"
            source3_path = Path(current_root) / "test_source3.tsv"
            if source2_path.is_file() and source3_path.is_file():
                return source1_path, source2_path, source3_path
    raise FileNotFoundError(
        f"Could not find test_source1.tsv, test_source2.tsv, and test_source3.tsv under {search_root}"
    )


def clean_values(values):
    return values.fillna("").astype(str).str.strip()


def normalize_texts(values):
    values = (
        values.fillna("")
        .astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(
            r"\b(pvt|ltd|limited|corp|corporation|inc|llc|co|sas|sarl|sa|sasu|eurl|snc|plc|gmbh|bv|nv|the|and|of)\b",
            "",
            regex=True,
        )
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.split()
        .str.join(" ")
    )
    return values


def score_chunk(pair_chunk, source1, targets, models, matched, probability_bins):
    source_ids = [pair[0] for pair in pair_chunk]
    target_ids = [pair[1] for pair in pair_chunk]

    left = source1.reindex(source_ids)
    right = targets.reindex(target_ids)
    name_left = normalize_texts(left["business_name"]).tolist()
    name_right = normalize_texts(right["business_name"]).tolist()
    address_left = normalize_texts(left["business_address"]).tolist()
    address_right = normalize_texts(right["business_address"]).tolist()
    country_left = clean_values(left["country"]).str.lower().tolist()
    country_right = clean_values(right["country"]).str.lower().tolist()

    name_scores = process.cpdist(
        name_left,
        name_right,
        scorer=fuzz.token_sort_ratio,
        workers=-1,
        dtype=np.float32,
    ) / 100.0
    address_scores = process.cpdist(
        address_left,
        address_right,
        scorer=fuzz.token_sort_ratio,
        workers=-1,
        dtype=np.float32,
    ) / 100.0
    name_scores[(np.asarray(name_left) == "") | (np.asarray(name_right) == "")] = 0.0
    address_scores[(np.asarray(address_left) == "") | (np.asarray(address_right) == "")] = 0.0
    same_country = np.asarray(
        [int(bool(a) and a == b) for a, b in zip(country_left, country_right)],
        dtype=np.float32,
    )

    features = pd.DataFrame(
        {
            "name_similarity": name_scores,
            "address_similarity": address_scores,
            "is_same_country": same_country,
        },
        columns=FEATURE_COLUMNS,
    )
    probabilities = np.mean(
        [model.predict_proba(features)[:, 1] for model in models], axis=0
    )
    probability_bins[:] += np.histogram(
        probabilities, bins=np.linspace(0.0, 1.0, 11)
    )[0]

    for (source_id, target_id), probability in zip(pair_chunk, probabilities):
        if probability >= THRESHOLD:
            matched.setdefault(source_id, []).append(target_id)


def find_resource_root(script_dir=None):
    script_dir = Path(script_dir or __file__).resolve()
    if script_dir.is_file():
        script_dir = script_dir.parent
    for parent in (script_dir, *script_dir.parents):
        if (parent / "output").is_dir() and (
            (parent / "dataset").is_dir()
            or (parent / "code" / "business_entity_resolution").is_dir()
        ):
            return parent
    raise FileNotFoundError("Could not locate the resource root containing output/ and code/.")


def main():
    script_dir = Path(__file__).resolve().parent
    resource_root = find_resource_root(script_dir)
    workspace_root = resource_root.parent
    output_root = resource_root / "output"
    model_path = workspace_root / "output" / "lgb_model.pkl"
    candidate_path = output_root / "candidate_pairs.tsv"
    matching_path = output_root / "matching_results.tsv"

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate file not found: {candidate_path}")

    source1_path, source2_path, source3_path = find_test_files(workspace_root)
    print(f"Test Source 1: {source1_path}")
    print(f"Test Source 2: {source2_path}")
    print(f"Test Source 3: {source3_path}")

    artifact = joblib.load(model_path)
    models = artifact["models"]
    print(f"Loaded {len(models)} fold models; using threshold {THRESHOLD:.2f}")

    source1 = pd.read_csv(source1_path, sep="\t", dtype=str).fillna("").set_index("entity_id")
    targets = pd.concat(
        [
            pd.read_csv(source2_path, sep="\t", dtype=str),
            pd.read_csv(source3_path, sep="\t", dtype=str),
        ],
        ignore_index=True,
    ).fillna("").set_index("entity_id")

    matched = {}
    probability_bins = np.zeros(10, dtype=np.int64)
    pair_chunk = []
    chunk_number = 0
    pair_count = 0

    with candidate_path.open("r", encoding="utf-8", newline="") as candidate_file:
        reader = csv.DictReader(candidate_file, delimiter="\t")
        expected_columns = {"source1_entity_id", "candidate_entity_ids"}
        if not expected_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected candidate header: {reader.fieldnames}")

        for row in reader:
            source_id = row["source1_entity_id"].strip()
            candidate_ids = row["candidate_entity_ids"] or ""
            for value in candidate_ids.split(","):
                target_id = value.strip()
                if target_id.startswith(("S2-", "S3-")):
                    pair_chunk.append((source_id, target_id))
                    pair_count += 1
                    if len(pair_chunk) == CHUNK_SIZE:
                        score_chunk(pair_chunk, source1, targets, models, matched, probability_bins)
                        pair_chunk.clear()
                        chunk_number += 1
                        if chunk_number % 10 == 0:
                            print(f"Scored {pair_count:,} candidate pairs ({chunk_number} chunks)")

    if pair_chunk:
        score_chunk(pair_chunk, source1, targets, models, matched, probability_bins)
        chunk_number += 1

    print(f"Total candidate pairs scored: {pair_count:,}")
    print("Probability distribution:")
    for index, count in enumerate(probability_bins):
        lower = index / 10
        upper = (index + 1) / 10
        closing = "]" if index == 9 else ")"
        print(f"  [{lower:.1f}, {upper:.1f}{closing}: {count:,}")

    rows_with_matches = 0
    output_root.mkdir(parents=True, exist_ok=True)
    with matching_path.open("w", encoding="utf-8", newline="") as matching_file:
        writer = csv.writer(matching_file, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for source_id in source1.index:
            target_ids = list(dict.fromkeys(matched.get(source_id, [])))
            rows_with_matches += bool(target_ids)
            writer.writerow([source_id, ",".join(target_ids)])

    total_matches = sum(len(ids) for ids in matched.values())
    print(f"Matches at threshold {THRESHOLD:.2f}: {total_matches:,}")
    print(f"Entities with matches: {rows_with_matches:,}")
    print(f"Singletons: {len(source1) - rows_with_matches:,}")
    print(f"Wrote {matching_path}")
    print(
        "python utils/validate_submission.py --matching output/matching_results.tsv "
        "--candidate output/candidate_pairs.tsv --test-dir dataset/test"
    )


if __name__ == "__main__":
    def find_resource_root(script_dir=None):
        script_dir = Path(script_dir or __file__).resolve()
        if script_dir.is_file():
            script_dir = script_dir.parent
        for parent in (script_dir, *script_dir.parents):
            if (parent / "output").is_dir() and (
                (parent / "dataset").is_dir()
                or (parent / "code" / "business_entity_resolution").is_dir()
            ):
                return parent
        raise FileNotFoundError("Could not locate the resource root containing output/ and code/.")


    def main():
        resource_root = find_resource_root()
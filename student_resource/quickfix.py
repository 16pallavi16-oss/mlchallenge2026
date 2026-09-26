import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


FEATURE_COLUMNS = ["name_similarity", "address_similarity", "is_same_country"]
CHUNK_SIZE = 2_048


def find_dataset_root(resource_root):
    candidates = [resource_root / "dataset", resource_root / "student_resource" / "dataset"]
    for path in candidates:
        if (path / "test" / "test_source1.tsv").is_file():
            return path
    raise FileNotFoundError("Could not locate test_source1.tsv under student_resource.")


def clean_name(value):
    if pd.isna(value):
        return ""
    import re

    value = re.sub(r"[^a-z0-9\s]", "", str(value).lower())
    value = re.sub(
        r"\b(pvt|ltd|limited|corp|corporation|inc|llc|the|and|of|sas|sarl|sa|sasu|eurl|snc)\b",
        "",
        value,
    )
    return " ".join(value.split())


def clean_names(values):
    return (
        values.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.replace(
            r"\b(pvt|ltd|limited|corp|corporation|inc|llc|the|and|of|sas|sarl|sa|sasu|eurl|snc)\b",
            "",
            regex=True,
        )
        .str.split()
        .str.join(" ")
    )


def text_value(value):
    return "" if pd.isna(value) else str(value).strip()


def score_source_batch(source_batch, target_data, models, threshold, matched, histogram):
    target_ids, target_names, target_addresses, target_countries = target_data
    query_names = [text_value(value).lower() for value in source_batch["business_name"]]
    query_addresses = [text_value(value).lower() for value in source_batch["business_address"]]
    query_countries = [text_value(value).lower() for value in source_batch["country"]]

    name_scores = process.cdist(
        query_names, target_names, scorer=fuzz.token_sort_ratio, workers=1
    ).astype(np.float32) / 100.0
    address_scores = process.cdist(
        query_addresses, target_addresses, scorer=fuzz.token_sort_ratio, workers=1
    ).astype(np.float32) / 100.0
    for index, value in enumerate(query_names):
        if not value:
            name_scores[index, :] = 0.0
    for index, value in enumerate(target_names):
        if not value:
            name_scores[:, index] = 0.0
    for index, value in enumerate(query_addresses):
        if not value:
            address_scores[index, :] = 0.0
    for index, value in enumerate(target_addresses):
        if not value:
            address_scores[:, index] = 0.0

    country_matches = np.asarray(
        [
            [bool(left) and left == right.lower() for right in target_countries]
            for left in query_countries
        ],
        dtype=np.float32,
    )
    feature_frame = pd.DataFrame(
        {
            "name_similarity": name_scores.ravel(),
            "address_similarity": address_scores.ravel(),
            "is_same_country": country_matches.ravel(),
        },
        columns=FEATURE_COLUMNS,
    )
    probabilities = np.mean(
        [model.predict_proba(feature_frame)[:, 1] for model in models], axis=0
    )
    histogram += np.histogram(probabilities, bins=np.linspace(0.0, 1.0, 11))[0]
    positive_indices = np.flatnonzero(probabilities >= threshold)
    query_count = len(source_batch)
    target_count = len(target_ids)
    for flat_index in positive_indices:
        query_index, target_index = divmod(int(flat_index), target_count)
        entity_id = source_batch.iloc[query_index]["entity_id"]
        matched.setdefault(entity_id, []).append(target_ids[target_index])


def main():
    resource_root = Path(__file__).resolve().parent
    workspace_root = resource_root.parent
    output_root = resource_root / "output"
    model_path = workspace_root / "output" / "lgb_model.pkl"
    dataset_root = find_dataset_root(resource_root)
    test_root = dataset_root / "test"

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    artifact = joblib.load(model_path)
    models = artifact["models"]
    threshold = 0.50
    print(
        f"Loaded {len(models)} fold models; saved OOF threshold: "
        f"{float(artifact['threshold']):.2f}; using threshold: {threshold:.2f}"
    )

    source1 = pd.read_csv(test_root / "test_source1.tsv", sep="\t", dtype=str)
    source2 = pd.read_csv(test_root / "test_source2.tsv", sep="\t", dtype=str)
    source3 = pd.read_csv(test_root / "test_source3.tsv", sep="\t", dtype=str)
    source1 = source1.fillna("")
    targets = pd.concat([source2, source3], ignore_index=True).fillna("")

    for frame in (source1, targets):
        frame["clean_name"] = clean_names(frame["business_name"])
        frame["block_key"] = (
            frame["country"].astype(str).str.lower() + "_" + frame["clean_name"].str[:6]
        )

    target_counts = targets["block_key"].value_counts()
    allowed_keys = target_counts[target_counts < 500].index
    target_groups = {}
    eligible_targets = targets[targets["block_key"].isin(allowed_keys)]
    for block_key, group in eligible_targets.groupby("block_key", sort=False):
        target_groups[block_key] = (
            group["entity_id"].tolist(),
            [text_value(value).lower() for value in group["business_name"]],
            [text_value(value).lower() for value in group["business_address"]],
            [text_value(value) for value in group["country"]],
        )

    output_root.mkdir(parents=True, exist_ok=True)
    candidates_path = output_root / "candidate_pairs.tsv"
    matching_path = output_root / "matching_results.tsv"
    matched = {}
    histogram = np.zeros(10, dtype=np.int64)
    pair_count = 0

    with candidates_path.open("w", encoding="utf-8", newline="") as candidate_file:
        candidate_writer = csv.writer(candidate_file, delimiter="\t")
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for source1_index, row in enumerate(source1.itertuples(index=False)):
            target_data = target_groups.get(row.block_key)
            target_ids = target_data[0] if target_data else []
            candidate_writer.writerow([row.entity_id, ",".join(target_ids)])
            pair_count += len(target_ids)

            if source1_index % 100_000 == 0:
                print(f"Processed {source1_index + 1:,} / {len(source1):,} Source-1 rows")

    processed_entities = 0
    for block_key, source_group in source1.groupby("block_key", sort=False):
        target_data = target_groups.get(block_key)
        if not target_data:
            continue
        for start in range(0, len(source_group), CHUNK_SIZE):
            source_batch = source_group.iloc[start : start + CHUNK_SIZE]
            score_source_batch(
                source_batch, target_data, models, threshold, matched, histogram
            )
            processed_entities += len(source_batch)
            if processed_entities // 100_000 > (processed_entities - len(source_batch)) // 100_000:
                print(f"Scored {processed_entities:,} Source-1 rows")

    print(f"Scored candidate pairs: {pair_count:,}")
    print("Probability distribution of scored pairs:")
    for index, count in enumerate(histogram):
        print(f"  [{index / 10:.1f}, {(index + 1) / 10:.1f}{']' if index == 9 else ')'}: {count:,}")
    print(f"Matches at threshold {threshold:.2f}: {sum(map(len, matched.values())):,}")

    with matching_path.open("w", encoding="utf-8", newline="") as matching_file:
        matching_writer = csv.writer(matching_file, delimiter="\t")
        matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for entity_id in source1["entity_id"]:
            matching_writer.writerow([entity_id, ",".join(dict.fromkeys(matched.get(entity_id, [])))])

    print(f"Wrote {candidates_path}")
    print(f"Wrote {matching_path}")
    print(
        "python utils/validate_submission.py --matching output/matching_results.tsv "
        "--candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test"
    )


if __name__ == "__main__":
    main()
import argparse
import os
import re

import pandas as pd
from rapidfuzz import fuzz

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "student_resource", "student_resource", "dataset")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")


def clean_text(value):
    if pd.isna(value):
        return ""
    value = re.sub(r"[^a-z0-9\s]", "", str(value).lower())
    value = re.sub(r"\b(pvt|ltd|limited|corp|corporation|inc|llc|the|and|of)\b", "", value)
    return " ".join(value.split())


def load_test_data(test_dir):
    source1 = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t")
    source2 = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t")
    source3 = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t")
    return source1, pd.concat([source2, source3], ignore_index=True)


def build_candidates(source1, targets):
    for frame in (source1, targets):
        frame["clean_name"] = frame["business_name"].apply(clean_text)
        frame["block_key"] = (
            frame["country"].astype(str).str.lower() + "_" + frame["clean_name"].str[:6]
        )

    counts = targets["block_key"].value_counts()
    valid_keys = set(counts[counts < 500].index)
    grouped = targets[targets["block_key"].isin(valid_keys)].groupby("block_key")
    target_ids = grouped["entity_id"].apply(list).to_dict()

    candidates = {}
    for row in source1.itertuples(index=False):
        candidates[row.entity_id] = target_ids.get(row.block_key, [])
    return candidates


def choose_matches(source1, targets, candidates):
    targets = targets.set_index("entity_id")
    matches = {}
    for row in source1.itertuples(index=False):
        selected = []
        for target_id in candidates[row.entity_id]:
            target = targets.loc[target_id]
            if str(row.country).lower() != str(target.country).lower():
                continue
            name_score = fuzz.token_sort_ratio(
                str(row.business_name).lower(), str(target.business_name).lower()
            )
            address_score = fuzz.token_sort_ratio(
                str(row.business_address).lower(), str(target.business_address).lower()
            )
            if name_score >= 85 or (name_score >= 70 and address_score >= 70):
                selected.append(target_id)
        matches[row.entity_id] = selected
    return matches


def write_output(output_dir, source1, candidates, matches):
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(
        {
            "source1_entity_id": source1["entity_id"],
            "candidate_entity_ids": [
                ",".join(candidates[entity_id]) for entity_id in source1["entity_id"]
            ],
        }
    ).to_csv(os.path.join(output_dir, "candidate_pairs.tsv"), sep="\t", index=False)
    pd.DataFrame(
        {
            "source1_entity_id": source1["entity_id"],
            "matched_entity_ids": [
                ",".join(matches[entity_id]) for entity_id in source1["entity_id"]
            ],
        }
    ).to_csv(os.path.join(output_dir, "matching_results.tsv"), sep="\t", index=False)


def find_default_test_dir():
    candidates = [
        os.path.join(PROJECT_ROOT, "student_resource", "dataset", "test"),
        os.path.join(PROJECT_ROOT, "student_resource", "student_resource", "dataset", "test"),
        os.path.join(PROJECT_ROOT, "dataset", "test"),
    ]
    for p in candidates:
        if os.path.isfile(os.path.join(p, "test_source1.tsv")):
            return p
    return os.path.join(DATA_ROOT, "test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default=find_default_test_dir())
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "output"))
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size for testing")
    args = parser.parse_args()

    source1, targets = load_test_data(args.test_dir)
    if args.sample_size:
        source1 = source1.iloc[:args.sample_size].copy()
        print(f"Sample mode enabled: evaluating first {len(source1):,} Source-1 entities.")
    candidates = build_candidates(source1, targets)
    matches = choose_matches(source1, targets, candidates)
    write_output(args.output_dir, source1, candidates, matches)
    print(f"Generated outputs for {len(source1)} Source-1 entities in {args.output_dir}.")


if __name__ == "__main__":
    main()
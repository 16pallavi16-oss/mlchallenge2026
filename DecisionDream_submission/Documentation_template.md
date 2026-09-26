# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** DecisionDream 
**Team Members:** Pallavi VR , Sughosha S Vasista , Rajan L  
**Submission Date:** September 26, 2026

---

## 1. Executive Summary

We built a two-stage Business Entity Resolution pipeline using name-based blocking followed by a LightGBM binary classifier. The classifier uses business-name similarity, address similarity, and country equality, and is trained as a five-fold ensemble. The test candidate and match files were generated and passed the official submission validator.

---

## 2. Methodology

### 2.1 Problem Analysis

- The test Source-1 file contains 1,732,544 entities.
- Business names and addresses may vary in punctuation, abbreviations, word order, and completeness.
- Country is treated as an open-set string label; the pipeline does not restrict countries to a fixed list.
- The generated training feature file contains 5,664,887 pairs: 127,965 positive and 5,536,922 negative examples, approximately 43.27 negatives per positive.
- In the inspected training feature file, `is_same_country` was 1 for every row, so it had no variation in that sample.

### 2.2 Solution Strategy

**Approach Type:** Blocking + binary classifier

**Core Approach:** Candidate blocks combine country with the first six characters of a normalized business name. Blocks with 500 or more target records are discarded to control memory use. Candidate pairs are then scored using string-similarity features and a LightGBM ensemble.

---

## 3. Candidate Generation (Blocking)

Business names are lowercased, punctuation is removed, and common legal/company terms (`pvt`, `ltd`, `limited`, `corp`, `corporation`, `inc`, `llc`, `the`, `and`, `of`, `sas`, `sarl`, `sa`, `sasu`, `eurl`, and `snc`) are removed. The blocking key is the lowercased country plus the first six characters of the cleaned name. Source-2 and Source-3 records are pooled as targets. Target blocks containing 500 or more records are excluded.

Observed test candidate output:

- **Test Source-1 entities:** 1,732,544
- **Rows with candidates:** 878,661
- **Rows with no candidates:** 853,883
- **Candidate pairs scored:** 92,622,524

The block-frequency cap is a scale/memory trade-off. True matches in excluded high-frequency blocks may not be retrieved, so blocking does not guarantee perfect recall.

---

## 4. Matching Model

### Features Used

| Feature | Method | Description |
|---|---|---|
| `name_similarity` | RapidFuzz `token_sort_ratio`, scaled to 0–1 for model input | Fuzzy business-name similarity, tolerant of token order |
| `address_similarity` | RapidFuzz `token_sort_ratio`, scaled to 0–1 for model input | Fuzzy address-string similarity |
| `is_same_country` | Case-insensitive exact string equality | Binary country agreement; no country list is hardcoded |

### Model Architecture

- **Model:** LightGBM binary classifier; five fold models are retained as an ensemble.
- **Validation:** 5-fold `StratifiedKFold`, shuffled with random seed 42.
- **Class imbalance:** `scale_pos_weight` is calculated as the number of negative examples divided by the number of positive examples (about 43.27 for the inspected training file).
- **Parameters in `model.py`:** 300 estimators, learning rate 0.05, and 31 leaves.
- **OOF threshold search:** 0.20 through 0.95 in 0.01 increments, evaluated on out-of-fold positive-class probabilities.
- **Best observed OOF threshold:** 0.95.
- **Best observed OOF F0.5:** 0.7675. This is a local OOF score, not a leaderboard score.

The test prediction run documented here used threshold **0.90**, separately from the OOF-selected threshold. The `repredict.py` script applies 0.90 to averaged positive-class probabilities.

### Test Probability Distribution

The following candidate-level probability distribution was printed during test scoring. It describes pair probabilities; it is not a count of unique matched entities.

| Probability interval | Candidate pairs |
|---|---:|
| [0.0, 0.1) | 80,952,358 |
| [0.1, 0.2) | 3,146,518 |
| [0.2, 0.3) | 1,920,775 |
| [0.3, 0.4) | 889,924 |
| [0.4, 0.5) | 556,190 |
| [0.5, 0.6) | 481,178 |
| [0.6, 0.7) | 339,260 |
| [0.7, 0.8) | 404,390 |
| [0.8, 0.9) | 367,908 |
| [0.9, 1.0] | 3,564,023 |

---

## 5. Results & Error Analysis

- **OOF F0.5:** 0.7675 at threshold 0.95.
- **Leaderboard F0.5:** Not provided.
- **Validated test output:** 1,732,544 rows; 859,241 entities with matches and 873,303 singletons in the validated matching file.
- **Validator:** `PASS — no blocking issues found. Safe to submit.`

The test set has no labels available here, so test false-positive and false-negative counts are unknown. Likely error sources include similar short names, address variation, transliteration, and true matches excluded by the block-frequency cap.

---

## 6. Conclusion

The pipeline reduces comparisons with deterministic country/name blocking and applies a five-fold LightGBM classifier to the retained pairs. The generated test outputs contain one row per Source-1 entity and passed the official format validator. The OOF metric is known, but no leaderboard score was supplied.

---

## Appendix

### A. Code Artefacts and Reproduction

The training code is under `code/business_entity_resolution/src/` (`blocking.py`, `features.py`, and `model.py`). Test candidate generation and inference helpers are `quickfix.py` and `repredict.py` under `student_resource/`. In this workspace, test data is nested at `student_resource/student_resource/dataset/`; submission outputs are in `student_resource/output/`.

Run from the workspace root with the virtual environment:

```powershell
.\.venv\Scripts\python.exe .\student_resource\code\business_entity_resolution\src\blocking.py
.\.venv\Scripts\python.exe .\student_resource\code\business_entity_resolution\src\features.py
.\.venv\Scripts\python.exe .\student_resource\code\business_entity_resolution\src\model.py
.\.venv\Scripts\python.exe .\student_resource\quickfix.py
.\.venv\Scripts\python.exe .\student_resource\repredict.py
.\.venv\Scripts\python.exe .\student_resource\utils\validate_submission.py --matching .\student_resource\output\matching_results.tsv --candidate .\student_resource\output\candidate_pairs.tsv --test-dir .\student_resource\student_resource\dataset\test
```

The first three scripts generate training data and train or load the model. `quickfix.py` generates test candidates; `repredict.py` generates test matches at threshold 0.90. The training scripts also use a workspace-level `output/` directory, so confirm the final submission files are in `student_resource/output/` before packaging.

### B. Dependencies

The project requirements include pandas, NumPy, LightGBM, RapidFuzz, scikit-learn, and joblib. The model bundle was loaded from the workspace `.venv` environment.

### C. Hardware

- **Operating system:** Windows
- **RAM:** 16GB 
- **GPU:** No GPU usage was reported

---
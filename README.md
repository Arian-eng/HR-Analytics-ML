# HR Analytics ML — Chapter 3 aligned pipeline (v2)

This repository contains the executed analysis code and published outputs for the four datasets used in the thesis. The implementation follows the documented 80/20 split, limited hyperparameter search, 3-fold stratified F1 tuning for classification, 5-fold RMSE tuning for GHRM regression, and numeric-only K-Means inputs for the three general HR datasets.

The repository is organized so that a reviewer can trace an analytical statement from the method to the numerical output and then to its visual interpretation.

## Reviewer quick path

1. [`docs/formulas.md`](docs/formulas.md) — equations and metric definitions.
2. [`results/MODEL_RESULTS.md`](results/MODEL_RESULTS.md) — readable model-result tables.
3. [`figures/README.md`](figures/README.md) — explanation of every chart, confusion matrix, K-Means plot and decision-tree figure.
4. [`docs/decision_trees.md`](docs/decision_trees.md) — fitted tree structures and interpretation.
5. [`results/PATTERNS_AND_FINDINGS.md`](results/PATTERNS_AND_FINDINGS.md) — table of patterns obtained from the analyses.
6. [`docs/conclusion.md`](docs/conclusion.md) — evidence-based conclusion and limitations.
7. [`docs/DEFENSE_QA.md`](docs/DEFENSE_QA.md) — concise answers to reviewer-sensitive questions about sample size, imbalance, tree complexity, GEE, reliability and dataset scope.
8. [`docs/thesis_mapping.md`](docs/thesis_mapping.md) — thesis-to-code/result/figure mapping.
9. [`validation/validation_report.md`](validation/validation_report.md) — independent cross-check against the existing Chapter 4 text.
10. `results/*.json` and `src/*.py` — machine-readable outputs and implementation details.

## The four datasets

| # | File | Role | n | Target |
|---|---|---|---:|---|
| 1 | `HRM_DATASETS.csv` | **Primary — direct GHRM measure** | 320 | FEP (continuous) |
| 2 | `WA_Fn-UseC_-HR-Employee-Attrition__3_.csv` | General HR benchmark | 1,470 | Attrition |
| 3 | `aug_train.csv` | General HR benchmark | 19,158 | target |
| 4 | `train_LZdllcl.csv` | General HR benchmark | 54,808 | is_promoted |

Only dataset 1 directly measures the Green HRM constructs used in the thesis. Datasets 2–4 are general-purpose HR benchmarks. They are modeled independently, and any connection to Green HRM is limited to methodological or interpretive discussion rather than direct measurement.

The expected local data layout and filenames are documented in [`data/README.md`](data/README.md). The four datasets are not merged record-by-record.

## What each script does

- **`src/generate_chapter4_charts.py`** generates the nine descriptive and summary figures referenced in Chapter 4: IBM age and income distributions, target/status distributions, train/test class balance, missing-value summaries, and cross-dataset K-Means/F1 comparisons.
- **`src/run_ghrm_reliability.py`** calculates Cronbach's alpha, construct descriptives and the construct-level correlation matrix. Five constructs have alpha above 0.70; FEP is 0.6043 and is retained as a limitation.
- **`src/run_ghrm.py`** runs the GHRM survey regression with a fixed 256/64 split, Base and GEE+ specifications, four regressors, 5-fold RMSE tuning, held-out R²/RMSE/MAE, 4,000-resample paired bootstrap comparisons, and K-Means on the five standardized GHRM practice scores.
- **`src/run_ibm.py` and `src/run_jobchange.py`** run Decision Tree, Random Forest, LinearSVC and MLP classification with 3-fold stratified F1 tuning, held-out metrics with bootstrap intervals, confusion matrices, McNemar pairwise tests, decision-tree plots, and numeric-only K-Means.
- **`src/run_promotion.py`** applies the same classification/clustering logic to the 54,808-row Promotion dataset. Its staged helper scripts separate the larger computations while preserving the same analysis design.

## Published outputs

The committed `results/*.json` files are the authoritative numerical outputs of the executed analysis. Reviewer-friendly tables are also provided under `results/tables/` and in `results/MODEL_RESULTS.md`.

Five directly viewable SVG summary figures are committed under `figures/`. The final analysis archive also contains 27 original PNG artifacts covering the Chapter 4 charts, confusion matrices, decision trees, K-Means criteria/PCA views, GHRM correlations, Base-vs-GEE comparison and the GEE bootstrap summary. Their filenames, byte sizes and SHA-256 digests are recorded in `figures/MANIFEST.md`; `figures/README.md` explains what each one is intended to show and what can, and cannot, be inferred from it.

## Reproducibility

```bash
pip install -r requirements.txt
# place the four CSV files under data/ using the exact filenames documented in data/README.md
python src/run_ghrm.py
python src/run_ibm.py
python src/run_jobchange.py
python src/run_promotion.py
```

All random seeds used by the current pipeline are fixed at 42. Paths are resolved relative to the repository root; no machine-specific `/home/...` path is required.

The GitHub Actions integrity workflow performs data-free checks on source syntax, published JSON structure, the fixed GHRM 320/256/64 sample sizes, and repository portability. The raw datasets are intentionally not committed.

`requirements.txt` uses compatible version ranges. The exact package snapshot of the original execution environment was not captured, so the repository does not claim bit-for-bit reproduction across arbitrary future library versions. The committed JSON outputs are the reference results for this repository.

## Main findings in brief

- IBM Attrition: MLP has the highest held-out F1 in the committed run (0.5000); Random Forest demonstrates why high accuracy can be misleading when positive-class recall is low.
- Job Change: Random Forest has the highest F1 (0.6042); the selected K-Means solution has k=3 and Silhouette=0.5773.
- Employee Promotion: MLP has the highest F1 (0.5054); the unrestricted Decision Tree reaches depth 63 and 4,855 leaves, which is reported as an overfitting risk.
- GHRM: the best Base held-out R² is 0.4374. MLP shows the largest positive point change after GEE is added, but its paired 95% bootstrap interval includes zero. GHRM K-Means selects two clusters with mean FEP values 3.851 and 3.041.
- Reliability: GRS, GTD, GPA, GCM and GEE have alpha above 0.70; FEP has alpha=0.6043 and is treated as a limitation.

Detailed interpretation is kept in `results/PATTERNS_AND_FINDINGS.md` and `docs/conclusion.md` rather than being inferred from a single metric.

## Project structure

```text
HR-Analytics-ML/
├── data/
│   └── README.md          # expected raw filenames; raw CSVs are not committed
├── src/                   # analysis scripts and shared utilities
├── results/
│   ├── *.json             # authoritative machine-readable outputs
│   ├── MODEL_RESULTS.md   # readable performance tables
│   ├── PATTERNS_AND_FINDINGS.md
│   └── tables/            # compact CSV summary tables
├── figures/
│   ├── summary_*.svg      # directly viewable numerical summary figures
│   ├── README.md          # figure-by-figure scientific explanation
│   └── MANIFEST.md        # expected PNG names, sizes and SHA-256 hashes
├── docs/
│   ├── formulas.md
│   ├── decision_trees.md
│   ├── DEFENSE_QA.md
│   ├── thesis_mapping.md
│   ├── FINAL_AUDIT.md
│   └── conclusion.md
├── validation/
│   └── validation_report.md
├── KEY_FINDINGS.md
├── requirements.txt
├── tests/
├── .github/workflows/
└── README.md
```

# HR Analytics ML — Chapter 3 aligned pipeline (v2)

Real, executed analysis code for the four datasets used in the thesis,
implementing Chapter 3's (روش پژوهش) exact specification — including its
80/20 split, its limited hyperparameter grid search (3-fold CV for
classification, 5-fold for regression), and K-Means's numeric-only feature
rule for the three general datasets. See `validation/validation_report.md`
for a full line-by-line cross-check of this run's numbers against the
thesis's existing Chapter 4 text — every K-Means result matches exactly,
which is strong evidence Chapter 4 is real and reproducible.

## The four datasets

| # | File | Role | n | Target |
|---|---|---|---:|---|
| 1 | `HRM_DATASETS.csv` | **Primary — direct GHRM measure** | 320 | FEP (continuous) |
| 2 | `WA_Fn-UseC_-HR-Employee-Attrition__3_.csv` | General HR benchmark | 1,470 | Attrition |
| 3 | `aug_train.csv` | General HR benchmark | 19,158 | target |
| 4 | `train_LZdllcl.csv` | General HR benchmark | 54,808 | is_promoted |

Only dataset 1 directly measures Green HRM constructs. Datasets 2-4 are
general-purpose public HR benchmarks; any connection to Green HRM concepts
is drawn only at the interpretation stage, per Chapter 1/3.

## What each script does

- **`src/generate_chapter4_charts.py`** — the 9 descriptive/summary
  figures Chapter 4 explicitly references by number (نمودار 1-4 through
  9-4: age histogram, income boxplot, target/status distributions, the
  80/20 split's preserved class balance, before/after missing-value
  completeness, and the two cross-dataset summary charts). These were
  requested explicitly by the advisor's original feedback ("even the
  schematic shape... can be included") and were missing from earlier
  commits, which only had the modeling-result figures (trees, confusion
  matrices, K-Means criteria).
- **`src/run_ghrm_reliability.py`** — Cronbach's alpha per construct,
  composite-score descriptives (cross-checked exactly against Chapter 4's
  Table 4-4), and a construct-level correlation matrix. This is genuinely
  new analysis — Chapter 4 reports descriptives but not its own reliability
  check on this study's sample — added because a committee member checking
  reliability independently would likely ask for it. Flags that the FEP
  construct's alpha (0.604) falls below the conventional 0.70 threshold.
- **`src/run_ghrm.py`** — GHRM survey regression: fixed 80/20 split (256/64,
  `random_state=42`); Base model (GRS+GTD+GPA+GCM→FEP) and GEE+ model each
  tuned via a limited grid search (5-fold CV, RMSE) over RandomForestRegressor,
  DecisionTreeRegressor, LinearSVR, and MLPRegressor; R²/RMSE/MAE on the
  same 64 held-out records; Base-vs-GEE+ compared via a paired bootstrap
  (4,000 resamples, seed=42); K-Means on standardized GRS/GTD/GPA/GCM/GEE
  reporting SSE, Silhouette, and Davies-Bouldin together (k=2..7).
- **`src/run_ibm.py`, `run_jobchange.py`** — for IBM Attrition and Job
  Change: 80/20 split; Decision Tree, Random Forest, LinearSVC, and
  MLPClassifier each tuned via a limited grid search (3-fold Stratified CV,
  F1 scoring — see `ch3_utils.py: GRIDS_CLASSIFICATION` for the exact grid);
  Accuracy/Precision/Recall/F1 each with a 95% bootstrap CI (2,000
  resamples); confusion matrices; a decision-tree structure plot; a McNemar
  pairwise test for all six model pairs; and a separate K-Means run on
  **numeric features only** (Chapter 3's explicit rule for these three
  datasets), reporting SSE/Silhouette/Davies-Bouldin (k=2..7).
- **`src/run_promotion.py`** — same pipeline as above for the largest
  dataset (54,808 rows), split into staged sub-scripts
  (`_promotion_step1_per_model.py` etc.) purely to fit a single-CPU
  runtime budget; `run_promotion.py` runs the full sequence and produces
  identical output structure to the other two.

The committed `results/*.json` files are the authoritative numerical outputs
of the executed analysis. The supplied final analysis archive also contains
27 generated PNG artifacts. Their exact filenames, byte sizes, and SHA-256
digests are recorded in `figures/MANIFEST.md`, and the scripts in `src/`
document how those figures are produced from the same data/results. No model
number or thesis finding is inferred from a figure file alone.

## Compute-budget note

Single-CPU environment. Grid search grids are intentionally small
("limited search," per Chapter 3's own wording) rather than exhaustive —
e.g. Random Forest tries at most 6 combinations, not every possible one.
This is documented per-dataset in `results/*.json` under `grids_searched`,
and its effect on how closely results match Chapter 4's own numbers is
discussed candidly in the validation report rather than hidden.

## Reproducibility

```bash
pip install -r requirements.txt
# place the 4 CSVs under data/ using the exact filenames in the table above
python src/run_ghrm.py
python src/run_ibm.py
python src/run_jobchange.py
python src/run_promotion.py
```
All random seeds are fixed at 42 throughout.

## Portability and repository checks

All scripts resolve `data/`, `results/`, and `figures/` relative to the repository root; no machine-specific absolute path is required. A lightweight GitHub Actions workflow compiles the source and validates the published result structure without rerunning the thesis models (the raw datasets are intentionally not committed).

`requirements.txt` uses compatible version ranges rather than pretending to know the exact package versions of the original execution environment; that original environment snapshot was not captured. The committed JSON result files remain the authoritative published outputs of this repository.

## Chapter 5 (discussion/conclusion)

Chapter 5 has no numbered figures of its own — it is a text discussion,
hypothesis-evaluation table, literature comparison, and limitations/future-
work list. `KEY_FINDINGS.md` (English) synthesizes the same patterns
Chapter 5 discusses, grounded in this repo's own verified numbers rather
than restating Chapter 5's text. The final archive also contains
`chart10_gee_effect_reliability.png`, which visualizes whether GEE's apparent
benefit to each regressor is statistically reliable under the paired
bootstrap analysis; its exact binary identity is listed in
`figures/MANIFEST.md`.

## Project structure

```text
HR-Analytics-ML/
├── data/              # raw CSVs (not committed — see .gitignore)
├── src/               # shared utilities + run_*.py per dataset
├── results/           # authoritative committed JSON outputs
├── figures/           # exact final-archive PNG manifest; figures generated by src/
├── validation/
│   └── validation_report.md
├── KEY_FINDINGS.md    # English synthesis of patterns across all 4 datasets
├── requirements.txt
├── tests/             # data-free repository integrity checks
├── .github/workflows/ # CI for syntax and repository integrity
├── .gitignore
└── README.md
```

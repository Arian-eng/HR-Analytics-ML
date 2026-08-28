# HR Analytics ML

Machine-learning and statistical analysis for the master's thesis on Green HRM and HR analytics.

## Scope — four datasets, two different roles

**1. Primary dataset — Green HRM survey (`HRM_DATASETS.csv`, n=320)**
This is the actual Green HRM measurement instrument (Likert 1-5 items grouped into
Green Recruitment & Selection (GRS), Green Training & Development (GTD), Green
Performance Appraisal (GPA), Green Compensation & Management (GCM), Firm
Environmental Performance (FEP), and Green Employee Engagement (GEE)). This is
the dataset that actually measures the thesis's core constructs, and is the
one the Chapter-4 "hidden patterns" narrative should be built around.

**2–4. Three general-purpose public HR datasets (benchmark / proxy analyses)**

| # | File | Target | n | Class balance |
|---|---|---|---:|---|
| 2 | `WA_Fn-UseC_-HR-Employee-Attrition__3_.csv` (IBM HR Attrition) | `Attrition` | 1,470 | 83.88% No / 16.12% Yes |
| 3 | `aug_train.csv` (HR Analytics: Job Change of Data Scientists) | `target` | 19,158 | 75.07% / 24.93% |
| 4 | `train_LZdllcl.csv` (Employee promotion, WNS/Analytics Vidhya) | `is_promoted` | 54,808 | 91.48% / 8.52% |

> **Methodological note (unchanged from the original scope):** these three
> general HR datasets do not directly measure Green HRM practices. Their
> outputs are HR-performance/behavioral proxy analyses — useful to
> demonstrate methodological competence (preprocessing, model comparison,
> validation) — not evidence about green HR practices. Only dataset 1 speaks
> directly to the thesis's Green HRM constructs. This gap was flagged by the
> thesis advisor and is not resolved by more modeling; it is a research-design
> question for the student and advisor.

## What each pipeline actually does

- **Survey dataset (`src/survey_analysis.py`):** Cronbach's alpha per
  construct, composite (mean) scores, a construct-level correlation matrix,
  OLS multiple regression (GRS+GTD+GPA+GCM → FEP, and → GEE), and K-Means
  respondent segmentation with silhouette-based k selection.
- **Three ML datasets (`src/run_attrition.py`, `run_jobchange.py`,
  `run_promotion.py`, sharing `src/ml_utils.py`):** median/most-frequent
  imputation + scaling/one-hot preprocessing, Decision Tree, Random Forest,
  RBF-kernel SVM, and an MLP neural network; stratified k-fold cross-validation;
  test-set accuracy/precision/recall/F1/ROC-AUC; confusion matrices; a
  decision-tree structure plot; permutation feature importance; and a separate
  K-Means segmentation of the feature space.

All numbers in `results/` and all charts in `figures/` were produced by
running these scripts against the four CSV files above — nothing in this
repository is hand-typed or estimated. See `validation/validation_report.md`
for the exact figures and how to reproduce them.

## Compute-budget note

This was run on a single-CPU machine. To keep runtime tractable:
- Random Forest uses 120 trees (not e.g. 500+) and cross-validation uses 3
  folds (not 5) for the two large datasets.
- The RBF-kernel SVM (which scales badly with n) is trained/cross-validated
  on a stratified 5,000-row subsample of the training set for the two large
  datasets; it is evaluated on the full held-out test set.
- Permutation importance uses 5 repeats on a capped test subsample (≤1,500
  rows) rather than the full test set.
These choices are recorded per-run in each `results/*_classification_report.json`
under `compute_settings`, so they are transparent and reproducible, not hidden.

## Project structure

```text
HR-Analytics-ML/
├── data/                  # Local datasets (not committed - see .gitignore)
├── src/                   # Analysis source code (see file list above)
├── results/               # Generated JSON/text results (real, committed)
├── figures/               # Generated charts (real, committed)
├── validation/
│   └── validation_report.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproducibility

```bash
pip install -r requirements.txt
# place the 4 CSV files under data/ using the exact filenames in the table above
python src/survey_analysis.py
python src/run_attrition.py
python src/run_jobchange.py
python src/run_promotion.py
```
Each script is self-contained and writes its own `results/*.json` and
`figures/*.png`. Random seeds are fixed (`random_state=42`) throughout.

## Data

Raw CSVs are not committed (see `.gitignore`) — three are public Kaggle/
Analytics-Vidhya datasets, and the survey file may contain the student's
own collected data. Filenames and targets are documented above and in the
analysis code.

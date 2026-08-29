# HR Analytics ML — Thesis Evidence Repository

This repository contains the executed analysis code, numerical outputs, tables, and **27 original PNG figures** used to document the thesis analyses. It is organized for reviewer traceability: **scope → method → code → result → table → figure → interpretation → limitation**.

> **Reviewer: start here → [`START_HERE.md`](START_HERE.md)**
>
> That page provides the shortest path to every important piece of evidence without searching the repository manually.

## Direct access

| Need | Open this |
|---|---|
| Fast reviewer dashboard | **[`START_HERE.md`](START_HERE.md)** |
| Main numerical results | **[`results/MODEL_RESULTS.md`](results/MODEL_RESULTS.md)** |
| All result tables + JSON outputs | **[`results/README.md`](results/README.md)** |
| All 27 PNG figures + interpretation | **[`figures/README.md`](figures/README.md)** |
| One explanation file per figure | **[`figures/descriptions/`](figures/descriptions/)** |
| Exact figure integrity hashes | **[`figures/MANIFEST.md`](figures/MANIFEST.md)** |
| Source-code execution map | **[`src/README.md`](src/README.md)** |
| Equations and metric definitions | **[`docs/formulas.md`](docs/formulas.md)** |
| Decision-tree depth/leaves | **[`docs/decision_trees.md`](docs/decision_trees.md)** |
| Thesis → code → result → figure mapping | **[`docs/thesis_mapping.md`](docs/thesis_mapping.md)** |
| Defense-sensitive Q&A | **[`docs/DEFENSE_QA.md`](docs/DEFENSE_QA.md)** |
| Validation vs thesis text | **[`validation/validation_report.md`](validation/validation_report.md)** |
| Final limitations/audit | **[`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md)** |

## Dataset roles

| # | Dataset | n | Role | Target |
|---|---|---:|---|---|
| 1 | `HRM_DATASETS.csv` | 320 | **Primary — direct Green HRM measurement** | FEP |
| 2 | IBM HR Analytics Attrition | 1,470 | General HR benchmark / methodological validation | Attrition |
| 3 | Job Change (`aug_train.csv`) | 19,158 | General HR benchmark / methodological validation | target |
| 4 | Employee Promotion (`train_LZdllcl.csv`) | 54,808 | General HR benchmark / methodological validation | is_promoted |

Only the 320-record GHRM survey directly contains the Green HRM constructs GRS, GTD, GPA, GCM, GEE and FEP. The three public HR datasets are analyzed independently as general HR benchmarks; they are **not** treated as direct Green HRM measurements and are not merged record-by-record with the GHRM survey.

Expected raw-data filenames and local placement are documented in [`data/README.md`](data/README.md).

## Analysis design

- Fixed random seed: **42**.
- Train/test split: **80/20** for all four analyses.
- Classification: Decision Tree, Random Forest, LinearSVC and MLP; limited hyperparameter search with **3-fold stratified CV using positive-class F1**.
- GHRM regression: Random Forest, Decision Tree, LinearSVR and MLP; **5-fold CV using RMSE** in the committed final pipeline.
- GHRM specifications: Base = GRS + GTD + GPA + GCM; GEE+ adds GEE.
- K-Means: candidates k=2..7; standardized inputs; selected by maximum Silhouette with SSE and Davies-Bouldin reported as supporting diagnostics.
- Reliability: Cronbach's alpha for GRS, GTD, GPA, GCM, GEE and FEP.
- Uncertainty: bootstrap intervals; classification comparisons also include McNemar tests.

For exact script responsibilities, see [`src/README.md`](src/README.md). For equations, see [`docs/formulas.md`](docs/formulas.md).

## Published evidence

### Numerical outputs

The committed `results/*.json` files are the machine-readable numerical source of truth for this repository. Reviewer-readable summaries are provided in [`results/MODEL_RESULTS.md`](results/MODEL_RESULTS.md) and compact CSV tables under [`results/tables/`](results/tables/).

### Figures

All **27 original PNG artifacts are committed under [`figures/`](figures/)**, including:

- Chapter 4 descriptive charts;
- confusion matrices;
- decision-tree plots;
- K-Means criteria plots;
- PCA cluster visualizations;
- GHRM correlation heatmap;
- Base-vs-GEE comparison;
- GEE effect / bootstrap summary.

Every PNG has a reviewer-facing explanation under [`figures/descriptions/`](figures/descriptions/). The consolidated clickable index is [`figures/README.md`](figures/README.md), and exact filenames, byte sizes and SHA-256 hashes are in [`figures/MANIFEST.md`](figures/MANIFEST.md).

## Main findings in brief

| Analysis | Main committed result |
|---|---|
| IBM Attrition | Best held-out F1: **MLP = 0.5000**; RF illustrates why high Accuracy can hide very low positive recall. |
| Job Change | Best held-out F1: **RF = 0.6042**; selected K-Means k=3, Silhouette=0.5773. |
| Employee Promotion | Best held-out F1: **MLP = 0.5054**; Decision Tree depth=63, leaves=4,855, reported as an overfitting/interpretability risk. |
| GHRM regression | Best Base held-out R²: **RF = 0.4374**. MLP has the largest positive GEE+ point change, but its paired 95% bootstrap interval includes zero. |
| GHRM clustering | k=2, Silhouette=0.4363, DB=0.8611; cluster mean FEP values 3.851 and 3.041. |
| Reliability | GRS/GTD/GPA/GCM/GEE alpha >0.70; **FEP alpha=0.6043**, explicitly treated as a limitation. |

Detailed interpretation is in [`results/PATTERNS_AND_FINDINGS.md`](results/PATTERNS_AND_FINDINGS.md) and [`docs/conclusion.md`](docs/conclusion.md).

## Reproducibility

```bash
pip install -r requirements.txt
# place the four CSV files under data/ using data/README.md
python src/run_ghrm.py
python src/run_ibm.py
python src/run_jobchange.py
python src/run_promotion.py
```

Paths are repository-relative; machine-specific `/home/...` paths are not required. A GitHub Actions integrity workflow performs data-free checks on syntax and published result structure.

The raw datasets are intentionally not committed. The exact original package snapshot was not captured, so this repository does **not** claim bit-for-bit reproduction across arbitrary future dependency versions. It does provide inspectable code, documented inputs, committed numerical outputs, tables, figures and validation evidence.

## Important interpretation boundaries

The repository intentionally does **not** claim that:

- the three public benchmark datasets directly measure Green HRM;
- correlation, clustering or predictive association establishes causality;
- Accuracy alone is sufficient for imbalanced classification;
- adding GEE significantly improves every GHRM model;
- the Promotion Decision Tree is preferable despite its extreme complexity;
- every result is identical to the thesis text where [`validation/validation_report.md`](validation/validation_report.md) explicitly records a difference.

## Repository map

```text
HR-Analytics-ML/
├── START_HERE.md           # reviewer evidence dashboard
├── README.md               # scope and top-level navigation
├── data/
│   └── README.md           # expected raw filenames and roles
├── src/
│   ├── README.md           # execution/code index
│   └── *.py                # analysis pipelines
├── results/
│   ├── README.md           # results index
│   ├── *.json              # authoritative machine-readable outputs
│   ├── MODEL_RESULTS.md
│   ├── PATTERNS_AND_FINDINGS.md
│   └── tables/*.csv
├── figures/
│   ├── 27 original PNGs
│   ├── README.md           # clickable figure index
│   ├── descriptions/       # one explanation per PNG
│   ├── MANIFEST.md         # filename/size/SHA-256 integrity
│   └── SUMMARY_FIGURES.md
├── docs/
│   ├── README.md           # documentation index
│   ├── formulas.md
│   ├── analysis_flow.md
│   ├── decision_trees.md
│   ├── thesis_mapping.md
│   ├── DEFENSE_QA.md
│   ├── conclusion.md
│   └── FINAL_AUDIT.md
├── validation/
│   └── validation_report.md
├── requirements.txt
├── tests/
└── .github/workflows/
```

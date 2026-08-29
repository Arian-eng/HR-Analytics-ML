# START HERE — Thesis Reviewer Evidence Dashboard

This page is the shortest route through the repository. It is designed so that a reviewer can move from **research scope → method → numerical output → tables → figures → interpretation → limitations** without searching through the directory tree.

## 1. Ten-minute reviewer path

| Step | What to inspect | Direct link | Why it matters |
|---|---|---|---|
| 1 | Scope and dataset roles | [`README.md`](README.md) | Separates the direct GHRM survey from the three general HR benchmark datasets. |
| 2 | Main numerical results | [`results/MODEL_RESULTS.md`](results/MODEL_RESULTS.md) | Reviewer-readable classification, regression, K-Means and reliability results. |
| 3 | Compact result tables | [`results/README.md`](results/README.md) | Direct index to every CSV summary and authoritative JSON output. |
| 4 | All 27 original figures | [`figures/README.md`](figures/README.md) | Direct PNG links, result interpretation and limitations for every figure. |
| 5 | Figure-specific explanations | [`figures/descriptions/`](figures/descriptions/) | One explanation file for each of the 27 PNG artifacts. |
| 6 | Algorithm equations and metrics | [`docs/formulas.md`](docs/formulas.md) | Accuracy, Precision, Recall, F1, RMSE, R², K-Means, Cronbach alpha, bootstrap and McNemar definitions. |
| 7 | Decision-tree transparency | [`docs/decision_trees.md`](docs/decision_trees.md) | Depth, leaf count and interpretation of each fitted tree. |
| 8 | Thesis-to-evidence traceability | [`docs/thesis_mapping.md`](docs/thesis_mapping.md) | Maps thesis analysis items to code, JSON and visual evidence. |
| 9 | Validation against the thesis | [`validation/validation_report.md`](validation/validation_report.md) | Shows both matches and documented differences instead of hiding them. |
| 10 | Defense-sensitive limitations | [`docs/DEFENSE_QA.md`](docs/DEFENSE_QA.md) | Concise answers on n=320, imbalance, FEP reliability, GEE+, tree complexity and dataset scope. |

## 2. Evidence by analysis question

| Question a reviewer may ask | Where the answer is |
|---|---|
| What data were analyzed? | [`README.md`](README.md), [`data/README.md`](data/README.md) |
| Which dataset directly measures Green HRM? | [`README.md`](README.md), [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md) |
| Why are the other three datasets included? | [`README.md`](README.md), [`docs/DEFENSE_QA.md`](docs/DEFENSE_QA.md) |
| What algorithms were actually run? | [`src/README.md`](src/README.md), [`src/`](src/) |
| What split and CV settings were used? | [`README.md`](README.md), [`src/README.md`](src/README.md) |
| What are the final model scores? | [`results/MODEL_RESULTS.md`](results/MODEL_RESULTS.md), [`results/tables/`](results/tables/) |
| Where are the raw machine-readable outputs? | [`results/README.md`](results/README.md), [`results/*.json`](results/) |
| Where are all confusion matrices and decision trees? | [`figures/README.md`](figures/README.md) |
| Where are all K-Means criteria and PCA plots? | [`figures/README.md`](figures/README.md) |
| What patterns were discovered? | [`results/PATTERNS_AND_FINDINGS.md`](results/PATTERNS_AND_FINDINGS.md) |
| What are the key limitations? | [`docs/conclusion.md`](docs/conclusion.md), [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md) |
| How do repo results compare with the thesis text? | [`validation/validation_report.md`](validation/validation_report.md) |

## 3. Dataset roles — critical interpretation boundary

| Dataset | n | Role in this repository | Target |
|---|---:|---|---|
| GHRM survey (`HRM_DATASETS.csv`) | 320 | **Primary direct Green HRM evidence** | FEP |
| IBM HR Analytics | 1,470 | General HR benchmark / methodological validation | Attrition |
| Job Change | 19,158 | General HR benchmark / methodological validation | target |
| Employee Promotion | 54,808 | General HR benchmark / methodological validation | is_promoted |

The three public HR benchmark datasets are **not** presented as direct measurements of Green HRM and are not merged record-by-record with the GHRM survey.

## 4. Main published findings

### Classification benchmarks

| Dataset | Best held-out F1 in committed run | Model |
|---|---:|---|
| IBM Attrition | 0.5000 | MLP |
| Job Change | 0.6042 | Random Forest |
| Employee Promotion | 0.5054 | MLP |

Because the classification targets are imbalanced, Accuracy is not interpreted alone; Precision, Recall and F1 are reported together with confusion matrices and uncertainty diagnostics.

### GHRM regression

The best Base held-out R² is **0.4374** (Random Forest). Adding GEE produces the largest positive point change for MLP, but every paired 95% bootstrap interval for ΔR² includes zero. Therefore the repository does **not** claim a statistically reliable general improvement from adding GEE.

### GHRM clustering

The selected GHRM K-Means solution is **k=2**, with Silhouette **0.4363** and Davies-Bouldin **0.8611**. The two clusters have mean FEP values **3.851** and **3.041**. This is interpreted as exploratory segmentation, not causation.

### Reliability

Cronbach's alpha is above 0.70 for GRS, GTD, GPA, GCM and GEE. FEP has alpha **0.6043** and is explicitly retained as a limitation.

## 5. Figure access

All **27 original PNG artifacts are committed in `figures/`**. Each PNG has a matching reviewer-facing Markdown explanation in [`figures/descriptions/`](figures/descriptions/). The consolidated index is [`figures/README.md`](figures/README.md), and exact filenames, sizes and SHA-256 digests are recorded in [`figures/MANIFEST.md`](figures/MANIFEST.md).

## 6. Reproducibility boundary

The analysis scripts are committed and use repository-relative paths with fixed random seed 42. The raw CSV files are not committed; their exact expected filenames and roles are documented in [`data/README.md`](data/README.md). The repository therefore supports inspection of the executed pipeline and published outputs, but does not claim bit-for-bit reproduction on arbitrary future library versions because the original exact package snapshot was not captured.

## 7. Claims intentionally avoided

This repository does not claim that:

- the three benchmark datasets directly measure Green HRM;
- observational correlations or clusters establish causality;
- high Accuracy alone is sufficient under class imbalance;
- adding GEE significantly improves every GHRM model;
- the Promotion Decision Tree is preferable despite its depth 63 and 4,855 leaves;
- every validation result is numerically identical to the thesis text where the validation report records a difference.

## 8. Repository audit

For a final reviewer-facing audit of what is present and what each evidence layer means, see [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md).

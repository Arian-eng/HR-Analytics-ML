# Output guide

Use this guide to locate the saved evidence for each technical question.

## Data identity and structure

| Question | File |
|---|---|
| How many rows and raw fields does each dataset contain? | `results/data/dataset_inventory.csv` |
| What is the source hash and duplicate-row count? | `results/data/data_quality.json` |
| What are the name, type, missingness, and distinct count of each field? | `results/data/column_dictionary.csv` |
| Which local rows were assigned to training or test? | `results/splits/<dataset>.csv` |

## Classification outputs

Each classifier uses this path:

```text
results/classification/<dataset>/<model>/
```

| File | Contents |
|---|---|
| `tuning_results.csv` | Every evaluated hyperparameter combination and CV score |
| `model_diagnostics.json` | Selected model, hyperparameters, and fitted complexity |
| `test_predictions.csv` | Local record ID, truth, prediction, and correctness |
| `confusion_matrix.csv` | TN, FP, FN, and TP |
| `permutation_importance.csv` | F1 change after permuting each source feature |
| `tree_nodes.csv` | Every Decision Tree node and positive-class probability |
| `tree_rules.txt` | Complete human-readable Decision Tree rules |
| `tree_top_levels.png` | Top four tree levels for readability |
| `forest_tree_summary.csv` | Depth and leaf count for every Random Forest tree |
| `representative_tree_nodes.csv` | Every node in forest tree 0 |
| `coefficients.csv` | Linear SVM coefficients |

The all-model summary is `results/tables/classification_metrics.csv`; all McNemar comparisons are in `results/tables/mcnemar.csv`.

## GHRM regression outputs

```text
results/regression/base/<model>/
results/regression/gee/<model>/
```

Tuning, prediction, importance, and diagnostics files follow the same pattern. `repeated_cv_metrics.csv` records all 25 folds of repeated cross-validation. `convergence_warnings.json` records warnings from tuning and repeated CV.

| Question | File |
|---|---|
| What are R², RMSE, MAE, and their intervals? | `results/tables/regression_metrics.csv` |
| What changed after adding GEE? | `results/tables/base_vs_gee_plus.csv` |
| Which variables matter most for each model? | `results/tables/regression_feature_importance.csv` |
| How are public metrics recomputed without row-level outputs? | `results/tables/regression_validation_aggregates.csv` |

## Hidden-pattern outputs

```text
results/clustering/<dataset>/
```

| File | Contents |
|---|---|
| `kmeans_metrics.csv` | SSE, Silhouette, and Davies–Bouldin for k=2..7 |
| `cluster_assignments.csv` | Local cluster membership for every row |
| `cluster_sizes.csv` | Count and share of each selected cluster |
| `cluster_distinguishing_features.csv` | Largest standardized feature differences and direction |
| `cluster_profile_numeric.csv` | Numeric mean, median, and standard deviation |
| `cluster_profile_categorical.csv` | Categorical mode and share |
| `cluster_target_summary.csv` | Target distribution or mean after clustering |

For GHRM, compare `cluster_distinguishing_features.csv` with `cluster_target_summary.csv`: the five green constructs form the clusters, while mean `FEP` is calculated afterward.

Detailed profiles for the three public employee datasets are generated locally only. The public repository retains k-selection metrics and sizes without publishing employee-level identifiers or sensitive income/rate aggregates. Detailed GHRM profiles remain public because they contain research constructs without respondent IDs.

## Ready-to-review outputs

Consolidated tables are in `results/tables/`; final figures are in `results/figures/`; `results/analysis_report.md` links the main findings to their detailed artifacts.

## Tracing one reported value

To audit IBM MLP F1:

1. The local split is recorded in `results/splits/ibm.csv`.
2. Truth and predictions are in `results/classification/ibm/mlp/test_predictions.csv`.
3. F1 is calculated from those two columns.
4. The value is repeated in the model and consolidated classification tables.
5. `scripts/validate_results.py` checks equality independently.

Steps 1 and 2 are local-only to protect employee-level outputs. The public validator recomputes the same F1 from the four confusion-matrix cells. Regression metrics are recomputed from record counts and aggregate squared/absolute-error sums, not individual predictions.

A value such as 0.465 must always be identified by metric, model, dataset, and source path. The current output contains no unlabeled thesis number.

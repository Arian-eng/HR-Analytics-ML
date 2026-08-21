# Final pipeline validation report

Date: 2026-08-21

Overall assessment: **Reproducible with stated statistical caveats**

## Scope and data identity

The validated pipeline analyzes four independent datasets and never merges
records across sources:

| Dataset | Rows | Raw columns | Target | Role |
|---|---:|---:|---|---|
| IBM HR Attrition | 1,470 | 35 | `Attrition` | classification and K-Means |
| Job Change | 19,158 | 14 | `target` | classification and K-Means |
| Employee Promotion | 54,808 | 14 | `is_promoted` | classification and K-Means |
| GHRM–Environmental Performance | 320 | 33 | continuous `FEP` | regression and K-Means |

The exact filenames, row counts, and SHA-256 hashes are machine-readable in
[`data_manifest.json`](data_manifest.json). The validation command verifies any
locally present source file and `--require-data` requires all four files.

## Reproduction result

A clean full `python run_all.py` execution completed successfully with the
version-pinned Python 3.12 environment. CI tests the publication bundle on both
Python 3.11 and 3.12. All classification and regression point
estimates reproduced the previously validated values at four decimal places.
The generated narrative, CSV tables, and figures were rebuilt in that same run.

The definitive numerical sources are:

- [`results/tables/classification_metrics.csv`](../results/tables/classification_metrics.csv)
- [`results/tables/regression_metrics.csv`](../results/tables/regression_metrics.csv)
- [`results/tables/regression_variant_comparison.csv`](../results/tables/regression_variant_comparison.csv)
- [`results/tables/kmeans_selection.csv`](../results/tables/kmeans_selection.csv)
- [`results/tables/cluster_sizes.csv`](../results/tables/cluster_sizes.csv)

The Persian Chapter 4 text is generated from these artifacts rather than
maintaining a second set of handwritten result values.

## Automated QA performed

- Python compilation completed for `run_all.py`, `src`, `tests`, and `scripts`.
- Eleven data-free unit tests passed in three consecutive runs.
- `pip check` reported no broken dependency requirements.
- All 12 classification metric rows were recalculated from the 12 confusion
  matrices and matched within floating-point tolerance.
- Confusion-matrix totals equal the expected 20% held-out sizes: IBM 294, Job
  Change 3,832, and Employee Promotion 10,962.
- All eight regression rows contain finite R², MAE, RMSE, and CV RMSE values.
- Every published point estimate lies inside its corresponding 95% bootstrap
  interval.
- Base and `GEE+` comparisons use paired predictions from the same 64 held-out
  GHRM rows.
- The selected K-Means `k` is between 2 and 7 and the published cluster sizes
  sum to each dataset's full row count.
- The Persian report contains Persian digits and uses `.` rather than `٫` as
  the decimal separator.
- The CI workflow runs compilation, unit tests, dependency checks, and the
  published-result validator without requiring redistribution of raw data.

Run the same checks locally with:

```bash
python -m compileall -q run_all.py src tests scripts
python -m unittest discover -s tests -v
python scripts/validate_published_results.py --require-data
python -m pip check
```

## Statistical findings relevant to interpretation

- IBM and Employee Promotion are imbalanced; Accuracy must be accompanied by
  positive-class Precision, Recall, F1, and confusion matrices.
- The 95% bootstrap F1 intervals disclose substantial uncertainty for the small
  IBM test set.
- The GHRM test set has only 64 observations. For LinearSVR, adding `GEE`
  changes R² by approximately 0.0007, and the paired 95% interval includes
  zero. This does not support a stable incremental predictive benefit.
- Job Change and Employee Promotion Silhouette values use deterministic
  5,000-row samples; the other clustering metrics use all rows.
- K-Means uses Euclidean distance on standardized numeric and one-hot encoded
  categorical features. These clusters are exploratory, not validated employee
  types or green/non-green labels.

## Remaining governance action

`CODEOWNERS` and CI configuration are committed. GitHub branch protection is a
repository setting, not a file. An administrator should protect `main`, require
both `validate` matrix checks, and require a pull-request review before merging.

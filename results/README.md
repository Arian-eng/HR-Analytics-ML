# Results index

This folder contains the published numerical evidence from the executed analyses. Use this page as the direct index.

## Reviewer-readable summaries

- [`MODEL_RESULTS.md`](MODEL_RESULTS.md) — consolidated classification, GHRM regression, K-Means and reliability results.
- [`PATTERNS_AND_FINDINGS.md`](PATTERNS_AND_FINDINGS.md) — evidence-linked interpretation of the main observed patterns.

## Compact CSV tables

- [`tables/classification_summary.csv`](tables/classification_summary.csv) — Accuracy, Precision, Recall and F1 for all classification models.
- [`tables/ghrm_regression_summary.csv`](tables/ghrm_regression_summary.csv) — Base and GEE+ R², RMSE, MAE and CV RMSE.
- [`tables/kmeans_summary.csv`](tables/kmeans_summary.csv) — selected k, SSE, Silhouette and Davies-Bouldin.
- [`tables/ghrm_reliability_summary.csv`](tables/ghrm_reliability_summary.csv) — Cronbach's alpha by construct.
- [`tables/decision_tree_summary.csv`](tables/decision_tree_summary.csv) — fitted tree depth and leaf count.
- [`tables/pattern_summary.csv`](tables/pattern_summary.csv) — compact map of reported patterns and interpretation boundaries.

## Authoritative machine-readable outputs

- [`ghrm_full_report.json`](ghrm_full_report.json) — GHRM regression, Base-vs-GEE+, paired bootstrap and K-Means outputs.
- [`ghrm_reliability_report.json`](ghrm_reliability_report.json) — Cronbach alpha, descriptives and construct correlations.
- [`ibm_attrition_classification_report.json`](ibm_attrition_classification_report.json) — IBM classification metrics, tuning and inferential diagnostics.
- [`ibm_attrition_kmeans_report.json`](ibm_attrition_kmeans_report.json) — IBM K-Means candidate and selected-cluster diagnostics.
- [`job_change_classification_report.json`](job_change_classification_report.json) — Job Change classification outputs.
- [`job_change_kmeans_report.json`](job_change_kmeans_report.json) — Job Change clustering outputs.
- [`promotion_classification_report.json`](promotion_classification_report.json) — Promotion classification outputs.
- [`promotion_kmeans_report.json`](promotion_kmeans_report.json) — Promotion clustering outputs.

## Interpretation rule

The JSON files are the numerical source of truth for this repository. Markdown and CSV files are reviewer-facing summaries of those published results. Figures are indexed separately in [`../figures/README.md`](../figures/README.md).

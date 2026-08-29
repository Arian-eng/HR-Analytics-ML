# Source-code index

This folder contains the executable analysis pipeline. The scripts are separated by dataset/analysis role so that a reviewer can see exactly where each published result comes from.

## GHRM survey

- [`run_ghrm.py`](run_ghrm.py) — 80/20 split (256/64), Base and GEE+ regression, four regressors, 5-fold RMSE tuning, held-out R²/RMSE/MAE, paired bootstrap comparison and GHRM K-Means.
- [`run_ghrm_reliability.py`](run_ghrm_reliability.py) — Cronbach's alpha, construct descriptives and construct correlation matrix.

## General HR benchmark classification

- [`run_ibm.py`](run_ibm.py) — IBM Attrition classification and clustering.
- [`run_jobchange.py`](run_jobchange.py) — Job Change classification and clustering.
- [`run_promotion.py`](run_promotion.py) — Promotion classification and clustering orchestration.
- [`_promotion_step1_per_model.py`](_promotion_step1_per_model.py) — staged Promotion model fitting.
- [`_promotion_step2_combine.py`](_promotion_step2_combine.py) — combines Promotion model outputs.
- [`_promotion_step3_tree_and_kmeans.py`](_promotion_step3_tree_and_kmeans.py) — Promotion decision-tree and K-Means outputs.

## Shared utilities and Chapter 4 figures

- [`ch3_utils.py`](ch3_utils.py) — shared preprocessing, metrics, model-selection and evaluation utilities.
- [`generate_chapter4_charts.py`](generate_chapter4_charts.py) — Chapter 4 descriptive and cross-dataset summary figures.

## Execution assumptions

- Random seed: **42**.
- Classification: 80/20 held-out split and 3-fold stratified F1 tuning.
- GHRM regression: 80/20 held-out split and 5-fold RMSE tuning in the committed final pipeline.
- General-dataset K-Means: standardized numeric-only inputs; candidates k=2..7; selected by maximum Silhouette with SSE and Davies-Bouldin reported as supporting diagnostics.
- Raw CSV filenames and placement are documented in [`../data/README.md`](../data/README.md).

## Traceability

For direct mapping from code to output and figures, use [`../docs/thesis_mapping.md`](../docs/thesis_mapping.md). For numerical outputs, use [`../results/README.md`](../results/README.md).

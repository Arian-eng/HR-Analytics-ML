# Reproducible Chapter 4 Results

This directory contains thesis-ready result specifications and generated-result placeholders.

## Important

Raw CSV datasets are intentionally not committed to the repository. Put the three supplied CSV files under `data/` locally and run `python src/thesis_analysis_full.py`.

The script writes:

- `results/model_comparison.csv`
- `results/confusion_matrices.csv`
- `results/*_kmeans.csv`
- `results/*_feature_importance.csv`
- `results/green_hrm_regression_results.csv` (when Green HRM files are present)
- `results/green_hrm_kmeans_results.csv` (when Green HRM files are present)
- `results/chapter_4_results.xlsx`

Only results reproduced from the supplied raw data should be copied into the final thesis. Values from older Word drafts are provisional until reproduced.

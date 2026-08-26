# GHRM–Environmental Performance validation

The current Green HRM analysis uses `HRM DATASETS(2).csv` (and also accepts the alias `HRM DATASETS.csv`), not the superseded Green Innovation or Sustainable Performance datasets.

## Construct preparation

The loader checks all item columns before analysis and constructs complete-case row means for:

- `GRS`: green recruitment and selection
- `GTD`: green training and development
- `GPA`: green performance appraisal
- `GCM`: green compensation management
- `GEE`: green employee empowerment/engagement
- `FEP`: firm environmental performance

The source file uses the misspelled column name `GDT3` for the third training/development item; the loader normalizes it to `GTD3` before constructing the score.

## Regression protocol

- continuous target: `FEP`
- base predictors: `GRS`, `GTD`, `GPA`, `GCM`
- supplementary predictors: base plus `GEE`
- 80/20 holdout with `random_state=42`
- three-fold shuffled KFold on training data only
- RMSE tuning objective
- held-out R², RMSE, and MAE

If any declared item is missing, that row's construct score is missing rather
than being calculated from a different subset of items. Rows without a complete
`FEP` composite are excluded from regression. Predictor missingness is
median-imputed inside the model pipeline.

The tracked R², RMSE, and MAE values are the values reported in thesis tables 4-17 and 4-18. Live predictions remain under `outputs/ghrm` and do not overwrite those publication values.

## Clustering protocol

K-Means uses the five GHRM predictor composites and excludes `FEP` from cluster formation. Missing predictor composites are median-imputed before standardization. The analysis evaluates `k=2..7` with Inertia/SSE, Silhouette, and Davies-Bouldin; thesis publication selects k=2 and reports cluster sizes and descriptive profiles from table 4-10.

## Interpretation boundary

The outputs are predictive and exploratory. They do not establish causal effects, mediation, or a substantively “green” versus “non-green” cluster label.

# Dedicated Green HRM validation

The dedicated datasets contain continuous composite constructs, so regression is used instead of forcing them into binary classification.

## Green Innovation dataset

- 422 observations, 34 variables.
- Predictors: `MGHRM`, `MGWE`, `MGTFL`.
- Target: `MGI` (green innovation composite).
- 80/20 holdout, `random_state=42`.
- Linear Regression: R² = **0.4735**, MAE = **0.4657**.
- Random Forest Regressor: R² = **0.3413**, MAE = **0.4655**.
- Random Forest feature importance: MGWE **0.7356**, MGTFL **0.1340**, MGHRM **0.1304**.
- Best K-Means silhouette among k=2..5: **k=4**, silhouette ≈ **0.6217**.

## Sustainable Performance dataset

- 409 observations, 32 variables.
- Predictors: `MGHRM`, `MPEI`, `MPES`.
- Target: `MSP` (sustainable performance composite).
- 80/20 holdout, `random_state=42`.
- Linear Regression: R² = **0.5938**, MAE = **0.4576**.
- Random Forest Regressor: R² = **0.5592**, MAE = **0.4655**.
- Random Forest feature importance: MPEI **0.8022**, MGHRM **0.1060**, MPES **0.0918**.
- Best K-Means silhouette among k=2..5: **k=4**, silhouette ≈ **0.5754**.

## Interpretation boundary

These results establish predictive associations in the public datasets. They do **not** establish causal effects. The Green HRM composite should be described as an explanatory/predictive feature, while green innovation and sustainable performance are continuous outcomes.

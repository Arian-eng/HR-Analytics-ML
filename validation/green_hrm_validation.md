# Dedicated Green HRM validation

The dedicated datasets contain continuous composite constructs, so regression is used instead of forcing them into binary classification. Results below were revalidated against the exact CSV files currently available for the thesis.

## Green Innovation dataset

- 422 observations, 34 variables.
- Predictors: `MGHRM`, `MGWE`, `MGTFL`.
- Target: `MGI` (green innovation composite).
- 80/20 holdout, `random_state=42`.
- Linear Regression: R² = **0.4735**, MAE = **0.4657**, RMSE ≈ **0.6719**.
- Random Forest Regressor: R² = **0.3413**, MAE = **0.4655**, RMSE ≈ **0.7515**.
- Random Forest feature importance: MGWE **0.7356**, MGTFL **0.1340**, MGHRM **0.1304**.
- Reproducible K-Means using the current script's standardized `MGHRM`, `MGWE`, `MGTFL`, and `MGI` features: best among k=2..5 is **k=4**, silhouette ≈ **0.4117**.

## Sustainable Performance dataset

- 409 observations, 32 variables.
- Predictors: `MGHRM`, `MPEI`, `MPES`.
- Target: `MSP` (sustainable performance composite).
- 80/20 holdout, `random_state=42`.
- Linear Regression: R² = **0.5938**, MAE = **0.4576**, RMSE ≈ **0.6271**.
- Random Forest Regressor: R² = **0.5592**, MAE = **0.4655**, RMSE ≈ **0.6533**.
- Random Forest feature importance: MPEI **0.8022**, MGHRM **0.1060**, MPES **0.0918**.
- Reproducible K-Means using the current script's standardized `MGHRM`, `MPEI`, `MPES`, and `MSP` features: best among k=2..5 is **k=4**, silhouette ≈ **0.3977**.

## Important correction

Earlier draft notes reported higher silhouette values for these two datasets. Those values are not reproduced by the current committed clustering specification. They have therefore been removed from the validated results rather than retained without reproducibility.

## Interpretation boundary

These results establish predictive associations in the public datasets. They do **not** establish causal effects. The Green HRM composite should be described as an explanatory/predictive feature, while green innovation and sustainable performance are continuous outcomes.

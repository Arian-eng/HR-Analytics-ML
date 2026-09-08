# Model results summary

Revised pre-defense execution. See [provenance and commands](../validation/defense_revision.md).

## IBM Attrition (test n=294)

| Model | Accuracy | Precision | Recall | F1 | Parameters |
|---|---:|---:|---:|---:|---|
| DecisionTree | 0.7483 | 0.3291 | 0.5532 | 0.4127 | {"max_depth": 6, "min_samples_split": 10} |
| RandomForest | 0.8469 | 0.6250 | 0.1064 | 0.1818 | {"max_depth": null, "n_estimators": 400} |
| LinearSVC | 0.7517 | 0.3488 | 0.6383 | 0.4511 | {"C": 0.1} |
| MLPClassifier | 0.8810 | 0.6667 | 0.5106 | 0.5783 | {"alpha": 0.001, "hidden_layer_sizes": [50], "learning_rate_init": 0.01} |

McNemar raw and within-dataset Holm-adjusted p-values are in the JSON. They assess error probability, not the F1 difference.

## Job Change (test n=3832)

| Model | Accuracy | Precision | Recall | F1 | Parameters |
|---|---:|---:|---:|---:|---|
| DecisionTree | 0.7116 | 0.4545 | 0.7853 | 0.5758 | {"max_depth": 6, "min_samples_split": 2} |
| RandomForest | 0.7630 | 0.5176 | 0.7257 | 0.6042 | {"max_depth": 15, "n_estimators": 400} |
| LinearSVC | 0.7437 | 0.4906 | 0.7361 | 0.5888 | {"C": 0.1} |
| MLPClassifier | 0.7850 | 0.5841 | 0.4764 | 0.5248 | {"alpha": 0.0001, "hidden_layer_sizes": [50], "learning_rate_init": 0.01} |

McNemar raw and within-dataset Holm-adjusted p-values are in the JSON. They assess error probability, not the F1 difference.

## Employee Promotion (test n=10962)

| Model | Accuracy | Precision | Recall | F1 | Parameters |
|---|---:|---:|---:|---:|---|
| DecisionTree | 0.8982 | 0.4113 | 0.4518 | 0.4306 | {"max_depth": null, "min_samples_split": 2} |
| RandomForest | 0.9350 | 0.8323 | 0.2976 | 0.4385 | {"max_depth": null, "n_estimators": 200} |
| LinearSVC | 0.7584 | 0.2387 | 0.8383 | 0.3716 | {"C": 0.1} |
| MLPClassifier | 0.9414 | 0.9011 | 0.3512 | 0.5054 | {"alpha": 0.0001, "hidden_layer_sizes": [32, 16], "learning_rate_init": 0.001} |

McNemar raw and within-dataset Holm-adjusted p-values are in the JSON. They assess error probability, not the F1 difference.

## GHRM base (test n=64)

| Model | R² | RMSE | MAE | CV RMSE |
|---|---:|---:|---:|---:|
| RandomForestRegressor | 0.4374 | 0.4870 | 0.3855 | 0.5682 |
| DecisionTreeRegressor | 0.4361 | 0.4876 | 0.3719 | 0.6019 |
| LinearSVR | 0.4287 | 0.4908 | 0.3831 | 0.5320 |
| MLPRegressor | 0.2349 | 0.5680 | 0.4488 | 0.5632 |

## GHRM gee_plus (test n=64)

| Model | R² | RMSE | MAE | CV RMSE |
|---|---:|---:|---:|---:|
| RandomForestRegressor | 0.4197 | 0.4946 | 0.3898 | 0.5632 |
| DecisionTreeRegressor | 0.4295 | 0.4904 | 0.3804 | 0.6116 |
| LinearSVR | 0.4058 | 0.5005 | 0.3884 | 0.5346 |
| MLPRegressor | 0.3697 | 0.5155 | 0.4100 | 0.5625 |

## Paired GEE+ minus Base uncertainty

| Model | Mean bootstrap ΔR² | 95% CI |
|---|---:|---|
| RandomForestRegressor | -0.0180 | [-0.0682, 0.0285] |
| DecisionTreeRegressor | -0.0073 | [-0.0686, 0.0547] |
| LinearSVR | -0.0263 | [-0.0812, 0.0110] |
| MLPRegressor | 0.1377 | [-0.0093, 0.2817] |

All intervals include zero. Five supplementary splits showed that OLS and RF change rank; complexity has no established stable advantage. These are post-hoc descriptive checks, not confirmatory tests.

K-means and reliability outputs are unchanged. FEP alpha remains 0.6043; this measurement limitation is not cured by resampling.

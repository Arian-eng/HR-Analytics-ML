# Final HR analytics execution report

This report is generated directly by `python run_all.py`. Thesis text is never used as a numeric input. Every reported value can be traced to a source-file hash, a train/test split, selected model parameters, and a saved evaluation artifact.

## Study scope and the role of each dataset

The GHRM dataset is the main Green Human Resource Management analysis and directly contains GRS, GTD, GPA, GCM, GEE, and FEP measures. The other three datasets are independent supplementary HR analyses of attrition, job change, and promotion. They are not direct evidence of green behavior or firm sustainability and are never merged with GHRM. This project contains no Digikala data.

| Display_Name | Role | Rows | Raw_Columns | Target | SHA256 |
| --- | --- | --- | --- | --- | --- |
| IBM HR Analytics | supplementary HR analysis | 1,470 | 35 | Attrition | `a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659` |
| Job Change | supplementary HR analysis | 19,158 | 14 | target | `8b78da3482032500df40d5359c36ba4a59e28ccd6fc272b050dcf6dee37f114c` |
| Employee Promotion | supplementary HR analysis | 54,808 | 14 | is_promoted | `3d7b679b3ba36d1cad25d4e0ea40bc84e5ac741c45e9c878d72adc3db74a984c` |
| GHRM - Environmental Performance | main Green HRM analysis | 320 | 33 | FEP items | `ac5c0005d3492aa664ec4d290e027f04a778241d3ce6195e10c5f87aebc65203` |

Column names, data types, missing-value counts, and distinct-value counts are recorded in [`data/column_dictionary.csv`](data/column_dictionary.csv). Sensitive ranges and frequencies are retained only in the local run.

## Analysis workflow

| Step | Action | Saved_Evidence |
| --- | --- | --- |
| 1 | Record data identity | SHA-256, row count, and column count |
| 2 | Check data quality | Missing values, duplicates, and target completeness |
| 3 | Preprocess | Training-only imputation, encoding, and scaling |
| 4 | Split data | 80% training and 20% test with random seed 42 |
| 5 | Tune models | Cross-validation and parameter selection on training data |
| 6 | Evaluate | Held-out test metrics, matrices, and confidence intervals |
| 7 | Explain models | Feature importance, tree structure, and coefficients |
| 8 | Cluster | Evaluate k from 2 to 7 and profile the selected clusters |

Preprocessing is fitted inside each model pipeline, so imputation, encoding, and scaling learn only from training data. Classification is tuned on positive-class F1; regression is tuned on RMSE.

## Classification results

![F1 comparison](figures/classification_f1.png)

![Confusion matrices](figures/confusion_matrices.png)

### IBM HR Analytics

| Model | Accuracy | Precision | Recall | F1 | CV_Best_F1 |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.8333 | 0.4375 | 0.1489 | 0.2222 | 0.3937 |
| Decision Tree | 0.7857 | 0.3667 | 0.4681 | 0.4112 | 0.3975 |
| Linear SVM | 0.7517 | 0.3488 | 0.6383 | 0.4511 | 0.4862 |
| MLP | 0.8776 | 0.7037 | 0.4043 | 0.5135 | 0.5377 |

**MLP** has the highest positive-class F1 (**0.5135**) in this split. This result concerns only the `Attrition` target and is not a direct measure of GHRM.

[Random Forest](classification/ibm/random_forest/model_diagnostics.json): selected parameters, held-out predictions, confusion matrix, feature importance, and tuning results

[Decision Tree](classification/ibm/decision_tree/model_diagnostics.json): selected parameters, held-out predictions, confusion matrix, feature importance, and tuning results

[Linear SVM](classification/ibm/linear_svm/model_diagnostics.json): selected parameters, held-out predictions, confusion matrix, feature importance, and tuning results

[MLP](classification/ibm/mlp/model_diagnostics.json): selected parameters, held-out predictions, confusion matrix, feature importance, and tuning results

![Feature importance](figures/feature_importance_ibm.png)

### Job Change

| Model | Accuracy | Precision | Recall | F1 | CV_Best_F1 |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.7620 | 0.5155 | 0.7476 | 0.6103 | 0.5841 |
| Decision Tree | 0.7474 | 0.4954 | 0.7393 | 0.5933 | 0.5664 |
| Linear SVM | 0.7437 | 0.4906 | 0.7361 | 0.5888 | 0.5712 |
| MLP | 0.7837 | 0.5680 | 0.5508 | 0.5593 | 0.5176 |

**Random Forest** has the highest positive-class F1 (**0.6103**) in this split. This result concerns only the `target` target and is not a direct measure of GHRM.

[Random Forest](classification/job_change/random_forest/model_diagnostics.json): selected parameters, held-out predictions, confusion matrix, feature importance, and tuning results

[Decision Tree](classification/job_change/decision_tree/model_diagnostics.json): selected parameters, held-out predictions, confusion matrix, feature importance, and tuning results

[Linear SVM](classification/job_change/linear_svm/model_diagnostics.json): selected parameters, held-out predictions, confusion matrix, feature importance, and tuning results

[MLP](classification/job_change/mlp/model_diagnostics.json): selected parameters, held-out predictions, confusion matrix, feature importance, and tuning results

![Feature importance](figures/feature_importance_job_change.png)

### Employee Promotion

| Model | Accuracy | Precision | Recall | F1 | CV_Best_F1 |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.9134 | 0.4911 | 0.4443 | 0.4666 | 0.4593 |
| Decision Tree | 0.7387 | 0.2309 | 0.8865 | 0.3664 | 0.3660 |
| Linear SVM | 0.7584 | 0.2387 | 0.8383 | 0.3716 | 0.3694 |
| MLP | 0.9431 | 0.9726 | 0.3415 | 0.5055 | 0.4947 |

**MLP** has the highest positive-class F1 (**0.5055**) in this split. This result concerns only the `is_promoted` target and is not a direct measure of GHRM.

[Random Forest](classification/promotion/random_forest/model_diagnostics.json): selected parameters, held-out predictions, confusion matrix, feature importance, and tuning results

[Decision Tree](classification/promotion/decision_tree/model_diagnostics.json): selected parameters, held-out predictions, confusion matrix, feature importance, and tuning results

[Linear SVM](classification/promotion/linear_svm/model_diagnostics.json): selected parameters, held-out predictions, confusion matrix, feature importance, and tuning results

[MLP](classification/promotion/mlp/model_diagnostics.json): selected parameters, held-out predictions, confusion matrix, feature importance, and tuning results

![Feature importance](figures/feature_importance_promotion.png)

## McNemar tests

McNemar's test compares two models on the same held-out records. Columns b01 and b10 count discordant outcomes. An exact binomial test is used when the discordant total is below 25; otherwise the continuity-corrected chi-square approximation is used. All six pairwise comparisons for every classification dataset are in [`tables/mcnemar.csv`](tables/mcnemar.csv).

## Main GHRM analysis and FEP prediction

The Base model uses GRS, GTD, GPA, and GCM. The GEE+ analysis adds GEE. Alongside the 20% holdout, each selected pipeline is evaluated with repeated 5x5 cross-validation to disclose the uncertainty associated with the 320-record sample.

| Variant | Model | R2 | RMSE | MAE | R2_CI_Lower | R2_CI_Upper | Repeated_CV_R2_Mean | Repeated_CV_R2_Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | Random Forest Regressor | 0.3847 | 0.5093 | 0.3905 | 0.0538 | 0.5944 | 0.4153 | 0.1356 |
| Base | Decision Tree Regressor | 0.4222 | 0.4936 | 0.3831 | 0.0415 | 0.6525 | 0.3684 | 0.1425 |
| Base | LinearSVR | 0.4287 | 0.4908 | 0.3831 | 0.1462 | 0.5944 | 0.4653 | 0.1164 |
| Base | MLPRegressor | 0.3717 | 0.5147 | 0.4061 | 0.0987 | 0.5312 | 0.3014 | 0.3546 |
| GEE+ | Random Forest Regressor | 0.3982 | 0.5037 | 0.3905 | 0.0706 | 0.6023 | 0.4257 | 0.1267 |
| GEE+ | Decision Tree Regressor | 0.3692 | 0.5157 | 0.4053 | 0.0526 | 0.5385 | 0.3405 | 0.1450 |
| GEE+ | LinearSVR | 0.4058 | 0.5005 | 0.3884 | 0.0779 | 0.5973 | 0.4568 | 0.1202 |
| GEE+ | MLPRegressor | 0.3422 | 0.5266 | 0.4180 | 0.1067 | 0.4839 | 0.3479 | 0.1827 |

The lowest-RMSE Base model is **LinearSVR**, with R²=0.4287 and RMSE=0.4908. The lowest-RMSE GEE+ model is **LinearSVR**, with R²=0.4058 and RMSE=0.5005. These are predictive results and do not establish causality or mediation.

### Solver convergence checks

Convergence warnings are counted for tuning and all 25 repeated-CV folds rather than being suppressed.

| Variant | Model | Tuning_Convergence_Warnings | Repeated_CV_Convergence_Warnings |
| --- | --- | --- | --- |
| Base | Random Forest Regressor | 0 | 0 |
| Base | Decision Tree Regressor | 0 | 0 |
| Base | LinearSVR | 0 | 0 |
| Base | MLPRegressor | 0 | 0 |
| GEE+ | Random Forest Regressor | 0 | 0 |
| GEE+ | Decision Tree Regressor | 0 | 0 |
| GEE+ | LinearSVR | 0 | 0 |
| GEE+ | MLPRegressor | 0 | 0 |

### Change after adding GEE

| Model | Delta_R2 | Delta_R2_CI_Lower | Delta_R2_CI_Upper | Delta_MAE | Delta_MAE_CI_Lower | Delta_MAE_CI_Upper | Delta_RMSE | Delta_RMSE_CI_Lower | Delta_RMSE_CI_Upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest Regressor | 0.0135 | -0.0303 | 0.0595 | -0.0001 | -0.0148 | 0.0153 | -0.0056 | -0.0239 | 0.0113 |
| Decision Tree Regressor | -0.0530 | -0.2336 | 0.1599 | 0.0222 | -0.0334 | 0.0840 | 0.0221 | -0.0565 | 0.1135 |
| LinearSVR | -0.0229 | -0.0797 | 0.0108 | 0.0053 | -0.0069 | 0.0181 | 0.0097 | -0.0057 | 0.0250 |
| MLPRegressor | -0.0295 | -0.1539 | 0.1162 | 0.0119 | -0.0277 | 0.0531 | 0.0119 | -0.0403 | 0.0676 |

![RMSE comparison](figures/ghrm_regression_rmse.png)

![GHRM feature importance](figures/ghrm_feature_importance.png)

![Actual versus predicted FEP](figures/ghrm_actual_vs_predicted.png)

Each model directory contains selected parameters, full tuning results, held-out predictions, feature importance, and—where applicable—tree-node exports and a readable top-level plot. Random forests contain hundreds of trees, so `forest_tree_summary.csv` records every tree's depth and leaf count while `representative_tree_nodes.csv` provides the complete structure of tree 0.

## Hidden patterns from K-Means

K-Means is run separately for every dataset. Targets and identifiers are excluded from cluster formation. Values of k from 2 through 7 are evaluated and the maximum-Silhouette solution is selected. Target outcomes are examined only after clustering for descriptive profiling.

### IBM HR Analytics

The selected k is **2**, with a Silhouette score of **0.1529** (weak separation; exploratory only).

- Cluster 0: 1,004 records.
- Cluster 1: 466 records.

Detailed profiles for this public employee dataset are retained only in the local run.

![K selection diagnostics](figures/kmeans_ibm.png)

### Job Change

The selected k is **3**, with a Silhouette score of **0.5773** (relatively clear separation in this dataset).

- Cluster 0: 11,493 records.
- Cluster 1: 2,435 records.
- Cluster 2: 5,230 records.

Detailed profiles for this public employee dataset are retained only in the local run.

![K selection diagnostics](figures/kmeans_job_change.png)

### Employee Promotion

The selected k is **5**, with a Silhouette score of **0.2565** (moderate exploratory separation).

- Cluster 0: 7,250 records.
- Cluster 1: 22,855 records.
- Cluster 2: 16,251 records.
- Cluster 3: 7,182 records.
- Cluster 4: 1,270 records.

Detailed profiles for this public employee dataset are retained only in the local run.

![K selection diagnostics](figures/kmeans_promotion.png)

### GHRM - Environmental Performance

The selected k is **2**, with a Silhouette score of **0.4363** (moderate exploratory separation).

- Cluster 0: 198 records; GPA higher, GEE higher, GCM higher; mean FEP = 3.8510.
- Cluster 1: 122 records; GPA lower, GEE lower, GCM lower; mean FEP = 3.0410.

![K selection diagnostics](figures/kmeans_ghrm.png)

## Interpretation limits

1. The three public HR datasets do not directly measure GHRM; their results are limited to general HR analytics.
2. GHRM contains 320 records. Confidence intervals and repeated CV disclose uncertainty but do not justify broad population-level generalization.
3. Clustering is exploratory; cluster labels are not causal conclusions.
4. Feature importance measures predictive contribution, not causal effect.

## Audit files

- [`run_log.txt`](run_log.txt): timestamped Python execution log
- [`run_manifest.json`](run_manifest.json): environment versions, seed, and data hashes
- [`data/column_dictionary.csv`](data/column_dictionary.csv): public field dictionary
- [`tables`](tables): consolidated result tables
- [`classification`](classification): details for all four classifiers
- [`regression`](regression): details for all four regressors in both variants
- [`clustering`](clustering): cluster metrics and sizes, plus detailed GHRM profiles

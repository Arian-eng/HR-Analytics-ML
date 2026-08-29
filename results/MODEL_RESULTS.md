# Model results summary

This file is a human-readable index of the numerical outputs already stored in `results/*.json`. The JSON files remain the machine-readable source of record.

## IBM Attrition — held-out test set (n=294)

| Model | Accuracy | Precision | Recall | F1 | Selected hyperparameters |
|---|---:|---:|---:|---:|---|
| Decision Tree | 0.7483 | 0.3291 | 0.5532 | 0.4127 | depth=6, min_samples_split=10 |
| Random Forest | 0.8435 | 0.5556 | 0.1064 | 0.1786 | 100 trees, depth=15 |
| LinearSVC | 0.7517 | 0.3488 | 0.6383 | 0.4511 | C=0.1 |
| MLPClassifier | **0.8776** | **0.7200** | 0.3830 | **0.5000** | hidden=(50,), alpha=0.0001 |

**Reading the table.** MLP has the highest F1 in the committed IBM run. Random Forest has relatively high accuracy but misses most positive Attrition cases (Recall=0.1064), which is why accuracy is not used as the sole criterion.

## Job Change — held-out test set (n=3,832)

| Model | Accuracy | Precision | Recall | F1 | Selected hyperparameters |
|---|---:|---:|---:|---:|---|
| Decision Tree | 0.7116 | 0.4545 | **0.7853** | 0.5758 | depth=6, min_samples_split=2 |
| Random Forest | 0.7630 | 0.5176 | 0.7257 | **0.6042** | 400 trees, depth=15 |
| LinearSVC | 0.7437 | 0.4906 | 0.7361 | 0.5888 | C=0.1 |
| MLPClassifier | **0.7850** | **0.5841** | 0.4764 | 0.5248 | hidden=(50,), alpha=0.0001, lr=0.01 |

**Reading the table.** Random Forest gives the highest F1. Decision Tree has the highest recall, while MLP has the highest accuracy and precision but substantially lower recall.

## Employee Promotion — held-out test set (n=10,962)

| Model | Accuracy | Precision | Recall | F1 | Selected hyperparameters |
|---|---:|---:|---:|---:|---|
| Decision Tree | 0.8982 | 0.4113 | 0.4518 | 0.4306 | unrestricted depth, min_samples_split=2 |
| Random Forest | 0.9350 | 0.8323 | 0.2976 | 0.4385 | 200 trees, unrestricted depth |
| LinearSVC | 0.7584 | 0.2387 | **0.8383** | 0.3716 | C=0.1 |
| MLPClassifier | **0.9414** | **0.9011** | 0.3512 | **0.5054** | hidden=(32,16), alpha=0.0001, lr=0.001 |

**Reading the table.** MLP provides the highest F1 and precision. LinearSVC detects the largest share of promoted employees but at the cost of many false positives. The unrestricted Decision Tree reaches depth 63 and 4,855 leaves, so its complexity is treated as an overfitting warning.

## GHRM regression — held-out test set (n=64)

### Base model: GRS + GTD + GPA + GCM → FEP

| Model | R² | RMSE | MAE | CV RMSE |
|---|---:|---:|---:|---:|
| Random Forest Regressor | **0.4374** | 0.4870 | 0.3855 | 0.5682 |
| Decision Tree Regressor | 0.4361 | 0.4876 | **0.3719** | 0.6019 |
| LinearSVR | 0.4287 | 0.4908 | 0.3831 | **0.5317** |
| MLPRegressor | 0.2350 | 0.5679 | 0.4487 | 0.5685 |

### GEE+ model: GRS + GTD + GPA + GCM + GEE → FEP

| Model | R² | RMSE | MAE | CV RMSE |
|---|---:|---:|---:|---:|
| Random Forest Regressor | 0.4197 | 0.4946 | 0.3898 | 0.5632 |
| Decision Tree Regressor | **0.4295** | **0.4904** | **0.3804** | 0.6116 |
| LinearSVR | 0.4058 | 0.5005 | 0.3884 | **0.5350** |
| MLPRegressor | 0.3707 | 0.5151 | 0.4098 | 0.5651 |

### Paired Base vs GEE+ bootstrap

| Model | Mean ΔR² (GEE+ − Base) | 95% CI | Excludes zero? |
|---|---:|---|---|
| Random Forest Regressor | -0.0180 | [-0.0682, 0.0285] | No |
| Decision Tree Regressor | -0.0073 | [-0.0686, 0.0547] | No |
| LinearSVR | -0.0263 | [-0.0812, 0.0110] | No |
| MLPRegressor | **+0.1387** | [-0.0071, 0.2828] | No |

The MLP has the largest positive point change after adding GEE, but its interval still includes zero. The repository therefore treats the effect as suggestive in this split, not statistically reliable.

## K-Means model selection

| Dataset | Selected k | SSE | Silhouette | Davies-Bouldin | Interpretation |
|---|---:|---:|---:|---:|---|
| IBM Attrition | 2 | 29,253.11 | 0.1529 | 2.3825 | Weak separation |
| Job Change | 3 | 11,219.41 | **0.5773** | **0.6639** | Clearest separation among the four analyses |
| Employee Promotion | 5 | 191,896.70 | 0.2565 | 1.2695 | Modest separation |
| GHRM | 2 | 770.65 | 0.4363 | 0.8611 | Moderate separation |

For GHRM, the two selected clusters contain 198 and 122 records. Their mean FEP values are 3.851 and 3.041 respectively.

## Reliability of the direct GHRM constructs

| Construct | Cronbach's alpha | Repository interpretation |
|---|---:|---|
| GRS | 0.7846 | Acceptable |
| GTD | 0.8026 | Acceptable |
| GPA | **0.8626** | Acceptable |
| GCM | 0.7962 | Acceptable |
| GEE | 0.7975 | Acceptable |
| FEP | **0.6043** | Questionable; reported as a limitation |

## Scope note

The IBM Attrition, Job Change and Promotion models validate the machine-learning workflow on public HR benchmarks. They do not directly measure Green HRM. The GHRM survey analysis is the direct construct-level evidence for the thesis topic.
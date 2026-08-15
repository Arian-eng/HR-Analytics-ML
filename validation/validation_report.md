# Chapter 4 validation report

Date: 2026-08-15

## Dataset checks

| Dataset | Rows | Columns | Target | Target distribution | Status |
|---|---:|---:|---|---|---|
| IBM HR Employee Attrition | 1,470 | 35 | `Attrition` | 83.88% No / 16.12% Yes | Valid |
| HR Analytics: Job Change of Data Scientists | 19,158 | 14 | `target` | 75.07% class 0 / 24.93% class 1 | Valid |
| Employee Performance | 17,417 | 13 | `KPIs_met_more_than_80` | 64.12% class 0 / 35.88% class 1 | Valid |

The IBM file supplied for this thesis was verified locally. The analysis loader accepts both the original Kaggle filename and the supplied renamed filename.

## Optimized model benchmark

A reproducible 80/20 stratified holdout (`random_state=42`) was used. Preprocessing is fitted on the training set only. Three-fold cross-validation is used for model hyperparameter selection. To keep the 19k-row public dataset computationally tractable, the SVM implementation is `LinearSVC` rather than an RBF-kernel `SVC`; it remains an SVM classifier but avoids the quadratic scaling of the kernel method.

| Dataset | Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---|---:|---:|---:|---:|
| IBM HR | Random Forest | 0.8401 | 0.7995 | 0.8401 | 0.8055 |
| IBM HR | Decision Tree | 0.7551 | 0.7610 | 0.7551 | 0.7609 |
| IBM HR | Linear SVM | 0.7517 | 0.8154 | 0.7517 | 0.7775 |
| IBM HR | Neural Network | 0.8741 | 0.8556 | 0.8741 | 0.8521 |
| Job Change | Random Forest | 0.7657 | 0.7754 | 0.7657 | 0.7754 |
| Job Change | Decision Tree | 0.7336 | 0.7615 | 0.7336 | 0.7491 |
| Job Change | Linear SVM | 0.7437 | 0.7754 | 0.7437 | 0.7578 |
| Job Change | Neural Network | 0.7868 | 0.7810 | 0.7868 | 0.7779 |
| Employee Performance | Random Forest | 0.6877 | 0.6936 | 0.6877 | 0.6901 |
| Employee Performance | Decision Tree | 0.6725 | 0.6816 | 0.6725 | 0.6759 |
| Employee Performance | Linear SVM | 0.6527 | 0.6867 | 0.6527 | 0.6597 |
| Employee Performance | Neural Network | 0.6963 | 0.6842 | 0.6963 | 0.6826 |

These benchmark results supersede the earlier preliminary RF/DT-only figures in this report. They should still be treated as validation results until the final thesis tables and narrative are generated.

## IBM class imbalance

IBM Attrition contains only 16.12% positive (`Yes`) cases. Therefore accuracy alone is not sufficient. The final Chapter 4 table should include positive-class recall and F1, plus the confusion matrix, so that the ability to identify employees likely to leave is not hidden by the majority class.

## K-Means validation

Numeric variables are standardized before clustering and identifier columns are excluded.

- IBM HR Attrition: best silhouette among k=2..7 was **k=2**, silhouette ≈ **0.1529**.
- Job Change: best silhouette among k=2..7 was **k=3**, silhouette ≈ **0.5711**.
- Employee Performance: best silhouette among k=2..7 was **k=4**, silhouette ≈ **0.2572**.

## Methodological note

The three general HR datasets do not contain direct Green HRM measures. They should therefore be described as HR-performance/behavioral proxy analyses. Dedicated Green Innovation and Sustainable Performance datasets should be analyzed separately for direct Green HRM/sustainability constructs.

## Reproducibility changes

The analysis script resolves both original dataset names and the exact renamed files supplied for this thesis. Raw CSV files remain excluded from Git through `.gitignore`.

## Remaining work

1. Run and preserve the final optimized pipeline outputs for all three datasets.
2. Add class-specific metrics (especially positive-class recall/F1) to the final Chapter 4 tables.
3. Analyze the dedicated Green HRM datasets with regression/continuous-target methods where appropriate.
4. Export final tables, confusion matrices, feature-importance, K-Means figures, and the Excel workbook for Chapter 4.

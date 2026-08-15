# Chapter 4 validation report

Date: 2026-08-15

## Dataset checks

| Dataset | Rows | Columns | Target | Target distribution | Status |
|---|---:|---:|---|---|---|
| IBM HR Employee Attrition | 1,470 | 35 | `Attrition` | 83.88% No / 16.12% Yes | Valid |
| HR Analytics: Job Change of Data Scientists | 19,158 | 14 | `target` | 75.07% class 0 / 24.93% class 1 | Valid |
| Employee Performance | 17,417 | 13 | `KPIs_met_more_than_80` | 64.12% class 0 / 35.88% class 1 | Valid |

The IBM file supplied for this thesis was verified locally. The analysis loader now accepts both the original Kaggle filename and the supplied renamed filename.

## IBM preliminary holdout validation

Using the fixed 80/20 stratified split (`random_state=42`) and the thesis preprocessing pipeline:

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|---|---:|---:|---:|---:|
| Random Forest | 0.8435 | 0.8048 | 0.8435 | 0.7920 |
| Decision Tree | 0.7789 | 0.7276 | 0.7789 | 0.7502 |

IBM confusion matrices:

- Random Forest: `[[244, 3], [43, 4]]`
- Decision Tree: `[[216, 31], [34, 13]]`

Because Attrition is imbalanced (only 16.12% positive cases), accuracy alone is not sufficient. Recall/F1 for the positive attrition class must be reported alongside accuracy in the thesis.

## Earlier validated results

Using the fixed 80/20 stratified split:

| Dataset | Model | Accuracy | Weighted F1 |
|---|---|---:|---:|
| Job Change | Random Forest | 0.7709 | 0.7570 |
| Job Change | Decision Tree | 0.7109 | 0.7123 |
| Employee Performance | Random Forest | 0.6825 | 0.6706 |
| Employee Performance | Decision Tree | 0.6200 | 0.6214 |

## K-Means validation

The numeric variables were standardized before clustering and identifier columns were excluded.

- IBM HR Attrition: best silhouette among k=2..7 was **k=2**, silhouette ≈ **0.1529**.
- Job Change: best silhouette among k=2..7 was **k=3**, silhouette ≈ **0.5711**.
- Employee Performance: best silhouette among k=2..7 was **k=4**, silhouette ≈ **0.2572**.

## Methodological note

The three general HR datasets do not contain direct Green HRM measures. They should therefore be described as HR-performance/behavioral proxy analyses. Dedicated Green Innovation and Sustainable Performance datasets should be analyzed separately for direct Green HRM/sustainability constructs.

## Reproducibility changes

The analysis script now resolves both the original dataset names and the exact renamed files supplied for this thesis. Raw CSV files remain excluded from Git through `.gitignore`.

## Remaining work

1. Run the complete optimized GridSearch pipeline for all three datasets.
2. Report class-specific metrics (especially positive-class recall/F1) in addition to weighted metrics.
3. Analyze the dedicated Green HRM datasets with regression/continuous-target methods where appropriate.
4. Export final tables, confusion matrices, feature importance, K-Means figures, and the Excel workbook for Chapter 4.

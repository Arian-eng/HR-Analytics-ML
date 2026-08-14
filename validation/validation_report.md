# Chapter 4 validation report

Date: 2026-08-14

## Validation scope

The currently available local datasets were checked against the targets defined in the thesis analysis script. The thesis methodology specifies separate analyses for the HR datasets rather than merging them.

## Dataset checks

| Dataset | Rows | Columns | Target | Target distribution | Status |
|---|---:|---:|---|---|---|
| HR Analytics: Job Change of Data Scientists | 19,158 | 14 | `target` | 75.07% class 0 / 24.93% class 1 | Valid |
| Employee Performance | 17,417 | 13 | `KPIs_met_more_than_80` | 64.12% class 0 / 35.88% class 1 | Valid |

The IBM HR Attrition CSV referenced by the current script was not present in the available local files, so its results were not fabricated or inferred.

## Preliminary model validation

Using the fixed 80/20 stratified split (`random_state=42`) and the same preprocessing principle used by the thesis script:

| Dataset | Model | Accuracy | Weighted F1 |
|---|---|---:|---:|
| Job Change | Random Forest | 0.7709 | 0.7570 |
| Job Change | Decision Tree | 0.7109 | 0.7123 |
| Employee Performance | Random Forest | 0.6825 | 0.6706 |
| Employee Performance | Decision Tree | 0.6200 | 0.6214 |

These are validation figures, not the final Chapter 4 figures. The final script additionally performs GridSearchCV for Random Forest, Decision Tree, SVM, and Neural Network.

## K-Means validation

The numeric variables were standardized before clustering and identifier columns were excluded.

- Job Change: best silhouette among k=2..7 was at **k=3**, silhouette ≈ **0.5711**.
- Employee Performance: best silhouette among k=2..7 was at **k=4**, silhouette ≈ **0.2572**.

## Methodological note

The three general HR datasets do not contain direct Green HRM measures. They should therefore be described as HR-performance/behavioral proxy analyses. The revised thesis also identifies dedicated Green Innovation and Sustainable Performance datasets to provide direct GHRM-related constructs.

## Next required validation

1. Add the IBM HR Attrition CSV used in the thesis.
2. Run the complete GridSearchCV pipeline on all three general HR datasets.
3. Analyze the dedicated Green HRM datasets separately with targets appropriate to their continuous GHRM/sustainability measures rather than forcing them into binary classification.
4. Export the final model-comparison, confusion-matrix, feature-importance, K-Means, and Excel outputs for Chapter 4.

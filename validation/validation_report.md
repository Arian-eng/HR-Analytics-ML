# Chapter 4 validation report

Date: 2026-08-15

## Validation scope

The three datasets named for the thesis were checked by exact filename against the available conversation and Library files. The thesis methodology specifies separate analyses for the HR datasets rather than merging them.

## Dataset inventory

| Dataset | Expected file | Available | Rows | Columns | Target |
|---|---|---|---:|---:|---|
| HR Analytics: Job Change of Data Scientists | `aug_train(4).csv` | Yes | 19,158 | 14 | `target` |
| Employee Performance | `Uncleaned_employees_final_dataset (1)(2).csv` | Yes | 17,417 | 13 | `KPIs_met_more_than_80` |
| IBM HR Employee Attrition & Performance | `WA_Fn-UseC_-HR-Employee-Attrition (3)(2).csv` | **No** | — | — | `Attrition` |

The IBM file was not available under the requested name (or the standard IBM filename) in the available files. Its results are therefore not fabricated or inferred.

## Preliminary model validation

Using the fixed 80/20 stratified split (`random_state=42`) and the preprocessing principle used by the thesis script:

| Dataset | Model | Accuracy | Weighted F1 |
|---|---|---:|---:|
| Job Change | Random Forest | 0.7709 | 0.7570 |
| Job Change | Decision Tree | 0.7109 | 0.7123 |
| Employee Performance | Random Forest | 0.6825 | 0.6706 |
| Employee Performance | Decision Tree | 0.6200 | 0.6214 |

These are validation figures, not the final Chapter 4 figures. The production analysis script additionally performs GridSearchCV for Random Forest, Decision Tree, SVM, and Neural Network.

## Full GridSearch execution status

A full 3-fold GridSearch over the original SVM/MLP parameter grids was attempted on the available datasets. The run was stopped by the execution environment before completion because the SVM search over the one-hot encoded 19k-row Job Change dataset is computationally expensive. No partial GridSearch result was promoted to a final thesis result.

This is a computational limitation, not a model failure. The production code remains methodologically explicit and reproducible; optimization of the search space or execution on a local machine with adequate CPU/RAM is required before treating GridSearch metrics as final.

## K-Means validation

The numeric variables were standardized before clustering and identifier columns were excluded.

- Job Change: best silhouette among k=2..7 was **k=3**, silhouette ≈ **0.5711**.
- Employee Performance: best silhouette among k=2..7 was **k=4**, silhouette ≈ **0.2572**.

## Methodological note

The three general HR datasets do not contain direct Green HRM measures. They should therefore be described as HR-performance/behavioral proxy analyses. The revised thesis identifies dedicated Green Innovation and Sustainable Performance datasets to provide direct GHRM-related constructs. fileciteturn24file2L81-L115

## Required next step

1. Upload/provide `WA_Fn-UseC_-HR-Employee-Attrition (3)(2).csv` so IBM analysis can be executed.
2. Run the complete model pipeline with a computationally practical search configuration, or run the existing full GridSearch locally.
3. Analyze the dedicated Green HRM datasets separately with targets appropriate to their continuous GHRM/sustainability measures.
4. Export the final model-comparison, confusion-matrix, feature-importance, K-Means, and Excel outputs for Chapter 4.

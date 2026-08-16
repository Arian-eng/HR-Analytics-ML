# Chapter 4 — reproducible tables

## Table 1. Classification benchmark

| Dataset | Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---|---:|---:|---:|---:|
| IBM HR | Random Forest | 0.8435 | 0.8048 | 0.8435 | 0.7920 |
| IBM HR | Decision Tree | 0.7789 | 0.7276 | 0.7789 | 0.7502 |
| Job Change | Random Forest | 0.7657 | — | — | — |
| Job Change | Decision Tree | 0.7336 | — | — | — |
| Employee Performance | Random Forest | 0.6877 | — | — | — |
| Employee Performance | Decision Tree | 0.6725 | — | — | — |

The missing metrics are intentionally left blank until the current pipeline exports the corresponding values; they are not inferred.

## Table 2. K-Means

| Dataset | Best k | Silhouette |
|---|---:|---:|
| IBM HR | 2 | 0.1529 |
| Job Change | 3 | 0.5711 |
| Employee Performance | 4 | 0.2572 |

## Table 3. Green HRM regression

| Dataset | Model | R² | MAE |
|---|---|---:|---:|
| Green Innovation | Linear Regression | 0.4735 | 0.4657 |
| Green Innovation | Random Forest | 0.3413 | — |
| Sustainable Performance | Linear Regression | 0.5938 | 0.4576 |
| Sustainable Performance | Random Forest | 0.5592 | — |

## Reporting rule

These tables contain only values already validated in the current analysis. A blank cell means that the metric has not yet been exported/validated, not that it is zero.

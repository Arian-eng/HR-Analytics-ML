# Chapter 4 — final fixed benchmark

This is the single classification benchmark to use for the thesis if the fixed-configuration pipeline is selected. It must not be mixed with earlier GridSearch results.

Configuration: `test_size=0.20`, `random_state=42`.

| Dataset | Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Positive Recall | Positive F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| IBM HR | Random Forest | 0.8435 | 0.8056 | 0.8435 | 0.7914 | 0.0851 | 0.1481 |
| IBM HR | Decision Tree | 0.7585 | 0.7764 | 0.7585 | 0.7668 | 0.3404 | 0.3107 |
| IBM HR | Linear SVM | 0.7517 | 0.8233 | 0.7517 | 0.7767 | 0.6170 | 0.4427 |
| IBM HR | Neural Network | 0.8673 | 0.8521 | 0.8673 | 0.8414 | 0.2766 | 0.4000 |
| Job Change | Random Forest | 0.7706 | 0.7521 | 0.7706 | 0.7568 | 0.3979 | 0.4637 |
| Job Change | Decision Tree | 0.7336 | 0.7920 | 0.7336 | 0.7491 | 0.7455 | 0.5824 |
| Job Change | Linear SVM | 0.7430 | 0.7928 | 0.7430 | 0.7570 | 0.7319 | 0.5867 |
| Job Change | Neural Network | 0.7852 | 0.7710 | 0.7852 | 0.7748 | 0.4503 | 0.5110 |
| Employee Performance | Random Forest | 0.6906 | 0.6787 | 0.6906 | 0.6791 | 0.4480 | 0.5096 |
| Employee Performance | Decision Tree | 0.6714 | 0.6805 | 0.6714 | 0.6748 | 0.5992 | 0.5668 |
| Employee Performance | Linear SVM | 0.6527 | 0.6867 | 0.6527 | 0.6597 | 0.6896 | 0.5876 |
| Employee Performance | Neural Network | 0.6949 | 0.6820 | 0.6949 | 0.6784 | 0.4152 | 0.4941 |

## Confusion matrices

- IBM HR: RF `[[244,3],[43,4]]`; DT `[[207,40],[31,16]]`; SVM `[[192,55],[18,29]]`; NN `[[242,5],[34,13]]`.
- Job Change: RF `[[2573,304],[575,380]]`; DT `[[2099,778],[243,712]]`; SVM `[[2148,729],[256,699]]`; NN `[[2579,298],[525,430]]`.
- Employee Performance: RF `[[1846,388],[690,560]]`; DT `[[1590,644],[501,749]]`; SVM `[[1412,822],[388,862]]`; NN `[[1902,332],[731,519]]`.

## K-Means

| Dataset | Best k | Silhouette |
|---|---:|---:|
| IBM HR | 2 | 0.1529 |
| Job Change | 3 | 0.5711 |
| Employee Performance | 4 | 0.2572 |

## Green HRM regression

| Dataset | Model | R² | MAE | RMSE |
|---|---|---:|---:|---:|
| Green Innovation | Linear Regression | 0.4735 | 0.4657 | 0.6719 |
| Green Innovation | Random Forest Regressor | 0.3413 | 0.4655 | 0.7515 |
| Sustainable Performance | Linear Regression | 0.5938 | 0.4576 | 0.6271 |
| Sustainable Performance | Random Forest Regressor | 0.5592 | 0.4655 | 0.6533 |

## Interpretation

For overall Accuracy and weighted F1, Neural Network is the strongest fixed-benchmark classifier on all three datasets. However, for IBM HR, Linear SVM has the highest positive-class Recall (0.6170) and positive-class F1 (0.4427), so the thesis should not call Neural Network universally superior without specifying the metric. IBM Attrition is imbalanced; positive-class performance is therefore essential.

# Final pipeline validation report

Date: 2026-08-20

Overall assessment: **Share with caveats**

## Question and scope

The validated pipeline analyzes four independent datasets. It does not merge records across sources and does not treat their combined row count as one sample.

| Dataset | Rows | Columns | Target | Positive/minority share | Role |
|---|---:|---:|---|---:|---|
| IBM HR Attrition | 1,470 | 35 | `Attrition` | 16.12% Yes | classification and K-Means |
| Job Change | 19,158 | 14 | `target` | 24.93% class 1 | classification and K-Means |
| Employee Promotion | 54,808 | 14 | `is_promoted` | 8.52% class 1 | classification and K-Means |
| GHRM–Environmental Performance | 320 | 33 raw | continuous `FEP` | not applicable | regression and K-Means |

The exact local source files were:

- `WA_Fn-UseC_-HR-Employee-Attrition (3)(2).csv` — SHA-256 `a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659`
- `aug_train(2).csv` — SHA-256 `8b78da3482032500df40d5359c36ba4a59e28ccd6fc272b050dcf6dee37f114c`
- `train_LZdllcl(2).csv` — SHA-256 `3d7b679b3ba36d1cad25d4e0ea40bc84e5ac741c45e9c878d72adc3db74a984c`
- `HRM DATASETS(2).csv` — SHA-256 `ac5c0005d3492aa664ec4d290e027f04a778241d3ce6195e10c5f87aebc65203`

## Data-quality review

All four files had zero exact duplicate rows, zero missing or duplicated candidate identifiers, zero missing target values, and no negative or infinite numeric cells. All 27 GHRM item columns stayed within the expected Likert range 1–5.

Material caveats:

- Job Change has substantial missingness in `company_type` (32.05%), `company_size` (31.00%), `gender` (23.53%), and `major_discipline` (14.68%). The pipeline imputes these fields inside each training fold, but interpretation of these features remains limited.
- Employee Promotion is highly imbalanced: only 8.52% of rows are positive. Accuracy alone is therefore misleading.
- IBM Attrition is also imbalanced at 16.12% positive.
- IBM constant columns (`EmployeeCount`, `Over18`, `StandardHours`) are removed before modeling.

## Methodology review

Classification uses an 80/20 stratified split (`random_state=42`), three-fold stratified CV on training data only, preprocessing inside each model pipeline, and positive-class F1 for tuning. The four models are Random Forest, Decision Tree, Linear SVM, and MLP.

GHRM regression uses an 80/20 holdout, three-fold shuffled KFold on training data, RMSE tuning, and the continuous `FEP` outcome. The base variant uses `GRS`, `GTD`, `GPA`, and `GCM`; the supplementary variant adds `GEE`.

K-Means is independent for each dataset and excludes targets and identifiers. Silhouette uses all 1,470 IBM and 320 GHRM rows, but a deterministic 5,000-row sample for Job Change and Promotion to bound the quadratic distance calculation.

## Classification results

All Precision, Recall, and F1 values below refer to the positive class.

| Dataset | Model | Accuracy | Precision | Recall | F1 | CV best F1 |
|---|---|---:|---:|---:|---:|---:|
| IBM | Random Forest | 0.8401 | 0.5000 | 0.1277 | 0.2034 | 0.3384 |
| IBM | Decision Tree | 0.7857 | 0.3788 | 0.5319 | 0.4425 | 0.3653 |
| IBM | Linear SVM | 0.7517 | 0.3488 | 0.6383 | 0.4511 | 0.4862 |
| IBM | MLP | 0.8673 | 0.6176 | 0.4468 | **0.5185** | 0.5489 |
| Job Change | Random Forest | 0.7630 | 0.5177 | 0.7194 | **0.6021** | 0.5770 |
| Job Change | Decision Tree | 0.7349 | 0.4793 | 0.7403 | 0.5819 | 0.5642 |
| Job Change | Linear SVM | 0.7437 | 0.4906 | 0.7361 | 0.5888 | 0.5712 |
| Job Change | MLP | 0.7868 | 0.5915 | 0.4670 | 0.5219 | 0.4890 |
| Promotion | Random Forest | 0.9282 | 0.6467 | 0.3469 | 0.4516 | 0.4477 |
| Promotion | Decision Tree | 0.8977 | 0.4090 | 0.4497 | 0.4284 | 0.4372 |
| Promotion | Linear SVM | 0.7584 | 0.2387 | **0.8383** | 0.3716 | 0.3694 |
| Promotion | MLP | 0.9423 | **0.9218** | 0.3533 | **0.5108** | 0.4983 |

There is no universally best classifier. MLP has the highest positive-class F1 for IBM and Promotion, while Random Forest leads Job Change. Promotion Linear SVM has the highest recall but low precision, demonstrating the tradeoff hidden by accuracy.

## McNemar comparisons

McNemar tests the difference in paired correctness, not the difference in positive-class F1.

| Dataset | RF comparison | b01 | b10 | p-value |
|---|---|---:|---:|---:|
| IBM | Decision Tree | 37 | 21 | 0.0489 |
| IBM | Linear SVM | 52 | 26 | 0.0046 |
| IBM | MLP | 11 | 19 | 0.2012 |
| Job Change | Decision Tree | 211 | 103 | <0.0001 |
| Job Change | Linear SVM | 214 | 140 | 0.0001 |
| Job Change | MLP | 244 | 335 | 0.0002 |
| Promotion | Decision Tree | 543 | 209 | <0.0001 |
| Promotion | Linear SVM | 2,347 | 486 | <0.0001 |
| Promotion | MLP | 67 | 222 | <0.0001 |

Here `b01` is RF correct/other wrong and `b10` is RF wrong/other correct. IBM RF versus MLP is not statistically significant at 0.05; the other paired comparisons are significant. These tests should not be presented as proof that the model with higher overall correctness also has higher minority-class F1.

## GHRM regression results

| Variant | Model | R² | MAE | RMSE | CV RMSE |
|---|---|---:|---:|---:|---:|
| Base | Random Forest Regressor | 0.3798 | 0.3938 | 0.5114 | 0.5563 |
| Base | Decision Tree Regressor | 0.3698 | 0.3942 | 0.5155 | 0.5892 |
| Base | LinearSVR | **0.4171** | **0.3859** | **0.4958** | **0.5312** |
| Base | MLPRegressor | 0.3534 | 0.4172 | 0.5222 | 0.5928 |
| GEE+ | Random Forest Regressor | 0.3868 | 0.3939 | 0.5085 | 0.5556 |
| GEE+ | Decision Tree Regressor | 0.4134 | 0.3848 | 0.4973 | 0.6073 |
| GEE+ | LinearSVR | **0.4178** | **0.3847** | **0.4955** | **0.5350** |
| GEE+ | MLPRegressor | 0.3712 | 0.3967 | 0.5149 | 0.5700 |

LinearSVR has the strongest held-out result in both variants. Adding `GEE` changes LinearSVR R² only from 0.4171 to 0.4178; its incremental contribution is therefore small in that model and must not be described as a causal or mediation effect.

## K-Means results

| Dataset | Selected k | Silhouette | Davies-Bouldin | Interpretation |
|---|---:|---:|---:|---|
| IBM | 2 | 0.1313 | 2.6054 | weak separation |
| Job Change | 2 | 0.1513 | 2.2778 | weak separation; sampled Silhouette |
| Promotion | 5 | 0.1473 | 1.7902 | weak separation; sampled Silhouette |
| GHRM | 2 | 0.4363 | 0.8611 | moderate exploratory separation |

The general HR datasets do not support strong substantive cluster labels. GHRM cluster 0 contains 198 rows with mean `FEP=3.8510`; cluster 1 contains 122 rows with mean `FEP=3.0410`. Since `FEP` was excluded from cluster formation, this is a descriptive post-clustering profile, not a causal effect or a predefined green/non-green classification.

## Calculation and artifact QA

- All 12 classification rows and all 8 regression rows contain no missing metrics.
- Classification metrics recomputed from saved predictions match the summary within `1.11e-16`.
- Regression metrics recomputed from saved predictions match within `2.22e-16`.
- All McNemar discordant counts independently match the saved tables exactly.
- Confusion-matrix totals and prediction row counts equal the expected 20% holdouts: IBM 294, Job Change 3,832, Promotion 10,962, and GHRM 64.
- `outputs/model_metrics.xlsx` was generated successfully.
- Twelve K-Means figures were rendered and visually inspected; axes and labels are complete, and sampled Silhouette titles disclose `n=5,000`.
- The data-free compile and four-unit-test CI suite passes.

## Required caveats for Chapters 4–5

1. Do not use accuracy alone for IBM or Promotion; always report positive-class Precision, Recall, and F1.
2. Disclose Job Change missingness and training-fold imputation.
3. Describe Job Change and Promotion Silhouette as a deterministic sample-based estimate.
4. Do not label weak K-Means clusters as inherently green/non-green.
5. Treat all findings as predictive or exploratory associations, not causation or mediation.
6. Do not mix these results with historical Employee Performance, Green Innovation, or Sustainable Performance benchmarks from the superseded pipeline.

# Chapter 4 validation report — thesis version 2026-09-08

Review date: 2026-09-08

## Scope and version

This review compares the supplied thesis **آرین پوراسد(20260908-072530).docx** with the repository's stored numerical outputs and selected implementation details.

Thesis SHA-256: `da03e96f7f29a7abca379dc6f242e18ef0558a8bbd94a3dcd290f353569ed7aa`.

**This was a document-to-output comparison, not a new execution of the training pipelines.** Agreement with stored outputs establishes consistency of the reviewed text and tables; it does not independently establish reproducibility from raw data. Figures embedded in the thesis were not comprehensively revalidated in this review.

The [2026-08-28 report](https://github.com/Arian-eng/HR-Analytics-ML/blob/9029f720d2208f76b564281a8a346fc20c2c4390/validation/validation_report.md) is preserved as a historical comparison against an earlier thesis version. Its old “Thesis F1” and “Thesis R²” columns must not be presented as values in the September 8 file. Its statements about independent execution describe that historical report's claims, not work performed during this review.

## Current comparison

| Item | Location in September 8 thesis | Finding |
|---|---|---|
| Employee Promotion target | Chapters 3 and 4 | `is_promoted`, consistent with the current README and training code; KPI is an input feature. |
| Classification metrics | Tables 4-11, 4-13, 4-15 | Accuracy, Precision, Recall and F1 agree with published outputs for the four models in each dataset. |
| Base and GEE+ regression | Tables 4-17, 4-18 | R², RMSE and MAE agree with the stored GHRM report. |
| K-Means candidates | Table 4-20 | SSE, Silhouette and Davies–Bouldin agree with the stored outputs for k=2..7 in all four datasets at the reported precision. |
| Decision-tree structure | Table 4-22 | Promotion depth=63 and leaves=4,855 are disclosed, together with complexity and interpretation limitations. |
| McNemar selected comparisons | Table 4-23 | Reported p-values agree with stored pairwise outputs at the displayed precision or inequality. |
| Construct reliability | Table 4-24 and Chapter 5 limitations | FEP alpha=0.6043 is present and explicitly treated as a measurement limitation. |
| Paired bootstrap | Table 4-25 | Mean changes and 95% interval bounds agree; every ΔR² interval includes zero. |

### Classification F1

Both columns below refer to the values checked in this review.

| Dataset | Model | Stored F1 | September 8 thesis F1 |
|---|---|---:|---:|
| IBM | Decision Tree | 0.4127 | 0.4127 |
| IBM | Random Forest | 0.1786 | 0.1786 |
| IBM | LinearSVC | 0.4511 | 0.4511 |
| IBM | MLPClassifier | 0.5000 | 0.5000 |
| Job Change | Decision Tree | 0.5758 | 0.5758 |
| Job Change | Random Forest | 0.6042 | 0.6042 |
| Job Change | LinearSVC | 0.5888 | 0.5888 |
| Job Change | MLPClassifier | 0.5248 | 0.5248 |
| Promotion | Decision Tree | 0.4306 | 0.4306 |
| Promotion | Random Forest | 0.4385 | 0.4385 |
| Promotion | LinearSVC | 0.3716 | 0.3716 |
| Promotion | MLPClassifier | 0.5054 | 0.5054 |

In particular, Job Change Random Forest F1 is now 0.6042, not the historical 0.483. The highest held-out F1 belongs to MLP for IBM and Promotion, and Random Forest for Job Change.

### GHRM Base versus GEE+

The following values are common to the reviewed thesis tables and stored outputs.

| Model | Base R² | GEE+ R² | Bootstrap mean ΔR² | 95% CI for ΔR² |
|---|---:|---:|---:|---|
| Random Forest Regressor | 0.4374 | 0.4197 | -0.0180 | [-0.0682, 0.0285] |
| Decision Tree Regressor | 0.4361 | 0.4295 | -0.0073 | [-0.0686, 0.0547] |
| LinearSVR | 0.4287 | 0.4058 | -0.0263 | [-0.0812, 0.0110] |
| MLPRegressor | 0.2350 | 0.3707 | +0.1387 | [-0.0071, 0.2828] |

Bootstrap mean differences need not equal the difference between the two original point estimates. MLP has the largest positive point change, but its interval also includes zero. These results do not establish a statistically reliable improvement, causality or mediation.

## Documentation corrections and remaining checks

1. **Classification preprocessing wording — correction proposed, updated thesis not yet inspected.** The supplied document says tree classifiers were not scaled in sections 4-3-1, 4-3-2, 4-3-3 and 4-5-1, with related wording in Table 4-5. The current classification code uses a shared numeric preprocessing pipeline containing StandardScaler for all four classifiers. The text should describe that implementation, while distinguishing “not required by tree models” from “not performed.” This observation does not by itself invalidate stored metrics.

2. **K selection wording — correction proposed, updated thesis not yet inspected.** The opening of section 4-4 describes a combined decision based on several criteria and interpretation. The code and technical section 4-6-4 use maximum Silhouette as the selection rule, with SSE and Davies–Bouldin as supporting diagnostics. The opening paragraph should use the same rule.

3. **Promotion grid metadata — repository correction completed.** The per-model training source searches Random Forest n_estimators=[100, 200]. Previously, the combine script copied the shared [100, 200, 400] grid into the report. The training script now records its grid with each partial output, and the combine script reads those records. Legacy partial outputs without grid metadata require the corresponding training step to be rerun; the combine script does not infer historical grids from current defaults. The published report's metadata was corrected from the training source with an explicit provenance note. This metadata correction was not a new training run and did not change performance metrics.

4. **Independent reproduction — pending.** A fresh execution on the four original CSVs, with package versions and run provenance recorded, is still needed to test whether the current code reproduces the stored outputs. Run into a separate checkout/output directory to preserve the existing evidence. Investigate any differences before changing thesis numbers.

The revised wording proposed to the author after the supplied file was reviewed has not been verified in a subsequently uploaded document.

## Interpretation boundaries

- FEP alpha=0.6043 remains a measurement limitation even though it is now disclosed. Four items alone do not establish the cause of its lower alpha.
- Promotion tree complexity warrants caution; depth and leaf count alone are not a conclusive diagnosis of overfitting.
- The three general HR datasets do not directly measure GHRM. Direct construct-level evidence comes from the 320-record GHRM survey.
- Consistency between a document and stored results is distinct from independent reproduction and does not remove sampling or generalization limitations.

## Evidence sources

- [Published results index](../results/README.md)
- [Model results summary](../results/MODEL_RESULTS.md), blob SHA `5daccc31ac1fcdd887f4ddaf91f5878ef4bc6d92`
- [IBM classification](../results/ibm_attrition_classification_report.json)
- [Job Change classification](../results/job_change_classification_report.json)
- [Promotion classification](../results/promotion_classification_report.json)
- [GHRM regression and paired bootstrap](../results/ghrm_full_report.json), blob SHA `34e03cbbc711cb02b0a570118d9c9d40a9236547`
- [IBM K-Means](../results/ibm_attrition_kmeans_report.json), [Job Change K-Means](../results/job_change_kmeans_report.json), [Promotion K-Means](../results/promotion_kmeans_report.json); GHRM K-Means is in the GHRM report.
- [Shared classification preprocessing and K selection](../src/ch3_utils.py)
- [Promotion training](../src/_promotion_step1_per_model.py) and [report assembly](../src/_promotion_step2_combine.py)

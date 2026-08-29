# Figure guide — 27 final analysis figures

This directory is the visual-evidence layer of the thesis analysis. **All 27 final PNG artifacts are committed in `figures/`**, and **each figure has a dedicated reviewer-facing explanation in `figures/descriptions/`**. The explanations identify the purpose of the figure, the principal result or interpretation, the underlying analysis source and the main limitation of interpretation.

The numerical source of truth remains `results/*.json`. Figures summarize or visualize those analyses and must not be treated as stand-alone evidence of causality.

## Reviewer index: all 27 figures

| # | PNG | Topic / analytical role | Dedicated explanation | Key interpretation / caution |
|---:|---|---|---|---|
| 1 | [`chart1_ibm_age_histogram.png`](chart1_ibm_age_histogram.png) | IBM employee-age distribution | [`description`](descriptions/chart1_ibm_age_histogram.md) | Descriptive sample characterization only; age distribution does not establish a cause of attrition. |
| 2 | [`chart2_ibm_income_boxplot.png`](chart2_ibm_income_boxplot.png) | IBM monthly-income distribution | [`description`](descriptions/chart2_ibm_income_boxplot.md) | Documents central tendency, dispersion and extreme values; descriptive rather than causal. |
| 3 | [`chart3_jobchange_target_distribution.png`](chart3_jobchange_target_distribution.png) | Job Change target balance | [`description`](descriptions/chart3_jobchange_target_distribution.md) | Approx. 75.07%/24.93% split motivates F1, precision and recall rather than accuracy alone. |
| 4 | [`chart4_promotion_status_distribution.png`](chart4_promotion_status_distribution.png) | Promotion target balance | [`description`](descriptions/chart4_promotion_status_distribution.md) | Positive class is about 8.52%; high accuracy can therefore hide weak minority detection. |
| 5 | [`chart5_ibm_class_distribution_split.png`](chart5_ibm_class_distribution_split.png) | IBM full/train/test class balance | [`description`](descriptions/chart5_ibm_class_distribution_split.md) | Documents preservation of class proportions under the 80/20 split. |
| 6 | [`chart6_jobchange_missing_before_after.png`](chart6_jobchange_missing_before_after.png) | Job Change missingness before/after preprocessing | [`description`](descriptions/chart6_jobchange_missing_before_after.md) | Shows preprocessing completeness; disappearance of missing values does not prove unbiased imputation. |
| 7 | [`chart7_promotion_missing_before_after.png`](chart7_promotion_missing_before_after.png) | Promotion missingness before/after preprocessing | [`description`](descriptions/chart7_promotion_missing_before_after.md) | Makes missing-value treatment visible; this is not a model-performance result. |
| 8 | [`chart8_cross_dataset_silhouette_db.png`](chart8_cross_dataset_silhouette_db.png) | Cross-dataset K-Means quality comparison | [`description`](descriptions/chart8_cross_dataset_silhouette_db.md) | Job Change has the clearest internal separation; IBM is weak. Cluster indices do not prove substantive or causal HR mechanisms. |
| 9 | [`chart9_f1_comparison_across_algorithms.png`](chart9_f1_comparison_across_algorithms.png) | F1 comparison across classifiers | [`description`](descriptions/chart9_f1_comparison_across_algorithms.md) | Best committed F1: IBM MLP 0.5000, Job Change RF 0.6042, Promotion MLP 0.5054. |
| 10 | [`chart10_gee_effect_reliability.png`](chart10_gee_effect_reliability.png) | GEE+ change with paired bootstrap uncertainty | [`description`](descriptions/chart10_gee_effect_reliability.md) | MLP has the largest positive point change, but all 95% paired bootstrap intervals include zero. |
| 11 | [`ghrm_base_vs_gee_r2.png`](ghrm_base_vs_gee_r2.png) | GHRM Base vs GEE+ held-out R² | [`description`](descriptions/ghrm_base_vs_gee_r2.md) | RF, DT and LinearSVR decline in point estimate; MLP improves, without statistically reliable paired-bootstrap evidence. |
| 12 | [`ghrm_correlation_heatmap.png`](ghrm_correlation_heatmap.png) | Correlations among GRS, GTD, GPA, GCM, GEE and FEP | [`description`](descriptions/ghrm_correlation_heatmap.md) | Correlations with FEP are positive (GRS .646, GTD .645, GPA .611, GCM .646, GEE .590); association is not causation. |
| 13 | [`ghrm_decision_tree.png`](ghrm_decision_tree.png) | GHRM Decision Tree Regressor | [`description`](descriptions/ghrm_decision_tree.md) | Fitted tree depth 4, 11 leaves; splits are predictive rules, not causal thresholds. |
| 14 | [`ghrm_kmeans_criteria.png`](ghrm_kmeans_criteria.png) | GHRM K-Means criteria for k=2..7 | [`description`](descriptions/ghrm_kmeans_criteria.md) | Selected k=2; SSE 770.65, Silhouette 0.4363, DB 0.8611. |
| 15 | [`ghrm_kmeans_pca.png`](ghrm_kmeans_pca.png) | PCA projection of GHRM k=2 clusters | [`description`](descriptions/ghrm_kmeans_pca.md) | Cluster 0 n=198, mean FEP 3.851; cluster 1 n=122, mean FEP 3.041. PCA is visualization only. |
| 16 | [`ibm_attrition_confusion_matrices.png`](ibm_attrition_confusion_matrices.png) | IBM confusion matrices for four classifiers | [`description`](descriptions/ibm_attrition_confusion_matrices.md) | Demonstrates accuracy/recall trade-offs; RF accuracy 0.8435 coexists with recall 0.1064. |
| 17 | [`ibm_attrition_decision_tree.png`](ibm_attrition_decision_tree.png) | IBM fitted Decision Tree | [`description`](descriptions/ibm_attrition_decision_tree.md) | Depth 6, 38 leaves; held-out F1 0.4127. Final fitted depth should be distinguished from earlier thesis wording. |
| 18 | [`ibm_attrition_kmeans_criteria.png`](ibm_attrition_kmeans_criteria.png) | IBM K-Means criteria | [`description`](descriptions/ibm_attrition_kmeans_criteria.md) | Selected k=2; Silhouette 0.1529 and DB 2.3825 indicate weak separation. |
| 19 | [`ibm_attrition_kmeans_pca.png`](ibm_attrition_kmeans_pca.png) | PCA projection of IBM clusters | [`description`](descriptions/ibm_attrition_kmeans_pca.md) | Visual inspection only; low Silhouette requires cautious cluster interpretation. |
| 20 | [`job_change_confusion_matrices.png`](job_change_confusion_matrices.png) | Job Change confusion matrices | [`description`](descriptions/job_change_confusion_matrices.md) | RF has the highest committed F1 (0.6042); matrices expose recall/precision trade-offs. |
| 21 | [`job_change_decision_tree.png`](job_change_decision_tree.png) | Job Change fitted Decision Tree | [`description`](descriptions/job_change_decision_tree.md) | Depth 6, 46 leaves; Accuracy 0.7116, Recall 0.7853, F1 0.5758. |
| 22 | [`job_change_kmeans_criteria.png`](job_change_kmeans_criteria.png) | Job Change K-Means criteria | [`description`](descriptions/job_change_kmeans_criteria.md) | Selected k=3; SSE 11219.41, Silhouette 0.5773, DB 0.6639. |
| 23 | [`job_change_kmeans_pca.png`](job_change_kmeans_pca.png) | PCA projection of Job Change clusters | [`description`](descriptions/job_change_kmeans_pca.md) | Visualization supports the comparatively strong internal separation; clustering itself uses standardized numeric inputs. |
| 24 | [`promotion_confusion_matrices.png`](promotion_confusion_matrices.png) | Promotion confusion matrices | [`description`](descriptions/promotion_confusion_matrices.md) | MLP has highest F1 0.5054; LinearSVC recall 0.8383 comes with precision 0.2387. |
| 25 | [`promotion_decision_tree.png`](promotion_decision_tree.png) | Promotion fitted Decision Tree | [`description`](descriptions/promotion_decision_tree.md) | Depth 63 with 4,855 leaves; explicitly treated as an overfitting/interpretability warning. |
| 26 | [`promotion_kmeans_criteria.png`](promotion_kmeans_criteria.png) | Promotion K-Means criteria | [`description`](descriptions/promotion_kmeans_criteria.md) | Selected k=5; SSE 191896.70, Silhouette 0.2565, DB 1.2695; separation is modest. |
| 27 | [`promotion_kmeans_pca.png`](promotion_kmeans_pca.png) | PCA projection of Promotion k=5 clusters | [`description`](descriptions/promotion_kmeans_pca.md) | Qualitative display only; PCA does not replace the quantitative clustering criteria. |

## Interpretation boundaries

Only the 320-record GHRM survey directly measures the Green HRM constructs used in the thesis. IBM Attrition, Job Change and Employee Promotion are independent general-HR benchmark analyses; their figures demonstrate methodological performance, preprocessing, classification and clustering behavior rather than direct measurement of Green HRM.

For imbalanced classification datasets, accuracy is never interpreted alone. Precision, recall and F1 are reported together with confusion matrices. For K-Means, higher Silhouette and lower Davies-Bouldin indicate stronger internal geometric separation, not causal mechanisms or naturally occurring employee types. For the GHRM Base-vs-GEE+ analysis, point estimates are interpreted together with paired bootstrap confidence intervals. For the Promotion Decision Tree, the depth-63 / 4,855-leaf complexity is explicitly documented as an overfitting and interpretability risk.

## Reproducibility and integrity

- [`MANIFEST.md`](MANIFEST.md) records the expected filename, byte size and SHA-256 identity of each of the 27 original PNG artifacts.
- [`descriptions/`](descriptions/) provides one dedicated scientific explanation for every PNG.
- [`SUMMARY_FIGURES.md`](SUMMARY_FIGURES.md) contains additional compact summary visualizations; these are supplementary and do not replace the 27 original figures.
- `results/*.json` remains the authoritative numerical output layer.
- `docs/thesis_mapping.md` links analysis claims to code, results and visual evidence.

A reviewer should therefore be able to move from **figure → dedicated explanation → numerical result → source code** without relying on an unsupported interpretation.
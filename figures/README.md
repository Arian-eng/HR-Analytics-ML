# Figure guide

This directory is the visual evidence layer for the analysis. Each figure has a specific analytical role and is tied to either a source script or a published result file. The figures should be read together with `results/*.json`, not as stand-alone proof of causal relationships.

## Chapter 4 descriptive and summary figures

| File | What it shows | How to interpret it |
|---|---|---|
| `chart1_ibm_age_histogram.png` | Employee age distribution in the IBM Attrition dataset. | A descriptive view of the sample age structure. It provides context for the classification analysis; it does not imply that age causes attrition. |
| `chart2_ibm_income_boxplot.png` | Distribution of monthly income in the IBM dataset. | Highlights central tendency, spread and extreme values. It is a descriptive data-quality and sample-characterization figure. |
| `chart3_jobchange_target_distribution.png` | Target-class distribution in Job Change. | The dataset is imbalanced (about 75.07% vs 24.93%), so F1, precision and recall are more informative than accuracy alone. |
| `chart4_promotion_status_distribution.png` | Promotion-class distribution. | The positive class is small (about 8.52%), which explains why high accuracy can coexist with weak minority-class recall. |
| `chart5_ibm_class_distribution_split.png` | IBM class balance in the full, training and test samples. | Confirms that the 80/20 split preserves the target distribution closely enough for a fair held-out comparison. |
| `chart6_jobchange_missing_before_after.png` | Missing-value counts before and after preprocessing in Job Change. | Documents the effect of the stated imputation procedure. The plot is about preprocessing completeness, not model performance. |
| `chart7_promotion_missing_before_after.png` | Missing-value counts before and after preprocessing in Promotion. | Makes the treatment of missing `previous_year_rating` and related fields visible rather than leaving it implicit. |
| `chart8_cross_dataset_silhouette_db.png` | Selected K-Means Silhouette and Davies-Bouldin values across datasets. | Higher Silhouette and lower Davies-Bouldin indicate cleaner separation. Job Change has the clearest separation among the four analyses; IBM is much weaker. |
| `chart9_f1_comparison_across_algorithms.png` | Held-out F1 comparison for the four classifiers across the three benchmark datasets. | Allows the reader to compare minority-class performance directly instead of relying on accuracy. |
| `chart10_gee_effect_reliability.png` | Point change from Base to GEE+ with bootstrap uncertainty in the GHRM regression. | MLP shows the largest positive point change, but the paired 95% interval includes zero; therefore the improvement is not treated as statistically reliable. |

## GHRM figures

| File | Analytical role | Interpretation |
|---|---|---|
| `ghrm_base_vs_gee_r2.png` | Compares held-out R² for Base and GEE+ models. | Random Forest, Decision Tree and LinearSVR do not improve after adding GEE in this split; MLP improves in point estimate. |
| `ghrm_correlation_heatmap.png` | Pearson correlations among GRS, GTD, GPA, GCM, GEE and FEP composite scores. | All reported relationships are positive. Correlations with FEP are moderate, not evidence of causality. |
| `ghrm_decision_tree.png` | Fitted GHRM Decision Tree Regressor. | Provides a transparent view of recursive splits used to predict FEP. The selected tree has depth 4 and 11 leaves, which is substantially simpler than the unconstrained Promotion tree. |
| `ghrm_kmeans_criteria.png` | SSE, Silhouette and Davies-Bouldin across k=2..7. | The documented selection rule chooses k=2 by maximum Silhouette. |
| `ghrm_kmeans_pca.png` | Two-dimensional PCA projection of the selected GHRM clusters. | A visualization aid only: clustering is performed in the standardized construct space, not in the two displayed PCA axes. |

## IBM Attrition figures

| File | Analytical role | Interpretation |
|---|---|---|
| `ibm_attrition_confusion_matrices.png` | Confusion matrices for Decision Tree, Random Forest, LinearSVC and MLP. | Shows the different false-positive/false-negative trade-offs hidden by accuracy. Random Forest has high accuracy but very low positive-class recall in the published run. |
| `ibm_attrition_decision_tree.png` | Fitted Decision Tree classifier. | The selected tree has depth 6 and 38 leaves. The plot exposes the actual recursive decision logic of the fitted model. |
| `ibm_attrition_kmeans_criteria.png` | K-Means quality criteria for k=2..7. | k=2 is selected, but Silhouette=0.1529 indicates weak cluster separation; the clusters should therefore be interpreted cautiously. |
| `ibm_attrition_kmeans_pca.png` | PCA view of IBM K-Means clusters. | Used for visual inspection only; the actual clustering uses the standardized numeric feature set listed in the JSON report. |

## Job Change figures

| File | Analytical role | Interpretation |
|---|---|---|
| `job_change_confusion_matrices.png` | Confusion matrices for the four classifiers. | Makes model-specific error patterns visible. Random Forest has the highest published F1 (0.6042) in this run. |
| `job_change_decision_tree.png` | Fitted Decision Tree classifier. | The selected tree has depth 6 and 46 leaves, providing a bounded and inspectable rule structure. |
| `job_change_kmeans_criteria.png` | K-Means criteria for k=2..7. | k=3 is selected with Silhouette=0.5773 and Davies-Bouldin=0.6639, the clearest clustering result among the datasets. |
| `job_change_kmeans_pca.png` | PCA projection of the three selected clusters. | A two-dimensional display of structure found using the standardized numeric inputs. |

## Employee Promotion figures

| File | Analytical role | Interpretation |
|---|---|---|
| `promotion_confusion_matrices.png` | Confusion matrices for the four classifiers. | Highlights the effect of strong class imbalance. MLP has the highest F1 (0.5054); LinearSVC has high recall but low precision. |
| `promotion_decision_tree.png` | Fitted Decision Tree classifier. | The selected unrestricted tree reaches depth 63 with 4,855 leaves. This complexity is reported explicitly because it is consistent with overfitting risk. |
| `promotion_kmeans_criteria.png` | K-Means criteria for k=2..7. | k=5 is selected by Silhouette (0.2565). Separation is modest, so clusters are descriptive segments rather than sharply separated natural groups. |
| `promotion_kmeans_pca.png` | PCA display of the five selected clusters. | Supports qualitative inspection; it does not replace the three numerical clustering criteria. |

## Reproducibility and integrity

The exact expected filenames, byte sizes and SHA-256 digests of the 27 PNG artifacts are recorded in `MANIFEST.md`. Numerical interpretation in this guide is taken from the committed JSON result files. The three general HR datasets are benchmark analyses; only the 320-record GHRM survey directly measures the Green HRM constructs used in the thesis.
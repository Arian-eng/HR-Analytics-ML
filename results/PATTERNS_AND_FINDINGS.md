# Patterns obtained from the analyses

This file separates **observed model/data patterns** from broader interpretation. The statements below are limited to what is supported by the committed result files.

| # | Observed pattern | Evidence in the repository | Interpretation boundary |
|---:|---|---|---|
| 1 | Accuracy alone is misleading in the imbalanced classification tasks. | IBM Random Forest reaches Accuracy=0.8435 but Recall=0.1064 and F1=0.1786. Promotion Random Forest reaches Accuracy=0.9350 but Recall=0.2976. | High overall accuracy does not mean that the positive/minority class is identified well. |
| 2 | The preferred classifier differs by dataset. | IBM: MLP F1=0.5000. Job Change: Random Forest F1=0.6042. Promotion: MLP F1=0.5054. | There is no single algorithm that dominates all three public HR benchmarks. |
| 3 | Models show different precision–recall trade-offs. | Promotion LinearSVC Recall=0.8383 but Precision=0.2387; Promotion MLP Precision=0.9011 but Recall=0.3512. | Model choice depends on whether missing positive cases or generating false positives is more costly. |
| 4 | A very deep Decision Tree can be selected by CV but still be difficult to justify as an interpretable model. | Promotion tree: depth=63, 4,855 leaves, held-out F1=0.4306. | The result is retained as evidence of model complexity/overfitting risk rather than presented as an ideal interpretable tree. |
| 5 | Cluster separation varies strongly across datasets. | Silhouette: IBM=0.1529, Job Change=0.5773, Promotion=0.2565, GHRM=0.4363. | Job Change has the clearest selected segmentation; IBM clusters are weak and should be interpreted cautiously. |
| 6 | The direct GHRM sample separates into two groups with different FEP levels. | GHRM k=2; cluster sizes 198 and 122; mean FEP=3.851 vs 3.041. | This is an unsupervised segmentation pattern, not proof that cluster membership causes environmental performance. |
| 7 | The four Base GHRM predictors explain a meaningful but incomplete share of held-out FEP variance. | Base R²: RF=0.4374, DT=0.4361, LinearSVR=0.4287, MLP=0.2350. | Prediction remains imperfect; substantial FEP variance is not explained by these predictors in the holdout. |
| 8 | Adding GEE does not produce a statistically reliable improvement in the paired holdout comparison. | ΔR² 95% CIs include zero for RF, DT, LinearSVR and MLP. | MLP has the largest positive point change (+0.1387), but the uncertainty interval prevents a definitive improvement claim. |
| 9 | GHRM practice constructs are positively correlated with FEP. | Pearson r with FEP: GRS=0.646, GTD=0.645, GPA=0.611, GCM=0.646, GEE=0.590. | These are observational correlations and should not be described as causal effects. |
| 10 | Most GHRM constructs have acceptable internal consistency, but FEP is weaker. | Cronbach alpha: GRS=0.7846, GTD=0.8026, GPA=0.8626, GCM=0.7962, GEE=0.7975, FEP=0.6043. | FEP reliability is a limitation and should be reported rather than ignored. |

## Cross-dataset pattern table

| Dataset | Main target / role | Best held-out predictive result in this run | Clustering result | Main caution |
|---|---|---|---|---|
| IBM Attrition | Attrition benchmark | MLP F1=0.5000 | k=2, Silhouette=0.1529 | Weak clustering; class imbalance makes accuracy insufficient. |
| Job Change | Job-change benchmark | Random Forest F1=0.6042 | k=3, Silhouette=0.5773 | General HR benchmark, not a direct Green HRM measure. |
| Employee Promotion | Promotion benchmark | MLP F1=0.5054 | k=5, Silhouette=0.2565 | Strong imbalance; unrestricted tree is extremely complex. |
| GHRM survey | Direct Green HRM evidence; FEP regression | Base RF R²=0.4374; GEE+ DT R²=0.4295 | k=2, Silhouette=0.4363 | n=320 and FEP alpha=0.6043 require cautious inference. |

## What counts as a "hidden pattern" here?

In this repository, the phrase refers to empirically observed structure that is not visible from a single raw variable alone, including:

- multi-feature classification behavior revealed by different models and confusion matrices;
- precision–recall trade-offs under class imbalance;
- unsupervised cluster structure detected by K-Means;
- differences in FEP between the two GHRM clusters;
- multivariate predictive behavior of the GHRM constructs;
- the model-dependent response to adding GEE;
- correlation structure among the six GHRM composite scores.

These patterns are data-analytic findings. They are not automatically causal mechanisms.

## Direct versus benchmark evidence

The distinction is essential for the thesis argument:

- **Direct evidence:** the GHRM survey (n=320) measures GRS, GTD, GPA, GCM, GEE and FEP directly.
- **Benchmark evidence:** IBM Attrition, Job Change and Employee Promotion demonstrate the classification and clustering methodology on larger public HR datasets.

Results from the benchmark datasets should therefore not be rewritten as if they directly measured Green HRM practices.
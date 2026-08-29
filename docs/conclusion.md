# Conclusion from the committed analysis

The repository supports two related but distinct analytical purposes: direct analysis of the Green HRM survey and methodological benchmarking on three larger public HR datasets.

## 1. Direct Green HRM findings

The 320-record GHRM dataset is the only dataset in the repository that directly measures the thesis constructs GRS, GTD, GPA, GCM, GEE and FEP.

The Base models using GRS, GTD, GPA and GCM achieve held-out R² values of 0.4374 (Random Forest), 0.4361 (Decision Tree), 0.4287 (LinearSVR) and 0.2350 (MLP). These results indicate that the four practice constructs contain useful predictive information for FEP, while also leaving a substantial part of the outcome variance unexplained.

Adding GEE changes model performance differently. Random Forest, Decision Tree and LinearSVR show small decreases in held-out R². MLP improves from 0.2350 to 0.3707, but the paired bootstrap interval for the R² difference is [-0.0071, 0.2828]. Because this interval includes zero, the repository does not treat the MLP improvement as statistically reliable evidence that GEE universally improves prediction.

The GHRM K-Means analysis selects two clusters. The groups contain 198 and 122 respondents and have mean FEP values of 3.851 and 3.041. This indicates a meaningful segmentation pattern in the standardized GHRM construct space, but the clustering result is descriptive rather than causal.

The reliability analysis is generally supportive for the GHRM practice constructs: GRS=0.7846, GTD=0.8026, GPA=0.8626, GCM=0.7962 and GEE=0.7975. FEP is lower at 0.6043. This value is retained as a limitation because it weakens confidence in the internal consistency of the FEP scale relative to the other constructs.

## 2. Public HR benchmark findings

IBM Attrition, Job Change and Employee Promotion do not directly measure Green HRM. They are used to test classification, validation and clustering procedures on larger public HR datasets.

The classification results show that model ranking depends on the dataset. MLP produces the highest held-out F1 on IBM Attrition (0.5000) and Employee Promotion (0.5054), while Random Forest is highest on Job Change (0.6042). This variability argues against selecting one algorithm as universally superior.

The benchmark datasets also show why accuracy must be interpreted carefully under class imbalance. For example, IBM Random Forest achieves Accuracy=0.8435 but Recall=0.1064 for the positive class. In Promotion, Random Forest has Accuracy=0.9350 and Precision=0.8323 but Recall=0.2976. These cases demonstrate that a model can classify the majority class well while missing many positive cases.

K-Means produces different levels of separation across the benchmarks. Job Change has the clearest selected structure (k=3, Silhouette=0.5773), while IBM has weak separation (k=2, Silhouette=0.1529). Promotion selects k=5 with Silhouette=0.2565. The clustering outputs should therefore not all be described with the same level of confidence.

## 3. Decision-tree evidence

The decision-tree artifacts provide an interpretable counterpart to the ensemble, linear and neural models. IBM and Job Change use selected depth-6 trees, while the GHRM regression tree has depth 4 and 11 leaves. In contrast, the Promotion grid search selected an unrestricted tree with depth 63 and 4,855 leaves. This unusually large structure is reported as evidence of overfitting risk rather than presented as a desirable interpretable model.

## 4. Overall interpretation

Taken together, the committed results support the use of machine-learning methods to identify predictive and clustering patterns in HR data, while also showing that model conclusions depend on target imbalance, hyperparameters, dataset structure and validation design.

For the thesis topic specifically, the defensible conclusion comes from the GHRM survey: the Green HRM constructs are positively related to FEP and provide meaningful predictive information, but the incremental contribution of GEE is model-dependent and not statistically reliable in the paired holdout bootstrap. The FEP reliability result is an additional limitation that should be acknowledged.

The three larger public HR datasets strengthen the methodological side of the work by showing how the same analytical family behaves on attrition, job-change and promotion outcomes. Their results should remain clearly labeled as benchmark evidence and should not be treated as direct measurements of Green HRM.

## 5. What this repository does not claim

The repository does not claim causal effects, does not merge the four datasets record-by-record, and does not claim that the three benchmark targets are Green HRM variables. It reports predictive performance, uncertainty, clustering structure and reliability using the committed code and outputs.
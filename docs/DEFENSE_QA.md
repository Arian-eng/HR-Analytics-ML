# Defense Q&A — reviewer-sensitive points

This note gives short, evidence-based answers to questions that are likely to arise when the repository is reviewed. The answers are limited to what the committed analyses support.

## Why are there four datasets if only one directly measures Green HRM?

`HRM_DATASETS.csv` is the primary dataset because it directly contains the GHRM constructs GRS, GTD, GPA, GCM, GEE and the FEP outcome. IBM Attrition, Job Change and Employee Promotion are general HR benchmark datasets. They are analyzed independently to test the same analytical workflow on larger public HR data and to provide benchmark evidence for classification and clustering. They are not treated as direct measurements of Green HRM.

## Is n=320 enough for the GHRM machine-learning analysis?

The GHRM analysis does not rely on the 320 records as if they were a large-scale benchmark. The data are split 256/64, model complexity is restricted through a limited grid, performance is evaluated on a held-out test set, and Base-vs-GEE+ differences are assessed with paired bootstrap resampling. The sample size remains a limitation, so the results are interpreted as sample-specific predictive evidence rather than population-level proof.

## Why report F1 instead of only accuracy for the three classification datasets?

The target classes are imbalanced, especially for Employee Promotion. Accuracy can therefore be high even when the positive class is poorly detected. Precision, Recall and F1 are reported together, and the confusion matrices are part of the visual evidence layer. IBM Random Forest is a concrete example: Accuracy=0.8435 but Recall=0.1064 and F1=0.1786.

## Why is the Promotion Decision Tree so large?

The selected grid-search result is an unrestricted tree (`max_depth=None`, `min_samples_split=2`). The fitted tree reaches depth 63 with 4,855 leaves. This is not presented as a strength. It is documented as an overfitting and interpretability warning, and the tree result is compared with the other classifiers rather than used alone.

## Does adding GEE significantly improve GHRM prediction?

Not as a general conclusion. MLP has the largest positive point change in held-out R² when GEE is added, but the paired 95% bootstrap interval includes zero. The other regressors also do not show a bootstrap interval that excludes zero. The defensible conclusion is therefore that GEE changes model performance in this split, but the repository does not establish a statistically reliable improvement across models.

## What does the FEP reliability value mean?

FEP has Cronbach's alpha=0.6043, below the conventional 0.70 reference used in the repository. This is reported explicitly as a measurement limitation. GRS, GTD, GPA, GCM and GEE are all above 0.70 in the same sample.

## What do the K-Means clusters prove?

They identify descriptive segmentation under the documented feature sets and standardization procedure. They do not establish causal mechanisms. Cluster quality is reported with SSE, Silhouette and Davies-Bouldin. Job Change gives the clearest separation in the committed run (k=3, Silhouette=0.5773, Davies-Bouldin=0.6639), whereas IBM clustering is much weaker (k=2, Silhouette=0.1529).

## Are the repository results exactly identical to every number in the thesis?

No blanket claim is made. `validation/validation_report.md` explicitly separates matching from non-matching outputs. The committed JSON files are the numerical source of truth for the current repository execution. Differences should be discussed rather than hidden.

## Why are the raw CSV files not committed?

The repository keeps the analysis code, result files and validation artifacts public while leaving the raw datasets outside version control. Reproduction instructions specify the required filenames and directory. The code resolves paths relative to the repository root.

## What is the strongest defensible summary of the repository?

The repository provides an inspectable chain from method to code to machine-readable results, tables, visual summaries, interpretation and limitations. It directly analyzes Green HRM only in the 320-record GHRM survey and uses the other three datasets as general HR benchmarks. The main conclusions are stated with uncertainty and class-imbalance limitations rather than from accuracy or point estimates alone.

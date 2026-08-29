# Mathematical definitions used in the analysis

This note records the formulas behind the metrics and procedures used in the repository. It is intended to make the implementation easier to audit against the code and the thesis methodology.

## 1. Construct scores in the GHRM survey

Each GHRM construct score is the row-wise mean of its selected questionnaire items:

\[
S_i = \frac{1}{m}\sum_{j=1}^{m} x_{ij}
\]

where \(x_{ij}\) is respondent \(i\)'s score on item \(j\), and \(m\) is the number of retained items for that construct. The implementation is visible in `src/run_ghrm.py` and `src/run_ghrm_reliability.py`.

The constructs used are GRS, GTD, GPA, GCM, GEE and FEP. The Base regression uses GRS+GTD+GPA+GCM to predict FEP; GEE+ adds GEE as a fifth predictor.

## 2. Standardization

For variables that are standardized, the usual z-score transformation is applied:

\[
z = \frac{x-\mu}{\sigma}
\]

where \(\mu\) and \(\sigma\) are estimated from the relevant training data when standardization is part of a supervised pipeline.

## 3. Classification metrics

For binary classification, TP, TN, FP and FN denote true positives, true negatives, false positives and false negatives.

**Accuracy**

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
\]

**Precision**

\[
Precision = \frac{TP}{TP+FP}
\]

**Recall**

\[
Recall = \frac{TP}{TP+FN}
\]

**F1-score**

\[
F1 = 2\frac{Precision\times Recall}{Precision+Recall}
\]

F1 is the tuning criterion for the three classification datasets because their positive classes are not balanced with the negative class.

## 4. Regression metrics

For observations \(y_i\) and predictions \(\hat y_i\):

**Mean Absolute Error**

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

**Root Mean Squared Error**

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
\]

**Coefficient of determination**

\[
R^2 = 1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}
\]

The GHRM grid search is optimized on cross-validated RMSE, while R², RMSE and MAE are reported on the fixed 64-record holdout.

## 5. Decision Trees

For classification trees, split quality is based on class impurity. For Gini impurity:

\[
Gini(t)=1-\sum_{c=1}^{C}p(c|t)^2
\]

A candidate split is preferred when it produces a larger weighted reduction in impurity.

For regression trees, node impurity is based on squared deviation around the node mean:

\[
MSE(t)=\frac{1}{N_t}\sum_{i\in t}(y_i-\bar y_t)^2
\]

The resulting tree partitions the feature space recursively until stopping rules or fitted hyperparameters prevent further splitting.

## 6. Random Forest

A Random Forest fits multiple decision trees to resampled data and feature subsets. For classification, the ensemble prediction can be written as a majority vote:

\[
\hat y = mode\{T_1(x),T_2(x),\ldots,T_B(x)\}
\]

For regression, the tree predictions are averaged:

\[
\hat y = \frac{1}{B}\sum_{b=1}^{B}T_b(x)
\]

where \(B\) is the number of trees.

## 7. Linear SVM and LinearSVR

The linear decision function has the form:

\[
f(x)=w^Tx+b
\]

For classification, the soft-margin objective balances margin size against classification violations:

\[
\min_{w,b,\xi}\frac{1}{2}\|w\|^2+C\sum_i\xi_i
\]

subject to the standard margin constraints.

LinearSVR uses an epsilon-insensitive loss, where small residuals within \(\epsilon\) are not penalized. The parameter \(C\) controls the trade-off between model flatness and deviations outside the epsilon tube.

## 8. Multilayer Perceptron

A hidden layer applies a linear transformation followed by a nonlinear activation:

\[
h=\phi(Wx+b)
\]

For multiple layers, the same operation is composed repeatedly. Model weights are learned by minimizing the task loss with regularization controlled by `alpha`. The exact candidate architectures and learning rates searched in this repository are stored in the JSON reports and source code.

## 9. K-Means

K-Means minimizes within-cluster squared distance:

\[
SSE=\sum_{k=1}^{K}\sum_{x_i\in C_k}\|x_i-\mu_k\|^2
\]

where \(\mu_k\) is the centroid of cluster \(C_k\).

### Silhouette coefficient

For observation \(i\):

\[
s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}
\]

where \(a(i)\) is mean within-cluster distance and \(b(i)\) is the smallest mean distance to another cluster. Values closer to 1 indicate clearer separation.

### Davies-Bouldin index

\[
DB=\frac{1}{K}\sum_{i=1}^{K}\max_{j\neq i}\frac{S_i+S_j}{M_{ij}}
\]

where \(S_i\) measures within-cluster scatter and \(M_{ij}\) is the distance between cluster centroids. Lower values are preferred.

The repository selects k by maximum Silhouette over k=2..7 and reports SSE and Davies-Bouldin as supplementary evidence.

## 10. Cronbach's alpha

For a construct with \(k\) items:

\[
\alpha=\frac{k}{k-1}\left(1-\frac{\sum_{j=1}^{k}\sigma_j^2}{\sigma_T^2}\right)
\]

where \(\sigma_j^2\) is item variance and \(\sigma_T^2\) is the variance of the summed scale. The committed reliability output reports alpha for all six GHRM constructs.

## 11. Paired bootstrap for Base vs GEE+

The GHRM comparison repeatedly samples the same test-record indices with replacement. For each resample, the metric difference is computed as:

\[
\Delta M = M_{GEE+}-M_{Base}
\]

For RMSE and MAE, negative differences favor GEE+; for R², positive differences favor GEE+. The reported 95% interval is the empirical percentile interval from 4,000 paired resamples. If the interval includes zero, the repository does not describe the change as statistically reliable.

## 12. McNemar paired test

For two classifiers evaluated on the same records, let \(b\) be cases correct only for model A and \(c\) cases correct only for model B. When the continuity-corrected chi-square approximation is used:

\[
\chi^2=\frac{(|b-c|-1)^2}{b+c}
\]

For small discordant counts, the implementation uses the exact binomial alternative. McNemar therefore tests whether the two classifiers have different error behavior on the same held-out observations, rather than comparing independent accuracy estimates.

## 13. Scope of inference

These formulas define prediction, clustering, reliability and paired-comparison procedures. None of them by itself establishes causality. IBM Attrition, Job Change and Promotion are general HR benchmark datasets; direct Green HRM interpretation is reserved for the GHRM survey variables.
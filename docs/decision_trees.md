# Decision-tree evidence and interpretation

This note gives the reader a compact audit trail for the fitted decision trees. The PNG tree plots are the visual representation; the numerical complexity reported here comes from the committed result files.

| Analysis | Tree type | Selected structure | Main interpretation |
|---|---|---:|---|
| IBM Attrition | Classification | depth 6, 38 leaves | A bounded tree that trades some accuracy for substantially better positive-class recall than the Random Forest in this run. |
| Job Change | Classification | depth 6, 46 leaves | A similarly bounded tree with F1=0.5758 and high recall (0.7853). |
| Employee Promotion | Classification | depth 63, 4,855 leaves | The unrestricted selected tree is extremely complex. Its size is reported as an overfitting warning rather than hidden. |
| GHRM | Regression | depth 4, 11 leaves | A comparatively small regression tree for FEP, consistent with an interpretable low-depth partition of the GHRM construct space. |

## IBM Attrition

**Figure:** `figures/ibm_attrition_decision_tree.png`

The selected hyperparameters are `max_depth=6` and `min_samples_split=10`. On the held-out 294 records the tree has Accuracy=0.7483, Precision=0.3291, Recall=0.5532 and F1=0.4127.

The important point is not that a single tree is universally the best model. Its value in the repository is interpretability: every prediction follows a visible sequence of feature thresholds. This makes it possible to inspect whether a split is plausible and whether a branch is supported by enough observations.

## Job Change

**Figure:** `figures/job_change_decision_tree.png`

The selected tree uses `max_depth=6` and `min_samples_split=2`, producing 46 leaves. Held-out performance is Accuracy=0.7116, Precision=0.4545, Recall=0.7853 and F1=0.5758.

The relatively high recall means the tree identifies many positive Job Change cases, but at the cost of more false positives than models with higher precision. The confusion-matrix figure should therefore be read together with the tree plot.

## Employee Promotion

**Figure:** `figures/promotion_decision_tree.png`

The grid search selected an unrestricted tree (`max_depth=None`, `min_samples_split=2`). The fitted structure has depth 63 and 4,855 leaves. Held-out F1 is 0.4306.

This is a useful negative result: the search criterion selected a very large tree, but the resulting model is difficult to interpret and is at clear risk of fitting sample-specific detail. The repository therefore reports both the tree's performance and its complexity. A simpler model would be preferable if interpretability or stability were the primary objective.

## GHRM regression tree

**Figure:** `figures/ghrm_decision_tree.png`

For the Base specification, the selected DecisionTreeRegressor uses `max_depth=4` and `min_samples_leaf=5`; its published held-out R² is 0.4361, RMSE 0.4876 and MAE 0.3719. The GEE+ version uses the same selected depth/leaf settings and produces R²=0.4295.

The tree is used to show how combinations of GRS, GTD, GPA and GCM scores partition the FEP outcome. Adding GEE does not improve the held-out R² for this model in the committed run.

## How to read a classification tree node

A classification node generally contains a feature threshold, sample count and class impurity. A branch such as `feature <= threshold` sends an observation left; otherwise it goes right. The final leaf returns the class supported by the observations that reached it.

For Gini impurity:

\[
Gini(t)=1-\sum_c p(c|t)^2
\]

A lower value means the node is more concentrated in one class.

## How to read the GHRM regression tree

A regression node predicts the mean target value of observations reaching that node. Candidate splits are selected by reduction in squared-error impurity. Leaves therefore represent local FEP predictions for regions of the construct-score space.

## Limits of interpretation

A decision-tree split is a predictive rule learned from the data; it is not a causal statement. This distinction is particularly important for the three public benchmark datasets, which do not directly measure Green HRM. The GHRM tree is directly tied to the thesis constructs, but it is still an observational predictive model.
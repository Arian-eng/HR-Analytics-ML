# Analysis methodology

## 1. Record data identity

The pipeline starts from four named files. Before processing, it records SHA-256, dimensions, exact duplicates, missingness, and target completeness. These checks identify the exact source version behind every result.

## 2. Prepare variables

For the three classification tasks, record identifiers and the target are removed from model inputs. Constant IBM fields are also removed. For GHRM, six composites are calculated from predefined survey items.

Preprocessing is inside each model `Pipeline`:

- numeric missing values: training-set median;
- categorical missing values: training-set mode;
- categorical variables: one-hot encoding;
- scaling: `StandardScaler` for SVM, MLP, and all GHRM regressors;
- unseen test categories: ignored by the encoder.

Because preprocessing is fitted after the split, the held-out set cannot influence imputation, encoding, scaling, or model selection.

## 3. Split training and test data

Each dataset uses one fixed 80/20 split with `random_state=42`. Local row membership is saved in `results/splits/`. Classification splits are stratified to preserve the positive-class rate; GHRM regression is not stratified.

The held-out set is never used for hyperparameter search and is evaluated only after model selection.

## 4. Classification

IBM, Job Change, and Promotion are modeled independently. Hyperparameters are selected by positive-class F1 in three-fold stratified cross-validation.

| Model | Search space |
|---|---|
| Random Forest | 200/400 trees, unlimited/20 depth, 1/2 minimum leaf samples, `max_features=sqrt` |
| Decision Tree | Gini/entropy, depth 5/8/12, minimum leaf size 2/5/10 |
| Linear SVM | `C` = 0.1, 1, or 10 |
| MLP | One or two hidden layers, 50 or 100 units, ReLU/Tanh, two alpha values and learning rates; eight deterministic random combinations |

Random Forest, Decision Tree, and Linear SVM use `class_weight=balanced`. MLP uses early stopping and a 15% internal validation split.

Held-out metrics are Accuracy, Precision, Recall, and F1. Each metric receives a paired 2,000-resample percentile bootstrap interval with seed 42.

Each confusion matrix is saved separately. All six pairwise comparisons among the four classifiers use McNemar's test. The exact binomial method is used below 25 discordant predictions; otherwise the continuity-corrected chi-square approximation is used.

## 5. Main GHRM regression

Two variants use identical training and test rows:

- `Base`: `GRS`, `GTD`, `GPA`, and `GCM`;
- `GEE+`: the Base predictors plus `GEE`.

Models are tuned on training data with five-fold `KFold` and RMSE.

| Model | Search space |
|---|---|
| Random Forest Regressor | 200/400 trees, unlimited/10 depth, minimum leaf size 1/2, all/square-root features |
| Decision Tree Regressor | Depth 3/5/8/unlimited and minimum leaf size 2/5/10 |
| LinearSVR | `C` = 0.1/1/10, epsilon = 0/0.1/0.2, maximum 100,000 iterations |
| MLPRegressor | 20, 50, or 50–25 units; two alpha values and learning rates; eight deterministic combinations; maximum 3,000 iterations |

R², MAE, and RMSE are evaluated on the same 64-record holdout, with 4,000 bootstrap resamples for 95% intervals. For stability analysis, each selected pipeline is also evaluated on all 320 records with five-fold, five-repeat `RepeatedKFold`.

Tuning and repeated-CV convergence warnings are counted in each model's `convergence_warnings.json`; reaching an iteration limit is never silently hidden.

Base and GEE+ predictions are paired on the same 64 held-out rows. Differences in R², MAE, and RMSE receive 4,000-resample paired bootstrap intervals. An interval containing zero is not treated as reliable improvement in this sample.

## 6. Clustering and hidden patterns

K-Means runs separately for each dataset. Identifiers, targets, and constant columns are excluded. Only nonconstant numeric features are used; missing values are median-imputed and features are standardized.

For GHRM, clustering uses only `GRS`, `GTD`, `GPA`, `GCM`, and `GEE`. `FEP` is deliberately excluded and is examined only after clustering to describe environmental-performance differences.

For every k from 2 through 7, K-Means uses 20 initializations and at most 300 iterations. Selection metrics are:

- Inertia/SSE: within-cluster compactness; lower is better;
- Silhouette: cohesion and separation; higher is better;
- Davies–Bouldin: relative overlap; lower is better.

The maximum-Silhouette k is selected. For large datasets, Silhouette uses a deterministic 5,000-record sample to bound memory use. Local runs save membership, sizes, numeric profiles, categorical modes, standardized feature differences, and post-clustering target distributions.

Target means or shares never influence cluster selection. Any target difference between clusters is descriptive, not causal evidence.

## 7. Algorithm explainability

- Permutation importance is saved for every model.
- Decision Tree exports include every node, child, split feature, threshold, impurity, sample count, class/prediction value, and positive-class leaf probability, plus full text rules.
- Random Forest exports include the depth and leaf count of every tree. Tree 0 is exported completely as a representative example, with a readable plot of its top four levels.
- Linear SVM and LinearSVR export coefficients for every transformed feature.
- MLP diagnostics record architecture, selected hyperparameters, iteration count, and final loss.

Exporting every node of hundreds of forest trees would add substantial repetition without proportional interpretive value. The repository therefore reports every tree's complexity and one complete representative structure.

## 8. Reproduction and independent validation

`run_all.py` removes and rebuilds only the exact `results/` directory at the beginning of a run. Python and library versions, runtime, seed, and source hashes are recorded in `run_manifest.json`.

`scripts/validate_results.py` independently recomputes the main metrics from saved model evidence rather than trusting summary tables. This prevents stale or manually entered thesis values from being presented as live pipeline results.

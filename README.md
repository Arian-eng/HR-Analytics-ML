# HR Analytics ML

Reproducible machine-learning analysis for the master's thesis on Green HRM and HR analytics.

## Study design

The thesis uses **four independent public datasets**. They are never merged and their row counts are not treated as one sample:

1. **IBM HR Analytics Employee Attrition & Performance** — target: `Attrition` — binary classification
2. **HR Analytics: Job Change of Data Scientists** — target: `target` — binary classification
3. **Employee Promotion Prediction** — target: `is_promoted` — binary classification
4. **GHRM–Environmental Performance** (`HRM DATASETS.csv`) — target: continuous `FEP` — regression and exploratory clustering

The older UCI dataset is not part of the final pipeline.

## Final Chapter 3 algorithms

### Exploratory clustering
- `KMeans`
- evaluated independently for **each of the four datasets** with `k=2...7`
- Inertia / SSE
- Silhouette Score
- Davies-Bouldin Index
- the target and identifier fields are excluded; for GHRM, `FEP` is excluded from cluster formation
- clusters are reported only as `Cluster 0`, `Cluster 1`, etc.; no cluster is labelled green/non-green

### Binary classification
For IBM HR, Job Change and Employee Promotion:
- `RandomForestClassifier`
- `DecisionTreeClassifier`
- `LinearSVC`
- `MLPClassifier`

### GHRM regression
For continuous `FEP`:
- `RandomForestRegressor`
- `DecisionTreeRegressor`
- `LinearSVR`
- `MLPRegressor`

The base GHRM regression uses `GRS`, `GTD`, `GPA` and `GCM`. A supplementary run adds `GEE` to assess its predictive contribution. This is predictive analysis, not a test of mediation or causality.

## Experimental design

### Classification
- 80% train / 20% test
- stratified split
- `random_state=42`
- 3-fold `StratifiedKFold` cross-validation on training data only
- `GridSearchCV` for Random Forest, Decision Tree and Linear SVM
- `RandomizedSearchCV` for MLP
- F1 score of the positive class is used for hyperparameter selection
- preprocessing is fitted inside the model pipeline using training data only
- identifier columns are removed before modelling
- categorical variables use `OneHotEncoder(handle_unknown="ignore")`
- missing values are imputed from training data only
- numeric features are standardized for Linear SVM and MLP

### GHRM regression
- 80% train / 20% test without stratification
- 3-fold `KFold` cross-validation on training data only
- preprocessing remains inside the model pipeline
- RMSE is the tuning objective
- final test metrics: `R²`, `RMSE`, `MAE`

## Classification evaluation

For every classifier and dataset the held-out test set is used once for:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

McNemar's test compares the same held-out predictions for:
- Random Forest vs Decision Tree
- Random Forest vs Linear SVM
- Random Forest vs MLP

No model is assumed to be best before evaluation.

## Outputs

`run_all.py` generates:

```text
outputs/
├── ibm/
├── job_change/
├── promotion/
├── ghrm/
└── model_metrics.csv
```

The dataset-specific folders contain test predictions, confusion matrices, K-Means metrics, cluster sizes and McNemar comparisons. The GHRM folder also contains regression metrics, predictions and cluster profiles. `figures/` contains the K-Means diagnostic plots.

## Project structure

```text
HR-Analytics-ML/
├── data/                  # Local datasets; not committed by default
├── src/
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── classification.py
│   ├── evaluation.py
│   └── green_hrm_analysis.py
├── outputs/               # Generated results
├── figures/               # Generated figures
├── run_all.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproducibility

Place these four current files in `data/`:

```text
WA_Fn-UseC_-HR-Employee-Attrition (3)(2).csv
aug_train(5).csv
train_LZdllcl.csv
HRM DATASETS.csv
```

Then install `requirements.txt` and run:

```bash
python run_all.py
```

Raw datasets are intentionally not committed unless their redistribution licenses permit it.

## GHRM interpretation

The three general HR datasets do not directly measure Green HRM or environmental behaviour. Their HR variables are therefore interpreted as theoretically related proxy/predictive features only. The fourth GHRM dataset contains direct GHRM dimensions (`GRS`, `GTD`, `GPA`, `GCM`, `GEE`) and continuous environmental performance (`FEP`). The machine-learning analysis does not establish causal or mediation effects.

# HR Analytics ML

Reproducible machine-learning analysis for the master's thesis on Green HRM and HR analytics.

## Scope

The project analyzes three **independent** public HR datasets. They are never merged and their row counts are not treated as one sample:

1. IBM HR Analytics Employee Attrition & Performance — target: `Attrition`
2. HR Analytics: Job Change of Data Scientists — target: `target`
3. Employee Promotion Prediction — target: `is_promoted`

The older UCI dataset and regression analysis are excluded from the final Chapter 4 pipeline.

## Final Chapter 4 algorithms

### Exploratory clustering
- K-Means, evaluated independently for each dataset with `k=2...7`
- Inertia / SSE
- Silhouette Score
- Davies-Bouldin Index

Clusters are reported only as `Cluster 0`, `Cluster 1`, etc. No cluster is labelled green/non-green.

### Binary classification
- Random Forest (`RandomForestClassifier`)
- Decision Tree (`DecisionTreeClassifier`)
- Linear SVM (`LinearSVC`)
- Multilayer Perceptron (`MLPClassifier`)

XGBoost, GMM, Logistic Regression, Gradient Boosting, KNN and Linear Regression are not part of the final executable Chapter 4 pipeline.

## Experimental design

Each dataset is processed independently using:

- 80% train / 20% test split
- `stratify=y`
- `random_state=42`
- 3-fold `StratifiedKFold` cross-validation on **training data only**
- `GridSearchCV` for Random Forest, Decision Tree and Linear SVM
- `RandomizedSearchCV` for MLP
- F1 score of the positive class as the hyperparameter-selection criterion
- Preprocessing is fitted inside the model pipeline using training data only
- Identifier columns are removed before modelling
- Categorical variables use `OneHotEncoder(handle_unknown="ignore")`
- Missing values are imputed using training-fold statistics
- Numeric features are standardized for Linear SVM and MLP; RF and DT do not require scaling

## Evaluation

For every model and dataset the final held-out test set is used once for:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

For each dataset, McNemar's exact test compares RF against DT, Linear SVM and MLP using the same held-out test predictions.

## Outputs

`run_all.py` writes:

```text
outputs/
├── ibm/
├── job_change/
├── promotion/
└── model_metrics.csv
```

Each dataset directory contains model test predictions, McNemar comparisons, K-Means metrics, cluster sizes and cluster summaries. `figures/` contains Elbow, Silhouette and Davies-Bouldin plots.

## Project structure

```text
HR-Analytics-ML/
├── data/                  # Local datasets; not committed by default
├── src/
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── classification.py
│   └── evaluation.py
├── outputs/               # Generated results; local unless explicitly committed
├── figures/               # Generated figures
├── run_all.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproducibility

Install `requirements.txt`, place the three current CSV files in `data/`, and run:

```bash
python run_all.py
```

Raw datasets are intentionally not committed unless their redistribution licenses permit it.

## GHRM interpretation

The three current HR datasets do not directly measure Green HRM or environmental behaviour. Any relationship to GHRM is therefore interpreted theoretically and must not be presented as a direct measured GHRM variable.

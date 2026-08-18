# HR Analytics ML

Machine-learning analysis for the master's thesis on Green HRM and HR analytics.

## Scope

The project analyzes three public HR datasets:

1. IBM HR Employee Attrition & Performance — target: `Attrition`
2. HR Analytics: Job Change of Data Scientists — target: `target`
3. Employee Performance and Productivity — target: `KPIs_met_more_than_80`

### Classification algorithms

The Chapter 3 classification benchmark includes eight algorithms:

1. Random Forest
2. Decision Tree
3. Linear SVM (`LinearSVC`)
4. Neural Network (`MLPClassifier`)
5. Logistic Regression
6. Gradient Boosting
7. K-Nearest Neighbors (KNN)
8. XGBoost

K-Means is evaluated separately for clustering. The Green HRM-specific regression analysis additionally uses Linear Regression and Random Forest Regressor.

The analysis includes preprocessing, train/test evaluation, 3-fold stratified cross-validation with GridSearchCV, Accuracy, weighted Precision/Recall/F1, positive-class metrics for binary targets, confusion matrices, K-Means/Silhouette evaluation, and permutation feature importance.

> **Methodological note:** the three general HR datasets do not directly measure Green HRM practices. Their outputs are therefore interpreted as HR-performance/behavioral proxy indicators rather than direct measures of green behavior.

## Project structure

```text
HR-Analytics-ML/
├── data/                  # Local datasets (not committed by default)
├── src/                   # Analysis source code
├── notebooks/             # Exploratory analysis
├── results/               # Generated tables and reports
├── figures/               # Generated charts
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproducibility

Create a Python environment and install the dependencies listed in `requirements.txt`. Place the required CSV files under `data/`, then run the analysis script from the project root.

The script resolves the supplied filename variants for the three datasets and writes `results/model_comparison.csv`, confusion-matrix outputs, K-Means metrics, feature-importance tables, and `results/chapter_4_results.xlsx`.

## Data

The repository intentionally does not include raw datasets unless their redistribution license permits it. Dataset filenames and expected targets are documented in the analysis code.

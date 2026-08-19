# HR Analytics ML

Machine-learning analysis for the master's thesis on Green HRM and HR analytics.

## Scope

The project analyzes the three current public HR datasets supplied for the thesis:

1. IBM HR Employee Attrition & Performance — target: `Attrition`
2. HR Analytics: Job Change of Data Scientists (`aug_train(5).csv`) — target: `target`
3. Current employee promotion dataset (`train_LZdllcl.csv`) — target: `is_promoted`

> **Advisor note:** Chapter 3 currently describes dataset 3 as an Employee Performance/Productivity dataset and specifies a continuous-performance regression analysis. The current supplied file `train_LZdllcl.csv` instead has the binary target `is_promoted`. The repository therefore does **not** fabricate a regression result for this dataset. The thesis Chapter 3/4 wording must be reconciled with the actual dataset before the final defense version.

### Classification algorithms explicitly selected in Chapter 3

1. Random Forest
2. Decision Tree
3. Support Vector Machine with RBF kernel (`SVC`)
4. Neural Network / MLP (`MLPClassifier`)
5. XGBoost

K-Means is evaluated separately for clustering. Linear Regression is supported only for a dataset that actually contains a continuous performance target; it is not applied to the current `is_promoted` target.

The implementation follows the Chapter 3 evaluation design: 80/20 stratified train/test split, 5-fold stratified cross-validation, GridSearchCV, Accuracy, weighted Precision/Recall/F1, positive-class metrics for binary targets, confusion matrices, K-Means/Silhouette evaluation, and Random Forest permutation feature importance.

### Chapter 3 hyperparameter alignment

- Random Forest: `n_estimators` up to 500, `max_depth` up to 50, `max_features` = `sqrt`/`log2`.
- Decision Tree: `max_depth` up to 20, `criterion` = `gini`/`entropy`.
- SVM: **RBF kernel**, `C` in {0.1, 1, 10}, `gamma` in {`scale`, `auto`}.
- MLP: hidden-layer, learning-rate and activation settings based on the ranges documented in Chapter 3.
- XGBoost: `n_estimators`, `max_depth`, and `learning_rate` searched using GridSearchCV.

> **Methodological note:** the three general HR datasets do not directly measure Green HRM practices. Their outputs are interpreted as HR-performance/behavioral proxy indicators rather than direct measures of green behavior.

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

Create a Python environment and install the dependencies listed in `requirements.txt`. Place the three current CSV files under `data/`, then run the analysis script from the project root.

The script resolves the supplied filename variants and writes `results/model_comparison.csv`, confusion-matrix outputs, K-Means metrics, feature-importance tables, and `results/chapter_4_results.xlsx`.

## Data

Raw datasets are intentionally not committed to the repository unless their redistribution licenses permit it. The exact expected filenames, targets, preprocessing rules, and model settings are documented in `src/thesis_analysis.py`.

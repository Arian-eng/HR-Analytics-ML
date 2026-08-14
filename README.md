# HR Analytics ML

Machine-learning analysis for the master's thesis on Green HRM and HR analytics.

## Scope

The project analyzes three public HR datasets:

1. IBM HR Employee Attrition & Performance — target: `Attrition`
2. HR Analytics: Job Change of Data Scientists — target: `target`
3. Employee Performance and Productivity — target: `KPIs_met_more_than_80`

The analysis includes preprocessing, K-Means clustering, Random Forest, Decision Tree, SVM, Neural Network, cross-validation, model metrics, confusion matrices, and permutation feature importance.

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

## Data

The repository intentionally does not include raw datasets unless their redistribution license permits it. Dataset filenames and expected targets are documented in the analysis code.

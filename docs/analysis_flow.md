# Analysis flow and evidence chain

The following diagrams summarize how the repository moves from raw data to reported findings. They are documentation diagrams, not substitutes for the fitted model figures.

## End-to-end workflow

```mermaid
flowchart LR
    A[Four independent datasets] --> B[Data checks]
    B --> C[80/20 train-test split]
    C --> D1[Classification pipeline]
    C --> D2[GHRM regression pipeline]
    B --> D3[K-Means pipeline]
    B --> D4[GHRM reliability]

    D1 --> E1[Decision Tree]
    D1 --> E2[Random Forest]
    D1 --> E3[LinearSVC]
    D1 --> E4[MLPClassifier]

    D2 --> F1[Random Forest Regressor]
    D2 --> F2[Decision Tree Regressor]
    D2 --> F3[LinearSVR]
    D2 --> F4[MLPRegressor]

    E1 --> G1[Accuracy / Precision / Recall / F1]
    E2 --> G1
    E3 --> G1
    E4 --> G1
    G1 --> G2[Bootstrap 95% CI]
    G1 --> G3[McNemar paired tests]
    G1 --> G4[Confusion matrices]

    F1 --> H1[R² / RMSE / MAE]
    F2 --> H1
    F3 --> H1
    F4 --> H1
    H1 --> H2[Base vs GEE+ paired bootstrap]

    D3 --> I1[SSE]
    D3 --> I2[Silhouette]
    D3 --> I3[Davies-Bouldin]
    I1 --> I4[Selected k]
    I2 --> I4
    I3 --> I4

    D4 --> J1[Cronbach alpha]
    D4 --> J2[Descriptives]
    D4 --> J3[Correlation matrix]

    G2 --> K[Pattern table]
    G3 --> K
    G4 --> K
    H2 --> K
    I4 --> K
    J1 --> K
    J3 --> K

    K --> L[Evidence-based conclusion]
```

## Dataset-role separation

```mermaid
flowchart TB
    A[HRM_DATASETS.csv<br/>n=320] --> B[Direct GHRM evidence]
    B --> C[GRS / GTD / GPA / GCM / GEE]
    C --> D[FEP regression, reliability, correlations, clustering]

    E[IBM Attrition<br/>n=1,470] --> H[General HR benchmark evidence]
    F[Job Change<br/>n=19,158] --> H
    G[Employee Promotion<br/>n=54,808] --> H
    H --> I[Classification + K-Means methodology]

    D --> J[Thesis interpretation]
    I --> J
```

The four datasets are not merged record-by-record. The second diagram is included specifically to prevent a reader from interpreting the three benchmark targets as direct Green HRM measures.

## Classification decision logic

```mermaid
flowchart LR
    A[Training set] --> B[3-fold Stratified CV]
    B --> C[Limited hyperparameter grid]
    C --> D[Select by positive-class F1]
    D --> E[Refit on full training set]
    E --> F[Evaluate once on held-out test set]
    F --> G[Metrics + bootstrap CI]
    F --> H[Confusion matrix]
    F --> I[McNemar pairwise comparisons]
```

## GHRM Base vs GEE+ logic

```mermaid
flowchart LR
    A[320 GHRM records] --> B[256 train / 64 test]
    B --> C1[Base: GRS+GTD+GPA+GCM]
    B --> C2[GEE+: Base+GEE]
    C1 --> D[5-fold RMSE tuning]
    C2 --> D
    D --> E[R² / RMSE / MAE on same 64 records]
    E --> F[4,000 paired bootstrap resamples]
    F --> G[Difference + 95% interval]
```

## Evidence chain used in the repository

```mermaid
flowchart LR
    A[src/*.py] --> B[results/*.json]
    B --> C[results/tables/*.csv]
    B --> D[figures/*]
    C --> E[MODEL_RESULTS.md]
    D --> F[figures/README.md]
    E --> G[PATTERNS_AND_FINDINGS.md]
    F --> G
    G --> H[docs/conclusion.md]
    A --> I[validation/validation_report.md]
    B --> I
```

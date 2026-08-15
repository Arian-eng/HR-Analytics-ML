# -*- coding: utf-8 -*-
"""Reproducible Chapter 4 HR machine-learning analysis.

The raw CSV files stay local. The script accepts the exact filenames supplied
for this thesis and writes model metrics, confusion matrices, K-Means
validation, feature importance, and an Excel workbook under results/.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, silhouette_score)
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, OUTPUT_DIR, FIG_DIR = ROOT / "data", ROOT / "results", ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

DATASETS = {
    "IBM_HR_Attrition": {
        "files": ["WA_Fn-UseC_-HR-Employee-Attrition.csv", "WA_Fn-UseC_-HR-Employee-Attrition (3)(2).csv"],
        "target": "Attrition", "ids": ["EmployeeNumber", "EmployeeCount", "StandardHours"]},
    "Job_Change": {
        "files": ["aug_train.csv", "aug_train(4).csv"],
        "target": "target", "ids": ["enrollee_id"]},
    "Employee_Performance": {
        "files": ["Uncleaned_employees_final_dataset.csv", "Uncleaned_employees_final_dataset (1)(2).csv"],
        "target": "KPIs_met_more_than_80", "ids": ["employee_id"]},
}


def resolve_file(candidates):
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError("Dataset not found: " + ", ".join(candidates))


def prepare(df, target, ids):
    y = LabelEncoder().fit_transform(df[target].astype(str))
    X = df.drop(columns=[target] + ids, errors="ignore").copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.drop(columns=[c for c in X if X[c].nunique(dropna=False) <= 1])
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return X, y, prep


def save_cm(cm, title, filename):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.xlabel("Predicted class"); plt.ylabel("Actual class")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout(); plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight"); plt.close()


def kmeans_analysis(df, name, target, ids):
    numeric = df.drop(columns=[target] + ids, errors="ignore").select_dtypes(include=np.number)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(numeric.median())
    numeric = numeric.drop(columns=[c for c in numeric if c.lower() in {"id", "employee_id", "employee_number", "enrollee_id"}], errors="ignore")
    if numeric.shape[1] < 2:
        return None
    Z = StandardScaler().fit_transform(numeric)
    ks = range(2, min(7, len(Z) - 1) + 1)
    rows = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(Z)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(Z, labels)})
    metrics = pd.DataFrame(rows)
    best_k = int(metrics.loc[metrics.silhouette.idxmax(), "k"])
    metrics.to_csv(OUTPUT_DIR / f"{name}_kmeans.csv", index=False, encoding="utf-8-sig")
    return best_k, metrics


# The original full SVC RBF search was computationally excessive for 19k rows.
# LinearSVC is still an SVM classifier and makes the experiment reproducible.
MODEL_SPECS = {
    "Random Forest": (RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
                      {"n_estimators": [100, 200], "max_depth": [None, 10]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=42, class_weight="balanced"),
                      {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]}),
    "SVM": (LinearSVC(class_weight="balanced", random_state=42, dual="auto"),
            {"C": [0.1, 1, 10]}),
    "Neural Network": (MLPClassifier(max_iter=250, early_stopping=True, random_state=42),
                       {"hidden_layer_sizes": [(50,), (100,)], "alpha": [0.0001, 0.001]}),
}

all_results, cms, feature_tables, kmeans_tables = [], {}, {}, {}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for dataset_name, info in DATASETS.items():
    path = resolve_file(info["files"])
    df = pd.read_csv(path)
    X, y, prep = prepare(df, info["target"], info["ids"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # Fit preprocessing once on training data, then cross-validate models on
    # the transformed training matrix. This avoids repeating one-hot encoding.
    X_train_t = prep.fit_transform(X_train)
    X_test_t = prep.transform(X_test)

    for model_name, (model, grid) in MODEL_SPECS.items():
        search = GridSearchCV(model, grid, scoring="f1_weighted", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train_t, y_train)
        pred = search.predict(X_test_t)
        row = {
            "Dataset": dataset_name, "Model": model_name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, pred, average="weighted", zero_division=0),
            "CV_Best_F1": search.best_score_, "Best_Params": str(search.best_params_),
        }
        if dataset_name == "IBM_HR_Attrition":
            row["Attrition_Yes_Recall"] = recall_score(y_test, pred, pos_label=1, zero_division=0)
            row["Attrition_Yes_F1"] = f1_score(y_test, pred, pos_label=1, zero_division=0)
        all_results.append(row)
        cms[(dataset_name, model_name)] = confusion_matrix(y_test, pred)
        save_cm(cms[(dataset_name, model_name)], f"{dataset_name} - {model_name}",
                f"{dataset_name}_{model_name.replace(' ', '_')}_confusion_matrix.png")

        if model_name == "Random Forest":
            perm = permutation_importance(search.best_estimator_, X_test_t, y_test,
                                          n_repeats=5, random_state=42, n_jobs=-1,
                                          scoring="f1_weighted")
            fi = pd.DataFrame({"feature_index": np.arange(X_test_t.shape[1]),
                               "importance_mean": perm.importances_mean,
                               "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
            fi.to_csv(OUTPUT_DIR / f"{dataset_name}_feature_importance.csv", index=False)
            feature_tables[dataset_name] = fi

    km = kmeans_analysis(df, dataset_name, info["target"], info["ids"])
    if km is not None:
        kmeans_tables[dataset_name] = km

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([{"Dataset": d, "Model": m, "Confusion_Matrix": str(cm.tolist())}
               for (d, m), cm in cms.items()]).to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False, encoding="utf-8-sig")

with pd.ExcelWriter(OUTPUT_DIR / "chapter_4_results.xlsx", engine="openpyxl") as writer:
    results_df.to_excel(writer, index=False, sheet_name="Model Results")
    for (d, m), cm in cms.items():
        pd.DataFrame(cm).to_excel(writer, index=False, header=False, sheet_name=f"CM_{d[:12]}_{m[:10]}"[:31])
    for d, fi in feature_tables.items():
        fi.head(20).to_excel(writer, index=False, sheet_name=f"FI_{d}"[:31])
    for d, (_, metrics) in kmeans_tables.items():
        metrics.to_excel(writer, index=False, sheet_name=f"KMeans_{d}"[:31])

print(results_df.round(4).to_string(index=False))
for d, (best_k, _) in kmeans_tables.items():
    print(f"{d}: optimal k={best_k}")

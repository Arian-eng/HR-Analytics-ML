# -*- coding: utf-8 -*-
"""Full reproducible Chapter 4 HR Analytics pipeline.

Place the three raw CSV files in data/ and run:
    python src/thesis_analysis_full.py

The script produces classification metrics, positive-class metrics,
confusion matrices, Random Forest permutation importance, K-Means metrics,
and a consolidated Excel workbook under results/.
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

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

DATASETS = {
    "IBM_HR_Attrition": {
        "files": ["WA_Fn-UseC_-HR-Employee-Attrition (3)(2).csv", "WA_Fn-UseC_-HR-Employee-Attrition.csv"],
        "target": "Attrition", "ids": ["EmployeeNumber", "EmployeeCount", "StandardHours"]},
    "Job_Change": {
        "files": ["aug_train(4).csv", "aug_train.csv"],
        "target": "target", "ids": ["enrollee_id"]},
    "Employee_Performance": {
        "files": ["Uncleaned_employees_final_dataset (1)(2).csv", "Uncleaned_employees_final_dataset.csv"],
        "target": "KPIs_met_more_than_80", "ids": ["employee_id"]},
}

MODELS = {
    "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
                      {"n_estimators": [100, 200], "max_depth": [None, 10]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
                      {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]}),
    # LinearSVC is used instead of RBF SVC because RBF SVC is impractical for the 19k-row Job Change data.
    "SVM": (LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, dual="auto"),
            {"C": [0.1, 1, 10]}),
    "Neural Network": (MLPClassifier(max_iter=250, early_stopping=True, random_state=RANDOM_STATE),
                       {"hidden_layer_sizes": [(50,), (100,)], "alpha": [0.0001, 0.001]}),
}


def find_file(names):
    for name in names:
        p = DATA_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError("Missing dataset: " + " | ".join(names))


def preprocess(df, target, ids):
    le = LabelEncoder()
    y = le.fit_transform(df[target].astype(str))
    X = df.drop(columns=[target] + ids, errors="ignore").copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.drop(columns=[c for c in X.columns if X[c].nunique(dropna=False) <= 1])
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    transformer = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return X, y, transformer, le


def save_cm(cm, title, filename):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def run_kmeans(df, name, target, ids):
    X = df.drop(columns=[target] + ids, errors="ignore").select_dtypes(include=np.number)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    if X.shape[1] < 2 or len(X) < 4:
        return None
    Z = StandardScaler().fit_transform(X)
    rows = []
    for k in range(2, min(7, len(Z) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(Z)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z, labels)})
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / f"{name}_kmeans.csv", index=False, encoding="utf-8-sig")
    return result


def main():
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    results, cms, feature_tables, kmeans_tables = [], [], {}, {}

    for name, info in DATASETS.items():
        path = find_file(info["files"])
        df = pd.read_csv(path)
        X, y, transformer, le = preprocess(df, info["target"], info["ids"])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
        Xtr = transformer.fit_transform(Xtr)
        Xte = transformer.transform(Xte)

        for model_name, (model, grid) in MODELS.items():
            search = GridSearchCV(model, grid, scoring="f1_weighted", cv=cv, n_jobs=-1, refit=True)
            search.fit(Xtr, ytr)
            pred = search.predict(Xte)
            row = {
                "Dataset": name, "Model": model_name,
                "Accuracy": accuracy_score(yte, pred),
                "Precision_weighted": precision_score(yte, pred, average="weighted", zero_division=0),
                "Recall_weighted": recall_score(yte, pred, average="weighted", zero_division=0),
                "F1_weighted": f1_score(yte, pred, average="weighted", zero_division=0),
                "CV_Best_F1_weighted": search.best_score_,
                "Best_Params": str(search.best_params_),
            }
            if len(le.classes_) == 2:
                row.update({
                    "Positive_Class": str(le.classes_[1]),
                    "Positive_Precision": precision_score(yte, pred, pos_label=1, zero_division=0),
                    "Positive_Recall": recall_score(yte, pred, pos_label=1, zero_division=0),
                    "Positive_F1": f1_score(yte, pred, pos_label=1, zero_division=0),
                })
            results.append(row)
            cm = confusion_matrix(yte, pred)
            cms.append({"Dataset": name, "Model": model_name, "Confusion_Matrix": cm.tolist()})
            save_cm(cm, f"{name} - {model_name}", f"{name}_{model_name.replace(' ', '_')}_confusion_matrix.png")

            if model_name == "Random Forest":
                perm = permutation_importance(search.best_estimator_, Xte, yte, n_repeats=5,
                                              random_state=RANDOM_STATE, n_jobs=-1, scoring="f1_weighted")
                fi = pd.DataFrame({"feature": transformer.get_feature_names_out(),
                                   "importance_mean": perm.importances_mean,
                                   "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
                fi.to_csv(RESULTS_DIR / f"{name}_feature_importance.csv", index=False, encoding="utf-8-sig")
                feature_tables[name] = fi

        km = run_kmeans(df, name, info["target"], info["ids"])
        if km is not None:
            kmeans_tables[name] = km

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cms).to_csv(RESULTS_DIR / "confusion_matrices.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(RESULTS_DIR / "chapter_4_results.xlsx", engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Model Results")
        pd.DataFrame(cms).to_excel(writer, index=False, sheet_name="Confusion Matrices")
        for name, fi in feature_tables.items():
            fi.head(30).to_excel(writer, index=False, sheet_name=f"FI_{name}"[:31])
        for name, km in kmeans_tables.items():
            km.to_excel(writer, index=False, sheet_name=f"KMeans_{name}"[:31])

    print("\n=== MODEL RESULTS ===")
    print(results_df.round(4).to_string(index=False))
    print("\n=== K-MEANS ===")
    for name, km in kmeans_tables.items():
        best = km.loc[km["silhouette"].idxmax()]
        print(f"{name}: best k={int(best['k'])}, silhouette={best['silhouette']:.4f}")


if __name__ == "__main__":
    main()

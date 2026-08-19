# -*- coding: utf-8 -*-
"""Reproducible thesis ML pipeline aligned with Chapters 1-3.

Classification models explicitly selected in Chapter 3:
Random Forest, Decision Tree, RBF-SVM, MLP Neural Network, XGBoost.
K-Means is evaluated separately. Linear Regression is supported only when a
continuous performance target is explicitly present in a dataset.

The supplied current datasets are resolved by their current filenames. Raw
CSV files remain local and are never committed to the repository.
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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, silhouette_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, OUTPUT_DIR, FIG_DIR = ROOT / "data", ROOT / "results", ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# Current project datasets. The second file is the current HR promotion dataset
# supplied by the researcher; it is not the older 17,417-row file.
DATASETS = {
    "IBM_HR_Attrition": {
        "files": ["WA_Fn-UseC_-HR-Employee-Attrition (3)(2).csv", "WA_Fn-UseC_-HR-Employee-Attrition.csv"],
        "target": "Attrition", "ids": ["EmployeeNumber", "EmployeeCount", "StandardHours"]
    },
    "Job_Change": {
        "files": ["aug_train(5).csv", "aug_train.csv", "aug_train(4).csv"],
        "target": "target", "ids": ["enrollee_id"]
    },
    "Employee_Promotion": {
        "files": ["train_LZdllcl.csv", "Uncleaned_employees_final_dataset (1).csv", "Uncleaned_employees_final_dataset.csv"],
        "target": "is_promoted", "ids": ["employee_id"]
    },
}


def resolve_file(candidates):
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError("Dataset not found: " + ", ".join(candidates))


def prepare_classification(df, target, ids):
    enc = LabelEncoder()
    y = enc.fit_transform(df[target].astype(str))
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
    return X, y, prep, enc


def save_cm(cm, title, filename):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def kmeans_analysis(df, name, target, ids):
    numeric = df.drop(columns=[target] + ids, errors="ignore").select_dtypes(include=np.number)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(numeric.median())
    numeric = numeric.drop(columns=[c for c in numeric if c.lower() in {
        "id", "employee_id", "employee_number", "enrollee_id"
    }], errors="ignore")
    if numeric.shape[1] < 2:
        return None
    Z = StandardScaler().fit_transform(numeric)
    rows = []
    # Chapter 3 specifies a search up to k=8 and reports the selected k from
    # the Elbow/Silhouette procedure; no k is hard-coded as the final answer.
    for k in range(2, min(8, len(Z) - 1) + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(Z)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Z, labels)})
    metrics = pd.DataFrame(rows)
    best_k = int(metrics.loc[metrics.silhouette.idxmax(), "k"])
    metrics.to_csv(OUTPUT_DIR / f"{name}_kmeans.csv", index=False, encoding="utf-8-sig")
    return best_k, metrics


# Exact Chapter-3 model family and documented hyperparameter search ranges.
# SVM is RBF, not LinearSVC: Chapter 3 explicitly specifies RBF and C/gamma.
MODEL_SPECS = {
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
        {"n_estimators": [50, 100, 200, 300, 400, 500],
         "max_depth": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
         "max_features": ["sqrt", "log2"]}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"max_depth": [5, 10, 15, 20], "criterion": ["gini", "entropy"]}
    ),
    "SVM": (
        SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=42),
        {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
    ),
    "Neural Network": (
        MLPClassifier(max_iter=400, early_stopping=True, random_state=42),
        {"hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
         "learning_rate_init": [0.001, 0.01, 0.1],
         "activation": ["relu", "tanh"]}
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist"),
        {"n_estimators": [100, 200], "max_depth": [3, 5, 7],
         "learning_rate": [0.01, 0.1, 0.2]}
    ),
}

all_results, cms, feature_tables, kmeans_tables = [], {}, {}, {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for dataset_name, info in DATASETS.items():
    path = resolve_file(info["files"])
    df = pd.read_csv(path)
    X, y, prep, label_encoder = prepare_classification(df, info["target"], info["ids"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train_t = prep.fit_transform(X_train)
    X_test_t = prep.transform(X_test)

    for model_name, (model, grid) in MODEL_SPECS.items():
        search = GridSearchCV(model, grid, scoring="f1_weighted", cv=cv,
                              n_jobs=-1, refit=True)
        search.fit(X_train_t, y_train)
        pred = search.predict(X_test_t)
        row = {
            "Dataset": dataset_name, "Model": model_name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
            "CV_Best_F1_weighted": search.best_score_,
            "Best_Params": str(search.best_params_),
        }
        if len(label_encoder.classes_) == 2:
            row["Positive_Class"] = str(label_encoder.classes_[1])
            row["Positive_Precision"] = precision_score(y_test, pred, pos_label=1, zero_division=0)
            row["Positive_Recall"] = recall_score(y_test, pred, pos_label=1, zero_division=0)
            row["Positive_F1"] = f1_score(y_test, pred, pos_label=1, zero_division=0)
        all_results.append(row)
        cms[(dataset_name, model_name)] = confusion_matrix(y_test, pred)
        save_cm(cms[(dataset_name, model_name)],
                f"{dataset_name} - {model_name}",
                f"{dataset_name}_{model_name.replace(' ', '_')}_confusion_matrix.png")

        if model_name == "Random Forest":
            perm = permutation_importance(
                search.best_estimator_, X_test_t, y_test, n_repeats=5,
                random_state=42, n_jobs=-1, scoring="f1_weighted"
            )
            names = np.asarray(prep.get_feature_names_out())
            fi = pd.DataFrame({
                "feature": names,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std
            }).sort_values("importance_mean", ascending=False)
            fi.to_csv(OUTPUT_DIR / f"{dataset_name}_feature_importance.csv",
                      index=False, encoding="utf-8-sig")
            feature_tables[dataset_name] = fi

    km = kmeans_analysis(df, dataset_name, info["target"], info["ids"])
    if km is not None:
        kmeans_tables[dataset_name] = km

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([
    {"Dataset": d, "Model": m, "Confusion_Matrix": str(cm.tolist())}
    for (d, m), cm in cms.items()
]).to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False, encoding="utf-8-sig")

with pd.ExcelWriter(OUTPUT_DIR / "chapter_4_results.xlsx", engine="openpyxl") as writer:
    results_df.to_excel(writer, index=False, sheet_name="Model Results")
    for (d, m), cm in cms.items():
        pd.DataFrame(cm).to_excel(
            writer, index=False, header=False,
            sheet_name=f"CM_{d[:12]}_{m[:10]}"[:31]
        )
    for d, fi in feature_tables.items():
        fi.head(20).to_excel(writer, index=False, sheet_name=f"FI_{d}"[:31])
    for d, (_, metrics) in kmeans_tables.items():
        metrics.to_excel(writer, index=False, sheet_name=f"KMeans_{d}"[:31])

print(results_df.round(4).to_string(index=False))
for d, (best_k, _) in kmeans_tables.items():
    print(f"{d}: optimal k={best_k}")

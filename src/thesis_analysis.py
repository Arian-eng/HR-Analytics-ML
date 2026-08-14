# -*- coding: utf-8 -*-
"""Reproducible Chapter 4 HR machine-learning analysis.

Datasets:
- IBM HR Employee Attrition -> Attrition
- HR Analytics: Job Change of Data Scientists -> target
- Employee Performance -> KPIs_met_more_than_80

Outputs: model metrics, confusion matrices, K-Means diagnostics, and
permutation feature importance.

Methodological note: these public HR datasets do not directly measure Green
HRM. Their results are therefore interpreted as HR-performance/behavioral
proxy indicators rather than direct measures of green behavior.
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
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

DATASETS = {
    "IBM_HR_Attrition": {
        "file": DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "target": "Attrition",
        "id_columns": ["EmployeeNumber", "EmployeeCount", "StandardHours"],
    },
    "Job_Change": {
        "file": DATA_DIR / "aug_train.csv",
        "target": "target",
        "id_columns": ["enrollee_id"],
    },
    "Employee_Performance": {
        "file": DATA_DIR / "Uncleaned_employees_final_dataset.csv",
        "target": "KPIs_met_more_than_80",
        "id_columns": ["employee_id"],
    },
}


def encode_target(y):
    y = y.copy()
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        encoder = LabelEncoder()
        encoded = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index, name=y.name)
        return encoded, dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    return pd.to_numeric(y, errors="coerce"), None


def prepare_xy(df, target, id_columns):
    data = df.replace([np.inf, -np.inf], np.nan).copy()
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found.")
    y, mapping = encode_target(data[target])
    X = data.drop(columns=[target])
    X = X.drop(columns=[c for c in id_columns if c in X.columns])
    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant)
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return X, y, preprocessor, mapping


def save_confusion_matrix(cm, title, filename):
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


def run_kmeans(df, name, excluded):
    numeric = df.drop(columns=[c for c in excluded if c in df.columns], errors="ignore").select_dtypes(include=np.number).copy()
    numeric = numeric.drop(columns=[c for c in numeric.columns if c.lower() in {"employee_id", "employee_number", "enrollee_id", "id"}], errors="ignore")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(numeric.median())
    if numeric.shape[1] < 2 or len(numeric) < 3:
        return None
    Z = StandardScaler().fit_transform(numeric)
    ks = list(range(2, min(8, len(Z) - 1) + 1))
    inertias, silhouettes = [], []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(Z)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(Z, labels))
    best_k = ks[int(np.argmax(silhouettes))]
    for values, ylabel, suffix in [(inertias, "Within-cluster SSE", "elbow"), (silhouettes, "Silhouette Score", "silhouette")]:
        plt.figure(figsize=(7, 5))
        plt.plot(ks, values, marker="o")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} - {name}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{name}_{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close()
    labels = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit_predict(Z)
    metrics = pd.DataFrame({"k": ks, "SSE": inertias, "Silhouette": silhouettes})
    metrics.to_csv(OUTPUT_DIR / f"{name}_kmeans_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"Cluster": pd.Series(labels).value_counts().sort_index().index, "Number_of_records": pd.Series(labels).value_counts().sort_index().values}).to_csv(OUTPUT_DIR / f"{name}_cluster_sizes.csv", index=False, encoding="utf-8-sig")
    return best_k


def main():
    all_results, all_cms = [], []
    for name, info in DATASETS.items():
        if not info["file"].exists():
            raise FileNotFoundError(f"Dataset not found: {info['file']}")
        df = pd.read_csv(info["file"])
        X, y, preprocessor, _ = prepare_xy(df, info["target"], info["id_columns"])
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=stratify)
        specs = {
            "Random Forest": (RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"), {"model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20], "model__min_samples_split": [2, 5]}),
            "Decision Tree": (DecisionTreeClassifier(random_state=42, class_weight="balanced"), {"model__max_depth": [None, 5, 10, 20], "model__min_samples_split": [2, 5, 10]}),
            "SVM": (SVC(probability=True, class_weight="balanced", random_state=42), {"model__C": [0.1, 1, 10], "model__kernel": ["rbf"], "model__gamma": ["scale", "auto"]}),
            "Neural Network": (MLPClassifier(max_iter=500, random_state=42, early_stopping=True), {"model__hidden_layer_sizes": [(50,), (100,), (100, 50)], "model__alpha": [0.0001, 0.001], "model__learning_rate_init": [0.001, 0.01]}),
        }
        best_rf = None
        for model_name, (base, params) in specs.items():
            pipe = Pipeline([("preprocessor", preprocessor), ("model", base)])
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            search = GridSearchCV(pipe, params, scoring="f1_weighted", cv=cv, n_jobs=-1, refit=True)
            search.fit(X_train, y_train)
            pred = search.best_estimator_.predict(X_test)
            result = {"Dataset": name, "Model": model_name, "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, average="weighted", zero_division=0), "Recall": recall_score(y_test, pred, average="weighted", zero_division=0), "F1": f1_score(y_test, pred, average="weighted", zero_division=0), "CV_Best_F1": search.best_score_, "Best_Params": str(search.best_params_)}
            all_results.append(result)
            cm = confusion_matrix(y_test, pred)
            all_cms.append({"Dataset": name, "Model": model_name, "Confusion_Matrix": str(cm.tolist())})
            save_confusion_matrix(cm, f"Confusion Matrix - {name} - {model_name}", f"{name}_{model_name.replace(' ', '_')}_confusion_matrix.png")
            if model_name == "Random Forest":
                best_rf = search.best_estimator_
        if best_rf is not None:
            perm = permutation_importance(best_rf, X_test, y_test, scoring="f1_weighted", n_repeats=10, random_state=42, n_jobs=-1)
            fi = pd.DataFrame({"Feature": best_rf.named_steps["preprocessor"].get_feature_names_out(), "Importance": perm.importances_mean}).sort_values("Importance", ascending=False)
            fi.to_csv(OUTPUT_DIR / f"{name}_feature_importance.csv", index=False, encoding="utf-8-sig")
        run_kmeans(df, name, [info["target"]] + info["id_columns"])
    results = pd.DataFrame(all_results)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_cms).to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False, encoding="utf-8-sig")
    print(results[["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

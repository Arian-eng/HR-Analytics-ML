# -*- coding: utf-8 -*-
"""Reproducible Chapter 4 HR machine-learning analysis.

Datasets:
- IBM HR Employee Attrition -> Attrition
- HR Analytics: Job Change of Data Scientists -> target
- Employee Performance -> KPIs_met_more_than_80

The loader accepts the original Kaggle filenames as well as the renamed
filenames supplied for this thesis. Raw CSV files remain local and are not
committed to the repository.

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
        "files": [
            "WA_Fn-UseC_-HR-Employee-Attrition.csv",
            "WA_Fn-UseC_-HR-Employee-Attrition (3)(2).csv",
        ],
        "target": "Attrition",
        "id_columns": ["EmployeeNumber", "EmployeeCount", "StandardHours"],
    },
    "Job_Change": {
        "files": ["aug_train.csv", "aug_train(4).csv"],
        "target": "target",
        "id_columns": ["enrollee_id"],
    },
    "Employee_Performance": {
        "files": ["Uncleaned_employees_final_dataset.csv", "Uncleaned_employees_final_dataset (1)(2).csv"],
        "target": "KPIs_met_more_than_80",
        "id_columns": ["employee_id"],
    },
}


def resolve_dataset_file(file_candidates):
    for name in file_candidates:
        path = DATA_DIR / name
        if path.exists():
            return path
    expected = ", ".join(file_candidates)
    raise FileNotFoundError(f"Dataset not found. Expected one of: {expected}")


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
        plt.title(f"{name} - {ylabel}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{name}_{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close()
    return {"best_k": best_k, "metrics": pd.DataFrame({"k": ks, "inertia": inertias, "silhouette": silhouettes})}


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, pred, average="weighted", zero_division=0),
    }
    return result, confusion_matrix(y_test, pred), model


def feature_importance_plot(model, X_test, y_test, dataset_name):
    perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1, scoring="f1_weighted")
    fi = pd.DataFrame({"feature": X_test.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    return fi


# ============================================================
# Main analysis
# ============================================================
all_model_results = []
all_confusion_matrices = {}
all_feature_importance = {}
all_kmeans = {}

for dataset_name, info in DATASETS.items():
    file_path = resolve_dataset_file(info["files"])
    print(f"\n{'=' * 80}\n{dataset_name}: {file_path.name}\n{'=' * 80}")
    df = pd.read_csv(file_path)
    print(f"Shape: {df.shape}")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    X, y, preprocessor, target_mapping = prepare_xy(df, info["target"], info["id_columns"])
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    class_counts = y.value_counts()
    stratify = y if y.nunique() > 1 and class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=stratify)

    model_specs = {
        "Random Forest": (RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"), {
            "model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20], "model__min_samples_split": [2, 5]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=42, class_weight="balanced"), {
            "model__max_depth": [None, 5, 10, 20], "model__min_samples_split": [2, 5, 10]}),
        "SVM": (SVC(probability=True, class_weight="balanced", random_state=42), {
            "model__C": [0.1, 1, 10], "model__kernel": ["rbf"], "model__gamma": ["scale", "auto"]}),
        "Neural Network": (MLPClassifier(max_iter=500, random_state=42, early_stopping=True), {
            "model__hidden_layer_sizes": [(50,), (100,), (100, 50)], "model__alpha": [0.0001, 0.001], "model__learning_rate_init": [0.001, 0.01]}),
    }

    best_rf = None
    for model_name, (base_model, param_grid) in model_specs.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", base_model)])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        search = GridSearchCV(pipe, param_grid=param_grid, scoring="f1_weighted", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        result, cm, fitted_model = evaluate_model(model_name, search.best_estimator_, X_train, X_test, y_train, y_test)
        result.update({"Dataset": dataset_name, "Best_Params": str(search.best_params_), "CV_Best_F1": search.best_score_})
        all_model_results.append(result)
        all_confusion_matrices[(dataset_name, model_name)] = cm
        save_confusion_matrix(cm, f"Confusion Matrix - {dataset_name} - {model_name}", f"{dataset_name}_{model_name.replace(' ', '_')}_confusion_matrix.png")
        if model_name == "Random Forest":
            best_rf = fitted_model
        print(f"{model_name}: Accuracy={result['Accuracy']:.4f}, Precision={result['Precision']:.4f}, Recall={result['Recall']:.4f}, F1={result['F1']:.4f}")

    if best_rf is not None:
        fi = feature_importance_plot(best_rf, X_test, y_test, dataset_name)
        fi.to_csv(OUTPUT_DIR / f"{dataset_name}_feature_importance.csv", index=False, encoding="utf-8-sig")
        all_feature_importance[dataset_name] = fi

    all_kmeans[dataset_name] = run_kmeans(df, dataset_name, [info["target"]] + info["id_columns"])

results_df = pd.DataFrame(all_model_results)[["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1", "CV_Best_F1", "Best_Params"]]
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

cm_rows = [{"Dataset": d, "Model": m, "Confusion_Matrix": str(cm.tolist())} for (d, m), cm in all_confusion_matrices.items()]
pd.DataFrame(cm_rows).to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False, encoding="utf-8-sig")

excel_file = OUTPUT_DIR / "chapter_4_results.xlsx"
with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
    results_df.to_excel(writer, index=False, sheet_name="Model Results")
    pd.DataFrame(cm_rows).to_excel(writer, index=False, sheet_name="Confusion Matrices")
    for dataset_name, fi in all_feature_importance.items():
        fi.head(20).to_excel(writer, index=False, sheet_name=f"FI_{dataset_name}"[:31])
    for dataset_name, result in all_kmeans.items():
        if result is not None:
            result["metrics"].to_excel(writer, index=False, sheet_name=f"KMeans_{dataset_name}"[:31])

print("\nFINAL RESULTS")
print(results_df[["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1"]].round(4).to_string(index=False))
print("\nOptimal K based on highest Silhouette Score:")
for dataset_name, result in all_kmeans.items():
    if result is not None:
        print(f"{dataset_name}: k = {result['best_k']}")
print(f"\nOutput directory: {OUTPUT_DIR}")

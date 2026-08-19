from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DATASETS = {
    "ibm": {
        "files": ["WA_Fn-UseC_-HR-Employee-Attrition (3)(2).csv", "WA_Fn-UseC_-HR-Employee-Attrition.csv"],
        "target": "Attrition", "ids": ["EmployeeNumber", "EmployeeCount", "StandardHours"]
    },
    "job_change": {
        "files": ["aug_train(5).csv", "aug_train.csv", "aug_train(4).csv"],
        "target": "target", "ids": ["enrollee_id"]
    },
    "promotion": {
        "files": ["train_LZdllcl.csv", "Uncleaned_employees_final_dataset (1).csv", "Uncleaned_employees_final_dataset.csv"],
        "target": "is_promoted", "ids": ["employee_id"]
    },
}


def resolve_file(candidates):
    for name in candidates:
        path = DATA_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError("Dataset not found: " + ", ".join(candidates))


def load_dataset(name):
    info = DATASETS[name]
    path = resolve_file(info["files"])
    return pd.read_csv(path), info, path


def split_xy(df, target, ids):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset")
    y = df[target].copy()
    X = df.drop(columns=[target] + ids, errors="ignore").copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    return X.drop(columns=constant), y


def make_preprocessor(X, scale_numeric):
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
    ], remainder="drop")

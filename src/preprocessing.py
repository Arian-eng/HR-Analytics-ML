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
    "ibm": {"files": ["WA_Fn-UseC_-HR-Employee-Attrition (3)(2).csv"], "target": "Attrition", "ids": ["EmployeeNumber", "EmployeeCount", "StandardHours"]},
    "job_change": {"files": ["aug_train(2).csv", "aug_train(5).csv"], "target": "target", "ids": ["enrollee_id"]},
    "promotion": {"files": ["train_LZdllcl(2).csv", "train_LZdllcl.csv"], "target": "is_promoted", "ids": ["employee_id"]},
}

def resolve_file(candidates):
    for name in candidates:
        p = DATA_DIR / name
        if p.exists(): return p
    raise FileNotFoundError("Dataset not found: " + ", ".join(candidates))

def load_dataset(name):
    info = DATASETS[name]; path = resolve_file(info["files"])
    return pd.read_csv(path), info, path

def split_xy(df, target, ids):
    y = df[target].copy()
    X = df.drop(columns=[target] + ids, errors="ignore").copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    return X.drop(columns=constant), y

def make_preprocessor(X, scale_numeric=False):
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric: num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), numeric)] if numeric else []
    if categorical:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers, remainder="drop")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import DATA_DIR, DATASETS


def load_public(name):
    info = DATASETS[name]
    path = DATA_DIR / info["file"]
    df = pd.read_csv(path)
    y_raw = df[info["target"]]
    y = (y_raw == info["positive"]).astype(int)
    drop = [info["target"], *info["ids"], *info["drop"]]
    X = df.drop(columns=drop, errors="ignore").replace([np.inf, -np.inf], np.nan)
    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant)
    return df, X, y, path, constant


def make_preprocessor(X, scale_numeric=False):
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop"), numeric, categorical

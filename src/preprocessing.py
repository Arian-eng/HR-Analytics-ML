import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATASETS, DATA_DIR, GHRM_ITEMS


def dataset_path(name):
    info = DATASETS[name]
    path = DATA_DIR / info["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def load_dataset(name):
    info = DATASETS[name]
    path = dataset_path(name)
    frame = pd.read_csv(path)
    if name == "ghrm":
        if "GDT3" in frame.columns and "GTD3" not in frame.columns:
            frame = frame.rename(columns={"GDT3": "GTD3"})
        required = {column for columns in GHRM_ITEMS.values() for column in columns}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"GHRM columns missing: {missing}")
        for construct, columns in GHRM_ITEMS.items():
            values = frame[columns].apply(pd.to_numeric, errors="coerce")
            frame[construct] = values.mean(axis=1, skipna=False)
    return frame, info, path


def split_xy(df, target, ids, drop=None):
    y = df[target].copy()
    X = df.drop(columns=[target] + list(ids) + list(drop or []), errors="ignore").copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    return X.drop(columns=constant), y


def make_preprocessor(X, scale_numeric=False):
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), numeric)] if numeric else []
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")

"""HR Analytics: Job Change of Data Scientists (n=19,158) - Chapter 3
compliant: 80/20 split, limited grid search (3-fold Stratified CV, F1
scoring) for DT/RF/LinearSVC/MLP, 95% bootstrap CI per metric, McNemar
pairwise tests, and K-Means on standardized numeric features only
(SSE/Silhouette/Davies-Bouldin, k=2..7).
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
from ch3_utils import run_classification_suite, run_kmeans_numeric_only

df = pd.read_csv(PROJECT_ROOT / "data" / "aug_train.csv")
df = df.dropna(subset=["target"]).copy()
df["target"] = df["target"].astype(int)

numeric_cols = ["city_development_index", "training_hours"]
categorical_cols = ["city", "gender", "relevent_experience", "enrolled_university",
                     "education_level", "major_discipline", "experience",
                     "company_size", "company_type", "last_new_job"]

res = run_classification_suite(df, target_col="target", numeric_cols=numeric_cols,
                                categorical_cols=categorical_cols, dataset_name="job_change")
print("=== Job Change: classification ===")
for m, v in res["models"].items():
    print(m, v["best_params"], {k: v[k] for k in ["accuracy", "precision", "recall", "f1"]})

km, _ = run_kmeans_numeric_only(df, numeric_cols, "job_change")
print("=== Job Change: K-Means ===")
print("best_k:", km["best_k"])
for r in km["by_k"]:
    print(r)

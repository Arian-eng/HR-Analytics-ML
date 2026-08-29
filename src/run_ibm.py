"""IBM HR Attrition (n=1,470) - Chapter 3 compliant: 80/20 split, limited
grid search (3-fold Stratified CV, F1 scoring) for DT/RF/LinearSVC/MLP,
95% bootstrap CI per metric, McNemar pairwise tests, and K-Means on
standardized numeric features only (SSE/Silhouette/Davies-Bouldin, k=2..7).
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
from ch3_utils import run_classification_suite, run_kmeans_numeric_only

df = pd.read_csv(PROJECT_ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition__3_.csv")
df["Attrition_bin"] = (df["Attrition"] == "Yes").astype(int)
drop_cols = {"Attrition", "Attrition_bin", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"}
numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in drop_cols]
categorical_cols = [c for c in df.select_dtypes(include="object").columns if c not in drop_cols]

res = run_classification_suite(df, target_col="Attrition_bin", numeric_cols=numeric_cols,
                                categorical_cols=categorical_cols, dataset_name="ibm_attrition")
print("=== IBM Attrition: classification ===")
for m, v in res["models"].items():
    print(m, v["best_params"], {k: v[k] for k in ["accuracy", "precision", "recall", "f1"]})

km, _ = run_kmeans_numeric_only(df, numeric_cols, "ibm_attrition")
print("=== IBM Attrition: K-Means ===")
print("best_k:", km["best_k"])
for r in km["by_k"]:
    print(r)

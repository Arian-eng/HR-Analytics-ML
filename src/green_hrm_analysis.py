"""Green HRM analysis for the dedicated GHRM datasets.

Data files are kept outside Git. The script expects:
- Data GI Step 1.csv: GHRM -> green work environment / green transformational
  leadership -> green innovation (MGI)
- Data sustainable performance.csv: GHRM -> perceived environmental
  improvement / environmental sustainability -> sustainable performance (MSP)

The continuous composite targets are analyzed with regression rather than
forcing them into binary classification.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_regression(file_name, target, features, name):
    df = pd.read_csv(DATA_DIR / file_name)
    X, y = df[features], df[target]
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.20, random_state=42)
    rows = []
    for model_name, model in [
        ("Linear Regression", LinearRegression()),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
    ]:
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        rows.append({
            "Dataset": name, "Target": target, "Model": model_name,
            "R2": r2_score(y.iloc[test_idx], pred),
            "MAE": mean_absolute_error(y.iloc[test_idx], pred),
            "RMSE": mean_squared_error(y.iloc[test_idx], pred) ** 0.5,
        })
        if hasattr(model, "feature_importances_"):
            pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False).to_csv(
                OUTPUT_DIR / f"{name}_feature_importance.csv", index=False, encoding="utf-8-sig")
    return rows


def run_kmeans(file_name, features, name):
    df = pd.read_csv(DATA_DIR / file_name)
    Z = StandardScaler().fit_transform(df[features])
    rows = []
    for k in range(2, 6):
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(Z)
        rows.append({"Dataset": name, "k": k, "Silhouette": silhouette_score(Z, labels)})
    return rows

rows = []
rows += run_regression("Data GI Step 1.csv", "MGI", ["MGHRM", "MGWE", "MGTFL"], "Green_Innovation")
rows += run_regression("Data sustainable performance.csv", "MSP", ["MGHRM", "MPEI", "MPES"], "Sustainable_Performance")
pd.DataFrame(rows).to_csv(OUTPUT_DIR / "green_hrm_regression_results.csv", index=False, encoding="utf-8-sig")

clusters = run_kmeans("Data GI Step 1.csv", ["MGHRM", "MGWE", "MGTFL", "MGI"], "Green_Innovation")
clusters += run_kmeans("Data sustainable performance.csv", ["MGHRM", "MPEI", "MPES", "MSP"], "Sustainable_Performance")
pd.DataFrame(clusters).to_csv(OUTPUT_DIR / "green_hrm_kmeans_results.csv", index=False, encoding="utf-8-sig")

print(pd.DataFrame(rows).round(4).to_string(index=False))
print(pd.DataFrame(clusters).round(4).to_string(index=False))

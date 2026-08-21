from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GHRM_FILES = ["HRM DATASETS(2).csv", "HRM DATASETS.csv"]
OUT = ROOT / "outputs" / "ghrm"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

ITEMS = {
    "GRS": ["GRS1", "GRS2", "GRS3", "GRS4"],
    "GTD": ["GTD1", "GTD2", "GDT3", "GTD4", "GTD5"],
    "GPA": ["GPA1", "GPA2", "GPA3", "GPA4", "GPA5", "GPA6"],
    "GCM": ["GCM1", "GCM2", "GCM3", "GCM4"],
    "GEE": ["GEE1", "GEE4", "GEE5", "GEE6"],
    "FEP": ["FEP1", "FEP5", "FEP7", "FEP9"],
}


def load():
    path = next(
        (DATA_DIR / name for name in GHRM_FILES if (DATA_DIR / name).exists()),
        None,
    )
    if path is None:
        raise FileNotFoundError(
            "GHRM dataset not found: " + ", ".join(GHRM_FILES)
        )
    df = pd.read_csv(path)
    required = {column for columns in ITEMS.values() for column in columns}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"GHRM dataset is missing required columns: {missing}")
    for name, cols in ITEMS.items():
        # A construct score is valid only when all of its stated items exist.
        # This avoids silently changing the construct definition row by row.
        df[name] = (
            df[cols]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=1, skipna=False)
        )
    return df


def run_regression(df, gee_plus=False):
    features = ["GRS", "GTD", "GPA", "GCM"] + (["GEE"] if gee_plus else [])
    valid_target = df["FEP"].notna()
    X, y = df.loc[valid_target, features], df.loc[valid_target, "FEP"]
    if len(y) < 4:
        raise ValueError("GHRM regression requires at least four non-missing FEP rows")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
    prep = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)])
    models = {
        "Random Forest Regressor": (RandomForestRegressor(random_state=42, n_jobs=-1), {
            "model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20],
            "model__max_features": [1.0, "sqrt"], "model__min_samples_split": [2, 5]}),
        "Decision Tree Regressor": (DecisionTreeRegressor(random_state=42), {
            "model__max_depth": [5, 10, 15, None], "model__min_samples_split": [2, 5], "model__min_samples_leaf": [1, 2]}),
        "LinearSVR": (LinearSVR(random_state=42, max_iter=20000), {
            "model__C": [0.1, 1, 10], "model__epsilon": [0, 0.1, 0.2]}),
        "MLPRegressor": (MLPRegressor(random_state=42, max_iter=1000, early_stopping=True), {
            "model__hidden_layer_sizes": [(50,), (100,), (50, 50)], "model__alpha": [0.0001, 0.001], "model__learning_rate_init": [0.001, 0.01]})}
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    rows = []
    for name, (estimator, grid) in models.items():
        pipe = Pipeline([("preprocess", prep), ("model", estimator)])
        if name == "MLPRegressor":
            search = RandomizedSearchCV(pipe, grid, n_iter=8, scoring="neg_root_mean_squared_error", cv=cv, random_state=42, n_jobs=-1, refit=True)
        else:
            search = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        rows.append({"Variant": "GEE+" if gee_plus else "Base", "Model": name,
                     "R2": r2_score(y_test, pred), "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                     "CV_RMSE": -search.best_score_, "Best_Params": str(search.best_params_)})
        pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": pred}).to_csv(
            OUT / f"{('gee_plus' if gee_plus else 'base')}_{name.replace(' ', '_')}_test_predictions.csv", index=False)
    return rows


def run_kmeans(df):
    features = ["GRS", "GTD", "GPA", "GCM", "GEE"]
    Z = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]).fit_transform(df[features])
    rows = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
        rows.append({"k": k, "Inertia_SSE": km.inertia_, "Silhouette": silhouette_score(Z, km.labels_), "Davies_Bouldin": davies_bouldin_score(Z, km.labels_)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "kmeans_metrics.csv", index=False, encoding="utf-8-sig")
    plots = [
        ("Inertia_SSE", "GHRM K-Means Elbow", "Inertia / SSE"),
        ("Silhouette", "GHRM K-Means Silhouette", "Silhouette Score"),
        (
            "Davies_Bouldin",
            "GHRM K-Means Davies-Bouldin",
            "Davies-Bouldin Index",
        ),
    ]
    for column, title, ylabel in plots:
        figure, axis = plt.subplots()
        axis.plot(metrics["k"], metrics[column], marker="o")
        axis.set_title(title)
        axis.set_xlabel("k")
        axis.set_ylabel(ylabel)
        figure.tight_layout()
        figure.savefig(FIG / f"ghrm_{column}.png", dpi=180)
        plt.close(figure)
    best_k = int(metrics.loc[metrics.Silhouette.idxmax(), "k"])
    labels = KMeans(n_clusters=best_k, n_init=20, random_state=42).fit_predict(Z)
    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Size")
    )
    cluster_sizes.to_csv(
        OUT / "cluster_sizes.csv", index=False, encoding="utf-8-sig"
    )
    profiles = pd.DataFrame({"Cluster": labels, "FEP": df["FEP"]}).groupby("Cluster").agg(Size=("FEP", "size"), Mean_FEP=("FEP", "mean"), SD_FEP=("FEP", "std"))
    profiles.to_csv(OUT / "cluster_profiles.csv", encoding="utf-8-sig")


def main():
    df = load()
    rows = run_regression(df, False) + run_regression(df, True)
    pd.DataFrame(rows).to_csv(OUT / "regression_metrics.csv", index=False, encoding="utf-8-sig")
    run_kmeans(df)

if __name__ == "__main__":
    main()

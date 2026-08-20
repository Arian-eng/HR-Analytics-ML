from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SILHOUETTE_SAMPLE_SIZE = 5000


def _matrix(df, drop_cols):
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    if X.shape[1] == 0:
        raise ValueError("K-Means requires at least one non-target feature")

    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers).fit_transform(X)


def run_kmeans(
    df, target, ids, out_dir, figure_dir, name, extra_drop=None
):
    drop_columns = list(ids) + [target] + (extra_drop or [])
    matrix = _matrix(df, drop_columns)
    if matrix.shape[0] <= 7:
        raise ValueError("K-Means k=2..7 requires at least eight rows")

    output_path = Path(out_dir)
    figure_path = Path(figure_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)
    dense_matrix = (
        matrix.toarray() if hasattr(matrix, "toarray") else matrix
    )

    rows = []
    for k in range(2, 8):
        model = KMeans(
            n_clusters=k, n_init=20, max_iter=300, random_state=42
        ).fit(matrix)
        silhouette_kwargs = {}
        if matrix.shape[0] > SILHOUETTE_SAMPLE_SIZE:
            silhouette_kwargs = {
                "sample_size": SILHOUETTE_SAMPLE_SIZE,
                "random_state": 42,
            }
        rows.append(
            {
                "Dataset": name,
                "k": k,
                "Inertia_SSE": model.inertia_,
                "Silhouette": silhouette_score(
                    matrix, model.labels_, **silhouette_kwargs
                ),
                "Davies_Bouldin": davies_bouldin_score(
                    dense_matrix, model.labels_
                ),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        output_path / "kmeans_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plots = [
        ("Inertia_SSE", "K-Means Elbow", "Inertia / SSE"),
        ("Silhouette", "K-Means Silhouette", "Silhouette Score"),
        ("Davies_Bouldin", "K-Means Davies-Bouldin", "Davies-Bouldin Index"),
    ]
    display_name = name.replace("_", " ").title()
    for column, title, ylabel in plots:
        figure, axis = plt.subplots()
        axis.plot(metrics["k"], metrics[column], marker="o")
        sample_note = ""
        if column == "Silhouette" and matrix.shape[0] > SILHOUETTE_SAMPLE_SIZE:
            sample_note = f" (sample n={SILHOUETTE_SAMPLE_SIZE:,})"
        axis.set_title(f"{display_name} {title}{sample_note}")
        axis.set_xlabel("k")
        axis.set_ylabel(ylabel)
        figure.tight_layout()
        figure.savefig(
            figure_path / f"{name}_{column}.png", dpi=180
        )
        plt.close(figure)

    best_k = int(metrics.loc[metrics["Silhouette"].idxmax(), "k"])
    final_model = KMeans(
        n_clusters=best_k, n_init=20, max_iter=300, random_state=42
    ).fit(matrix)
    cluster_sizes = (
        pd.Series(final_model.labels_)
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Size")
    )
    cluster_sizes.to_csv(
        output_path / "cluster_sizes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return metrics, best_k, final_model.labels_

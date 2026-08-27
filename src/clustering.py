"""Independent, exploratory K-Means analysis for each dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, SILHOUETTE_SAMPLE_SIZE


def numeric_matrix(frame, drop_columns):
    features = frame.drop(columns=drop_columns, errors="ignore")
    features = features.select_dtypes(include="number").copy()
    constant = [
        column
        for column in features.columns
        if features[column].nunique(dropna=False) <= 1
    ]
    features = features.drop(columns=constant)
    if features.shape[1] == 0:
        raise ValueError("K-Means needs at least one numeric non-target feature")
    transformer = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    matrix = transformer.fit_transform(features)
    return features, matrix


def _profile_numeric(frame, labels, target, dataset):
    data = frame.copy()
    data["Cluster"] = labels
    numeric = data.select_dtypes(include="number").columns.tolist()
    rows = []
    for cluster, group in data.groupby("Cluster", sort=True):
        for column in numeric:
            if column == "Cluster":
                continue
            rows.append(
                {
                    "Dataset": dataset,
                    "Cluster": int(cluster),
                    "Feature": column,
                    "Mean": float(group[column].mean()),
                    "Median": float(group[column].median()),
                    "Std": float(group[column].std()),
                    "Is_Target": column == target,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "Dataset",
            "Cluster",
            "Feature",
            "Mean",
            "Median",
            "Std",
            "Is_Target",
        ],
    )


def _profile_categorical(frame, labels, target, dataset):
    data = frame.copy()
    data["Cluster"] = labels
    categorical = data.select_dtypes(include=["object", "category", "bool"]).columns
    rows = []
    for cluster, group in data.groupby("Cluster", sort=True):
        for column in categorical:
            counts = group[column].fillna("<missing>").astype(str).value_counts()
            if counts.empty:
                continue
            rows.append(
                {
                    "Dataset": dataset,
                    "Cluster": int(cluster),
                    "Feature": column,
                    "Most_Common_Value": counts.index[0],
                    "Count": int(counts.iloc[0]),
                    "Share": float(counts.iloc[0] / len(group)),
                    "Is_Target": column == target,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "Dataset",
            "Cluster",
            "Feature",
            "Most_Common_Value",
            "Count",
            "Share",
            "Is_Target",
        ],
    )


def _target_distribution(frame, labels, target, dataset):
    data = pd.DataFrame({"Cluster": labels, "Target": frame[target].to_numpy()})
    rows = []
    if pd.api.types.is_numeric_dtype(data["Target"]):
        for cluster, group in data.groupby("Cluster", sort=True):
            rows.append(
                {
                    "Dataset": dataset,
                    "Cluster": int(cluster),
                    "Target": target,
                    "Target_Value": "mean",
                    "Count": len(group),
                    "Share_or_Mean": float(group["Target"].mean()),
                }
            )
    else:
        for (cluster, value), count in (
            data.groupby(["Cluster", "Target"], dropna=False).size().items()
        ):
            size = int((data["Cluster"] == cluster).sum())
            rows.append(
                {
                    "Dataset": dataset,
                    "Cluster": int(cluster),
                    "Target": target,
                    "Target_Value": str(value),
                    "Count": int(count),
                    "Share_or_Mean": float(count / size),
                }
            )
    return pd.DataFrame(rows)


def _distinguishing_features(features, labels, dataset):
    imputed = features.copy()
    for column in imputed.columns:
        imputed[column] = imputed[column].fillna(imputed[column].median())
    overall_mean = imputed.mean()
    overall_std = imputed.std(ddof=0).replace(0, np.nan)
    rows = []
    labelled = imputed.assign(Cluster=labels)
    for cluster, group in labelled.groupby("Cluster", sort=True):
        z_difference = ((group.drop(columns="Cluster").mean() - overall_mean) / overall_std)
        for feature, value in z_difference.abs().nlargest(8).items():
            signed = float(z_difference[feature])
            rows.append(
                {
                    "Dataset": dataset,
                    "Cluster": int(cluster),
                    "Feature": feature,
                    "Cluster_Mean": float(group[feature].mean()),
                    "Overall_Mean": float(overall_mean[feature]),
                    "Standardized_Difference": signed,
                    "Direction": "higher" if signed > 0 else "lower",
                }
            )
    return pd.DataFrame(rows)


def run_kmeans(frame, target, ids, out_dir, dataset, extra_drop=None):
    """Evaluate k=2..7, select maximum Silhouette, and save profiles."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    drop_columns = list(ids) + [target] + list(extra_drop or [])
    numeric_features, matrix = numeric_matrix(frame, drop_columns)

    rows = []
    for k in range(2, 8):
        model = KMeans(
            n_clusters=k,
            n_init=20,
            max_iter=300,
            random_state=RANDOM_STATE,
        ).fit(matrix)
        silhouette_kwargs = {}
        if len(frame) > SILHOUETTE_SAMPLE_SIZE:
            silhouette_kwargs = {
                "sample_size": SILHOUETTE_SAMPLE_SIZE,
                "random_state": RANDOM_STATE,
            }
        rows.append(
            {
                "Dataset": dataset,
                "k": k,
                "Numeric_Features": numeric_features.shape[1],
                "Inertia_SSE": float(model.inertia_),
                "Silhouette": float(
                    silhouette_score(matrix, model.labels_, **silhouette_kwargs)
                ),
                "Davies_Bouldin": float(
                    davies_bouldin_score(np.asarray(matrix), model.labels_)
                ),
                "Iterations": int(model.n_iter_),
                "Silhouette_N": min(len(frame), SILHOUETTE_SAMPLE_SIZE),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "kmeans_metrics.csv", index=False, encoding="utf-8-sig")
    selected = metrics.loc[metrics["Silhouette"].idxmax()]
    selected_k = int(selected["k"])
    final_model = KMeans(
        n_clusters=selected_k,
        n_init=20,
        max_iter=300,
        random_state=RANDOM_STATE,
    ).fit(matrix)
    labels = final_model.labels_

    sizes = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Size")
    )
    sizes.insert(0, "Dataset", dataset)
    sizes["Share"] = sizes["Size"] / len(frame)
    sizes.to_csv(output / "cluster_sizes.csv", index=False, encoding="utf-8-sig")
    profile_frame = frame.drop(
        columns=list(ids) + list(extra_drop or []), errors="ignore"
    )
    _profile_numeric(profile_frame, labels, target, dataset).to_csv(
        output / "cluster_profile_numeric.csv", index=False, encoding="utf-8-sig"
    )
    _profile_categorical(profile_frame, labels, target, dataset).to_csv(
        output / "cluster_profile_categorical.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _target_distribution(frame, labels, target, dataset).to_csv(
        output / "cluster_target_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _distinguishing_features(numeric_features, labels, dataset).to_csv(
        output / "cluster_distinguishing_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({"Row_Index": frame.index, "Cluster": labels}).to_csv(
        output / "cluster_assignments.csv", index=False, encoding="utf-8-sig"
    )
    return metrics, selected_k, labels

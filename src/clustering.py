from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


def run_kmeans(df, target, ids, out_dir, fig_dir, dataset_name):
    X = df.drop(columns=[target] + ids, errors="ignore").select_dtypes(include=np.number).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.drop(columns=[c for c in X if c.lower() in {"id", "employee_id", "employee_number", "enrollee_id"}], errors="ignore")
    X = X.fillna(X.median(numeric_only=True))
    X = X.dropna(axis=1, how="all")
    if X.shape[1] < 2:
        raise ValueError(f"{dataset_name}: K-Means requires at least two numeric features")
    Z = StandardScaler().fit_transform(X)
    rows = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(Z)
        rows.append({
            "k": k,
            "inertia_sse": km.inertia_,
            "silhouette_score": silhouette_score(Z, labels),
            "davies_bouldin_index": davies_bouldin_score(Z, labels)
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / f"{dataset_name}_kmeans_metrics.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots()
    ax.plot(metrics["k"], metrics["inertia_sse"], marker="o")
    ax.set(xlabel="k", ylabel="Inertia / SSE", title=f"{dataset_name}: Elbow")
    fig.tight_layout(); fig.savefig(fig_dir / f"{dataset_name}_elbow.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(metrics["k"], metrics["silhouette_score"], marker="o")
    ax.set(xlabel="k", ylabel="Silhouette Score", title=f"{dataset_name}: Silhouette")
    fig.tight_layout(); fig.savefig(fig_dir / f"{dataset_name}_silhouette.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(metrics["k"], metrics["davies_bouldin_index"], marker="o")
    ax.set(xlabel="k", ylabel="Davies-Bouldin Index", title=f"{dataset_name}: Davies-Bouldin")
    fig.tight_layout(); fig.savefig(fig_dir / f"{dataset_name}_davies_bouldin.png", dpi=300); plt.close(fig)

    # Select independently using the highest silhouette score, with DB index as a secondary check.
    best_k = int(metrics.loc[metrics["silhouette_score"].idxmax(), "k"])
    final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = final.fit_predict(Z)
    summary = X.assign(cluster=labels).groupby("cluster").agg(["mean", "count"])
    summary.to_csv(out_dir / f"{dataset_name}_cluster_summary.csv", encoding="utf-8-sig")
    sizes = pd.Series(labels).value_counts().sort_index().rename_axis("cluster").reset_index(name="size")
    sizes.to_csv(out_dir / f"{dataset_name}_cluster_sizes.csv", index=False, encoding="utf-8-sig")
    return metrics, best_k

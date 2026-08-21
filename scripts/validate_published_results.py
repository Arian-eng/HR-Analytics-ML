"""Validate tracked Chapter 4 artifacts without requiring the raw datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
REPORT = ROOT / "results" / "chapter4_results.md"
MANIFEST = ROOT / "validation" / "data_manifest.json"
PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_table(name: str) -> pd.DataFrame:
    path = TABLES / f"{name}.csv"
    require(path.is_file(), f"Missing published table: {path.relative_to(ROOT)}")
    return pd.read_csv(path, encoding="utf-8-sig")


def between(frame: pd.DataFrame, metric: str) -> pd.Series:
    return (frame[f"{metric}_CI_Lower"] <= frame[metric]) & (
        frame[metric] <= frame[f"{metric}_CI_Upper"]
    )


def validate_tables() -> None:
    overview = read_table("dataset_overview")
    classification = read_table("classification_metrics")
    confusion = read_table("confusion_matrices")
    regression = read_table("regression_metrics")
    comparison = read_table("regression_variant_comparison")
    kmeans = read_table("kmeans_selection")
    cluster_sizes = read_table("cluster_sizes")

    require(len(overview) == 4, "dataset_overview must contain four datasets")
    require(len(classification) == 12, "classification_metrics must contain 12 rows")
    require(len(confusion) == 12, "confusion_matrices must contain 12 rows")
    require(len(regression) == 8, "regression_metrics must contain eight rows")
    require(len(comparison) == 4, "regression comparison must contain four rows")
    require(len(kmeans) == 4, "kmeans_selection must contain four rows")

    merged = classification.merge(
        confusion, on=["Dataset", "Model"], validate="one_to_one"
    )
    require(len(merged) == 12, "classification and confusion rows do not align")
    total = merged[["TN", "FP", "FN", "TP"]].sum(axis=1)
    accuracy = (merged["TN"] + merged["TP"]) / total
    precision = merged["TP"] / (merged["TP"] + merged["FP"])
    recall = merged["TP"] / (merged["TP"] + merged["FN"])
    f1 = 2 * precision * recall / (precision + recall)
    for name, computed in {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }.items():
        require(
            (computed - merged[name]).abs().max() < 1e-12,
            f"{name} does not match the published confusion matrices",
        )
        require(between(merged, name).all(), f"{name} lies outside its 95% CI")

    row_counts = overview.set_index("Dataset")["Rows"]
    for row in confusion.itertuples(index=False):
        expected_test_n = math.ceil(float(row_counts[row.Dataset]) * 0.20)
        require(row.Test_N == expected_test_n, f"Unexpected test size for {row.Dataset}")

    require(regression[["R2", "MAE", "RMSE", "CV_RMSE"]].notna().all().all(), "Regression metrics contain missing values")
    for metric in ("R2", "MAE", "RMSE"):
        require(between(regression, metric).all(), f"Regression {metric} lies outside its CI")
    for metric in ("Delta_R2", "Delta_MAE", "Delta_RMSE"):
        require(between(comparison, metric).all(), f"{metric} lies outside its paired CI")

    require(kmeans["Selected_k"].between(2, 7).all(), "Selected k must be between 2 and 7")
    require(kmeans["Silhouette"].between(-1, 1).all(), "Invalid Silhouette score")
    require((kmeans["Silhouette_N"] <= kmeans["Dataset"].map(row_counts)).all(), "Silhouette sample exceeds source rows")
    size_totals = cluster_sizes.groupby("Dataset")["Size"].sum()
    require(set(size_totals.index) == set(row_counts.index), "Cluster sizes do not cover all datasets")
    for dataset, count in size_totals.items():
        require(int(count) == int(row_counts[dataset]), f"Cluster sizes do not sum for {dataset}")
        selected_k = int(kmeans.loc[kmeans["Dataset"] == dataset, "Selected_k"].iloc[0])
        actual_k = int((cluster_sizes["Dataset"] == dataset).sum())
        require(actual_k == selected_k, f"Cluster count does not match selected k for {dataset}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_data(require_data: bool) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing = []
    for item in manifest["datasets"]:
        path = ROOT / "data" / item["filename"]
        if not path.exists():
            missing.append(item["filename"])
            continue
        require(sha256(path) == item["sha256"], f"SHA-256 mismatch for {item['filename']}")
    if require_data:
        require(not missing, "Raw data missing: " + ", ".join(missing))


def persian_number(value: float) -> str:
    return f"{value:.4f}".translate(PERSIAN_DIGITS)


def validate_report() -> None:
    report = REPORT.read_text(encoding="utf-8")
    require("٫" not in report, "Use '.' rather than the Arabic decimal separator")
    require(any(character in report for character in "۰۱۲۳۴۵۶۷۸۹"), "Persian report has no Persian digits")
    classification = read_table("classification_metrics")
    regression = read_table("regression_metrics")
    comparison = read_table("regression_variant_comparison")
    cluster_sizes = read_table("cluster_sizes")
    expected_values = [
        *(persian_number(value) for value in classification["F1"]),
        *(persian_number(value) for value in regression["R2"]),
        *(persian_number(value) for value in regression["RMSE"]),
        *(persian_number(value) for value in comparison["Delta_R2"]),
        *(f"{int(value):,}".translate(PERSIAN_DIGITS) for value in cluster_sizes["Size"]),
    ]
    require(
        all(value in report for value in expected_values),
        "Persian narrative is stale relative to the published tables",
    )
    for name in ("classification_f1.png", "ghrm_regression_rmse.png", "kmeans_selected_quality.png"):
        require((ROOT / "results" / "figures" / name).is_file(), f"Missing figure: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Fail when a raw dataset is absent; hashes are always checked when present.",
    )
    args = parser.parse_args()
    validate_tables()
    validate_report()
    validate_data(args.require_data)
    print("Published Chapter 4 artifacts are internally consistent.")


if __name__ == "__main__":
    main()

"""Recompute published metrics and check that all audit artifacts are consistent."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
MANIFEST = ROOT / "validation" / "data_manifest.json"
sys.path.insert(0, str(ROOT))

from src.evaluation import classification_metrics, regression_metrics


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def require(path):
    if not path.is_file():
        raise AssertionError(f"required file is missing: {path.relative_to(ROOT)}")
    return path


def close(actual, expected, label, tolerance=1e-10):
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_data(require_data):
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["datasets"]
    inventory = pd.read_csv(require(RESULTS / "data" / "dataset_inventory.csv"))
    by_id = inventory.set_index("Dataset")
    for item in expected:
        dataset = item["id"]
        if dataset not in by_id.index:
            raise AssertionError(f"dataset missing from inventory: {dataset}")
        row = by_id.loc[dataset]
        if int(row["Rows"]) != int(item["rows"]):
            raise AssertionError(f"row count mismatch in inventory: {dataset}")
        if str(row["SHA256"]) != item["sha256"]:
            raise AssertionError(f"SHA-256 mismatch in inventory: {dataset}")
        path = DATA / item["filename"]
        if require_data:
            require(path)
        if path.is_file():
            if sha256(path) != item["sha256"]:
                raise AssertionError(f"raw file SHA-256 mismatch: {path.name}")
            if len(pd.read_csv(path)) != item["rows"]:
                raise AssertionError(f"raw file row count mismatch: {path.name}")
    require(RESULTS / "data" / "column_dictionary.csv")
    require(RESULTS / "data" / "data_quality.json")
    return len(expected)


def validate_classification():
    table = pd.read_csv(require(RESULTS / "classification" / "classification_metrics.csv"))
    aggregate = pd.read_csv(require(RESULTS / "tables" / "classification_metrics.csv"))
    pd.testing.assert_frame_equal(table, aggregate)
    checked = 0
    for _, row in table.iterrows():
        dataset, model = row["Dataset"], row["Model"]
        output = RESULTS / "classification" / dataset / slug(model)
        saved_matrix = pd.read_csv(
            require(output / "confusion_matrix.csv"), index_col=0
        ).to_numpy()
        tn, fp, fn, tp = saved_matrix.ravel()
        aggregate_true = np.r_[np.zeros(tn + fp), np.ones(fn + tp)]
        aggregate_pred = np.r_[
            np.zeros(tn), np.ones(fp), np.zeros(fn), np.ones(tp)
        ]
        values = classification_metrics(aggregate_true, aggregate_pred)
        for metric, value in values.items():
            close(row[metric], value, f"{dataset}/{model}/{metric}")
        prediction_path = output / "test_predictions.csv"
        if prediction_path.is_file():
            predictions = pd.read_csv(prediction_path)
            if predictions["Row_Index"].duplicated().any():
                raise AssertionError(f"duplicate test row: {dataset}/{model}")
            split = pd.read_csv(require(RESULTS / "splits" / f"{dataset}.csv"))
            expected_test = split.loc[split["Split"].eq("test")]
            if set(predictions["Row_Index"]) != set(expected_test["Row_Index"]):
                raise AssertionError(f"test split mismatch: {dataset}/{model}")
            matrix = confusion_matrix(
                predictions["y_true"], predictions["y_pred"], labels=[0, 1]
            )
            if not np.array_equal(matrix, saved_matrix):
                raise AssertionError(f"confusion matrix mismatch: {dataset}/{model}")
        require(output / "tuning_results.csv")
        require(output / "permutation_importance.csv")
        diagnostics = json.loads(require(output / "model_diagnostics.json").read_text(encoding="utf-8"))
        if model == "Decision Tree":
            nodes = pd.read_csv(require(output / "tree_nodes.csv"))
            require(output / "tree_rules.txt")
            require(output / "tree_top_levels.png")
            if "Positive_Probability" not in nodes or len(nodes) != diagnostics["nodes"]:
                raise AssertionError(f"decision tree nodes incomplete: {dataset}")
        elif model == "Random Forest":
            forest = pd.read_csv(require(output / "forest_tree_summary.csv"))
            require(output / "representative_tree_nodes.csv")
            require(output / "representative_tree_top_levels.png")
            if len(forest) != diagnostics["number_of_trees"]:
                raise AssertionError(f"forest tree count mismatch: {dataset}")
        elif model == "Linear SVM":
            require(output / "coefficients.csv")
        elif model == "MLP":
            if not diagnostics.get("architecture") or not diagnostics.get("iterations"):
                raise AssertionError(f"MLP diagnostics incomplete: {dataset}")
        checked += 1
    mcnemar = pd.read_csv(require(RESULTS / "classification" / "mcnemar.csv"))
    if len(mcnemar) != 18 or (mcnemar[["b01", "b10"]].sum(axis=1) < 0).any():
        raise AssertionError("McNemar output does not contain all 18 valid comparisons")
    return checked


def validate_regression():
    table = pd.read_csv(require(RESULTS / "regression" / "regression_metrics.csv"))
    aggregate = pd.read_csv(require(RESULTS / "tables" / "regression_metrics.csv"))
    pd.testing.assert_frame_equal(table, aggregate)
    validation = pd.read_csv(
        require(RESULTS / "regression" / "regression_validation_aggregates.csv")
    ).set_index(["Variant", "Model"])
    predictions_by_model = {}
    checked = 0
    for _, row in table.iterrows():
        variant, model = row["Variant"], row["Model"]
        output = RESULTS / "regression" / slug(variant) / slug(model)
        evidence = validation.loc[(variant, model)]
        n = int(evidence["N"])
        sse = float(evidence["Sum_Squared_Error"])
        sae = float(evidence["Sum_Absolute_Error"])
        sum_y = float(evidence["Sum_Y"])
        sum_y_squared = float(evidence["Sum_Y_Squared"])
        total_sum_squares = sum_y_squared - (sum_y**2 / n)
        values = {
            "R2": 1.0 - sse / total_sum_squares,
            "MAE": sae / n,
            "RMSE": math.sqrt(sse / n),
        }
        for metric, value in values.items():
            close(row[metric], value, f"{variant}/{model}/{metric}")
        prediction_path = output / "test_predictions.csv"
        if prediction_path.is_file():
            predictions = pd.read_csv(prediction_path)
            direct = regression_metrics(predictions["y_true"], predictions["y_pred"])
            for metric, value in direct.items():
                close(row[metric], value, f"local {variant}/{model}/{metric}")
            split = pd.read_csv(require(RESULTS / "splits" / "ghrm.csv"))
            expected_test = split.loc[split["Split"].eq("test"), "Row_Index"]
            if set(predictions["Row_Index"]) != set(expected_test):
                raise AssertionError(f"GHRM test split mismatch: {variant}/{model}")
            predictions_by_model[(variant, model)] = predictions
        if len(pd.read_csv(require(output / "repeated_cv_metrics.csv"))) != 25:
            raise AssertionError(f"repeated CV must contain 25 folds: {variant}/{model}")
        require(output / "tuning_results.csv")
        require(output / "permutation_importance.csv")
        convergence = json.loads(
            require(output / "convergence_warnings.json").read_text(encoding="utf-8")
        )
        if min(
            convergence["tuning_convergence_warnings"],
            convergence["repeated_cv_convergence_warnings"],
        ) < 0:
            raise AssertionError(f"invalid convergence warning count: {variant}/{model}")
        diagnostics = json.loads(require(output / "model_diagnostics.json").read_text(encoding="utf-8"))
        if model == "Decision Tree Regressor":
            nodes = pd.read_csv(require(output / "tree_nodes.csv"))
            require(output / "tree_rules.txt")
            require(output / "tree_top_levels.png")
            if len(nodes) != diagnostics["nodes"]:
                raise AssertionError(f"regression tree nodes incomplete: {variant}")
        elif model == "Random Forest Regressor":
            forest = pd.read_csv(require(output / "forest_tree_summary.csv"))
            require(output / "representative_tree_nodes.csv")
            require(output / "representative_tree_top_levels.png")
            if len(forest) != diagnostics["number_of_trees"]:
                raise AssertionError(f"regression forest count mismatch: {variant}")
        elif model == "LinearSVR":
            require(output / "coefficients.csv")
        elif model == "MLPRegressor" and not diagnostics.get("architecture"):
            raise AssertionError(f"regression MLP diagnostics incomplete: {variant}")
        checked += 1

    comparison = pd.read_csv(require(RESULTS / "regression" / "base_vs_gee_plus.csv"))
    for _, row in comparison.iterrows():
        model = row["Model"]
        if ("Base", model) in predictions_by_model:
            base = predictions_by_model[("Base", model)].sort_values("Row_Index")
            plus = predictions_by_model[("GEE+", model)].sort_values("Row_Index")
            if not np.array_equal(base["Row_Index"], plus["Row_Index"]):
                raise AssertionError(f"Base/GEE+ row pairing mismatch: {model}")
            base_metrics = regression_metrics(base["y_true"], base["y_pred"])
            plus_metrics = regression_metrics(plus["y_true"], plus["y_pred"])
        else:
            base_row = table[(table["Variant"] == "Base") & (table["Model"] == model)].iloc[0]
            plus_row = table[(table["Variant"] == "GEE+") & (table["Model"] == model)].iloc[0]
            base_metrics = {metric: base_row[metric] for metric in ("R2", "MAE", "RMSE")}
            plus_metrics = {metric: plus_row[metric] for metric in ("R2", "MAE", "RMSE")}
        for metric in ("R2", "MAE", "RMSE"):
            close(
                row[f"Delta_{metric}"],
                plus_metrics[metric] - base_metrics[metric],
                f"GEE+ difference/{model}/{metric}",
            )
    return checked


def validate_clustering():
    inventory = pd.read_csv(require(RESULTS / "data" / "dataset_inventory.csv")).set_index("Dataset")
    summary = pd.read_csv(require(RESULTS / "tables" / "kmeans_selection.csv")).set_index("Dataset")
    checked = 0
    for dataset in ("ibm", "job_change", "promotion", "ghrm"):
        output = RESULTS / "clustering" / dataset
        metrics = pd.read_csv(require(output / "kmeans_metrics.csv"))
        if set(metrics["k"]) != set(range(2, 8)):
            raise AssertionError(f"k=2..7 is incomplete: {dataset}")
        selected = metrics.loc[metrics["Silhouette"].idxmax()]
        if int(summary.loc[dataset, "k"]) != int(selected["k"]):
            raise AssertionError(f"selected k is not maximum Silhouette: {dataset}")
        sizes = pd.read_csv(require(output / "cluster_sizes.csv"))
        expected_rows = int(inventory.loc[dataset, "Rows"])
        if int(sizes["Size"].sum()) != expected_rows:
            raise AssertionError(f"cluster membership count mismatch: {dataset}")
        assignment_path = output / "cluster_assignments.csv"
        if assignment_path.is_file():
            assignments = pd.read_csv(assignment_path)
            if len(assignments) != expected_rows or assignments["Row_Index"].duplicated().any():
                raise AssertionError(f"cluster assignments incomplete: {dataset}")
        if sizes["Cluster"].nunique() != int(selected["k"]):
            raise AssertionError(f"cluster labels incomplete: {dataset}")
        if dataset == "ghrm":
            require(output / "cluster_profile_numeric.csv")
            require(output / "cluster_profile_categorical.csv")
            require(output / "cluster_target_summary.csv")
            require(output / "cluster_distinguishing_features.csv")
        checked += 1
    return checked


def validate_report():
    report = require(RESULTS / "analysis_report.md").read_text(encoding="utf-8")
    forbidden = ("thesis_chapter4_reference", "sole numeric source of truth", "sync_thesis")
    for phrase in forbidden:
        if phrase in report:
            raise AssertionError(f"obsolete publication mechanism appears in report: {phrase}")
    for path in (
        RESULTS / "run_manifest.json",
        RESULTS / "run_log.txt",
        RESULTS / "tables" / "model_complexity.csv",
        RESULTS / "tables" / "cluster_patterns.csv",
    ):
        require(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="fail if any of the four raw CSV files is unavailable",
    )
    args = parser.parse_args()
    datasets = validate_data(args.require_data)
    classifiers = validate_classification()
    regressors = validate_regression()
    clusters = validate_clustering()
    validate_report()
    print(
        "validation passed: "
        f"{datasets} datasets, {classifiers} classifiers, "
        f"{regressors} regressors, {clusters} clustering runs"
    )


if __name__ == "__main__":
    main()

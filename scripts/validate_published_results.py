"""Validate every published Chapter 4 value against the final thesis reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sync_thesis_chapter4 import _source_rows, table_map  # noqa: E402


TABLES = ROOT / "results" / "tables"
REPORT = ROOT / "results" / "chapter4_results.md"
REFERENCE = ROOT / "results" / "thesis_chapter4_reference.json"
MANIFEST = ROOT / "validation" / "data_manifest.json"
CLASSIFICATION_TABLES = {
    "classification_ibm": "IBM HR Analytics",
    "classification_job_change": "Job Change",
    "classification_promotion": "Employee Promotion",
}
CONFUSION_TABLES = {
    "confusion_ibm": "IBM HR Analytics",
    "confusion_job_change": "Job Change",
    "confusion_promotion": "Employee Promotion",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    require(path.is_file(), f"Missing published table: {path.relative_to(ROOT)}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    require(bool(rows), f"Empty published table: {path.relative_to(ROOT)}")
    return rows[0], rows[1:]


def read_csv_dicts(name: str) -> list[dict[str, str]]:
    path = TABLES / name
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_all_19_source_tables(reference: dict) -> None:
    require(len(reference["tables"]) == 19, "The reference must contain tables 4-1 through 4-19")
    expected_numbers = {f"4-{number}" for number in range(1, 20)}
    require({table["number"] for table in reference["tables"]} == expected_numbers, "Chapter 4 table numbering is incomplete")

    for table in reference["tables"]:
        number = table["number"].replace("-", "_")
        path = TABLES / f"thesis_table_{number}_{table['slug']}.csv"
        columns, rows = read_csv_rows(path)
        require(columns == table["columns"], f"Column mismatch in {path.name}")
        require(rows == _source_rows(table), f"Value mismatch in {path.name}")


def validate_classification(reference: dict) -> None:
    tables = table_map(reference)
    metrics = read_csv_dicts("classification_metrics.csv")
    confusion = read_csv_dicts("confusion_matrices.csv")
    require(len(metrics) == 12, "classification_metrics.csv must contain 12 rows")
    require(len(confusion) == 12, "confusion_matrices.csv must contain 12 rows")

    expected_metrics = {}
    for slug, dataset in CLASSIFICATION_TABLES.items():
        for model, accuracy, precision, recall, f1 in tables[slug]["rows"]:
            expected_metrics[(dataset, model)] = tuple(f"{value:.3f}" for value in (accuracy, precision, recall, f1))
    for row in metrics:
        key = (row["Dataset"], row["Model"])
        require(key in expected_metrics, f"Unexpected classification row: {key}")
        require(
            tuple(row[name] for name in ("Accuracy", "Precision", "Recall", "F1")) == expected_metrics[key],
            f"Classification values differ from the thesis for {key}",
        )

    expected_confusion = {}
    for slug, dataset in CONFUSION_TABLES.items():
        for model, tn, fp, fn, tp in tables[slug]["rows"]:
            expected_confusion[(dataset, model)] = (tn, fp, fn, tp)
    for row in confusion:
        key = (row["Dataset"], row["Model"])
        require(key in expected_confusion, f"Unexpected confusion row: {key}")
        values = tuple(int(row[name]) for name in ("TN", "FP", "FN", "TP"))
        require(values == expected_confusion[key], f"Confusion values differ from the thesis for {key}")
        tn, fp, fn, tp = values
        require(int(row["Test_N"]) == sum(values), f"Test_N mismatch for {key}")
        computed = (
            (tn + tp) / sum(values),
            tp / (tp + fp),
            tp / (tp + fn),
            2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))),
        )
        published = expected_metrics[key]
        require(tuple(f"{value:.3f}" for value in computed) == published, f"Metrics do not round from the thesis confusion matrix for {key}")


def validate_regression_and_clustering(reference: dict) -> None:
    tables = table_map(reference)
    regression = read_csv_dicts("regression_metrics.csv")
    expected_regression = {}
    for slug, variant in (("regression_base", "Base"), ("regression_gee_plus", "GEE+")):
        for model, r2, rmse, mae in tables[slug]["rows"]:
            expected_regression[(variant, model)] = (f"{r2:.3f}", f"{rmse:.3f}", f"{mae:.3f}")
    require(len(regression) == 8, "regression_metrics.csv must contain eight rows")
    for row in regression:
        key = (row["Variant"], row["Model"])
        require(tuple(row[name] for name in ("R2", "RMSE", "MAE")) == expected_regression[key], f"Regression values differ from the thesis for {key}")

    kmeans = read_csv_dicts("kmeans_selection.csv")
    expected_kmeans = {
        dataset: (str(k), f"{sse:.2f}", f"{silhouette:.3f}", f"{db:.3f}")
        for dataset, k, sse, silhouette, db in tables["kmeans_selection"]["rows"]
    }
    require(len(kmeans) == 4, "kmeans_selection.csv must contain four rows")
    for row in kmeans:
        values = tuple(row[name] for name in ("Selected_k", "Inertia_SSE", "Silhouette", "Davies_Bouldin"))
        require(values == expected_kmeans[row["Dataset"]], f"K-Means values differ from the thesis for {row['Dataset']}")

    overview = {row["Dataset"]: int(row["Rows"]) for row in read_csv_dicts("dataset_overview.csv")}
    clusters = read_csv_dicts("cluster_sizes.csv")
    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    for row in clusters:
        totals[row["Dataset"]] = totals.get(row["Dataset"], 0) + int(row["Size"])
        counts[row["Dataset"]] = counts.get(row["Dataset"], 0) + 1
    for dataset, total in totals.items():
        require(total == overview[dataset], f"Cluster sizes do not sum to source rows for {dataset}")
        require(counts[dataset] == int(expected_kmeans[dataset][0]), f"Cluster count differs from selected k for {dataset}")


def validate_narrative_tables(reference: dict) -> None:
    expected_mcnemar = [[str(cell) for cell in row] for row in reference["narrative_results"]["mcnemar"]]
    columns, rows = read_csv_rows(TABLES / "mcnemar.csv")
    require(columns == ["Dataset", "Model_A", "Model_B", "p_value"], "mcnemar.csv columns changed")
    require(rows == expected_mcnemar, "McNemar values differ from the thesis narrative")

    report = REPORT.read_text(encoding="utf-8")
    require("٫" not in report, "Use an ASCII full stop as the decimal separator")
    require(reference["source"]["sha256"] in report, "Report does not identify the exact thesis source")
    for table in reference["tables"]:
        require(f"جدول {table['number'].translate(str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹'))}" in report, f"Report omits table {table['number']}")
        require(table["caption"] in report, f"Report omits the caption for table {table['number']}")

    figure_dir = ROOT / "results" / "figures"
    required_figures = {
        "class_balance.png",
        "classification_f1.png",
        "confusion_matrices.png",
        "ghrm_cluster_fep.png",
        "ghrm_regression_rmse.png",
        "kmeans_selected_quality.png",
    }
    require(
        all((figure_dir / name).is_file() for name in required_figures),
        "A thesis-backed result figure is missing",
    )


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
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
            row_count = sum(1 for _ in stream) - 1
        require(row_count == item["rows"], f"Row-count mismatch for {item['filename']}")
    if require_data:
        require(not missing, "Raw data missing: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-data", action="store_true", help="Require all four raw datasets and validate their hashes.")
    args = parser.parse_args()
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    validate_all_19_source_tables(reference)
    validate_classification(reference)
    validate_regression_and_clustering(reference)
    validate_narrative_tables(reference)
    validate_data(args.require_data)
    print("All 19 Chapter 4 tables and canonical outputs match the final thesis reference.")


if __name__ == "__main__":
    main()

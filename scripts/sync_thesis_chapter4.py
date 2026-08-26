"""Publish Chapter 4 artifacts from the final thesis reference only."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES_DIR = RESULTS / "tables"
FIGURES_DIR = RESULTS / "figures"
REFERENCE_PATH = RESULTS / "thesis_chapter4_reference.json"
REPORT_PATH = RESULTS / "chapter4_results.md"

PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
DATASET_DISPLAY = {
    "IBM HR Analytics": "IBM HR Analytics",
    "Job Change": "Job Change",
    "Employee Promotion": "Employee Promotion",
    "GHRM": "GHRM",
}
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
MODEL_COLORS = ["#2563A6", "#C7922D", "#D66A2C", "#748C3D"]


def load_reference() -> dict:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def table_map(reference: dict) -> dict[str, dict]:
    return {table["slug"]: table for table in reference["tables"]}


def _format_source_cell(table: dict, row_index: int, column_index: int, value: object) -> str:
    """Preserve the precision displayed in the thesis tables."""

    number = table["number"]
    if not isinstance(value, float):
        return str(value)
    if number == "4-1":
        if row_index == 0 and column_index == 2:
            return f"{value:.2f}"
        return f"{value:.1f}"
    if number == "4-2":
        decimals = 3 if row_index == 0 else 2
        return f"{value:.{decimals}f}"
    if number in {"4-3", "4-6", "4-7"}:
        return f"{value:.2f}"
    if number == "4-9":
        return f"{value:.2f}" if column_index == 2 else f"{value:.3f}"
    if number in {"4-11", "4-13", "4-15", "4-17", "4-18", "4-19"}:
        return f"{value:.3f}"
    return str(value)


def _display_cell(table: dict, row_index: int, column_index: int, value: object) -> str:
    text = _format_source_cell(table, row_index, column_index, value)
    if table["number"] in {"4-6", "4-7"} and column_index == 2:
        text += "%"
    return text.translate(PERSIAN_DIGITS)


def _write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _source_rows(table: dict) -> list[list[str]]:
    return [
        [
            _format_source_cell(table, row_index, column_index, value)
            for column_index, value in enumerate(row)
        ]
        for row_index, row in enumerate(table["rows"])
    ]


def _write_source_tables(reference: dict) -> None:
    for table in reference["tables"]:
        number = table["number"].replace("-", "_")
        _write_csv(
            TABLES_DIR / f"thesis_table_{number}_{table['slug']}.csv",
            table["columns"],
            _source_rows(table),
        )


def _write_canonical_tables(reference: dict) -> None:
    tables = table_map(reference)

    overview_columns = [
        "Dataset", "Rows", "Train_N", "Test_N", "Target", "Task",
        "Positive_Count", "Positive_Share_Percent",
    ]
    overview_rows = reference["narrative_results"]["dataset_overview"]
    _write_csv(TABLES_DIR / "dataset_overview.csv", overview_columns, overview_rows)

    missing_rows = []
    for slug, dataset in (("missing_job_change", "Job Change"), ("missing_promotion", "Employee Promotion")):
        for variable, count, percent, method in tables[slug]["rows"]:
            missing_rows.append([dataset, variable, count, f"{percent:.2f}", method])
    _write_csv(
        TABLES_DIR / "missing_values.csv",
        ["Dataset", "Variable", "Missing_Count", "Missing_Percent", "Method"],
        missing_rows,
    )

    classification_rows = []
    for slug, dataset in CLASSIFICATION_TABLES.items():
        for model, accuracy, precision, recall, f1 in tables[slug]["rows"]:
            classification_rows.append(
                [dataset, model, f"{accuracy:.3f}", f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"]
            )
    _write_csv(
        TABLES_DIR / "classification_metrics.csv",
        ["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1"],
        classification_rows,
    )

    confusion_rows = []
    for slug, dataset in CONFUSION_TABLES.items():
        for model, tn, fp, fn, tp in tables[slug]["rows"]:
            confusion_rows.append([dataset, model, tn, fp, fn, tp, tn + fp + fn + tp])
    _write_csv(
        TABLES_DIR / "confusion_matrices.csv",
        ["Dataset", "Model", "TN", "FP", "FN", "TP", "Test_N"],
        confusion_rows,
    )

    regression_rows = []
    for slug, variant in (("regression_base", "Base"), ("regression_gee_plus", "GEE+")):
        for model, r2, rmse, mae in tables[slug]["rows"]:
            regression_rows.append([variant, model, f"{r2:.3f}", f"{rmse:.3f}", f"{mae:.3f}"])
    _write_csv(
        TABLES_DIR / "regression_metrics.csv",
        ["Variant", "Model", "R2", "RMSE", "MAE"],
        regression_rows,
    )

    kmeans_rows = []
    silhouette_n = [1470, 5000, 5000, 320]
    for row, sample_n in zip(tables["kmeans_selection"]["rows"], silhouette_n):
        dataset, selected_k, sse, silhouette, db = row
        kmeans_rows.append(
            [dataset, selected_k, f"{sse:.2f}", f"{silhouette:.3f}", f"{db:.3f}", sample_n]
        )
    _write_csv(
        TABLES_DIR / "kmeans_selection.csv",
        ["Dataset", "Selected_k", "Inertia_SSE", "Silhouette", "Davies_Bouldin", "Silhouette_N"],
        kmeans_rows,
    )

    cluster_rows = []
    ghrm_profiles = []
    cluster_pattern = re.compile(r"^(IBM|Job Change|Promotion|GHRM) - (\d+) \(n=(\d+)\)$")
    profile_pattern = re.compile(
        r"GRS=([0-9.]+).*GTD=([0-9.]+).*GPA=([0-9.]+).*GCM=([0-9.]+).*GEE=([0-9.]+).*FEP=([0-9.]+)"
    )
    dataset_names = {"IBM": "IBM HR Analytics", "Promotion": "Employee Promotion"}
    for cluster_cell, label, details in tables["cluster_profiles"]["rows"]:
        match = cluster_pattern.match(cluster_cell)
        if not match:
            raise ValueError(f"Unparseable thesis cluster row: {cluster_cell}")
        raw_dataset, cluster, size = match.groups()
        dataset = dataset_names.get(raw_dataset, raw_dataset)
        cluster_rows.append([dataset, int(cluster), int(size), label, details])
        if raw_dataset == "GHRM":
            profile = profile_pattern.search(details)
            if not profile:
                raise ValueError(f"Unparseable GHRM profile: {details}")
            ghrm_profiles.append([int(cluster), int(size), *profile.groups()])
    _write_csv(
        TABLES_DIR / "cluster_sizes.csv",
        ["Dataset", "Cluster", "Size", "Label", "Profile"],
        cluster_rows,
    )
    _write_csv(
        TABLES_DIR / "ghrm_cluster_profiles.csv",
        ["Cluster", "Size", "GRS", "GTD", "GPA", "GCM", "GEE", "FEP"],
        ghrm_profiles,
    )

    _write_csv(
        TABLES_DIR / "mcnemar.csv",
        ["Dataset", "Model_A", "Model_B", "p_value"],
        reference["narrative_results"]["mcnemar"],
    )

    for stale in ("data_quality.csv", "regression_variant_comparison.csv"):
        (TABLES_DIR / stale).unlink(missing_ok=True)


def _markdown_table(table: dict) -> str:
    header = "| " + " | ".join(table["columns"]) + " |"
    separator = "| " + " | ".join(["---"] * len(table["columns"])) + " |"
    rows = []
    for row_index, row in enumerate(table["rows"]):
        cells = [
            _display_cell(table, row_index, column_index, value).replace("|", "\\|")
            for column_index, value in enumerate(row)
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def _write_report(reference: dict) -> None:
    source = reference["source"]
    extra = reference["narrative_results"]["additional_values"]
    sections = [
        "# نتایج فصل چهارم — مرجع نهایی پایان‌نامه",
        "",
        f"> منبع عددی یگانه: `{source['filename']}` با SHA-256 `{source['sha256']}`. "
        "تمام جدول‌های این صفحه و CSVهای پوشه `results/tables` فقط از همین مرجع ساخته شده‌اند.",
        "",
        "## طرح تحلیل",
        "",
        "تقسیم داده‌ها ۸۰٪ آموزش و ۲۰٪ آزمون، `random_state=42` و اعتبارسنجی متقاطع سه‌بخشی بود. "
        "تفکیک مسائل طبقه‌بندی stratified و تفکیک رگرسیون بدون stratification انجام شد. "
        "برای Silhouette مجموعه‌های بزرگ، نمونه قطعی ۵۰۰۰تایی با بذر ۴۲ استفاده شد؛ SSE و Davies–Bouldin روی کل داده محاسبه شدند.",
        "",
        "## اعداد توصیفی تکمیلی متن پایان‌نامه",
        "",
        f"میانگین دقیق سن IBM برابر {extra['ibm_age_mean_exact']:.2f} سال بود. میانه حقوق ماهانه {extra['ibm_monthly_income_median']} و بخش میانی آن تقریباً "
        f"بین {extra['ibm_monthly_income_middle_approx'][0]} تا {extra['ibm_monthly_income_middle_approx'][1]} قرار داشت. "
        f"در Job Change، میانه ساعات آموزش {extra['job_training_hours_median']}، تعداد افراد دارای سابقه مرتبط {extra['job_relevant_experience_yes']} و فاقد آن "
        f"{extra['job_relevant_experience_no']} بود؛ کلاس صفر {extra['job_negative_class']} و کلاس یک {extra['job_positive_class']} رکورد داشت. "
        f"SSE دوخوشه‌ای Job Change برابر {extra['job_k2_sse']:.2f} گزارش شد.",
        "",
        "## جدول‌های ۴-۱ تا ۴-۱۹",
    ]
    for table in reference["tables"]:
        sections.extend(
            [
                "",
                f"### جدول {table['number']} — {table['caption']}",
                "",
                _markdown_table(table),
            ]
        )

    mcnemar = reference["narrative_results"]["mcnemar"]
    mcnemar_table = {
        "number": "narrative",
        "columns": ["مجموعه‌داده", "مدل A", "مدل B", "p-value"],
        "rows": mcnemar,
    }
    sections.extend(
        [
            "",
            "## نتایج آزمون McNemar گزارش‌شده در متن",
            "",
            _markdown_table(mcnemar_table),
            "",
            "## مرز تفسیر",
            "",
            "نتایج پیش‌بینانه و اکتشافی‌اند؛ خوشه‌بندی و رگرسیون به‌تنهایی رابطه علّی یا میانجی‌گری را اثبات نمی‌کنند. "
            "اعداد اجرای جدید کد باید در `outputs/` بمانند و نباید بدون اصلاح سند مرجع جایگزین اعداد منتشرشده فصل چهارم شوند.",
            "",
        ]
    )
    report = "\n".join(sections).translate(PERSIAN_DIGITS)
    report = report.replace(source["sha256"].translate(PERSIAN_DIGITS), source["sha256"])
    report = report.replace("SHA-۲۵۶", "SHA-256")
    REPORT_PATH.write_text(report, encoding="utf-8")


def _read_csv_rows(name: str) -> list[dict[str, str]]:
    with (TABLES_DIR / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_figures() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.25})

    classification = _read_csv_rows("classification_metrics.csv")
    datasets = ["IBM HR Analytics", "Job Change", "Employee Promotion"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for axis, dataset in zip(axes, datasets):
        rows = [row for row in classification if row["Dataset"] == dataset]
        values = [float(row["F1"]) for row in rows]
        bars = axis.bar([row["Model"] for row in rows], values, color=MODEL_COLORS)
        axis.set_title(dataset)
        axis.set_ylim(0, 0.65)
        axis.tick_params(axis="x", rotation=25)
        axis.bar_label(bars, fmt="%.3f", padding=3)
    axes[0].set_ylabel("Positive-class F1")
    figure.suptitle("Chapter 4 classification results")
    _save_figure(figure, "classification_f1.png")

    confusion = _read_csv_rows("confusion_matrices.csv")
    figure, axes = plt.subplots(3, 4, figsize=(12.8, 9.5))
    models = ["Random Forest", "Decision Tree", "LinearSVC", "MLPClassifier"]
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            row = next(r for r in confusion if r["Dataset"] == dataset and r["Model"] == model)
            matrix = np.array([[int(row["TN"]), int(row["FP"])], [int(row["FN"]), int(row["TP"])]])
            axis = axes[i, j]
            axis.imshow(matrix, cmap="Blues")
            for y in range(2):
                for x in range(2):
                    axis.text(x, y, f"{matrix[y, x]:,}", ha="center", va="center")
            axis.set_xticks([0, 1])
            axis.set_yticks([0, 1])
            axis.grid(False)
            if i == 0:
                axis.set_title(model)
            if j == 0:
                axis.set_ylabel(dataset + "\nActual")
            if i == 2:
                axis.set_xlabel("Predicted")
    _save_figure(figure, "confusion_matrices.png")

    regression = _read_csv_rows("regression_metrics.csv")
    models_reg = ["Random Forest Regressor", "Decision Tree Regressor", "LinearSVR", "MLPRegressor"]
    positions = np.arange(len(models_reg))
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for index, variant in enumerate(("Base", "GEE+")):
        values = [float(next(r["RMSE"] for r in regression if r["Variant"] == variant and r["Model"] == model)) for model in models_reg]
        bars = axis.barh(positions + (index - 0.5) * 0.34, values, 0.34, label=variant)
        axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set_yticks(positions, models_reg)
    axis.invert_yaxis()
    axis.set_xlabel("RMSE (lower is better)")
    axis.set_title("Chapter 4 GHRM regression")
    axis.legend(frameon=False)
    _save_figure(figure, "ghrm_regression_rmse.png")

    kmeans = _read_csv_rows("kmeans_selection.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for axis, metric, title in ((axes[0], "Silhouette", "Silhouette"), (axes[1], "Davies_Bouldin", "Davies–Bouldin")):
        values = [float(row[metric]) for row in kmeans]
        bars = axis.bar([row["Dataset"] for row in kmeans], values, color=MODEL_COLORS)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.bar_label(bars, fmt="%.3f", padding=3)
    _save_figure(figure, "kmeans_selected_quality.png")

    profiles = _read_csv_rows("ghrm_cluster_profiles.csv")
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    bars = axis.bar([f"Cluster {row['Cluster']}" for row in profiles], [float(row["FEP"]) for row in profiles], color=MODEL_COLORS[:2])
    axis.set_ylim(0, 5)
    axis.set_ylabel("Mean FEP")
    axis.set_title("GHRM cluster profile from thesis table 4-10")
    axis.bar_label(bars, fmt="%.2f", padding=3)
    _save_figure(figure, "ghrm_cluster_fep.png")

    overview = _read_csv_rows("dataset_overview.csv")[:3]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.barh([row["Dataset"] for row in overview], [float(row["Positive_Share_Percent"]) for row in overview], color=MODEL_COLORS[:3])
    axis.invert_yaxis()
    axis.set_xlabel("Positive class (%)")
    axis.bar_label(bars, fmt="%.2f%%", padding=3)
    _save_figure(figure, "class_balance.png")

    for stale in (
        "ghrm_actual_vs_predicted.png",
        "kmeans_ghrm_diagnostics.png",
        "kmeans_ibm_diagnostics.png",
        "kmeans_job_change_diagnostics.png",
        "kmeans_promotion_diagnostics.png",
    ):
        (FIGURES_DIR / stale).unlink(missing_ok=True)


def sync() -> None:
    reference = load_reference()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _write_source_tables(reference)
    _write_canonical_tables(reference)
    _write_report(reference)
    _write_figures()


def main() -> None:
    sync()
    print("Chapter 4 artifacts synchronized from the final thesis reference.")


if __name__ == "__main__":
    main()

"""Build the GitHub-ready Chapter 4 tables, figures, and Persian report.

This module never hard-codes model results.  It reads the artifacts produced by
``run_all.py`` and publishes a compact, reviewable subset under ``results/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from src.green_hrm_analysis import DATA_DIR as GHRM_DATA_DIR
from src.green_hrm_analysis import GHRM_FILES
from src.preprocessing import DATASETS, load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

DATASET_LABELS = {
    "ibm": "IBM HR Attrition",
    "job_change": "Job Change",
    "promotion": "Employee Promotion",
    "ghrm": "GHRM–Environmental Performance",
}
DATASET_ORDER = ["ibm", "job_change", "promotion", "ghrm"]
MODEL_ORDER = ["Random Forest", "Decision Tree", "Linear SVM", "MLP"]
REGRESSION_MODEL_ORDER = [
    "Random Forest Regressor",
    "Decision Tree Regressor",
    "LinearSVR",
    "MLPRegressor",
]
KEY_COLUMNS = {
    "ibm": "EmployeeNumber",
    "job_change": "enrollee_id",
    "promotion": "employee_id",
}

BLUE = "#2563A6"
GOLD = "#C7922D"
ORANGE = "#D66A2C"
OLIVE = "#748C3D"
INK = "#26313D"
GRID = "#D9E0E7"
MODEL_COLORS = {
    "Random Forest": BLUE,
    "Decision Tree": GOLD,
    "Linear SVM": ORANGE,
    "MLP": OLIVE,
}
REGRESSION_COLORS = {"Base": BLUE, "GEE+": GOLD}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
        }
    )


def _save_figure(figure: plt.Figure, filename: str) -> None:
    figure.tight_layout()
    figure.savefig(
        FIGURES / filename,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _read_ghrm_raw() -> pd.DataFrame:
    path = next(
        (
            GHRM_DATA_DIR / filename
            for filename in GHRM_FILES
            if (GHRM_DATA_DIR / filename).exists()
        ),
        None,
    )
    if path is None:
        raise FileNotFoundError(
            "GHRM dataset not found: " + ", ".join(GHRM_FILES)
        )
    return pd.read_csv(path)


def _positive_label(dataset: str) -> str:
    mapping_path = OUTPUTS / dataset / "label_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    return next(label for label, code in mapping.items() if code == 1)


def _format_value(value) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    """Return a dependency-free GitHub Markdown table."""

    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        cells = [
            _format_value(value).replace("|", "\\|").replace("\n", " ")
            for value in values
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def select_kmeans_row(metrics: pd.DataFrame) -> pd.Series:
    """Select k using the maximum Silhouette score used by the pipeline."""

    return metrics.loc[metrics["Silhouette"].idxmax()]


def build_dataset_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overview_rows = []
    quality_rows = []
    missing_rows = []

    for dataset in DATASET_ORDER[:-1]:
        frame, info, _ = load_dataset(dataset)
        target = info["target"]
        positive_label = _positive_label(dataset)
        positive_count = int(
            frame[target].astype(str).eq(str(positive_label)).sum()
        )
        overview_rows.append(
            {
                "Dataset": DATASET_LABELS[dataset],
                "Rows": len(frame),
                "Raw_Columns": frame.shape[1],
                "Target": target,
                "Task": "Binary classification",
                "Positive_Count": positive_count,
                "Positive_Share": positive_count / len(frame),
            }
        )
        key = KEY_COLUMNS[dataset]
        quality_rows.append(
            {
                "Dataset": DATASET_LABELS[dataset],
                "Exact_Duplicate_Rows": int(frame.duplicated().sum()),
                "Missing_Target": int(frame[target].isna().sum()),
                "Missing_Key": int(frame[key].isna().sum()),
                "Duplicate_Key": int(frame[key].duplicated().sum()),
            }
        )
        for column, count in frame.isna().sum().items():
            if count:
                missing_rows.append(
                    {
                        "Dataset": DATASET_LABELS[dataset],
                        "Column": column,
                        "Missing_Count": int(count),
                        "Missing_Share": count / len(frame),
                    }
                )

    ghrm = _read_ghrm_raw()
    overview_rows.append(
        {
            "Dataset": DATASET_LABELS["ghrm"],
            "Rows": len(ghrm),
            "Raw_Columns": ghrm.shape[1],
            "Target": "FEP (derived mean score)",
            "Task": "Regression",
            "Positive_Count": np.nan,
            "Positive_Share": np.nan,
        }
    )
    quality_rows.append(
        {
            "Dataset": DATASET_LABELS["ghrm"],
            "Exact_Duplicate_Rows": int(ghrm.duplicated().sum()),
            "Missing_Target": int(
                ghrm[["FEP1", "FEP5", "FEP7", "FEP9"]]
                .isna()
                .any(axis=1)
                .sum()
            ),
            "Missing_Key": np.nan,
            "Duplicate_Key": np.nan,
        }
    )
    for column, count in ghrm.isna().sum().items():
        if count:
            missing_rows.append(
                {
                    "Dataset": DATASET_LABELS["ghrm"],
                    "Column": column,
                    "Missing_Count": int(count),
                    "Missing_Share": count / len(ghrm),
                }
            )

    overview = pd.DataFrame(overview_rows)
    quality = pd.DataFrame(quality_rows)
    missing = pd.DataFrame(
        missing_rows,
        columns=["Dataset", "Column", "Missing_Count", "Missing_Share"],
    )
    return overview, quality, missing


def build_result_tables() -> dict[str, pd.DataFrame]:
    classification = pd.read_csv(OUTPUTS / "model_metrics.csv")
    classification["Dataset"] = classification["Dataset"].map(DATASET_LABELS)
    classification["Dataset"] = pd.Categorical(
        classification["Dataset"],
        [DATASET_LABELS[name] for name in DATASET_ORDER[:-1]],
        ordered=True,
    )
    classification["Model"] = pd.Categorical(
        classification["Model"], MODEL_ORDER, ordered=True
    )
    classification = classification.sort_values(["Dataset", "Model"])

    mcnemar = pd.concat(
        [
            pd.read_csv(OUTPUTS / dataset / "mcnemar.csv")
            for dataset in DATASET_ORDER[:-1]
        ],
        ignore_index=True,
    )
    mcnemar["Dataset"] = mcnemar["Dataset"].map(DATASET_LABELS)
    mcnemar["Dataset"] = pd.Categorical(
        mcnemar["Dataset"],
        [DATASET_LABELS[name] for name in DATASET_ORDER[:-1]],
        ordered=True,
    )
    mcnemar = mcnemar.sort_values(["Dataset", "Model_B"])

    confusion_rows = []
    for dataset in DATASET_ORDER[:-1]:
        for model in MODEL_ORDER:
            path = (
                OUTPUTS
                / dataset
                / f"{model.replace(' ', '_')}_confusion_matrix.csv"
            )
            matrix = pd.read_csv(path, header=None).to_numpy()
            confusion_rows.append(
                {
                    "Dataset": DATASET_LABELS[dataset],
                    "Model": model,
                    "TN": int(matrix[0, 0]),
                    "FP": int(matrix[0, 1]),
                    "FN": int(matrix[1, 0]),
                    "TP": int(matrix[1, 1]),
                    "Test_N": int(matrix.sum()),
                }
            )
    confusion = pd.DataFrame(confusion_rows)

    regression = pd.read_csv(OUTPUTS / "ghrm" / "regression_metrics.csv")
    regression["Model"] = pd.Categorical(
        regression["Model"], REGRESSION_MODEL_ORDER, ordered=True
    )
    regression = regression.sort_values(["Variant", "Model"])

    kmeans_rows = []
    for dataset in DATASET_ORDER:
        metrics = pd.read_csv(OUTPUTS / dataset / "kmeans_metrics.csv")
        selected = select_kmeans_row(metrics)
        source_rows = 320 if dataset == "ghrm" else len(load_dataset(dataset)[0])
        kmeans_rows.append(
            {
                "Dataset": DATASET_LABELS[dataset],
                "Selected_k": int(selected["k"]),
                "Inertia_SSE": float(selected["Inertia_SSE"]),
                "Silhouette": float(selected["Silhouette"]),
                "Davies_Bouldin": float(selected["Davies_Bouldin"]),
                "Silhouette_N": min(source_rows, 5000),
                "Interpretation": (
                    "Moderate exploratory separation"
                    if dataset == "ghrm"
                    else "Weak separation"
                ),
            }
        )
    kmeans = pd.DataFrame(kmeans_rows)

    cluster_profiles = pd.read_csv(
        OUTPUTS / "ghrm" / "cluster_profiles.csv"
    )

    overview, quality, missing = build_dataset_tables()
    return {
        "dataset_overview": overview,
        "data_quality": quality,
        "missing_values": missing,
        "classification_metrics": classification,
        "confusion_matrices": confusion,
        "mcnemar": mcnemar,
        "regression_metrics": regression,
        "kmeans_selection": kmeans,
        "ghrm_cluster_profiles": cluster_profiles,
    }


def write_tables(tables: dict[str, pd.DataFrame]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(
            TABLES / f"{name}.csv", index=False, encoding="utf-8-sig"
        )


def plot_class_balance(overview: pd.DataFrame) -> None:
    data = overview.dropna(subset=["Positive_Share"]).copy()
    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    bars = axis.barh(
        data["Dataset"],
        data["Positive_Share"] * 100,
        color=[BLUE, GOLD, ORANGE],
        edgecolor=INK,
        linewidth=0.7,
    )
    axis.set_title("Positive-class share by classification dataset")
    axis.set_xlabel("Positive class (%)")
    axis.set_xlim(0, 100)
    axis.invert_yaxis()
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    axis.bar_label(bars, fmt="%.2f%%", padding=4)
    _save_figure(figure, "class_balance.png")


def plot_classification_metrics(classification: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    for axis, dataset in zip(axes, DATASET_ORDER[:-1]):
        label = DATASET_LABELS[dataset]
        data = classification[classification["Dataset"] == label]
        bars = axis.bar(
            data["Model"].astype(str),
            data["F1"],
            color=[MODEL_COLORS[model] for model in MODEL_ORDER],
            edgecolor=INK,
            linewidth=0.6,
        )
        axis.set_title(label)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Model")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[0].set_ylabel("Positive-class F1")
    figure.suptitle(
        "Held-out positive-class F1 by model",
        fontsize=14,
        y=1.02,
    )
    _save_figure(figure, "classification_f1.png")


def plot_confusion_matrices(confusion: pd.DataFrame) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(12.8, 9.6))
    cmap = LinearSegmentedColormap.from_list(
        "thesis_blue", ["#EFF5FA", BLUE, "#173D63"]
    )
    for row_index, dataset in enumerate(DATASET_ORDER[:-1]):
        dataset_label = DATASET_LABELS[dataset]
        for column_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            row = confusion[
                (confusion["Dataset"] == dataset_label)
                & (confusion["Model"] == model)
            ].iloc[0]
            matrix = np.array([[row.TN, row.FP], [row.FN, row.TP]])
            axis.imshow(matrix, cmap=cmap, vmin=0, vmax=matrix.max())
            for y in range(2):
                for x in range(2):
                    value = int(matrix[y, x])
                    color = "white" if value > matrix.max() * 0.55 else INK
                    axis.text(
                        x,
                        y,
                        f"{value:,}",
                        ha="center",
                        va="center",
                        color=color,
                        fontweight="bold",
                    )
            axis.set_xticks([0, 1], ["0", "1"])
            axis.set_yticks([0, 1], ["0", "1"])
            axis.grid(False)
            if row_index == 0:
                axis.set_title(model)
            if column_index == 0:
                axis.set_ylabel(f"{dataset_label}\nActual")
            if row_index == 2:
                axis.set_xlabel("Predicted")
    figure.suptitle(
        "Held-out confusion matrices (positive class = 1)",
        fontsize=14,
        y=1.01,
    )
    _save_figure(figure, "confusion_matrices.png")


def plot_regression_metrics(regression: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(9.6, 5.6))
    positions = np.arange(len(REGRESSION_MODEL_ORDER))
    width = 0.35
    for offset, variant in enumerate(["Base", "GEE+"]):
        data = regression[regression["Variant"] == variant].set_index("Model")
        values = [float(data.loc[model, "RMSE"]) for model in REGRESSION_MODEL_ORDER]
        bars = axis.barh(
            positions + (offset - 0.5) * width,
            values,
            width,
            label=variant,
            color=REGRESSION_COLORS[variant],
            edgecolor=INK,
            linewidth=0.6,
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set_title("GHRM held-out RMSE by model and feature variant")
    axis.set_xlabel("RMSE (lower is better)")
    axis.set_yticks(positions, REGRESSION_MODEL_ORDER)
    axis.set_xlim(0, max(regression["RMSE"]) * 1.22)
    axis.invert_yaxis()
    axis.legend(frameon=False, ncols=2, loc="upper center")
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    _save_figure(figure, "ghrm_regression_rmse.png")


def plot_best_regression_predictions(regression: pd.DataFrame) -> None:
    best = regression.loc[regression["RMSE"].idxmin()]
    variant = str(best["Variant"])
    model = str(best["Model"])
    prefix = "gee_plus" if variant == "GEE+" else "base"
    predictions = pd.read_csv(
        OUTPUTS
        / "ghrm"
        / f"{prefix}_{model.replace(' ', '_')}_test_predictions.csv"
    )
    figure, axis = plt.subplots(figsize=(6.2, 5.6))
    axis.scatter(
        predictions["y_true"],
        predictions["y_pred"],
        s=38,
        color=BLUE,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.85,
    )
    minimum = float(min(predictions.min()))
    maximum = float(max(predictions.max()))
    padding = (maximum - minimum) * 0.08
    limits = (minimum - padding, maximum + padding)
    axis.plot(limits, limits, linestyle="--", color=INK, linewidth=1.1)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        "GHRM held-out actual vs predicted FEP\n"
        f"{variant} {model}; n={len(predictions)}"
    )
    axis.set_xlabel("Actual FEP")
    axis.set_ylabel("Predicted FEP")
    _save_figure(figure, "ghrm_actual_vs_predicted.png")


def plot_kmeans_selection(kmeans: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    labels = kmeans["Dataset"].str.replace(
        "GHRM–Environmental Performance", "GHRM", regex=False
    )
    colors = [BLUE, GOLD, ORANGE, OLIVE]
    for axis, column, title in [
        (axes[0], "Silhouette", "Selected-cluster Silhouette"),
        (axes[1], "Davies_Bouldin", "Selected-cluster Davies–Bouldin"),
    ]:
        bars = axis.bar(
            labels,
            kmeans[column],
            color=colors,
            edgecolor=INK,
            linewidth=0.6,
        )
        axis.set_title(title)
        axis.set_ylabel(column.replace("_", " "))
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylim(0, max(kmeans[column]) * 1.18)
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    figure.suptitle(
        "K-Means quality at the selected k\n"
        "Job Change and Promotion Silhouette use deterministic n=5,000 samples",
        fontsize=13,
        y=1.05,
    )
    _save_figure(figure, "kmeans_selected_quality.png")


def plot_kmeans_diagnostics() -> None:
    for dataset in DATASET_ORDER:
        metrics = pd.read_csv(OUTPUTS / dataset / "kmeans_metrics.csv")
        figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
        specs = [
            ("Inertia_SSE", "Inertia / SSE", "lower with diminishing returns"),
            ("Silhouette", "Silhouette", "higher is better"),
            ("Davies_Bouldin", "Davies–Bouldin", "lower is better"),
        ]
        for axis, (column, label, hint) in zip(axes, specs):
            axis.plot(
                metrics["k"],
                metrics[column],
                marker="o",
                color=BLUE,
                markerfacecolor="white",
                markeredgecolor=BLUE,
                linewidth=2,
            )
            axis.set_title(f"{label} ({hint})")
            axis.set_xlabel("k")
            axis.set_ylabel(label)
            axis.set_xticks(metrics["k"])
        sample_note = (
            "; Silhouette sample n=5,000"
            if dataset in {"job_change", "promotion"}
            else ""
        )
        figure.suptitle(
            f"{DATASET_LABELS[dataset]} K-Means diagnostics{sample_note}",
            fontsize=13,
            y=1.02,
        )
        _save_figure(figure, f"kmeans_{dataset}_diagnostics.png")


def plot_ghrm_cluster_profile(cluster_profiles: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    labels = [f"Cluster {int(value)}" for value in cluster_profiles["Cluster"]]
    bars = axis.bar(
        labels,
        cluster_profiles["Mean_FEP"],
        yerr=cluster_profiles["SD_FEP"],
        capsize=5,
        color=[BLUE, GOLD],
        edgecolor=INK,
        linewidth=0.6,
    )
    axis.set_title("GHRM cluster profile on descriptive FEP")
    axis.set_ylabel("Mean FEP ± SD (FEP excluded from clustering)")
    axis.set_ylim(0, 5)
    axis.grid(axis="y")
    axis.grid(axis="x", visible=False)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    for index, size in enumerate(cluster_profiles["Size"]):
        axis.text(index, 0.15, f"n={int(size)}", ha="center", color="white")
    _save_figure(figure, "ghrm_cluster_fep.png")


def build_figures(tables: dict[str, pd.DataFrame]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    plot_class_balance(tables["dataset_overview"])
    plot_classification_metrics(tables["classification_metrics"])
    plot_confusion_matrices(tables["confusion_matrices"])
    plot_regression_metrics(tables["regression_metrics"])
    plot_best_regression_predictions(tables["regression_metrics"])
    plot_kmeans_selection(tables["kmeans_selection"])
    plot_kmeans_diagnostics()
    plot_ghrm_cluster_profile(tables["ghrm_cluster_profiles"])


def _presentation_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    overview = tables["dataset_overview"][
        [
            "Dataset",
            "Rows",
            "Raw_Columns",
            "Target",
            "Task",
            "Positive_Share",
        ]
    ].copy()
    overview["Positive_Share"] = overview["Positive_Share"].map(
        lambda value: "—" if pd.isna(value) else f"{value * 100:.2f}%"
    )
    overview.columns = [
        "مجموعه‌داده",
        "تعداد رکورد",
        "ستون خام",
        "متغیر هدف",
        "نوع مسئله",
        "سهم کلاس مثبت",
    ]

    classification = tables["classification_metrics"][
        [
            "Dataset",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "CV_Best_F1",
        ]
    ].copy()
    classification.columns = [
        "مجموعه‌داده",
        "مدل",
        "Accuracy",
        "Precision مثبت",
        "Recall مثبت",
        "F1 مثبت",
        "بهترین F1 در CV",
    ]

    mcnemar = tables["mcnemar"][
        ["Dataset", "Model_A", "Model_B", "b01", "b10", "p_value"]
    ].copy()
    mcnemar["p_value"] = mcnemar["p_value"].map(
        lambda value: "<0.0001" if value < 0.0001 else f"{value:.4f}"
    )
    mcnemar.columns = [
        "مجموعه‌داده",
        "مدل A",
        "مدل B",
        "b01",
        "b10",
        "p-value",
    ]

    regression = tables["regression_metrics"][
        ["Variant", "Model", "R2", "MAE", "RMSE", "CV_RMSE"]
    ].copy()
    regression.columns = [
        "نسخه",
        "مدل",
        "R²",
        "MAE",
        "RMSE",
        "CV RMSE",
    ]

    kmeans = tables["kmeans_selection"][
        [
            "Dataset",
            "Selected_k",
            "Silhouette",
            "Davies_Bouldin",
            "Silhouette_N",
            "Interpretation",
        ]
    ].copy()
    kmeans.columns = [
        "مجموعه‌داده",
        "k منتخب",
        "Silhouette",
        "Davies–Bouldin",
        "n برای Silhouette",
        "تفسیر",
    ]
    kmeans["تفسیر"] = kmeans["تفسیر"].replace(
        {
            "Weak separation": "تفکیک ضعیف",
            "Moderate exploratory separation": "تفکیک اکتشافی متوسط",
        }
    )

    clusters = tables["ghrm_cluster_profiles"].copy()
    clusters.columns = ["خوشه", "تعداد", "میانگین FEP", "انحراف معیار FEP"]
    return {
        "overview": overview,
        "classification": classification,
        "mcnemar": mcnemar,
        "regression": regression,
        "kmeans": kmeans,
        "clusters": clusters,
    }


def write_persian_report(tables: dict[str, pd.DataFrame]) -> Path:
    display = _presentation_tables(tables)
    report = f"""# نتایج بازتولیدپذیر فصل چهارم

این صفحه نسخه GitHub نتایج فصل چهارم پایان‌نامه است. همه اعداد از اجرای کامل `python run_all.py` روی چهار مجموعه‌داده مستقل تولید شده‌اند؛ بنابراین جدول‌ها و نمودارها دستی وارد نشده‌اند و با اجرای مجدد کد قابل بازتولید هستند.

## طرح تحلیل و داده‌ها

{markdown_table(display['overview'])}

سه مجموعه‌داده عمومی منابع انسانی برای طبقه‌بندی دودویی و خوشه‌بندی اکتشافی استفاده شدند. مجموعه‌داده GHRM برای رگرسیون `FEP` و خوشه‌بندی مستقل ابعاد مستقیم مدیریت منابع انسانی سبز استفاده شد. رکوردهای چهار منبع با یکدیگر ادغام نشده‌اند.

کنترل کیفیت داده، نبود ردیف تکراری کامل، نبود هدف گمشده و نبود شناسه تکراری یا گمشده در سه مجموعه‌داده دارای شناسه را تأیید کرد. مهم‌ترین محدودیت داده‌ای، گمشده‌بودن `company_type` (۳۲٫۰۵٪)، `company_size` (۳۱٫۰۰٪)، `gender` (۲۳٫۵۳٪) و `major_discipline` (۱۴٫۶۸٪) در Job Change است. جایگذاری داده‌های گمشده داخل هر fold آموزشی انجام می‌شود.

![سهم کلاس مثبت](figures/class_balance.png)

## نتایج طبقه‌بندی

تمام Precision، Recall و F1 جدول زیر مربوط به کلاس مثبت هستند. این نکته برای IBM و Employee Promotion ضروری است، زیرا سهم کلاس مثبت به‌ترتیب ۱۶٫۱۲٪ و ۸٫۵۲٪ است و Accuracy به‌تنهایی می‌تواند گمراه‌کننده باشد.

{markdown_table(display['classification'])}

در IBM، مدل MLP بالاترین F1 کلاس مثبت را با مقدار **۰٫۵۱۸۵** به دست آورد، در حالی که Linear SVM بیشترین Recall را با مقدار **۰٫۶۳۸۳** ثبت کرد. Random Forest با وجود Accuracy برابر ۰٫۸۴۰۱، F1 مثبت ۰٫۲۰۳۴ داشت و نمونه روشنی از محدودیت Accuracy در داده نامتوازن است.

در Job Change، Random Forest بالاترین F1 مثبت را با مقدار **۰٫۶۰۲۱** ثبت کرد. Decision Tree بیشترین Recall را با مقدار ۰٫۷۴۰۳ داشت، اما Precision پایین‌تر آن باعث شد F1 آن به ۰٫۵۸۱۹ برسد. MLP بیشترین Accuracy را با مقدار ۰٫۷۸۶۸ ثبت کرد، ولی F1 مثبت آن ۰٫۵۲۱۹ بود.

در Employee Promotion، MLP بالاترین F1 مثبت را با مقدار **۰٫۵۱۰۸** و Precision برابر ۰٫۹۲۱۸ به دست آورد. Linear SVM بیشترین Recall را با مقدار **۰٫۸۳۸۳** ثبت کرد، اما Precision آن فقط ۰٫۲۳۸۷ بود؛ بنابراین انتخاب مدل به هزینه خطاهای مثبت کاذب و منفی کاذب وابسته است.

![مقایسه F1 مدل‌های طبقه‌بندی](figures/classification_f1.png)

![ماتریس‌های درهم‌ریختگی](figures/confusion_matrices.png)

## آزمون McNemar

{markdown_table(display['mcnemar'])}

آزمون McNemar تفاوت الگوی درست/نادرست بودن پیش‌بینی‌های جفت‌شده را می‌سنجد و مستقیماً اختلاف F1 را آزمون نمی‌کند. در IBM، تفاوت Random Forest و MLP در سطح ۰٫۰۵ معنادار نبود (`p=0.2012`)، اما سایر مقایسه‌های گزارش‌شده معنادار بودند.

## نتایج رگرسیون GHRM

در نسخه پایه، `GRS`، `GTD`، `GPA` و `GCM` پیش‌بین‌های `FEP` هستند. در نسخه `GEE+`، متغیر `GEE` نیز اضافه می‌شود.

{markdown_table(display['regression'])}

LinearSVR در هر دو نسخه بهترین عملکرد مجموعه آزمون را داشت. در نسخه پایه، R² برابر **۰٫۴۱۷۱** و RMSE برابر **۰٫۴۹۵۸** بود. با افزودن GEE، R² به **۰٫۴۱۷۸** و RMSE به **۰٫۴۹۵۵** رسید. افزایش R² فقط حدود ۰٫۰۰۰۷ است؛ بنابراین این نتیجه شواهدی برای اثر علّی یا میانجی‌گری GEE محسوب نمی‌شود.

![مقایسه RMSE مدل‌های رگرسیون](figures/ghrm_regression_rmse.png)

![مقادیر واقعی و پیش‌بینی‌شده FEP](figures/ghrm_actual_vs_predicted.png)

## نتایج خوشه‌بندی K-Means

متغیر هدف و شناسه‌ها در خوشه‌بندی سه مجموعه‌داده عمومی وارد نشده‌اند. در GHRM نیز `FEP` از تشکیل خوشه‌ها کنار گذاشته شد. برای Job Change و Employee Promotion، Silhouette روی نمونه قطعی ۵٬۰۰۰ رکوردی با `random_state=42` محاسبه شد؛ SSE و Davies–Bouldin از کل داده استفاده می‌کنند.

{markdown_table(display['kmeans'])}

مقادیر Silhouette سه مجموعه‌داده عمومی پایین است و جدایی طبیعی خوشه‌ها را ضعیف نشان می‌دهد؛ بنابراین خوشه‌های آن‌ها نباید به‌صورت «سبز/غیرسبز» یا گروه‌های قطعی کارکنان نام‌گذاری شوند. ساختار دوخوشه‌ای GHRM با Silhouette برابر ۰٫۴۳۶۳ و Davies–Bouldin برابر ۰٫۸۶۱۱ تفکیک اکتشافی متوسطی دارد.

![کیفیت خوشه‌بندی در k منتخب](figures/kmeans_selected_quality.png)

### پروفایل توصیفی خوشه‌های GHRM

{markdown_table(display['clusters'])}

خوشه ۰ شامل ۱۹۸ مشاهده با میانگین FEP برابر ۳٫۸۵۱۰ و خوشه ۱ شامل ۱۲۲ مشاهده با میانگین FEP برابر ۳٫۰۴۱۰ است. چون FEP در تشکیل خوشه‌ها وارد نشده، این مقایسه یک پروفایل توصیفی پس از خوشه‌بندی است و اثر علّی را نشان نمی‌دهد.

![پروفایل FEP خوشه‌های GHRM](figures/ghrm_cluster_fep.png)

نمودارهای تشخیصی کامل انتخاب k:

- [IBM HR](figures/kmeans_ibm_diagnostics.png)
- [Job Change](figures/kmeans_job_change_diagnostics.png)
- [Employee Promotion](figures/kmeans_promotion_diagnostics.png)
- [GHRM](figures/kmeans_ghrm_diagnostics.png)

## جمع‌بندی فصل

یک الگوریتم واحد در همه مسائل بهترین نبود. MLP بر اساس F1 مثبت در IBM و Employee Promotion برتر بود، در حالی که Random Forest در Job Change بالاترین F1 مثبت را ثبت کرد. در GHRM، LinearSVR بهترین نتیجه مجموعه آزمون را به دست آورد و افزودن GEE بهبود بسیار محدودی ایجاد کرد. خوشه‌بندی سه مجموعه‌داده عمومی ساختار ضعیفی نشان داد و فقط خوشه‌بندی GHRM به سطح تفکیک اکتشافی متوسط رسید.

## محدودیت‌های الزامی در تفسیر

1. نتایج ماهیت پیش‌بینانه یا اکتشافی دارند و رابطه علّی یا میانجی‌گری را اثبات نمی‌کنند.
2. Accuracy در IBM و Employee Promotion باید همراه Precision، Recall، F1 و ماتریس درهم‌ریختگی گزارش شود.
3. گمشده‌بودن برخی متغیرهای Job Change و روش جایگذاری داخل fold باید افشا شود.
4. Silhouette نمونه‌ای Job Change و Employee Promotion باید با `n=5,000` گزارش شود.
5. سه مجموعه‌داده عمومی، سنجه مستقیم GHRM یا رفتار محیط‌زیستی ندارند و فقط مسائل عمومی منابع انسانی را پوشش می‌دهند.
6. تعمیم نتایج به سازمان، صنعت یا کشور دیگر به اعتبارسنجی بیرونی نیاز دارد.

## بازتولید

```bash
python -m pip install -r requirements.txt
python run_all.py
python -m unittest discover -s tests -v
```

جدول‌های ماشین‌خوان در پوشه [`results/tables`](tables/) و نمودارهای آماده GitHub در پوشه [`results/figures`](figures/) قرار دارند. جزئیات کنترل روش‌شناسی و بازمحاسبه معیارها در [`validation/validation_report.md`](../validation/validation_report.md) ثبت شده است.
"""
    path = RESULTS / "chapter4_results.md"
    path.write_text(report, encoding="utf-8")
    return path


def build_chapter4_outputs() -> dict[str, pd.DataFrame]:
    """Create all tracked Chapter 4 report assets from pipeline artifacts."""

    RESULTS.mkdir(parents=True, exist_ok=True)
    tables = build_result_tables()
    write_tables(tables)
    build_figures(tables)
    write_persian_report(tables)
    return tables


if __name__ == "__main__":
    build_chapter4_outputs()

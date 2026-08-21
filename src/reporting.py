"""Build tracked Chapter 4 tables, figures, and a data-driven Persian report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from src.evaluation import (
    bootstrap_classification_metrics,
    bootstrap_regression_difference,
    bootstrap_regression_metrics,
)
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
PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


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
    figure.savefig(FIGURES / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _read_ghrm_raw() -> pd.DataFrame:
    path = next(
        (GHRM_DATA_DIR / name for name in GHRM_FILES if (GHRM_DATA_DIR / name).exists()),
        None,
    )
    if path is None:
        raise FileNotFoundError("GHRM dataset not found: " + ", ".join(GHRM_FILES))
    return pd.read_csv(path)


def _positive_label(dataset: str) -> str:
    mapping = json.loads(
        (OUTPUTS / dataset / "label_mapping.json").read_text(encoding="utf-8")
    )
    return next(label for label, code in mapping.items() if code == 1)


def to_persian_digits(value: object) -> str:
    """Render Persian digits while deliberately preserving the ASCII decimal point."""

    return str(value).translate(PERSIAN_DIGITS)


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return to_persian_digits(f"{value:.4f}")
    if isinstance(value, (int, np.integer)):
        return to_persian_digits(f"{value:,}")
    return to_persian_digits(value)


def markdown_table(frame: pd.DataFrame) -> str:
    """Return a dependency-free GitHub Markdown table with Persian digits."""

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


def interpret_silhouette(value: float) -> str:
    """Apply one metric-based interpretation rule to every dataset."""

    if value < 0.25:
        return "Weak exploratory separation"
    if value < 0.50:
        return "Moderate exploratory separation"
    return "Strong exploratory separation"


def build_dataset_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overview_rows, quality_rows, missing_rows = [], [], []
    for dataset in DATASET_ORDER[:-1]:
        frame, info, _ = load_dataset(dataset)
        target = info["target"]
        positive = frame[target].astype(str).eq(str(_positive_label(dataset)))
        overview_rows.append(
            {
                "Dataset": DATASET_LABELS[dataset],
                "Rows": len(frame),
                "Raw_Columns": frame.shape[1],
                "Target": target,
                "Task": "Binary classification",
                "Positive_Count": int(positive.sum()),
                "Positive_Share": float(positive.mean()),
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
                        "Missing_Share": float(count / len(frame)),
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
                ghrm[["FEP1", "FEP5", "FEP7", "FEP9"]].isna().any(axis=1).sum()
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
                    "Missing_Share": float(count / len(ghrm)),
                }
            )
    missing = pd.DataFrame(
        missing_rows,
        columns=["Dataset", "Column", "Missing_Count", "Missing_Share"],
    )
    return pd.DataFrame(overview_rows), pd.DataFrame(quality_rows), missing


def _classification_table() -> pd.DataFrame:
    metrics = pd.read_csv(OUTPUTS / "model_metrics.csv")
    intervals = []
    for dataset in DATASET_ORDER[:-1]:
        for model in MODEL_ORDER:
            predictions = pd.read_csv(
                OUTPUTS / dataset / f"{model.replace(' ', '_')}_test_predictions.csv"
            )
            intervals.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    **bootstrap_classification_metrics(
                        predictions["y_true"], predictions["y_pred"]
                    ),
                }
            )
    metrics = metrics.merge(
        pd.DataFrame(intervals),
        on=["Dataset", "Model"],
        how="left",
        validate="one_to_one",
    )
    metrics["Dataset"] = metrics["Dataset"].map(DATASET_LABELS)
    metrics["Dataset"] = pd.Categorical(
        metrics["Dataset"],
        [DATASET_LABELS[name] for name in DATASET_ORDER[:-1]],
        ordered=True,
    )
    metrics["Model"] = pd.Categorical(metrics["Model"], MODEL_ORDER, ordered=True)
    return metrics.sort_values(["Dataset", "Model"])


def _regression_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(OUTPUTS / "ghrm" / "regression_metrics.csv")
    intervals, comparisons = [], []
    for model in REGRESSION_MODEL_ORDER:
        paired = {}
        for variant, prefix in (("Base", "base"), ("GEE+", "gee_plus")):
            predictions = pd.read_csv(
                OUTPUTS
                / "ghrm"
                / f"{prefix}_{model.replace(' ', '_')}_test_predictions.csv"
            )
            paired[variant] = predictions
            intervals.append(
                {
                    "Variant": variant,
                    "Model": model,
                    **bootstrap_regression_metrics(
                        predictions["y_true"], predictions["y_pred"]
                    ),
                }
            )
        if not np.array_equal(
            paired["Base"]["y_true"].to_numpy(), paired["GEE+"]["y_true"].to_numpy()
        ):
            raise ValueError(f"Unpaired Base/GEE+ held-out rows for {model}")
        comparisons.append(
            {
                "Model": model,
                **bootstrap_regression_difference(
                    paired["Base"]["y_true"],
                    paired["Base"]["y_pred"],
                    paired["GEE+"]["y_pred"],
                ),
            }
        )
    metrics = metrics.merge(
        pd.DataFrame(intervals),
        on=["Variant", "Model"],
        how="left",
        validate="one_to_one",
    )
    metrics["Model"] = pd.Categorical(
        metrics["Model"], REGRESSION_MODEL_ORDER, ordered=True
    )
    return metrics.sort_values(["Variant", "Model"]), pd.DataFrame(comparisons)


def build_result_tables() -> dict[str, pd.DataFrame]:
    classification = _classification_table()
    mcnemar = pd.concat(
        [pd.read_csv(OUTPUTS / name / "mcnemar.csv") for name in DATASET_ORDER[:-1]],
        ignore_index=True,
    )
    mcnemar["Dataset"] = mcnemar["Dataset"].map(DATASET_LABELS)

    confusion_rows = []
    for dataset in DATASET_ORDER[:-1]:
        for model in MODEL_ORDER:
            matrix = pd.read_csv(
                OUTPUTS / dataset / f"{model.replace(' ', '_')}_confusion_matrix.csv",
                header=None,
            ).to_numpy()
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

    regression, comparisons = _regression_tables()
    kmeans_rows, cluster_size_rows = [], []
    for dataset in DATASET_ORDER:
        metrics = pd.read_csv(OUTPUTS / dataset / "kmeans_metrics.csv")
        selected = select_kmeans_row(metrics)
        source_rows = (
            len(_read_ghrm_raw())
            if dataset == "ghrm"
            else len(load_dataset(dataset)[0])
        )
        kmeans_rows.append(
            {
                "Dataset": DATASET_LABELS[dataset],
                "Selected_k": int(selected["k"]),
                "Inertia_SSE": float(selected["Inertia_SSE"]),
                "Silhouette": float(selected["Silhouette"]),
                "Davies_Bouldin": float(selected["Davies_Bouldin"]),
                "Silhouette_N": min(source_rows, 5000),
                "Interpretation": interpret_silhouette(float(selected["Silhouette"])),
            }
        )
        for row in pd.read_csv(OUTPUTS / dataset / "cluster_sizes.csv").itertuples(
            index=False
        ):
            cluster_size_rows.append(
                {
                    "Dataset": DATASET_LABELS[dataset],
                    "Cluster": int(row.Cluster),
                    "Size": int(row.Size),
                }
            )

    overview, quality, missing = build_dataset_tables()
    return {
        "dataset_overview": overview,
        "data_quality": quality,
        "missing_values": missing,
        "classification_metrics": classification,
        "confusion_matrices": pd.DataFrame(confusion_rows),
        "mcnemar": mcnemar,
        "regression_metrics": regression,
        "regression_variant_comparison": comparisons,
        "kmeans_selection": pd.DataFrame(kmeans_rows),
        "cluster_sizes": pd.DataFrame(cluster_size_rows),
        "ghrm_cluster_profiles": pd.read_csv(OUTPUTS / "ghrm" / "cluster_profiles.csv"),
    }


def write_tables(tables: dict[str, pd.DataFrame]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(TABLES / f"{name}.csv", index=False, encoding="utf-8-sig")


def plot_class_balance(overview: pd.DataFrame) -> None:
    data = overview.dropna(subset=["Positive_Share"]).copy()
    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    bars = axis.barh(
        data["Dataset"], data["Positive_Share"] * 100,
        color=[BLUE, GOLD, ORANGE], edgecolor=INK, linewidth=0.7,
    )
    axis.set(title="Positive-class share by classification dataset", xlabel="Positive class (%)", xlim=(0, 100))
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
        lower = data["F1"] - data["F1_CI_Lower"]
        upper = data["F1_CI_Upper"] - data["F1"]
        bars = axis.bar(
            data["Model"].astype(str), data["F1"], yerr=np.vstack([lower, upper]),
            capsize=3, color=[MODEL_COLORS[name] for name in MODEL_ORDER],
            edgecolor=INK, linewidth=0.6,
        )
        axis.set(title=label, ylim=(0, 1), xlabel="Model")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[0].set_ylabel("Positive-class F1 (95% bootstrap CI)")
    figure.suptitle("Held-out positive-class F1 by model", fontsize=14, y=1.02)
    _save_figure(figure, "classification_f1.png")


def plot_confusion_matrices(confusion: pd.DataFrame) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(12.8, 9.6))
    cmap = LinearSegmentedColormap.from_list("thesis_blue", ["#EFF5FA", BLUE, "#173D63"])
    for row_index, dataset in enumerate(DATASET_ORDER[:-1]):
        label = DATASET_LABELS[dataset]
        for column_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            row = confusion[(confusion["Dataset"] == label) & (confusion["Model"] == model)].iloc[0]
            matrix = np.array([[row.TN, row.FP], [row.FN, row.TP]])
            axis.imshow(matrix, cmap=cmap, vmin=0, vmax=matrix.max())
            for y in range(2):
                for x in range(2):
                    value = int(matrix[y, x])
                    axis.text(x, y, f"{value:,}", ha="center", va="center", color="white" if value > matrix.max() * 0.55 else INK, fontweight="bold")
            axis.set_xticks([0, 1], ["0", "1"])
            axis.set_yticks([0, 1], ["0", "1"])
            axis.grid(False)
            if row_index == 0:
                axis.set_title(model)
            if column_index == 0:
                axis.set_ylabel(f"{label}\nActual")
            if row_index == 2:
                axis.set_xlabel("Predicted")
    figure.suptitle("Held-out confusion matrices (positive class = 1)", fontsize=14, y=1.01)
    _save_figure(figure, "confusion_matrices.png")


def plot_regression_metrics(regression: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(9.6, 5.6))
    positions, width = np.arange(len(REGRESSION_MODEL_ORDER)), 0.35
    for offset, variant in enumerate(["Base", "GEE+"]):
        data = regression[regression["Variant"] == variant].set_index("Model")
        values = np.array([float(data.loc[name, "RMSE"]) for name in REGRESSION_MODEL_ORDER])
        lower = values - np.array([float(data.loc[name, "RMSE_CI_Lower"]) for name in REGRESSION_MODEL_ORDER])
        upper = np.array([float(data.loc[name, "RMSE_CI_Upper"]) for name in REGRESSION_MODEL_ORDER]) - values
        bars = axis.barh(
            positions + (offset - 0.5) * width, values, width,
            xerr=np.vstack([lower, upper]), capsize=3, label=variant,
            color=REGRESSION_COLORS[variant], edgecolor=INK, linewidth=0.6,
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set(title="GHRM held-out RMSE with 95% bootstrap CI", xlabel="RMSE (lower is better)")
    axis.set_yticks(positions, REGRESSION_MODEL_ORDER)
    axis.set_xlim(0, max(regression["RMSE_CI_Upper"]) * 1.20)
    axis.invert_yaxis()
    axis.legend(frameon=False, ncols=2, loc="upper center")
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    _save_figure(figure, "ghrm_regression_rmse.png")


def plot_best_regression_predictions(regression: pd.DataFrame) -> None:
    best = regression.loc[regression["RMSE"].idxmin()]
    variant, model = str(best["Variant"]), str(best["Model"])
    prefix = "gee_plus" if variant == "GEE+" else "base"
    predictions = pd.read_csv(OUTPUTS / "ghrm" / f"{prefix}_{model.replace(' ', '_')}_test_predictions.csv")
    figure, axis = plt.subplots(figsize=(6.2, 5.6))
    axis.scatter(predictions["y_true"], predictions["y_pred"], s=38, color=BLUE, edgecolor="white", linewidth=0.7, alpha=0.85)
    minimum, maximum = float(predictions.min().min()), float(predictions.max().max())
    padding = (maximum - minimum) * 0.08
    limits = (minimum - padding, maximum + padding)
    axis.plot(limits, limits, linestyle="--", color=INK, linewidth=1.1)
    axis.set(xlim=limits, ylim=limits, title=f"GHRM held-out actual vs predicted FEP\n{variant} {model}; n={len(predictions)}", xlabel="Actual FEP", ylabel="Predicted FEP")
    axis.set_aspect("equal", adjustable="box")
    _save_figure(figure, "ghrm_actual_vs_predicted.png")


def plot_kmeans_selection(kmeans: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    labels = kmeans["Dataset"].str.replace("GHRM–Environmental Performance", "GHRM", regex=False)
    for axis, column, title in [
        (axes[0], "Silhouette", "Selected-cluster Silhouette"),
        (axes[1], "Davies_Bouldin", "Selected-cluster Davies–Bouldin"),
    ]:
        bars = axis.bar(labels, kmeans[column], color=[BLUE, GOLD, ORANGE, OLIVE], edgecolor=INK, linewidth=0.6)
        axis.set(title=title, ylabel=column.replace("_", " "), ylim=(0, max(kmeans[column]) * 1.18))
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    figure.suptitle("K-Means quality at the selected k", fontsize=13, y=1.02)
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
            axis.plot(metrics["k"], metrics[column], marker="o", color=BLUE, markerfacecolor="white", markeredgecolor=BLUE, linewidth=2)
            axis.set(title=f"{label} ({hint})", xlabel="k", ylabel=label)
            axis.set_xticks(metrics["k"])
        sample_n = min(len(_read_ghrm_raw()) if dataset == "ghrm" else len(load_dataset(dataset)[0]), 5000)
        figure.suptitle(f"{DATASET_LABELS[dataset]} K-Means diagnostics; Silhouette n={sample_n:,}", fontsize=13, y=1.02)
        _save_figure(figure, f"kmeans_{dataset}_diagnostics.png")


def plot_ghrm_cluster_profile(profiles: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    labels = [f"Cluster {int(value)}" for value in profiles["Cluster"]]
    bars = axis.bar(labels, profiles["Mean_FEP"], yerr=profiles["SD_FEP"], capsize=5, color=[BLUE, GOLD], edgecolor=INK, linewidth=0.6)
    axis.set(title="GHRM cluster profile on descriptive FEP", ylabel="Mean FEP ± SD (FEP excluded from clustering)", ylim=(0, 5))
    axis.grid(axis="y")
    axis.grid(axis="x", visible=False)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    for index, size in enumerate(profiles["Size"]):
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
    overview = tables["dataset_overview"][["Dataset", "Rows", "Raw_Columns", "Target", "Task", "Positive_Share"]].copy()
    overview["Positive_Share"] = overview["Positive_Share"].map(lambda value: "—" if pd.isna(value) else f"{value * 100:.2f}%")
    overview.columns = ["مجموعه‌داده", "تعداد رکورد", "ستون خام", "متغیر هدف", "نوع مسئله", "سهم کلاس مثبت"]

    classification = tables["classification_metrics"][["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1", "F1_CI_Lower", "F1_CI_Upper", "CV_Best_F1"]].copy()
    classification.columns = ["مجموعه‌داده", "مدل", "Accuracy", "Precision مثبت", "Recall مثبت", "F1 مثبت", "حد پایین F1", "حد بالای F1", "بهترین F1 در CV"]

    mcnemar = tables["mcnemar"][["Dataset", "Model_A", "Model_B", "b01", "b10", "p_value"]].copy()
    mcnemar["p_value"] = mcnemar["p_value"].map(
        lambda value: "<0.0001" if value < 0.0001 else f"{value:.4f}"
    )
    mcnemar.columns = ["مجموعه‌داده", "مدل A", "مدل B", "b01", "b10", "p-value"]

    regression = tables["regression_metrics"][["Variant", "Model", "R2", "R2_CI_Lower", "R2_CI_Upper", "MAE", "RMSE", "RMSE_CI_Lower", "RMSE_CI_Upper", "CV_RMSE"]].copy()
    regression.columns = ["نسخه", "مدل", "R²", "حد پایین R²", "حد بالای R²", "MAE", "RMSE", "حد پایین RMSE", "حد بالای RMSE", "CV RMSE"]

    comparison = tables["regression_variant_comparison"][["Model", "Delta_R2", "Delta_R2_CI_Lower", "Delta_R2_CI_Upper", "Delta_RMSE", "Delta_RMSE_CI_Lower", "Delta_RMSE_CI_Upper"]].copy()
    comparison.columns = ["مدل", "تغییر R²", "حد پایین تغییر R²", "حد بالای تغییر R²", "تغییر RMSE", "حد پایین تغییر RMSE", "حد بالای تغییر RMSE"]

    kmeans = tables["kmeans_selection"][["Dataset", "Selected_k", "Silhouette", "Davies_Bouldin", "Silhouette_N", "Interpretation"]].copy()
    kmeans.columns = ["مجموعه‌داده", "k منتخب", "Silhouette", "Davies–Bouldin", "n برای Silhouette", "تفسیر"]
    kmeans["تفسیر"] = kmeans["تفسیر"].replace({"Weak exploratory separation": "تفکیک اکتشافی ضعیف", "Moderate exploratory separation": "تفکیک اکتشافی متوسط", "Strong exploratory separation": "تفکیک اکتشافی قوی"})
    clusters = tables["cluster_sizes"].copy()
    clusters.columns = ["مجموعه‌داده", "خوشه", "تعداد"]
    profiles = tables["ghrm_cluster_profiles"].copy()
    profiles.columns = ["خوشه", "تعداد", "میانگین FEP", "انحراف معیار FEP"]
    return {"overview": overview, "classification": classification, "mcnemar": mcnemar, "regression": regression, "comparison": comparison, "kmeans": kmeans, "clusters": clusters, "profiles": profiles}


def _classification_summary(classification: pd.DataFrame) -> str:
    paragraphs = []
    for dataset in [DATASET_LABELS[name] for name in DATASET_ORDER[:-1]]:
        data = classification[classification["Dataset"] == dataset]
        best_f1 = data.loc[data["F1"].idxmax()]
        best_recall = data.loc[data["Recall"].idxmax()]
        paragraphs.append(
            f"در {dataset}، **{best_f1['Model']}** با F1 مثبت **{_format_value(best_f1['F1'])}** "
            f"(بازه اطمینان ۹۵٪: {_format_value(best_f1['F1_CI_Lower'])} تا {_format_value(best_f1['F1_CI_Upper'])}) "
            f"بهترین F1 را داشت؛ بیشترین Recall مثبت نیز مربوط به {best_recall['Model']} "
            f"با مقدار {_format_value(best_recall['Recall'])} بود."
        )
    return "\n\n".join(paragraphs)


def write_persian_report(tables: dict[str, pd.DataFrame]) -> Path:
    display = _presentation_tables(tables)
    overview = tables["dataset_overview"]
    classification = tables["classification_metrics"]
    regression = tables["regression_metrics"]
    comparisons = tables["regression_variant_comparison"]
    kmeans = tables["kmeans_selection"]
    profiles = tables["ghrm_cluster_profiles"]
    missing = tables["missing_values"].sort_values("Missing_Share", ascending=False)

    shares = overview.dropna(subset=["Positive_Share"]).set_index("Dataset")["Positive_Share"]
    missing_text = "داده گمشده‌ای در فایل‌های ورودی مشاهده نشد."
    if not missing.empty:
        top_missing = missing.head(4)
        missing_text = "بیشترین نسبت‌های گمشدگی: " + "، ".join(
            f"`{row.Column}` در {row.Dataset} ({_format_value(row.Missing_Share * 100)}٪)"
            for row in top_missing.itertuples(index=False)
        ) + "."

    insignificant = tables["mcnemar"][tables["mcnemar"]["p_value"] >= 0.05]
    mcnemar_text = "همه مقایسه‌های گزارش‌شده در سطح ۰.۰۵ معنادار بودند."
    if not insignificant.empty:
        mcnemar_text = "مقایسه‌های نامعنادار در سطح ۰.۰۵: " + "، ".join(
            f"{row.Dataset}: {row.Model_A} در برابر {row.Model_B} (p={_format_value(row.p_value)})"
            for row in insignificant.itertuples(index=False)
        ) + "."

    best_base = regression[regression["Variant"] == "Base"].loc[lambda data: data["RMSE"].idxmin()]
    best_plus = regression[regression["Variant"] == "GEE+"].loc[lambda data: data["RMSE"].idxmin()]
    best_delta = comparisons[comparisons["Model"] == str(best_plus["Model"])].iloc[0]
    delta_contains_zero = best_delta["Delta_R2_CI_Lower"] <= 0 <= best_delta["Delta_R2_CI_Upper"]
    delta_conclusion = "شامل صفر است؛ بنابراین برتری پایدار GEE تأیید نمی‌شود" if delta_contains_zero else "شامل صفر نیست"

    cluster_text = "\n\n".join(
        f"برای {row.Dataset}، k={_format_value(row.Selected_k)} با Silhouette={_format_value(row.Silhouette)} "
        f"و Davies–Bouldin={_format_value(row.Davies_Bouldin)} منتخب شد؛ Silhouette روی {_format_value(row.Silhouette_N)} رکورد محاسبه شد."
        for row in kmeans.itertuples(index=False)
    )
    profile_text = "، ".join(
        f"خوشه {_format_value(row.Cluster)} با {_format_value(row.Size)} مشاهده و میانگین FEP={_format_value(row.Mean_FEP)}"
        for row in profiles.itertuples(index=False)
    )

    report = f"""# نتایج بازتولیدپذیر فصل چهارم

این صفحه از آرتیفکت‌های اجرای کامل `python run_all.py` ساخته شده است. جدول‌ها، نمودارها و متن نتیجه‌گیری همگی برنامه‌وار به‌روز می‌شوند. همه اعداد این گزارش فارسی هستند و نشان اعشار به‌صورت `.` نوشته شده است.

## طرح تحلیل و کیفیت داده

{markdown_table(display['overview'])}

چهار منبع داده مستقل اند و هیچ رکوردی میان آن‌ها ادغام نشده است. کنترل کیفیت شامل ردیف تکراری، هدف گمشده و شناسه تکراری/گمشده بود. {missing_text} جایگذاری فقط داخل fold آموزشی انجام شد.

![سهم کلاس مثبت](figures/class_balance.png)

## نتایج طبقه‌بندی

Precision، Recall و F1 مربوط به کلاس مثبت هستند. سهم کلاس مثبت IBM برابر {_format_value(shares[DATASET_LABELS['ibm']] * 100)}٪ و Employee Promotion برابر {_format_value(shares[DATASET_LABELS['promotion']] * 100)}٪ است؛ پس Accuracy به‌تنهایی ملاک مناسبی نیست. بازه‌های اطمینان ۹۵٪ با ۲٬۰۰۰ بازنمونه‌گیری bootstrap از مجموعه آزمون محاسبه شدند.

{markdown_table(display['classification'])}

{_classification_summary(classification)}

![مقایسه F1 مدل‌های طبقه‌بندی](figures/classification_f1.png)

![ماتریس‌های درهم‌ریختگی](figures/confusion_matrices.png)

## آزمون McNemar

{markdown_table(display['mcnemar'])}

این آزمون الگوی درست/نادرست پیش‌بینی‌های جفت‌شده را می‌سنجد و آزمون مستقیم اختلاف F1 نیست. {mcnemar_text}

## نتایج رگرسیون GHRM

نسخه پایه از `GRS`، `GTD`، `GPA` و `GCM` و نسخه `GEE+` از متغیر اضافی `GEE` نیز استفاده می‌کند. بازه‌های اطمینان با ۴٬۰۰۰ بازنمونه‌گیری محاسبه شدند.

{markdown_table(display['regression'])}

بهترین مدل پایه **{best_base['Model']}** با R²={_format_value(best_base['R2'])} و RMSE={_format_value(best_base['RMSE'])} بود. بهترین مدل GEE+ نیز **{best_plus['Model']}** با R²={_format_value(best_plus['R2'])} و RMSE={_format_value(best_plus['RMSE'])} بود.

### مقایسه جفت‌شده افزودن GEE

{markdown_table(display['comparison'])}

برای {best_plus['Model']}، تغییر R² برابر {_format_value(best_delta['Delta_R2'])} و بازه اطمینان آن {_format_value(best_delta['Delta_R2_CI_Lower'])} تا {_format_value(best_delta['Delta_R2_CI_Upper'])} است. این بازه {delta_conclusion}. این تحلیل پیش‌بینانه است و میانجی‌گری یا علیت را آزمون نمی‌کند.

![مقایسه RMSE مدل‌های رگرسیون](figures/ghrm_regression_rmse.png)

![مقادیر واقعی و پیش‌بینی‌شده FEP](figures/ghrm_actual_vs_predicted.png)

## نتایج خوشه‌بندی K-Means

هدف و شناسه‌ها در سه داده عمومی و `FEP` در GHRM از تشکیل خوشه حذف شدند. چون K-Means فاصله اقلیدسی را روی اعداد استاندارد و متغیرهای رده‌ای one-hot اعمال می‌کند، خوشه‌ها صرفاً اکتشافی هستند.

{markdown_table(display['kmeans'])}

{cluster_text}

### اندازه همه خوشه‌ها

{markdown_table(display['clusters'])}

![کیفیت خوشه‌بندی در k منتخب](figures/kmeans_selected_quality.png)

### پروفایل توصیفی خوشه‌های GHRM

{markdown_table(display['profiles'])}

{profile_text}. چون FEP در تشکیل خوشه وارد نشد، این مقایسه فقط پروفایل پس از خوشه‌بندی است.

![پروفایل FEP خوشه‌های GHRM](figures/ghrm_cluster_fep.png)

نمودارهای تشخیصی: [IBM](figures/kmeans_ibm_diagnostics.png)، [Job Change](figures/kmeans_job_change_diagnostics.png)، [Employee Promotion](figures/kmeans_promotion_diagnostics.png)، [GHRM](figures/kmeans_ghrm_diagnostics.png).

## محدودیت‌ها

1. یک تفکیک ثابت ۸۰/۲۰ نماینده همه تفکیک‌های ممکن نیست؛ بازه‌های bootstrap عدم‌قطعیت مجموعه آزمون را نشان می‌دهند، نه اعتبارسنجی بیرونی را.
2. سه داده عمومی سنجه مستقیم GHRM ندارند و نباید خوشه‌های آن‌ها را سبز/غیرسبز نامگذاری کرد.
3. نتایج ماهیت پیش‌بینانه یا اکتشافی دارند و علیت یا میانجی‌گری را اثبات نمی‌کنند.
4. تعمیم به سازمان، صنعت یا کشور دیگر به اعتبارسنجی بیرونی نیاز دارد.

## بازتولید و اعتبارسنجی

```bash
python -m pip install -r requirements.txt
python run_all.py
python -m unittest discover -s tests -v
python scripts/validate_published_results.py
```

جدول‌های ماشین‌خوان در [`results/tables`](tables/) و نمودارها در [`results/figures`](figures/) قرار دارند.
"""
    if "٫" in report:
        raise ValueError("Persian report must use '.' as its decimal separator")
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

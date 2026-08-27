"""Build tables, figures, and the Persian execution report from live outputs."""

from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from src.config import (
    CLASSIFICATION_MODELS,
    DATASETS,
    REGRESSION_MODELS,
    RESULTS_DIR,
)
from src.explainability import slug


TABLES = RESULTS_DIR / "tables"
FIGURES = RESULTS_DIR / "figures"
PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
COLORS = ["#2D5F8B", "#BE8A2D", "#C65D3B", "#687B45"]
MODEL_COLORS = dict(zip(CLASSIFICATION_MODELS, COLORS))


def to_persian_digits(value):
    return str(value).translate(PERSIAN_DIGITS)


def _format(value):
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return to_persian_digits(f"{value:.4f}")
    if isinstance(value, (int, np.integer)):
        return to_persian_digits(f"{value:,}")
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        return f"`{value}`"
    return to_persian_digits(value)


def markdown_table(frame):
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(_format(value).replace("|", "\\|") for value in row)
            + " |"
        )
    return "\n".join(lines)


def select_kmeans_row(metrics):
    return metrics.loc[metrics["Silhouette"].idxmax()]


def interpret_silhouette(value):
    if value < 0.25:
        return "تفکیک ضعیف و صرفاً اکتشافی"
    if value < 0.50:
        return "تفکیک متوسط و اکتشافی"
    return "تفکیک نسبتاً روشن در همین داده"


def _configure_plotting():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _save(figure, filename):
    figure.tight_layout()
    figure.savefig(FIGURES / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


def aggregate_tables(classification, mcnemar, regression, comparisons):
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(RESULTS_DIR / "data" / "dataset_inventory.csv")
    inventory.to_csv(TABLES / "dataset_inventory.csv", index=False, encoding="utf-8-sig")
    classification.to_csv(
        TABLES / "classification_metrics.csv", index=False, encoding="utf-8-sig"
    )
    mcnemar.to_csv(TABLES / "mcnemar.csv", index=False, encoding="utf-8-sig")
    regression.to_csv(
        TABLES / "regression_metrics.csv", index=False, encoding="utf-8-sig"
    )
    comparisons.to_csv(
        TABLES / "base_vs_gee_plus.csv", index=False, encoding="utf-8-sig"
    )
    pd.read_csv(
        RESULTS_DIR / "regression" / "regression_validation_aggregates.csv"
    ).to_csv(
        TABLES / "regression_validation_aggregates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    confusion_rows = []
    importance_rows = []
    complexity_rows = []
    for dataset in ("ibm", "job_change", "promotion"):
        for model in CLASSIFICATION_MODELS:
            output = RESULTS_DIR / "classification" / dataset / slug(model)
            matrix = pd.read_csv(output / "confusion_matrix.csv", index_col=0).to_numpy()
            confusion_rows.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "TN": int(matrix[0, 0]),
                    "FP": int(matrix[0, 1]),
                    "FN": int(matrix[1, 0]),
                    "TP": int(matrix[1, 1]),
                }
            )
            importance = pd.read_csv(output / "permutation_importance.csv").head(10)
            importance_rows.extend(importance.to_dict(orient="records"))
            diagnostics = json.loads(
                (output / "model_diagnostics.json").read_text(encoding="utf-8")
            )
            complexity_rows.append(
                {
                    "Dataset": dataset,
                    "Variant": "classification",
                    "Model": model,
                    "Depth": diagnostics.get("depth"),
                    "Leaves": diagnostics.get("leaves"),
                    "Number_of_Trees": diagnostics.get("number_of_trees"),
                    "Mean_Forest_Depth": diagnostics.get("depth_mean"),
                    "Iterations": diagnostics.get("iterations"),
                    "Architecture": json.dumps(diagnostics.get("architecture")),
                }
            )

    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(
        TABLES / "confusion_matrices.csv", index=False, encoding="utf-8-sig"
    )
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(
        TABLES / "classification_feature_importance_top10.csv",
        index=False,
        encoding="utf-8-sig",
    )

    regression_importance_rows = []
    for variant in ("Base", "GEE+"):
        for model in REGRESSION_MODELS:
            output = RESULTS_DIR / "regression" / slug(variant) / slug(model)
            frame = pd.read_csv(output / "permutation_importance.csv").head(10)
            frame.insert(0, "Variant", variant)
            regression_importance_rows.extend(frame.to_dict(orient="records"))
            diagnostics = json.loads(
                (output / "model_diagnostics.json").read_text(encoding="utf-8")
            )
            complexity_rows.append(
                {
                    "Dataset": "ghrm",
                    "Variant": variant,
                    "Model": model,
                    "Depth": diagnostics.get("depth"),
                    "Leaves": diagnostics.get("leaves"),
                    "Number_of_Trees": diagnostics.get("number_of_trees"),
                    "Mean_Forest_Depth": diagnostics.get("depth_mean"),
                    "Iterations": diagnostics.get("iterations"),
                    "Architecture": json.dumps(diagnostics.get("architecture")),
                }
            )
    regression_importance = pd.DataFrame(regression_importance_rows)
    regression_importance.to_csv(
        TABLES / "regression_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(complexity_rows).to_csv(
        TABLES / "model_complexity.csv", index=False, encoding="utf-8-sig"
    )

    kmeans_rows = []
    cluster_sizes = []
    cluster_targets = []
    cluster_patterns = []
    for dataset in DATASETS:
        output = RESULTS_DIR / "clustering" / dataset
        metrics = pd.read_csv(output / "kmeans_metrics.csv")
        selected = select_kmeans_row(metrics)
        kmeans_rows.append(
            {
                **selected.to_dict(),
                "Interpretation": interpret_silhouette(float(selected["Silhouette"])),
            }
        )
        cluster_sizes.extend(
            pd.read_csv(output / "cluster_sizes.csv").to_dict(orient="records")
        )
        cluster_targets.extend(
            pd.read_csv(output / "cluster_target_summary.csv").to_dict(orient="records")
        )
        cluster_patterns.extend(
            pd.read_csv(output / "cluster_distinguishing_features.csv").to_dict(
                orient="records"
            )
        )
    kmeans = pd.DataFrame(kmeans_rows)
    sizes = pd.DataFrame(cluster_sizes)
    targets = pd.DataFrame(cluster_targets)
    patterns = pd.DataFrame(cluster_patterns)
    kmeans.to_csv(TABLES / "kmeans_selection.csv", index=False, encoding="utf-8-sig")
    sizes.to_csv(TABLES / "cluster_sizes.csv", index=False, encoding="utf-8-sig")
    targets[targets["Dataset"] == "ghrm"].to_csv(
        TABLES / "cluster_target_summary.csv", index=False, encoding="utf-8-sig"
    )
    patterns[patterns["Dataset"] == "ghrm"].to_csv(
        TABLES / "cluster_patterns.csv", index=False, encoding="utf-8-sig"
    )

    workflow = pd.DataFrame(
        [
            [1, "ثبت هویت داده", "هش SHA-256، تعداد ردیف و ستون"],
            [2, "کنترل کیفیت", "مقادیر گمشده، تکراری و کامل بودن هدف"],
            [3, "پیش پردازش", "جایگذاری فقط در داده آموزش، کدگذاری و مقیاس بندی"],
            [4, "تقسیم داده", "۸۰ درصد آموزش و ۲۰ درصد آزمون با بذر ۴۲"],
            [5, "تنظیم مدل", "اعتبارسنجی متقاطع و انتخاب پارامتر روی آموزش"],
            [6, "ارزیابی نهایی", "آزمون نگه داشته شده، ماتریس و بازه اطمینان"],
            [7, "تفسیر مدل", "اهمیت متغیرها، ساختار درخت و ضرایب"],
            [8, "خوشه بندی", "بررسی k از ۲ تا ۷ و پروفایل خوشه ها"],
        ],
        columns=["Step", "Action", "Saved_Evidence"],
    )
    workflow.to_csv(TABLES / "workflow_steps.csv", index=False, encoding="utf-8-sig")
    return {
        "inventory": inventory,
        "classification": classification,
        "mcnemar": mcnemar,
        "regression": regression,
        "comparisons": comparisons,
        "confusion": confusion,
        "classification_importance": importance,
        "regression_importance": regression_importance,
        "kmeans": kmeans,
        "sizes": sizes,
        "targets": targets,
        "patterns": patterns,
        "workflow": workflow,
    }


def build_figures(tables):
    _configure_plotting()
    classification = tables["classification"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for axis, dataset in zip(axes, ("ibm", "job_change", "promotion")):
        data = classification[classification["Dataset"] == dataset]
        lower = data["F1"] - data["F1_CI_Lower"]
        upper = data["F1_CI_Upper"] - data["F1"]
        bars = axis.bar(
            data["Model"],
            data["F1"],
            yerr=np.vstack([lower, upper]),
            color=[MODEL_COLORS[model] for model in data["Model"]],
            capsize=3,
        )
        axis.set_title(DATASETS[dataset]["display_name"])
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=25)
        axis.bar_label(bars, fmt="%.3f", padding=2)
    axes[0].set_ylabel("Positive-class F1 with 95% bootstrap CI")
    _save(figure, "classification_f1.png")

    confusion = tables["confusion"]
    cmap = LinearSegmentedColormap.from_list("blue", ["#F2F6FA", "#2D5F8B"])
    figure, axes = plt.subplots(3, 4, figsize=(12, 9))
    for row_index, dataset in enumerate(("ibm", "job_change", "promotion")):
        for column_index, model in enumerate(CLASSIFICATION_MODELS):
            axis = axes[row_index, column_index]
            row = confusion[
                (confusion["Dataset"] == dataset) & (confusion["Model"] == model)
            ].iloc[0]
            matrix = np.array([[row.TN, row.FP], [row.FN, row.TP]])
            axis.imshow(matrix, cmap=cmap)
            for y in range(2):
                for x in range(2):
                    axis.text(x, y, f"{matrix[y, x]:,}", ha="center", va="center")
            axis.set_xticks([0, 1], ["0", "1"])
            axis.set_yticks([0, 1], ["0", "1"])
            axis.grid(False)
            if row_index == 0:
                axis.set_title(model)
            if column_index == 0:
                axis.set_ylabel(DATASETS[dataset]["display_name"] + "\nActual")
            if row_index == 2:
                axis.set_xlabel("Predicted")
    _save(figure, "confusion_matrices.png")

    importance = tables["classification_importance"]
    for dataset in ("ibm", "job_change", "promotion"):
        figure, axes = plt.subplots(2, 2, figsize=(12, 9))
        for axis, model in zip(axes.ravel(), CLASSIFICATION_MODELS):
            data = importance[
                (importance["Dataset"] == dataset) & (importance["Model"] == model)
            ].head(10).sort_values("Importance_Mean")
            axis.barh(data["Feature"], data["Importance_Mean"], color=MODEL_COLORS[model])
            axis.set_title(model)
            axis.set_xlabel("Permutation importance (F1 decrease)")
        figure.suptitle(DATASETS[dataset]["display_name"] + " - feature importance")
        _save(figure, f"feature_importance_{dataset}.png")

    regression = tables["regression"]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(REGRESSION_MODELS))
    width = 0.35
    for index, variant in enumerate(("Base", "GEE+")):
        data = regression[regression["Variant"] == variant].set_index("Model")
        values = np.array([data.loc[model, "RMSE"] for model in REGRESSION_MODELS])
        bars = axis.barh(
            positions + (index - 0.5) * width,
            values,
            height=width,
            label=variant,
            color=COLORS[index],
        )
        axis.bar_label(bars, fmt="%.3f", padding=2)
    axis.set_yticks(positions, REGRESSION_MODELS)
    axis.invert_yaxis()
    axis.set_xlabel("Held-out RMSE (lower is better)")
    axis.legend()
    _save(figure, "ghrm_regression_rmse.png")

    regression_importance = tables["regression_importance"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, model in zip(axes.ravel(), REGRESSION_MODELS):
        data = regression_importance[
            (regression_importance["Variant"] == "GEE+")
            & (regression_importance["Model"] == f"GEE+ {model}")
        ].sort_values("Importance_Mean")
        axis.barh(data["Feature"], data["Importance_Mean"], color=COLORS[2])
        axis.set_title(model)
        axis.set_xlabel("Permutation importance")
    figure.suptitle("GHRM GEE+ - feature importance")
    _save(figure, "ghrm_feature_importance.png")

    for dataset in DATASETS:
        metrics = pd.read_csv(
            RESULTS_DIR / "clustering" / dataset / "kmeans_metrics.csv"
        )
        figure, axes = plt.subplots(1, 3, figsize=(13, 4))
        for axis, column, label in (
            (axes[0], "Inertia_SSE", "SSE"),
            (axes[1], "Silhouette", "Silhouette"),
            (axes[2], "Davies_Bouldin", "Davies-Bouldin"),
        ):
            axis.plot(metrics["k"], metrics[column], marker="o", color=COLORS[0])
            axis.set_xlabel("k")
            axis.set_ylabel(label)
        figure.suptitle(DATASETS[dataset]["display_name"] + " - K-Means diagnostics")
        _save(figure, f"kmeans_{dataset}.png")

    best = regression.loc[regression["RMSE"].idxmin()]
    output = RESULTS_DIR / "regression" / slug(best["Variant"]) / slug(best["Model"])
    predictions = pd.read_csv(output / "test_predictions.csv")
    figure, axis = plt.subplots(figsize=(5.8, 5.2))
    axis.scatter(predictions["y_true"], predictions["y_pred"], color=COLORS[0])
    low = min(predictions["y_true"].min(), predictions["y_pred"].min())
    high = max(predictions["y_true"].max(), predictions["y_pred"].max())
    axis.plot([low, high], [low, high], linestyle="--", color="black")
    axis.set_xlabel("Actual FEP")
    axis.set_ylabel("Predicted FEP")
    axis.set_title(f"{best['Variant']} {best['Model']} - held-out predictions")
    _save(figure, "ghrm_actual_vs_predicted.png")


def _classification_sections(classification):
    sections = []
    for dataset in ("ibm", "job_change", "promotion"):
        data = classification[classification["Dataset"] == dataset].copy()
        display = data[["Model", "Accuracy", "Precision", "Recall", "F1", "CV_Best_F1"]]
        best = data.loc[data["F1"].idxmax()]
        links = []
        for model in CLASSIFICATION_MODELS:
            path = f"classification/{dataset}/{slug(model)}"
            links.append(
                f"[{model}]({path}/model_diagnostics.json): پارامترها، پیش بینی آزمون، "
                f"ماتریس، اهمیت متغیرها و خروجی تنظیم مدل"
            )
        sections.append(
            f"### {DATASETS[dataset]['display_name']}\n\n"
            f"{markdown_table(display)}\n\n"
            f"در این تفکیک، بیشترین F1 مربوط به **{best['Model']}** با مقدار "
            f"**{_format(best['F1'])}** است. این نتیجه فقط درباره متغیر هدف "
            f"`{DATASETS[dataset]['target']}` است و سنجش مستقیم GHRM نیست.\n\n"
            + "\n\n".join(links)
            + f"\n\n![اهمیت متغیرها](figures/feature_importance_{dataset}.png)"
        )
    return "\n\n".join(sections)


def _cluster_sections(tables):
    sections = []
    for dataset in DATASETS:
        selected = tables["kmeans"][tables["kmeans"]["Dataset"] == dataset].iloc[0]
        sizes = tables["sizes"][tables["sizes"]["Dataset"] == dataset]
        targets = tables["targets"][tables["targets"]["Dataset"] == dataset]
        patterns = tables["patterns"][tables["patterns"]["Dataset"] == dataset]
        lines = []
        for cluster in sorted(sizes["Cluster"].unique()):
            size = int(sizes.loc[sizes["Cluster"] == cluster, "Size"].iloc[0])
            if dataset != "ghrm":
                lines.append(f"- خوشه {_format(cluster)} با {_format(size)} رکورد.")
                continue
            target_rows = targets[targets["Cluster"] == cluster]
            target_row = target_rows.iloc[0]
            target_text = f"میانگین FEP برابر {_format(target_row['Share_or_Mean'])}"
            top = patterns[patterns["Cluster"] == cluster].head(3)
            feature_text = "، ".join(
                f"{row.Feature} {('بالاتر' if row.Direction == 'higher' else 'پایین تر')}"
                for row in top.itertuples(index=False)
            )
            lines.append(
                f"- خوشه {_format(cluster)} با {_format(size)} رکورد: {feature_text}؛ {target_text}."
            )
        sections.append(
            f"### {DATASETS[dataset]['display_name']}\n\n"
            f"مقدار منتخب k برابر **{_format(selected['k'])}**، Silhouette برابر "
            f"**{_format(selected['Silhouette'])}** و تفسیر آن «{selected['Interpretation']}» است.\n\n"
            + "\n".join(lines)
            + (
                "\n\nپروفایل آماری تفصیلی این دیتاست عمومی منابع انسانی "
                "فقط در اجرای محلی نگه داری می شود."
                if dataset != "ghrm"
                else ""
            )
            + f"\n\n![نمودارهای انتخاب k](figures/kmeans_{dataset}.png)"
        )
    return "\n\n".join(sections)


def write_report(tables):
    inventory = tables["inventory"][
        ["Display_Name", "Role", "Rows", "Raw_Columns", "Target", "SHA256"]
    ].copy()
    classification = tables["classification"]
    regression = tables["regression"]
    comparisons = tables["comparisons"]
    best_base = regression[regression["Variant"] == "Base"].loc[
        regression[regression["Variant"] == "Base"]["RMSE"].idxmin()
    ]
    best_plus = regression[regression["Variant"] == "GEE+"].loc[
        regression[regression["Variant"] == "GEE+"]["RMSE"].idxmin()
    ]

    report = f"""# گزارش اجرای نهایی تحلیل منابع انسانی

این گزارش مستقیماً با اجرای `python run_all.py` ساخته شده است. هیچ عددی از فایل پایان نامه به نتایج کد تزریق نشده است. مسیر هر عدد از فایل داده، تقسیم آموزش و آزمون، پارامتر مدل و فایل پیش بینی قابل پیگیری است.

## حدود پژوهش و نقش چهار دیتاست

دیتاست GHRM تحلیل اصلی مدیریت منابع انسانی سبز است و متغیرهای GRS، GTD، GPA، GCM، GEE و FEP را مستقیماً دارد. سه دیتاست دیگر تحلیل های مستقل و تکمیلی منابع انسانی هستند و برای ترک خدمت، تغییر شغل و ارتقا استفاده شده اند. نتایج آن سه دیتاست شاهد مستقیم رفتار سبز یا پایداری شرکت نیستند و با دیتاست GHRM ادغام نشده اند. این پروژه هیچ داده ای از دیجی کالا ندارد.

{markdown_table(inventory)}

نام تمام ستون ها، نوع داده، تعداد مقادیر گمشده و تعداد مقادیر متمایز در [`data/column_dictionary.csv`](data/column_dictionary.csv) ثبت شده است. دامنه و فراوانی های حساس فقط در اجرای محلی قرار می گیرند.

## مراحل انجام کار

{markdown_table(tables['workflow'])}

پیش پردازش داخل Pipeline انجام شده است؛ بنابراین میانه، مد، کدگذاری و مقیاس بندی فقط از داده آموزش یاد گرفته می شوند. معیار تنظیم طبقه بندی F1 کلاس مثبت و معیار تنظیم رگرسیون RMSE است.

## نتایج مدل های طبقه بندی

![مقایسه F1](figures/classification_f1.png)

![ماتریس های درهم ریختگی](figures/confusion_matrices.png)

{_classification_sections(classification)}

## آزمون McNemar

آزمون McNemar پیش بینی های دو مدل روی همان رکوردهای آزمون را مقایسه می کند. ستون های b01 و b10 تعداد موارد اختلاف دو مدل هستند. روش دقیق دو جمله ای برای تعداد اختلاف کمتر از ۲۵ و تقریب کای دو با تصحیح پیوستگی برای نمونه های بزرگ تر استفاده شده است. جدول کامل هر شش مقایسه برای هر دیتاست در [`tables/mcnemar.csv`](tables/mcnemar.csv) قرار دارد.

## تحلیل اصلی GHRM و پیش بینی FEP

در مدل پایه، GRS، GTD، GPA و GCM پیش بین هستند. در تحلیل GEE+، متغیر GEE نیز اضافه شده است. علاوه بر آزمون ۲۰ درصدی، برای بررسی محدودیت حجم ۳۲۰ رکورد، عملکرد مدل نهایی در اعتبارسنجی متقاطع تکرارشونده ۵×۵ نیز ثبت شده است.

{markdown_table(regression[['Variant', 'Model', 'R2', 'RMSE', 'MAE', 'R2_CI_Lower', 'R2_CI_Upper', 'Repeated_CV_R2_Mean', 'Repeated_CV_R2_Std']])}

بهترین مدل پایه از نظر RMSE، **{best_base['Model']}** با R²={_format(best_base['R2'])} و RMSE={_format(best_base['RMSE'])} است. در حالت GEE+، بهترین مدل **{best_plus['Model']}** با R²={_format(best_plus['R2'])} و RMSE={_format(best_plus['RMSE'])} است. این نتایج پیش بینانه هستند و رابطه علّی یا میانجی گری را اثبات نمی کنند.

### کنترل همگرایی حل گرها

تعداد هشدارهای همگرایی پنهان نشده و برای مرحله تنظیم و ۲۵ Fold اعتبارسنجی تکرارشونده ثبت شده است.

{markdown_table(regression[['Variant', 'Model', 'Tuning_Convergence_Warnings', 'Repeated_CV_Convergence_Warnings']])}

### تغییر عملکرد پس از افزودن GEE

{markdown_table(comparisons)}

![مقایسه RMSE](figures/ghrm_regression_rmse.png)

![اهمیت متغیرهای GHRM](figures/ghrm_feature_importance.png)

![واقعی در برابر پیش بینی](figures/ghrm_actual_vs_predicted.png)

خروجی کامل هر مدل شامل پارامترها، تمام نتایج تنظیم، پیش بینی های آزمون، اهمیت متغیرها و برای مدل های درختی ساختار گره ها و شکل چند سطح اول است. در جنگل تصادفی صدها درخت وجود دارد؛ بنابراین مشخصات تمام درخت ها در `forest_tree_summary.csv` و ساختار کامل درخت اول به عنوان نمونه در `representative_tree_nodes.csv` ذخیره شده است.

## الگوهای پنهان حاصل از K-Means

K-Means برای هر دیتاست جداگانه اجرا شده است. متغیر هدف و شناسه ها در تشکیل خوشه وارد نشده اند. k از ۲ تا ۷ بررسی و مقدار دارای بیشترین Silhouette انتخاب شده است. وضعیت متغیر هدف فقط بعد از تشکیل خوشه برای توصیف الگو گزارش شده است.

{_cluster_sections(tables)}

## محدودیت های تفسیر

1. سه دیتاست عمومی، GHRM را مستقیماً اندازه گیری نمی کنند و نتایج آنها فقط در حوزه عمومی تحلیل منابع انسانی تفسیر می شود.
2. نمونه GHRM شامل ۳۲۰ رکورد است؛ بنابراین بازه های اطمینان و اعتبارسنجی تکرارشونده گزارش شده و تعمیم گسترده مجاز نیست.
3. خوشه بندی ماهیت اکتشافی دارد و نام گذاری خوشه ها نتیجه علّی نیست.
4. اهمیت متغیر نشان دهنده نقش پیش بینانه در مدل است، نه اثر علّی.

## فایل های قابل بررسی

- [`run_log.txt`](run_log.txt): گزارش زمانی اجرای پایتون
- [`run_manifest.json`](run_manifest.json): نسخه محیط، بذر و هش داده ها
- [`data/column_dictionary.csv`](data/column_dictionary.csv): فهرست تمام فیلدها
- [`tables`](tables): جدول های تجمیعی
- [`classification`](classification): جزئیات چهار الگوریتم طبقه بندی
- [`regression`](regression): جزئیات چهار الگوریتم رگرسیون در دو حالت
- [`clustering`](clustering): معیار و اندازه خوشه ها؛ پروفایل تفصیلی GHRM
"""
    if "٫" in report:
        raise ValueError("Use an ASCII full stop as decimal separator")
    (RESULTS_DIR / "analysis_report_fa.md").write_text(report, encoding="utf-8")


def write_results_readme():
    text = """# خروجی اجرای نهایی

تمام فایل های این پوشه با اجرای `python run_all.py` از چهار CSV محلی ساخته شده اند. متن پایان نامه ورودی عددی برنامه نیست.

از [گزارش فارسی](analysis_report_fa.md) شروع کنید. جدول های تجمیعی در `tables/`، جزئیات هر مدل در پوشه های دیتاست، زمان اجرا در `run_log.txt` و نسخه محیط و هش داده در `run_manifest.json` است.

فایل های خام داده در مخزن قرار ندارند. هویت دقیق SHA-256 آن ها در `data/dataset_inventory.csv` ثبت شده است.
"""
    (RESULTS_DIR / "README.md").write_text(text, encoding="utf-8")


def build_report(classification, mcnemar, regression, comparisons):
    tables = aggregate_tables(classification, mcnemar, regression, comparisons)
    build_figures(tables)
    write_report(tables)
    write_results_readme()
    return tables

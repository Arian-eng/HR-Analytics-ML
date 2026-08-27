"""Build tables, figures, and the English execution report from live outputs."""

from __future__ import annotations

import json
from pathlib import Path

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
COLORS = ["#2D5F8B", "#BE8A2D", "#C65D3B", "#687B45"]
MODEL_COLORS = dict(zip(CLASSIFICATION_MODELS, COLORS))


def _format(value):
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    ):
        return f"`{value}`"
    return str(value)


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
        return "weak separation; exploratory only"
    if value < 0.50:
        return "moderate exploratory separation"
    return "relatively clear separation in this dataset"


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
            [1, "Record data identity", "SHA-256, row count, and column count"],
            [2, "Check data quality", "Missing values, duplicates, and target completeness"],
            [3, "Preprocess", "Training-only imputation, encoding, and scaling"],
            [4, "Split data", "80% training and 20% test with random seed 42"],
            [5, "Tune models", "Cross-validation and parameter selection on training data"],
            [6, "Evaluate", "Held-out test metrics, matrices, and confidence intervals"],
            [7, "Explain models", "Feature importance, tree structure, and coefficients"],
            [8, "Cluster", "Evaluate k from 2 to 7 and profile the selected clusters"],
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
                f"[{model}]({path}/model_diagnostics.json): selected parameters, "
                f"held-out predictions, confusion matrix, feature importance, and tuning results"
            )
        sections.append(
            f"### {DATASETS[dataset]['display_name']}\n\n"
            f"{markdown_table(display)}\n\n"
            f"**{best['Model']}** has the highest positive-class F1 "
            f"(**{_format(best['F1'])}**) in this split. This result concerns only "
            f"the `{DATASETS[dataset]['target']}` target and is not a direct measure of GHRM.\n\n"
            + "\n\n".join(links)
            + f"\n\n![Feature importance](figures/feature_importance_{dataset}.png)"
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
                lines.append(f"- Cluster {_format(cluster)}: {_format(size)} records.")
                continue
            target_rows = targets[targets["Cluster"] == cluster]
            target_row = target_rows.iloc[0]
            target_text = f"mean FEP = {_format(target_row['Share_or_Mean'])}"
            top = patterns[patterns["Cluster"] == cluster].head(3)
            feature_text = ", ".join(
                f"{row.Feature} {('higher' if row.Direction == 'higher' else 'lower')}"
                for row in top.itertuples(index=False)
            )
            lines.append(
                f"- Cluster {_format(cluster)}: {_format(size)} records; {feature_text}; {target_text}."
            )
        sections.append(
            f"### {DATASETS[dataset]['display_name']}\n\n"
            f"The selected k is **{_format(selected['k'])}**, with a Silhouette score of "
            f"**{_format(selected['Silhouette'])}** ({selected['Interpretation']}).\n\n"
            + "\n".join(lines)
            + (
                "\n\nDetailed profiles for this public employee dataset are retained "
                "only in the local run."
                if dataset != "ghrm"
                else ""
            )
            + f"\n\n![K selection diagnostics](figures/kmeans_{dataset}.png)"
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

    report = f"""# Final HR analytics execution report

This report is generated directly by `python run_all.py`. Thesis text is never used as a numeric input. Every reported value can be traced to a source-file hash, a train/test split, selected model parameters, and a saved evaluation artifact.

## Study scope and the role of each dataset

The GHRM dataset is the main Green Human Resource Management analysis and directly contains GRS, GTD, GPA, GCM, GEE, and FEP measures. The other three datasets are independent supplementary HR analyses of attrition, job change, and promotion. They are not direct evidence of green behavior or firm sustainability and are never merged with GHRM. This project contains no Digikala data.

{markdown_table(inventory)}

Column names, data types, missing-value counts, and distinct-value counts are recorded in [`data/column_dictionary.csv`](data/column_dictionary.csv). Sensitive ranges and frequencies are retained only in the local run.

## Analysis workflow

{markdown_table(tables['workflow'])}

Preprocessing is fitted inside each model pipeline, so imputation, encoding, and scaling learn only from training data. Classification is tuned on positive-class F1; regression is tuned on RMSE.

## Classification results

![F1 comparison](figures/classification_f1.png)

![Confusion matrices](figures/confusion_matrices.png)

{_classification_sections(classification)}

## McNemar tests

McNemar's test compares two models on the same held-out records. Columns b01 and b10 count discordant outcomes. An exact binomial test is used when the discordant total is below 25; otherwise the continuity-corrected chi-square approximation is used. All six pairwise comparisons for every classification dataset are in [`tables/mcnemar.csv`](tables/mcnemar.csv).

## Main GHRM analysis and FEP prediction

The Base model uses GRS, GTD, GPA, and GCM. The GEE+ analysis adds GEE. Alongside the 20% holdout, each selected pipeline is evaluated with repeated 5x5 cross-validation to disclose the uncertainty associated with the 320-record sample.

{markdown_table(regression[['Variant', 'Model', 'R2', 'RMSE', 'MAE', 'R2_CI_Lower', 'R2_CI_Upper', 'Repeated_CV_R2_Mean', 'Repeated_CV_R2_Std']])}

The lowest-RMSE Base model is **{best_base['Model']}**, with R²={_format(best_base['R2'])} and RMSE={_format(best_base['RMSE'])}. The lowest-RMSE GEE+ model is **{best_plus['Model']}**, with R²={_format(best_plus['R2'])} and RMSE={_format(best_plus['RMSE'])}. These are predictive results and do not establish causality or mediation.

### Solver convergence checks

Convergence warnings are counted for tuning and all 25 repeated-CV folds rather than being suppressed.

{markdown_table(regression[['Variant', 'Model', 'Tuning_Convergence_Warnings', 'Repeated_CV_Convergence_Warnings']])}

### Change after adding GEE

{markdown_table(comparisons)}

![RMSE comparison](figures/ghrm_regression_rmse.png)

![GHRM feature importance](figures/ghrm_feature_importance.png)

![Actual versus predicted FEP](figures/ghrm_actual_vs_predicted.png)

Each model directory contains selected parameters, full tuning results, held-out predictions, feature importance, and—where applicable—tree-node exports and a readable top-level plot. Random forests contain hundreds of trees, so `forest_tree_summary.csv` records every tree's depth and leaf count while `representative_tree_nodes.csv` provides the complete structure of tree 0.

## Hidden patterns from K-Means

K-Means is run separately for every dataset. Targets and identifiers are excluded from cluster formation. Values of k from 2 through 7 are evaluated and the maximum-Silhouette solution is selected. Target outcomes are examined only after clustering for descriptive profiling.

{_cluster_sections(tables)}

## Interpretation limits

1. The three public HR datasets do not directly measure GHRM; their results are limited to general HR analytics.
2. GHRM contains 320 records. Confidence intervals and repeated CV disclose uncertainty but do not justify broad population-level generalization.
3. Clustering is exploratory; cluster labels are not causal conclusions.
4. Feature importance measures predictive contribution, not causal effect.

## Audit files

- [`run_log.txt`](run_log.txt): timestamped Python execution log
- [`run_manifest.json`](run_manifest.json): environment versions, seed, and data hashes
- [`data/column_dictionary.csv`](data/column_dictionary.csv): public field dictionary
- [`tables`](tables): consolidated result tables
- [`classification`](classification): details for all four classifiers
- [`regression`](regression): details for all four regressors in both variants
- [`clustering`](clustering): cluster metrics and sizes, plus detailed GHRM profiles
"""
    (RESULTS_DIR / "analysis_report.md").write_text(report, encoding="utf-8")


def write_results_readme():
    text = """# Final run outputs

Every file in this directory is generated from the four local CSV files by `python run_all.py`. Thesis text is not a numeric input to the pipeline.

Start with the [final execution report](analysis_report.md). Consolidated tables are in `tables/`, model details are grouped by dataset, execution timing is in `run_log.txt`, and environment versions plus data hashes are in `run_manifest.json`.

Raw data files are not committed. Their exact SHA-256 identities are recorded in `data/dataset_inventory.csv`.
"""
    (RESULTS_DIR / "README.md").write_text(text, encoding="utf-8")


def build_report(classification, mcnemar, regression, comparisons):
    tables = aggregate_tables(classification, mcnemar, regression, comparisons)
    build_figures(tables)
    write_report(tables)
    write_results_readme()
    return tables

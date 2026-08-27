"""Human-readable model diagnostics and tree exports."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text, plot_tree

from src.config import RANDOM_STATE


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def feature_names(pipeline):
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return np.asarray([str(name) for name in names])


def permutation_table(pipeline, X, y, scoring, dataset, model, repeats=5):
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring=scoring,
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "Dataset": dataset,
            "Model": model,
            "Feature": X.columns,
            "Importance_Mean": result.importances_mean,
            "Importance_Std": result.importances_std,
        }
    ).sort_values("Importance_Mean", ascending=False)


def tree_nodes(estimator, names, dataset, model, tree_index=None):
    tree = estimator.tree_
    rows = []
    for node in range(tree.node_count):
        feature_index = int(tree.feature[node])
        feature = "leaf" if feature_index < 0 else str(names[feature_index])
        values = np.asarray(tree.value[node]).reshape(-1)
        value_sum = float(values.sum())
        row = {
            "Dataset": dataset,
            "Model": model,
            "Tree_Index": tree_index,
            "Node": node,
            "Left_Child": int(tree.children_left[node]),
            "Right_Child": int(tree.children_right[node]),
            "Feature": feature,
            "Threshold": None if feature_index < 0 else float(tree.threshold[node]),
            "Impurity": float(tree.impurity[node]),
            "Samples": int(tree.n_node_samples[node]),
            "Weighted_Samples": float(tree.weighted_n_node_samples[node]),
            "Is_Leaf": feature_index < 0,
        }
        for index, value in enumerate(values):
            row[f"Value_{index}"] = float(value)
        if len(values) == 2 and value_sum:
            row["Positive_Probability"] = float(values[1] / value_sum)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_tree(estimator, names, output, title, class_names=None):
    figure, axis = plt.subplots(figsize=(20, 10))
    plot_tree(
        estimator,
        feature_names=names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=3,
        proportion=True,
        fontsize=7,
        ax=axis,
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_model_diagnostics(pipeline, output_dir, dataset, model, class_names=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimator = pipeline.named_steps["model"]
    names = feature_names(pipeline)
    summary = {
        "dataset": dataset,
        "model": model,
        "estimator": estimator.__class__.__name__,
        "parameters": estimator.get_params(),
    }

    if model in {"Decision Tree", "Decision Tree Regressor"}:
        summary.update(
            {
                "depth": int(estimator.get_depth()),
                "leaves": int(estimator.get_n_leaves()),
                "nodes": int(estimator.tree_.node_count),
            }
        )
        tree_nodes(estimator, names, dataset, model).to_csv(
            output / "tree_nodes.csv", index=False, encoding="utf-8-sig"
        )
        (output / "tree_rules.txt").write_text(
            export_text(estimator, feature_names=list(names), max_depth=20),
            encoding="utf-8",
        )
        _plot_tree(
            estimator,
            names,
            output / "tree_top_levels.png",
            f"{dataset} - {model} (first four levels)",
            class_names=class_names,
        )

    if model in {"Random Forest", "Random Forest Regressor"}:
        depths = np.asarray([tree.get_depth() for tree in estimator.estimators_])
        leaves = np.asarray([tree.get_n_leaves() for tree in estimator.estimators_])
        summary.update(
            {
                "number_of_trees": len(estimator.estimators_),
                "depth_min": int(depths.min()),
                "depth_mean": float(depths.mean()),
                "depth_max": int(depths.max()),
                "leaves_min": int(leaves.min()),
                "leaves_mean": float(leaves.mean()),
                "leaves_max": int(leaves.max()),
            }
        )
        pd.DataFrame(
            {
                "Tree_Index": np.arange(len(depths)),
                "Depth": depths,
                "Leaves": leaves,
            }
        ).to_csv(output / "forest_tree_summary.csv", index=False, encoding="utf-8-sig")
        representative = estimator.estimators_[0]
        tree_nodes(representative, names, dataset, model, tree_index=0).to_csv(
            output / "representative_tree_nodes.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _plot_tree(
            representative,
            names,
            output / "representative_tree_top_levels.png",
            f"{dataset} - first tree from {model}",
            class_names=class_names,
        )

    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_).reshape(-1)
        pd.DataFrame(
            {
                "Transformed_Feature": names,
                "Coefficient": coefficients,
                "Absolute_Coefficient": np.abs(coefficients),
            }
        ).sort_values("Absolute_Coefficient", ascending=False).to_csv(
            output / "coefficients.csv", index=False, encoding="utf-8-sig"
        )

    if hasattr(estimator, "n_iter_"):
        summary["iterations"] = np.asarray(estimator.n_iter_).tolist()
    if hasattr(estimator, "loss_"):
        summary["final_loss"] = float(estimator.loss_)
    if hasattr(estimator, "hidden_layer_sizes"):
        summary["architecture"] = [
            int(estimator.n_features_in_),
            *[int(value) for value in np.atleast_1d(estimator.hidden_layer_sizes)],
            int(estimator.n_outputs_),
        ]

    (output / "model_diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary

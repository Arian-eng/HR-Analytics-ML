import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.clustering import run_kmeans
from src.config import GHRM_ITEMS
from src.evaluation import (
    bootstrap_classification_metrics,
    bootstrap_regression_difference,
    classification_metrics,
    mcnemar_pair,
)
from src.explainability import tree_nodes
from src.preprocessing import load_dataset, make_preprocessor, split_xy
from src.reporting import interpret_silhouette, markdown_table, select_kmeans_row


class EvaluationTests(unittest.TestCase):
    def test_classification_metrics_use_positive_class(self):
        metrics = classification_metrics(
            np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
        )
        self.assertAlmostEqual(metrics["Accuracy"], 0.75)
        self.assertAlmostEqual(metrics["Precision"], 2 / 3)
        self.assertAlmostEqual(metrics["Recall"], 1.0)
        self.assertAlmostEqual(metrics["F1"], 0.8)

    def test_mcnemar_records_counts_and_method(self):
        result = mcnemar_pair(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 0]),
            np.array([0, 0, 0, 0]),
        )
        self.assertEqual(result["b01"], 1)
        self.assertEqual(result["b10"], 1)
        self.assertEqual(result["method"], "exact binomial")

    def test_bootstrap_intervals_are_deterministic(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        first = bootstrap_classification_metrics(
            y_true, y_pred, n_resamples=200, random_state=7
        )
        second = bootstrap_classification_metrics(
            y_true, y_pred, n_resamples=200, random_state=7
        )
        self.assertEqual(first, second)

    def test_paired_regression_difference_uses_same_rows(self):
        y_true = np.arange(1.0, 11.0)
        base = y_true + np.array([0.4, -0.3] * 5)
        plus = y_true + np.array([0.2, -0.1] * 5)
        difference = bootstrap_regression_difference(
            y_true, base, plus, n_resamples=200, random_state=11
        )
        self.assertGreater(difference["Delta_R2"], 0)
        self.assertLess(difference["Delta_RMSE"], 0)


class PreprocessingTests(unittest.TestCase):
    def test_split_and_transform_mixed_features(self):
        frame = pd.DataFrame(
            {
                "employee_id": [1, 2, 3],
                "target": [0, 1, 0],
                "age": [20.0, np.nan, 40.0],
                "department": ["A", None, "B"],
                "constant": [1, 1, 1],
            }
        )
        X, y = split_xy(frame, "target", ["employee_id"])
        transformed = make_preprocessor(X, scale_numeric=True).fit_transform(X)
        self.assertEqual(y.tolist(), [0, 1, 0])
        self.assertNotIn("constant", X.columns)
        self.assertEqual(transformed.shape[0], 3)

    def test_ghrm_constructs_require_all_items(self):
        row = {
            column: 4.0
            for columns in GHRM_ITEMS.values()
            for column in columns
        }
        row["Timestamp"] = "2026-01-01"
        incomplete = dict(row)
        incomplete["GRS2"] = np.nan
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ghrm.csv"
            pd.DataFrame([row, incomplete]).rename(
                columns={"GTD3": "GDT3"}
            ).to_csv(path, index=False)
            info = {
                "filename": "ghrm.csv",
                "target": "FEP",
            }
            with patch("src.preprocessing.DATA_DIR", Path(temp_dir)), patch.dict(
                "src.preprocessing.DATASETS", {"ghrm": info}, clear=False
            ):
                frame, _, _ = load_dataset("ghrm")
        self.assertEqual(frame.loc[0, "GRS"], 4.0)
        self.assertTrue(pd.isna(frame.loc[1, "GRS"]))


class ClusteringTests(unittest.TestCase):
    def test_kmeans_saves_all_k_values_and_profiles(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame(
            {
                "id": range(60),
                "x": np.r_[rng.normal(-3, 0.2, 30), rng.normal(3, 0.2, 30)],
                "y": np.r_[rng.normal(-3, 0.2, 30), rng.normal(3, 0.2, 30)],
                "target": [0] * 30 + [1] * 30,
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics, selected_k, labels = run_kmeans(
                frame, "target", ["id"], temp_dir, "synthetic"
            )
            output = Path(temp_dir)
            self.assertTrue((output / "cluster_target_summary.csv").is_file())
            self.assertTrue((output / "cluster_distinguishing_features.csv").is_file())
        self.assertEqual(metrics["k"].tolist(), [2, 3, 4, 5, 6, 7])
        self.assertEqual(selected_k, 2)
        self.assertEqual(len(labels), len(frame))


class ExplainabilityTests(unittest.TestCase):
    def test_tree_nodes_include_leaf_probabilities(self):
        X = pd.DataFrame({"x": [0, 1, 2, 3]})
        y = np.array([0, 0, 1, 1])
        model = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
        nodes = tree_nodes(model, np.array(["x"]), "demo", "Decision Tree")
        self.assertEqual(len(nodes), model.tree_.node_count)
        self.assertIn("Positive_Probability", nodes.columns)
        self.assertTrue(nodes["Is_Leaf"].any())


class ReportingTests(unittest.TestCase):
    def test_select_kmeans_row_uses_maximum_silhouette(self):
        metrics = pd.DataFrame(
            {
                "k": [2, 3, 4],
                "Silhouette": [0.2, 0.4, 0.3],
                "Davies_Bouldin": [1.1, 1.2, 0.9],
            }
        )
        self.assertEqual(int(select_kmeans_row(metrics)["k"]), 3)

    def test_markdown_table_uses_ascii_digits_and_decimal(self):
        table = markdown_table(pd.DataFrame({"label": ["A|B"], "value": [0.125]}))
        self.assertIn("A\\|B", table)
        self.assertIn("0.1250", table)

    def test_markdown_table_preserves_sha256_identity(self):
        digest = "a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659"
        table = markdown_table(pd.DataFrame({"SHA256": [digest]}))
        self.assertIn(f"`{digest}`", table)

    def test_silhouette_interpretation_is_consistent(self):
        self.assertIn("weak", interpret_silhouette(0.1))
        self.assertIn("moderate", interpret_silhouette(0.4))
        self.assertIn("clear", interpret_silhouette(0.6))


if __name__ == "__main__":
    unittest.main()

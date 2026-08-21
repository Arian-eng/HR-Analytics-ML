import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.clustering import run_kmeans
from src.evaluation import (
    bootstrap_classification_metrics,
    bootstrap_regression_difference,
    bootstrap_regression_metrics,
    classification_metrics,
    mcnemar_pair,
    regression_metrics,
)
from src.green_hrm_analysis import ITEMS, load as load_ghrm
from src.preprocessing import make_preprocessor, split_xy
from src.reporting import (
    interpret_silhouette,
    markdown_table,
    select_kmeans_row,
    to_persian_digits,
)


class EvaluationTests(unittest.TestCase):
    def test_classification_metrics_use_positive_class(self):
        metrics = classification_metrics(
            np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
        )

        self.assertAlmostEqual(metrics["Accuracy"], 0.75)
        self.assertAlmostEqual(metrics["Precision"], 2 / 3)
        self.assertAlmostEqual(metrics["Recall"], 1.0)
        self.assertAlmostEqual(metrics["F1"], 0.8)

    def test_mcnemar_counts_discordant_predictions(self):
        result = mcnemar_pair(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 0]),
            np.array([0, 0, 0, 0]),
        )

        self.assertEqual(result["b01"], 1)
        self.assertEqual(result["b10"], 1)

    def test_bootstrap_intervals_are_deterministic_and_contain_points(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        first = bootstrap_classification_metrics(
            y_true, y_pred, n_resamples=200, random_state=7
        )
        second = bootstrap_classification_metrics(
            y_true, y_pred, n_resamples=200, random_state=7
        )

        self.assertEqual(first, second)
        point = classification_metrics(y_true, y_pred)
        for metric, value in point.items():
            self.assertLessEqual(first[f"{metric}_CI_Lower"], value)
            self.assertGreaterEqual(first[f"{metric}_CI_Upper"], value)

    def test_regression_bootstrap_and_paired_difference(self):
        y_true = np.arange(1.0, 11.0)
        base = y_true + np.array([0.4, -0.3] * 5)
        plus = y_true + np.array([0.2, -0.1] * 5)

        intervals = bootstrap_regression_metrics(
            y_true, plus, n_resamples=200, random_state=11
        )
        difference = bootstrap_regression_difference(
            y_true, base, plus, n_resamples=200, random_state=11
        )

        point_rmse = regression_metrics(y_true, plus)["RMSE"]
        self.assertLessEqual(intervals["RMSE_CI_Lower"], point_rmse)
        self.assertGreaterEqual(intervals["RMSE_CI_Upper"], point_rmse)
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
        dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        self.assertTrue(np.isfinite(dense).all())


class ClusteringTests(unittest.TestCase):
    def test_large_inputs_use_deterministic_silhouette_sample(self):
        frame = pd.DataFrame(
            {
                "employee_id": range(30),
                "target": [0, 1] * 15,
                "value": np.linspace(0, 1, 30),
                "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "outputs"
            figures = Path(temp_dir) / "figures"
            with patch("src.clustering.SILHOUETTE_SAMPLE_SIZE", 10), patch(
                "src.clustering.silhouette_score", return_value=0.5
            ) as score:
                metrics, best_k, labels = run_kmeans(
                    frame,
                    "target",
                    ["employee_id"],
                    output,
                    figures,
                    "synthetic",
                )

        self.assertEqual(best_k, 2)
        self.assertEqual(len(labels), len(frame))
        self.assertEqual(metrics["k"].tolist(), [2, 3, 4, 5, 6, 7])
        self.assertEqual(score.call_count, 6)
        for call in score.call_args_list:
            self.assertEqual(call.kwargs["sample_size"], 10)
            self.assertEqual(call.kwargs["random_state"], 42)


class ReportingTests(unittest.TestCase):
    def test_select_kmeans_row_uses_maximum_silhouette(self):
        metrics = pd.DataFrame(
            {
                "k": [2, 3, 4],
                "Silhouette": [0.2, 0.4, 0.3],
                "Davies_Bouldin": [1.1, 1.2, 0.9],
            }
        )

        selected = select_kmeans_row(metrics)

        self.assertEqual(int(selected["k"]), 3)

    def test_markdown_table_formats_numbers_and_escapes_pipes(self):
        table = markdown_table(
            pd.DataFrame({"label": ["A|B"], "value": [0.123456]})
        )

        self.assertIn("A\\|B", table)
        self.assertIn("۰.۱۲۳۵", table)
        self.assertNotIn("٫", table)

    def test_persian_digits_preserve_ascii_decimal_point(self):
        self.assertEqual(to_persian_digits("12.50"), "۱۲.۵۰")

    def test_silhouette_interpretation_depends_only_on_score(self):
        self.assertEqual(interpret_silhouette(0.10), "Weak exploratory separation")
        self.assertEqual(
            interpret_silhouette(0.40), "Moderate exploratory separation"
        )
        self.assertEqual(interpret_silhouette(0.60), "Strong exploratory separation")


class GreenHRMTests(unittest.TestCase):
    def test_construct_means_require_all_declared_items(self):
        row = {
            column: 4.0
            for columns in ITEMS.values()
            for column in columns
        }
        incomplete = dict(row)
        incomplete["GRS2"] = np.nan

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ghrm.csv"
            pd.DataFrame([row, incomplete]).to_csv(path, index=False)
            with patch("src.green_hrm_analysis.DATA_DIR", Path(temp_dir)), patch(
                "src.green_hrm_analysis.GHRM_FILES", ["ghrm.csv"]
            ):
                frame = load_ghrm()

        self.assertEqual(frame.loc[0, "GRS"], 4.0)
        self.assertTrue(pd.isna(frame.loc[1, "GRS"]))


if __name__ == "__main__":
    unittest.main()

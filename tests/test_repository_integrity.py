import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FIGURES = {
    "chart10_gee_effect_reliability.png",
    "chart1_ibm_age_histogram.png",
    "chart2_ibm_income_boxplot.png",
    "chart3_jobchange_target_distribution.png",
    "chart4_promotion_status_distribution.png",
    "chart5_ibm_class_distribution_split.png",
    "chart6_jobchange_missing_before_after.png",
    "chart7_promotion_missing_before_after.png",
    "chart8_cross_dataset_silhouette_db.png",
    "chart9_f1_comparison_across_algorithms.png",
    "ghrm_base_vs_gee_r2.png",
    "ghrm_correlation_heatmap.png",
    "ghrm_decision_tree.png",
    "ghrm_kmeans_criteria.png",
    "ghrm_kmeans_pca.png",
    "ibm_attrition_confusion_matrices.png",
    "ibm_attrition_decision_tree.png",
    "ibm_attrition_kmeans_criteria.png",
    "ibm_attrition_kmeans_pca.png",
    "job_change_confusion_matrices.png",
    "job_change_decision_tree.png",
    "job_change_kmeans_criteria.png",
    "job_change_kmeans_pca.png",
    "promotion_confusion_matrices.png",
    "promotion_decision_tree.png",
    "promotion_kmeans_criteria.png",
    "promotion_kmeans_pca.png",
}

RESULT_FILES = [
    "ghrm_full_report.json",
    "ghrm_reliability_report.json",
    "ibm_attrition_classification_report.json",
    "ibm_attrition_kmeans_report.json",
    "job_change_classification_report.json",
    "job_change_kmeans_report.json",
    "promotion_classification_report.json",
    "promotion_kmeans_report.json",
]

class RepositoryIntegrityTests(unittest.TestCase):
    def test_figure_manifest_tracks_final_archive(self):
        manifest = ROOT / "figures" / "MANIFEST.md"
        self.assertTrue(manifest.is_file())
        text = manifest.read_text(encoding="utf-8")
        for name in EXPECTED_FIGURES:
            self.assertIn(f"`{name}`", text)
        self.assertEqual(27, len(EXPECTED_FIGURES))

    def test_result_json_files_parse(self):
        for name in RESULT_FILES:
            path = ROOT / "results" / name
            self.assertTrue(path.is_file(), name)
            with path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertIsInstance(payload, dict)
            self.assertTrue(payload, name)

    def test_primary_ghrm_split_is_unchanged(self):
        with (ROOT / "results" / "ghrm_full_report.json").open(encoding="utf-8") as fh:
            report = json.load(fh)
        self.assertEqual(320, report["n_total"])
        self.assertEqual(256, report["n_train"])
        self.assertEqual(64, report["n_test"])

    def test_no_machine_specific_repo2_paths_remain(self):
        for path in (ROOT / "src").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/home/claude/repo2", text, path.name)

if __name__ == "__main__":
    unittest.main()

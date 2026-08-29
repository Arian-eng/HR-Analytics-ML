# Thesis-to-repository evidence map

This file helps a reviewer trace each analytical claim to the code, machine-readable output and figure used to support it.

| Thesis / analysis item | Source code | Numerical evidence | Visual evidence |
|---|---|---|---|
| GHRM construct scores | `src/run_ghrm.py`, `src/run_ghrm_reliability.py` | `results/ghrm_reliability_report.json` | `figures/ghrm_correlation_heatmap.png` |
| GHRM reliability | `src/run_ghrm_reliability.py` | `results/ghrm_reliability_report.json` | `figures/ghrm_correlation_heatmap.png` |
| GHRM Base regression | `src/run_ghrm.py` | `results/ghrm_full_report.json` | `figures/ghrm_base_vs_gee_r2.png`, `figures/ghrm_decision_tree.png` |
| GHRM GEE+ comparison | `src/run_ghrm.py` | `results/ghrm_full_report.json` | `figures/ghrm_base_vs_gee_r2.png`, `figures/chart10_gee_effect_reliability.png` |
| GHRM K-Means | `src/run_ghrm.py` | `results/ghrm_full_report.json` | `figures/ghrm_kmeans_criteria.png`, `figures/ghrm_kmeans_pca.png` |
| IBM classification | `src/run_ibm.py`, `src/ch3_utils.py` | `results/ibm_attrition_classification_report.json` | `figures/ibm_attrition_confusion_matrices.png`, `figures/ibm_attrition_decision_tree.png` |
| IBM K-Means | `src/run_ibm.py`, `src/ch3_utils.py` | `results/ibm_attrition_kmeans_report.json` | `figures/ibm_attrition_kmeans_criteria.png`, `figures/ibm_attrition_kmeans_pca.png` |
| Job Change classification | `src/run_jobchange.py`, `src/ch3_utils.py` | `results/job_change_classification_report.json` | `figures/job_change_confusion_matrices.png`, `figures/job_change_decision_tree.png` |
| Job Change K-Means | `src/run_jobchange.py`, `src/ch3_utils.py` | `results/job_change_kmeans_report.json` | `figures/job_change_kmeans_criteria.png`, `figures/job_change_kmeans_pca.png` |
| Promotion classification | `src/run_promotion.py` and staged promotion scripts | `results/promotion_classification_report.json` | `figures/promotion_confusion_matrices.png`, `figures/promotion_decision_tree.png` |
| Promotion K-Means | `src/run_promotion.py`, `src/_promotion_step3_tree_and_kmeans.py` | `results/promotion_kmeans_report.json` | `figures/promotion_kmeans_criteria.png`, `figures/promotion_kmeans_pca.png` |
| Chapter 4 descriptive figures | `src/generate_chapter4_charts.py` | underlying dataset summaries and committed model outputs | `figures/chart1_*` through `figures/chart9_*` |
| Cross-dataset model comparison | result JSON files | `results/MODEL_RESULTS.md` | `figures/chart9_f1_comparison_across_algorithms.png` |
| Hidden-pattern synthesis | all four result families | `results/PATTERNS_AND_FINDINGS.md` | K-Means, confusion-matrix and GHRM comparison figures |
| Final repository conclusion | all committed evidence | `results/MODEL_RESULTS.md`, `results/PATTERNS_AND_FINDINGS.md` | relevant figures above |

## Reviewer reading order

A reviewer who wants to verify the work without reading every source file can follow this sequence:

1. `README.md` — project scope and dataset roles.
2. `docs/formulas.md` — formulas and metric definitions.
3. `results/MODEL_RESULTS.md` — readable model tables.
4. `figures/README.md` — purpose and interpretation of each figure.
5. `docs/decision_trees.md` — fitted tree complexity and interpretation.
6. `results/PATTERNS_AND_FINDINGS.md` — table of patterns obtained from the analyses.
7. `docs/conclusion.md` — evidence-based conclusion and limits.
8. `validation/validation_report.md` — cross-check against the existing Chapter 4 text.
9. `results/*.json` and `src/*.py` — detailed machine-readable evidence and implementation.

## Scope boundary

The mapping intentionally keeps the three public HR datasets separate from the direct GHRM survey. IBM Attrition, Job Change and Promotion are methodological benchmark analyses. The Green HRM constructs and FEP are measured directly only in `HRM_DATASETS.csv`.
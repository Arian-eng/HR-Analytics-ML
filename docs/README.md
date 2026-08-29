# Documentation index

This folder contains the reviewer-facing methodological, interpretive and defense documentation.

## Core methodology

- [`formulas.md`](formulas.md) — equations and definitions for classification metrics, regression metrics, Decision Tree criteria, Random Forest, SVM/SVR, MLP, K-Means, Cronbach alpha, bootstrap intervals and McNemar tests.
- [`analysis_flow.md`](analysis_flow.md) — end-to-end workflow diagrams, dataset-role separation and evidence chain.
- [`thesis_mapping.md`](thesis_mapping.md) — thesis analysis item → code → JSON evidence → figure/table evidence.

## Model transparency

- [`decision_trees.md`](decision_trees.md) — fitted tree depth, leaf count and interpretation for IBM, Job Change, Promotion and GHRM.
- [`../results/MODEL_RESULTS.md`](../results/MODEL_RESULTS.md) — direct reviewer-readable numerical model results.
- [`../figures/README.md`](../figures/README.md) — all 27 PNG figures with direct links and interpretation.

## Interpretation and limitations

- [`conclusion.md`](conclusion.md) — evidence-based conclusions with explicit limits on causal interpretation.
- [`DEFENSE_QA.md`](DEFENSE_QA.md) — concise defense answers for n=320, class imbalance, Promotion tree complexity, FEP reliability, GEE+, K-Means and dataset scope.
- [`FINAL_AUDIT.md`](FINAL_AUDIT.md) — final repository audit and statements that should not be overclaimed.

## Validation

- [`../validation/validation_report.md`](../validation/validation_report.md) — independent comparison between repository results and the existing thesis text, including documented differences.

## Recommended reviewer order

Start at [`../START_HERE.md`](../START_HERE.md), then open `MODEL_RESULTS.md`, `figures/README.md`, `thesis_mapping.md`, and `FINAL_AUDIT.md`.

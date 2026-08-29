# ghrm_decision_tree

**Purpose:** documents the fitted Decision Tree Regressor for the GHRM Base model.

**Structure:** depth 4 with 11 leaves.

**Interpretation:** provides an inspectable recursive rule structure for predicting FEP from GRS, GTD, GPA and GCM.

**Source:** `src/run_ghrm.py` and `results/ghrm_full_report.json`.

**Limitation:** tree splits are predictive rules, not causal thresholds.
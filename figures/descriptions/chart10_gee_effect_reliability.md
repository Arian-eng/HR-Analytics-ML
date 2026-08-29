# chart10_gee_effect_reliability

**Purpose:** summarizes the change from the GHRM Base specification to GEE+ together with paired bootstrap uncertainty.

**Result:** mean ΔR² (GEE+−Base) is RF −0.0180, DT −0.0073, LinearSVR −0.0263 and MLP +0.1387. The 95% paired bootstrap intervals for all four models include zero.

**Interpretation:** MLP has the largest positive point change, but the evidence is insufficient to claim a statistically reliable general improvement from adding GEE.

**Source:** `results/ghrm_full_report.json` and `src/run_ghrm.py`.

**Limitation:** predictive comparison on one held-out split; no causal claim.
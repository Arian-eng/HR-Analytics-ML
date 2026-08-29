# ghrm_base_vs_gee_r2

**Purpose:** compares held-out R² between the Base GHRM predictors (GRS, GTD, GPA, GCM) and the GEE+ specification that adds GEE.

**Results:** Base R²: RF 0.4374, DT 0.4361, LinearSVR 0.4287, MLP 0.2350. GEE+: RF 0.4197, DT 0.4295, LinearSVR 0.4058, MLP 0.3707.

**Interpretation:** only MLP improves in point estimate; the paired bootstrap comparison does not establish a statistically reliable improvement.

**Limitation:** no causal interpretation of GEE.
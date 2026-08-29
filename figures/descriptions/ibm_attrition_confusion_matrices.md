# ibm_attrition_confusion_matrices

**Purpose:** compares false positives and false negatives for the four IBM Attrition classifiers.

**Held-out F1:** DT 0.4127, RF 0.1786, LinearSVC 0.4511, MLP 0.5000. RF accuracy is 0.8435 but positive-class recall is only 0.1064.

**Interpretation:** demonstrates why accuracy alone is misleading under imbalance.

**Source:** `results/ibm_attrition_classification_report.json`.

**Limitation:** confusion matrices reflect the documented test split.
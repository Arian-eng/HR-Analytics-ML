# job_change_confusion_matrices

**Purpose:** compares classification error patterns for Job Change.

**Held-out F1:** DT 0.5758, RF 0.6042, LinearSVC 0.5888, MLP 0.5248.

**Interpretation:** Random Forest has the highest F1 in the committed run; the matrices make recall/precision trade-offs visible.

**Source:** `results/job_change_classification_report.json`.

**Limitation:** results are split- and preprocessing-specific.
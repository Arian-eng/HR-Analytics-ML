# promotion_confusion_matrices

**Purpose:** compares error patterns for Promotion classifiers under strong class imbalance.

**Held-out F1:** DT 0.4306, RF 0.4385, LinearSVC 0.3716, MLP 0.5054. LinearSVC recall is 0.8383 but precision only 0.2387; MLP has the highest F1.

**Interpretation:** illustrates the precision/recall trade-off hidden by accuracy.

**Source:** `results/promotion_classification_report.json`.
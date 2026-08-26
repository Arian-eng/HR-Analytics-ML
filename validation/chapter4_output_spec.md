# Chapter 4 output specification

## Publication authority

The final thesis document is the sole numeric source for tracked GitHub results. Its identity, protocol values, all 19 Chapter 4 tables, and narrative-only results are stored in `results/thesis_chapter4_reference.json`.

`scripts/sync_thesis_chapter4.py` must generate:

- one exact CSV for every table from 4-1 through 4-19;
- canonical classification metrics and confusion matrices;
- Base and `GEE+` GHRM regression metrics;
- K-Means selection, sizes, labels, and GHRM profiles;
- missing-value and McNemar tables;
- the Persian Chapter 4 report and numeric figures.

## Required analysis scope

- IBM HR Analytics: binary `Attrition` classification and independent K-Means.
- Job Change: binary `target` classification and independent K-Means.
- Employee Promotion: binary `is_promoted` classification and independent K-Means.
- GHRM: continuous `FEP` regression and independent K-Means without `FEP` in cluster formation.

The split is 80/20 with `random_state=42`; classification is stratified. K-Means evaluates k=2..7 on standardized numeric features. Job Change and Employee Promotion use a deterministic 5,000-row Silhouette sample.

## Separation from live runs

`run_all.py` may write detailed diagnostics and predictions under ignored `outputs/`. Those files may be used to investigate reproducibility, but they are not a second publication authority. The final publication step always re-synchronizes tracked files from the thesis reference.

Persian presentation uses Persian digits and an ASCII full stop. Machine-readable CSV files use ASCII digits.

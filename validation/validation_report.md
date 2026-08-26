# Final Chapter 4 publication validation

Date: 2026-08-26

Overall assessment: **all tracked Chapter 4 numbers are synchronized with the final thesis document**.

## Source identity

The publication source is `پایان نامه آرین پوراسد.docx`, SHA-256:

```text
712c29ee29dda7a53159ae3ddab08569e334682bd8ad996b79a72420385c0d9f
```

All 19 tables are represented in `results/thesis_chapter4_reference.json` and in separate CSV files under `results/tables/`.

## Raw-data checks

The four local data files matched `validation/data_manifest.json` by SHA-256 and row count:

| Dataset | Rows | Target | Role |
|---|---:|---|---|
| IBM HR Analytics | 1,470 | `Attrition` | classification and K-Means |
| Job Change | 19,158 | `target` | classification and K-Means |
| Employee Promotion | 54,808 | `is_promoted` | classification and K-Means |
| GHRM | 320 | continuous `FEP` | regression and K-Means |

## Automated QA

- Tables 4-1 through 4-19 match the versioned reference cell by cell.
- All 12 classification rows match thesis tables 4-11, 4-13, and 4-15.
- Classification values recompute from all 12 thesis confusion matrices after three-decimal rounding.
- All eight regression rows match thesis tables 4-17 and 4-18.
- K-Means k, SSE, Silhouette, Davies–Bouldin, cluster sizes, labels, and profiles match tables 4-9 and 4-10.
- Cluster sizes sum to the source row counts and the cluster count matches selected k.
- McNemar p-values match the values reported in the thesis narrative.
- The Persian report includes all 19 captions, Persian digits, and ASCII decimal points.
- Unsupported old confidence intervals, paired-difference tables, prediction scatterplots, and non-thesis K-Means diagnostic figures were removed from the tracked publication bundle.

Run the checks with:

```bash
python scripts/sync_thesis_chapter4.py
python -m compileall -q run_all.py src tests scripts
python -m unittest discover -s tests -v
python scripts/validate_published_results.py --require-data
python -m pip check
```

Live estimates and predictions under ignored `outputs/` are diagnostic artifacts. They do not replace the final values without a corresponding revision to the thesis reference.

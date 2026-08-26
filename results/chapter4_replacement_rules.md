# Chapter 4 publication rules

1. The final file `پایان نامه آرین پوراسد.docx` is the sole numeric publication authority.
2. Its recorded SHA-256 is `712c29ee29dda7a53159ae3ddab08569e334682bd8ad996b79a72420385c0d9f`.
3. Tables 4-1 through 4-19 are versioned in `thesis_chapter4_reference.json`; do not maintain a second benchmark list.
4. Run `python scripts/sync_thesis_chapter4.py` to regenerate the report, all table CSVs, and all numeric figures.
5. Live model outputs belong under ignored `outputs/`. They must not overwrite the tracked thesis publication values.
6. If the thesis itself changes, update the reference, its source hash, all generated artifacts, and the validator in the same commit.
7. Classification metrics must round from the matching thesis confusion matrix to three decimals.
8. Cluster sizes must sum to 1,470, 19,158, 54,808, and 320, with selected k values 2, 3, 5, and 2.
9. GHRM Base and `GEE+` results must remain separate; predictive results do not establish causality or mediation.
10. Use Persian digits with an ASCII full stop in the Persian report; retain ASCII digits in machine-readable CSVs.
11. Run `python scripts/validate_published_results.py --require-data` before committing when all four raw files are available.

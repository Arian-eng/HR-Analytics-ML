# Pre-defense revision — 2026-09-08

This revision aligns the revised thesis with executed outputs, not vice versa.

## Evidence and changes

- The completed verification run of source commit `8d80b9c7bb1db8f954737f9733e80acac0f6866c` reproduced Job Change, Promotion, all K-means outputs and GHRM to saved precision. The raw uploaded datasets were used unchanged.
- IBM RF and MLP differed. Current IBM F1 values are 0.1818 and 0.5783. Stored historical values were 0.1786 and 0.5000. The old environment/search provenance is insufficient to assign a proven historical cause. Current outputs, selected parameters, confidence intervals, confusion matrices and summaries now use the completed re-run.
- GHRM scaling was moved inside the inner-CV Pipeline for LinearSVR and MLPRegressor. GHRM was rerun. MLP Base R² is 0.2349 and GEE+ R² is 0.3697. All paired ΔR² intervals still include zero.
- McNemar uses the survival function to avoid cancellation for tiny p-values. Holm corrections use all six comparisons per dataset. This is not an F1 significance test.
- `results/defense_checks.json` contains post-hoc dummy baselines and five GHRM 80/20 splits (42–46), with RF re-tuned on training-only folds. OLS is close to RF; ranking changes across splits. Overlapping splits are not independent replicates or external validation.

## Reproduction

Use the version of this file and the scripts from the same git commit. The thesis appendix contains excerpts; the repository scripts are the executable source.

```bash
python -m pip install -r requirements-reproduction.txt
python src/run_ibm.py
python src/run_jobchange.py
python src/run_promotion.py
python src/run_ghrm.py
python src/run_ghrm_reliability.py
python src/defense_checks.py
```

Promotion intentionally uses its per-model orchestration and four stored search grids. Run its entry point rather than applying the general RF grid to it.

Python: 3.12.13. Numerical packages: numpy 2.3.5, pandas 2.2.3, scipy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8. Execution used single-thread BLAS/OpenMP. Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for the same compute policy.

## Limits retained

Only GHRM directly measures green constructs; its n=320 is not increased by the other datasets. FEP is a self-report composite with alpha 0.6043. No new data, external validation, causal identification, or formal power guarantee was produced. Existing model bootstrap intervals are conditional on fixed fitted predictions. Some candidate LinearSVR fits emitted convergence warnings; this limits claims about exhaustive optimizer convergence. Prior source and numerical reports remain available in git history.

## Method references

- https://scikit-learn.org/stable/common_pitfalls.html
- https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html
- https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html

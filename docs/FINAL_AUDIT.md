# Final repository audit for thesis review

This file is a reviewer-facing checklist of what is actually present in the repository and how each evidence layer should be interpreted. It does not change any numerical result.

## 1. Evidence chain

The repository now exposes the analysis in the following order:

1. **Data role and scope** — `README.md` and `docs/thesis_mapping.md` identify the four datasets and separate the direct GHRM survey from the three general HR benchmark datasets.
2. **Method** — `src/` contains the executable pipelines; `docs/formulas.md` documents the metrics and statistical criteria used.
3. **Machine-readable results** — `results/*.json` are the authoritative published numerical outputs.
4. **Tables** — `results/tables/` provides compact reviewer-readable summaries of model results, K-Means selection, reliability, tree structure, and discovered patterns.
5. **Figures** — `figures/README.md` explains the intended role of every final figure; `figures/MANIFEST.md` records the exact binary identity of the 27 original PNG artifacts from the final analysis archive; `figures/summary_*.svg` provides directly viewable summary plots generated from the committed result numbers.
6. **Interpretation** — `results/PATTERNS_AND_FINDINGS.md`, `docs/decision_trees.md`, and `docs/conclusion.md` distinguish descriptive patterns, predictive findings, uncertainty, and limitations.
7. **Validation** — `validation/validation_report.md` documents the cross-check against the thesis and explicitly reports both matching and non-matching results instead of hiding differences.

## 2. Numerical integrity

No published model score, hyperparameter, split size, random seed, K-Means result, reliability coefficient, or Base-vs-GEE+ bootstrap result was changed while adding reviewer documentation and portability improvements.

The committed JSON files remain the numerical source of truth for this repository.

## 3. Direct GHRM evidence versus benchmark evidence

The 320-record GHRM survey is the only dataset that directly contains the Green HRM constructs GRS, GTD, GPA, GCM, GEE and FEP. The IBM Attrition, Job Change and Employee Promotion datasets are retained as general HR benchmark / methodological validation analyses and are not described as direct measurements of Green HRM.

This distinction is intentional and should be preserved in any defense explanation.

## 4. Model evidence available to a reviewer

### Classification

For IBM Attrition, Job Change and Employee Promotion the repository reports, for each of Decision Tree, Random Forest, LinearSVC and MLPClassifier:

- selected hyperparameters,
- Accuracy, Precision, Recall and F1,
- 95% bootstrap confidence intervals,
- pairwise McNemar comparisons,
- confusion-matrix figure references,
- decision-tree structure information where applicable.

### GHRM regression

For the Base and GEE+ specifications the repository reports, for Random Forest, Decision Tree, LinearSVR and MLPRegressor:

- selected hyperparameters,
- R², RMSE and MAE,
- five-fold CV RMSE,
- paired bootstrap Base-vs-GEE+ comparisons,
- decision-tree complexity,
- K-Means results on the GHRM construct scores.

### Unsupervised analysis

K-Means reports SSE, Silhouette and Davies-Bouldin for k=2..7. The selected k is based on maximum Silhouette, with the other two criteria reported as supporting diagnostics.

## 5. Reliability and construct-level evidence

The GHRM reliability report contains Cronbach's alpha for all six constructs and the construct-level Pearson correlation matrix. Five constructs are at or above the conventional 0.70 reference; FEP is below it and is explicitly reported as a limitation rather than hidden.

## 6. Decision-tree transparency

`docs/decision_trees.md` reports the fitted tree sizes and interprets them cautiously. In particular, the Promotion Decision Tree has depth 63 and 4,855 leaves; this is documented as an overfitting / interpretability warning rather than presented as an unqualified strength.

## 7. Figure status

The repository contains directly rendered SVG summary figures based on the committed JSON values. The final analysis archive also contains 27 original PNG artifacts (descriptive charts, confusion matrices, decision trees, K-Means criteria and PCA plots). Their expected filenames, sizes and SHA-256 values are recorded in `figures/MANIFEST.md` so their binary identity is auditable.

The numerical conclusions in this repository do not depend on reverse-reading values from image files; they come from the committed JSON outputs.

## 8. Reproducibility / portability

Machine-specific `/home/claude/repo2` paths were removed. Scripts now resolve project paths relative to the repository root. Dependency ranges are documented in `requirements.txt`, and a lightweight GitHub Actions integrity check validates syntax and the published result structure without requiring the raw datasets to be committed.

## 9. Statements that should NOT be made in the defense

To keep the repository scientifically defensible, avoid claiming any of the following:

- that the three public HR benchmark datasets directly measure Green HRM;
- that high accuracy alone establishes good minority-class prediction;
- that K-Means clusters prove causal mechanisms;
- that adding GEE significantly improves every GHRM model;
- that the Promotion Decision Tree is preferable simply because it was selected by CV;
- that every classification result is bit-for-bit identical to the thesis when the validation report explicitly documents differences.

## 10. Defensible summary

The strongest repository-supported conclusions are:

- the analysis pipeline is explicit and inspectable;
- the 80/20 split and fixed random seed are documented;
- the K-Means results are fully reported with three quality criteria;
- model comparisons include metrics that are appropriate for imbalanced classes;
- uncertainty is reported with bootstrap intervals and McNemar tests;
- GHRM results are separated from benchmark HR results;
- reliability, tree complexity and limitations are visible rather than omitted;
- every important numerical conclusion can be traced back to a committed machine-readable result file.

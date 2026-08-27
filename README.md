# Reproducible HR Analytics with Machine Learning

This repository contains the code and reproducible outputs for my thesis analysis. The main study examines Green Human Resource Management (GHRM) dimensions and predicts firm environmental performance (`FEP`). Three public HR datasets are analyzed separately for employee attrition, job change, and promotion.

The four datasets are never merged. This repository contains no Digikala data.

## Dataset roles

| Dataset | Rows | Raw fields | Role in the project |
|---|---:|---:|---|
| IBM HR Analytics | 1,470 | 35 | Attrition classification and supplementary clustering |
| Job Change | 19,158 | 14 | Job-change classification and supplementary clustering |
| Employee Promotion | 54,808 | 14 | Promotion classification and supplementary clustering |
| GHRM–Environmental Performance | 320 | 33 | Main GHRM analysis, `FEP` prediction, and green-pattern discovery |

Only the fourth dataset directly measures green constructs: green recruitment and selection (`GRS`), green training and development (`GTD`), green performance appraisal (`GPA`), green compensation management (`GCM`), green employee empowerment/engagement (`GEE`), and firm environmental performance (`FEP`). The other datasets are not direct evidence of GHRM or firm sustainability.

None of the four files contains a separate direct measure named “HR productivity” or “overall firm sustainability,” so the analysis makes no empirical claim about those outcomes. See [Data and scope](docs/data_and_scope.md) for the full rationale.

## Experimental design

- Fixed 80/20 train/test split with `random_state=42`
- All preprocessing fitted only on training data
- Hyperparameter selection by cross-validation on the training split
- Four classifiers: Random Forest, Decision Tree, Linear SVM, and MLP
- Four GHRM regressors: Random Forest, Decision Tree, LinearSVR, and MLPRegressor
- Independent K-Means analysis for each dataset with `k=2..7`
- Bootstrap confidence intervals for held-out metrics
- Repeated 5x5 cross-validation for the 320-record GHRM dataset
- All six pairwise McNemar comparisons on identical held-out classification records

The complete procedure is documented in [Methodology](docs/methodology.md).

## Reproducing the analysis

Place these four files in `data/`:

```text
WA_Fn-UseC_-HR-Employee-Attrition (3).csv
aug_train.csv
train_LZdllcl.csv
HRM DATASETS.csv
```

Then run:

```bash
python -m pip install -r requirements.txt
python run_all.py
python scripts/validate_results.py --require-data
```

The full reference run took about eight minutes. `run_all.py` rebuilds `results/` from scratch so stale outputs cannot be mixed with the current run.

## Outputs

Start with the [final execution report](results/analysis_report.md) or the [executed walkthrough notebook](notebooks/analysis_walkthrough.ipynb).

| Evidence | Location |
|---|---|
| Dataset dimensions, field structure, missingness, and SHA-256 | `results/data/` |
| Metrics, confidence intervals, and selected parameters | `results/tables/` |
| Public aggregate values used to recompute metrics | confusion matrices and `regression_validation_aggregates.csv` |
| Confusion matrices | each classification model directory and `results/figures/` |
| Every node and rule in each fitted Decision Tree | the corresponding `decision_tree` directory |
| Depth and leaf count for every Random Forest tree | each `forest_tree_summary.csv` |
| Full structure of a representative forest tree | `representative_tree_nodes.csv` and its plot |
| Linear SVM and LinearSVR coefficients | `coefficients.csv` |
| MLP architecture, iteration count, and final loss | `model_diagnostics.json` |
| K selection, cluster sizes, and GHRM patterns | `results/clustering/` |
| Runtime, environment versions, and source hashes | `results/run_log.txt` and `results/run_manifest.json` |

See [Output guide](docs/output_guide.md) for an exact map from each technical question to its saved evidence.

## Main GHRM analysis

The Base regression uses `GRS`, `GTD`, `GPA`, and `GCM` to predict `FEP`; the GEE+ variant adds `GEE`. A paired bootstrap comparison reports how much adding `GEE` changes prediction performance in this sample.

For GHRM clustering, `FEP` is excluded from cluster formation and used only afterward for descriptive profiling. This prevents target leakage and shows how the five GHRM constructs group together independently of the environmental-performance outcome.

## Interpretation boundary

The analysis is predictive and exploratory. Feature importance, cluster differences, or superior model performance do not establish causality or mediation. The GHRM sample contains 320 records; confidence intervals and repeated cross-validation disclose uncertainty but do not justify broad generalization without independent data.

## Validation

```bash
python -m compileall -q run_all.py src tests scripts
python -m unittest discover -s tests -v
python scripts/validate_results.py
```

With local raw data, the validator recomputes metrics from held-out predictions. In the public repository, it performs the same checks using confusion matrices and privacy-safe aggregate error sums. It also checks cluster-size totals and the expected explainability artifacts.

Row-level predictions, split membership, and cluster assignments are generated locally but are not committed because they contain employee-level identifiers or outcomes.

## License

Code and generated artifacts are distributed under [LICENSE](LICENSE). Raw datasets remain the property of their original sources and are not included in this repository.

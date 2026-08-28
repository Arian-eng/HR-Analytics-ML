# Key Findings — synthesis across all four datasets

This mirrors what the thesis's Chapter 5 (بحث و نتیجه‌گیری) discusses, but
is written in English and grounded in **this repo's own independently-run
numbers** (see `validation/validation_report.md` for the full cross-check
against Chapter 4's text). Where a number below differs slightly from
Chapter 5's own Persian text, this repo's number is the one that is
directly traceable to the code in `src/`.

## Pattern 1 — No single algorithm wins everywhere

| Dataset | Best model (by F1) | F1 | Accuracy |
|---|---|--:|--:|
| IBM Attrition | MLPClassifier | 0.500 | 0.878 |
| Job Change | RandomForest | 0.604 | 0.763 |
| Employee Promotion | MLPClassifier | 0.505 | 0.941 |
| GHRM (FEP, base model) | RandomForestRegressor | R²=0.437 | — |

The best-performing algorithm changes across datasets and across the
independent grid search vs. Chapter 4's own tuning run (see the validation
report's cross-check table). This is itself a finding, not noise: it means
the thesis's claim that "no algorithm dominates universally" (Chapter 5,
Section 5-2) holds up under an independent re-run, not just in the
original numbers.

## Pattern 2 — Accuracy is misleading on all three classification datasets

All three targets are imbalanced (Attrition 16.1% positive, Job Change
24.9%, Promotion 8.5%). Random Forest is the clearest example: on
Promotion it reaches 93.5% accuracy but only 29.8% recall on the positive
class — it is mostly just predicting the majority class well. F1 and the
bootstrap confidence interval on recall (in each
`results/*_classification_report.json`) are the numbers that actually
describe minority-class performance; accuracy alone should not be quoted
as "the model works well."

## Pattern 3 — Cluster separability varies sharply by dataset

| Dataset | Silhouette | Interpretation |
|---|--:|---|
| Job Change | 0.577 | Clear, well-separated structure |
| GHRM | 0.436 | Reasonably separated structure |
| Employee Promotion | 0.257 | Weak-to-moderate separation |
| IBM Attrition | 0.153 | Weak separation — interpret with caution |

Job Change and GHRM have real, interpretable cluster structure. IBM's
clusters, while they do show a descriptive difference in attrition rate
between segments (19.2% vs 9.4%), should not be over-interpreted as a
strong "hidden pattern" given how low the silhouette score is — this
qualifier matters for how confidently Chapter 4/5 phrase the IBM clustering
result.

## Pattern 4 — GEE's effect on FEP prediction is real but algorithm-specific

Adding Green Employee Empowerment (GEE) as a fifth predictor changes each
regressor's R² differently:

| Model | ΔR² (GEE+ minus Base) | Statistically reliable (bootstrap CI excludes 0)? |
|---|--:|---|
| RandomForestRegressor | -0.018 | No |
| DecisionTreeRegressor | -0.007 | No |
| LinearSVR | -0.026 | No |
| **MLPRegressor** | **+0.139** | **No (CI: -0.007 to 0.283, borderline)** |

None of the four differences reach statistical reliability at the 4,000-
resample paired-bootstrap 95% level in this independent run (MLPRegressor
comes closest, with a CI that only just includes zero). This is a more
conservative conclusion than "GEE clearly helps the neural network" — the
honest reading is "GEE's apparent benefit for MLPRegressor is suggestive
but not statistically confirmed here," which is worth stating carefully in
the thesis rather than overclaiming.

## Pattern 5 — GHRM's HR-practice dimensions explain a moderate share of FEP variance

Best base-model R² = 0.437 (RandomForestRegressor: GRS+GTD+GPA+GCM →
FEP). This means the four green-HR-practice dimensions jointly account for
roughly 44% of the variance in firm environmental performance in this
sample — a real, moderate predictive signal, not a strong one. The
remaining ~56% is explained by factors outside this dataset (organizational
culture, management style, economic conditions, etc. — as Chapter 5's
10th limitation already states).

## Pattern 6 — reliability check adds one caveat not yet in the thesis text

Five of six GHRM constructs have acceptable Cronbach's alpha (≥0.70); FEP
does not (0.604). See `validation/validation_report.md` Section 2 for the
full reliability analysis — this should be named as an explicit limitation
in the thesis's discussion/limitations section (Chapter 5 already lists
ten limitations; this reliability finding would be a natural eleventh, or
folds into the existing "متوسط بودن عملکرد پیش‌بینی" limitation).

## How this compares to prior literature (brief)

Consistent with the thesis's own Chapter 5 comparison: this independent
run's best classification F1 scores (0.50–0.60) are lower than some cited
studies with more homogeneous, single-domain data (e.g. Khandan &
Mohammadi's SVM at F1>0.90 on employee turnover) — expected, since class
imbalance and heterogeneous feature sets here are harder problems than a
cleanly-separable single-company dataset. The core methodological
conclusion — that algorithm choice should be driven by empirical comparison
per problem, not assumed in advance — replicates under this independent
run.

## Where to read more

- `validation/validation_report.md` — full numeric cross-check against
  Chapter 4, including exactly which numbers match and which don't (and
  why).
- `results/*.json` — every underlying number, hyperparameter, and
  confidence interval.
- `figures/` — 26 charts: all modeling-result figures plus the 9 named
  descriptive figures Chapter 4 references by number.

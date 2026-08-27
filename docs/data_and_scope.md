# Study scope and data identity

## Main research problem

The main study uses `HRM DATASETS.csv`, which contains 320 responses to Green Human Resource Management and firm environmental-performance items. The modeling target is the composite `FEP` score.

The available data do not contain separate targets for “HR productivity” or “overall firm sustainability.” The code therefore makes no claim to measure or predict those outcomes. The direct main outcome is firm environmental performance (`FEP`); the supplementary outcomes are attrition, job change, and promotion.

The three other files cover general HR problems. They are supplementary analyses, not direct measurements of GHRM or firm sustainability. Records from different sources are never joined or pooled.

## Relationship among the four datasets

| ID | Unit of analysis | Target | Relationship to the thesis topic |
|---|---|---|---|
| `ibm` | Employees | `Attrition` | Supplementary work-behavior analysis; no direct green construct |
| `job_change` | Applicants/workers | `target` | Supplementary job-change analysis; no direct green construct |
| `promotion` | Employees | `is_promoted` | Supplementary promotion analysis; no direct green construct |
| `ghrm` | SME survey respondents | `FEP` | Main analysis with direct GHRM dimensions |

The common rationale is to evaluate the selected algorithms on three established HR tasks while restricting direct Green HRM claims to the dataset that contains the required constructs. Results remain dataset-specific: model performance on IBM is not generalized to GHRM or Ghanaian SMEs.

## Digikala status

No file, field, or record from Digikala exists in this repository. If the final thesis population or data source differs from the original proposal, that change must be explained and approved through the university process; code cannot replace that approval.

## GHRM construct creation

Each construct is the complete row mean of the listed items. If a required item is missing for a row, that row's construct score is not calculated.

| Construct | Items | Meaning |
|---|---|---|
| `GRS` | `GRS1` to `GRS4` | Green recruitment and selection |
| `GTD` | `GTD1` to `GTD5` | Green training and development |
| `GPA` | `GPA1` to `GPA6` | Green performance appraisal |
| `GCM` | `GCM1` to `GCM4` | Green compensation management |
| `GEE` | `GEE1`, `GEE4`, `GEE5`, `GEE6` | Green empowerment/engagement |
| `FEP` | `FEP1`, `FEP5`, `FEP7`, `FEP9` | Firm environmental performance |

The raw column `GDT3` is renamed to `GTD3` before construct calculation. This corrects the field name only and does not alter its values.

## Auditable data-quality checks

Every run records:

- exact source filename and SHA-256;
- raw row and column counts;
- field data types;
- missing-value counts and percentages;
- distinct-value counts;
- numeric range and mean in a local private profile;
- exact duplicate-row count;
- target completeness.

The public field structure is in `results/data/column_dictionary.csv`; file-level checks are in `results/data/data_quality.json`. The local `column_dictionary_private.csv` also contains sensitive ranges and frequencies and is not committed. Expected source hashes are stored in `validation/data_manifest.json`.

## Sample-size limitation

The three public datasets contain between 1,470 and 54,808 records. The main GHRM dataset has only 320 records. To make its uncertainty visible, the pipeline reports a held-out test, bootstrap confidence intervals, and repeated 5x5 cross-validation. These procedures do not remove the sample-size limitation; they only quantify instability more transparently.

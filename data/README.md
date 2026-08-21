# Local datasets

Place the four thesis datasets in this directory when running the project locally:

- `WA_Fn-UseC_-HR-Employee-Attrition (3)(2).csv`
- `aug_train(2).csv`
- `train_LZdllcl(2).csv`
- `HRM DATASETS(2).csv`

The loader also accepts the earlier aliases `aug_train(5).csv`, `train_LZdllcl.csv`, and `HRM DATASETS.csv`. The datasets are intentionally not committed to GitHub. Each dataset is analyzed independently and no records are merged across sources.

## Sources

- IBM HR Attrition: [IBM HR Analytics Employee Attrition & Performance on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Job Change: [HR Analytics: Job Change of Data Scientists on Kaggle](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)
- Employee Promotion: [WNS Analytics Wizard 2018 challenge](https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018-1/)
- GHRM: [PLOS ONE source article](https://doi.org/10.1371/journal.pone.0293957) and its [PLOS Figshare materials](https://plos.figshare.com/articles/dataset/Variance_Inflation_Factor_VIF_/25630768)

Confirm each source's current access and redistribution terms before sharing a
raw file. The repository deliberately tracks only metadata and results, not the
third-party CSV files.

## File identity

Exact filenames, row counts, and SHA-256 hashes for the validated thesis run are
recorded in [`validation/data_manifest.json`](../validation/data_manifest.json).
Run `python scripts/validate_published_results.py --require-data` to verify all
four local files before reproducing the final results.

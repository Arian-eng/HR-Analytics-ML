# Local data files

The full analysis requires these four files:

- `WA_Fn-UseC_-HR-Employee-Attrition (3).csv`
- `aug_train.csv`
- `train_LZdllcl.csv`
- `HRM DATASETS.csv`

Raw files are not committed. Before analysis, `run_all.py` records each filename, row and column count, SHA-256 hash, missingness, and exact duplicate count. Expected identities for the final run are stored in `validation/data_manifest.json`.

Documented sources:

- [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)
- [WNS Analytics Wizard 2018](https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018-1/)
- [GHRM study in PLOS ONE](https://doi.org/10.1371/journal.pone.0293957)

Check each source's terms before redistributing any raw file.

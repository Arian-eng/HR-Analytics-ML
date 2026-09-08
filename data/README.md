# Raw datasets

The four author-supplied CSV files are committed in this directory without content changes. Expected names are `aug_train.csv`, `HRM_DATASETS.csv`, `train_LZdllcl.csv`, and `WA_Fn-UseC_-HR-Employee-Attrition__3_.csv`. Provenance hashes and dimensions are in `validation/defense_provenance.json`.

# Data directory

The four raw CSV files are committed. The scripts expect the following filenames:

| File | Role | Rows used | Target |
|---|---|---:|---|
| `HRM_DATASETS.csv` | Primary GHRM survey | 320 | `FEP_score` derived from FEP items |
| `WA_Fn-UseC_-HR-Employee-Attrition__3_.csv` | General HR benchmark | 1,470 | Attrition |
| `aug_train.csv` | General HR benchmark | 19,158 | `target` |
| `train_LZdllcl.csv` | General HR benchmark | 54,808 | `is_promoted` |

The four datasets are analyzed independently; they are not concatenated or matched record-by-record. Only `HRM_DATASETS.csv` directly measures the Green HRM constructs used in the thesis. The other three datasets are public HR benchmark analyses used for methodological comparison and validation.

After placing the files here, run the scripts from the repository root as described in `README.md`. Project paths are resolved relative to the repository root.

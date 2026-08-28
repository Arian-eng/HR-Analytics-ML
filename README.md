# HR Analytics ML — Chapter 3 aligned pipeline

This repository was rebuilt from a clean slate around one execution entry point: `python run_all.py`.

`HRM_DATASETS.csv` (n=320) is the primary survey and the only dataset that directly measures Green HRM constructs. IBM Attrition, Job Change, and Promotion are separate public HR benchmark/proxy analyses and are not direct Green HRM evidence.

Place the four raw CSV files under `data/` with these exact names:
- `HRM_DATASETS.csv`
- `WA_Fn-UseC_-HR-Employee-Attrition__3_.csv`
- `aug_train.csv`
- `train_LZdllcl.csv`

Raw CSV files are intentionally excluded from Git.

The pipeline implements the Chapter 3 specification: Decision Tree, Random Forest, LinearSVC and MLPClassifier for the three public classification datasets; RandomForestRegressor, DecisionTreeRegressor, LinearSVR and MLPRegressor for GHRM FEP prediction; fixed 80/20 GHRM split; Base vs GEE+ comparison on the same test cases; bootstrap uncertainty; McNemar pairwise classification comparison; and K-Means reporting SSE, Silhouette and Davies-Bouldin.

This commit is an execution checkpoint, not the final all-dataset result release. See `results/CHECKPOINT_STATUS.md` for exactly which real-computation outputs are complete.

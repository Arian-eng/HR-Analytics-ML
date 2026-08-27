# Result validation

The repository has two validation layers:

1. `tests/test_pipeline.py` tests metric calculations, bootstrap intervals, McNemar logic, GHRM construct creation, preprocessing, clustering, and tree exports on small synthetic data.
2. `scripts/validate_results.py` recomputes outputs from held-out predictions during a local run. In the public repository, it recomputes classification metrics from confusion matrices and regression metrics from privacy-safe aggregate error sums.

Run without raw files:

```bash
python scripts/validate_results.py
```

This mode is used in CI and verifies the internal consistency of all published outputs.

Run locally with the four CSV files:

```bash
python scripts/validate_results.py --require-data
```

This mode also checks filenames, row counts, and SHA-256 hashes against `data_manifest.json`.

The validator does not rely on a manually edited Chapter 4 table. Its numeric authority is the saved model evidence from the current pipeline.

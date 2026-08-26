# Chapter 4 results

The executable analysis pipeline is `run_all.py`. It analyzes three binary HR datasets and the separate GHRM–Environmental Performance dataset.

The final thesis document is the publication authority. Its SHA-256 and all 19 Chapter 4 tables are recorded in [`thesis_chapter4_reference.json`](thesis_chapter4_reference.json). The Persian report is in [`chapter4_results.md`](chapter4_results.md), exact table CSVs are under [`tables/`](tables/), and derived figures are under [`figures/`](figures/).

Regenerate and validate the tracked publication bundle with:

```bash
python scripts/sync_thesis_chapter4.py
python scripts/validate_published_results.py
```

Detailed predictions, tuning parameters, confusion-matrix CSV files and intermediate live outputs remain under the ignored local `outputs/` directory. They are diagnostics, not an alternate publication source. Historical Employee Performance and Green Innovation/Sustainable Performance tables must not be mixed with the final Employee Promotion and continuous-`FEP` methodology.

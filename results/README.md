# Chapter 4 results

The current executable pipeline is `run_all.py`. It analyzes three binary HR datasets and the separate GHRM–Environmental Performance dataset.

The Persian, GitHub-ready result narrative is in [`chapter4_results.md`](chapter4_results.md). Its committed CSV tables live under [`tables/`](tables/) and its figures under [`figures/`](figures/). All files are regenerated from the same model outputs by `src/reporting.py`; the narrative does not contain manually maintained result values.

The tables include 95% bootstrap intervals, a paired Base-versus-`GEE+`
regression comparison, and cluster sizes for all four datasets. Validate the
tracked publication bundle with:

```bash
python scripts/validate_published_results.py
```

Detailed predictions, tuning parameters, confusion-matrix CSV files and intermediate outputs remain under the ignored local `outputs/` directory. Historical fixed-configuration tables that used Employee Performance and Green Innovation/Sustainable Performance must not be mixed with the current Employee Promotion and continuous-`FEP` methodology.

"""Data identity and quality checks saved with every final run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATASETS, RESULTS_DIR
from src.preprocessing import dataset_path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_number(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def build_data_profile():
    overview_rows = []
    column_rows = []
    for dataset, info in DATASETS.items():
        path = dataset_path(dataset)
        frame = pd.read_csv(path)
        target = info["target"] if info["target"] in frame.columns else "FEP items"
        target_missing = (
            int(frame[info["target"]].isna().sum())
            if info["target"] in frame.columns
            else int(frame[["FEP1", "FEP5", "FEP7", "FEP9"]].isna().any(axis=1).sum())
        )
        overview_rows.append(
            {
                "Dataset": dataset,
                "Display_Name": info["display_name"],
                "Role": info["role"],
                "File": info["filename"],
                "SHA256": sha256(path),
                "Rows": len(frame),
                "Raw_Columns": frame.shape[1],
                "Exact_Duplicate_Rows": int(frame.duplicated().sum()),
                "Target": target,
                "Missing_Target": target_missing,
            }
        )

        for column in frame.columns:
            series = frame[column]
            numeric = pd.to_numeric(series, errors="coerce")
            numeric_count = int(numeric.notna().sum())
            row = {
                "Dataset": dataset,
                "Column": column,
                "Dtype": str(series.dtype),
                "Missing_Count": int(series.isna().sum()),
                "Missing_Percent": float(series.isna().mean() * 100),
                "Distinct_Count": int(series.nunique(dropna=True)),
                "Minimum": None,
                "Maximum": None,
                "Mean": None,
                "Most_Common": None,
                "Most_Common_Count": None,
            }
            if numeric_count == series.notna().sum() and numeric_count:
                row.update(
                    {
                        "Minimum": _safe_number(numeric.min()),
                        "Maximum": _safe_number(numeric.max()),
                        "Mean": _safe_number(numeric.mean()),
                    }
                )
            else:
                counts = series.fillna("<missing>").astype(str).value_counts()
                if not counts.empty:
                    row["Most_Common"] = counts.index[0]
                    row["Most_Common_Count"] = int(counts.iloc[0])
            column_rows.append(row)
    return pd.DataFrame(overview_rows), pd.DataFrame(column_rows)


def write_data_profile():
    output = RESULTS_DIR / "data"
    output.mkdir(parents=True, exist_ok=True)
    overview, columns = build_data_profile()
    overview.to_csv(output / "dataset_inventory.csv", index=False, encoding="utf-8-sig")
    columns.to_csv(
        output / "column_dictionary_private.csv", index=False, encoding="utf-8-sig"
    )
    columns[
        [
            "Dataset",
            "Column",
            "Dtype",
            "Missing_Count",
            "Missing_Percent",
            "Distinct_Count",
        ]
    ].to_csv(output / "column_dictionary.csv", index=False, encoding="utf-8-sig")
    summary = {
        "datasets": overview.to_dict(orient="records"),
        "checks": {
            "exact_duplicate_rows": "checked on all raw columns",
            "missing_values": "reported per raw column",
            "target_completeness": "checked before modelling",
            "hash_algorithm": "SHA-256",
        },
    }
    (output / "data_quality.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return overview, columns

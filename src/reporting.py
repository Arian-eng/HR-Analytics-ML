"""Publish the tracked Chapter 4 bundle from the final thesis reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def to_persian_digits(value: object) -> str:
    """Render Persian digits while preserving the ASCII decimal point."""

    return str(value).translate(PERSIAN_DIGITS)


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return to_persian_digits(f"{value:.4f}")
    if isinstance(value, (int, np.integer)):
        return to_persian_digits(f"{value:,}")
    return to_persian_digits(value)


def markdown_table(frame: pd.DataFrame) -> str:
    """Return a dependency-free GitHub Markdown table with Persian digits."""

    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        cells = [
            _format_value(value).replace("|", "\\|").replace("\n", " ")
            for value in values
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def select_kmeans_row(metrics: pd.DataFrame) -> pd.Series:
    """Return the maximum-Silhouette row for exploratory diagnostics."""

    return metrics.loc[metrics["Silhouette"].idxmax()]


def interpret_silhouette(value: float) -> str:
    if value < 0.25:
        return "Weak exploratory separation"
    if value < 0.50:
        return "Moderate exploratory separation"
    return "Strong exploratory separation"


def build_chapter4_outputs() -> None:
    """Regenerate tracked artifacts without importing live model outputs."""

    from scripts.sync_thesis_chapter4 import sync

    sync()

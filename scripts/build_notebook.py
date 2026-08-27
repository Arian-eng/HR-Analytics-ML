"""Build and execute the lightweight audit notebook without a Jupyter dependency."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import traceback


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "analysis_walkthrough.ipynb"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def cells():
    return [
        markdown(
            """# مرور قابل بازتولید تحلیل پایان‌نامه

این Notebook خروجی‌های ساخته‌شده با `python run_all.py` را مرحله‌به‌مرحله مرور می‌کند. اجرای مدل‌ها در اسکریپت اصلی انجام می‌شود و این فایل برای کنترل مسیر داده تا نتیجه است.
"""
        ),
        code(
            """from pathlib import Path
import json
import subprocess
import sys
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "results").is_dir():
    ROOT = ROOT.parent
RESULTS = ROOT / "results"
print(f"project root: {ROOT}")
"""
        ),
        markdown("## ۱. شناسنامه چهار دیتاست\n"),
        code(
            """inventory = pd.read_csv(RESULTS / "data" / "dataset_inventory.csv")
print(inventory[["Dataset", "Display_Name", "Role", "Rows", "Raw_Columns", "Target", "SHA256"]].to_string(index=False))
"""
        ),
        markdown("## ۲. مدل‌های طبقه‌بندی و مدل منتخب هر مسئله\n"),
        code(
            """classification = pd.read_csv(RESULTS / "tables" / "classification_metrics.csv")
best_classifiers = classification.loc[classification.groupby("Dataset")["F1"].idxmax(), ["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1", "CV_Best_F1"]]
print(best_classifiers.to_string(index=False))
"""
        ),
        markdown("## ۳. تحلیل اصلی GHRM و پیش‌بینی FEP\n"),
        code(
            """regression = pd.read_csv(RESULTS / "tables" / "regression_metrics.csv")
columns = ["Variant", "Model", "R2", "RMSE", "MAE", "R2_CI_Lower", "R2_CI_Upper", "Repeated_CV_R2_Mean", "Tuning_Convergence_Warnings", "Repeated_CV_Convergence_Warnings"]
print(regression[columns].to_string(index=False))
"""
        ),
        markdown("## ۴. اثر پیش‌بینانه افزودن GEE\n"),
        code(
            """comparison = pd.read_csv(RESULTS / "tables" / "base_vs_gee_plus.csv")
print(comparison.to_string(index=False))
"""
        ),
        markdown("## ۵. انتخاب تعداد خوشه‌ها\n"),
        code(
            """kmeans = pd.read_csv(RESULTS / "tables" / "kmeans_selection.csv")
print(kmeans[["Dataset", "k", "Numeric_Features", "Inertia_SSE", "Silhouette", "Davies_Bouldin", "Interpretation"]].to_string(index=False))
"""
        ),
        markdown(
            """## ۶. الگوی پنهان مستقیم GHRM

در این بخش خوشه‌ها فقط با پنج سازه سبز ساخته شده‌اند. `FEP` در تشکیل خوشه دخالت نداشته و بعد از آن برای توصیف عملکرد محیط‌زیستی هر خوشه اضافه شده است.
"""
        ),
        code(
            """patterns = pd.read_csv(RESULTS / "clustering" / "ghrm" / "cluster_distinguishing_features.csv")
targets = pd.read_csv(RESULTS / "clustering" / "ghrm" / "cluster_target_summary.csv")
print("ویژگی‌های متمایزکننده:")
print(patterns.to_string(index=False))
print("\\nعملکرد محیط‌زیستی پس از خوشه‌بندی:")
print(targets.to_string(index=False))
"""
        ),
        markdown("## ۷. کنترل مستقل فایل‌های منتشرشده\n"),
        code(
            """validation = subprocess.run([sys.executable, "scripts/validate_results.py"], cwd=ROOT, check=True, capture_output=True, text=True)
print(validation.stdout.strip())
"""
        ),
    ]


def execute(notebook_cells):
    namespace = {"__name__": "__notebook__"}
    count = 0
    for cell in notebook_cells:
        if cell["cell_type"] != "code":
            continue
        count += 1
        source = "".join(cell["source"])
        stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile(source, f"cell-{count}", "exec"), namespace)
        except Exception as error:
            cell["execution_count"] = count
            cell["outputs"] = [
                {
                    "output_type": "error",
                    "ename": error.__class__.__name__,
                    "evalue": str(error),
                    "traceback": traceback.format_exc().splitlines(),
                }
            ]
            raise
        cell["execution_count"] = count
        text = stream.getvalue()
        cell["outputs"] = (
            [{"output_type": "stream", "name": "stdout", "text": text.splitlines(True)}]
            if text
            else []
        )


def main():
    notebook_cells = cells()
    for index, cell in enumerate(notebook_cells, start=1):
        cell["id"] = f"cell-{index:02d}"
    execute(notebook_cells)
    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote and executed {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

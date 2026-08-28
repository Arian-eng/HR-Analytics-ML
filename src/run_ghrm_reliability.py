"""
GHRM survey (HRM_DATASETS.csv, n=320) - reliability and descriptive
analysis of the six constructs (GRS, GTD, GPA, GCM, GEE, FEP).

NOTE ON SCOPE: Chapter 4 of the thesis reports composite-score descriptives
(Table 4-4) but does NOT compute its own Cronbach's alpha from this
320-respondent sample - it only states that reliability/validity were
checked in the original source study (Adu Sarfo et al.). This script adds
that missing piece: it independently computes Cronbach's alpha for each
construct from the actual data, plus a construct-level correlation matrix,
as evidence the thesis can point to if asked "did you check your own
sample's reliability, not just cite the original paper's."

Descriptive stats below are cross-checked against Chapter 4 Table 4-4 in
the validation report.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/home/claude/repo2/results"
FIGURES = "/home/claude/repo2/figures"

df = pd.read_csv("/home/claude/repo2/data/HRM_DATASETS.csv")
df = df.rename(columns={"GDT3": "GTD3"})

CONSTRUCTS = {
    "GRS": ["GRS1", "GRS2", "GRS3", "GRS4"],
    "GTD": ["GTD1", "GTD2", "GTD3", "GTD4", "GTD5"],
    "GPA": ["GPA1", "GPA2", "GPA3", "GPA4", "GPA5", "GPA6"],
    "GCM": ["GCM1", "GCM2", "GCM3", "GCM4"],
    "GEE": ["GEE1", "GEE4", "GEE5", "GEE6"],
    "FEP": ["FEP1", "FEP5", "FEP7", "FEP9"],
}


def cronbach_alpha(items: pd.DataFrame) -> float:
    items = items.dropna()
    k = items.shape[1]
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))


report = {"n": int(len(df))}

# ---------------- Cronbach's alpha ----------------
alphas = {name: round(cronbach_alpha(df[cols]), 4) for name, cols in CONSTRUCTS.items()}
report["cronbach_alpha"] = alphas
report["alpha_interpretation"] = {
    name: ("acceptable (>=0.70)" if a >= 0.70 else "questionable (<0.70)")
    for name, a in alphas.items()
}

# ---------------- Composite scores + descriptives (cross-check vs Chapter 4 Table 4-4) ----------------
for name, cols in CONSTRUCTS.items():
    df[f"{name}_score"] = df[cols].mean(axis=1)
score_cols = [f"{n}_score" for n in CONSTRUCTS]

desc = df[score_cols].describe().T[["mean", "std", "min", "max"]].round(2)
report["descriptives"] = desc.to_dict(orient="index")

# ---------------- Correlation matrix ----------------
corr = df[score_cols].corr(method="pearson")
report["correlation_matrix"] = corr.round(3).to_dict()

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
labels = [c.replace("_score", "") for c in score_cols]
ax.set_xticks(range(len(score_cols))); ax.set_xticklabels(labels)
ax.set_yticks(range(len(score_cols))); ax.set_yticklabels(labels)
for i in range(len(score_cols)):
    for j in range(len(score_cols)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("GHRM construct correlation matrix (n=320)")
plt.tight_layout()
plt.savefig(f"{FIGURES}/ghrm_correlation_heatmap.png", dpi=150)
plt.close()

with open(f"{RESULTS}/ghrm_reliability_report.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print("n =", report["n"])
print("\nCronbach's alpha:")
for name, a in alphas.items():
    print(f"  {name}: {a}  ({report['alpha_interpretation'][name]})")
print("\nDescriptives (mine) vs Chapter 4 Table 4-4:")
thesis_table = {"GRS": (3.44, 0.89), "GTD": (3.47, 0.84), "GPA": (3.46, 0.87),
                 "GCM": (3.45, 0.88), "GEE": (3.43, 0.87), "FEP": (3.54, 0.73)}
for name in CONSTRUCTS:
    mine = report["descriptives"][f"{name}_score"]
    t_mean, t_std = thesis_table[name]
    print(f"  {name}: mine mean={mine['mean']} std={mine['std']}  |  thesis mean={t_mean} std={t_std}")
print("\nCorrelation matrix:")
print(corr.round(3))

"""
The 9 descriptive/summary figures Chapter 4 explicitly refers to by number
(نمودار 1-4 through 9-4) that were missing from the repo - the earlier
figures/ additions (confusion matrices, decision trees, per-dataset K-Means
criteria, correlation heatmap) covered the modeling results but not these.
All built from the same real data/results already in this repo.
"""
import sys, json
sys.path.insert(0, "/home/claude/repo2/src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

FIGURES = "/home/claude/repo2/figures"
RESULTS = "/home/claude/repo2/results"
RNG_SEED = 42

# ============================================================
# 1-4: IBM age histogram
# ============================================================
ibm = pd.read_csv("/home/claude/repo2/data/WA_Fn-UseC_-HR-Employee-Attrition__3_.csv")
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(ibm["Age"], bins=20, color="#2b6cb0", edgecolor="white")
ax.set_xlabel("Age"); ax.set_ylabel("Count")
ax.set_title(f"IBM Attrition: employee age distribution (n={len(ibm)}, mean={ibm['Age'].mean():.2f})")
plt.tight_layout(); plt.savefig(f"{FIGURES}/chart1_ibm_age_histogram.png", dpi=150); plt.close()

# ============================================================
# 2-4: IBM monthly income boxplot
# ============================================================
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.boxplot(ibm["MonthlyIncome"], vert=True)
ax.set_ylabel("Monthly Income")
ax.set_title(f"IBM Attrition: monthly income distribution\n(median={ibm['MonthlyIncome'].median():.0f})")
ax.set_xticks([])
plt.tight_layout(); plt.savefig(f"{FIGURES}/chart2_ibm_income_boxplot.png", dpi=150); plt.close()

# ============================================================
# 3-4: Job Change target distribution
# ============================================================
jc = pd.read_csv("/home/claude/repo2/data/aug_train.csv")
jc = jc.dropna(subset=["target"]).copy()
counts = jc["target"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["0 (no change)", "1 (looking to change)"], counts.values, color=["#2b6cb0", "#e53e3e"])
for b, c in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, c, f"{int(c)}\n({c/counts.sum()*100:.1f}%)", ha="center", va="bottom")
ax.set_ylabel("Count")
ax.set_title(f"Job Change: target class distribution (n={len(jc)})")
plt.tight_layout(); plt.savefig(f"{FIGURES}/chart3_jobchange_target_distribution.png", dpi=150); plt.close()

# ============================================================
# 4-4: Promotion status distribution
# ============================================================
promo = pd.read_csv("/home/claude/repo2/data/train_LZdllcl.csv")
counts = promo["is_promoted"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["0 (not promoted)", "1 (promoted)"], counts.values, color=["#2b6cb0", "#38a169"])
for b, c in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, c, f"{int(c)}\n({c/counts.sum()*100:.1f}%)", ha="center", va="bottom")
ax.set_ylabel("Count")
ax.set_title(f"Employee Promotion: is_promoted distribution (n={len(promo)})")
plt.tight_layout(); plt.savefig(f"{FIGURES}/chart4_promotion_status_distribution.png", dpi=150); plt.close()

# ============================================================
# 5-4: IBM Attrition class distribution - full vs train vs test
# ============================================================
ibm["Attrition_bin"] = (ibm["Attrition"] == "Yes").astype(int)
drop_cols = {"Attrition", "Attrition_bin", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"}
X = ibm.drop(columns=[c for c in drop_cols if c in ibm.columns])
y = ibm["Attrition_bin"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RNG_SEED, stratify=y)

full_pct = y.value_counts(normalize=True).sort_index() * 100
train_pct = y_train.value_counts(normalize=True).sort_index() * 100
test_pct = y_test.value_counts(normalize=True).sort_index() * 100

fig, ax = plt.subplots(figsize=(6.5, 4.5))
x = np.arange(2); width = 0.25
ax.bar(x - width, full_pct.values, width, label=f"Full (n={len(y)})", color="#2b6cb0")
ax.bar(x, train_pct.values, width, label=f"Train (n={len(y_train)})", color="#38a169")
ax.bar(x + width, test_pct.values, width, label=f"Test (n={len(y_test)})", color="#dd6b20")
ax.set_xticks(x); ax.set_xticklabels(["No Attrition (0)", "Attrition (1)"])
ax.set_ylabel("% of records")
ax.set_title("IBM Attrition: class balance preserved across full/train/test\n(stratified 80/20 split)")
ax.legend()
plt.tight_layout(); plt.savefig(f"{FIGURES}/chart5_ibm_class_distribution_split.png", dpi=150); plt.close()

# ============================================================
# 6-4: Job Change missing values before/after imputation
# ============================================================
jc_cols = ["gender", "enrolled_university", "education_level", "major_discipline",
           "experience", "company_size", "company_type", "last_new_job"]
before = [(1 - jc[c].isna().mean()) * 100 for c in jc_cols]
after = [100.0] * len(jc_cols)  # imputation guarantees full completeness

fig, ax = plt.subplots(figsize=(9, 5))
y_pos = np.arange(len(jc_cols))
ax.barh(y_pos - 0.2, before, height=0.4, label="Before imputation", color="#e53e3e")
ax.barh(y_pos + 0.2, after, height=0.4, label="After imputation", color="#38a169")
ax.set_yticks(y_pos); ax.set_yticklabels(jc_cols)
ax.set_xlabel("% of records with a value")
ax.set_title("Job Change: data completeness before vs after imputation")
ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(f"{FIGURES}/chart6_jobchange_missing_before_after.png", dpi=150); plt.close()

# ============================================================
# 7-4: Promotion missing values before/after imputation
# ============================================================
promo_cols = ["education", "previous_year_rating"]
before = [(1 - promo[c].isna().mean()) * 100 for c in promo_cols]
after = [100.0, 100.0]

fig, ax = plt.subplots(figsize=(6, 4))
y_pos = np.arange(len(promo_cols))
ax.barh(y_pos - 0.2, before, height=0.4, label="Before imputation", color="#e53e3e")
ax.barh(y_pos + 0.2, after, height=0.4, label="After imputation", color="#38a169")
ax.set_yticks(y_pos); ax.set_yticklabels(["education", "previous_year_rating"])
ax.set_xlabel("% of records with a value")
ax.set_title("Employee Promotion: data completeness before vs after imputation")
ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(f"{FIGURES}/chart7_promotion_missing_before_after.png", dpi=150); plt.close()

# ============================================================
# 8-4: Cross-dataset Silhouette + Davies-Bouldin comparison (chosen k)
# ============================================================
datasets = ["ibm_attrition", "job_change", "promotion", "ghrm"]
labels = ["IBM\n(k=2)", "Job Change\n(k=3)", "Promotion\n(k=5)", "GHRM\n(k=2)"]
sil_vals, db_vals = [], []
for ds in datasets[:3]:
    with open(f"{RESULTS}/{ds}_kmeans_report.json") as f:
        d = json.load(f)
    row = [r for r in d["by_k"] if r["k"] == d["best_k"]][0]
    sil_vals.append(row["silhouette"]); db_vals.append(row["davies_bouldin"])
with open(f"{RESULTS}/ghrm_full_report.json") as f:
    g = json.load(f)
row = [r for r in g["kmeans"]["by_k"] if r["k"] == g["kmeans"]["best_k"]][0]
sil_vals.append(row["silhouette"]); db_vals.append(row["davies_bouldin"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
colors = ["#2b6cb0", "#38a169", "#dd6b20", "#805ad5"]
axes[0].bar(labels, sil_vals, color=colors)
axes[0].set_ylabel("Silhouette Score"); axes[0].set_title("Silhouette (higher = better separation)")
for i, v in enumerate(sil_vals):
    axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
axes[1].bar(labels, db_vals, color=colors)
axes[1].set_ylabel("Davies-Bouldin Index"); axes[1].set_title("Davies-Bouldin (lower = better separation)")
for i, v in enumerate(db_vals):
    axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
fig.suptitle("K-Means cluster quality across all four datasets (at each dataset's chosen k)")
plt.tight_layout(); plt.savefig(f"{FIGURES}/chart8_cross_dataset_silhouette_db.png", dpi=150); plt.close()

# ============================================================
# 9-4: F1-Score comparison across classification algorithms (3 datasets)
# ============================================================
model_names = ["DecisionTree", "RandomForest", "LinearSVC", "MLPClassifier"]
ds_labels = ["IBM Attrition", "Job Change", "Promotion"]
f1_matrix = []
for ds in ["ibm_attrition", "job_change", "promotion"]:
    with open(f"{RESULTS}/{ds}_classification_report.json") as f:
        d = json.load(f)
    f1_matrix.append([d["models"][m]["f1"] for m in model_names])
f1_matrix = np.array(f1_matrix)  # shape (3 datasets, 4 models)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(ds_labels)); width = 0.2
colors = ["#2b6cb0", "#38a169", "#dd6b20", "#805ad5"]
for i, m in enumerate(model_names):
    ax.bar(x + (i - 1.5) * width, f1_matrix[:, i], width, label=m, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(ds_labels)
ax.set_ylabel("F1-Score (positive class)")
ax.set_title("F1-Score comparison across classification algorithms\n(three general HR datasets)")
ax.legend()
plt.tight_layout(); plt.savefig(f"{FIGURES}/chart9_f1_comparison_across_algorithms.png", dpi=150); plt.close()

print("All 9 figures generated:")
import os
for f in sorted(os.listdir(FIGURES)):
    if f.startswith("chart"):
        print(" ", f)

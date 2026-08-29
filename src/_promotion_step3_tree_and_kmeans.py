import sys, json, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from ch3_utils import build_preprocessor, run_kmeans_numeric_only, RNG_SEED, RESULTS, FIGURES

df = pd.read_csv(PROJECT_ROOT / "data" / "train_LZdllcl.csv")
numeric_cols = ["no_of_trainings", "age", "previous_year_rating", "length_of_service",
                 "KPIs_met >80%", "awards_won?", "avg_training_score"]
categorical_cols = ["department", "region", "education", "gender", "recruitment_channel"]

X = df[numeric_cols + categorical_cols]
y = df["is_promoted"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RNG_SEED, stratify=y)

with open(f"{RESULTS}/promotion_DecisionTree_partial.json") as f:
    dt_params = json.load(f)["best_params"]

preproc = build_preprocessor(numeric_cols, categorical_cols)
Xt_train = preproc.fit_transform(X_train)
dt = DecisionTreeClassifier(random_state=RNG_SEED, class_weight="balanced", **dt_params)
dt.fit(Xt_train, y_train)
feat_names = list(preproc.get_feature_names_out())

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt, feature_names=feat_names, class_names=["0", "1"], filled=True, max_depth=3, fontsize=8, ax=ax)
ax.set_title("promotion: Decision Tree (best-CV params, first 3 levels shown)")
plt.tight_layout()
plt.savefig(f"{FIGURES}/promotion_decision_tree.png", dpi=150)
plt.close()

dt_structure = {"n_leaves": int(dt.get_n_leaves()), "depth": int(dt.get_depth())}
print("DT structure:", dt_structure)

# patch into the classification report
with open(f"{RESULTS}/promotion_classification_report.json") as f:
    report = json.load(f)
report["decision_tree_structure"] = dt_structure
with open(f"{RESULTS}/promotion_classification_report.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

# K-Means (numeric only, corrected)
t0 = time.time()
km, _ = run_kmeans_numeric_only(df, numeric_cols, "promotion")
print("kmeans runtime:", time.time() - t0)
print("best_k:", km["best_k"])
for r in km["by_k"]:
    print(r)

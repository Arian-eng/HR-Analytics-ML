import sys, json, time
sys.path.insert(0, "/home/claude/repo2/src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from ch3_utils import kmeans_three_criteria, plot_kmeans_criteria, RESULTS, FIGURES, RNG_SEED

df = pd.read_csv("/home/claude/repo2/data/HRM_DATASETS.csv")
df = df.rename(columns={"GDT3": "GTD3"})
CONSTRUCTS = {
    "GRS": ["GRS1", "GRS2", "GRS3", "GRS4"], "GTD": ["GTD1", "GTD2", "GTD3", "GTD4", "GTD5"],
    "GPA": ["GPA1", "GPA2", "GPA3", "GPA4", "GPA5", "GPA6"], "GCM": ["GCM1", "GCM2", "GCM3", "GCM4"],
    "FEP": ["FEP1", "FEP5", "FEP7", "FEP9"], "GEE": ["GEE1", "GEE4", "GEE5", "GEE6"],
}
for name, cols in CONSTRUCTS.items():
    df[f"{name}_score"] = df[cols].mean(axis=1)

train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=RNG_SEED)
train_df = df.loc[train_idx].reset_index(drop=True)
test_df = df.loc[test_idx].reset_index(drop=True)
assert len(test_df) == 64

BASE_PREDICTORS = ["GRS_score", "GTD_score", "GPA_score", "GCM_score"]
GEE_PREDICTORS = BASE_PREDICTORS + ["GEE_score"]
TARGET = "FEP_score"

rmse_scorer = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)
cv5 = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)

GRIDS = {
    "RandomForestRegressor": (RandomForestRegressor(random_state=RNG_SEED, n_jobs=1),
                                {"n_estimators": [100, 200, 400], "max_depth": [None, 5, 10], "max_features": ["sqrt", 1.0]}),
    "DecisionTreeRegressor": (DecisionTreeRegressor(random_state=RNG_SEED),
                                {"max_depth": [3, 4, 6, None], "min_samples_leaf": [1, 5, 10]}),
    "LinearSVR": (LinearSVR(random_state=RNG_SEED, max_iter=20000),
                   {"C": [0.1, 1, 10], "epsilon": [0.0, 0.1, 0.2]}),
    "MLPRegressor": (MLPRegressor(max_iter=3000, early_stopping=True, random_state=RNG_SEED),
                       {"hidden_layer_sizes": [(50,), (32, 16), (16, 8)], "alpha": [0.0001, 0.001, 0.01],
                        "learning_rate_init": [0.001, 0.01]}),
}
NEEDS_SCALING = {"LinearSVR", "MLPRegressor"}


def tune_and_fit(predictors, label):
    Xtr = train_df[predictors].values
    Xte = test_df[predictors].values
    ytr = train_df[TARGET].values
    yte = test_df[TARGET].values

    fitted, preds, best_params_all, cv_scores = {}, {}, {}, {}
    for name, (est, grid) in GRIDS.items():
        if name in NEEDS_SCALING:
            scaler = StandardScaler().fit(Xtr)
            Xtr_use, Xte_use = scaler.transform(Xtr), scaler.transform(Xte)
        else:
            Xtr_use, Xte_use = Xtr, Xte
        t0 = time.time()
        gs = GridSearchCV(est, grid, scoring=rmse_scorer, cv=cv5, n_jobs=1, refit=True)
        gs.fit(Xtr_use, ytr)
        best = gs.best_estimator_
        yp = best.predict(Xte_use)
        fitted[name] = best
        preds[name] = yp
        best_params_all[name] = gs.best_params_
        cv_scores[name] = round(float(-gs.best_score_), 4)  # back to positive RMSE
        print(f"  [{label}] {name}: best_params={gs.best_params_} cv_rmse={cv_scores[name]:.4f} runtime={time.time()-t0:.1f}s")
    return fitted, preds, yte, best_params_all, cv_scores


print("=== BASE model ===")
base_fitted, base_preds, y_test_true, base_params, base_cv = tune_and_fit(BASE_PREDICTORS, "base")
print("=== GEE+ model ===")
gee_fitted, gee_preds, _, gee_params, gee_cv = tune_and_fit(GEE_PREDICTORS, "gee+")


def metric_set(y_true, y_pred):
    return {"r2": round(float(r2_score(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4)}

model_results = {}
for name in base_preds:
    model_results[name] = {
        "base": {**metric_set(y_test_true, base_preds[name]), "best_params": base_params[name], "cv_rmse": base_cv[name]},
        "gee_plus": {**metric_set(y_test_true, gee_preds[name]), "best_params": gee_params[name], "cv_rmse": gee_cv[name]},
    }
    print(name, "BASE", model_results[name]["base"], "| GEE+", model_results[name]["gee_plus"])

# ---------------- Paired bootstrap: base vs GEE+ (4000 resamples) ----------------
rng = np.random.RandomState(RNG_SEED)
n_test = len(y_test_true)
N_BOOT = 4000
bootstrap_results = {}
for name in base_preds:
    bp, gp = base_preds[name], gee_preds[name]
    diffs = {"r2": [], "rmse": [], "mae": []}
    for _ in range(N_BOOT):
        idx = rng.randint(0, n_test, n_test)
        yt = y_test_true[idx]
        try:
            b_r2 = r2_score(yt, bp[idx]); g_r2 = r2_score(yt, gp[idx])
        except Exception:
            b_r2 = g_r2 = np.nan
        diffs["r2"].append(g_r2 - b_r2)
        diffs["rmse"].append(np.sqrt(mean_squared_error(yt, gp[idx])) - np.sqrt(mean_squared_error(yt, bp[idx])))
        diffs["mae"].append(mean_absolute_error(yt, gp[idx]) - mean_absolute_error(yt, bp[idx]))
    out = {}
    for k, v in diffs.items():
        v = np.array(v); v = v[~np.isnan(v)]
        out[k] = {"mean_diff_gee_minus_base": round(float(np.mean(v)), 4),
                   "ci_low": round(float(np.percentile(v, 2.5)), 4),
                   "ci_high": round(float(np.percentile(v, 97.5)), 4),
                   "ci_excludes_zero": bool(np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0)}
    bootstrap_results[name] = out
    print(name, "diff CI (r2):", out["r2"])

report = {"n_total": int(len(df)), "n_train": int(len(train_df)), "n_test": int(len(test_df)),
          "base_predictors": BASE_PREDICTORS, "gee_predictors": GEE_PREDICTORS, "target": TARGET,
          "model_results": model_results, "bootstrap_base_vs_gee_4000resamples": bootstrap_results,
          "compute_settings": {"cv_folds": 5, "cv_scoring": "RMSE (KFold, not stratified)", "bootstrap_resamples": 4000, "seed": RNG_SEED}}

# ---------------- Figures ----------------
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, name in zip(axes, base_preds.keys()):
    r2b = model_results[name]["base"]["r2"]; r2g = model_results[name]["gee_plus"]["r2"]
    ax.bar(["Base", "GEE+"], [r2b, r2g], color=["#2b6cb0", "#38a169"])
    ax.set_title(f"{name}\nR²: {r2b:.2f} -> {r2g:.2f}")
    ax.set_ylim(min(0, r2b, r2g) - 0.05, 1)
fig.suptitle("GHRM: FEP prediction R² - Base vs GEE+ (test set, n=64)")
plt.tight_layout(); plt.savefig(f"{FIGURES}/ghrm_base_vs_gee_r2.png", dpi=150); plt.close()

dt_reg = base_fitted["DecisionTreeRegressor"]
fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(dt_reg, feature_names=BASE_PREDICTORS, filled=True, fontsize=10, ax=ax)
ax.set_title(f"GHRM: Decision Tree Regressor (base model, best-CV params: {base_params['DecisionTreeRegressor']})")
plt.tight_layout(); plt.savefig(f"{FIGURES}/ghrm_decision_tree.png", dpi=150); plt.close()
report["decision_tree_structure"] = {"n_leaves": int(dt_reg.get_n_leaves()), "depth": int(dt_reg.get_depth())}

# ---------------- K-Means on GRS/GTD/GPA/GCM/GEE (FEP excluded) ----------------
cluster_cols = ["GRS_score", "GTD_score", "GPA_score", "GCM_score", "GEE_score"]
Xc = StandardScaler().fit_transform(df[cluster_cols].values)
km_result = kmeans_three_criteria(Xc, seed=RNG_SEED, sample_cap_silhouette=320)
km_result["dataset"] = "ghrm"; km_result["cluster_input_vars"] = cluster_cols
plot_kmeans_criteria(km_result, "ghrm")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
km = KMeans(n_clusters=km_result["best_k"], n_init=20, max_iter=300, random_state=RNG_SEED)
labels = km.fit_predict(Xc)
df["cluster"] = labels
fep_by_cluster = df.groupby("cluster")["FEP_score"].agg(["mean", "std", "count"]).round(3)
km_result["fep_by_cluster"] = fep_by_cluster.to_dict(orient="index")

pca = PCA(n_components=2, random_state=RNG_SEED)
coords = pca.fit_transform(Xc)
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=25, alpha=0.8)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title(f"GHRM: K-Means clusters (k={km_result['best_k']}) on GRS/GTD/GPA/GCM/GEE")
legend1 = ax.legend(*sc.legend_elements(), title="Cluster"); ax.add_artist(legend1)
plt.tight_layout(); plt.savefig(f"{FIGURES}/ghrm_kmeans_pca.png", dpi=150); plt.close()
report["kmeans"] = km_result

with open(f"{RESULTS}/ghrm_full_report.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print("\nkmeans best_k:", km_result["best_k"])
for r in km_result["by_k"]:
    print(r)
print("fep by cluster:", km_result["fep_by_cluster"])

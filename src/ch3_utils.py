"""
Chapter-3-compliant pipeline, v2 - corrects 3 bugs found by cross-checking
against the thesis's own Chapter 4 text:

  1. Train/test split must be 80/20 for ALL four datasets (Chapter 3,
     Table 3-3), not 75/25.
  2. K-Means for the three general datasets must use ONLY standardized
     numeric features (Chapter 3: "تحلیل بر اساس ویژگی‌های عددی
     استانداردشده انجام می‌گیرد") - no one-hot categoricals. K range is
     2-7 (not 2-6), n_init=20, max_iter=300, random_state=42. Silhouette
     on large datasets uses a fixed 5000-row sample.
  3. Hyperparameters are chosen by a *limited* grid search (Chapter 3's own
     wording) with 3-fold Stratified CV scored on positive-class F1
     (classification) or 5-fold CV scored on RMSE (GHRM regression) -
     fixed arbitrary hyperparameters are not used.
"""
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, make_scorer,
    confusion_matrix, ConfusionMatrixDisplay,
    r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score, davies_bouldin_score,
)
from scipy import stats as scipy_stats

RESULTS = "/home/claude/repo2/results"
FIGURES = "/home/claude/repo2/figures"
RNG_SEED = 42


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, numeric_cols), ("cat", categorical_pipe, categorical_cols)])


def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "method": "no_discordant_pairs", "statistic": None, "p_value": 1.0}
    if n_discordant < 25:
        k = min(b, c)
        p = min(2 * scipy_stats.binom.cdf(k, n_discordant, 0.5), 1.0)
        return {"b": b, "c": c, "n_discordant": n_discordant, "method": "exact_binomial", "statistic": None, "p_value": float(p)}
    stat = (abs(b - c) - 1) ** 2 / n_discordant
    p = float(1 - scipy_stats.chi2.cdf(stat, df=1))
    return {"b": b, "c": c, "n_discordant": n_discordant, "method": "chi2_continuity_corrected", "statistic": float(stat), "p_value": p}


def bootstrap_classification_ci(y_true, y_pred, n_boot=2000, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        metrics["accuracy"].append(accuracy_score(yt, yp))
        metrics["precision"].append(precision_score(yt, yp, zero_division=0))
        metrics["recall"].append(recall_score(yt, yp, zero_division=0))
        metrics["f1"].append(f1_score(yt, yp, zero_division=0))
    return {k: {"ci_low": round(float(np.percentile(v, 2.5)), 4), "ci_high": round(float(np.percentile(v, 97.5)), 4)} for k, v in metrics.items()}


def kmeans_three_criteria(X, k_range=range(2, 8), seed=RNG_SEED, n_init=20, max_iter=300, sample_cap_silhouette=5000):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=seed)
        labels = km.fit_predict(X)
        sse = float(km.inertia_)
        sil = float(silhouette_score(X, labels, sample_size=min(sample_cap_silhouette, len(X)), random_state=seed))
        db = float(davies_bouldin_score(X, labels))
        rows.append({"k": k, "sse": round(sse, 2), "silhouette": round(sil, 4), "davies_bouldin": round(db, 4)})
    best = max(rows, key=lambda r: r["silhouette"])
    return {"by_k": rows, "best_k": best["k"], "selection_rule": "max silhouette (2<=k<=7); SSE and Davies-Bouldin reported as supplementary criteria"}


def plot_kmeans_criteria(kmeans_result, dataset_name):
    rows = kmeans_result["by_k"]
    ks = [r["k"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(ks, [r["sse"] for r in rows], marker="o"); axes[0].set_title("SSE"); axes[0].set_xlabel("k")
    axes[1].plot(ks, [r["silhouette"] for r in rows], marker="o", color="green"); axes[1].set_title("Silhouette (higher=better)"); axes[1].set_xlabel("k")
    axes[1].axvline(kmeans_result["best_k"], color="gray", linestyle="--")
    axes[2].plot(ks, [r["davies_bouldin"] for r in rows], marker="o", color="orange"); axes[2].set_title("Davies-Bouldin (lower=better)"); axes[2].set_xlabel("k")
    fig.suptitle(f"{dataset_name}: K-Means model selection (SSE / Silhouette / Davies-Bouldin)")
    plt.tight_layout(); plt.savefig(f"{FIGURES}/{dataset_name}_kmeans_criteria.png", dpi=150); plt.close()


def run_kmeans_numeric_only(df, numeric_cols, dataset_name, sample_cap=None, seed=RNG_SEED):
    """K-Means using ONLY standardized numeric features - matches Chapter 3
    exactly for the three general HR datasets. Missing numeric values are
    median-imputed first (Chapter 3's stated missing-value handling)."""
    t0 = time.time()
    d = df
    if sample_cap and len(d) > sample_cap:
        d = d.sample(sample_cap, random_state=seed)
    X = d[numeric_cols].values
    X = SimpleImputer(strategy="median").fit_transform(X)
    Xt = StandardScaler().fit_transform(X)

    result = kmeans_three_criteria(Xt, seed=seed)
    result["dataset"] = dataset_name
    result["n_used"] = int(len(d))
    result["cluster_input_vars"] = numeric_cols
    plot_kmeans_criteria(result, dataset_name)

    km = KMeans(n_clusters=result["best_k"], n_init=20, max_iter=300, random_state=seed)
    labels = km.fit_predict(Xt)
    d = d.copy()
    d["cluster"] = labels

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(Xt)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=8, alpha=0.6)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"{dataset_name}: K-Means clusters (k={result['best_k']}, numeric features only)")
    legend1 = ax.legend(*sc.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    plt.tight_layout(); plt.savefig(f"{FIGURES}/{dataset_name}_kmeans_pca.png", dpi=150); plt.close()

    result["runtime_seconds"] = round(time.time() - t0, 1)
    result["cluster_labels"] = labels.tolist()
    result["_d_index"] = d.index.tolist()
    with open(f"{RESULTS}/{dataset_name}_kmeans_report.json", "w") as f:
        json.dump({k: v for k, v in result.items() if k not in ("cluster_labels", "_d_index")}, f, ensure_ascii=False, indent=2, default=str)
    return result, d


# ---------------------------------------------------------------------------
# Grid-searched classification suite
# ---------------------------------------------------------------------------
GRIDS_CLASSIFICATION = {
    "DecisionTree": {
        "clf__max_depth": [6, 10, 15, None],
        "clf__min_samples_split": [2, 10],
    },
    "RandomForest": {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 15],
    },
    "LinearSVC": {
        "clf__C": [0.1, 1, 10],
    },
    "MLPClassifier": {
        "clf__hidden_layer_sizes": [(50,), (32, 16)],
        "clf__alpha": [0.0001, 0.001],
        "clf__learning_rate_init": [0.001, 0.01],
    },
}


def run_classification_suite(df, target_col, numeric_cols, categorical_cols, dataset_name,
                              random_state=RNG_SEED, n_boot=2000, cv_folds=3):
    t0 = time.time()
    X = df[numeric_cols + categorical_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)

    preproc = build_preprocessor(numeric_cols, categorical_cols)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    f1_scorer = make_scorer(f1_score, zero_division=0)

    base_models = {
        "DecisionTree": DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(random_state=random_state, class_weight="balanced", n_jobs=1, max_features="sqrt", min_samples_split=2),
        "LinearSVC": LinearSVC(class_weight="balanced", random_state=random_state, max_iter=5000, dual="auto"),
        "MLPClassifier": MLPClassifier(max_iter=300, early_stopping=True, random_state=random_state),
    }

    results = {"dataset": dataset_name, "n_total": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
               "target": target_col, "models": {}, "grids_searched": GRIDS_CLASSIFICATION,
               "compute_settings": {"cv_folds": cv_folds, "bootstrap_resamples": n_boot, "seed": random_state, "scoring": "f1 (positive class)"}}

    fitted_pipes = {}
    preds = {}
    fig, axes = plt.subplots(1, len(base_models), figsize=(4.2 * len(base_models), 4))
    for ax, (name, clf) in zip(axes, base_models.items()):
        pipe = Pipeline([("preproc", preproc), ("clf", clf)])
        gs = GridSearchCV(pipe, GRIDS_CLASSIFICATION[name], scoring=f1_scorer, cv=cv, n_jobs=1, refit=True)
        gs.fit(X_train, y_train)
        best_pipe = gs.best_estimator_
        y_pred = best_pipe.predict(X_test)
        fitted_pipes[name] = best_pipe
        preds[name] = np.asarray(y_pred)

        m = {
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            "cv_best_f1": round(float(gs.best_score_), 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        }
        ci = bootstrap_classification_ci(y_test.values, y_pred, n_boot=n_boot, seed=random_state)
        m["bootstrap_95ci"] = ci
        results["models"][name] = m

        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name}\nF1={m['f1']:.3f}")

    fig.suptitle(f"{dataset_name}: confusion matrices (test set, n={len(X_test)})")
    plt.tight_layout(); plt.savefig(f"{FIGURES}/{dataset_name}_confusion_matrices.png", dpi=150); plt.close()

    dt_pipe = fitted_pipes["DecisionTree"]
    feat_names = list(dt_pipe.named_steps["preproc"].get_feature_names_out())
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt_pipe.named_steps["clf"], feature_names=feat_names,
              class_names=[str(c) for c in dt_pipe.named_steps["clf"].classes_],
              filled=True, max_depth=3, fontsize=8, ax=ax)
    ax.set_title(f"{dataset_name}: Decision Tree (best-CV params, first 3 levels shown)")
    plt.tight_layout(); plt.savefig(f"{FIGURES}/{dataset_name}_decision_tree.png", dpi=150); plt.close()
    results["decision_tree_structure"] = {"n_leaves": int(dt_pipe.named_steps["clf"].get_n_leaves()), "depth": int(dt_pipe.named_steps["clf"].get_depth())}

    names = list(base_models.keys())
    mcnemar_results = {}
    y_test_arr = y_test.values
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            mcnemar_results[f"{a}_vs_{b}"] = mcnemar_test(y_test_arr, preds[a], preds[b])
    results["mcnemar_pairwise"] = mcnemar_results

    results["runtime_seconds"] = round(time.time() - t0, 1)
    with open(f"{RESULTS}/{dataset_name}_classification_report.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    return results

import sys, json, time
sys.path.insert(0, "/home/claude/repo2/src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ch3_utils import mcnemar_test, RESULTS, FIGURES, GRIDS_CLASSIFICATION

names = ["DecisionTree", "RandomForest", "LinearSVC", "MLPClassifier"]
models = {}
preds = {}
y_test_ref = None
for name in names:
    with open(f"{RESULTS}/promotion_{name}_partial.json") as f:
        models[name] = json.load(f)
    yt = np.load(f"{RESULTS}/promotion_{name}_ytest.npy")
    yp = np.load(f"{RESULTS}/promotion_{name}_ypred.npy")
    preds[name] = yp
    if y_test_ref is None:
        y_test_ref = yt
    else:
        assert np.array_equal(y_test_ref, yt), "test sets differ between model runs!"

results = {
    "dataset": "promotion", "n_total": 54808, "n_train": 43846, "n_test": 10962,
    "target": "is_promoted", "models": models, "grids_searched": GRIDS_CLASSIFICATION,
    "compute_settings": {"cv_folds": 3, "bootstrap_resamples": 2000, "seed": 42, "scoring": "f1 (positive class)",
                          "note": "each model's grid search was run in a separate process to fit the single-CPU runtime budget; combined here"},
}

fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax, name in zip(axes, names):
    cm = confusion_matrix(y_test_ref, preds[name])
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
    ax.set_title(f"{name}\nF1={models[name]['f1']:.3f}")
fig.suptitle("promotion: confusion matrices (test set, n=10962)")
plt.tight_layout()
plt.savefig(f"{FIGURES}/promotion_confusion_matrices.png", dpi=150)
plt.close()

mcnemar_results = {}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        mcnemar_results[f"{a}_vs_{b}"] = mcnemar_test(y_test_ref, preds[a], preds[b])
results["mcnemar_pairwise"] = mcnemar_results

with open(f"{RESULTS}/promotion_classification_report.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

print("Combined report written.")
for name in names:
    print(name, models[name]["best_params"], {k: models[name][k] for k in ["accuracy","precision","recall","f1"]})
print("\nMcNemar:")
for k, v in mcnemar_results.items():
    print(" ", k, "p=", round(v["p_value"], 5), v["method"])

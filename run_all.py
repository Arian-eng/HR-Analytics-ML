from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from src.preprocessing import DATASETS, load_dataset, split_xy, make_preprocessor
from src.classification import MODELS, build_search
from src.clustering import run_kmeans
from src.evaluation import classification_metrics, mcnemar_pair
from src.green_hrm_analysis import main as run_ghrm

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
all_rows, predictions = [], {}

for name, info in DATASETS.items():
    df, info, path = load_dataset(name)
    X, y_raw = split_xy(df, info["target"], info["ids"])
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw.astype(str))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, (estimator, _) in MODELS.items():
        scaled = model_name in {"Linear SVM", "MLP"}
        prep = make_preprocessor(X_train, scale_numeric=scaled)
        search = build_search(model_name, estimator, prep, cv)
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        all_rows.append({"Dataset": name, "Model": model_name, **classification_metrics(y_test, pred),
                         "CV_Best_F1": float(search.best_score_), "Best_Params": json.dumps(search.best_params_, default=str)})
        predictions[(name, model_name)] = (y_test, pred)
        pd.DataFrame({"y_true": y_test, "y_pred": pred}).to_csv(out_dir / f"{model_name.replace(' ', '_')}_test_predictions.csv", index=False)
        pd.DataFrame(confusion_matrix(y_test, pred)).to_csv(out_dir / f"{model_name.replace(' ', '_')}_confusion_matrix.csv", index=False, header=False)

    pairs = [("Random Forest", "Decision Tree"), ("Random Forest", "Linear SVM"), ("Random Forest", "MLP")]
    mcnemar_rows = []
    for a, b in pairs:
        _, pred_a = predictions[(name, a)]
        _, pred_b = predictions[(name, b)]
        mcnemar_rows.append({"Dataset": name, "Model_A": a, "Model_B": b, **mcnemar_pair(y_test, pred_a, pred_b)})
    pd.DataFrame(mcnemar_rows).to_csv(out_dir / "mcnemar.csv", index=False, encoding="utf-8-sig")
    run_kmeans(df, info["target"], info["ids"], out_dir, FIGURES, name)

results = pd.DataFrame(all_rows)
results.to_csv(RESULTS / "model_metrics.csv", index=False, encoding="utf-8-sig")
results.to_excel(RESULTS / "model_metrics.xlsx", index=False)

# Chapter 3 also specifies a separate fourth GHRM dataset: continuous FEP regression
# and independent K-Means analysis. It is never merged with the three HR datasets.
run_ghrm()
print(results.round(4).to_string(index=False))

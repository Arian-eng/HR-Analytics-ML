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
from src.reporting import build_chapter4_outputs

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs"
FIGURES = ROOT / "figures"


def run_classification_analyses():
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    all_rows = []

    for name, info in DATASETS.items():
        print(f"[{name}] loading and validating data", flush=True)
        df, info, _ = load_dataset(name)
        X, y_raw = split_xy(df, info["target"], info["ids"])

        valid_target = y_raw.notna()
        X = X.loc[valid_target]
        y_raw = y_raw.loc[valid_target]
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw.astype(str))
        if len(encoder.classes_) != 2:
            raise ValueError(
                f"{name} target {info['target']!r} must contain exactly two classes; "
                f"found {encoder.classes_.tolist()}"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )
        out_dir = RESULTS / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "label_mapping.json").write_text(
            json.dumps(
                {str(label): int(code) for code, label in enumerate(encoder.classes_)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        predictions = {}
        for model_name, (estimator, _) in MODELS.items():
            print(f"[{name}] tuning {model_name}", flush=True)
            scaled = model_name in {"Linear SVM", "MLP"}
            prep = make_preprocessor(X_train, scale_numeric=scaled)
            search = build_search(model_name, estimator, prep, cv)
            search.fit(X_train, y_train)
            pred = search.predict(X_test)
            all_rows.append(
                {
                    "Dataset": name,
                    "Model": model_name,
                    **classification_metrics(y_test, pred),
                    "CV_Best_F1": float(search.best_score_),
                    "Best_Params": json.dumps(search.best_params_, default=str),
                }
            )
            predictions[model_name] = pred
            print(
                f"[{name}] {model_name} test F1="
                f"{all_rows[-1]['F1']:.4f}",
                flush=True,
            )
            pd.DataFrame({"y_true": y_test, "y_pred": pred}).to_csv(
                out_dir / f"{model_name.replace(' ', '_')}_test_predictions.csv",
                index=False,
            )
            pd.DataFrame(confusion_matrix(y_test, pred)).to_csv(
                out_dir / f"{model_name.replace(' ', '_')}_confusion_matrix.csv",
                index=False,
                header=False,
            )

        pairs = [
            ("Random Forest", "Decision Tree"),
            ("Random Forest", "Linear SVM"),
            ("Random Forest", "MLP"),
        ]
        mcnemar_rows = [
            {
                "Dataset": name,
                "Model_A": model_a,
                "Model_B": model_b,
                **mcnemar_pair(
                    y_test, predictions[model_a], predictions[model_b]
                ),
            }
            for model_a, model_b in pairs
        ]
        pd.DataFrame(mcnemar_rows).to_csv(
            out_dir / "mcnemar.csv", index=False, encoding="utf-8-sig"
        )
        print(f"[{name}] evaluating K-Means", flush=True)
        run_kmeans(
            df, info["target"], info["ids"], out_dir, FIGURES, name
        )

    return pd.DataFrame(all_rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    results = run_classification_analyses()
    results.to_csv(
        RESULTS / "model_metrics.csv", index=False, encoding="utf-8-sig"
    )
    results.to_excel(RESULTS / "model_metrics.xlsx", index=False)

    # Chapter 3 specifies a separate fourth GHRM dataset: continuous FEP
    # regression and independent K-Means analysis. It is never merged with the
    # three classification datasets.
    print("[ghrm] running regression and K-Means", flush=True)
    run_ghrm()
    print("[report] building GitHub Chapter 4 tables and figures", flush=True)
    build_chapter4_outputs()
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

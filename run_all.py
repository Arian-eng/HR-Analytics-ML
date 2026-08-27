"""Run the complete, reproducible thesis analysis from the four raw CSV files."""

from __future__ import annotations

import itertools
import json
import logging
import platform
import shutil
import sys
import time
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.classification import MODELS, build_search
from src.clustering import run_kmeans
from src.config import (
    CLASSIFICATION_MODELS,
    CV_FOLDS,
    DATASETS,
    GHRM_ITEMS,
    RANDOM_STATE,
    RESULTS_DIR,
    ROOT,
    TEST_SIZE,
)
from src.data_quality import write_data_profile
from src.evaluation import (
    bootstrap_classification_metrics,
    classification_metrics,
    mcnemar_pair,
)
from src.explainability import (
    permutation_table,
    slug,
    write_model_diagnostics,
)
from src.green_hrm_analysis import run_regression_analysis
from src.preprocessing import load_dataset, make_preprocessor, split_xy
from src.reporting import build_report


warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
    category=UserWarning,
)


def prepare_results():
    if RESULTS_DIR.name != "results" or RESULTS_DIR.parent != ROOT:
        raise RuntimeError(f"Refusing to replace unexpected path: {RESULTS_DIR}")
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True)


def configure_logging():
    logger = logging.getLogger("thesis")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(RESULTS_DIR / "run_log.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.handlers[:] = [file_handler, stream_handler]
    return logger


def cv_results_table(search):
    frame = pd.DataFrame(search.cv_results_)
    columns = [
        column
        for column in frame.columns
        if column.startswith("param_")
        or column
        in {
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score",
            "mean_fit_time",
        }
    ]
    return frame[columns].sort_values("rank_test_score")


def run_classification(logger):
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    metric_rows = []
    mcnemar_rows = []
    for dataset in ("ibm", "job_change", "promotion"):
        frame, info, _ = load_dataset(dataset)
        X, y_raw = split_xy(
            frame, info["target"], info["ids"], info.get("drop", [])
        )
        valid = y_raw.notna()
        X = X.loc[valid]
        y = y_raw.loc[valid].eq(info["positive_label"]).astype(int)
        train_index, test_index = train_test_split(
            X.index,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        split = pd.DataFrame(
            {
                "Row_Index": X.index,
                "Split": np.where(X.index.isin(test_index), "test", "train"),
                "Target": y,
            }
        )
        split_dir = RESULTS_DIR / "splits"
        split_dir.mkdir(exist_ok=True)
        split.to_csv(
            split_dir / f"{dataset}.csv", index=False, encoding="utf-8-sig"
        )
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        dataset_output = RESULTS_DIR / "classification" / dataset
        dataset_output.mkdir(parents=True, exist_ok=True)
        (dataset_output / "target_mapping.json").write_text(
            json.dumps(
                {"negative": 0, "positive": 1, "positive_source_label": info["positive_label"]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        predictions = {}
        logger.info(
            "%s: %s rows, %s train, %s test, positive share %.4f",
            dataset,
            len(X),
            len(train_index),
            len(test_index),
            y.mean(),
        )
        for model_name in CLASSIFICATION_MODELS:
            logger.info("%s: tuning %s", dataset, model_name)
            estimator, _ = MODELS[model_name]
            preprocessor = make_preprocessor(
                X_train, scale_numeric=model_name in {"Linear SVM", "MLP"}
            )
            search = build_search(model_name, estimator, preprocessor, cv)
            search.fit(X_train, y_train)
            prediction = search.predict(X_test)
            predictions[model_name] = prediction
            output = dataset_output / slug(model_name)
            output.mkdir(parents=True, exist_ok=True)

            point = classification_metrics(y_test, prediction)
            intervals = bootstrap_classification_metrics(y_test, prediction)
            row = {
                "Dataset": dataset,
                "Model": model_name,
                "Train_N": len(train_index),
                "Test_N": len(test_index),
                "Positive_Test_N": int(y_test.sum()),
                **point,
                **intervals,
                "CV_Best_F1": float(search.best_score_),
                "Best_Params": json.dumps(search.best_params_, default=str),
            }
            metric_rows.append(row)
            logger.info(
                "%s: %s F1=%.4f, accuracy=%.4f",
                dataset,
                model_name,
                point["F1"],
                point["Accuracy"],
            )

            identifier = info["ids"][0] if info["ids"] else None
            prediction_table = pd.DataFrame(
                {
                    "Row_Index": test_index,
                    "Record_ID": (
                        frame.loc[test_index, identifier].to_numpy()
                        if identifier
                        else test_index
                    ),
                    "y_true": y_test.to_numpy(),
                    "y_pred": prediction,
                    "Correct": y_test.to_numpy() == prediction,
                }
            )
            prediction_table.to_csv(
                output / "test_predictions.csv", index=False, encoding="utf-8-sig"
            )
            matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
            pd.DataFrame(
                matrix,
                index=["Actual_0", "Actual_1"],
                columns=["Predicted_0", "Predicted_1"],
            ).to_csv(output / "confusion_matrix.csv", encoding="utf-8-sig")
            cv_results_table(search).to_csv(
                output / "tuning_results.csv", index=False, encoding="utf-8-sig"
            )
            permutation_table(
                search.best_estimator_,
                X_test,
                y_test,
                "f1",
                dataset,
                model_name,
            ).to_csv(
                output / "permutation_importance.csv",
                index=False,
                encoding="utf-8-sig",
            )
            write_model_diagnostics(
                search.best_estimator_,
                output,
                dataset,
                model_name,
                class_names=["negative", "positive"],
            )

        for model_a, model_b in itertools.combinations(CLASSIFICATION_MODELS, 2):
            mcnemar_rows.append(
                {
                    "Dataset": dataset,
                    "Model_A": model_a,
                    "Model_B": model_b,
                    **mcnemar_pair(
                        y_test, predictions[model_a], predictions[model_b]
                    ),
                }
            )

        logger.info("%s: running K-Means k=2..7", dataset)
        run_kmeans(
            frame,
            info["target"],
            info["ids"],
            RESULTS_DIR / "clustering" / dataset,
            dataset,
            extra_drop=info.get("drop", []),
        )

    classification_root = RESULTS_DIR / "classification"
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        classification_root / "classification_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mcnemar = pd.DataFrame(mcnemar_rows)
    mcnemar.to_csv(
        classification_root / "mcnemar.csv", index=False, encoding="utf-8-sig"
    )
    return metrics, mcnemar


def run_ghrm_clustering(frame, logger):
    logger.info("ghrm: running K-Means k=2..7 on five GHRM constructs")
    raw_items = sorted({column for columns in GHRM_ITEMS.values() for column in columns})
    run_kmeans(
        frame,
        "FEP",
        DATASETS["ghrm"]["ids"],
        RESULTS_DIR / "clustering" / "ghrm",
        "ghrm",
        extra_drop=raw_items,
    )


def write_manifest(started, finished, duration, data_overview):
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "openpyxl"]
    manifest = {
        "started_utc": started,
        "finished_utc": finished,
        "duration_seconds": duration,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in packages},
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "classification_cv_folds": CV_FOLDS,
        "datasets": data_overview.to_dict(orient="records"),
        "command": "python run_all.py",
    }
    (RESULTS_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    started_wall = time.time()
    started = pd.Timestamp.now(tz="UTC").isoformat()
    prepare_results()
    logger = configure_logging()
    logger.info("final analysis started")
    data_overview, _ = write_data_profile()
    classification, mcnemar = run_classification(logger)
    ghrm_frame, regression, comparisons = run_regression_analysis()
    run_ghrm_clustering(ghrm_frame, logger)
    build_report(classification, mcnemar, regression, comparisons)
    finished = pd.Timestamp.now(tz="UTC").isoformat()
    duration = time.time() - started_wall
    write_manifest(started, finished, duration, data_overview)
    logger.info("final analysis completed in %.1f seconds", duration)


if __name__ == "__main__":
    main()

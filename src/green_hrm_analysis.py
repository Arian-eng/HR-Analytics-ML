"""Main Green HRM analysis: predict environmental performance (FEP)."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

from src.config import RANDOM_STATE, REGRESSION_MODELS, RESULTS_DIR, TEST_SIZE
from src.evaluation import (
    bootstrap_regression_difference,
    bootstrap_regression_metrics,
    regression_metrics,
)
from src.explainability import (
    permutation_table,
    slug,
    write_model_diagnostics,
)
from src.preprocessing import load_dataset


MODELS = {
    "Random Forest Regressor": (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": [1.0, "sqrt"],
        },
    ),
    "Decision Tree Regressor": (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {
            "model__max_depth": [3, 5, 8, None],
            "model__min_samples_leaf": [2, 5, 10],
        },
    ),
    "LinearSVR": (
        LinearSVR(random_state=RANDOM_STATE, max_iter=100_000),
        {
            "model__C": [0.1, 1, 10],
            "model__epsilon": [0.0, 0.1, 0.2],
        },
    ),
    "MLPRegressor": (
        MLPRegressor(
            random_state=RANDOM_STATE,
            max_iter=3_000,
            early_stopping=True,
            validation_fraction=0.15,
        ),
        {
            "model__hidden_layer_sizes": [(20,), (50,), (50, 25)],
            "model__alpha": [0.0001, 0.001],
            "model__learning_rate_init": [0.001, 0.01],
        },
    ),
}


def _preprocessor(features):
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
    )


def _search(model_name, estimator, grid, features):
    pipeline = Pipeline(
        [("preprocess", _preprocessor(features)), ("model", estimator)]
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    common = {
        "estimator": pipeline,
        "param_distributions" if model_name == "MLPRegressor" else "param_grid": grid,
        "scoring": "neg_root_mean_squared_error",
        "cv": cv,
        "n_jobs": -1,
        "refit": True,
        "return_train_score": True,
    }
    if model_name == "MLPRegressor":
        common.update({"n_iter": 8, "random_state": RANDOM_STATE})
        return RandomizedSearchCV(**common)
    return GridSearchCV(**common)


def _cv_table(search):
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


def run_regression_analysis():
    frame, _, _ = load_dataset("ghrm")
    valid = frame["FEP"].notna()
    frame = frame.loc[valid].copy()
    y = frame["FEP"]
    train_index, test_index = train_test_split(
        frame.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    split_dir = RESULTS_DIR / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Row_Index": frame.index,
            "Split": np.where(frame.index.isin(test_index), "test", "train"),
            "FEP": y,
        }
    ).to_csv(split_dir / "ghrm.csv", index=False, encoding="utf-8-sig")

    metric_rows = []
    all_predictions = {}
    for variant, features in {
        "Base": ["GRS", "GTD", "GPA", "GCM"],
        "GEE+": ["GRS", "GTD", "GPA", "GCM", "GEE"],
    }.items():
        X = frame[features]
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        for model_name in REGRESSION_MODELS:
            estimator, grid = MODELS[model_name]
            output = RESULTS_DIR / "regression" / slug(variant) / slug(model_name)
            output.mkdir(parents=True, exist_ok=True)
            search = _search(model_name, estimator, grid, features)
            with warnings.catch_warnings(record=True) as tuning_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                search.fit(X_train, y_train)
            tuning_convergence_warnings = sum(
                issubclass(item.category, ConvergenceWarning)
                for item in tuning_warnings
            )
            prediction = search.predict(X_test)
            point = regression_metrics(y_test, prediction)
            intervals = bootstrap_regression_metrics(y_test, prediction)

            repeated = RepeatedKFold(
                n_splits=5, n_repeats=5, random_state=RANDOM_STATE
            )
            with warnings.catch_warnings(record=True) as repeated_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                stability = cross_validate(
                    search.best_estimator_,
                    X,
                    y,
                    cv=repeated,
                    scoring={
                        "r2": "r2",
                        "rmse": "neg_root_mean_squared_error",
                        "mae": "neg_mean_absolute_error",
                    },
                    n_jobs=-1,
                )
            repeated_convergence_warnings = sum(
                issubclass(item.category, ConvergenceWarning)
                for item in repeated_warnings
            )
            stability_frame = pd.DataFrame(
                {
                    "Fold": np.arange(1, len(stability["test_r2"]) + 1),
                    "R2": stability["test_r2"],
                    "RMSE": -stability["test_rmse"],
                    "MAE": -stability["test_mae"],
                }
            )
            stability_frame.to_csv(
                output / "repeated_cv_metrics.csv", index=False, encoding="utf-8-sig"
            )
            row = {
                "Variant": variant,
                "Model": model_name,
                "Train_N": len(train_index),
                "Test_N": len(test_index),
                **point,
                **intervals,
                "Tuning_CV_RMSE": float(-search.best_score_),
                "Repeated_CV_R2_Mean": float(stability_frame["R2"].mean()),
                "Repeated_CV_R2_Std": float(stability_frame["R2"].std()),
                "Repeated_CV_RMSE_Mean": float(stability_frame["RMSE"].mean()),
                "Repeated_CV_RMSE_Std": float(stability_frame["RMSE"].std()),
                "Tuning_Convergence_Warnings": tuning_convergence_warnings,
                "Repeated_CV_Convergence_Warnings": repeated_convergence_warnings,
                "Best_Params": json.dumps(search.best_params_, default=str),
            }
            metric_rows.append(row)
            predictions = pd.DataFrame(
                {
                    "Row_Index": test_index,
                    "y_true": y_test.to_numpy(),
                    "y_pred": prediction,
                }
            )
            predictions.to_csv(
                output / "test_predictions.csv", index=False, encoding="utf-8-sig"
            )
            all_predictions[(variant, model_name)] = predictions
            _cv_table(search).to_csv(
                output / "tuning_results.csv", index=False, encoding="utf-8-sig"
            )
            importance = permutation_table(
                search.best_estimator_,
                X_test,
                y_test,
                "neg_root_mean_squared_error",
                "ghrm",
                f"{variant} {model_name}",
            )
            importance.to_csv(
                output / "permutation_importance.csv",
                index=False,
                encoding="utf-8-sig",
            )
            write_model_diagnostics(
                search.best_estimator_, output, "ghrm", model_name
            )
            (output / "convergence_warnings.json").write_text(
                json.dumps(
                    {
                        "tuning_convergence_warnings": tuning_convergence_warnings,
                        "repeated_cv_convergence_warnings": repeated_convergence_warnings,
                        "note": "Counts are recorded instead of suppressing model convergence warnings.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    comparison_rows = []
    for model_name in REGRESSION_MODELS:
        base = all_predictions[("Base", model_name)]
        plus = all_predictions[("GEE+", model_name)]
        if not np.array_equal(base["Row_Index"], plus["Row_Index"]):
            raise ValueError(f"Base and GEE+ rows differ for {model_name}")
        comparison_rows.append(
            {
                "Model": model_name,
                **bootstrap_regression_difference(
                    base["y_true"], base["y_pred"], plus["y_pred"]
                ),
            }
        )

    regression_root = RESULTS_DIR / "regression"
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        regression_root / "regression_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(
        regression_root / "base_vs_gee_plus.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_rows = []
    for (variant, model_name), predictions in all_predictions.items():
        residual = predictions["y_true"] - predictions["y_pred"]
        validation_rows.append(
            {
                "Variant": variant,
                "Model": model_name,
                "N": len(predictions),
                "Sum_Squared_Error": float((residual**2).sum()),
                "Sum_Absolute_Error": float(residual.abs().sum()),
                "Sum_Y": float(predictions["y_true"].sum()),
                "Sum_Y_Squared": float((predictions["y_true"] ** 2).sum()),
            }
        )
    pd.DataFrame(validation_rows).to_csv(
        regression_root / "regression_validation_aggregates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return frame, metrics, comparisons

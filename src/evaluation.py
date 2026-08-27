import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def classification_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def mcnemar_pair(y_true, pred_a, pred_b):
    a_correct = np.asarray(pred_a) == np.asarray(y_true)
    b_correct = np.asarray(pred_b) == np.asarray(y_true)
    b01 = int(np.sum(a_correct & ~b_correct))
    b10 = int(np.sum(~a_correct & b_correct))
    discordant = b01 + b10

    if discordant == 0:
        statistic, p_value, method = 0.0, 1.0, "no discordant pairs"
    elif discordant < 25:
        statistic = float("nan")
        p_value = binomtest(min(b01, b10), discordant, 0.5).pvalue
        method = "exact binomial"
    else:
        statistic = (abs(b01 - b10) - 1) ** 2 / discordant
        p_value = chi2.sf(statistic, df=1)
        method = "continuity-corrected chi-square"
    return {
        "b01": b01,
        "b10": b10,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "method": method,
    }


def _interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def bootstrap_classification_metrics(
    y_true, y_pred, n_resamples=2000, random_state=42, confidence=0.95
):
    """Paired non-parametric bootstrap intervals for held-out metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be same-length one-dimensional arrays")
    if y_true.size == 0:
        raise ValueError("bootstrap requires at least one held-out observation")

    rng = np.random.default_rng(random_state)
    samples = {name: [] for name in ("Accuracy", "Precision", "Recall", "F1")}
    for _ in range(n_resamples):
        index = rng.integers(0, y_true.size, size=y_true.size)
        metrics = classification_metrics(y_true[index], y_pred[index])
        for name, value in metrics.items():
            samples[name].append(value)

    result = {}
    for name, values in samples.items():
        lower, upper = _interval(values, confidence)
        result[f"{name}_CI_Lower"] = lower
        result[f"{name}_CI_Upper"] = upper
    return result


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def bootstrap_regression_metrics(
    y_true, y_pred, n_resamples=4000, random_state=42, confidence=0.95
):
    """Bootstrap intervals for R2, MAE, and RMSE on a held-out sample."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be same-length one-dimensional arrays")
    if y_true.size < 2:
        raise ValueError("regression bootstrap requires at least two observations")

    rng = np.random.default_rng(random_state)
    samples = {name: [] for name in ("R2", "MAE", "RMSE")}
    for _ in range(n_resamples):
        index = rng.integers(0, y_true.size, size=y_true.size)
        if np.unique(y_true[index]).size < 2:
            continue
        metrics = regression_metrics(y_true[index], y_pred[index])
        for name, value in metrics.items():
            samples[name].append(value)

    result = {}
    for name, values in samples.items():
        lower, upper = _interval(values, confidence)
        result[f"{name}_CI_Lower"] = lower
        result[f"{name}_CI_Upper"] = upper
    return result


def bootstrap_regression_difference(
    y_true,
    pred_base,
    pred_plus,
    n_resamples=4000,
    random_state=42,
    confidence=0.95,
):
    """Paired bootstrap for GEE+ minus Base metric differences."""

    y_true = np.asarray(y_true, dtype=float)
    pred_base = np.asarray(pred_base, dtype=float)
    pred_plus = np.asarray(pred_plus, dtype=float)
    if not (y_true.shape == pred_base.shape == pred_plus.shape) or y_true.ndim != 1:
        raise ValueError("all paired regression arrays must have the same 1-D shape")

    point_base = regression_metrics(y_true, pred_base)
    point_plus = regression_metrics(y_true, pred_plus)
    rng = np.random.default_rng(random_state)
    samples = {"Delta_R2": [], "Delta_MAE": [], "Delta_RMSE": []}
    for _ in range(n_resamples):
        index = rng.integers(0, y_true.size, size=y_true.size)
        if np.unique(y_true[index]).size < 2:
            continue
        base = regression_metrics(y_true[index], pred_base[index])
        plus = regression_metrics(y_true[index], pred_plus[index])
        for metric in ("R2", "MAE", "RMSE"):
            samples[f"Delta_{metric}"].append(plus[metric] - base[metric])

    result = {}
    for name, values in samples.items():
        metric = name.removeprefix("Delta_")
        result[name] = float(point_plus[metric] - point_base[metric])
        lower, upper = _interval(values, confidence)
        result[f"{name}_CI_Lower"] = lower
        result[f"{name}_CI_Upper"] = upper
    return result

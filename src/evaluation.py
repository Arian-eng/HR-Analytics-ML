import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
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
        statistic, p_value = 0.0, 1.0
    else:
        statistic = (abs(b01 - b10) - 1) ** 2 / discordant
        p_value = chi2.sf(statistic, df=1)
    return {
        "b01": b01,
        "b10": b10,
        "statistic": float(statistic),
        "p_value": float(p_value),
    }

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar


def classification_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Confusion_Matrix": json.dumps(confusion_matrix(y_true, y_pred).tolist())
    }


def mcnemar_pair(y_true, pred_a, pred_b):
    correct_a = np.asarray(pred_a) == np.asarray(y_true)
    correct_b = np.asarray(pred_b) == np.asarray(y_true)
    table = np.array([
        [np.sum(correct_a & correct_b), np.sum(correct_a & ~correct_b)],
        [np.sum(~correct_a & correct_b), np.sum(~correct_a & ~correct_b)]
    ])
    result = mcnemar(table, exact=True, correction=False)
    return {
        "b": int(table[0, 1]), "c": int(table[1, 0]),
        "statistic": float(result.statistic), "p_value": float(result.pvalue)
    }

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

def classification_metrics(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred), "Precision": precision_score(y_true, y_pred, zero_division=0), "Recall": recall_score(y_true, y_pred, zero_division=0), "F1": f1_score(y_true, y_pred, zero_division=0)}

def mcnemar_pair(y_true, pred_a, pred_b):
    a_correct = np.asarray(pred_a) == np.asarray(y_true)
    b_correct = np.asarray(pred_b) == np.asarray(y_true)
    table = [[int(np.sum(a_correct & b_correct)), int(np.sum(a_correct & ~b_correct))], [int(np.sum(~a_correct & b_correct)), int(np.sum(~a_correct & ~b_correct))]]
    result = mcnemar(table, exact=False, correction=True)
    return {"b01": table[0][1], "b10": table[1][0], "statistic": float(result.statistic), "p_value": float(result.pvalue)}

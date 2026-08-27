"""Shared configuration for the final thesis analysis."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
SILHOUETTE_SAMPLE_SIZE = 5_000


DATASETS = {
    "ibm": {
        "filename": "WA_Fn-UseC_-HR-Employee-Attrition (3).csv",
        "display_name": "IBM HR Analytics",
        "target": "Attrition",
        "positive_label": "Yes",
        "ids": ["EmployeeNumber"],
        "drop": ["EmployeeCount", "Over18", "StandardHours"],
        "task": "classification",
        "role": "supplementary HR analysis",
    },
    "job_change": {
        "filename": "aug_train.csv",
        "display_name": "Job Change",
        "target": "target",
        "positive_label": 1.0,
        "ids": ["enrollee_id"],
        "drop": [],
        "task": "classification",
        "role": "supplementary HR analysis",
    },
    "promotion": {
        "filename": "train_LZdllcl.csv",
        "display_name": "Employee Promotion",
        "target": "is_promoted",
        "positive_label": 1,
        "ids": ["employee_id"],
        "drop": [],
        "task": "classification",
        "role": "supplementary HR analysis",
    },
    "ghrm": {
        "filename": "HRM DATASETS.csv",
        "display_name": "GHRM - Environmental Performance",
        "target": "FEP",
        "ids": ["Timestamp"],
        "drop": [],
        "task": "regression",
        "role": "main Green HRM analysis",
    },
}


GHRM_ITEMS = {
    "GRS": ["GRS1", "GRS2", "GRS3", "GRS4"],
    "GTD": ["GTD1", "GTD2", "GTD3", "GTD4", "GTD5"],
    "GPA": ["GPA1", "GPA2", "GPA3", "GPA4", "GPA5", "GPA6"],
    "GCM": ["GCM1", "GCM2", "GCM3", "GCM4"],
    "GEE": ["GEE1", "GEE4", "GEE5", "GEE6"],
    "FEP": ["FEP1", "FEP5", "FEP7", "FEP9"],
}


CLASSIFICATION_MODELS = [
    "Random Forest",
    "Decision Tree",
    "Linear SVM",
    "MLP",
]

REGRESSION_MODELS = [
    "Random Forest Regressor",
    "Decision Tree Regressor",
    "LinearSVR",
    "MLPRegressor",
]

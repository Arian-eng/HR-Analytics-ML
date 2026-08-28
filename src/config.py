from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
VALIDATION_DIR = ROOT / "validation"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CLASSIFICATION_CV_FOLDS = 3
REGRESSION_CV_FOLDS = 5
CLASSIFICATION_BOOTSTRAPS = 2000
REGRESSION_BOOTSTRAPS = 4000
K_VALUES = range(2, 8)
KMEANS_N_INIT = 20
KMEANS_MAX_ITER = 300
SILHOUETTE_SAMPLE_SIZE = 5000

DATASETS = {
    "ibm": {
        "file": "WA_Fn-UseC_-HR-Employee-Attrition__3_.csv",
        "target": "Attrition",
        "positive": "Yes",
        "ids": ["EmployeeNumber"],
        "drop": ["EmployeeCount", "Over18", "StandardHours"],
        "role": "public HR benchmark/proxy; not direct Green HRM evidence",
    },
    "job_change": {
        "file": "aug_train.csv",
        "target": "target",
        "positive": 1.0,
        "ids": ["enrollee_id"],
        "drop": [],
        "role": "public HR benchmark/proxy; not direct Green HRM evidence",
    },
    "promotion": {
        "file": "train_LZdllcl.csv",
        "target": "is_promoted",
        "positive": 1,
        "ids": ["employee_id"],
        "drop": [],
        "role": "public HR benchmark/proxy; not direct Green HRM evidence",
    },
}

SURVEY_FILE = "HRM_DATASETS.csv"
SURVEY_CONSTRUCTS = {
    "GRS": ["GRS1", "GRS2", "GRS3", "GRS4"],
    "GTD": ["GTD1", "GTD2", "GTD3", "GTD4", "GTD5"],
    "GPA": ["GPA1", "GPA2", "GPA3", "GPA4", "GPA5", "GPA6"],
    "GCM": ["GCM1", "GCM2", "GCM3", "GCM4"],
    "GEE": ["GEE1", "GEE4", "GEE5", "GEE6"],
    "FEP": ["FEP1", "FEP5", "FEP7", "FEP9"],
}
BASE_FEATURES = ["GRS", "GTD", "GPA", "GCM"]
GEE_PLUS_FEATURES = BASE_FEATURES + ["GEE"]

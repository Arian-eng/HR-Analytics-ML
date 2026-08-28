import sys, time, json, os
sys.path.insert(0, "/home/claude/repo2/src")
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from ch3_utils import build_preprocessor, bootstrap_classification_ci, RNG_SEED, RESULTS

df = pd.read_csv("/home/claude/repo2/data/train_LZdllcl.csv")
numeric_cols = ["no_of_trainings", "age", "previous_year_rating", "length_of_service",
                 "KPIs_met >80%", "awards_won?", "avg_training_score"]
categorical_cols = ["department", "region", "education", "gender", "recruitment_channel"]
target_col = "is_promoted"

X = df[numeric_cols + categorical_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RNG_SEED, stratify=y)

preproc = build_preprocessor(numeric_cols, categorical_cols)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG_SEED)
f1_scorer = make_scorer(f1_score, zero_division=0)

MODEL_NAME = sys.argv[1]

GRIDS = {
    "DecisionTree": (DecisionTreeClassifier(random_state=RNG_SEED, class_weight="balanced"),
                      {"clf__max_depth": [6, 10, 15, None], "clf__min_samples_split": [2, 10]}),
    "RandomForest": (RandomForestClassifier(random_state=RNG_SEED, class_weight="balanced", n_jobs=1, max_features="sqrt", min_samples_split=2),
                      {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 15]}),
    "LinearSVC": (LinearSVC(class_weight="balanced", random_state=RNG_SEED, max_iter=5000, dual="auto"),
                   {"clf__C": [0.1, 1, 10]}),
    "MLPClassifier": (MLPClassifier(max_iter=300, early_stopping=True, random_state=RNG_SEED),
                        {"clf__hidden_layer_sizes": [(50,), (32, 16)], "clf__alpha": [0.0001, 0.001], "clf__learning_rate_init": [0.001, 0.01]}),
}

clf, grid = GRIDS[MODEL_NAME]
pipe = Pipeline([("preproc", preproc), ("clf", clf)])

t0 = time.time()
gs = GridSearchCV(pipe, grid, scoring=f1_scorer, cv=cv, n_jobs=1, refit=True)
gs.fit(X_train, y_train)
best_pipe = gs.best_estimator_
y_pred = best_pipe.predict(X_test)

m = {
    "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
    "cv_best_f1": round(float(gs.best_score_), 4),
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
    "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
    "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
}
ci = bootstrap_classification_ci(y_test.values, y_pred, n_boot=2000, seed=RNG_SEED)
m["bootstrap_95ci"] = ci
m["runtime_seconds"] = round(time.time() - t0, 1)

out_path = f"{RESULTS}/promotion_{MODEL_NAME}_partial.json"
with open(out_path, "w") as f:
    json.dump(m, f, ensure_ascii=False, indent=2, default=str)

# also save predictions for later McNemar + confusion matrix + n_test bookkeeping
np.save(f"{RESULTS}/promotion_{MODEL_NAME}_ytest.npy", y_test.values)
np.save(f"{RESULTS}/promotion_{MODEL_NAME}_ypred.npy", y_pred)

print(MODEL_NAME, m)
print("n_train", len(X_train), "n_test", len(X_test))

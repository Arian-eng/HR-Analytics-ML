from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

MODELS = {
    "Random Forest": (RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"), {
        "model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20],
        "model__max_features": ["sqrt", "log2"], "model__min_samples_split": [2, 5]
    }),
    "Decision Tree": (DecisionTreeClassifier(random_state=42, class_weight="balanced"), {
        "model__max_depth": [5, 10, 15, None], "model__criterion": ["gini", "entropy"],
        "model__min_samples_split": [2, 5], "model__min_samples_leaf": [1, 2]
    }),
    "Linear SVM": (LinearSVC(random_state=42, class_weight="balanced", max_iter=10000), {
        "model__C": [0.1, 1, 10]
    }),
    "MLP": (MLPClassifier(max_iter=500, early_stopping=True, random_state=42), {
        "model__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
        "model__alpha": [0.0001, 0.001], "model__learning_rate_init": [0.001, 0.01],
        "model__activation": ["relu", "tanh"]
    }),
}

def build_search(name, estimator, preprocessor, cv):
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    grid = MODELS[name][1]
    if name == "MLP":
        return RandomizedSearchCV(pipe, grid, n_iter=8, scoring="f1", cv=cv, random_state=42, n_jobs=-1, refit=True)
    return GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)

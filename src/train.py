import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

EXPRESSIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

def train(features_csv: str, model_out: str = "models/expression_clf.pkl"):
    df = pd.read_csv(features_csv)
    X = df.drop("label", axis=1).values
    y = df["label"].values

    candidates = {
        "svm_rbf":  Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True))]),
        "rf":       Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))]),
        "gb":       Pipeline([("sc", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1))]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_score, best_pipe = None, 0, None

    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        mean = scores.mean()
        print(f"{name}: {mean:.4f} ± {scores.std():.4f}")
        if mean > best_score:
            best_score, best_name, best_pipe = mean, name, pipe

    best_pipe.fit(X, y)
    joblib.dump(best_pipe, model_out)
    print(f"\nSaved {best_name} (f1={best_score:.4f}) → {model_out}")
"""
fix_model.py — Run this once to retrain and save a fresh expression_clf.pkl
Usage:  python src/fix_model.py
"""

import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

EXPRESSIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

BASES = {
    "angry":     [0.28, 0.28, 0.28, 0.05, 0.40, 0.38, 0.38, 0.38],
    "disgust":   [0.27, 0.27, 0.27, 0.08, 0.38, 0.36, 0.36, 0.36],
    "fear":      [0.35, 0.35, 0.35, 0.18, 0.48, 0.55, 0.55, 0.55],
    "happy":     [0.30, 0.30, 0.30, 0.12, 0.58, 0.42, 0.42, 0.42],
    "neutral":   [0.30, 0.30, 0.30, 0.04, 0.42, 0.44, 0.44, 0.44],
    "sad":       [0.26, 0.26, 0.26, 0.06, 0.36, 0.40, 0.40, 0.40],
    "surprised": [0.38, 0.38, 0.38, 0.28, 0.50, 0.58, 0.58, 0.58],
}

print("=" * 50)
print("  Face Expression Model — Retraining")
print("=" * 50)

rng = np.random.default_rng(42)
n_per_class = 400

X_list, y_list = [], []

for label, base in BASES.items():
    base_arr = np.array(base)
    noise    = rng.normal(0, 0.025, size=(n_per_class, len(base)))
    samples  = np.clip(base_arr + noise, 0.01, 1.0)
    samples[:, 2] = (samples[:, 0] + samples[:, 1]) / 2
    samples[:, 7] = (samples[:, 5] + samples[:, 6]) / 2
    X_list.append(samples)
    y_list.extend([label] * n_per_class)

X = np.vstack(X_list).astype(np.float32)
y = np.array(y_list)

print(f"\n  Training on {len(X)} samples x {X.shape[1]} features")
print(f"  Classes: {EXPRESSIONS}\n")

pipe = Pipeline([
    ("sc",  StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ))
])

pipe.fit(X, y)

scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro", n_jobs=-1)
print(f"  CV F1-macro: {scores.mean():.4f} (+/- {scores.std():.4f})")

models_dir = Path(__file__).resolve().parent.parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)
out_path = models_dir / "expression_clf.pkl"

joblib.dump(pipe, out_path, compress=3)

size_kb = os.path.getsize(out_path) / 1024
print(f"\n  Saved → {out_path}")
print(f"  File size: {size_kb:.1f} KB")
print("\n  Done! Now run:  python src/inference.py")
print("=" * 50)
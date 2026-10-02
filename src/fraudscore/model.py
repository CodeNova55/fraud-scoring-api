import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .data import FEATURES, generate_transactions

THRESHOLD = 0.5
DEFAULT_MODEL_PATH = "models/fraud_model.joblib"


def risk_level(probability):
    """Turn a fraud probability into a simple risk band."""
    if probability >= 0.7:
        return "high"
    if probability >= 0.3:
        return "medium"
    return "low"


def evaluate(model, X, y, threshold=THRESHOLD):
    """Measure the model on held-out data.

    Fraud is rare, so accuracy alone is misleading: `baseline_accuracy` is what
    you would get by always predicting "not fraud". Look at precision, recall,
    and average precision instead.
    """
    y = np.asarray(y)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "samples": int(len(y)),
        "fraud_rate": round(float(np.mean(y)), 4),
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y, probabilities)), 4),
        "precision": round(float(precision_score(y, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y, predictions, zero_division=0)), 4),
        "accuracy": round(float(np.mean(predictions == y)), 4),
        "baseline_accuracy": round(float(1 - np.mean(y)), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def train_model(n=20000, fraud_rate=0.03, seed=42):
    """Generate data, train a random forest, and evaluate it on a held-out split."""
    data = generate_transactions(n, fraud_rate, seed)
    X = data[FEATURES]
    y = data["is_fraud"]
    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        raise ValueError("Need at least two fraud and two normal transactions to train")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    metrics["train_samples"] = int(len(y_train))
    return {"model": model, "metrics": metrics, "features": FEATURES}


def score(bundle, transaction):
    """Score one transaction given as a dict with the feature names as keys."""
    row = pd.DataFrame([[transaction[name] for name in FEATURES]], columns=FEATURES)
    probability = float(bundle["model"].predict_proba(row)[0, 1])
    return {
        "fraud_probability": round(probability, 4),
        "risk_level": risk_level(probability),
        "flagged": probability >= THRESHOLD,
    }


def model_path():
    return Path(os.environ.get("FRAUD_MODEL_PATH", DEFAULT_MODEL_PATH))


def save_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path):
    """Load a saved model. Only load files you trained yourself (they are pickles)."""
    return joblib.load(path)


def load_or_train(path=None):
    """Load the saved model, or train and save one if none exists yet."""
    path = Path(path) if path else model_path()
    if path.exists():
        return load_bundle(path)
    bundle = train_model()
    save_bundle(bundle, path)
    return bundle

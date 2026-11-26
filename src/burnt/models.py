"""Lightweight pixel classifiers: LightGBM (primary), Random Forest (secondary), Logistic Regression (tiny)."""
from __future__ import annotations

import pickle
import random

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def make(model, params, seed, n_jobs=8):
    params = dict(params or {})
    if model == "lgbm":
        params.pop("n_estimators_best", None)
        return lgb.LGBMClassifier(objective="binary", random_state=seed, n_jobs=n_jobs, verbose=-1,
                                  deterministic=True, force_row_wise=True, **params)
    if model == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **params)
    if model == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed, **params))
    raise ValueError(model)


def fit(model, params, X, y, seed, X_val=None, y_val=None, early_stopping=100, n_jobs=8):
    set_seed(seed)
    clf = make(model, params, seed, n_jobs)
    if model == "lgbm" and X_val is not None:
        clf.fit(X, y, eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
    else:
        clf.fit(X, y)
    return clf


def predict_proba(clf, X, batch=2_000_000):
    out = np.empty(len(X), np.float32)
    for i in range(0, len(X), batch):
        out[i:i + batch] = clf.predict_proba(X.iloc[i:i + batch] if hasattr(X, "iloc") else X[i:i + batch])[:, 1]
    return out


def save(clf, path):
    with open(path, "wb") as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def complexity(clf):
    """Number of trees / leaves / parameters, for the cost table."""
    if isinstance(clf, lgb.LGBMClassifier):
        d = clf.booster_.dump_model()
        leaves = sum(t["num_leaves"] for t in d["tree_info"])
        return {"trees": len(d["tree_info"]), "leaves": leaves, "params": 2 * leaves - len(d["tree_info"])}
    if isinstance(clf, RandomForestClassifier):
        nodes = sum(e.tree_.node_count for e in clf.estimators_)
        leaves = sum(e.tree_.n_leaves for e in clf.estimators_)
        return {"trees": len(clf.estimators_), "leaves": leaves, "params": nodes}
    lr = clf[-1]
    return {"trees": 0, "leaves": 0, "params": int(lr.coef_.size + lr.intercept_.size)}


def best_threshold(prob, y, grid=None):
    """Decision threshold maximising F1 on validation probabilities (grid 0.05..0.95)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    y = np.asarray(y, bool)
    pos = np.sort(prob[y]); neg = np.sort(prob[~y])
    tp = len(pos) - np.searchsorted(pos, grid, side="right")
    fp = len(neg) - np.searchsorted(neg, grid, side="right")
    fn = len(pos) - tp
    f1 = np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0)
    i = int(np.argmax(f1))
    return float(grid[i]), float(f1[i])

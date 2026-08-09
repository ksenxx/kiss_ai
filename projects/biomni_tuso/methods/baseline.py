"""Naive baseline method.

A single linear / logistic model on the RAW features with no engineering. This
is intentionally weak: it establishes that the benchmarks are non-trivial (a
naive approach scores well below 99), so any high score reported by the evolved
method reflects genuine method development rather than a degenerate benchmark.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


def _is_classification(y: np.ndarray) -> bool:
    """Detect a binary-label target."""
    uniq = np.unique(y)
    return uniq.size <= 2 and set(uniq.tolist()) <= {0.0, 1.0}


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Train a plain linear model on raw features and predict ``x_eval``."""
    scaler = StandardScaler().fit(x_train)
    xt = scaler.transform(x_train)
    xe = scaler.transform(x_eval)
    if _is_classification(y_train):
        clf = LogisticRegression(max_iter=1000).fit(xt, y_train)
        return clf.predict_proba(xe)[:, 1]
    reg = LinearRegression().fit(xt, y_train)
    return reg.predict(xe)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

def fit_logistic_calibrator(
    S: np.ndarray,
    m: np.ndarray,
    y: np.ndarray,
    l2_C: float = 1.0,
    max_iter: int = 2000,
) -> Tuple[float, float, float]:
    """Fit logistic reg: p = sigmoid(b0 + b1*S + b2*m).
    Returns (b0,b1,b2).
    """
    from sklearn.linear_model import LogisticRegression
    X = np.stack([S, m], axis=1)
    clf = LogisticRegression(C=float(l2_C), max_iter=int(max_iter), solver="lbfgs")
    clf.fit(X, y.astype(int))
    b0 = float(clf.intercept_[0])
    b1 = float(clf.coef_[0, 0])
    b2 = float(clf.coef_[0, 1])
    return b0, b1, b2

def fit_isotonic_calibrator(
    score: np.ndarray, y: np.ndarray
):
    """Monotone calibration mapping score -> probability using isotonic regression."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(score, y.astype(float))
    return ir

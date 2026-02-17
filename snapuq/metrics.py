from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true.astype(int), y_score.astype(float)))

def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true.astype(int), y_score.astype(float)))

def risk_coverage_curve(y_correct: np.ndarray, y_score: np.ndarray, n_points: int = 101) -> Dict[str, np.ndarray]:
    """Risk-coverage where y_score higher means *more uncertain* (worse)."""
    y_score = y_score.astype(float)
    y_correct = y_correct.astype(bool)
    qs = np.linspace(0.0, 1.0, n_points)
    cov = []
    risk = []
    for q in qs:
        thr = np.quantile(y_score, q)
        keep = y_score <= thr
        coverage = keep.mean()
        if keep.sum() == 0:
            cov.append(0.0)
            risk.append(0.0)
        else:
            cov.append(float(coverage))
            risk.append(float((~y_correct[keep]).mean()))
    return {"quantiles": qs, "coverage": np.array(cov), "risk": np.array(risk)}

def detection_delay(event_mask: np.ndarray, alarm_mask: np.ndarray) -> Optional[int]:
    """Return delay in timesteps from first event to first alarm during event.
    Returns None if no alarm during event.
    """
    event_idx = np.where(event_mask)[0]
    if len(event_idx) == 0:
        return None
    start = event_idx[0]
    alarm_idx = np.where(alarm_mask & (np.arange(len(alarm_mask)) >= start))[0]
    alarm_idx = [i for i in alarm_idx if event_mask[i]]
    if len(alarm_idx) == 0:
        return None
    return int(alarm_idx[0] - start)

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    y_true = y_true.astype(int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        # Only one class present — AUPRC is undefined; return 0.0 as sentinel
        return 0.0
    return float(average_precision_score(y_true, y_score.astype(float)))

def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    y_true = y_true.astype(int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5  # undefined — return chance level
    return float(roc_auc_score(y_true, y_score.astype(float)))

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
    """Return delay in timesteps from first event to first alarm *during* the event window.
    Returns None if no alarm occurs during the event.
    """
    event_idxs = np.where(event_mask)[0]
    if len(event_idxs) == 0:
        return None
    start = int(event_idxs[0])
    # Find alarms that occur at or after the event start AND within the event window
    during_event = event_mask & alarm_mask & (np.arange(len(alarm_mask)) >= start)
    alarm_during = np.where(during_event)[0]
    if len(alarm_during) == 0:
        return None
    return int(alarm_during[0] - start)

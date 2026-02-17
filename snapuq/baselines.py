from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def maxsoftmax_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return 1.0 - probs.max(dim=-1).values

@torch.no_grad()
def entropy_uncertainty(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1).clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)

def temperature_scale_logits(logits: torch.Tensor, T: float) -> torch.Tensor:
    return logits / float(max(1e-6, T))

def odin_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    T: float = 1000.0,
    eps: float = 0.001,
) -> torch.Tensor:
    """ODIN-style: input perturbation to increase max-softmax separation.
    Returns uncertainty = 1 - maxsoftmax(perturbed_logits/T).
    """
    x = x.detach().clone().requires_grad_(True)
    logits = model(x)
    logits_T = logits / float(T)
    y = logits_T.argmax(dim=-1)
    loss = F.cross_entropy(logits_T, y)
    loss.backward()
    grad = x.grad.detach().sign()
    x_pert = (x - eps * grad).detach()
    logits2 = model(x_pert) / float(T)
    return maxsoftmax_uncertainty(logits2)

@torch.no_grad()
def mahalanobis_uncertainty(
    feats: torch.Tensor,
    class_means: torch.Tensor,
    inv_cov: torch.Tensor,
) -> torch.Tensor:
    """Mahalanobis distance in feature space; returns min distance across classes."""
    # feats: [B,D], class_means: [C,D], inv_cov: [D,D]
    diff = feats[:, None, :] - class_means[None, :, :]
    # [B,C,D] @ [D,D] -> [B,C,D]
    md = torch.einsum("bcd,dd->bcd", diff, inv_cov)
    dist2 = (md * diff).sum(dim=-1)  # [B,C]
    return dist2.min(dim=-1).values

def fit_class_stats(
    feats: np.ndarray,
    y: np.ndarray,
    reg: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit class means and shared covariance inverse (Ledoit-style shrinkage)."""
    feats = feats.astype(np.float64)
    y = y.astype(int)
    C = int(y.max() + 1)
    means = np.stack([feats[y == c].mean(axis=0) for c in range(C)], axis=0)
    centered = feats - means[y]
    cov = centered.T @ centered / max(1, len(feats) - 1)
    cov = cov + reg * np.eye(cov.shape[0])
    inv = np.linalg.inv(cov)
    return means.astype(np.float32), inv.astype(np.float32)

def mc_dropout_uncertainty(model: nn.Module, x: torch.Tensor, n: int = 10) -> torch.Tensor:
    """MC Dropout predictive entropy. Assumes model has dropout layers."""
    model.train()  # enable dropout
    probs = []
    for _ in range(n):
        logits = model(x)
        probs.append(torch.softmax(logits, dim=-1))
    p = torch.stack(probs, dim=0).mean(dim=0)
    model.eval()
    return -(p * (p + 1e-12).log()).sum(dim=-1)

def ensemble_uncertainty(models: List[nn.Module], x: torch.Tensor) -> torch.Tensor:
    """Ensemble predictive entropy."""
    probs = []
    for m in models:
        m.eval()
        with torch.no_grad():
            probs.append(torch.softmax(m(x), dim=-1))
    p = torch.stack(probs, dim=0).mean(dim=0)
    return -(p * (p + 1e-12).log()).sum(dim=-1)

def hydra_multihead_uncertainty(head_logits: List[torch.Tensor]) -> torch.Tensor:
    """HYDRA-like: disagreement across multiple lightweight heads (predictive entropy of mean prob)."""
    probs = [torch.softmax(l, dim=-1) for l in head_logits]
    p = torch.stack(probs, dim=0).mean(dim=0)
    return -(p * (p + 1e-12).log()).sum(dim=-1)

def qute_threshold(score: np.ndarray, q: float) -> float:
    """QUTE-style threshold by score quantile on ID dev set."""
    return float(np.quantile(score, q))

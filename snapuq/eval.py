from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .metrics import auprc, auroc, risk_coverage_curve

@dataclass
class EvalConfig:
    batch_size: int = 256
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def collect_scores(snapuq_model, ds, cfg: EvalConfig):
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    device = torch.device(cfg.device)
    snapuq_model.to(device)
    snapuq_model.eval()

    all_logits = []
    all_S = []
    all_m = []
    all_U = []
    all_y = []
    for x, y in loader:
        x = x.to(device)
        out = snapuq_model(x, return_uq=True)
        logits = out.logits.detach().cpu()
        all_logits.append(logits)
        all_S.append(out.S.detach().cpu())
        all_m.append(out.m.detach().cpu())
        all_y.append(y.cpu())
        if out.U is not None:
            all_U.append(out.U.detach().cpu())
    logits = torch.cat(all_logits, dim=0).numpy()
    S = torch.cat(all_S, dim=0).numpy()
    m = torch.cat(all_m, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()
    U = torch.cat(all_U, dim=0).numpy() if len(all_U) else None
    return {"logits": logits, "S": S, "m": m, "U": U, "y": y}

def eval_ood_detection(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    return {"auprc": auprc(y_true, y_score), "auroc": auroc(y_true, y_score)}

def eval_selective_risk(logits: np.ndarray, y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    pred = logits.argmax(axis=1)
    correct = (pred == y)
    rc = risk_coverage_curve(correct, score, n_points=101)
    # area under risk-coverage (lower better); simple trapezoid
    auc = float(np.trapz(rc["risk"], rc["coverage"]))
    return {"aurc": auc}

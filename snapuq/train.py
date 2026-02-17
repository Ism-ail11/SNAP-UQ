from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils.torchutil import AverageMeter, accuracy_top1
from .utils.io import ensure_dir, save_json

@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 0.01
    weight_decay: float = 5e-4
    batch_size: int = 128
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 50

def train_classifier(
    model: nn.Module,
    train_ds,
    val_ds,
    out_dir: str,
    cfg: TrainConfig,
) -> Dict[str, float]:
    out = ensure_dir(out_dir)
    device = torch.device(cfg.device)
    model.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    best_acc = -1.0
    best_path = Path(out) / "backbone_best.pt"
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        loss_m = AverageMeter()
        acc_m = AverageMeter()
        for it, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_m.update(float(loss.item()), n=len(x))
            acc_m.update(float(accuracy_top1(logits.detach(), y).item()), n=1)
        sched.step()

        # val
        model.eval()
        accs = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                accs.append(float(accuracy_top1(logits, y).item()))
        val_acc = float(np.mean(accs)) if len(accs) else 0.0

        history.append({"epoch": epoch, "train_loss": loss_m.avg, "train_acc": acc_m.avg, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, best_path)

    save_json({"best_val_acc": best_acc, "history": history}, Path(out) / "train_backbone.json")
    return {"best_val_acc": best_acc, "best_ckpt": str(best_path)}

def train_snapuq_posthoc(
    snapuq_model,
    train_ds,
    out_dir: str,
    cfg: TrainConfig,
    lambda_ss: float = 1.0,
    lambda_var_l1: float = 0.0,
) -> Dict[str, float]:
    """Freeze backbone; train only SNAP-UQ predictors using self-supervised Gaussian NLL."""
    out = ensure_dir(out_dir)
    device = torch.device(cfg.device)
    snapuq_model.to(device)
    snapuq_model.backbone.eval()
    for p in snapuq_model.backbone.parameters():
        p.requires_grad_(False)

    params = list(snapuq_model.predictors.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    best_loss = 1e9
    best_path = Path(out) / "snapuq_best.pt"
    history = []

    for epoch in range(cfg.epochs):
        snapuq_model.train()
        loss_m = AverageMeter()
        for x, _y in loader:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            loss = snapuq_model.gaussian_nll_loss(x) * float(lambda_ss)

            # Optional L1 on log(var) via mean log(var)
            if lambda_var_l1 > 0.0:
                reg = 0.0
                for pred in snapuq_model.predictors:
                    # pred.head outputs raw -> var; approximate with weight magnitude
                    reg = reg + pred.head.weight.abs().mean()
                loss = loss + float(lambda_var_l1) * reg

            loss.backward()
            opt.step()
            loss_m.update(float(loss.item()), n=len(x))

        history.append({"epoch": epoch, "loss": loss_m.avg})
        if loss_m.avg < best_loss:
            best_loss = loss_m.avg
            torch.save({"predictors": snapuq_model.predictors.state_dict(), "epoch": epoch, "loss": best_loss}, best_path)

    save_json({"best_loss": best_loss, "history": history}, Path(out) / "train_snapuq.json")
    return {"best_loss": best_loss, "best_ckpt": str(best_path)}

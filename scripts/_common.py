from __future__ import annotations
from pathlib import Path
import argparse
import torch

from snapuq.utils.config import load_yaml
from snapuq.utils.seed import seed_everything
from snapuq.models.factory import build_backbone
from snapuq.snapuq import SnapUQ, LogisticCalibrator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="runs")
    return p.parse_args()

def build_from_config(cfg, device: str):
    backbone, meta = build_backbone(cfg)
    # Load optional backbone ckpt
    ckpt_path = cfg.get("paths", {}).get("backbone_ckpt", None)
    if ckpt_path:
        d = torch.load(ckpt_path, map_location="cpu")
        backbone.load_state_dict(d["model"], strict=True)

    # Resolve tap names
    taps = cfg.get("snapuq", {}).get("taps", None) or meta.get("default_taps", None)
    if taps is None:
        raise ValueError("No taps specified and no default_taps available.")
    # Find modules by name
    name_to_module = dict(backbone.named_modules())
    tap_modules = []
    for name in taps:
        if name not in name_to_module:
            raise KeyError(f"Tap module name '{name}' not found in backbone. Available keys include: "
                           f"{list(name_to_module.keys())[:20]} ...")
        tap_modules.append(name_to_module[name])

    suq = SnapUQ(
        backbone=backbone,
        tap_modules=tap_modules,
        rank=int(cfg.get("snapuq", {}).get("rank", 16)),
        tap_weights=cfg.get("snapuq", {}).get("tap_weights", None),
        alpha=float(cfg.get("snapuq", {}).get("alpha", 0.5)),
        var_floor=float(cfg.get("snapuq", {}).get("var_floor", 1e-4)),
    )
    # infer dims
    example_shape = cfg.get("data", {}).get("example_shape", None)
    if example_shape is None:
        raise ValueError("data.example_shape must be set (e.g., [1,3,32,32]).")
    x0 = torch.zeros(*example_shape, dtype=torch.float32)
    suq.infer_dims(x0)

    # load optional snapuq predictors
    suq_ckpt = cfg.get("paths", {}).get("snapuq_ckpt", None)
    if suq_ckpt:
        d = torch.load(suq_ckpt, map_location="cpu")
        suq.predictors.load_state_dict(d.get("predictors", d), strict=True)

    # load optional calibrator params
    cal = cfg.get("paths", {}).get("calibrator_json", None)
    if cal and Path(cal).exists():
        import json
        j = json.loads(Path(cal).read_text(encoding="utf-8"))
        suq.set_calibrator(LogisticCalibrator(j["b0"], j["b1"], j["b2"]))

    suq.to(torch.device(device))
    return suq

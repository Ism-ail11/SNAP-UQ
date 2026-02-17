from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from snapuq.utils.config import load_yaml
from snapuq.utils.seed import seed_everything
from snapuq.data.vision import VisionDataConfig, build_vision_dataset
from snapuq.data.audio import SpeechCommandsWrapper
from snapuq.eval import EvalConfig, collect_scores, eval_ood_detection, eval_selective_risk
from snapuq.utils.io import save_json
from scripts._common import build_from_config

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(args.seed)
    device = cfg.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    suq = build_from_config(cfg, device=device)
    ec = EvalConfig(device=device, batch_size=int(cfg["eval"].get("batch_size", 256)))

    # ID eval
    ds_name = cfg["data"]["dataset"]
    if ds_name in ("mnist", "cifar10", "tinyimagenet"):
        id_ds = build_vision_dataset(VisionDataConfig(dataset=ds_name, root=cfg["data"]["root"], train=False))
    elif ds_name == "speechcommands":
        id_ds = SpeechCommandsWrapper(root=cfg["data"]["root"], subset="testing")
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    id_scores = collect_scores(suq, id_ds, ec)
    logits = id_scores["logits"]
    y = id_scores["y"]
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())

    # If U exists, use it; else use S as uncertainty score
    score = id_scores["U"][:, 0] if (id_scores["U"] is not None and id_scores["U"].ndim > 1) else (id_scores["U"] if id_scores["U"] is not None else id_scores["S"])
    sel = eval_selective_risk(logits, y, score)

    out = {"id_acc": acc, **sel}

    # Optional: OOD dataset path specified in config
    ood_cfg = cfg.get("eval", {}).get("ood", None)
    if ood_cfg:
        ood_ds_name = ood_cfg.get("dataset", None)
        if ood_ds_name:
            if ood_ds_name in ("mnist", "cifar10", "tinyimagenet"):
                ood_ds = build_vision_dataset(VisionDataConfig(dataset=ood_ds_name, root=cfg["data"]["root"], train=False))
            elif ood_ds_name == "speechcommands":
                ood_ds = SpeechCommandsWrapper(root=cfg["data"]["root"], subset="testing")
            else:
                raise ValueError(f"Unknown ood dataset: {ood_ds_name}")
            ood_scores = collect_scores(suq, ood_ds, ec)
            ood_score = ood_scores["U"][:, 0] if (ood_scores["U"] is not None and ood_scores["U"].ndim > 1) else (ood_scores["U"] if ood_scores["U"] is not None else ood_scores["S"])
            out["ood"] = eval_ood_detection(score, ood_score)

    out_dir = Path(args.out_dir) / cfg["exp"]["name"] / "results"
    save_json(out, out_dir / "offline_metrics.json")
    print(out)

if __name__ == "__main__":
    main()

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from snapuq.utils.config import load_yaml
from snapuq.utils.seed import seed_everything
from snapuq.data.vision import VisionDataConfig, build_vision_dataset
from snapuq.data.audio import SpeechCommandsWrapper
from snapuq.eval import EvalConfig, collect_scores
from snapuq.calibration import fit_logistic_calibrator
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

    ds_name = cfg["data"]["dataset"]
    if ds_name in ("mnist", "cifar10", "tinyimagenet"):
        dev_ds = build_vision_dataset(VisionDataConfig(dataset=ds_name, root=cfg["data"]["root"], train=False))
    elif ds_name == "speechcommands":
        dev_ds = SpeechCommandsWrapper(root=cfg["data"]["root"], subset="validation")
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    scores = collect_scores(suq, dev_ds, EvalConfig(device=device, batch_size=int(cfg["eval"].get("batch_size", 256))))
    logits = scores["logits"]
    y = scores["y"]
    pred = logits.argmax(axis=1)
    # Calibration target: 1 if incorrect (proxy for 'should abstain')
    y_cal = (pred != y).astype(np.int32)

    b0, b1, b2 = fit_logistic_calibrator(scores["S"], scores["m"], y_cal, l2_C=float(cfg["calib"].get("C", 1.0)))
    out_dir = Path(args.out_dir) / cfg["exp"]["name"] / "calib"
    save_json({"b0": b0, "b1": b1, "b2": b2}, out_dir / "calibrator.json")
    print({"b0": b0, "b1": b1, "b2": b2, "saved_to": str(out_dir / "calibrator.json")})

if __name__ == "__main__":
    main()

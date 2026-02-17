from __future__ import annotations
from pathlib import Path
import torch

from snapuq.utils.config import load_yaml
from snapuq.utils.seed import seed_everything
from snapuq.data.vision import VisionDataConfig, build_vision_dataset
from snapuq.data.audio import SpeechCommandsWrapper
from snapuq.train import TrainConfig, train_snapuq_posthoc
from scripts._common import build_from_config

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["posthoc"], default="posthoc")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(args.seed)

    device = cfg.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    suq = build_from_config(cfg, device=device)

    ds_name = cfg["data"]["dataset"]
    if ds_name in ("mnist", "cifar10", "tinyimagenet"):
        train_ds = build_vision_dataset(VisionDataConfig(dataset=ds_name, root=cfg["data"]["root"], train=True))
    elif ds_name == "speechcommands":
        train_ds = SpeechCommandsWrapper(root=cfg["data"]["root"], subset="training")
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    out_dir = Path(args.out_dir) / cfg["exp"]["name"] / "snapuq"
    tc = TrainConfig(
        epochs=int(cfg["snapuq_train"]["epochs"]),
        lr=float(cfg["snapuq_train"]["lr"]),
        weight_decay=float(cfg["snapuq_train"].get("weight_decay", 0.0)),
        batch_size=int(cfg["snapuq_train"]["batch_size"]),
        num_workers=int(cfg["snapuq_train"].get("num_workers", 2)),
        device=str(device),
    )
    res = train_snapuq_posthoc(
        suq, train_ds, str(out_dir), tc,
        lambda_ss=float(cfg["snapuq_train"].get("lambda_ss", 1.0)),
        lambda_var_l1=float(cfg["snapuq_train"].get("lambda_var_l1", 0.0)),
    )
    print(res)
    (out_dir / "snapuq_best_path.txt").write_text(res["best_ckpt"], encoding="utf-8")

if __name__ == "__main__":
    main()

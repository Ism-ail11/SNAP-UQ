from __future__ import annotations
from pathlib import Path
import torch

from snapuq.data.vision import VisionDataConfig, build_vision_dataset
from snapuq.data.audio import SpeechCommandsWrapper
from snapuq.train import TrainConfig, train_classifier
from snapuq.utils.config import load_yaml
from snapuq.utils.seed import seed_everything
from snapuq.models.factory import build_backbone

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

    model, _meta = build_backbone(cfg)

    ds_name = cfg["data"]["dataset"]
    if ds_name in ("mnist", "cifar10", "tinyimagenet"):
        train_ds = build_vision_dataset(VisionDataConfig(dataset=ds_name, root=cfg["data"]["root"], train=True))
        val_ds = build_vision_dataset(VisionDataConfig(dataset=ds_name, root=cfg["data"]["root"], train=False))
    elif ds_name == "speechcommands":
        train_ds = SpeechCommandsWrapper(root=cfg["data"]["root"], subset="training")
        val_ds = SpeechCommandsWrapper(root=cfg["data"]["root"], subset="validation")
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    out_dir = Path(args.out_dir) / cfg["exp"]["name"] / "backbone"
    tc = TrainConfig(
        epochs=int(cfg["train"]["epochs"]),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 5e-4)),
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"].get("num_workers", 2)),
        device=str(device),
    )
    res = train_classifier(model, train_ds, val_ds, str(out_dir), tc)
    print(res)
    # write path for convenience
    (out_dir / "backbone_best_path.txt").write_text(res["best_ckpt"], encoding="utf-8")

if __name__ == "__main__":
    main()

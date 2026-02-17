from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from snapuq.utils.config import load_yaml
from snapuq.utils.seed import seed_everything
from snapuq.eval import EvalConfig
from snapuq.metrics import auprc, detection_delay
from snapuq.streaming import build_monotone_severity_stream, rolling_mean
from snapuq.utils.io import save_json
from scripts._common import build_from_config

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=100)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(args.seed)
    device = cfg.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    suq = build_from_config(cfg, device=device)
    ec = EvalConfig(device=device, batch_size=int(cfg["eval"].get("batch_size", 256)))

    # Stream dataset is formed by reusing ID test set; corruption/ood injection is implemented in your custom stream runner.
    # Here we demonstrate the evaluation mechanics on an ID-only stream: event labels are based on rolling accuracy drop.

    # Load ID test set
    ds_name = cfg["data"]["dataset"]
    if ds_name in ("mnist", "cifar10", "tinyimagenet"):
        from snapuq.data.vision import VisionDataConfig, build_vision_dataset
        test_ds = build_vision_dataset(VisionDataConfig(dataset=ds_name, root=cfg["data"]["root"], train=False))
    elif ds_name == "speechcommands":
        from snapuq.data.audio import SpeechCommandsWrapper
        test_ds = SpeechCommandsWrapper(root=cfg["data"]["root"], subset="testing")
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=int(cfg["eval"].get("num_workers", 1)))
    suq.eval()
    suq.to(torch.device(device))

    scores = []
    correct = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(torch.device(device))
            out = suq(x, return_uq=True)
            logits = out.logits.detach().cpu().numpy()[0]
            y0 = int(y.numpy()[0])
            pred = int(np.argmax(logits))
            correct.append(1 if pred == y0 else 0)
            score = out.U.detach().cpu().numpy()[0] if out.U is not None else out.S.detach().cpu().numpy()[0]
            if np.ndim(score) > 0:
                score = float(np.ravel(score)[0])
            scores.append(float(score))

    correct = np.array(correct, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    roll_acc = rolling_mean(correct, window=args.window)
    # Define event when rolling accuracy below (mean - 3*std) of an ID reference band.
    # For demo we use the first 20% as reference.
    n_ref = max(50, int(0.2 * len(correct)))
    ref = roll_acc[:n_ref]
    ref = ref[~np.isnan(ref)]
    mu = float(np.mean(ref)) if len(ref) else 1.0
    sigma = float(np.std(ref)) if len(ref) else 0.0
    thresh_acc = mu - 3.0 * sigma
    event = (roll_acc < thresh_acc) & ~np.isnan(roll_acc)

    # Alarm by thresholding uncertainty at 95th percentile of reference
    ref_scores = scores[:n_ref]
    thr_u = float(np.quantile(ref_scores, 0.95))
    alarm = scores >= thr_u

    pr = auprc(event.astype(int), scores)
    delay = detection_delay(event.astype(bool), alarm.astype(bool))
    out = {"auprc_event": pr, "delay": delay, "thr_u": thr_u, "acc_event_thr": thresh_acc}

    out_dir = Path(args.out_dir) / cfg["exp"]["name"] / "results"
    save_json(out, out_dir / "stream_metrics.json")
    print(out)

if __name__ == "__main__":
    main()

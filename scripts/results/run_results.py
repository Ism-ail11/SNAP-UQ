from __future__ import annotations
import subprocess
from pathlib import Path

# This script runs the main set of experiments for a given config.
# Usage:
#   python scripts/results/run_results.py --config configs/experiments/cifar10_resnet20.yaml

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", default="runs")
    args = p.parse_args()

    cmds = [
        ["python", "scripts/train_backbone.py", "--config", args.config, "--out_dir", args.out_dir],
        ["python", "scripts/train_snapuq.py",   "--config", args.config, "--out_dir", args.out_dir],
        ["python", "scripts/calibrate.py",      "--config", args.config, "--out_dir", args.out_dir],
        ["python", "scripts/eval_offline.py",   "--config", args.config, "--out_dir", args.out_dir],
        ["python", "scripts/eval_stream.py",    "--config", args.config, "--out_dir", args.out_dir],
    ]
    for c in cmds:
        print("Running:", " ".join(c))
        subprocess.check_call(c)

if __name__ == "__main__":
    main()

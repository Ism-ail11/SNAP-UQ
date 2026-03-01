from __future__ import annotations
from pathlib import Path
import subprocess

# Runs a superset of experiments referenced in the appendix.
# It expects you to provide multiple configs (mnist, cifar10, tinyimagenet, speechcommands).
# Example:
#   python scripts/appendix/run_appendix.py --configs configs/experiments/mnist_lenet.yaml configs/experiments/cifar10_resnet20.yaml

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--out_dir", default="runs")
    args = p.parse_args()

    for cfg in args.configs:
        cmds = [
            ["python", "scripts/results/run_results.py", "--config", cfg, "--out_dir", args.out_dir],
            ["python", "scripts/ablations/run_ablations.py", "--base_config", cfg, "--out_dir", args.out_dir],
        ]
        for c in cmds:
            print("Running:", " ".join(c))
            subprocess.check_call(c)

if __name__ == "__main__":
    main()

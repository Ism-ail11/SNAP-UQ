from __future__ import annotations
import copy
from pathlib import Path
import subprocess
import yaml

# Runs ablations by editing a base config in-memory and writing temp configs.

ABLATIONS = [
    ("rank_8",  {"snapuq": {"rank": 8}}),
    ("rank_16", {"snapuq": {"rank": 16}}),
    ("rank_32", {"snapuq": {"rank": 32}}),
    ("alpha_0.0", {"snapuq": {"alpha": 0.0}}),
    ("alpha_0.5", {"snapuq": {"alpha": 0.5}}),
    ("alpha_1.0", {"snapuq": {"alpha": 1.0}}),
]

def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base_config", required=True)
    p.add_argument("--out_dir", default="runs")
    args = p.parse_args()

    base = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    temp_dir = Path(args.out_dir) / "tmp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for name, patch in ABLATIONS:
        cfg = copy.deepcopy(base)
        cfg["exp"]["name"] = f'{base["exp"]["name"]}__abl_{name}'
        deep_update(cfg, patch)
        tmp = temp_dir / f"{cfg['exp']['name']}.yaml"
        tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        cmds = [
            ["python", "scripts/train_backbone.py", "--config", str(tmp), "--out_dir", args.out_dir],
            ["python", "scripts/train_snapuq.py",   "--config", str(tmp), "--out_dir", args.out_dir],
            ["python", "scripts/calibrate.py",      "--config", str(tmp), "--out_dir", args.out_dir],
            ["python", "scripts/eval_offline.py",   "--config", str(tmp), "--out_dir", args.out_dir],
        ]
        for c in cmds:
            print("Running:", " ".join(c))
            subprocess.check_call(c)

if __name__ == "__main__":
    main()

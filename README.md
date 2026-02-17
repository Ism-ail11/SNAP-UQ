# SNAP-UQ (MCU-friendly uncertainty) — Reproducible codebase

This repository provides a clean, end-to-end implementation of **SNAP-UQ** as described in the paper
(see `paper/` if you copy the PDF in later). It includes:

- Data preprocessing (vision + audio)
- SNAP-UQ module (taps, low-rank projections, Gaussian NLL self-supervision)
- Offline evaluation (ID / corruption / OOD)
- Streaming evaluation (event detection with drift/corruption bursts)
- Baselines used in the paper (entropy, max-softmax, temperature scaling, ODIN/G-ODIN-style, MC Dropout, ensemble, Mahalanobis, HYDRA-like multihead, QUTE-style quantile thresholding)
- Ablations (taps/rank/quantization/calibration/loss variants)
- Appendix experiment runners

> Note: Datasets like CIFAR-10-C, MNIST-C, TinyImageNet-C are often distributed as separate artifacts.
> Scripts in `scripts/preprocess/` will help you download/prepare them (where URLs exist) or create
> on-the-fly corruptions when precomputed sets are unavailable.

## Quickstart

### 1) Create env
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Train a backbone (example: CIFAR-10 ResNet20)
```bash
python scripts/train_backbone.py --config configs/experiments/cifar10_resnet20.yaml
```

### 3) Train SNAP-UQ heads (post-hoc; backbone frozen)
```bash
python scripts/train_snapuq.py --config configs/experiments/cifar10_resnet20.yaml --mode posthoc
```

### 4) Calibrate mapping (logistic or isotonic)
```bash
python scripts/calibrate.py --config configs/experiments/cifar10_resnet20.yaml
```

### 5) Evaluate (offline + streaming)
```bash
python scripts/eval_offline.py --config configs/experiments/cifar10_resnet20.yaml
python scripts/eval_stream.py  --config configs/experiments/cifar10_resnet20.yaml
```

## Folder layout

- `snapuq/` — main library
- `scripts/` — runnable entry points for each section (preprocess, method, results, ablations, appendix)
- `configs/` — YAML configs (datasets, models, taps, training, eval)
- `tests/` — lightweight sanity tests (synthetic tensors; no dataset downloads)

## Reproducing paper tables/figures

See:
- `scripts/results/run_results.py`
- `scripts/ablations/run_ablations.py`
- `scripts/appendix/run_appendix.py`

They log outputs to `runs/` as CSV + JSON.

## License
MIT (see `LICENSE`).

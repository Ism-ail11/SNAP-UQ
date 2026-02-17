# SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty in TinyML

<p align="center">
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/pytorch.svg" height="44" alt="PyTorch"/>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/python.svg" height="44" alt="Python"/>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/arxiv.svg" height="44" alt="arXiv"/>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" height="44" alt="GitHub"/>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=YdK1ZRhrKi&referrer=%5Bthe%20profile%20of%20Ismail%20Lamaakal%5D(%2Fprofile%3Fid%3D~Ismail_Lamaakal1)">
    <img src="https://img.shields.io/badge/OpenReview-ICLR%202026-blue" alt="OpenReview (ICLR 2026)"/>
  </a>
  <a href="https://arxiv.org/abs/2508.12907">
    <img src="https://img.shields.io/badge/arXiv-2508.12907-b31b1b" alt="arXiv"/>
  </a>
  <img src="https://img.shields.io/badge/Status-Accepted%20ICLR%202026-success" alt="Accepted ICLR 2026"/>
  <img src="https://img.shields.io/badge/CORE-A*-informational" alt="CORE A*"/>
  <img src="https://img.shields.io/badge/License-MIT-black" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB" alt="Python 3.9+"/>
</p>

---

## Overview

**SNAP-UQ** is a lightweight uncertainty estimation method designed for **single-pass inference** and **TinyML / MCU-friendly deployment**.

This repository provides a clean, end-to-end, **reproducible** codebase for:
- data preprocessing,
- the SNAP-UQ method implementation,
- main results,
- ablation studies,
- and all appendix experiments.

✅ **Accepted in ICLR 2026** (**CORE A\***)

### Paper links
- **OpenReview (ICLR 2026):** https://openreview.net/forum?id=YdK1ZRhrKi&referrer=%5Bthe%20profile%20of%20Ismail%20Lamaakal%5D(%2Fprofile%3Fid%3D~Ismail_Lamaakal1)
- **arXiv:** https://arxiv.org/abs/2508.12907
- **Local PDF (optional):** place the camera-ready PDF in `paper/` (e.g., `paper/SNAP-UQ.pdf`)

---

## What’s included

- **Data preprocessing** (vision + audio)
- **SNAP-UQ method**
  - activation tap selection (intermediate layers)
  - low-rank projections
  - self-supervision via **Gaussian NLL** for next-activation prediction
- **Offline evaluation**
  - In-distribution (ID)
  - Corruptions (e.g., synthetic or precomputed corruption suites)
  - Out-of-distribution (OOD)
- **Streaming evaluation**
  - event detection under drift/corruption bursts (single-pass uncertainty signals)
- **Baselines used in the paper**
  - entropy, max-softmax
  - temperature scaling
  - ODIN / G-ODIN-style scoring
  - MC Dropout
  - ensembles
  - Mahalanobis distance
  - HYDRA-like multihead approaches
  - QUTE-style quantile thresholding
- **Ablations**
  - taps / rank / quantization / calibration / loss variants
- **Appendix experiment runners**
  - all additional experiments organized as runnable scripts + configs

> Note: Some corruption benchmarks (e.g., CIFAR-10-C, MNIST-C, TinyImageNet-C) may be distributed separately.  
> The scripts in `scripts/preprocess/` help you download/prepare datasets when possible, or generate corruptions  
> on-the-fly when precomputed sets are unavailable.

---

## Repository layout

```text
.
├─ snapuq/                      # Main library (models, method, losses, eval)
├─ scripts/                     # Entry points for each section
│  ├─ preprocess/               # Data preprocessing pipelines
│  ├─ method/                   # SNAP-UQ training / fitting
│  ├─ results/                  # Reproduce main paper results
│  ├─ ablations/                # Run ablation studies
│  └─ appendix/                 # All appendix experiments
├─ configs/                     # YAML configs (datasets, models, taps, training, eval)
│  ├─ datasets/
│  ├─ models/
│  └─ experiments/
├─ tests/                       # Lightweight sanity tests (no dataset downloads required)
├─ paper/                       # Optional: place PDF here
├─ runs/                        # Outputs: logs, CSV/JSON summaries, checkpoints
├─ requirements.txt
└─ README.md

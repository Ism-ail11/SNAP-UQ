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

## Abstract

Reliable uncertainty estimation is a key missing piece for on-device monitoring in TinyML: microcontrollers must detect failures, distribution shift, or accuracy
drops under strict flash/latency budgets, yet common uncertainty approaches (deep ensembles, MC dropout, early exits, temporal buffering) typically require multiple
passes, extra branches, or state that is impractical on milliwatt hardware. This paper proposes a novel and practical method, SNAP-UQ, for single-pass, label-free uncertainty estimation based on depth-wise next-activation prediction. SNAP-UQ taps a small set of backbone layers and uses tiny int8 heads to predict the mean and scale of the next activation from a low-rank projection of the previous one; the resulting standardized prediction error forms a depth-wise surprisal signal that is aggregated and mapped through a lightweight monotone calibrator into an
actionable uncertainty score. The design introduces no temporal buffers or auxiliary exits and preserves state-free inference, while increasing deployment footprint by only a few tens of kilobytes. Across vision and audio backbones, SNAPUQ reduces flash and latency relative to early-exit and deep-ensemble baselines (typically ∼40–60% smaller and ∼25–35% faster), with several competing methods at similar accuracy often exceeding MCU memory limits. On corrupted streams, it improves accuracy-drop event detection by multiple AUPRC points and maintains strong failure detection (AUROC ≈ 0.9) in a single forward pass. By grounding uncertainty in layer-to-layer dynamics rather than solely in output confidence, SNAP-UQ offers a novel, resource-efficient basis for robust TinyML monitoring.

This repository provides a clean, end-to-end, **reproducible** codebase for:
- data preprocessing,
- the SNAP-UQ method implementation,
- main results,
- ablation studies,
- and all appendix experiments.

✅ **Accepted in ICLR 2026** (**CORE A\***)

## Method overview (architecture)

<p align="center">
  <img src="SNAPUQ1 (3).jpg" alt="SNAP-UQ architecture" width="100%"/>
</p>


### Paper links
- **OpenReview (ICLR 2026):** https://openreview.net/forum?id=YdK1ZRhrKi
- **arXiv:** https://arxiv.org/abs/2508.12907

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

## Citation

If you use **SNAP-UQ** or this codebase in your work, please cite:

```bibtex
@misc{lamaakal2026snapuqselfsupervisednextactivationprediction,
  title        = {SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty in TinyML},
  author       = {Ismail Lamaakal and Chaymae Yahyati and Khalid El Makkaoui and Ibrahim Ouahbi and Yassine Maleh},
  year         = {2026},
  eprint       = {2508.12907},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2508.12907}
}


from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np

def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def gaussian_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    s = [0.05, 0.08, 0.12, 0.18, 0.26][severity-1]
    return _clip01(x + np.random.normal(0, s, size=x.shape).astype(np.float32))

def shot_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    lam = [60, 25, 12, 5, 3][severity-1]
    return _clip01(np.random.poisson(x * lam).astype(np.float32) / lam)

def impulse_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    p = [0.03, 0.06, 0.09, 0.17, 0.27][severity-1]
    mask = np.random.rand(*x.shape[:2], 1) < p
    salt = np.random.rand(*x.shape) < 0.5
    out = x.copy()
    out[mask[...,0] & salt[...,0]] = 1.0
    out[mask[...,0] & ~salt[...,0]] = 0.0
    return out.astype(np.float32)

def gaussian_blur(x: np.ndarray, severity: int = 1) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    sigma = [0.5, 1.0, 1.5, 2.0, 2.5][severity-1]
    return _clip01(gaussian_filter(x, sigma=(sigma, sigma, 0)))

def brightness(x: np.ndarray, severity: int = 1) -> np.ndarray:
    delta = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
    return _clip01(x + delta)

VISION_CORRUPTIONS: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "gaussian_blur": gaussian_blur,
    "brightness": brightness,
}

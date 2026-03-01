from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import numpy as np
from scipy import signal

def add_awgn(x: np.ndarray, snr_db: float) -> np.ndarray:
    sigp = np.mean(x**2) + 1e-12
    noisep = sigp / (10**(snr_db/10))
    n = np.random.normal(0, np.sqrt(noisep), size=x.shape).astype(np.float32)
    return (x + n).astype(np.float32)

def clipping(x: np.ndarray, clip: float) -> np.ndarray:
    return np.clip(x, -clip, clip).astype(np.float32)

def time_stretch(x: np.ndarray, rate: float) -> np.ndarray:
    # Simple resample-based stretch (no phase vocoder), good enough for corruption.
    n = int(len(x) / rate)
    return signal.resample(x, n).astype(np.float32)

def pitch_shift_resample(x: np.ndarray, semitones: float) -> np.ndarray:
    # Approx pitch shift by resampling.
    rate = 2 ** (semitones / 12.0)
    y = signal.resample(x, int(len(x) / rate)).astype(np.float32)
    # back to original length
    return signal.resample(y, len(x)).astype(np.float32)

def lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    b, a = signal.butter(4, cutoff / (sr / 2), btype="low")
    return signal.lfilter(b, a, x).astype(np.float32)

def reverb_simple(x: np.ndarray, sr: int, rt60_ms: float) -> np.ndarray:
    # Exponential decay impulse response
    ir_len = int(sr * (rt60_ms / 1000.0))
    if ir_len <= 8:
        return x.astype(np.float32)
    t = np.arange(ir_len, dtype=np.float32) / sr
    decay = np.exp(-6.91 * t / (rt60_ms / 1000.0))  # approx 60dB decay
    ir = decay * (np.random.randn(ir_len).astype(np.float32) * 0.02 + 1.0)
    y = signal.fftconvolve(x, ir, mode="full")[: len(x)]
    return y.astype(np.float32)

# Severity maps (matching the paper's typical 5-level severities)
SNR_LEVELS_DB = {
    "awgn": [35, 25, 20, 15, 10],
    "bg_noise": [25, 15, 10, 5, 0],  # use add_awgn as proxy
}

def apply_audio_corruption(x: np.ndarray, sr: int, name: str, severity: int) -> np.ndarray:
    severity = int(np.clip(severity, 1, 5))
    if name == "awgn":
        return add_awgn(x, SNR_LEVELS_DB["awgn"][severity-1])
    if name == "bg_noise":
        return add_awgn(x, SNR_LEVELS_DB["bg_noise"][severity-1])
    if name == "clipping":
        clip = [0.95, 0.8, 0.65, 0.5, 0.35][severity-1]
        return clipping(x, clip)
    if name == "time_stretch":
        rate = [0.9, 0.8, 0.7, 0.6, 0.5][severity-1]
        y = time_stretch(x, rate)
        return signal.resample(y, len(x)).astype(np.float32)
    if name == "pitch":
        semis = [1, 2, 3, 4, 5][severity-1]
        return pitch_shift_resample(x, float(semis))
    if name == "lowpass":
        cutoff = [3500, 2500, 2000, 1500, 1000][severity-1]
        return lowpass(x, sr, float(cutoff))
    if name == "reverb":
        rt = [50, 100, 150, 250, 400][severity-1]
        return reverb_simple(x, sr, float(rt))
    raise ValueError(f"Unknown audio corruption: {name}")

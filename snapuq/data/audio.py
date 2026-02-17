from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .safe_imports import try_import_torchaudio

@dataclass
class AudioFeaturizerConfig:
    sample_rate: int = 16000
    n_mels: int = 40
    win_length_ms: float = 30.0
    hop_length_ms: float = 10.0
    fmin: float = 20.0
    fmax: float = 4000.0
    log_eps: float = 1e-6

def log_mel_spectrogram(wav: np.ndarray, cfg: AudioFeaturizerConfig) -> np.ndarray:
    """Return log-mel spectrogram as float32 array [T, F]."""
    sr = cfg.sample_rate
    # Prefer torchaudio if available, else librosa.
    ok, _ = try_import_torchaudio()
    if ok:
        import torchaudio
        import torch
        wave = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        n_fft = int(sr * cfg.win_length_ms / 1000.0)
        hop = int(sr * cfg.hop_length_ms / 1000.0)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window_fn=torch.hann_window,
            n_mels=cfg.n_mels,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            power=2.0,
        )(wave)
        mel = mel.squeeze(0).clamp_min(cfg.log_eps)
        logmel = mel.log().transpose(0, 1).contiguous()  # [T,F]
        return logmel.numpy().astype(np.float32)

    import librosa
    n_fft = int(sr * cfg.win_length_ms / 1000.0)
    hop = int(sr * cfg.hop_length_ms / 1000.0)
    S = librosa.feature.melspectrogram(
        y=wav.astype(np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window="hann",
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    S = np.maximum(S, cfg.log_eps)
    return np.log(S).T.astype(np.float32)

class SpeechCommandsWrapper(Dataset):
    """Speech Commands v0.02 wrapper that returns log-mel tensors [1,T,F]."""
    def __init__(
        self,
        root: str = "data",
        subset: str = "training",
        featurizer: Optional[AudioFeaturizerConfig] = None,
        download: bool = True,
        classes: Optional[list[str]] = None,
    ):
        self.root = Path(root)
        self.subset = subset
        self.featurizer = featurizer or AudioFeaturizerConfig()
        self.classes = classes

        ok, err = try_import_torchaudio()
        if not ok:
            raise RuntimeError(
                "torchaudio is required for SpeechCommands dataset loading. "
                "Install matching wheels for your torch build."
            ) from err

        import torchaudio
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(self.root),
            download=download,
            subset=subset,
        )

        if self.classes is None:
            self.classes = sorted(list(set([label for _, _, label, *_ in self.ds])))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Cache label indices for speed
        self._targets = [self.class_to_idx.get(label, -1) for _, _, label, *_ in self.ds]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        wav, sr, label, *_ = self.ds[idx]
        y = self._targets[idx]
        # resample if needed
        if sr != self.featurizer.sample_rate:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, self.featurizer.sample_rate)
        wav = wav.squeeze(0).numpy()
        # pad/trim to 1s
        target_len = self.featurizer.sample_rate
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)), mode="constant")
        elif len(wav) > target_len:
            wav = wav[:target_len]
        feat = log_mel_spectrogram(wav, self.featurizer)  # [T,F]
        x = torch.from_numpy(feat).unsqueeze(0)  # [1,T,F]
        return x, int(y)

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def global_avg_pool(x: torch.Tensor) -> torch.Tensor:
    """Pool over all non-(B,C) dimensions."""
    if x.dim() <= 2:
        return x
    dims = tuple(range(2, x.dim()))
    return x.mean(dim=dims)

@dataclass
class SnapUQOutputs:
    logits: torch.Tensor
    S: torch.Tensor
    m: torch.Tensor
    U: Optional[torch.Tensor]
    tap_surprisals: List[torch.Tensor]
    tap_nll: List[torch.Tensor]

class GaussianPredictor(nn.Module):
    """Low-rank projector + Gaussian head predicting next pooled activation."""
    def __init__(self, in_dim: int, out_dim: int, rank: int, var_floor: float = 1e-4):
        super().__init__()
        rank = int(min(rank, in_dim))
        self.proj = nn.Linear(in_dim, rank)
        self.head = nn.Linear(rank, out_dim * 2)
        self.var_floor = float(var_floor)

    def forward(self, a_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.proj(a_prev)
        h = self.head(z)
        mu, raw = torch.chunk(h, 2, dim=-1)
        # Positive variance with softplus + floor
        var = F.softplus(raw) + self.var_floor
        return mu, var

class LogisticCalibrator(nn.Module):
    """U = sigmoid(b0 + b1*S + b2*m)."""
    def __init__(self, b0: float = 0.0, b1: float = 1.0, b2: float = 1.0):
        super().__init__()
        self.b0 = nn.Parameter(torch.tensor(float(b0)))
        self.b1 = nn.Parameter(torch.tensor(float(b1)))
        self.b2 = nn.Parameter(torch.tensor(float(b2)))

    def forward(self, S: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.b0 + self.b1 * S + self.b2 * m)

class SnapUQ(nn.Module):
    """Wraps a backbone and provides SNAP-UQ scores.

    You provide a list of *tap module names*; SNAP-UQ will create predictors for consecutive tap pairs.
    If tap names = [t0,t1,t2,t3], then we train heads for (t0->t1), (t1->t2), (t2->t3).
    """
    def __init__(
        self,
        backbone: nn.Module,
        tap_modules: Sequence[nn.Module],
        rank: int = 16,
        tap_weights: Optional[Sequence[float]] = None,
        alpha: float = 0.5,
        var_floor: float = 1e-4,
    ):
        super().__init__()
        if len(tap_modules) < 2:
            raise ValueError("Need at least 2 tap modules to form (prev->next) pairs.")
        self.backbone = backbone
        self.tap_modules = list(tap_modules)
        self.alpha = float(alpha)
        self.var_floor = float(var_floor)

        self._acts: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

        # Build predictors based on channel dims observed with a dry run; call `infer_dims()` first.
        self.predictors: nn.ModuleList = nn.ModuleList()
        self._tap_weights = None
        self._init_rank = int(rank)
        self._requested_weights = tap_weights

        # Calibrator is optional and trained separately.
        self.calibrator: Optional[nn.Module] = None

    def _register_hooks(self) -> None:
        self._hooks.clear()
        for idx, m in enumerate(self.tap_modules):
            handle = m.register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

    def _make_hook(self, idx: int):
        def hook(_m, _inp, out):
            self._acts[idx] = out
        return hook

    @torch.no_grad()
    def infer_dims(self, example_input: torch.Tensor) -> None:
        """Run one forward pass to infer pooled channel dims and create predictor heads."""
        self._acts.clear()
        _ = self.backbone(example_input)
        pooled = [global_avg_pool(self._acts[i]).detach() for i in range(len(self.tap_modules))]
        dims = [p.shape[-1] for p in pooled]
        self.predictors = nn.ModuleList([
            GaussianPredictor(in_dim=dims[i], out_dim=dims[i+1], rank=self._init_rank, var_floor=self.var_floor)
            for i in range(len(dims) - 1)
        ])
        if self._requested_weights is None:
            w = [1.0] * (len(dims) - 1)
        else:
            w = list(self._requested_weights)
            if len(w) != len(dims) - 1:
                raise ValueError("tap_weights must have length len(taps)-1.")
        wsum = sum(w)
        self._tap_weights = [float(x) / max(1e-12, wsum) for x in w]

    def set_calibrator(self, calibrator: Optional[nn.Module]) -> None:
        self.calibrator = calibrator

    def _confidence_proxy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        maxp = probs.max(dim=-1).values
        top2 = probs.topk(k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        # m(x) = α(1-maxp) + (1-α)(1-margin)
        return self.alpha * (1.0 - maxp) + (1.0 - self.alpha) * (1.0 - margin)

    def forward(self, x: torch.Tensor, return_uq: bool = False) -> torch.Tensor | SnapUQOutputs:
        logits = self.backbone(x)
        if not return_uq:
            return logits
        if len(self.predictors) == 0 or self._tap_weights is None:
            raise RuntimeError("Call infer_dims(example_input) once before using return_uq=True.")

        pooled = [global_avg_pool(self._acts[i]) for i in range(len(self.tap_modules))]
        tap_surps: List[torch.Tensor] = []
        tap_nll: List[torch.Tensor] = []
        for i, pred in enumerate(self.predictors):
            a_prev = pooled[i]
            a_next = pooled[i+1]
            mu, var = pred(a_prev)
            # Standardized residual
            u = (a_next - mu) / torch.sqrt(var + 1e-12)
            q = (u * u).sum(dim=-1)  # per sample
            d = float(a_next.shape[-1])
            tap_surps.append(q / max(1.0, d))
            nll = 0.5 * (((a_next - mu) ** 2) / (var + 1e-12) + torch.log(var + 1e-12)).sum(dim=-1) / max(1.0, d)
            tap_nll.append(nll)

        S = torch.zeros_like(tap_surps[0])
        for w, s in zip(self._tap_weights, tap_surps):
            S = S + float(w) * s

        m = self._confidence_proxy(logits)
        U = self.calibrator(S, m) if self.calibrator is not None else None
        return SnapUQOutputs(logits=logits, S=S, m=m, U=U, tap_surprisals=tap_surps, tap_nll=tap_nll)

    def gaussian_nll_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Self-supervised SNAP-UQ loss (mean over batch and taps)."""
        out = self.forward(x, return_uq=True)
        assert isinstance(out, SnapUQOutputs)
        # Each tap_nll[i] is [B]; average across taps then batch.
        loss = torch.stack([t.mean() for t in out.tap_nll]).mean()
        return loss

    def quantize_predictors_int8(self) -> None:
        """Dynamic INT8 quantization for linear layers in SNAP-UQ heads (host-side)."""
        try:
            import torch.ao.quantization as tq
        except Exception as e:
            raise RuntimeError("torch.ao.quantization not available in this PyTorch build.") from e
        self.predictors = tq.quantize_dynamic(self.predictors, {nn.Linear}, dtype=torch.qint8)

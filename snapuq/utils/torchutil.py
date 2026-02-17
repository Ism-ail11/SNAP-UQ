from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import torch

@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0
    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)
    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)

def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean()

@torch.no_grad()
def model_num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def state_dict_size_bytes(model: torch.nn.Module) -> int:
    # Rough estimate: number of elements * bytes-per-element for each parameter/buffer.
    size = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if not hasattr(t, "element_size"):
            continue
        size += t.numel() * t.element_size()
    return int(size)

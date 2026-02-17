from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .safe_imports import try_import_torchvision

@dataclass
class VisionDataConfig:
    dataset: str
    root: str = "data"
    train: bool = True
    download: bool = True

def build_vision_dataset(cfg: VisionDataConfig):
    ok, err = try_import_torchvision()
    if not ok:
        raise RuntimeError(
            "torchvision is required for MNIST/CIFAR datasets. "
            "Install matching wheels for your torch build (see README)."
        ) from err
    from torchvision import datasets, transforms

    root = Path(cfg.root)
    if cfg.dataset.lower() == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        return datasets.MNIST(root=str(root), train=cfg.train, download=cfg.download, transform=tfm)

    if cfg.dataset.lower() == "cifar10":
        if cfg.train:
            tfm = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            tfm = transforms.Compose([transforms.ToTensor()])
        return datasets.CIFAR10(root=str(root), train=cfg.train, download=cfg.download, transform=tfm)

    if cfg.dataset.lower() == "tinyimagenet":
        # Expect you have extracted TinyImageNet into data/tinyimagenet/
        # train: data/tinyimagenet/train/<class>/images/*.JPEG
        # val:   data/tinyimagenet/val/images/*.JPEG with val_annotations.txt
        from .tinyimagenet import TinyImageNet
        return TinyImageNet(root=str(root / "tinyimagenet"), split="train" if cfg.train else "val")

    raise ValueError(f"Unknown vision dataset: {cfg.dataset}")

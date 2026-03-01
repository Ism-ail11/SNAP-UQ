from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class TinyImageNet(Dataset):
    """Minimal TinyImageNet loader (expects standard folder structure)."""
    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self._build_index()

    def _build_index(self) -> None:
        wnids_path = self.root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(
                f"{wnids_path} not found. Download and extract TinyImageNet to {self.root}."
            )
        wnids = [l.strip() for l in wnids_path.read_text().splitlines() if l.strip()]
        self.class_to_idx = {c: i for i, c in enumerate(wnids)}

        if self.split == "train":
            for c in wnids:
                img_dir = self.root / "train" / c / "images"
                if not img_dir.exists():
                    continue
                for p in img_dir.glob("*.JPEG"):
                    self.samples.append((p, self.class_to_idx[c]))
        elif self.split in ("val", "valid", "validation"):
            ann = self.root / "val" / "val_annotations.txt"
            img_dir = self.root / "val" / "images"
            if not ann.exists():
                raise FileNotFoundError(f"{ann} not found.")
            for line in ann.read_text().splitlines():
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                img_name, c = parts[0], parts[1]
                if c not in self.class_to_idx:
                    continue
                self.samples.append((img_dir / img_name, self.class_to_idx[c]))
        else:
            raise ValueError(f"Unknown split: {self.split}")

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for TinyImageNet split={self.split} at {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is None:
            # Lightweight default: resize to 64x64 and normalize to [0,1]
            import numpy as np
            img = img.resize((64,64))
            x = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        else:
            x = self.transform(img)
        return x, int(y)

from __future__ import annotations
from typing import Dict, Any, Tuple
import torch.nn as nn

from .vision.lenet import LeNetMNIST
from .vision.resnet_cifar import resnet20_cifar
from .vision.mobilenetv2_tiny import mobilenetv2_tiny
from .audio.dscnn import dscnn_small

def build_backbone(cfg: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build backbone model from config.
    Returns (model, meta) where meta includes tap module name defaults.
    """
    name = cfg["model"]["name"]
    num_classes = int(cfg["data"]["num_classes"])
    in_ch = int(cfg["data"].get("in_channels", 1))

    if name == "lenet_mnist":
        m = LeNetMNIST(num_classes=num_classes, in_channels=in_ch)
        meta = {"default_taps": ["features.0", "features.3", "features.6", "classifier.1"]}
        return m, meta
    if name == "resnet20_cifar":
        m = resnet20_cifar(num_classes=num_classes)
        meta = {"default_taps": ["conv1", "layer1.2", "layer2.2", "layer3.2"]}
        return m, meta
    if name == "mobilenetv2_tiny":
        width_mult = float(cfg["model"].get("width_mult", 1.0))
        m = mobilenetv2_tiny(num_classes=num_classes, width_mult=width_mult, in_channels=in_ch)
        meta = {"default_taps": ["features.0", "features.3", "features.6", "features.12"]}
        return m, meta
    if name == "dscnn_small":
        m = dscnn_small(num_classes=num_classes, in_channels=in_ch)
        meta = {"default_taps": ["features.0", "features.2", "features.4", "features.6"]}
        return m, meta

    raise ValueError(f"Unknown model.name: {name}")

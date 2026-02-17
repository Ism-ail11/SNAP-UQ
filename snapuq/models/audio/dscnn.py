from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class DSCNNSmall(nn.Module):
    """Small DS-CNN for keyword spotting (expects log-mel [B,1,T,F])."""
    def __init__(self, num_classes: int = 12, in_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(base_ch, base_ch, 3, 1, 1),
            DepthwiseSeparableConv(base_ch, base_ch, 3, 1, 1),
            DepthwiseSeparableConv(base_ch, base_ch, 3, 1, 1),

            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(base_ch, base_ch, 3, 1, 1),
            DepthwiseSeparableConv(base_ch, base_ch, 3, 1, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(base_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

def dscnn_small(num_classes: int, in_channels: int = 1) -> DSCNNSmall:
    return DSCNNSmall(num_classes=num_classes, in_channels=in_channels)

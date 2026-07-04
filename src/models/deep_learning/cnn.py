"""1D convolutional network for tabular intrusion detection.

The feature vector is treated as a one-channel 1D signal; stacked
``Conv1d -> BatchNorm -> ReLU`` blocks (configurable channels and kernel
sizes) feed an adaptive-average-pooled linear head. Configuration:
``configs/deep_learning.yaml > models.cnn.params``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from src.models.deep_learning.base import TorchModelBase

logger = logging.getLogger(__name__)


class CNNNetwork(nn.Module):
    """1D CNN over the feature axis with a pooled linear classifier head."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        channels: list[int],
        kernel_sizes: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(channels) != len(kernel_sizes):
            raise ValueError(
                f"channels ({len(channels)}) and kernel_sizes "
                f"({len(kernel_sizes)}) must have equal length."
            )
        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels, kernel in zip(channels, kernel_sizes):
            blocks.extend([
                nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ])
            in_channels = out_channels
        self.convolutions = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head = nn.Linear(in_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a ``(batch, features)`` tensor."""
        signal = x.unsqueeze(1)  # (batch, 1, features)
        features = self.pool(self.convolutions(signal)).squeeze(-1)
        return self.head(self.dropout(features))


class CNNModel(TorchModelBase):
    """1D-CNN wrapper exposing the Engine A model interface."""

    name = "cnn"

    def build_network(self, input_dim: int, n_classes: int) -> nn.Module:
        """Construct the CNN from configuration."""
        arch = self.arch_params
        return CNNNetwork(
            input_dim=input_dim,
            n_classes=n_classes,
            channels=[int(c) for c in arch.get("channels", [32, 64])],
            kernel_sizes=[int(k) for k in arch.get("kernel_sizes", [3, 3])],
            dropout=float(arch.get("dropout", 0.0)),
        )

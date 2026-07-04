"""Multi-Layer Perceptron for tabular intrusion detection.

Configurable hidden layers, dropout and batch normalization
(``configs/deep_learning.yaml > models.mlp.params``). Training/prediction is
inherited from :class:`src.models.deep_learning.base.TorchModelBase`.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from src.models.deep_learning.base import TorchModelBase

logger = logging.getLogger(__name__)


class MLPNetwork(nn.Module):
    """Feed-forward classifier: ``[Linear -> (BN) -> ReLU -> Dropout] x N``."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_layers: list[int],
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(previous, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a ``(batch, features)`` tensor."""
        return self.network(x)


class MLPModel(TorchModelBase):
    """MLP wrapper exposing the Engine A model interface."""

    name = "mlp"

    def build_network(self, input_dim: int, n_classes: int) -> nn.Module:
        """Construct the MLP from configuration."""
        arch = self.arch_params
        return MLPNetwork(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_layers=[int(w) for w in arch.get("hidden_layers", [256, 128])],
            dropout=float(arch.get("dropout", 0.0)),
            batch_norm=bool(arch.get("batch_norm", False)),
        )

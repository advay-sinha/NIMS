"""LSTM network for tabular intrusion detection.

The feature vector is treated as a synthetic sequence (one timestep per
feature, scalar input size); the final hidden state (concatenated directions
when bidirectional) feeds a linear classifier head. Configuration:
``configs/deep_learning.yaml > models.lstm.params``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from src.models.deep_learning.base import TorchModelBase

logger = logging.getLogger(__name__)


class LSTMNetwork(nn.Module):
    """LSTM over the feature axis with a linear classifier head."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        directions = 2 if bidirectional else 1
        self.head = nn.Linear(hidden_size * directions, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a ``(batch, features)`` tensor."""
        sequence = x.unsqueeze(-1)  # (batch, features, 1)
        _, (hidden, _) = self.lstm(sequence)
        if self.lstm.bidirectional:
            final = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final = hidden[-1]
        return self.head(final)


class LSTMModel(TorchModelBase):
    """LSTM wrapper exposing the Engine A model interface."""

    name = "lstm"

    def build_network(self, input_dim: int, n_classes: int) -> nn.Module:
        """Construct the LSTM from configuration."""
        arch = self.arch_params
        return LSTMNetwork(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_size=int(arch.get("hidden_size", 64)),
            num_layers=int(arch.get("num_layers", 1)),
            bidirectional=bool(arch.get("bidirectional", False)),
            dropout=float(arch.get("dropout", 0.0)),
        )

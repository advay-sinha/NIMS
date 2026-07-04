"""Transformer encoder for tabular intrusion detection.

Each scalar feature becomes a token: a shared linear layer embeds the value
into ``embedding_dim`` and a learned per-feature positional embedding is
added. A standard TransformerEncoder processes the tokens; mean pooling feeds
the classifier head. Configuration:
``configs/deep_learning.yaml > models.transformer.params``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from src.models.deep_learning.base import TorchModelBase

logger = logging.getLogger(__name__)


class TransformerNetwork(nn.Module):
    """Feature-tokenized transformer encoder with a mean-pooled linear head."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        embedding_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        feedforward_dim: int | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.value_embedding = nn.Linear(1, embedding_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, input_dim, embedding_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim or embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a ``(batch, features)`` tensor."""
        tokens = self.value_embedding(x.unsqueeze(-1)) + self.position_embedding
        encoded = self.encoder(tokens)
        return self.head(encoded.mean(dim=1))


class TransformerModel(TorchModelBase):
    """Transformer wrapper exposing the Engine A model interface."""

    name = "transformer"

    def build_network(self, input_dim: int, n_classes: int) -> nn.Module:
        """Construct the transformer from configuration."""
        arch = self.arch_params
        return TransformerNetwork(
            input_dim=input_dim,
            n_classes=n_classes,
            embedding_dim=int(arch.get("embedding_dim", 32)),
            num_heads=int(arch.get("num_heads", 4)),
            num_layers=int(arch.get("num_layers", 2)),
            dropout=float(arch.get("dropout", 0.1)),
            feedforward_dim=(
                int(arch["feedforward_dim"]) if arch.get("feedforward_dim") else None
            ),
        )

"""Engine B (deep learning) model registry.

Maps deep-learning model ids to their wrapper classes. Merged into the main
model registry (:mod:`src.models.registry`) lazily, so Engine A code paths do
not import PyTorch unless a deep model is actually requested.
"""

from __future__ import annotations

import logging
from typing import Type

from src.models.base import BaseModel
from src.models.deep_learning.cnn import CNNModel
from src.models.deep_learning.lstm import LSTMModel
from src.models.deep_learning.mlp import MLPModel
from src.models.deep_learning.transformer import TransformerModel

logger = logging.getLogger(__name__)

# Single source of truth mapping deep model id -> class.
DEEP_MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "mlp": MLPModel,
    "cnn": CNNModel,
    "lstm": LSTMModel,
    "transformer": TransformerModel,
}

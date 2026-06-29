"""Deterministic seeding for reproducibility.

Purpose
-------
Guarantee reproducible runs by seeding every source of randomness in one call
(CLAUDE.md > Machine Learning Standards: set random / numpy / torch seeds).
Used by data splitting in Phase 1 and by training in later phases.

Inputs
------
- An integer seed (typically ``config["project"]["seed"]``).

Outputs
-------
- Side effects only: global RNG state is set.

Examples
--------
>>> from src.utils.seed import set_global_seed
>>> set_global_seed(42)              # doctest: +SKIP

Limitations
-----------
Full determinism on GPU also requires deterministic cuDNN algorithms, which
can reduce throughput; this is opt-in via ``deterministic_torch``.
"""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed python, numpy and (if installed) torch RNGs.

    Parameters
    ----------
    seed:
        Non-negative integer seed.
    deterministic_torch:
        When ``True`` and torch is available, enable deterministic cuDNN
        behaviour (``torch.backends.cudnn.deterministic = True``) at the cost
        of performance.

    Raises
    ------
    ValueError
        If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import numpy as np

    np.random.seed(seed)

    # torch is optional (Phase 1 needs no GPU); seed it only if installed.
    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding")
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    logger.debug("Global seed set to %d", seed)

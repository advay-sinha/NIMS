"""Repository path resolution.

Purpose
-------
Resolve every filesystem location from the ``paths`` configuration block so
that no path is hardcoded in business logic (CLAUDE.md > Repository
Principles). Also distinguishes the read-only raw dataset roots from writable
pipeline-stage directories.

Inputs
------
- The ``paths`` block of the merged configuration (``configs/paths.yaml``).

Outputs
-------
- Absolute, normalised ``Path`` objects.

Examples
--------
>>> paths = Paths.from_config(config)         # doctest: +SKIP
>>> paths.processed_dir                        # doctest: +SKIP
>>> paths.raw_dir("nsl_kdd")                   # doctest: +SKIP

Limitations
-----------
Directory creation is explicit (``ensure_dir``); resolution never creates the
read-only raw roots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# Repository root = two levels above this file (src/utils/paths.py).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Resolved repository paths.

    All attributes are absolute paths. Writable pipeline-stage directories are
    distinct from the read-only ``datasets/`` raw roots accessed via
    :meth:`raw_dir`.
    """

    root: Path
    data_dir: Path
    datasets_dir: Path
    models_dir: Path
    outputs_dir: Path
    logs_dir: Path
    interim_dir: Path
    processed_dir: Path
    features_dir: Path
    metadata_dir: Path
    reports_dir: Path
    data_reports_dir: Path
    fingerprints_dir: Path
    figures_dir: Path
    # Phase 2 preprocessing outputs (reports, processed splits, fitted artefacts).
    preprocessing_dir: Path
    processed_out_dir: Path
    artifacts_dir: Path
    # Phase 3 feature-engineering outputs (transformed datasets + reports).
    features_out_dir: Path
    # Phase 4 training experiment outputs (models, metrics, manifests).
    experiments_dir: Path
    # Explainability artefacts (SHAP values, importance tables, plots).
    explainability_dir: Path
    # Mapping of dataset id -> read-only raw directory.
    raw: Mapping[str, Path]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Paths":
        """Build a :class:`Paths` instance from the ``paths`` config block.

        Parameters
        ----------
        config:
            Effective configuration containing a ``paths`` section.

        Returns
        -------
        Paths
            Resolved, absolute paths.
        """
        try:
            block = config["paths"]
        except (KeyError, TypeError) as exc:
            raise KeyError("Configuration is missing a 'paths' section.") from exc

        def resolve(key: str) -> Path:
            value = block[key]
            candidate = Path(value)
            return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)

        raw_block: Mapping[str, Any] = block.get("raw", {})
        raw = {
            dataset_id: (
                Path(loc) if Path(loc).is_absolute() else (PROJECT_ROOT / loc)
            )
            for dataset_id, loc in raw_block.items()
        }

        return cls(
            root=PROJECT_ROOT,
            data_dir=resolve("data_dir"),
            datasets_dir=resolve("datasets_dir"),
            models_dir=resolve("models_dir"),
            outputs_dir=resolve("outputs_dir"),
            logs_dir=resolve("logs_dir"),
            interim_dir=resolve("interim_dir"),
            processed_dir=resolve("processed_dir"),
            features_dir=resolve("features_dir"),
            metadata_dir=resolve("metadata_dir"),
            reports_dir=resolve("reports_dir"),
            data_reports_dir=resolve("data_reports_dir"),
            fingerprints_dir=resolve("fingerprints_dir"),
            figures_dir=resolve("figures_dir"),
            preprocessing_dir=resolve("preprocessing_dir"),
            processed_out_dir=resolve("processed_out_dir"),
            artifacts_dir=resolve("artifacts_dir"),
            features_out_dir=resolve("features_out_dir"),
            experiments_dir=resolve("experiments_dir"),
            explainability_dir=resolve("explainability_dir"),
            raw=raw,
        )

    def raw_dir(self, dataset_id: str) -> Path:
        """Return the read-only raw directory for a dataset.

        Parameters
        ----------
        dataset_id:
            Dataset identifier (e.g. ``"cicids2017"``).

        Returns
        -------
        Path

        Raises
        ------
        KeyError
            If ``dataset_id`` has no configured raw directory.
        """
        if dataset_id not in self.raw:
            raise KeyError(f"No raw directory configured for dataset '{dataset_id}'")
        return self.raw[dataset_id]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if absent and return it.

    Never call this on a read-only raw dataset root.

    Parameters
    ----------
    path:
        Directory to create.

    Returns
    -------
    Path
        The same path, guaranteed to exist.
    """
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved

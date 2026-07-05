"""Entry point: rebuild the model registry from experiment manifests.

Scans every completed experiment manifest and rewrites
``outputs/registry/registry.json`` and ``best_per_dataset.json``. Existing
registration timestamps, tags and lifecycle statuses are preserved;
``production.json`` is never overwritten (promotion is explicit via
``scripts/promote_model.py``). No model files are moved and no metrics are
recomputed.

Usage
-----
    python -m scripts.build_model_registry
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.registry.artifacts import load_production, rebuild_registry
from src.registry.reporting import registry_summary
from src.utils.io import read_json

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Rebuild the model registry from experiment manifests."
    )
    add_common_arguments(parser)
    return parser


def artifact_roots(paths: "object") -> dict[str, Path]:
    """Map the optional-artefact roots the registry references."""
    return {
        "explainability": Path(paths.explainability_dir),
        "error_analysis": Path(paths.error_analysis_dir),
        "visualizations": Path(paths.visualizations_dir),
        "features": Path(paths.features_out_dir),
        "preprocessing_artifacts": Path(paths.artifacts_dir),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        ``0`` on success, ``1`` when no experiment could be registered.
    """
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    cfg = dict(ctx.config.get("registry") or {})

    registry_dir = Path(ctx.paths.registry_dir)
    document = rebuild_registry(
        Path(ctx.paths.experiments_dir),
        registry_dir,
        artifact_roots(ctx.paths),
        selection_metric=str(cfg.get("selection_metric", "test_f1")),
        higher_is_better=bool(cfg.get("higher_is_better", True)),
        require_test_metrics=bool(cfg.get("require_test_metrics", True)),
        allow_optimized_models=bool(cfg.get("allow_optimized_models", True)),
    )
    if not document["entries"]:
        logger.error("No registrable experiments found under %s.",
                     ctx.paths.experiments_dir)
        return 1

    best = read_json(registry_dir / "best_per_dataset.json")
    summary = registry_summary(
        document, best, load_production(registry_dir), registry_dir
    )
    for line in summary.splitlines():
        if line.strip():
            logger.info("%s", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

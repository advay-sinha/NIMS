"""Entry point: resolve the model serving a dataset from the registry.

Read-only lookup over ``outputs/registry/`` via
:func:`src.registry.resolver.resolve_model` — nothing is retrained,
recomputed or modified. Human-readable by default; ``--json`` emits valid
JSON on stdout for scripting.

Usage
-----
    python -m scripts.resolve_model --dataset unsw_nb15
    python -m scripts.resolve_model --dataset unsw_nb15 --stage best --json
    python -m scripts.resolve_model --dataset nsl_kdd --model xgboost
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from scripts._bootstrap import add_common_arguments, bootstrap
from src.registry.registry import RegistryError
from src.registry.resolver import resolve_model

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Resolve the registered model serving a dataset."
    )
    add_common_arguments(parser)
    parser.add_argument("--dataset", required=True, help="Dataset id.")
    parser.add_argument(
        "--stage", default="production", choices=["production", "best"],
        help="Lookup stage (default: production).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Expected model type; resolution fails if it differs.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit the resolution as JSON on stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        ``0`` on success; ``1`` when no model is assigned/registered for the
        dataset/stage, or the resolved model type mismatches ``--model``.
    """
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    try:
        resolved = resolve_model(
            args.dataset, args.stage, registry_dir=ctx.paths.registry_dir
        )
    except RegistryError as exc:
        logger.error("%s", exc)
        return 1

    if args.model is not None and resolved["model_type"] != args.model:
        logger.error(
            "Resolved model for '%s' at stage '%s' is '%s' (run %s), not the "
            "requested '%s'.", args.dataset, args.stage,
            resolved["model_type"], resolved["experiment_id"], args.model,
        )
        return 1

    if args.as_json:
        # Machine output contract: bare JSON on stdout for piping.
        sys.stdout.write(json.dumps(resolved, indent=2) + "\n")
        return 0

    logger.info("experiment_id: %s", resolved["experiment_id"])
    logger.info("model_type: %s", resolved["model_type"])
    logger.info("status: %s | stage: %s", resolved["status"], resolved["stage"])
    logger.info("model_artifact_path: %s", resolved["model_artifact_path"])
    logger.info("manifest_path: %s", resolved["manifest_path"])
    for split, values in resolved["metrics"].items():
        rendered = ", ".join(f"{k}={v:.4f}" for k, v in values.items())
        logger.info("metrics[%s]: %s", split, rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())

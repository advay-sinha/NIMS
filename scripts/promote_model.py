"""Entry point: promote a registered model to production for a dataset.

Updates ``outputs/registry/production.json`` (and the entry statuses in
``registry.json``). Model files are never moved, copied or deleted.

Without ``--experiment-id``, the best registered candidate for the
(dataset, model) pair — by the configured selection metric — is chosen and
the alternatives are logged; pass an explicit id to pin a specific run.

Usage
-----
    python -m scripts.promote_model --dataset unsw_nb15 --model xgboost \
        --reason "Best validated baseline"
    python -m scripts.promote_model --dataset unsw_nb15 --model xgboost \
        --experiment-id unsw_nb15_xgboost_20260704T214803 --reason "Pinned"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.registry.artifacts import load_registry, promote
from src.registry.registry import RegistryError, metric_value

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Promote a registered model to production."
    )
    add_common_arguments(parser)
    parser.add_argument("--dataset", required=True, help="Dataset id.")
    parser.add_argument("--model", required=True, help="Model type to promote.")
    parser.add_argument(
        "--experiment-id", default=None,
        help="Specific run to promote (default: best registered candidate).",
    )
    parser.add_argument(
        "--reason", required=True, help="Promotion rationale (recorded)."
    )
    return parser


def _pick_candidate(
    registry_dir: Path, dataset: str, model: str, selection_metric: str
) -> str:
    """Choose the best registered (dataset, model) candidate deterministically."""
    entries = [
        e for e in load_registry(registry_dir).get("entries", [])
        if e["dataset"] == dataset and e["model_type"] == model
        and e["status"] != "archived"
    ]
    if not entries:
        raise RegistryError(
            f"No registered '{model}' runs for dataset '{dataset}'. Run "
            f"'python -m scripts.build_model_registry' first."
        )
    ranked = sorted(
        entries,
        key=lambda e: (-(metric_value(e, selection_metric) or float("-inf")),
                       e["experiment_id"]),
    )
    chosen = ranked[0]
    if len(ranked) > 1:
        logger.info(
            "%d candidate(s) for %s/%s; choosing the best by %s: %s "
            "(alternatives: %s). Pass --experiment-id to pin another.",
            len(ranked), dataset, model, selection_metric,
            chosen["experiment_id"],
            ", ".join(e["experiment_id"] for e in ranked[1:]),
        )
    return chosen["experiment_id"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        ``0`` on success, ``1`` on any expected failure.
    """
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    registry_dir = Path(ctx.paths.registry_dir)
    metric = str(ctx.config.get("registry", {}).get("selection_metric", "test_f1"))

    try:
        experiment_id = args.experiment_id or _pick_candidate(
            registry_dir, args.dataset, args.model, metric
        )
        assignment = promote(
            registry_dir,
            dataset=args.dataset,
            experiment_id=experiment_id,
            reason=args.reason,
        )
    except RegistryError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Production model for '%s': %s (%s) — %s",
        args.dataset, assignment["experiment_id"], assignment["model_type"],
        assignment["reason"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

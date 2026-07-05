"""Entry point: generate SHAP explainability artefacts for a completed run.

Loads an already-trained model and the feature-engineered split from disk —
nothing is retrained — and writes the explainability artefacts to
``outputs/explainability/<experiment_id>/``.

Usage
-----
    python -m scripts.run_explainability --dataset nsl_kdd --model xgboost
    python -m scripts.run_explainability --dataset nsl_kdd --model xgboost \
        --run-id nsl_kdd_xgboost_20260703T185430 --max-samples 5000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.explainability.base import ExplainabilityError
from src.explainability.runner import explain_model
from src.models.base import BaseModel
from src.training.trainer import _load_xy
from src.utils.config import deep_merge, load_dataset_config
from src.utils.io import read_parquet

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SHAP explainability artefacts for a trained model."
    )
    add_common_arguments(parser)
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. nsl_kdd).")
    parser.add_argument("--model", required=True, help="Model id (e.g. xgboost).")
    parser.add_argument(
        "--run-id", default=None,
        help="Experiment id to explain (defaults to the newest run).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Override explainability.max_samples for this invocation.",
    )
    return parser


def _resolve_run_dir(experiments_dir: Path, dataset: str, model: str,
                     run_id: str | None) -> Path:
    """Locate the experiment directory to explain (newest when unspecified)."""
    model_dir = experiments_dir / dataset / model
    if run_id is not None:
        run_dir = model_dir / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Experiment not found: {run_dir}")
        return run_dir
    runs = sorted(d for d in model_dir.glob("*") if (d / "manifest.json").is_file())
    if not runs:
        raise FileNotFoundError(
            f"No completed experiments under {model_dir}; train the model first."
        )
    return runs[-1]  # run ids embed a sortable UTC timestamp


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        ``0`` on success, ``1`` on any expected failure (missing run/split,
        unsupported model).
    """
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    config = ctx.config
    if args.max_samples is not None:
        config = deep_merge(config, {"explainability": {"max_samples": args.max_samples}})

    try:
        run_dir = _resolve_run_dir(
            Path(ctx.paths.experiments_dir), args.dataset, args.model, args.run_id
        )
        model = BaseModel.load(run_dir / "model.joblib")

        split = str(config.get("explainability", {}).get("split", "test"))
        label_column = load_dataset_config(args.dataset).get("label_column")
        split_path = Path(ctx.paths.features_out_dir) / args.dataset / f"{split}.parquet"
        if not split_path.is_file():
            raise FileNotFoundError(f"Feature split not found: {split_path}")
        x, _y = _load_xy(read_parquet(split_path), str(label_column))

        artefacts = explain_model(
            model,
            x,
            experiment_id=run_dir.name,
            dataset_id=args.dataset,
            config=config,
            output_root=Path(ctx.paths.explainability_dir),
        )
    except (FileNotFoundError, ExplainabilityError) as exc:
        logger.error("%s", exc)
        return 1

    for name, path in artefacts.items():
        logger.info("%s -> %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

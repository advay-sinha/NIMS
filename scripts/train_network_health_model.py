"""Entry point: train the network-health Isolation Forest baseline.

Reads the preprocessed splits, engineers the health features, trains the
baseline (supervised on healthy rows when labels exist, unsupervised
otherwise), persists the experiment and generates the summary report.

Usage
-----
    python -m scripts.train_network_health_model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.network_health.artifacts import (
    write_experiment,
    write_feature_splits,
    write_report,
)
from src.network_health.baseline import train_baseline
from src.network_health.features import build_features
from src.network_health.reporting import network_health_report
from src.network_health.schema import TelemetrySchema
from src.utils.io import read_json

logger = logging.getLogger(__name__)

_SPLITS = ("train", "validation", "test")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Train the network-health anomaly baseline."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--dataset", default=None,
        help="Registered dataset id (namespaces the processed splits and "
             "experiment; see network_health.datasets).",
    )
    parser.add_argument(
        "--model", default="isolation_forest", choices=("isolation_forest",),
        help="Baseline model (only the Isolation Forest baseline exists so far).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` on success)."""
    import pandas as pd

    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    cfg = dict(ctx.config.get("network_health") or {})
    schema = TelemetrySchema.from_config(cfg.get("schema") or {})
    dataset_id = str(cfg.get("dataset_id", "telemetry"))
    if args.dataset:
        from src.network_health.dataset_registry import resolve_pipeline_source

        try:
            _, dataset_id = resolve_pipeline_source(ctx.config, args.dataset)
        except (KeyError, ValueError) as exc:
            logger.error("%s", exc)
            return 1
    root = Path(ctx.paths.network_health_dir)
    seed = int(ctx.config.get("project", {}).get("seed", 42))

    processed_dir = root / "processed" / dataset_id
    if not (processed_dir / "train.parquet").is_file():
        logger.error(
            "No processed telemetry under %s; run "
            "'python -m scripts.run_network_health_preprocessing' first.",
            processed_dir,
        )
        return 1

    feature_splits = {}
    metadata = {}
    for split_name in _SPLITS:
        path = processed_dir / f"{split_name}.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        feature_splits[split_name], metadata = build_features(
            frame, schema, dict(cfg.get("features") or {})
        )
    write_feature_splits(feature_splits, metadata, root, dataset_id)

    baseline, metrics = train_baseline(
        feature_splits,
        list(metadata["feature_columns"]),
        schema,
        dict(cfg.get("model", {}).get("isolation_forest") or {}),
        seed,
    )
    experiment = write_experiment(
        baseline, metrics, root, dataset_id,
        config_snapshot=cfg, seed=seed,
    )

    validation_path = root / "validation" / "validation_report.json"
    validation = read_json(validation_path) if validation_path.is_file() else None
    manifest = read_json(processed_dir / "preprocessing_manifest.json")
    report = network_health_report(
        dataset_id=dataset_id,
        validation=validation,
        preprocessing_manifest=manifest,
        feature_metadata=metadata,
        metrics=metrics,
        experiment_id=experiment["run_dir"].name,
    )
    report_path = write_report(report, root)
    for line in report.splitlines():
        if line.strip():
            logger.info("%s", line)
    logger.info("Report: %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

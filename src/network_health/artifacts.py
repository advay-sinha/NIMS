"""Engine B artefact persistence.

Purpose
-------
Write every network-health pipeline stage to its fixed location under
``outputs/network_health/``::

    validation/{validation_report.json, validation_report.md}
    processed/<dataset_id>/{train,validation,test}.parquet + manifest
    features/<dataset_id>/{train,validation,test}.parquet + feature_metadata
    experiments/<dataset_id>/isolation_forest/<run_id>/{model,metrics,manifest}
    reports/network_health_report.md
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def write_validation_report(report: Any, root: Path) -> dict[str, Path]:
    """Persist a validation report as JSON + Markdown."""
    from src.utils.io import write_json
    from src.utils.paths import ensure_dir

    out_dir = ensure_dir(Path(root) / "validation")
    payload = report.to_dict()
    json_path = write_json(payload, out_dir / "validation_report.json")

    lines = [
        f"# Network-Health Validation — {payload['dataset_id']}",
        "",
        f"- Result: {'PASSED' if payload['passed'] else 'FAILED'}",
        f"- Rows: {payload['n_rows']:,} | Devices: {payload['n_devices']} "
        f"| Interfaces: {payload['n_interfaces']}",
        f"- Errors: {payload['n_errors']} | Warnings: {payload['n_warnings']}",
        "",
    ]
    if payload["issues"]:
        lines.extend(["| Severity | Check | Message |", "|---|---|---|"])
        lines.extend(
            f"| {i['severity']} | {i['check']} | {i['message']} |"
            for i in payload["issues"]
        )
        lines.append("")
    md_path = out_dir / "validation_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Validation report written to %s.", out_dir)
    return {"json": json_path, "markdown": md_path}


def write_processed_splits(
    result: Any, root: Path, dataset_id: str
) -> dict[str, Path]:
    """Persist preprocessed splits and the preprocessing manifest."""
    from src.utils.io import write_json
    from src.utils.paths import ensure_dir

    out_dir = ensure_dir(Path(root) / "processed" / dataset_id)
    paths: dict[str, Path] = {}
    for name, frame in result.splits.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    paths["manifest"] = write_json(
        result.manifest, out_dir / "preprocessing_manifest.json"
    )
    logger.info("Processed splits written to %s.", out_dir)
    return paths


def write_feature_splits(
    splits: Mapping[str, Any],
    metadata: Mapping[str, Any],
    root: Path,
    dataset_id: str,
) -> dict[str, Path]:
    """Persist feature splits and the feature metadata."""
    from src.utils.io import write_json
    from src.utils.paths import ensure_dir

    out_dir = ensure_dir(Path(root) / "features" / dataset_id)
    paths: dict[str, Path] = {}
    for name, frame in splits.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    paths["metadata"] = write_json(
        dict(metadata), out_dir / "feature_metadata.json"
    )
    logger.info("Feature splits written to %s.", out_dir)
    return paths


def write_experiment(
    baseline: Any,
    metrics: Mapping[str, Any],
    root: Path,
    dataset_id: str,
    *,
    config_snapshot: Mapping[str, Any],
    seed: int,
) -> dict[str, Path]:
    """Persist one baseline training run under a unique run directory."""
    import joblib

    from src.utils.io import write_json
    from src.utils.paths import ensure_dir

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{dataset_id}_isolation_forest_{stamp}"
    parent = Path(root) / "experiments" / dataset_id / "isolation_forest"
    run_dir = parent / run_id
    suffix = 1
    while run_dir.exists():  # never overwrite an experiment
        run_dir = parent / f"{run_id}_{suffix}"
        suffix += 1
    ensure_dir(run_dir)

    model_path = run_dir / "model.joblib"
    joblib.dump(baseline, model_path)
    metrics_path = write_json(dict(metrics), run_dir / "metrics.json")
    manifest = {
        "experiment_id": run_dir.name,
        "dataset_id": dataset_id,
        "model_name": "isolation_forest",
        "engine": "network_health",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "labeled": bool(baseline.labeled),
        "threshold": baseline.threshold,
        "threshold_quantile": baseline.threshold_quantile,
        "n_features": len(baseline.feature_columns),
        "feature_columns": baseline.feature_columns,
        "params": baseline.params,
        "config_snapshot": dict(config_snapshot),
        "metrics": dict(metrics),
        "artefacts": {
            "model": str(model_path),
            "metrics": str(metrics_path),
        },
    }
    manifest_path = write_json(manifest, run_dir / "manifest.json")
    logger.info("Network-health experiment %s written to %s.",
                run_dir.name, run_dir)
    return {
        "model": model_path,
        "metrics": metrics_path,
        "manifest": manifest_path,
        "run_dir": run_dir,
    }


def write_canonical_dataset(
    result: Any, output_path: Path, root: Path, dataset_id: str
) -> dict[str, Path]:
    """Persist a converted canonical CSV plus its adapter report (JSON + MD).

    Parameters
    ----------
    result:
        An :class:`~src.network_health.adapters.AdapterResult`.
    output_path:
        Destination canonical CSV (from the dataset's ``output_path``).
    root:
        ``outputs/network_health`` root (adapter report location).
    dataset_id:
        Dataset identity (report subdirectory name).
    """
    from src.utils.io import write_json
    from src.utils.paths import ensure_dir

    csv_path = Path(output_path)
    ensure_dir(csv_path.parent)
    result.frame.to_csv(csv_path, index=False)

    report_dir = ensure_dir(Path(root) / "adapters" / dataset_id)
    payload = {"dataset_id": dataset_id, "output_path": str(csv_path),
               **result.report.to_dict()}
    json_path = write_json(payload, report_dir / "adapter_report.json")
    md_path = report_dir / "adapter_report.md"
    md_path.write_text(_adapter_markdown(payload), encoding="utf-8")
    logger.info("Canonical dataset '%s' written to %s.", dataset_id, csv_path)
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def _adapter_markdown(payload: Mapping[str, Any]) -> str:
    """Render an adapter report as Markdown."""
    lines = [
        f"# Network-Health Adapter Report — {payload['dataset_id']}",
        "",
        f"- Adapter type: `{payload['dataset_type']}`",
        f"- Output: `{payload['output_path']}`",
        f"- Rows: {payload['n_rows']:,} | Devices: {payload['n_devices']} "
        f"| Interfaces: {payload['n_interfaces']}",
        f"- Source columns: {payload['n_source_columns']} | "
        f"Warnings: {payload['n_warnings']}",
        f"- Label mapping applied: {payload['label_mapping_applied']}",
    ]
    span = payload.get("timestamp_span") or {}
    if span:
        lines.append(f"- Timestamp span: {span.get('start')} → {span.get('end')}")
    if payload["mapped_columns"]:
        lines += ["", "## Column mapping (canonical ← raw)", "",
                   "| Canonical | Raw |", "|---|---|"]
        lines += [f"| {c} | {r} |" for c, r in payload["mapped_columns"].items()]
    for title, key in (("Generated columns", "generated_columns"),
                       ("Preserved columns", "preserved_columns"),
                       ("Dropped columns", "dropped_columns")):
        values = payload.get(key) or []
        if values:
            lines += ["", f"## {title}", "", ", ".join(map(str, values))]
    if payload["warnings"]:
        lines += ["", "## Warnings", ""]
        lines += [f"- {w}" for w in payload["warnings"]]
    lines.append("")
    return "\n".join(lines)


def write_report(markdown: str, root: Path) -> Path:
    """Persist the network-health summary report."""
    from src.utils.paths import ensure_dir

    out_dir = ensure_dir(Path(root) / "reports")
    path = out_dir / "network_health_report.md"
    path.write_text(markdown, encoding="utf-8")
    logger.info("Network-health report written to %s.", path)
    return path

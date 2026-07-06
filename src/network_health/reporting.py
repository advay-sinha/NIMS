"""Engine B reporting helper.

Purpose
-------
Summarise one network-health pipeline run — dataset shape, validation
issues, generated features and model outcome — as a Markdown document.
"""

from __future__ import annotations

from typing import Any, Mapping


def network_health_report(
    *,
    dataset_id: str,
    validation: Mapping[str, Any] | None,
    preprocessing_manifest: Mapping[str, Any],
    feature_metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    experiment_id: str,
) -> str:
    """Render the network-health summary report.

    Parameters
    ----------
    dataset_id:
        Telemetry dataset identity.
    validation:
        ``validation_report.json`` payload (``None`` when validation was not
        run in this invocation).
    preprocessing_manifest, feature_metadata:
        Stage artefact payloads.
    metrics:
        Per-split baseline metrics.
    experiment_id:
        The trained run's id.

    Returns
    -------
    str
        The complete Markdown document.
    """
    lines = [
        "# Network Health Report",
        "",
        f"- Dataset: {dataset_id}",
        f"- Experiment: `{experiment_id}`",
        f"- Rows after preprocessing: "
        f"{preprocessing_manifest.get('n_rows_after_preprocessing', 0):,} "
        f"(from {preprocessing_manifest.get('n_raw_rows', 0):,} raw)",
        f"- Splits: {preprocessing_manifest.get('split_rows', {})}",
        "",
    ]
    if validation is not None:
        lines.extend(
            [
                "## Validation",
                "",
                f"- Result: {'PASSED' if validation.get('passed') else 'FAILED'} "
                f"({validation.get('n_errors', 0)} error(s), "
                f"{validation.get('n_warnings', 0)} warning(s))",
                f"- Devices: {validation.get('n_devices', 0)} | Interfaces: "
                f"{validation.get('n_interfaces', 0)}",
                "",
            ]
        )
        issues = validation.get("issues", [])
        if issues:
            lines.extend(
                f"- [{i['severity']}] {i['check']}: {i['message']}"
                for i in issues[:10]
            )
            lines.append("")

    lines.extend(
        [
            "## Features",
            "",
            f"- {feature_metadata.get('n_features', 0)} feature column(s): "
            f"{len(feature_metadata.get('base_features', []))} base, "
            f"{len(feature_metadata.get('rolling_features', []))} rolling, "
            f"{len(feature_metadata.get('lag_features', []))} lag, "
            f"{len(feature_metadata.get('status_change_features', []))} "
            f"status-change",
            "",
            "## Model",
            "",
        ]
    )

    findings: list[str] = []
    for split_name, split_metrics in metrics.items():
        if split_metrics.get("mode") == "labeled":
            lines.append(
                f"- {split_name}: precision={split_metrics['precision']:.4f} "
                f"recall={split_metrics['recall']:.4f} "
                f"f1={split_metrics['f1']:.4f} "
                f"roc_auc={split_metrics['roc_auc'] if split_metrics['roc_auc'] is None else format(split_metrics['roc_auc'], '.4f')}"
            )
        else:
            dist = split_metrics["score_distribution"]
            lines.append(
                f"- {split_name}: anomaly_rate="
                f"{split_metrics['anomaly_rate']:.4f} "
                f"threshold={split_metrics['threshold']:.4f} "
                f"score p50/p99={dist['p50']:.4f}/{dist['p99']:.4f}"
            )
    lines.append("")

    test = metrics.get("test") or {}
    if test.get("mode") == "labeled":
        findings.append(
            f"Test-split anomaly detection reaches F1 {test['f1']:.4f} "
            f"(recall {test['recall']:.4f})."
        )
    elif test:
        findings.append(
            f"{test['anomaly_rate']:.2%} of test readings exceed the anomaly "
            f"threshold."
        )
    if findings:
        lines.extend(["## Key Findings", ""])
        lines.extend(f"- {finding}" for finding in findings)
        lines.append("")
    return "\n".join(lines)

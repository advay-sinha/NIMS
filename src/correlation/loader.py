"""Read-only signal loaders for the correlation engine.

Each loader turns one engine's already-persisted artefacts into normalised
:class:`~src.correlation.models.Signal` objects. Loaders never run a pipeline,
never contact a device and never mutate an artefact; every missing or malformed
artefact is tolerated and recorded as a warning rather than raised, so a
correlation run can proceed on whatever engines are available.

Design note
-----------
Engine A and Engine B currently expose only *aggregate* artefacts (benchmark
metrics, anomaly counts) — there is no per-flow alert log. Their signals are
therefore created conservatively and flagged ``aggregate=True`` so downstream
scoring never treats them as precise, per-event alerts (CLAUDE.md: "Never
fabricate metrics"; prompt: "Do not fabricate detailed alerts").
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.correlation.models import (
    ENGINE_A,
    ENGINE_B,
    ENGINE_C,
    LoadResult,
    Signal,
    signal_id,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- config access


def cfg(config: dict[str, Any], dotted: str, default: Any) -> Any:
    """Fetch a dotted config path with a fallback default."""
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _read_json(path: Path, warnings: list[str], label: str) -> Any:
    if not path.is_file():
        warnings.append(f"{label} artefact not found: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        warnings.append(f"could not read {label} artefact {path.name}: {exc}")
        return None


# --------------------------------------------------------------- Engine C


def load_engine_c_signals(
    snapshot_dir: str | Path, snapshot_id: str, config: dict[str, Any]
) -> LoadResult:
    """Load Engine C findings, topology warnings and device health cards.

    ``inventory.json`` is not required here — the correlation view only needs
    the finding/warning artefacts. A snapshot with none of them yields an empty
    (warned) result rather than an error.
    """
    root = Path(snapshot_dir)
    result = LoadResult(engine=ENGINE_C, source=snapshot_id)
    if not root.is_dir():
        result.warnings.append(f"Engine C snapshot directory not found: {root}")
        return result

    conf_map = cfg(config, "engine_c.finding_confidence",
                   {"high": 0.9, "medium": 0.75, "low": 0.55})
    findings = _read_json(root / "findings.json", result.warnings, "Engine C findings")
    for finding in findings if isinstance(findings, list) else []:
        result.signals.append(_finding_signal(finding, snapshot_id, conf_map))

    topology = _read_json(root / "topology.json", result.warnings, "Engine C topology")
    warn_conf = float(cfg(config, "engine_c.topology_warning_confidence", 0.6))
    warn_sev = cfg(config, "engine_c.topology_warning_severity",
                   {"critical": "critical", "warning": "medium", "info": "info"})
    if isinstance(topology, dict):
        for warning in topology.get("warnings") or []:
            result.signals.append(
                _topology_signal(warning, snapshot_id, warn_conf, warn_sev))

    card_conf = float(cfg(config, "engine_c.device_card_confidence", 0.5))
    card_sev = cfg(config, "engine_c.device_card_severity",
                   {"critical": "high", "warning": "medium"})
    cards = _read_json(root / "dashboard" / "device_health_cards.json",
                       result.warnings, "Engine C device health cards")
    if isinstance(cards, dict):
        for card in cards.get("cards") or []:
            sig = _device_card_signal(card, snapshot_id, card_conf, card_sev)
            if sig is not None:
                result.signals.append(sig)

    logger.info("Engine C '%s': %d signal(s) loaded (offline, read-only).",
                snapshot_id, len(result.signals))
    return result


def _finding_signal(finding: dict[str, Any], snapshot_id: str,
                    conf_map: dict[str, float]) -> Signal:
    src = f"network_config/{snapshot_id}/findings.json"
    category = str(finding.get("category", "config"))
    severity = str(finding.get("severity", "info"))
    device = _opt(finding.get("device"))
    interface = _opt(finding.get("interface"))
    title = str(finding.get("title", finding.get("rule_id", "finding")))
    confidence = float(conf_map.get(str(finding.get("confidence", "medium")), 0.7))
    description = str(finding.get("evidence") or finding.get("recommendation") or "")
    return Signal(
        signal_id=signal_id(ENGINE_C, src, category, device, interface, title),
        engine=ENGINE_C, source_artifact=src, category=category,
        severity=severity, confidence=confidence, title=title,
        description=description, raw_reference=str(finding.get("finding_id", "")),
        device=device, interface=interface, vlan=_opt(finding.get("vlan")),
        aggregate=False,
        tags=tuple(str(t) for t in finding.get("tags") or ())
        + (str(finding.get("rule_id", "")),))


def _topology_signal(warning: dict[str, Any], snapshot_id: str,
                     confidence: float, sev_map: dict[str, str]) -> Signal:
    src = f"network_config/{snapshot_id}/topology.json"
    raw_sev = str(warning.get("severity", "info"))
    severity = str(sev_map.get(raw_sev, "info"))
    category = str(warning.get("category", "topology"))
    device = _opt(warning.get("device"))
    interface = _opt(warning.get("interface"))
    title = str(warning.get("message", "topology warning"))
    return Signal(
        signal_id=signal_id(ENGINE_C, src, f"topology:{category}", device,
                            interface, title),
        engine=ENGINE_C, source_artifact=src, category="topology",
        severity=severity, confidence=confidence, title=title,
        description=str(warning.get("evidence") or ""),
        raw_reference=str(warning.get("warning_id", "")),
        device=device, interface=interface, aggregate=False,
        tags=("topology", category))


def _device_card_signal(card: dict[str, Any], snapshot_id: str,
                        confidence: float, sev_map: dict[str, str]
                        ) -> Optional[Signal]:
    status = str(card.get("status", "")).lower()
    if status not in sev_map:            # healthy / unknown -> no signal
        return None
    src = f"network_config/{snapshot_id}/dashboard/device_health_cards.json"
    device = _opt(card.get("device_id"))
    severity = str(sev_map[status])
    title = f"Device health {status}: {device}"
    return Signal(
        signal_id=signal_id(ENGINE_C, src, "device_health", device, None, title),
        engine=ENGINE_C, source_artifact=src, category="device_health",
        severity=severity, confidence=confidence, title=title,
        description=f"{card.get('finding_count', 0)} finding(s); highest "
                    f"severity {card.get('highest_severity', 'n/a')}.",
        raw_reference=str(device or ""), device=device, aggregate=True,
        tags=("device_health", status))


# --------------------------------------------------------------- Engine B


def load_engine_b_signals(
    network_health_dir: str | Path, dataset: str, config: dict[str, Any]
) -> LoadResult:
    """Load the latest network-health experiment for ``dataset`` as a signal.

    Only aggregate experiment metrics exist, so a single dataset-level anomaly
    signal is emitted (``aggregate=True``). Severity is keyed on the predicted-
    anomaly ratio via configurable thresholds.
    """
    result = LoadResult(engine=ENGINE_B, source=None)
    exp_root = Path(network_health_dir) / "experiments" / dataset
    manifest_path, metrics = _latest_experiment(exp_root, result.warnings)
    if manifest_path is None or metrics is None:
        result.warnings.append(
            f"no usable Engine B experiment for dataset '{dataset}' under "
            f"{exp_root}")
        return result

    manifest = _read_json(manifest_path, result.warnings, "Engine B manifest") or {}
    experiment_id = str(manifest.get("experiment_id", manifest_path.parent.name))
    result.source = experiment_id
    src = (f"network_health/experiments/{dataset}/"
           f"{manifest.get('model_name', 'model')}/{experiment_id}/metrics.json")

    test = metrics.get("test") if isinstance(metrics, dict) else None
    test = test if isinstance(test, dict) else {}
    n_samples = int(test.get("n_samples", 0) or 0)
    n_pred = int(test.get("n_anomalous_predicted", 0) or 0)
    ratio = (n_pred / n_samples) if n_samples else 0.0

    thresholds = cfg(config, "engine_b.anomaly_severity_thresholds",
                     {"high": 0.2, "medium": 0.05})
    severity = _ratio_severity(ratio, thresholds)
    confidence = float(cfg(config, "engine_b.aggregate_confidence", 0.6))

    f1 = test.get("f1")
    roc = test.get("roc_auc")
    description = (
        f"{n_pred} of {n_samples} test rows flagged anomalous "
        f"({ratio:.1%}); model test f1={_fmt(f1)}, roc_auc={_fmt(roc)}. "
        "Aggregate experiment metric — not a live per-interface alert.")
    title = f"Network-health anomalies (aggregate, {dataset})"
    tags = ("anomaly", "degradation", "aggregate")
    result.signals.append(Signal(
        signal_id=signal_id(ENGINE_B, src, "network_health", None, None, title),
        engine=ENGINE_B, source_artifact=src, category="network_health",
        severity=severity, confidence=confidence, title=title,
        description=description, raw_reference=experiment_id,
        timestamp=_opt(manifest.get("created_at")), aggregate=True, tags=tags))

    logger.info("Engine B '%s': anomaly ratio %.3f -> severity %s.",
                dataset, ratio, severity)
    return result


# --------------------------------------------------------------- Engine A


def load_engine_a_signals(
    experiments_dir: str | Path,
    registry_dir: str | Path,
    error_analysis_dir: str | Path,
    dataset: str,
    config: dict[str, Any],
) -> LoadResult:
    """Load an aggregate cyber signal for ``dataset`` from Engine A artefacts.

    The best/promoted experiment for the dataset (via the registry, falling back
    to the newest experiment on disk) supplies a dataset/model-level signal. It
    is tagged with the dataset's known attack families (config-driven) so that
    correlation rules can reason about attack exposure without inventing a
    per-flow alert log that does not exist.
    """
    result = LoadResult(engine=ENGINE_A, source=None)
    experiment_id, model_type = _registry_pick(Path(registry_dir), dataset,
                                                result.warnings)

    metrics_path = _engine_a_metrics_path(Path(experiments_dir), dataset,
                                          experiment_id, model_type,
                                          result.warnings)
    if metrics_path is None:
        result.warnings.append(
            f"no usable Engine A experiment for dataset '{dataset}' under "
            f"{Path(experiments_dir) / dataset}")
        return result

    experiment_id = metrics_path.parent.name
    model_type = model_type or metrics_path.parent.parent.name
    result.source = experiment_id
    metrics = _read_json(metrics_path, result.warnings, "Engine A metrics") or {}

    families = tuple(str(x) for x in cfg(
        config, f"engine_a.attack_families_by_dataset.{dataset}", ()))
    severity = str(cfg(config, "engine_a.aggregate_severity", "medium"))
    confidence = float(cfg(config, "engine_a.aggregate_confidence", 0.6))

    ea = _engine_a_metadata(Path(error_analysis_dir), experiment_id)
    perf = _engine_a_performance(metrics, ea)
    fam_text = ", ".join(families) if families else "attack classes"
    description = (
        f"Engine A intrusion model '{model_type}' promoted for dataset "
        f"'{dataset}' ({fam_text}). {perf} Aggregate model-level indicator — "
        "there is no live per-flow alert stream.")
    title = f"Intrusion-detection coverage (aggregate, {dataset})"
    src = f"experiments/{dataset}/{model_type}/{experiment_id}/metrics.json"
    result.signals.append(Signal(
        signal_id=signal_id(ENGINE_A, src, "intrusion", None, None, title),
        engine=ENGINE_A, source_artifact=src, category="intrusion",
        severity=severity, confidence=confidence, title=title,
        description=description, raw_reference=experiment_id,
        aggregate=True, tags=("attack", "aggregate") + families))

    logger.info("Engine A '%s': aggregate cyber signal from %s.",
                dataset, experiment_id)
    return result


# --------------------------------------------------------------- helpers


def _opt(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _ratio_severity(ratio: float, thresholds: dict[str, float]) -> str:
    if ratio >= float(thresholds.get("high", 0.2)):
        return "high"
    if ratio >= float(thresholds.get("medium", 0.05)):
        return "medium"
    if ratio > 0:
        return "low"
    return "info"


def _latest_experiment(
    exp_root: Path, warnings: list[str]
) -> tuple[Optional[Path], Optional[dict[str, Any]]]:
    """Return (manifest_path, metrics) for the newest run under ``exp_root``."""
    if not exp_root.is_dir():
        return None, None
    runs = sorted(
        (p for p in exp_root.glob("*/*") if (p / "metrics.json").is_file()),
        key=lambda p: p.name)
    if not runs:
        return None, None
    run = runs[-1]                       # timestamped run ids sort chronologically
    metrics = _read_json(run / "metrics.json", warnings, "Engine B metrics")
    return run / "manifest.json", metrics if isinstance(metrics, dict) else None


def _registry_pick(
    registry_dir: Path, dataset: str, warnings: list[str]
) -> tuple[Optional[str], Optional[str]]:
    """Resolve (experiment_id, model_type) from the registry for ``dataset``."""
    for name in ("production.json", "best_per_dataset.json"):
        data = _read_json(registry_dir / name, [], "registry")
        entry = data.get(dataset) if isinstance(data, dict) else None
        if isinstance(entry, dict) and entry.get("experiment_id"):
            return str(entry["experiment_id"]), _opt(entry.get("model_type"))
    warnings.append(
        f"no registry entry for dataset '{dataset}'; falling back to newest "
        "experiment on disk")
    return None, None


def _engine_a_metrics_path(
    experiments_dir: Path, dataset: str, experiment_id: Optional[str],
    model_type: Optional[str], warnings: list[str]
) -> Optional[Path]:
    dataset_root = experiments_dir / dataset
    if experiment_id:
        if model_type:
            candidate = dataset_root / model_type / experiment_id / "metrics.json"
            if candidate.is_file():
                return candidate
        matches = list(dataset_root.glob(f"*/{experiment_id}/metrics.json"))
        if matches:
            return matches[0]
        warnings.append(
            f"registry experiment '{experiment_id}' not found on disk for "
            f"dataset '{dataset}'; using newest experiment instead")
    runs = sorted(
        (p for p in dataset_root.glob("*/*") if (p / "metrics.json").is_file()),
        key=lambda p: p.name)
    return runs[-1] / "metrics.json" if runs else None


def _engine_a_metadata(error_analysis_dir: Path,
                       experiment_id: str) -> Optional[dict[str, Any]]:
    path = error_analysis_dir / experiment_id / "metadata.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _engine_a_performance(metrics: dict[str, Any],
                          ea: Optional[dict[str, Any]]) -> str:
    """Human-readable one-liner of model performance (best-effort, defensive)."""
    test = metrics.get("test") if isinstance(metrics, dict) else None
    if isinstance(test, dict) and test.get("f1") is not None:
        return f"Test f1={_fmt(test.get('f1'))}, roc_auc={_fmt(test.get('roc_auc'))}."
    if ea and ea.get("accuracy") is not None:
        return (f"Accuracy={_fmt(ea.get('accuracy'))}, "
                f"macro_f1={_fmt(ea.get('macro_f1'))}.")
    return "Model performance metrics available in the experiment manifest."

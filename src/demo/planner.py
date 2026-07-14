"""Deterministic planner for the offline full-demo workflow.

Inspects existing artefacts and produces an ordered list of stages, each either
a pure-Python readiness check or a set of **allowlisted** local command
invocations. Building a plan runs nothing — execution is a separate, explicit
step (see :mod:`src.demo.runner`). Engine A/B training commands are produced by
reusing the existing ``src.ml_workflow`` planner and the network-health scripts;
this module never re-implements a training or correlation pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from src.demo import readiness
from src.demo.models import (
    KIND_COMMAND,
    KIND_INTERNAL,
    PENDING,
    REUSED_EXISTING,
    DemoCommand,
    DemoConfig,
    StageResult,
)
from src.ml_workflow import planner as ml

logger = logging.getLogger(__name__)

_PROMOTE_REASON = "Prepared for offline full-system demo"
# Engine A training steps (lean; explainability/error-analysis excluded here).
_ENGINE_A_TRAIN_STEPS = ("validate", "preprocess", "features", "train",
                         "validation_report", "experiment_index", "registry")


def resolve_config(demo_yaml: Mapping[str, Any], overrides: Mapping[str, Any]
                   ) -> DemoConfig:
    """Merge ``configs/demo.yaml`` defaults with CLI overrides into a DemoConfig."""
    demo = dict(demo_yaml.get("demo", {}) or {})
    syslog_block = dict(demo_yaml.get("syslog", {}) or {})

    def pick(key: str, default: Any) -> Any:
        val = overrides.get(key)
        if val is not None:
            return val
        return demo.get(key, default)

    datasets = pick("engine_a_datasets",
                    ["nsl_kdd", "unsw_nb15", "cicids2017"])
    if isinstance(datasets, str):
        datasets = ml.expand_datasets([datasets])
    else:
        datasets = ml.expand_datasets(list(datasets))

    return DemoConfig(
        engine_c_input_dir=str(pick("engine_c_input_dir",
                                    "datasets/samples/network_config")),
        engine_c_snapshot=str(pick("engine_c_snapshot", "demo_assessment")),
        engine_a_datasets=tuple(datasets),
        engine_a_model=str(pick("engine_a_model", "xgboost")),
        engine_b_dataset=str(pick("engine_b_dataset", "synthetic")),
        engine_b_model=str(pick("engine_b_model", "isolation_forest")),
        syslog_run=str(pick("syslog_run", "latest")),
        correlation_id=str(pick("correlation_id", "demo_full_correlation")),
        refresh_engine_c=bool(pick("refresh_engine_c", True)),
        conditional_training=bool(pick("conditional_training", True)),
        skip_training=bool(overrides.get("skip_training", False)),
        force_train_engine_a=bool(overrides.get("force_train_engine_a", False)),
        force_train_engine_b=bool(overrides.get("force_train_engine_b", False)),
        require_syslog=bool(overrides.get("require_syslog", False)),
        reuse_assessment=bool(overrides.get("reuse_assessment", False)),
        continue_on_error=bool(overrides.get("continue_on_error", False)),
        launch_dashboard=bool(pick("launch_dashboard", False)),
        dry_run=bool(overrides.get("dry_run", False)),
        syslog_fallback_fixture=(syslog_block.get("fallback_fixture")
                                 if syslog_block.get("fallback_fixture") else None),
    )


def build_plan(config: DemoConfig, paths) -> list[StageResult]:
    """Build the ordered demo plan as pre-populated (PENDING) stage results."""
    stages: list[StageResult] = [
        _env_stage(config, paths),
        _engine_a_stage(config, paths),
        _engine_b_stage(config, paths),
        _engine_c_assessment_stage(config, paths),
        _syslog_stage(config, paths),
        _engine_c_dashboard_stage(config, paths),
        _correlation_stage(config, paths),
        _streaming_stage(config),
        _frontend_readiness_stage(config),
        _final_report_stage(),
    ]
    return stages


# --------------------------------------------------------------- stages
def _env_stage(config: DemoConfig, paths) -> StageResult:
    return StageResult(name="env_validation", title="Environment & configuration",
                       kind=KIND_INTERNAL, required=True,
                       details={"recheck": "env"})


def _engine_a_stage(config: DemoConfig, paths) -> StageResult:
    ready = readiness.engine_a_ready(paths.registry_dir, config.engine_a_datasets)
    stage = StageResult(name="engine_a", title="Engine A model readiness",
                        required=True, details={"recheck": "engine_a",
                                                "readiness": ready})
    if config.force_train_engine_a:
        train_for = list(config.engine_a_datasets)
    else:
        train_for = ready["missing_datasets"]

    if not train_for:
        stage.status = REUSED_EXISTING
        stage.reused = True
        stage.details["reused_datasets"] = ready["ready_datasets"]
        return stage

    if config.skip_training:
        stage.status = PENDING       # runner recheck will fail (required, missing)
        stage.warnings.append(
            f"--skip-training set but Engine A model(s) missing for: "
            f"{', '.join(train_for)}.")
        return stage

    stage.commands = _engine_a_training_commands(train_for, config.engine_a_model)
    stage.details["train_datasets"] = train_for
    return stage


def _engine_a_training_commands(datasets: list[str], model: str
                                ) -> list[DemoCommand]:
    """Reuse the ml_workflow planner for training, then promote with a reason."""
    plan = ml.build_plan(datasets, [model], list(_ENGINE_A_TRAIN_STEPS))
    cmds = [DemoCommand(module=s.module, args=s.args) for s in plan]
    for dataset in datasets:
        cmds.append(DemoCommand(
            module="scripts.promote_model",
            args=("--dataset", dataset, "--model", model,
                  "--reason", _PROMOTE_REASON)))
    return cmds


def _engine_b_stage(config: DemoConfig, paths) -> StageResult:
    ready = readiness.engine_b_ready(paths.network_health_dir,
                                     config.engine_b_dataset)
    stage = StageResult(name="engine_b", title="Engine B experiment readiness",
                        required=True, details={"recheck": "engine_b",
                                                "readiness": ready})
    needs_train = config.force_train_engine_b or not ready["ready"]
    if not needs_train:
        stage.status = REUSED_EXISTING
        stage.reused = True
        return stage
    if config.skip_training:
        stage.status = PENDING
        stage.warnings.append(
            f"--skip-training set but no Engine B experiment for "
            f"'{config.engine_b_dataset}'.")
        return stage
    d = config.engine_b_dataset
    stage.commands = [
        DemoCommand("scripts.prepare_network_health_dataset", ("--dataset", d)),
        DemoCommand("scripts.validate_network_health", ("--dataset", d)),
        DemoCommand("scripts.run_network_health_preprocessing", ("--dataset", d)),
        DemoCommand("scripts.train_network_health_model",
                    ("--dataset", d, "--model", config.engine_b_model)),
    ]
    return stage


def _engine_c_assessment_stage(config: DemoConfig, paths) -> StageResult:
    ready = readiness.engine_c_ready(paths.network_config_dir,
                                     config.engine_c_snapshot)
    stage = StageResult(name="engine_c_assessment", title="Engine C assessment",
                        required=True, details={"recheck": "engine_c",
                                                "readiness": ready})
    snap = config.engine_c_snapshot
    if config.reuse_assessment and ready["ready"]:
        stage.status = REUSED_EXISTING
        stage.reused = True
        return stage
    stage.commands = [
        DemoCommand("scripts.analyze_network_config",
                    ("--input-dir", config.engine_c_input_dir,
                     "--snapshot-id", snap)),
        DemoCommand("scripts.dry_run_network_actions", ("--snapshot-id", snap)),
        DemoCommand("scripts.generate_network_config_report",
                    ("--snapshot-id", snap)),
    ]
    return stage


def _engine_c_dashboard_stage(config: DemoConfig, paths) -> StageResult:
    return StageResult(
        name="engine_c_dashboard", title="Engine C dashboard export",
        required=True,
        commands=[DemoCommand("scripts.export_network_config_dashboard",
                              ("--snapshot-id", config.engine_c_snapshot))],
        details={"recheck": "engine_c_dashboard"})


def _syslog_stage(config: DemoConfig, paths) -> StageResult:
    ready = readiness.syslog_ready(paths.outputs_dir, config.syslog_run)
    stage = StageResult(name="syslog", title="Syslog ingestion readiness",
                        required=config.require_syslog,
                        details={"recheck": "syslog", "readiness": ready})
    if ready["ready"]:
        stage.status = REUSED_EXISTING
        stage.reused = True
        stage.details["run_id"] = ready["run_id"]
        return stage
    fixture = config.syslog_fallback_fixture
    if fixture:
        arg = "--input-dir" if _looks_like_dir(fixture) else "--input-file"
        stage.commands = [DemoCommand(
            "scripts.ingest_switch_syslog",
            (arg, fixture, "--run-id", "demo_syslog"))]
        return stage
    # No run and no fixture: optional unless --require-syslog.
    stage.warnings.append(
        "No syslog ingestion run found and no fallback fixture configured; "
        "continuing without syslog evidence.")
    stage.status = PENDING  # runner marks SKIPPED (optional) or FAILED (required)
    return stage


def _correlation_stage(config: DemoConfig, paths) -> StageResult:
    ready = readiness.engine_a_ready(paths.registry_dir, config.engine_a_datasets)
    a_dataset = ("unsw_nb15" if "unsw_nb15" in ready["ready_datasets"]
                 else (ready["ready_datasets"][0] if ready["ready_datasets"]
                       else config.engine_a_datasets[0]))
    args = ["--engine-c-snapshot", config.engine_c_snapshot,
            "--engine-b-dataset", config.engine_b_dataset,
            "--engine-a-dataset", a_dataset,
            "--correlation-id", config.correlation_id]
    syslog = readiness.syslog_ready(paths.outputs_dir, config.syslog_run)
    if syslog["ready"]:
        args += ["--syslog-run", config.syslog_run]
    return StageResult(
        name="correlation", title="Unified syslog-enhanced correlation",
        required=True, commands=[DemoCommand("scripts.run_correlation",
                                             tuple(args))],
        details={"recheck": "correlation", "engine_a_dataset": a_dataset,
                 "with_syslog": syslog["ready"]})


def _streaming_stage(config: DemoConfig) -> StageResult:
    return StageResult(
        name="streaming", title="Streaming replay / current state", required=True,
        commands=[DemoCommand("scripts.run_streaming_demo",
                              ("--correlation-id", config.correlation_id,
                               "--snapshot-id", config.engine_c_snapshot,
                               "--no-sleep"))],
        details={"recheck": "streaming"})


def _frontend_readiness_stage(config: DemoConfig) -> StageResult:
    return StageResult(
        name="frontend_readiness", title="Frontend readiness + safety validation",
        required=True,
        commands=[DemoCommand("scripts.validate_engine_c_safety", ())],
        details={"recheck": "dashboard"})


def _final_report_stage() -> StageResult:
    return StageResult(name="final_report", title="Final demo-readiness report",
                       kind=KIND_INTERNAL, required=True,
                       details={"recheck": "report"})


def _looks_like_dir(path: str) -> bool:
    from pathlib import Path
    p = Path(path)
    return p.is_dir() or not p.suffix

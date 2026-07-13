"""Deterministic planner for the offline Engine A ML workflow.

Maps a selection of datasets, models and workflow steps onto the exact,
already-existing offline entry points under ``scripts/`` and returns an ordered,
inspectable plan. Pure Python — building a plan runs nothing; execution is a
separate, explicit step in the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Offline Engine A datasets and models (the only things this workflow touches).
DATASETS: tuple[str, ...] = ("nsl_kdd", "unsw_nb15", "cicids2017")
MODELS: tuple[str, ...] = ("xgboost", "lightgbm", "isolation_forest", "mlp",
                           "cnn", "lstm", "transformer")

# Canonical step order. ``scope`` decides how many commands a step expands to:
#   dataset -> one per dataset; model -> one per (dataset, model); global -> one.
_STEP_SPECS: tuple[tuple[str, str, str], ...] = (
    ("validate", "scripts.validate_datasets", "dataset"),
    ("audit", "scripts.run_audit", "dataset"),
    ("preprocess", "scripts.run_preprocessing", "dataset"),
    ("features", "scripts.run_feature_engineering", "dataset"),
    ("train", "scripts.train_model", "model"),
    ("validation_report", "scripts.generate_validation_report", "global"),
    ("experiment_index", "scripts.build_experiment_index", "global"),
    ("explainability", "scripts.run_explainability", "model"),
    ("error_analysis", "scripts.run_error_analysis", "model"),
    ("visualizations", "scripts.run_visualizations", "model"),
    ("registry", "scripts.build_model_registry", "global"),
    ("promote", "scripts.promote_model", "model"),
    ("resolve", "scripts.resolve_model", "dataset"),
)
STEP_ORDER: tuple[str, ...] = tuple(name for name, _, _ in _STEP_SPECS)
_STEP_INDEX = {name: i for i, name in enumerate(STEP_ORDER)}
_MODULE = {name: mod for name, mod, _ in _STEP_SPECS}
_SCOPE = {name: scope for name, _, scope in _STEP_SPECS}

# Friendly aliases accepted from the CLI / dashboard (prompt wording).
STEP_ALIASES: dict[str, str] = {
    "validate dataset": "validate", "validate": "validate",
    "audit dataset": "audit", "audit": "audit",
    "preprocess dataset": "preprocess", "preprocess": "preprocess",
    "feature engineering": "features", "features": "features",
    "train model": "train", "train": "train",
    "generate validation report": "validation_report",
    "validation report": "validation_report", "validation_report": "validation_report",
    "build experiment index": "experiment_index",
    "experiment index": "experiment_index", "experiment_index": "experiment_index",
    "run explainability": "explainability", "explainability": "explainability",
    "run error analysis": "error_analysis", "error analysis": "error_analysis",
    "error_analysis": "error_analysis",
    "run visualizations": "visualizations", "visualizations": "visualizations",
    "build model registry": "registry", "model registry": "registry",
    "registry": "registry",
    "promote best model": "promote", "promote model": "promote", "promote": "promote",
    "resolve production model": "resolve", "resolve model": "resolve",
    "resolve": "resolve",
}


@dataclass(frozen=True)
class WorkflowStep:
    """One resolved, runnable offline command in the plan."""

    step: str
    module: str
    scope: str
    args: tuple[str, ...] = ()
    dataset: Optional[str] = None
    model: Optional[str] = None

    @property
    def display(self) -> str:
        """The exact command a user would type (for dry-run / dashboard)."""
        return " ".join(("python", "-m", self.module, *self.args))


def expand_datasets(selected: list[str]) -> list[str]:
    """Resolve dataset selection (``all`` -> every dataset), validated + ordered."""
    if not selected or "all" in selected:
        return list(DATASETS)
    unknown = [d for d in selected if d not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}. "
                         f"Valid: {', '.join(DATASETS)} or 'all'.")
    return [d for d in DATASETS if d in set(selected)]


def expand_models(selected: list[str]) -> list[str]:
    """Resolve model selection (``all`` -> every model), validated + ordered."""
    if not selected or "all" in selected:
        return list(MODELS)
    unknown = [m for m in selected if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}. "
                         f"Valid: {', '.join(MODELS)} or 'all'.")
    return [m for m in MODELS if m in set(selected)]


def normalize_steps(selected: list[str]) -> list[str]:
    """Resolve step selection (aliases + ``all``) to canonical keys, in order."""
    if not selected or "all" in selected:
        return list(STEP_ORDER)
    resolved: list[str] = []
    for raw in selected:
        key = STEP_ALIASES.get(raw.strip().lower())
        if key is None:
            raise ValueError(
                f"Unknown workflow step: '{raw}'. Valid steps: "
                f"{', '.join(STEP_ORDER)} (or 'all').")
        if key not in resolved:
            resolved.append(key)
    return sorted(resolved, key=lambda s: _STEP_INDEX[s])


def build_plan(datasets: list[str], models: list[str], steps: list[str]
               ) -> list[WorkflowStep]:
    """Build the ordered, offline-only workflow plan.

    Steps run in canonical order; within a step, dataset-scoped commands expand
    per dataset and model-scoped commands per (dataset, model). Global steps
    (validation report, experiment index, registry) run once.
    """
    ds = expand_datasets(datasets)
    ms = expand_models(models)
    step_keys = normalize_steps(steps)

    plan: list[WorkflowStep] = []
    for step in step_keys:
        module, scope = _MODULE[step], _SCOPE[step]
        if scope == "global":
            plan.append(WorkflowStep(step=step, module=module, scope=scope))
        elif scope == "dataset":
            for dataset in ds:
                plan.append(WorkflowStep(
                    step=step, module=module, scope=scope,
                    args=("--dataset", dataset), dataset=dataset))
        else:  # model
            for dataset in ds:
                for model in ms:
                    plan.append(WorkflowStep(
                        step=step, module=module, scope=scope,
                        args=("--dataset", dataset, "--model", model),
                        dataset=dataset, model=model))
    return plan

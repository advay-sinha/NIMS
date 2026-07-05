"""Registry persistence (file-based, deterministic).

Purpose
-------
Read and write the three registry files::

    outputs/registry/
        registry.json          # every registered experiment
        best_per_dataset.json  # best candidate per dataset (by config metric)
        production.json        # explicit production assignments

Rebuilds are idempotent: existing entries keep their ``registered_at``,
``tags`` and lifecycle ``status`` (production status is re-derived from
``production.json``, the single source of promotion truth).

Limitations
-----------
No locking — the registry is single-writer by design (CLI scripts).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.registry.registry import (
    RegistryError,
    entry_from_manifest,
    filter_registrable,
    select_best_per_dataset,
)

logger = logging.getLogger(__name__)

REGISTRY_FILE = "registry.json"
BEST_FILE = "best_per_dataset.json"
PRODUCTION_FILE = "production.json"


def load_registry(registry_dir: Path) -> dict[str, Any]:
    """Load ``registry.json`` (empty structure when absent)."""
    from src.utils.io import read_json

    path = Path(registry_dir) / REGISTRY_FILE
    if not path.is_file():
        return {"generated_at": None, "entries": []}
    return read_json(path)


def load_production(registry_dir: Path) -> dict[str, Any]:
    """Load ``production.json`` (empty mapping when absent)."""
    from src.utils.io import read_json

    path = Path(registry_dir) / PRODUCTION_FILE
    return read_json(path) if path.is_file() else {}


def rebuild_registry(
    experiments_dir: Path,
    registry_dir: Path,
    roots: Mapping[str, Path],
    *,
    selection_metric: str = "test_f1",
    higher_is_better: bool = True,
    require_test_metrics: bool = True,
    allow_optimized_models: bool = True,
) -> dict[str, Any]:
    """Scan all experiment manifests and rewrite the registry files.

    Existing entries keep their ``registered_at``/``tags``/``status``;
    production status is re-applied from ``production.json``. Invalid
    manifests are skipped with a warning, never fatal.

    Parameters
    ----------
    experiments_dir:
        Root of the experiment tree.
    registry_dir:
        Registry output directory.
    roots:
        Optional-artefact roots (see
        :func:`src.registry.registry.entry_from_manifest`).
    selection_metric, higher_is_better:
        Best-candidate selection policy.
    require_test_metrics, allow_optimized_models:
        Registration policy.

    Returns
    -------
    dict
        The written registry document.
    """
    from src.utils.io import write_json
    from src.utils.paths import ensure_dir

    previous = {
        entry["experiment_id"]: entry
        for entry in load_registry(registry_dir).get("entries", [])
    }
    production = load_production(registry_dir)
    production_ids = {
        assignment.get("experiment_id") for assignment in production.values()
    }

    entries: list[dict[str, Any]] = []
    for manifest_path in sorted(Path(experiments_dir).glob("*/*/*/manifest.json")):
        old = previous.get(manifest_path.parent.name, {})
        try:
            entry = entry_from_manifest(
                manifest_path, roots,
                tags=old.get("tags", ()),
                status=old.get("status", "candidate"),
                registered_at=old.get("registered_at"),
            )
        except RegistryError as exc:
            logger.warning("Skipping unregistrable experiment: %s", exc)
            continue
        # production.json is authoritative for the production status.
        if entry["experiment_id"] in production_ids:
            entry["status"] = "production"
        elif entry["status"] == "production":
            entry["status"] = "candidate"
        entries.append(entry)

    entries = filter_registrable(
        entries,
        require_test_metrics=require_test_metrics,
        allow_optimized_models=allow_optimized_models,
    )
    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_metric": selection_metric,
        "entries": entries,
    }
    best = select_best_per_dataset(
        entries,
        selection_metric=selection_metric,
        higher_is_better=higher_is_better,
    )

    out_dir = ensure_dir(registry_dir)
    write_json(document, out_dir / REGISTRY_FILE)
    write_json(best, out_dir / BEST_FILE)
    if not (out_dir / PRODUCTION_FILE).is_file():
        write_json({}, out_dir / PRODUCTION_FILE)
    logger.info(
        "Registry rebuilt: %d entrie(s), %d dataset best pick(s) -> %s",
        len(entries), len(best), out_dir,
    )
    return document


def promote(
    registry_dir: Path,
    *,
    dataset: str,
    experiment_id: str,
    reason: str,
) -> dict[str, Any]:
    """Assign one registered experiment as the production model for a dataset.

    Updates ``production.json`` and the entry statuses in ``registry.json``
    (previous production entry for the dataset returns to ``candidate``).
    Model files are never moved, copied or deleted.

    Parameters
    ----------
    registry_dir:
        Registry directory (must contain a built ``registry.json``).
    dataset, experiment_id:
        The assignment.
    reason:
        Human-readable promotion rationale.

    Returns
    -------
    dict
        The dataset's new production assignment.

    Raises
    ------
    RegistryError
        When the experiment is not registered for the dataset.
    """
    from src.utils.io import write_json

    document = load_registry(registry_dir)
    entries: Sequence[dict[str, Any]] = document.get("entries", [])
    target = next(
        (e for e in entries
         if e["experiment_id"] == experiment_id and e["dataset"] == dataset),
        None,
    )
    if target is None:
        registered = sorted(
            e["experiment_id"] for e in entries if e["dataset"] == dataset
        )
        raise RegistryError(
            f"Experiment '{experiment_id}' is not registered for dataset "
            f"'{dataset}'. Registered runs: {registered or 'none'} — run "
            f"'python -m scripts.build_model_registry' first."
        )

    for entry in entries:
        if entry["dataset"] == dataset and entry["status"] == "production":
            entry["status"] = "candidate"
    target["status"] = "production"

    production = load_production(registry_dir)
    assignment = {
        "experiment_id": experiment_id,
        "model_type": target["model_type"],
        "registered_at": target["registered_at"],
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    production[dataset] = assignment

    write_json(document, Path(registry_dir) / REGISTRY_FILE)
    write_json(production, Path(registry_dir) / PRODUCTION_FILE)
    logger.info("Promoted %s to production for %s (%s).",
                experiment_id, dataset, reason)
    return assignment

"""Model validation / benchmark report generation.

Purpose
-------
Aggregate the persisted experiment manifests (Layer 4 reads Layer 3's outputs;
no model is loaded or re-run) into a single Markdown benchmark report:
per-model summaries, a cross-model comparison table, timings, model sizes and
train/validation/test metrics for every dataset.

Inputs
------
``<experiments_dir>/<dataset>/<model>/<run_id>/manifest.json`` files.

Outputs
-------
A Markdown string (persisted by the caller, e.g.
``outputs/reports/model_validation_report.md``) containing an executive
summary, best-model and ranking tables, an overall (cross-dataset) model
ranking, a classical-vs-deep comparison, an efficiency analysis, key
findings, a reproducibility section and the per-dataset detail tables.

Limitations
-----------
Only the LATEST run per (dataset, model) — by manifest ``created_at`` — is
reported; historical runs remain on disk untouched. Summary metrics use each
run's test split, falling back to validation then train when absent, with
the evaluation average configured at training time (e.g. weighted).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

_SPLITS = ("train", "validation", "test")

# Preference order for the single "headline" split used in summary tables.
_SPLIT_PREFERENCE = ("test", "validation", "train")

# Model-family classification for the classical-vs-deep comparison. Names are
# registry names (lowercase); unknown models are reported as unclassified.
CLASSICAL_MODEL_NAMES = frozenset(
    {
        "decision_tree",
        "random_forest",
        "logistic_regression",
        "isolation_forest",
        "xgboost",
        "lightgbm",
        "catboost",
    }
)
DEEP_MODEL_NAMES = frozenset(
    {"mlp", "cnn", "lstm", "gru", "transformer", "autoencoder"}
)


def collect_latest_manifests(experiments_dir: Path) -> dict[str, dict[str, Any]]:
    """Return the latest manifest per (dataset, model).

    Parameters
    ----------
    experiments_dir:
        Root experiments directory.

    Returns
    -------
    dict
        ``{dataset_id: {model_name: manifest_dict}}``, empty when no
        experiments exist.
    """
    from src.utils.io import read_json

    latest: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(Path(experiments_dir).glob("*/*/*/manifest.json")):
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable manifest %s: %s", manifest_path, exc)
            continue
        dataset = manifest.get("dataset_id")
        model = manifest.get("model_name")
        if not dataset or not model:
            logger.warning("Skipping manifest without identity: %s", manifest_path)
            continue
        current = latest.setdefault(dataset, {}).get(model)
        if current is None or manifest.get("created_at", "") > current.get("created_at", ""):
            latest[dataset][model] = manifest
    return latest


def count_manifests(experiments_dir: Path) -> int:
    """Count every persisted experiment manifest (completed runs).

    Parameters
    ----------
    experiments_dir:
        Root experiments directory.

    Returns
    -------
    int
        Number of ``manifest.json`` files across all runs (not only the
        latest per (dataset, model)).
    """
    return sum(1 for _ in Path(experiments_dir).glob("*/*/*/manifest.json"))


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a metric value for a Markdown table cell."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _split_metrics(manifest: dict[str, Any], split: str) -> dict[str, Any]:
    """Return one split's metrics dict (empty when the split was absent)."""
    return manifest.get("metrics", {}).get(split, {}) or {}


def _model_section(model_name: str, manifest: dict[str, Any]) -> list[str]:
    """Render one model's summary section."""
    timings = manifest.get("timings", {})
    model_info = manifest.get("model", {})
    size_kb = timings.get("model_size_bytes", 0) / 1024

    lines = [
        f"### {model_name}",
        "",
        f"- Run: `{manifest.get('experiment_id')}`",
        f"- Device: `{model_info.get('device', 'unknown')}`"
        f" | Training time: {_fmt(timings.get('train_seconds'), 2)} s"
        f" | Model size: {size_kb:,.1f} KB",
        "",
        "| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR (macro) "
        "| Predict (s) |",
        "|-------|---------:|----------:|-------:|---:|--------:|------------:"
        "|------------:|",
    ]
    for split in _SPLITS:
        metrics = _split_metrics(manifest, split)
        if not metrics:
            continue
        fpr = (metrics.get("false_positive_rate") or {}).get("macro")
        predict_s = timings.get(f"predict_{split}_seconds")
        lines.append(
            f"| {split} | {_fmt(metrics.get('accuracy'))} "
            f"| {_fmt(metrics.get('precision'))} | {_fmt(metrics.get('recall'))} "
            f"| {_fmt(metrics.get('f1'))} | {_fmt(metrics.get('roc_auc'))} "
            f"| {_fmt(fpr, 5)} | {_fmt(predict_s, 3)} |"
        )
    lines.append("")
    return lines


def _comparison_table(models: dict[str, dict[str, Any]]) -> list[str]:
    """Render the cross-model comparison table for one dataset."""
    lines = [
        "| Model | Val F1 | Test F1 | Test ROC-AUC | Train time (s) "
        "| Predict test (s) | Size (KB) |",
        "|-------|-------:|--------:|-------------:|---------------:"
        "|-----------------:|----------:|",
    ]
    for model_name, manifest in sorted(models.items()):
        timings = manifest.get("timings", {})
        val = _split_metrics(manifest, "validation")
        test = _split_metrics(manifest, "test")
        lines.append(
            f"| {model_name} | {_fmt(val.get('f1'))} | {_fmt(test.get('f1'))} "
            f"| {_fmt(test.get('roc_auc'))} | {_fmt(timings.get('train_seconds'), 2)} "
            f"| {_fmt(timings.get('predict_test_seconds'), 3)} "
            f"| {timings.get('model_size_bytes', 0) / 1024:,.1f} |"
        )
    lines.append("")
    return lines


@dataclass(frozen=True)
class _RunSummary:
    """One (dataset, model) benchmark row derived from its latest manifest.

    Metric fields come from the run's headline split (test, falling back to
    validation then train); ``None`` marks values the run did not record.
    """

    dataset: str
    model: str
    run_id: str
    split: str
    accuracy: float | None
    f1: float | None
    roc_auc: float | None
    train_seconds: float | None
    predict_seconds: float | None
    size_bytes: float | None
    gpu: str | None


def _summarise_run(
    dataset: str, model: str, manifest: Mapping[str, Any]
) -> _RunSummary:
    """Build the benchmark summary row for one manifest."""
    split = next(
        (s for s in _SPLIT_PREFERENCE if _split_metrics(dict(manifest), s)), ""
    )
    metrics = _split_metrics(dict(manifest), split) if split else {}
    timings = manifest.get("timings", {}) or {}
    hardware = manifest.get("hardware", {}) or {}
    return _RunSummary(
        dataset=dataset,
        model=model,
        run_id=str(manifest.get("experiment_id", "")),
        split=split or "—",
        accuracy=metrics.get("accuracy"),
        f1=metrics.get("f1"),
        roc_auc=metrics.get("roc_auc"),
        train_seconds=timings.get("train_seconds"),
        predict_seconds=timings.get(f"predict_{split}_seconds") if split else None,
        size_bytes=timings.get("model_size_bytes"),
        gpu=hardware.get("gpu_name") or hardware.get("device"),
    )


def _run_summaries(latest: dict[str, dict[str, Any]]) -> list[_RunSummary]:
    """Flatten the latest manifests into deterministic benchmark rows."""
    return [
        _summarise_run(dataset, model, latest[dataset][model])
        for dataset in sorted(latest)
        for model in sorted(latest[dataset])
    ]


def _desc(value: float | None) -> float:
    """Sort key placing ``None`` last in a descending metric ordering."""
    return value if value is not None else float("-inf")


def _mean(values: Iterable[float | None]) -> float | None:
    """Mean of the non-``None`` values, or ``None`` when none exist."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _ranked(rows: list[_RunSummary]) -> list[_RunSummary]:
    """Rank rows by F1, then accuracy, then ROC-AUC (all descending)."""
    return sorted(
        rows,
        key=lambda r: (-_desc(r.f1), -_desc(r.accuracy), -_desc(r.roc_auc), r.model),
    )


def _by_dataset(summaries: list[_RunSummary]) -> dict[str, list[_RunSummary]]:
    """Group summary rows by dataset (insertion order is already sorted)."""
    grouped: dict[str, list[_RunSummary]] = {}
    for row in summaries:
        grouped.setdefault(row.dataset, []).append(row)
    return grouped


def _size_kb(size_bytes: float | None) -> str:
    """Format a model size in KB for a table cell."""
    return f"{size_bytes / 1024:,.1f}" if size_bytes is not None else "—"


def _model_family(model: str) -> str | None:
    """Return ``"Classical"``/``"Deep"`` for a known model name, else ``None``."""
    if model in CLASSICAL_MODEL_NAMES:
        return "Classical"
    if model in DEEP_MODEL_NAMES:
        return "Deep"
    return None


# ------------------------------------------------------- benchmark sections


def _executive_summary(
    summaries: list[_RunSummary], total_experiments: int, stamp: str
) -> list[str]:
    """Render section 1: dataset/model/experiment counts and hardware."""
    datasets = sorted({s.dataset for s in summaries})
    models = sorted({s.model for s in summaries})
    hardware = sorted({s.gpu for s in summaries if s.gpu})
    return [
        "## 1. Executive Summary",
        "",
        f"- Datasets benchmarked: {len(datasets)} ({', '.join(datasets)})",
        f"- Models benchmarked: {len(models)} ({', '.join(models)})",
        f"- Completed experiments: {total_experiments}",
        f"- Hardware: {', '.join(hardware) if hardware else 'unknown'}",
        f"- Report generated: {stamp}",
        "",
        "Summary metrics use each run's test split (falling back to "
        "validation/train when absent) with the evaluation average configured "
        "at training time.",
        "",
    ]


def _best_per_dataset_section(summaries: list[_RunSummary]) -> list[str]:
    """Render section 2: the top-ranked model per dataset, sorted by F1."""
    best = [_ranked(rows)[0] for rows in _by_dataset(summaries).values()]
    best.sort(key=lambda r: (-_desc(r.f1), r.dataset))
    lines = [
        "## 2. Best Model Per Dataset",
        "",
        "| Dataset | Best Model | Accuracy | F1 | ROC-AUC | Train time (s) "
        "| Run ID |",
        "|---------|------------|---------:|---:|--------:|---------------:"
        "|--------|",
    ]
    for row in best:
        lines.append(
            f"| {row.dataset} | {row.model} | {_fmt(row.accuracy)} "
            f"| {_fmt(row.f1)} | {_fmt(row.roc_auc)} "
            f"| {_fmt(row.train_seconds, 2)} | `{row.run_id}` |"
        )
    lines.append("")
    return lines


def _ranking_section(summaries: list[_RunSummary]) -> list[str]:
    """Render section 3: full per-dataset rankings with the winner bolded."""
    lines = ["## 3. Ranking Per Dataset", ""]
    for dataset, rows in _by_dataset(summaries).items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| Rank | Model | F1 | ROC-AUC | Train time (s) "
                "| Predict (s) | Size (KB) |",
                "|-----:|-------|---:|--------:|---------------:"
                "|------------:|----------:|",
            ]
        )
        for rank, row in enumerate(_ranked(rows), start=1):
            name = f"**{row.model}**" if rank == 1 else row.model
            lines.append(
                f"| {rank} | {name} | {_fmt(row.f1)} | {_fmt(row.roc_auc)} "
                f"| {_fmt(row.train_seconds, 2)} | {_fmt(row.predict_seconds, 3)} "
                f"| {_size_kb(row.size_bytes)} |"
            )
        lines.append("")
    return lines


def _model_aggregates(
    summaries: list[_RunSummary],
) -> list[tuple[str, int, dict[str, float | None]]]:
    """Aggregate rows per model across datasets.

    Returns
    -------
    list
        ``(model, datasets_completed, averages)`` tuples sorted descending by
        average F1; ``averages`` holds ``accuracy``/``f1``/``roc_auc``/
        ``train_seconds`` means over the datasets the model completed.
    """
    by_model: dict[str, list[_RunSummary]] = {}
    for row in summaries:
        by_model.setdefault(row.model, []).append(row)
    aggregates = [
        (
            model,
            len(rows),
            {
                "accuracy": _mean(r.accuracy for r in rows),
                "f1": _mean(r.f1 for r in rows),
                "roc_auc": _mean(r.roc_auc for r in rows),
                "train_seconds": _mean(r.train_seconds for r in rows),
            },
        )
        for model, rows in by_model.items()
    ]
    aggregates.sort(key=lambda item: (-_desc(item[2]["f1"]), item[0]))
    return aggregates


def _overall_ranking_section(summaries: list[_RunSummary]) -> list[str]:
    """Render section 4: cross-dataset averages per model, sorted by F1."""
    lines = [
        "## 4. Overall Model Ranking",
        "",
        "Averages across every dataset the model completed.",
        "",
        "| Rank | Model | Datasets | Avg Accuracy | Avg F1 | Avg ROC-AUC "
        "| Avg train time (s) |",
        "|-----:|-------|---------:|-------------:|-------:|------------:"
        "|-------------------:|",
    ]
    for rank, (model, n_datasets, avg) in enumerate(
        _model_aggregates(summaries), start=1
    ):
        lines.append(
            f"| {rank} | {model} | {n_datasets} | {_fmt(avg['accuracy'])} "
            f"| {_fmt(avg['f1'])} | {_fmt(avg['roc_auc'])} "
            f"| {_fmt(avg['train_seconds'], 2)} |"
        )
    lines.append("")
    return lines


def _family_section(summaries: list[_RunSummary]) -> list[str]:
    """Render section 5: classical vs deep-learning comparison."""
    lines = [
        "## 5. Classical vs Deep Learning",
        "",
        "| Family | Models | Avg F1 | Avg ROC-AUC | Avg train time (s) "
        "| Fastest training | Highest F1 |",
        "|--------|-------:|-------:|------------:|-------------------:"
        "|------------------|------------|",
    ]
    for family in ("Classical", "Deep"):
        rows = [s for s in summaries if _model_family(s.model) == family]
        if not rows:
            continue
        fastest = min(
            (r for r in rows if r.train_seconds is not None),
            key=lambda r: (r.train_seconds, r.dataset, r.model),
            default=None,
        )
        top = max(rows, key=lambda r: (_desc(r.f1), r.dataset, r.model))
        fastest_cell = (
            f"{fastest.model} ({_fmt(fastest.train_seconds, 2)} s)"
            if fastest else "—"
        )
        top_cell = f"{top.model} ({_fmt(top.f1)} on {top.dataset})"
        lines.append(
            f"| {family} | {len({r.model for r in rows})} "
            f"| {_fmt(_mean(r.f1 for r in rows))} "
            f"| {_fmt(_mean(r.roc_auc for r in rows))} "
            f"| {_fmt(_mean(r.train_seconds for r in rows), 2)} "
            f"| {fastest_cell} | {top_cell} |"
        )
    lines.append("")
    unclassified = sorted(
        {s.model for s in summaries if _model_family(s.model) is None}
    )
    if unclassified:
        lines.extend([f"Unclassified models: {', '.join(unclassified)}", ""])
    return lines


def _efficiency_section(summaries: list[_RunSummary]) -> list[str]:
    """Render section 6: the leading run per efficiency criterion."""

    def _leader(
        rows: list[_RunSummary],
        value: Callable[[_RunSummary], float | None],
        best: Callable[..., tuple[_RunSummary, float]],
    ) -> tuple[_RunSummary, float] | None:
        scored = [(r, v) for r in rows if (v := value(r)) is not None]
        if not scored:
            return None
        row, score = best(
            scored, key=lambda item: (item[1], item[0].dataset, item[0].model)
        )
        return row, score

    criteria: list[tuple[str, tuple[_RunSummary, float] | None, str]] = [
        (
            "Fastest training",
            _leader(summaries, lambda r: r.train_seconds, min),
            "{:.2f} s",
        ),
        (
            "Smallest model",
            _leader(
                summaries,
                lambda r: r.size_bytes / 1024 if r.size_bytes is not None else None,
                min,
            ),
            "{:,.1f} KB",
        ),
        (
            "Fastest inference",
            _leader(summaries, lambda r: r.predict_seconds, min),
            "{:.3f} s",
        ),
        (
            "Best F1 per training minute",
            _leader(
                summaries,
                lambda r: (
                    r.f1 / (r.train_seconds / 60.0)
                    if r.f1 is not None and r.train_seconds
                    else None
                ),
                max,
            ),
            "{:.2f} F1/min",
        ),
        (
            "Best F1 per model MB",
            _leader(
                summaries,
                lambda r: (
                    r.f1 / (r.size_bytes / 1_048_576)
                    if r.f1 is not None and r.size_bytes
                    else None
                ),
                max,
            ),
            "{:.2f} F1/MB",
        ),
    ]
    lines = [
        "## 6. Efficiency Analysis",
        "",
        "Leader per criterion across the latest runs.",
        "",
        "| Criterion | Model | Dataset | Value |",
        "|-----------|-------|---------|------:|",
    ]
    for label, leader, fmt in criteria:
        if leader is None:
            continue
        row, score = leader
        lines.append(
            f"| {label} | {row.model} | {row.dataset} | {fmt.format(score)} |"
        )
    lines.append("")
    return lines


def _key_findings_section(summaries: list[_RunSummary]) -> list[str]:
    """Render section 7: factual observations computed from the benchmark."""
    findings: list[str] = []
    grouped = _by_dataset(summaries)
    winners = {dataset: _ranked(rows)[0] for dataset, rows in grouped.items()}

    winner_models = sorted({w.model for w in winners.values()})
    if len(winner_models) == 1 and len(winners) > 1:
        findings.append(
            f"{winner_models[0]} achieves the highest F1 on all "
            f"{len(winners)} datasets."
        )
    else:
        leaders = ", ".join(
            f"{dataset}: {winners[dataset].model} ({_fmt(winners[dataset].f1)})"
            for dataset in sorted(winners)
        )
        findings.append(f"Per-dataset F1 leaders — {leaders}.")

    winner_families = {_model_family(w.model) for w in winners.values()}
    if winner_families == {"Classical"}:
        findings.append("Classical models lead every dataset benchmarked.")
    elif winner_families == {"Deep"}:
        findings.append("Deep models lead every dataset benchmarked.")

    for dataset in sorted(grouped):
        ranked = [r for r in _ranked(grouped[dataset]) if r.f1 is not None]
        if len(ranked) < 2:
            continue
        delta = ranked[0].f1 - ranked[1].f1
        if delta <= 0.002:
            gap = f"{delta:.4f} F1" if delta >= 0.0001 else "less than 0.0001 F1"
            findings.append(
                f"On {dataset}, {ranked[0].model} and {ranked[1].model} are "
                f"separated by {gap}."
            )

    aggregates = _model_aggregates(summaries)
    multi = [a for a in aggregates if a[1] > 1 and a[2]["f1"] is not None]
    if multi:
        model, n_datasets, avg = multi[0]
        findings.append(
            f"{model} has the highest average F1 across datasets "
            f"({_fmt(avg['f1'])} over {n_datasets})."
        )
    deep = [a for a in aggregates if _model_family(a[0]) == "Deep"]
    if deep:
        model, n_datasets, avg = deep[0]
        findings.append(
            f"Strongest deep model by average F1: {model} "
            f"({_fmt(avg['f1'])} over {n_datasets} dataset(s))."
        )

    lines = ["## 7. Key Findings", ""]
    lines.extend(f"- {finding}" for finding in findings)
    lines.append("")
    return lines


def _reproducibility_section(latest: dict[str, dict[str, Any]]) -> list[str]:
    """Render section 8: how every benchmark number can be reproduced."""
    seeds = sorted(
        {
            manifest.get("seed")
            for models in latest.values()
            for manifest in models.values()
            if manifest.get("seed") is not None
        }
    )
    seed_text = ", ".join(str(s) for s in seeds) if seeds else "recorded per run"
    return [
        "## 8. Reproducibility",
        "",
        f"- Deterministic seed: {seed_text} (recorded in every manifest).",
        "- Every run persists a manifest with its full configuration "
        "snapshot, hardware summary, timings and metrics under "
        "`outputs/experiments/<dataset>/<model>/<run_id>/`.",
        "- Hardware is detected centrally (`src/utils/hardware.py`) and the "
        "selected device is recorded per run.",
        "- `outputs/experiments/experiment_index.csv` indexes every completed "
        "run (rebuild: `python -m scripts.build_experiment_index`).",
        "- The pipeline is configuration-driven: hyperparameters come from "
        "`configs/*.yaml`, never from code.",
        "",
    ]


def build_validation_report(
    latest: dict[str, dict[str, Any]],
    analysis: str | None = None,
    total_experiments: int | None = None,
) -> str:
    """Assemble the full Markdown validation / benchmark report.

    Parameters
    ----------
    latest:
        Output of :func:`collect_latest_manifests`.
    analysis:
        Optional Markdown block (bottlenecks, recommendations) appended after
        the per-dataset sections.
    total_experiments:
        Total number of completed experiments on disk (see
        :func:`count_manifests`); defaults to the number of latest runs when
        not provided.

    Returns
    -------
    str
        The complete Markdown document: benchmark sections 1–8 followed by
        the per-dataset detail tables.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summaries = _run_summaries(latest)
    if total_experiments is None:
        total_experiments = len(summaries)

    lines = [
        "# Model Validation Report",
        "",
        f"Generated: {stamp}",
        "",
        "Latest experiment per (dataset, model), aggregated from the run "
        "manifests under `outputs/experiments/`.",
        "",
    ]
    lines.extend(_executive_summary(summaries, total_experiments, stamp))
    lines.extend(_best_per_dataset_section(summaries))
    lines.extend(_ranking_section(summaries))
    lines.extend(_overall_ranking_section(summaries))
    lines.extend(_family_section(summaries))
    lines.extend(_efficiency_section(summaries))
    lines.extend(_key_findings_section(summaries))
    lines.extend(_reproducibility_section(latest))

    lines.append("## Detailed Results")
    lines.append("")
    for dataset_id, models in sorted(latest.items()):
        lines.append(f"## Dataset: {dataset_id}")
        lines.append("")
        lines.extend(_comparison_table(models))
        for model_name in sorted(models):
            lines.extend(_model_section(model_name, models[model_name]))
    if analysis:
        lines.append(analysis.rstrip())
        lines.append("")
    return "\n".join(lines)

"""Plot renderers (pure matplotlib, no experiment I/O).

Purpose
-------
Render each visualization from an in-memory DataFrame to a PNG. Every
function is deterministic (no randomness, fixed ordering) and headless
(``Agg`` backend), so plots are directly testable.

Design notes
------------
Single-series bar charts carry one restrained hue (identity is in the title,
so no legend is needed); the confusion heatmap uses a single-hue sequential
ramp (magnitude job). Grids and spines are recessive; annotation ink flips to
white on dark cells. Axis text stays in neutral ink, never the series color.

Limitations
-----------
Static PNGs only — no interactivity, no dashboard (later phase).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Single restrained hue for single-series bars; sequential ramp for magnitude.
_BAR_COLOR = "#4269d0"
_HEATMAP_CMAP = "Blues"
_INK = "#37474f"
# Above this class count, per-cell annotations become unreadable noise.
_MAX_ANNOTATED_CLASSES = 25


def _new_axes(figsize: tuple[float, float]) -> tuple[Any, Any]:
    """Create a headless figure/axes pair with recessive styling."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=_INK, labelsize=8)
    return fig, ax


def _decode_labels(raw_labels: Any, class_names: Any | None) -> list[str]:
    """Map encoded integer labels to decoded names when a full mapping exists."""
    labels = [str(c) for c in raw_labels]
    if class_names is None:
        return labels
    names = [str(n) for n in class_names]
    try:
        decoded = [names[int(label)] for label in labels]
    except (ValueError, IndexError):
        logger.warning("Class-name mapping does not cover the confusion labels; "
                       "keeping raw ids.")
        return labels
    return decoded


def _save(fig: Any, path: Path, dpi: int) -> None:
    """Persist and close a figure (never leak figure state between plots)."""
    import matplotlib.pyplot as plt

    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_confusion_matrix(
    frame: "Any",
    path: Path,
    *,
    normalized: bool = False,
    dpi: int = 150,
    class_names: "Any | None" = None,
) -> None:
    """Render a confusion-matrix heatmap (true rows x predicted columns).

    Parameters
    ----------
    frame:
        Confusion matrix indexed by true label, one column per predicted
        label (as written by the error-analysis subsystem).
    path:
        Output PNG path.
    normalized:
        Row-normalize to recall proportions (zero-support rows stay zero).
    dpi:
        Output resolution.
    class_names:
        Decoded class names indexed by encoded id. Used as tick labels when
        every encoded label resolves; otherwise the frame's own labels stay.
    """
    import numpy as np

    matrix = frame.to_numpy(dtype=float)
    if normalized:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix),
                           where=row_sums > 0)

    n_classes = len(frame.index)
    side = max(4.0, min(14.0, 0.45 * n_classes + 2.5))
    fig, ax = _new_axes((side, side))
    image = ax.imshow(matrix, cmap=_HEATMAP_CMAP, aspect="equal")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    labels = _decode_labels(frame.index, class_names)
    ax.set_xticks(range(n_classes), labels,
                  rotation=90 if n_classes > 10 else 0)
    ax.set_yticks(range(n_classes), labels)
    ax.set_xlabel("Predicted label", color=_INK)
    ax.set_ylabel("True label", color=_INK)
    ax.set_title(
        "Confusion matrix" + (" (row-normalized)" if normalized else " (counts)"),
        color=_INK,
    )

    if n_classes <= _MAX_ANNOTATED_CLASSES:
        threshold = matrix.max() / 2 if matrix.max() > 0 else 0.5
        for row in range(n_classes):
            for col in range(n_classes):
                value = matrix[row, col]
                text = f"{value:.2f}" if normalized else f"{int(value):,}"
                ax.text(col, row, text, ha="center", va="center", fontsize=7,
                        color="white" if value > threshold else _INK)
    _save(fig, path, dpi)


def plot_feature_importance(
    frame: "Any", path: Path, *, top_n: int = 20, dpi: int = 150
) -> None:
    """Render the top-N global feature importances as horizontal bars.

    Parameters
    ----------
    frame:
        ``global_feature_importance.csv`` contents (``feature``,
        ``mean_abs_shap`` required); already ranked by the explainability
        subsystem — SHAP is never recomputed here.
    path:
        Output PNG path.
    top_n:
        Number of features shown (all when fewer exist).
    dpi:
        Output resolution.
    """
    top = frame.nlargest(min(int(top_n), len(frame)), "mean_abs_shap")
    fig, ax = _new_axes((8, max(3.0, 0.35 * len(top))))
    ax.barh(top["feature"].astype(str), top["mean_abs_shap"],
            color=_BAR_COLOR, height=0.7)
    ax.invert_yaxis()  # most important on top
    ax.set_xlabel("Mean |SHAP value|", color=_INK)
    ax.set_title(f"Top {len(top)} features by mean |SHAP|", color=_INK)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, path, dpi)


def plot_hardest_classes(
    frame: "Any", path: Path, *, top_n: int = 10, dpi: int = 150
) -> None:
    """Render the lowest-F1 classes as horizontal bars annotated with support.

    Parameters
    ----------
    frame:
        ``hardest_classes.csv`` contents (``class_label``, ``f1_score``,
        ``support`` required), already ranked hardest-first.
    path:
        Output PNG path.
    top_n:
        Number of classes shown (all when fewer exist — binary datasets
        simply show both classes).
    dpi:
        Output resolution.
    """
    top = frame.head(min(int(top_n), len(frame)))
    labels = [
        f"{row.class_label}  (n={row.support:,})"
        for row in top.itertuples(index=False)
    ]
    fig, ax = _new_axes((8, max(2.5, 0.4 * len(top))))
    ax.barh(labels, top["f1_score"], color=_BAR_COLOR, height=0.7)
    ax.invert_yaxis()  # hardest on top
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score", color=_INK)
    ax.set_title(f"{len(top)} hardest classes (lowest F1)", color=_INK)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for position, value in enumerate(top["f1_score"]):
        ax.text(min(value + 0.01, 0.99), position, f"{value:.3f}",
                va="center", fontsize=8, color=_INK)
    _save(fig, path, dpi)


def plot_misclassification_pairs(
    frame: "Any", path: Path, *, top_n: int = 20, dpi: int = 150
) -> None:
    """Render the most common true -> predicted mistakes as horizontal bars.

    Parameters
    ----------
    frame:
        ``misclassified_examples.csv`` contents (``true_label``,
        ``predicted_label`` required). Must be non-empty — the runner skips
        this plot (with metadata) when there are no misclassifications.
    path:
        Output PNG path.
    top_n:
        Number of pairs shown.
    dpi:
        Output resolution.

    Raises
    ------
    ValueError
        When ``frame`` has no rows (nothing to plot).
    """
    if len(frame) == 0:
        raise ValueError("No misclassified examples to plot.")

    pairs = (
        frame.groupby(["true_label", "predicted_label"], sort=True)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "true_label", "predicted_label"],
                     ascending=[False, True, True])
        .head(int(top_n))
    )
    labels = [
        f"{row.true_label} → {row.predicted_label}"
        for row in pairs.itertuples(index=False)
    ]
    fig, ax = _new_axes((8, max(2.5, 0.4 * len(pairs))))
    ax.barh(labels, pairs["count"], color=_BAR_COLOR, height=0.7)
    ax.invert_yaxis()  # most common on top
    ax.set_xlabel("Misclassified examples", color=_INK)
    ax.set_title(
        f"Top {len(pairs)} true → predicted misclassification pairs",
        color=_INK,
    )
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, path, dpi)

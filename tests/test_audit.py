"""Tests for src.data.audit (Markdown rendering)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import audit
from src.data.validation import build_report


def _report(label_skew: bool = False):
    n = 100
    labels = ["attack"] * (99 if label_skew else 50) + ["normal"] * (
        1 if label_skew else 50
    )
    frame = pd.DataFrame(
        {
            "duration": [0.0, np.inf] + [1.0] * (n - 2),
            "protocol_type": ["tcp"] * n,
            "label": labels,
        }
    )
    config = {
        "id": "demo",
        "name": "Demo",
        "engine": "A",
        "categorical_columns": ["protocol_type"],
        "label_column": "label",
    }
    return build_report(frame, config)


def test_render_audit_contains_required_sections() -> None:
    report = _report()
    fingerprints = {"demo": {"sha256": "a" * 64, "schema_version": "1.0"}}
    md = audit.render_audit_markdown({"demo": report}, fingerprints)

    for heading in (
        "# NetSentinel — Dataset Audit Report",
        "### Dataset overview",
        "### Duplicate analysis",
        "### Missing value analysis",
        "### Infinite values",
        "### Class imbalance",
        "### Memory usage",
        "### Recommended preprocessing actions",
        "### Risks",
        "### Notes for future model training",
    ):
        assert heading in md


def test_render_audit_flags_infinite_and_imbalance() -> None:
    report = _report(label_skew=True)
    md = audit.render_audit_markdown(
        {"demo": report}, {"demo": {"sha256": "b" * 64, "schema_version": "1.0"}}
    )
    # Infinite value in `duration` should drive a recommendation.
    assert "±inf" in md
    # Severe imbalance (99:1) should appear as a risk.
    assert "imbalance" in md.lower()


def test_render_audit_recommendations_present() -> None:
    report = _report()
    md = audit.render_audit_markdown(
        {"demo": report}, {"demo": {"sha256": "c" * 64, "schema_version": "1.0"}}
    )
    assert "Encode categorical features" in md
    assert "Scale numeric features" in md

"""Engine C Phase 3 — structured finding model.

Purpose
-------
One typed, JSON-serialisable :class:`Finding` produced by the rule engine
(:mod:`src.network_config.rules`) plus a deterministic id helper. Findings are
detection-only: they describe a problem, its evidence and a recommendation, but
never execute any change (remediation is a later, gated phase).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

# Severity ordering (most severe first) for deterministic sorting / reporting.
SEVERITY_ORDER: dict[str, int] = {
    "critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4,
}

VALID_STATUSES = ("open", "suppressed", "resolved")


@dataclass(frozen=True)
class Finding:
    """A single rule-engine finding against the parsed network state."""

    finding_id: str
    rule_id: str
    title: str
    severity: str                              # critical/high/medium/low/info
    category: str                              # vlan/poe/port/stp/topology/...
    device: Optional[str] = None
    interface: Optional[str] = None
    vlan: Optional[str] = None
    status: str = "open"                       # open/suppressed/resolved
    evidence: Optional[str] = None
    recommendation: Optional[str] = None
    confidence: str = "medium"                 # high/medium/low
    source: str = "inventory"                  # inventory/topology
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def severity_rank(self) -> int:
        """Numeric rank for sorting (unknown severities sort last)."""
        return SEVERITY_ORDER.get(self.severity, len(SEVERITY_ORDER))


def make_finding_id(
    rule_id: str,
    device: Optional[str],
    interface: Optional[str],
    vlan: Optional[str] = None,
) -> str:
    """Deterministic finding id from its natural key.

    The id is stable across runs and independent of evaluation order: the same
    (rule, device, interface, vlan) always yields the same id.
    """
    key = "|".join([
        rule_id,
        device or "",
        interface or "",
        "" if vlan is None else str(vlan),
    ])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{rule_id}-{digest}"

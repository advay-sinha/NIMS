"""Offline ML workflow orchestration (Phase 11 add-on).

A controlled, **offline-only** way to run the existing Engine A dataset/model
pipeline steps (validate → audit → preprocess → features → train → reports →
explainability/error-analysis/visualizations → registry → promote → resolve) so
the prototype is easy to demo and the dashboard can then load the freshly
generated artefacts.

Hard boundary
-------------
This orchestrates only the existing **local, offline** Engine A entry points. It
must never touch live devices, live traffic, SNMP, SSH, packet capture, firewall
logs or remediation — those steps do not exist in the plan and are rejected.
"""

from __future__ import annotations

__all__ = ["planner"]

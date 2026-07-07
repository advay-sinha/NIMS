# Engine C — Integration Handoff

Engine C is functionally complete (phases 1–9) and offline/read-only. This
document explains how future phases — the **monitoring dashboard** and the
**correlation layer** — can consume Engine C outputs, and the gated path that
would be required if live-device support were ever added.

Engine C exposes everything as stable JSON/CSV artifacts under
`outputs/network_config/`. Downstream consumers should read those artifacts;
they should not import Engine C internals or re-run analysis.

---

## A. Dashboard integration

The dashboard exporter (`scripts/export_network_config_dashboard`) writes flat,
frontend-friendly views to
`outputs/network_config/<snapshot_id>/dashboard/`. Each view carries a
`safety_note` and stable field names (`export_version` in
`export_metadata.json` tracks the schema).

| File | What it is for |
|---|---|
| `dashboard_summary.json` | Top-level KPIs: device/interface/VLAN/edge counts, findings by severity/category, remediation action counts, availability flags (`dry_run_available`, `batfish_available`, `diff_available`) and `top_risk_devices`. Drives the landing page. |
| `inventory_view.json` | Devices plus interfaces/VLANs/trunks/PoE/STP grouped by device — flat lists ready for tables/detail panels. |
| `topology_view.json` | `nodes` (id/label/type/`risk_score`/`finding_count`) and `edges` (source/target/interfaces/protocol/confidence/`warning_count`) plus `warnings` — drives the topology graph. |
| `findings_view.json` | All findings enriched with `risk_score`, grouped by severity/category/device, plus `top_findings` — drives the findings page. |
| `remediation_view.json` | All dry-run actions with command/investigation/blocked splits, grouped by device/risk; `human_confirmation_required` and `dry_run_only` are always true — drives the remediation review panel (display only). |
| `action_audit_view.json` | Dry-run execution summary, records grouped by status, `executed_count` (always 0), or `{available:false, reason}` — drives an audit panel. |
| `risk_timeline.json` | Artifact-lifecycle events (snapshot/topology/findings/remediation/dry-run/diff) with timestamps. **Not** a live time series. |
| `device_health_cards.json` | One card per device (interface/trunk/access-port/finding/PoE/STP counts, highest severity, status `healthy/warning/critical/unknown`) — drives a device-health grid. |
| `export_metadata.json` | `snapshot_id`, `diff_id`, `generated_at`, `source_artifacts_used`/`missing`, `export_version`, `safety_note` — provenance for the UI. |
| `diff_view.json` / `verification_view.json` | Only when `--diff-id` is passed: changes grouped by category/type, findings new/resolved/persistent, and verification grouped by `passed/failed/unknown/not_applicable`. |

**Dashboard guidance**

- Treat all remediation/action data as **display-only**. The UI must not offer a
  one-click "apply"; Engine C never executes, and the dashboard must not either.
- Always surface the `safety_note` and the dry-run/no-execution status.
- Use `export_metadata.json` to show which artifacts were present and to guard
  against schema drift via `export_version`.
- The dashboard is Phase 10; this handoff provides its data contract only — no
  UI is built in Engine C.

---

## B. Correlation layer integration

The correlation engine (planned next milestone) should ingest Engine C findings
and the intelligence summary alongside Engine A and Engine B signals, and group
related events into incidents rather than emitting them separately.

**Engine C inputs the correlator can consume**

- `findings.json` — structured configuration findings (rule_id, severity,
  category, device, interface, evidence, deterministic ids).
- `config_intelligence_summary.json` — per-device risk, top risks, root-cause
  hypothesis and action-item counts.
- `topology.json` — device/interface adjacency for spatial correlation.
- `dashboard/device_health_cards.json` — per-device rollups keyed by device.

**Correlation examples**

- **Engine A DoS alert + Engine B interface saturation + Engine C trunk
  warning** on the same segment → one "possible DDoS causing switch overload"
  incident instead of three alerts.
- **Engine B port degradation + Engine C PoE/STP/port-state findings** on the
  same device/interface → a hardware/config-degradation incident.
- **Engine C unauthorized-VLAN finding + Engine A suspicious traffic** on that
  VLAN → a policy-violation / lateral-movement incident.

**Correlation guidance**

- Join primarily on `(device, interface)` and topology adjacency; Engine C ids
  are deterministic, so findings are stable across runs.
- Preserve Engine C's cautious wording — configuration findings are evidence,
  not verdicts. Confidence should compose, not inflate.
- Engine C provides *configuration* context; it does not itself decide that an
  incident is malicious.

---

## C. Future live support (gated)

Live device support is **not implemented and out of scope**. If it is ever
added, it must be a separate, gated subsystem. Minimum requirements (see
`engine_c_safety_audit.md` for the full list):

- separate package/module, isolated from detection and planning
- explicit config enablement (off by default)
- test lab only, read-only collection first
- human confirmation, rollback, verification and audit logs for every action
- an allowlist of permitted devices and actions
- no default execution — dry-run stays the default
- mockable clients; no tests requiring real hardware; no secrets in source

Until all of the above exist, downstream consumers should assume Engine C is
and remains **offline, read-only and execution-free**.

---

## Stability contract

- Artifacts are additive and versioned where it matters
  (`export_metadata.export_version`).
- Consumers should read artifacts, tolerate missing optional files, and never
  depend on Engine C internal Python APIs.
- Existing artifact schemas are frozen for this closure; new fields may be added
  but existing fields should not change meaning.

# Engine C — Network Configuration Intelligence & Remediation

Engine C is the network-operations layer of NetSentinel (NIMS). It turns saved
network-device command outputs into a structured inventory, a derived topology,
rule-based configuration findings, safe **dry-run** remediation plans, offline
snapshot diffs with remediation verification, a consolidated intelligence
report, optional external Batfish validation, and dashboard-ready JSON exports.

Engine C is **offline-first and read-only by default**. It does not access live
devices, does not poll SNMP, and never executes a command. Every phase after
detection (remediation, dry-run execution) produces *plans and validations
only*.

---

## Purpose

- Understand real network state (switches, routers, ports, VLANs, PoE, STP,
  topology) from saved command outputs.
- Detect configuration errors and loop/topology risks with evidence.
- Propose **safe, reversible, human-confirmed** remediation — never execute it.
- Compare snapshots over time and verify whether remediation goals were met.
- Summarise everything for an operator and export it for a future dashboard.

---

## Design principles

- **Offline-first.** All inputs are saved text captures; the live pipeline is
  intentionally not implemented.
- **Read-only by default.** Detection is strictly separated from action.
- **Configuration-driven.** Rules, thresholds, expected VLANs, remediation
  templates and safety flags live in YAML — no site policy is hardcoded in
  Python.
- **Deterministic & reproducible.** Ids are derived from natural keys; the same
  inputs always yield the same outputs.
- **Cautious wording.** Inferred issues use `candidate` / `possible` / `likely`;
  verification prefers `unknown` over false confidence.
- **No secrets.** No credentials are read, stored, printed or logged.

---

## Supported offline inputs

Saved Cisco-style command outputs (filenames configurable in
`configs/network_config.yaml`):

| Logical input | Default filename |
|---|---|
| interface status | `show_interface_status.txt` |
| VLAN brief | `show_vlan_brief.txt` |
| trunks | `show_interfaces_trunk.txt` |
| spanning-tree | `show_spanning_tree.txt` |
| MAC table | `show_mac_address_table.txt` |
| PoE | `show_power_inline.txt` |
| LLDP neighbors | `show_lldp_neighbors.txt` |
| CDP neighbors | `show_cdp_neighbors.txt` |
| running-config | `show_running_config.txt` |

Missing files are warned about and skipped. Batfish (optional) additionally
consumes a snapshot directory of raw `.cfg` files.

---

## Package structure

```
src/network_config/
    models.py                 # typed dataclasses (device/interface/vlan/...)
    parsers.py                # tolerant show-output parsers
    inventory.py              # inventory builder + summary
    artifacts.py              # inventory/report persistence
    reporting.py              # network_config_report.md
    topology.py               # LLDP/CDP/MAC/STP topology + warnings
    topology_artifacts.py
    rules.py, findings.py     # YAML rule engine -> structured findings
    rule_artifacts.py
    safety.py                 # remediation safety primitives (DO_NOT_EXECUTE)
    remediation.py            # dry-run remediation plan generator
    remediation_artifacts.py
    executor.py, audit.py     # dry-run action executor + audit log
    execution_artifacts.py
    diff.py, verification.py  # snapshot diff + remediation verification
    diff_artifacts.py
    intelligence.py           # risk scoring, root-cause, action items
    intelligence_artifacts.py # consolidated report + summary
    batfish_adapter.py        # optional Batfish validation (lazy import)
    batfish_artifacts.py
    dashboard_export.py       # frontend-friendly JSON views
    dashboard_artifacts.py
```

---

## Config files

| File | Purpose |
|---|---|
| `configs/network_config.yaml` | input filenames, topology thresholds, phase toggles, safety posture |
| `configs/network_rules.yaml` | rule definitions, severities, thresholds, expected VLANs, suppressions |
| `configs/remediation.yaml` | remediation templates (dry-run only) |
| `configs/network_action_executor.yaml` | dry-run executor + audit safety flags |
| `configs/batfish.yaml` | optional Batfish adapter (disabled by default) |

---

## Scripts

| Script | Role |
|---|---|
| `analyze_network_config` | parse → inventory → topology → findings → dry-run remediation plan |
| `dry_run_network_actions` | validate a remediation plan in dry-run mode + write audit log |
| `compare_network_snapshots` | diff two snapshots + verify remediation goals |
| `generate_network_config_report` | consolidated intelligence report + summary |
| `export_network_config_dashboard` | dashboard-ready JSON views |
| `run_batfish_validation` | optional external Batfish validation (disabled by default) |
| `validate_engine_c_safety` | static safety audit of Engine C source + configs |

---

## Artifact outputs

Under `outputs/network_config/<snapshot_id>/`:

- Inventory: `inventory.json`, `metadata.json`, one CSV per object type,
  `network_config_report.md`
- Topology: `topology.json`, `topology_nodes.csv`, `topology_edges.csv`,
  `topology_warnings.csv`
- Findings: `findings.json`, `findings.csv`, `rule_summary.json`
- Remediation (dry-run): `remediation_plan.json`, `remediation_plan.md`,
  `remediation_commands.csv`, `remediation_summary.json`
- Dry-run execution: `dry_run_execution.json`, `dry_run_execution.csv`,
  `action_audit_log.jsonl`, `execution_summary.json`
- Intelligence: `config_intelligence_report.md`,
  `config_intelligence_summary.json` (+ `_with_diff.md` when a diff is used)
- Batfish (optional): `batfish/batfish_summary.json` + tables + findings
- Dashboard: `dashboard/*.json`

Under `outputs/network_config/diffs/<before>__to__<after>/`:
`snapshot_diff.json/.csv`, `verification_results.json/.csv`, `diff_summary.json`,
`network_diff_report.md`.

---

## Development phases completed

| Phase | Scope |
|---|---|
| 1 | Offline parsers + inventory builder |
| 2 | Topology builder (LLDP/CDP/MAC/STP) |
| 3 | YAML rule findings engine |
| 4 | Safe (dry-run) remediation plan generator |
| 5 | Dry-run action executor + audit logging |
| 6 | Offline snapshot diff + remediation verification |
| 7 | Consolidated configuration-intelligence report |
| 8 | Optional Batfish validation adapter |
| 9 | Dashboard-ready JSON exports |

All nine phases are complete. Live device access and command execution are
**intentionally not implemented** — see `engine_c_safety_audit.md`.

---

## Safety model

- Default mode is **read-only**; detection never calls action execution.
- Remediation produces **plans only**. Every command-bearing action is
  `dry_run_only`, `requires_confirmation`, and carries a rollback and a
  verification step, or it is blocked.
- The dry-run executor sets `executed = false` and `would_execute = false` on
  every record and writes an append-only audit log.
- No live-device libraries (netmiko/napalm/paramiko), no SSH/SNMP, no command
  execution, no sockets to devices.
- Batfish is optional and disabled by default; the whole engine runs without
  Docker or pybatfish installed.

Run the static audit any time:

```bash
python -m scripts.validate_engine_c_safety
```

---

## Running the sample workflow

```bash
# 1. Parse saved outputs -> inventory, topology, findings, dry-run remediation
python -m scripts.analyze_network_config \
    --input-dir datasets/samples/network_config --snapshot-id sample_offline

# 2. Dry-run validate the remediation plan + write the audit log
python -m scripts.dry_run_network_actions --snapshot-id sample_offline

# 3. Compare two snapshots and verify remediation goals
python -m scripts.compare_network_snapshots --before sample_before --after sample_after

# 4. Consolidated operator intelligence report
python -m scripts.generate_network_config_report --snapshot-id sample_remediation

# 5. Dashboard-ready JSON exports
python -m scripts.export_network_config_dashboard --snapshot-id sample_remediation

# 6. (Optional) external Batfish validation — disabled by default, exits cleanly
python -m scripts.run_batfish_validation --snapshot-id sample_remediation
```

---

## Inspecting outputs

- **Findings** — `outputs/network_config/<id>/findings.json` (structured) or the
  "Configuration Findings" section of `network_config_report.md`.
- **Remediation (dry-run)** — `remediation_plan.md` (human) /
  `remediation_plan.json` (machine); every command has a rollback and a
  verification step and is marked *no command executed*.
- **Dry-run audit** — `dry_run_execution.json` and `action_audit_log.jsonl`;
  every record has `executed = false`.
- **Diff / verification** — `diffs/<...>/network_diff_report.md` and
  `verification_results.json` (`passed`/`failed`/`unknown`/`not_applicable`).
- **Intelligence report** — `config_intelligence_report.md` (risk scores,
  root-cause hypotheses, operator action items, safety notes).
- **Dashboard exports** — `dashboard/*.json` (see
  `engine_c_integration_handoff.md`).

---

## Batfish optional adapter behavior

Batfish is an **optional external validator**, disabled by default in
`configs/batfish.yaml` (`global.enabled: false`).

- `pybatfish` is imported lazily, only inside the adapter; the whole project
  imports and runs without Batfish, Docker or pybatfish present.
- When disabled, the script exits cleanly reporting "disabled".
- When enabled but unavailable (no pybatfish/service, missing snapshot, parse
  failure), the adapter records a `skipped`/`failed` status and does not crash
  Engine C — unless `--fail-if-unavailable` is set, which forces an attempt and
  a non-zero exit.
- Results are clearly marked as **external validation evidence** and never
  replace the offline parsers, rules, remediation or reports.
- Tests use a mocked session and never require a running Batfish service.

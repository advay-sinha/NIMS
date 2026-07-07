# Engine C — Safety Audit

Engine C is operationally sensitive: it reasons about production network
configuration and proposes changes. Its entire design keeps it **offline,
read-only, and execution-free**. This document records the safety checklist,
the known boundaries, what is intentionally not implemented, and the
requirements that would have to be met before live-device support could ever be
considered.

A static verifier enforces the code/config parts of this checklist:

```bash
python -m scripts.validate_engine_c_safety
```

It scans `src/network_config/` and the Engine C scripts for forbidden imports
and usages, and checks the shipped configs for safe defaults. It exits non-zero
on any violation.

---

## Safety checklist

| # | Guarantee | Status | Where enforced |
|---|---|---|---|
| 1 | No live SSH access | ✅ | no SSH client imported anywhere in Engine C |
| 2 | No SNMP polling | ✅ | no SNMP client; SNMP is offline dataset input only |
| 3 | No Netmiko / NAPALM / Paramiko | ✅ | `validate_engine_c_safety` forbids these imports |
| 4 | No command execution on devices | ✅ | no device client; nothing sends config |
| 5 | Remediation produces plans only | ✅ | `remediation.py` builds plans; never executes |
| 6 | Dry-run executor sets `executed = false` | ✅ | `executor.py` hardcodes `EXECUTED = False`, `WOULD_EXECUTE = False` |
| 7 | Rollback required for command-bearing plans | ✅ | `safety.validate_command_action` blocks actions without rollback |
| 8 | Verification required for command-bearing plans | ✅ | same validator blocks actions without a verification step |
| 9 | Human confirmation required | ✅ | every command action is `requires_confirmation = true` |
| 10 | Audit logs written | ✅ | `audit.py` writes append-only `action_audit_log.jsonl` |
| 11 | Batfish optional | ✅ | `configs/batfish.yaml` `enabled: false`; lazy import |
| 12 | No Docker / Batfish required for tests | ✅ | tests use a mocked Batfish session |
| 13 | No source artifacts mutated by report/export scripts | ✅ | intelligence + dashboard builders read-only; tested |

Config safe-default checks (also enforced by the verifier):

- `network_config.safety.live_device_access` is `false`
- `network_config.safety.remediation_enabled` is `false`
- `remediation.global.dry_run_only` is `true`
- `network_action_executor.global.allow_live_execution` is `false`
- `network_action_executor.global.mode` is `dry_run`
- `batfish.global.enabled` is `false`

---

## Forbidden usages (statically checked)

The validator flags any of the following in Engine C source:

- `import netmiko` / `from netmiko ...`
- `import napalm` / `from napalm ...`
- `import paramiko` / `from paramiko ...`
- `ConnectHandler(...)`
- `send_config_set(...)`
- `import socket` / `socket.socket(...)`
- `subprocess.*` invoking `ssh` / `telnet` / `netcat` / `nc`

Prose in docstrings and comments is stripped before scanning, so documentation
that *names* these libraries (like this file) does not trip the check.

---

## Known safety boundaries

- **Evidence quality.** Findings and topology are only as good as the saved
  command outputs. MAC-derived adjacency and inferred issues are marked
  low-confidence / `candidate`, never asserted as fact.
- **Verification is offline.** Snapshot-diff verification compares saved
  artifacts; it is *not* live-device confirmation. It returns `unknown` when the
  after-state lacks the data to judge.
- **Dry-run is not execution.** A `validated` dry-run record means the plan is
  internally safe to *plan*, not that it was applied or that it would succeed on
  a real device.
- **No time series.** The risk timeline is an artifact-lifecycle timeline built
  from artifact timestamps, not a live monitoring feed.
- **Batfish is advisory.** External validation evidence is additive; it never
  overrides the offline analysis and requires an operator-run Batfish service.

---

## Intentionally not implemented

- Live device access of any kind (SSH, SNMP, NETCONF, RESTCONF, vendor APIs).
- Any command execution or configuration push.
- Automatic / autonomous remediation.
- Credential handling, secret storage, or connection management.
- A web dashboard UI (only the JSON export layer exists).
- The correlation engine combining Engine A/B/C (planned next milestone).

---

## Requirements before live-device support could be considered

Live support is **out of scope** and must not be added casually. If it is ever
pursued, it must satisfy all of the following before any write path exists:

1. A **separate package/module** isolated from detection and planning.
2. **Explicit config enablement** (off by default; no implicit enabling).
3. **Test-lab only** initially — never production by default.
4. **Read-only collection first** (inventory/state), before any action path.
5. **Human confirmation** required for every action.
6. **Rollback** generated and shown before every action.
7. **Verification** run after every action, with results logged.
8. **Append-only audit logs** for every attempt, confirmation and result.
9. A **device/action allowlist** — only explicitly permitted targets/actions.
10. **No default execution** — dry-run remains the default forever.
11. Mockable clients and tests that never require real hardware.
12. No credentials in source, configs, manifests or logs.

Until every item above is met, Engine C remains offline, read-only and
execution-free.

# Live Integration Setup

NetSentinel ingestion is **offline-first**. Every source runs in `offline` (saved
samples) or `mock` mode by default and contacts nothing. **Live mode is disabled
by default** and only runs a source whose config has `enabled: true`,
`mode: live` and `read_only: true`, after you complete the setup below.

The whole live layer is **read-only**: no SNMP SET, no configuration mode, no
write command, no firewall/VLAN/PoE/STP change, no remediation. Credentials come
**only from environment variables** — never from CLI arguments, config files, or
the frontend, and never appear in logs, events, checkpoints, reports or API
responses.

## Workflow for every source

```
# 1. Check what is missing (never prints secrets, never connects to a device):
python -m scripts.check_live_readiness --source <source>

# 2. When it reports READY, do a single controlled live run:
python -m scripts.run_live_logger --source <source> --live --run-once
```

Readiness statuses: `READY`, `NOT_READY`, `DISABLED`, `BLOCKED_BY_SAFETY`,
`MISSING_DEPENDENCY`, `MISSING_CONFIGURATION`, `MISSING_CREDENTIALS`.

---

## Sophos Firewall syslog  (`sophos_firewall_syslog`)

**Status: Not required — offline/mock works with no setup. Required only for live.**

- External: a Sophos Firewall admin forwards syslog to this host over **UDP 5514**
  (non-privileged); a network route from the firewall to the receiver; the log
  categories you want forwarded; the firewall's source IP.
- Dependency: none (stdlib UDP socket).
- Env vars: none (UDP reception needs no credentials).
- Config (`configs/sophos_logging.yaml` → `sophos.firewall_syslog`): set
  `enabled: true`, `mode: live`, `bind_host`, `bind_port: 5514`, and
  `allowed_sources: [<firewall-ip>]`.
- Minimum permission: read-only log forwarding on the firewall.
- Verify: `python -m scripts.check_live_readiness --source sophos_firewall_syslog`
  then `python -m scripts.run_sophos_ingestor --source firewall_syslog --live --run-once`.
- Success: one datagram received, parsed and persisted; process exits.
- Common failures: UDP 5514 blocked by a firewall/route; source IP not in
  `allowed_sources`; nothing forwarded within the wait window.

## Sophos Central SIEM API  (`sophos_central`)

**Status: Required but not configured (needs vendor credentials).**

- External: a Sophos Central **SIEM Integration** API credential (client id +
  secret), your tenant id and data region, the SIEM API entitlement, and
  outbound HTTPS.
- Dependency: `httpx` (already installed).
- Env vars: `SOPHOS_CLIENT_ID`, `SOPHOS_CLIENT_SECRET`, `SOPHOS_TENANT_ID`,
  `SOPHOS_REGION`.
- Config (`sophos.central_api`): set `enabled: true`, `mode: live`, and an
  explicit `region` (the adapter never guesses tenant/region/host).
- Minimum permission: read-only SIEM events/alerts.
- Verify: `check_live_readiness --source sophos_central` → run-once.
- Success: OAuth token obtained, a page of events fetched, cursor checkpointed.
- Common failures: missing/incorrect env vars; wrong region; no SIEM
  entitlement; outbound HTTPS blocked.

## Hirschmann SNMPv3  (`hirschmann_snmp`)

**Status: Required but not configured (needs `pysnmp` + device + credentials).**

- External: switch management IP; an **SNMPv3 read-only** user; auth protocol +
  password; privacy protocol + password; a supported OID profile; UDP 161
  reachability.
- Dependency: `pysnmp` (`pip install pysnmp`).
- Env vars: `HIRSCHMANN_SNMP_USER`, `HIRSCHMANN_SNMP_AUTH_PASSWORD`,
  `HIRSCHMANN_SNMP_PRIV_PASSWORD`.
- Config: copy `configs/hirschmann_targets.local.yaml.example` to
  `configs/hirschmann_targets.local.yaml` (gitignored) and list your targets; in
  `hirschmann.snmp_polling` set `enabled: true`, `mode: live`. **Read-only GET
  only — there is no SNMP SET path, and arbitrary OIDs are rejected; targets use
  a predefined `profile`.**
- Minimum permission: SNMPv3 read-only view.
- Verify: `check_live_readiness --source hirschmann_snmp` → run-once.
- Common failures: `pysnmp` not installed; UDP 161 unreachable; wrong SNMPv3
  auth/priv; unknown profile; missing targets file.

## Hirschmann traps  (`hirschmann_traps`)

**Status: Not required for offline/mock. Required for live.**

- External: this host's receiver IP + port; the switch configured to send traps
  there; UDP reachability; the switch source IP.
- Dependency: none for text traps (binary PDU decode would need `pysnmp`).
- Env vars: none.
- Config (`hirschmann.traps`): set `enabled: true`, `mode: live`, `bind_port`,
  `allowed_sources: [<switch-ip>]`.
- Verify: `check_live_readiness --source hirschmann_traps` → run-once.
- Common failures: UDP port blocked; source IP not allowlisted; no trap sent in
  the wait window.

## Hirschmann configuration retrieval  (`hirschmann_config`)

**Status: Disabled by default. Required for live (needs `paramiko` + read-only SSH).**

- External: a **strictly read-only** SSH account; switch management IP; the
  switch host key; a supported safe `show` command; **no** configuration
  privilege.
- Dependency: `paramiko` (`pip install paramiko`).
- Env vars: `HIRSCHMANN_SSH_USER`, `HIRSCHMANN_SSH_PASSWORD`.
- Config (`hirschmann.config_retrieval`): set `enabled: true`, `mode: live`,
  `known_hosts_file: configs/known_hosts.local` (host-key verification is
  required), `allowed_commands: [show running-config]`. Startup is **rejected**
  if `allow_config_mode` or `allow_write_commands` is true or a mutating command
  is configured.
- Minimum permission: read-only `show` access, no enable/config mode.
- Verify: `check_live_readiness --source hirschmann_config` → run-once.
- Common failures: `paramiko` not installed; unknown host key (add it to
  `known_hosts.local`); account has write privilege; command not allowlisted.

---

## What stays disabled until you confirm

- All five sources remain `mode: offline` until you explicitly set `mode: live`
  and `enabled: true` per source and provide the environment variables above.
- The `POST /api/live-ingestion/run-once` endpoints return **403** until
  `configs/webapp.yaml` `live_ingestion.allow_run_once: true`, and even then they
  only record intent — they never execute a device connection.
- No device-changing capability exists anywhere in the codebase.

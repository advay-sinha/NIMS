"""Entry point: read-only live retrieval of Huawei ``display`` output over SSH.

Connects to a Huawei VRP switch with host-key verification, runs a fixed
allowlist of **read-only** ``display`` commands (never config mode, never a
write/save/reset command), and writes each command's output to a snapshot
directory that :mod:`src.network_config.vendors.huawei` can parse. This is the
live equivalent of pasting saved output — no configuration is changed.

Safety
------
- Host-key verification is required (``RejectPolicy``); use ``--save-host-key``
  once to record the key after verifying its fingerprint.
- Every command is checked against a forbidden-token list; a config-mode or
  write command aborts the run.
- Credentials come from environment variables only, never the command line.

Usage
-----
    # one-time: record + verify the switch host key
    python -m scripts.retrieve_huawei_config --host 10.90.10.72 --save-host-key

    # read-only retrieval into the snapshot dir
    python -m scripts.retrieve_huawei_config --host 10.90.10.72 \
        --output-dir datasets/samples/network_config/huawei_s5720
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Read-only command -> output filename (matches huawei.DEFAULT_FILES).
COMMANDS: dict[str, str] = {
    "display version": "display_version.txt",
    "display device": "display_device.txt",
    "display interface brief": "display_interface_brief.txt",
    "display vlan": "display_vlan.txt",
    "display stp brief": "display_stp_brief.txt",
    "display poe power": "display_poe_power.txt",
    "display mac-address": "display_mac_address.txt",
    "display lldp neighbor brief": "display_lldp_neighbor_brief.txt",
    "display current-configuration": "display_current_configuration.txt",
}

# Write/config verbs that must never appear in a read-only command. Matched as
# whole words so read-only display sub-commands (interface, vlan, stp, poe,
# mac-address, current-configuration) are never false-positives.
_FORBIDDEN_RE = re.compile(
    r"\b(system-view|save|reset|reboot|shutdown|undo|delete|commit|write|"
    r"erase|format|startup|rename|clear|batch)\b"
)

# Live captures land under outputs/ (gitignored) — never the tracked datasets dir,
# because a live running-config contains real credential material.
DEFAULT_OUTPUT_DIR = "outputs/network_config/live_capture/huawei_s5720"
DEFAULT_KNOWN_HOSTS = "configs/known_hosts.local"


def _redact_config(text: str) -> str:
    """Strip credential material from a live running-config before it is stored."""
    import re

    text = re.sub(r"%\^%#.*?%\^%#", "<REDACTED>", text)
    text = re.sub(r"(irreversible-cipher )\S+", r"\1<REDACTED>", text)
    text = re.sub(r"(cipher )\S+", r"\1<REDACTED>", text)
    return text


def _assert_read_only(command: str) -> None:
    lowered = command.lower().strip()
    if not lowered.startswith("display "):
        raise SystemExit(f"refused: '{command}' is not a read-only display command")
    if any(sep in lowered for sep in (";", "|", "\n", "&&")):
        raise SystemExit(f"refused: '{command}' contains a command separator")
    hit = _FORBIDDEN_RE.search(lowered)
    if hit:
        raise SystemExit(f"refused: '{command}' contains forbidden verb '{hit.group(1)}'")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    p = argparse.ArgumentParser(description="Read-only Huawei display retrieval over SSH.")
    p.add_argument("--host", required=True, help="Switch management IP/hostname.")
    p.add_argument("--port", type=int, default=22)
    p.add_argument("--user-env", default="HUAWEI_SSH_USER",
                   help="Env var holding the SSH username (default: HUAWEI_SSH_USER).")
    p.add_argument("--password-env", default="HUAWEI_SSH_PASSWORD",
                   help="Env var holding the SSH password (default: HUAWEI_SSH_PASSWORD).")
    p.add_argument("--known-hosts", default=DEFAULT_KNOWN_HOSTS)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--save-host-key", action="store_true",
                   help="Record the switch host key to --known-hosts (verify the printed "
                        "fingerprint), then exit without retrieving.")
    p.add_argument("--timeout", type=int, default=20)
    return p


def _credentials(args) -> tuple[str, str]:
    user = os.environ.get(args.user_env)
    password = os.environ.get(args.password_env)
    if not (user and password):
        raise SystemExit(f"set {args.user_env} and {args.password_env} (session env vars); "
                         "never pass credentials on the command line")
    return user, password


def _save_host_key(args) -> int:
    import paramiko

    user, password = _credentials(args)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # record-then-verify (TOFU)
    try:
        client.connect(args.host, port=args.port, username=user, password=password,
                       timeout=args.timeout, look_for_keys=False, allow_agent=False)
        key = client.get_transport().get_remote_server_key()
        fingerprint = key.fingerprint if hasattr(key, "fingerprint") else key.get_base64()[:32]
        Path(args.known_hosts).parent.mkdir(parents=True, exist_ok=True)
        client.save_host_keys(args.known_hosts)
    finally:
        client.close()
    logger.info("Host key for %s recorded to %s", args.host, args.known_hosts)
    logger.info("Fingerprint (%s): %s", key.get_name(), fingerprint)
    logger.info("Verify this fingerprint out-of-band before trusting live retrieval.")
    return 0


def _read_until_idle(chan, idle: float = 2.5, hard: float = 90.0) -> str:
    buf: list[str] = []
    deadline = time.time() + hard
    last = time.time()
    while time.time() < deadline:
        if chan.recv_ready():
            buf.append(chan.recv(65535).decode("utf-8", errors="replace"))
            last = time.time()
        elif time.time() - last > idle:
            break
        else:
            time.sleep(0.1)
    return "".join(buf)


def _retrieve(args) -> int:
    import paramiko

    for command in COMMANDS:            # refuse before opening any socket
        _assert_read_only(command)

    if not os.path.isfile(args.known_hosts):
        raise SystemExit(f"host key not found in {args.known_hosts}; run --save-host-key first")

    user, password = _credentials(args)
    client = paramiko.SSHClient()
    client.load_host_keys(args.known_hosts)
    client.set_missing_host_key_policy(paramiko.RejectPolicy())  # host-key verification required
    try:
        client.connect(args.host, port=args.port, username=user, password=password,
                       timeout=args.timeout, look_for_keys=False, allow_agent=False)
        outputs = _capture(client, args.timeout)
    finally:
        client.close()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for command, fname in COMMANDS.items():
        body = outputs.get(command, "").strip("\n")
        if not body:
            logger.warning("no output captured for '%s'", command)
            continue
        if command == "display current-configuration":
            body = _redact_config(body)              # never store live credential material
        (out_dir / fname).write_text(body + "\n", encoding="utf-8")
        written += 1
    logger.info("Wrote %d/%d read-only command outputs to %s (no config changed).",
                written, len(COMMANDS), out_dir)
    logger.info("Next: python -m scripts.analyze_network_config --vendor huawei "
                "--input-dir %s --snapshot-id huawei_s5720_live", out_dir)
    return 0 if written else 1


def _capture(client, timeout: int) -> dict[str, str]:
    """Run each read-only command in one shell; return {command: cleaned output}."""
    chan = client.invoke_shell()
    chan.settimeout(timeout)
    _read_until_idle(chan, idle=1.5, hard=12)             # drain login banner
    chan.send("screen-length 0 temporary\n")              # disable paging (session only)
    _read_until_idle(chan, idle=1.2, hard=6)
    chan.send("\n")                                       # settle to a clean prompt
    _read_until_idle(chan, idle=1.0, hard=4)
    results: dict[str, str] = {}
    for command in COMMANDS:
        chan.send(command + "\n")
        results[command] = _clean(_read_until_idle(chan, idle=1.3, hard=50), command)
    return results


def _clean(raw: str, command: str) -> str:
    """Drop the echoed command line and prompt lines from one command's output."""
    lines = []
    for line in raw.replace("\r", "").splitlines():
        if re.match(r"^<[^>]+>", line):        # prompt / echoed-command line
            continue
        if line.strip() == command:            # bare echo without prompt
            continue
        lines.append(line)
    return "\n".join(lines).strip("\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` on success)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    if args.save_host_key:
        return _save_host_key(args)
    return _retrieve(args)


if __name__ == "__main__":
    raise SystemExit(main())

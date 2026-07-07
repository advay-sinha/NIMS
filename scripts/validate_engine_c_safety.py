"""Engine C safety validator (static audit).

Statically checks that Engine C keeps its offline, read-only, no-execution
safety posture: no live-device libraries or connection primitives are imported
in ``src/network_config/`` or the Engine C scripts, and the shipped configs keep
their safe defaults (no live access, dry-run remediation, Batfish disabled).

This is an auditing tool — it reads source and config files only and never runs
Engine C, contacts a device or executes a command. It exits non-zero if any
forbidden usage or unsafe default is found.

Usage
-----
    python -m scripts.validate_engine_c_safety
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Repo root = one level above scripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Source trees that must stay free of live-device access primitives.
DEFAULT_SOURCE_ROOTS = [
    REPO_ROOT / "src" / "network_config",
]
DEFAULT_SCRIPT_FILES = [
    "analyze_network_config.py", "dry_run_network_actions.py",
    "compare_network_snapshots.py", "generate_network_config_report.py",
    "export_network_config_dashboard.py", "run_batfish_validation.py",
    # validate_engine_c_safety.py is intentionally excluded: it contains the
    # forbidden token names as detection patterns, not as usages.
]

# token -> regex matched against code with docstrings/comments stripped.
FORBIDDEN_PATTERNS: dict[str, str] = {
    "netmiko": r"(?:^|\W)(?:import\s+netmiko|from\s+netmiko\b)",
    "napalm": r"(?:^|\W)(?:import\s+napalm|from\s+napalm\b)",
    "paramiko": r"(?:^|\W)(?:import\s+paramiko|from\s+paramiko\b)",
    "ConnectHandler": r"\bConnectHandler\b",
    "send_config_set": r"\bsend_config_set\b",
    "socket_connection": r"\bimport\s+socket\b|socket\.socket\s*\(",
    "subprocess_network_exec": r"subprocess\.\w+\([^\n]*\b(?:ssh|telnet|netcat|nc)\b",
}

_TRIPLE = re.compile(r'""".*?"""|\'\'\'.*?\'\'\'', re.DOTALL)


@dataclass(frozen=True)
class Violation:
    """One forbidden usage found in a source file."""

    file: str
    line: int
    token: str
    text: str


def _strip_noncode(source: str) -> str:
    """Blank out triple-quoted blocks and comments so prose is not scanned."""
    without_docstrings = _TRIPLE.sub(lambda m: "\n" * m.group(0).count("\n"),
                                     source)
    cleaned_lines = []
    for line in without_docstrings.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            cleaned_lines.append("")
            continue
        # Drop trailing inline comments (best-effort; ignores '#' in strings).
        cleaned_lines.append(line.split("#", 1)[0] if "#" in line else line)
    return "\n".join(cleaned_lines)


def scan_file(path: Path) -> list[Violation]:
    """Return forbidden-usage violations in a single Python file."""
    source = path.read_text(encoding="utf-8")
    code = _strip_noncode(source)
    lines = code.splitlines()
    violations: list[Violation] = []
    for token, pattern in FORBIDDEN_PATTERNS.items():
        for match in re.finditer(pattern, code):
            line_no = code.count("\n", 0, match.start()) + 1
            text = lines[line_no - 1].strip() if line_no - 1 < len(lines) else ""
            violations.append(Violation(str(path), line_no, token, text))
    return violations


def scan_source_tree(roots: Iterable[Path]) -> list[Violation]:
    """Scan every ``*.py`` file under the given roots (dirs or files)."""
    violations: list[Violation] = []
    for root in roots:
        root = Path(root)
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            if path.is_file() and path.suffix == ".py":
                violations.extend(scan_file(path))
    return violations


def check_config_defaults(configs_dir: Path) -> list[tuple[str, bool]]:
    """Verify the shipped Engine C configs keep safe defaults."""
    import yaml

    def load(name: str) -> dict[str, Any]:
        path = Path(configs_dir) / name
        if not path.is_file():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    nc = load("network_config.yaml").get("network_config", {})
    safety = (nc or {}).get("safety", {})
    rem = load("remediation.yaml").get("global", {})
    exe = load("network_action_executor.yaml").get("global", {})
    bf = load("batfish.yaml").get("global", {})

    return [
        ("network_config.safety.live_device_access is false",
         safety.get("live_device_access") is False),
        ("network_config.safety.remediation_enabled is false",
         safety.get("remediation_enabled") is False),
        ("remediation.global.dry_run_only is true",
         rem.get("dry_run_only") is True),
        ("action_executor.global.allow_live_execution is false",
         exe.get("allow_live_execution") is False),
        ("action_executor.global.mode is dry_run",
         exe.get("mode") == "dry_run"),
        ("batfish.global.enabled is false (disabled by default)",
         bf.get("enabled") is False),
    ]


def run_audit(
    source_roots: Iterable[Path] | None = None,
    configs_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full static safety audit and return a structured result."""
    roots = list(source_roots) if source_roots is not None else (
        DEFAULT_SOURCE_ROOTS
        + [REPO_ROOT / "scripts" / name for name in DEFAULT_SCRIPT_FILES])
    configs = Path(configs_dir) if configs_dir is not None else (
        REPO_ROOT / "configs")

    violations = scan_source_tree(roots)
    config_checks = check_config_defaults(configs)
    passed = not violations and all(ok for _, ok in config_checks)
    return {"violations": violations, "config_checks": config_checks,
            "passed": passed}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` when the audit passes, ``1`` otherwise)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_audit()

    logger.info("Engine C safety audit")
    logger.info("=====================")
    for label, ok in result["config_checks"]:
        logger.info("  [%s] %s", "PASS" if ok else "FAIL", label)
    if result["violations"]:
        logger.error("Forbidden usages found:")
        for v in result["violations"]:
            logger.error("  %s:%d  %s -> %s", v.file, v.line, v.token, v.text)
    else:
        logger.info("  [PASS] no forbidden live-device usages in Engine C source")

    if result["passed"]:
        logger.info("RESULT: PASS — Engine C is offline, read-only and "
                    "execution-free.")
        return 0
    logger.error("RESULT: FAIL — safety audit found issues (see above).")
    return 1


if __name__ == "__main__":
    sys.exit(main())

/**
 * Live-ingestion loader — read-only view of outputs/live_logging.
 *
 * Reads what the Phase 9 offline ingestion pipeline persisted (normalized
 * events, ingestion status/report, checkpoints) and returns SANITIZED data for
 * the frontend. The frontend must never receive secrets, credential values,
 * authorization headers, SNMP communities, private keys or raw payloads —
 * events on disk are already redacted, and this loader applies a defensive
 * second scrub and never exposes raw_events.jsonl.
 *
 * Nothing here polls a device, opens a socket or executes a command.
 */

import { readJson, readJsonl, readText, join, isFile, isDir } from "../readers.js";
import fs from "node:fs";

// Defence-in-depth: key names whose values are dropped if they ever appear.
const SECRET_KEY_RE =
  /(password|passwd|secret|token|api[_-]?key|authorization|community|private[_-]?key|credential|bearer)/i;

const SOURCE_LABELS = {
  sophos_api: "Sophos Central",
  sophos_syslog: "Sophos Firewall",
  hirschmann_snmp: "Hirschmann SNMP",
  hirschmann_traps: "Hirschmann Traps",
  hirschmann_config: "Hirschmann Config",
};

const ALL_SOURCES = Object.keys(SOURCE_LABELS);

/** Recursively drop secret-named keys from a plain object (defensive scrub). */
function scrub(value) {
  if (Array.isArray(value)) return value.map(scrub);
  if (value && typeof value === "object") {
    const out = {};
    for (const [k, v] of Object.entries(value)) {
      if (SECRET_KEY_RE.test(k)) continue;
      out[k] = scrub(v);
    }
    return out;
  }
  return value;
}

/** Sanitise one normalized event for the API (no raw payload, scrubbed maps). */
function sanitizeEvent(e) {
  return {
    event_id: e.event_id,
    timestamp: e.timestamp,
    observed_at: e.observed_at ?? null,
    source_vendor: e.source_vendor,
    source_product: e.source_product,
    source_type: e.source_type,
    source_name: e.source_name ?? null,
    device_id: e.device_id ?? null,
    device_ip: e.device_ip ?? null,
    hostname: e.hostname ?? null,
    category: e.category,
    subcategory: e.subcategory ?? null,
    severity: e.severity,
    message: e.message,
    engine_target: e.engine_target,
    normalized_fields: scrub(e.normalized_fields ?? {}),
    correlation_keys: scrub(e.correlation_keys ?? {}),
  };
}

function statusPath(dir) {
  return join(dir, "ingestion_status.json");
}

/** Overall ingestion status (sanitized). */
export function loadStatus(liveDir) {
  const status = readJson(statusPath(liveDir));
  if (!status) {
    return {
      available: false,
      message:
        "No ingestion status found. Run: python -m scripts.run_live_logger",
    };
  }
  return {
    available: true,
    mode: status.mode ?? "offline",
    readOnly: status.read_only ?? true,
    startedAt: status.started_at ?? null,
    finishedAt: status.finished_at ?? null,
    totalEvents: status.total_events ?? 0,
    eventsByEngine: status.events_by_engine ?? {},
    eventsBySeverity: status.events_by_severity ?? {},
    eventsByVendor: status.events_by_vendor ?? {},
    healthy: (status.sources ?? []).every((s) => s.status !== "failed"),
  };
}

/** Per-source status rows (sanitized — never any credential detail). */
export function loadSources(liveDir) {
  const status = readJson(statusPath(liveDir));
  const rows = (status?.sources ?? []).map((s) => ({
    source: s.source,
    label: SOURCE_LABELS[s.source] ?? s.source,
    engineTarget: s.engine_target,
    status: s.status,
    mode: s.mode,
    events: s.events ?? 0,
    attempts: s.attempts ?? 1,
    errorCategory: s.error_category ?? null,
    errorMessage: s.error_message ?? null,
  }));
  return { available: rows.length > 0, sources: rows };
}

function readEvents(liveDir) {
  const path = join(liveDir, "events.jsonl");
  if (!isFile(path)) return [];
  return readJsonl(path).map(sanitizeEvent);
}

/** Filter + paginate recent normalized events (newest first). */
export function loadEvents(liveDir, filters = {}, limit = 200) {
  let events = readEvents(liveDir);
  const match = (val, q) =>
    q == null || q === "" || String(val ?? "").toLowerCase().includes(String(q).toLowerCase());
  events = events.filter(
    (e) =>
      match(e.source_vendor, filters.vendor) &&
      match(e.source_name, filters.source) &&
      match(e.severity, filters.severity) &&
      match(e.category, filters.category) &&
      match(e.engine_target, filters.engine) &&
      (filters.device == null ||
        filters.device === "" ||
        match(e.device_id, filters.device) ||
        match(e.hostname, filters.device) ||
        match(e.device_ip, filters.device)),
  );
  events.reverse(); // JSONL is append order; newest last -> show newest first
  return {
    available: isFile(join(liveDir, "events.jsonl")),
    total: events.length,
    events: events.slice(0, limit),
  };
}

/** One event by id (sanitized) or null. */
export function getEvent(liveDir, id) {
  const event = readEvents(liveDir).find((e) => e.event_id === id);
  return event ?? null;
}

/** Checkpoint freshness per source (no cursor secrets — our cursors carry none). */
export function loadCheckpoints(liveDir) {
  const dir = join(liveDir, "checkpoints");
  if (!isDir(dir)) return { available: false, checkpoints: [] };
  const now = Date.now();
  const checkpoints = [];
  for (const file of fs.readdirSync(dir)) {
    if (!file.endsWith("_checkpoint.json")) continue;
    const data = readJson(join(dir, file));
    if (!data) continue;
    const updatedAt = data.updated_at ?? data.cursor?.last_poll_time ?? null;
    const ageSeconds = updatedAt ? Math.round((now - Date.parse(updatedAt)) / 1000) : null;
    checkpoints.push({
      source: data.source ?? file.replace("_checkpoint.json", ""),
      label: SOURCE_LABELS[data.source] ?? data.source ?? file,
      updatedAt,
      ageSeconds: Number.isFinite(ageSeconds) ? ageSeconds : null,
      eventCount: data.cursor?.event_count ?? null,
    });
  }
  checkpoints.sort((a, b) => a.source.localeCompare(b.source));
  return { available: checkpoints.length > 0, checkpoints };
}

/** Live-readiness reports (produced by scripts.check_live_readiness). */
export function loadReadiness(liveDir) {
  const data = readJson(join(liveDir, "readiness.json"));
  if (!data) {
    return {
      available: false,
      message:
        "No readiness report found. Run: python -m scripts.check_live_readiness --source all",
    };
  }
  // readiness.json is already secret-free (booleans + env-var names only); scrub
  // defensively anyway in case a future field carries a value.
  return {
    available: true,
    generatedAt: data.generated_at ?? null,
    sources: (data.sources ?? []).map((s) => ({
      source: s.source,
      friendlyName: s.friendly_name ?? s.source,
      engineTarget: s.engine_target,
      status: s.status,
      dependency: s.dependency ?? null,
      dependencyOk: s.dependency_ok ?? true,
      mode: s.mode,
      enabled: Boolean(s.enabled),
      liveEnabled: Boolean(s.live_enabled),
      readOnly: s.read_only ?? true,
      envPresent: scrub(s.env_present ?? {}),
      requiredEnvVars: s.required_env_vars ?? [],
      bindPort: s.bind_port ?? null,
      bindPortAvailable: s.bind_port_available ?? null,
      targetsConfigured: s.targets_configured ?? null,
      safetyProblems: s.safety_problems ?? [],
      remainingSteps: s.remaining_steps ?? [],
      canRunOnceLive: Boolean(s.can_run_once_live),
    })),
  };
}

/** Readiness for one source, or null. */
export function loadReadinessSource(liveDir, source) {
  const all = loadReadiness(liveDir);
  if (!all.available) return all;
  const one = all.sources.find((s) => s.source === source);
  return one ? { available: true, generatedAt: all.generatedAt, source: one } : null;
}

/** The Markdown ingestion report as text. */
export function loadReport(liveDir) {
  const text = readText(join(liveDir, "ingestion_report.md"));
  return { available: Boolean(text), markdown: text ?? "" };
}

/** Health summary: overall + per-source ok/failed + retry/failure rollup. */
export function loadHealth(liveDir) {
  const status = readJson(statusPath(liveDir));
  if (!status) {
    return {
      available: false,
      message: "No ingestion status found. Run: python -m scripts.run_live_logger",
    };
  }
  const sources = status.sources ?? [];
  const failures = sources
    .filter((s) => s.status === "failed" || s.error_category)
    .map((s) => ({
      source: s.source,
      label: SOURCE_LABELS[s.source] ?? s.source,
      status: s.status,
      category: s.error_category ?? null,
      attempts: s.attempts ?? 1,
    }));
  return {
    available: true,
    healthy: sources.every((s) => s.status !== "failed"),
    mode: status.mode ?? "offline",
    readOnly: status.read_only ?? true,
    lastRunAt: status.finished_at ?? null,
    totalEvents: status.total_events ?? 0,
    sourceCount: sources.length,
    okCount: sources.filter((s) => s.status === "ok").length,
    failedCount: sources.filter((s) => s.status === "failed").length,
    disabledCount: sources.filter((s) => s.status === "disabled").length,
    failures,
  };
}

/** Append a sanitized audit entry for a run-once invocation. */
export function appendRunOnceAudit(liveDir, entry) {
  try {
    fs.mkdirSync(liveDir, { recursive: true });
    const line = JSON.stringify({ at: new Date().toISOString(), ...scrub(entry) });
    fs.appendFileSync(join(liveDir, "run_once_audit.jsonl"), line + "\n", "utf-8");
  } catch {
    /* auditing must never break the request path */
  }
}

export { ALL_SOURCES };

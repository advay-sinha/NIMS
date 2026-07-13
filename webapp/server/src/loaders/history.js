/**
 * History loader — labelled past runs across every engine.
 *
 * Ports the labelled-run logic of src/dashboard/loader.py: attach
 * human-readable timestamps to Engine C snapshots and correlation runs
 * (newest first, latest marked) and add the experiment timeline.
 */

import { readJson, join } from "../readers.js";
import { listEngineCSnapshots } from "./engines.js";
import { listCorrelationRuns } from "./correlation.js";

/** Format an ISO timestamp as "YYYY-MM-DD HH:MM UTC" (tolerant). */
export function humanizeTimestamp(value) {
  if (!value) return null;
  const parsed = new Date(String(value).trim().replace("Z", "+00:00"));
  if (Number.isNaN(parsed.getTime())) return null;
  const pad = (n) => String(n).padStart(2, "0");
  return (
    `${parsed.getUTCFullYear()}-${pad(parsed.getUTCMonth() + 1)}-` +
    `${pad(parsed.getUTCDate())} ${pad(parsed.getUTCHours())}:` +
    `${pad(parsed.getUTCMinutes())} UTC`
  );
}

function snapshotTimestamp(networkConfigDir, snapshot) {
  const candidates = [
    ["dashboard/export_metadata.json", "generated_at"],
    ["dashboard/dashboard_summary.json", "generated_at"],
    ["metadata.json", "generated_at"],
    ["metadata.json", "timestamp"],
    ["metadata.json", "created_at"],
  ];
  for (const [rel, key] of candidates) {
    const data = readJson(join(networkConfigDir, snapshot, rel));
    if (data && typeof data === "object" && data[key]) return String(data[key]);
  }
  return null;
}

function correlationTimestamp(correlationDir, run) {
  const data = readJson(join(correlationDir, run, "correlation_summary.json"));
  return data && typeof data === "object" && data.timestamp
    ? String(data.timestamp)
    : null;
}

/** Attach labels + timestamps to ids, newest first, latest marked. */
function labeled(ids, timestampFn, kind) {
  const items = ids.map((id) => {
    const timestamp = timestampFn(id);
    const human = humanizeTimestamp(timestamp);
    return {
      id,
      timestamp,
      human,
      label: human ? `${kind} · ${human}` : `${kind} · ${id}`,
      isLatest: false,
    };
  });
  items.sort((a, b) =>
    `${b.timestamp ?? ""}${b.id}`.localeCompare(`${a.timestamp ?? ""}${a.id}`),
  );
  if (items.length) {
    items[0].isLatest = true;
    items[0].label += " (latest)";
  }
  return items;
}

/** Assessment runs (Engine C snapshots), newest first. */
export function labeledSnapshots(networkConfigDir) {
  return labeled(
    listEngineCSnapshots(networkConfigDir),
    (s) => snapshotTimestamp(networkConfigDir, s),
    "Assessment Run",
  );
}

/** Incident runs (correlation runs), newest first. */
export function labeledCorrelationRuns(correlationDir) {
  return labeled(
    listCorrelationRuns(correlationDir),
    (r) => correlationTimestamp(correlationDir, r),
    "Incident Run",
  );
}

/** The default selection: latest available run, else the configured id. */
export function resolveDefault(items, configured = null) {
  return items.length ? items[0].id : configured;
}

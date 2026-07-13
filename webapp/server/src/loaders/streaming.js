/**
 * Live-monitor loader — offline streaming demo state.
 *
 * Reads what scripts/run_streaming_demo.py persisted under outputs/streaming/
 * (current state, active incidents, recent events, run summary). Read-only:
 * nothing here polls, captures or executes.
 */

import { readJson, readJsonl, join, isFile } from "../readers.js";

/** Load the streaming demo's persisted current state for the live monitor. */
export function loadLiveMonitor(streamingDir) {
  const currentDir = join(streamingDir, "current");
  const state = readJson(join(currentDir, "current_state.json"));
  const incidents = readJson(join(currentDir, "active_incidents.json"));
  const events = readJson(join(currentDir, "recent_events.json"));
  const summary = readJson(join(streamingDir, "stream_summary.json"));

  if (!state && !summary) {
    return {
      available: false,
      message:
        "No streaming demo output found. Run: python -m scripts.run_streaming_demo",
    };
  }
  const s = state ?? summary ?? {};
  return {
    available: true,
    startedAt: s.started_at ?? null,
    lastEventAt: s.last_event_at ?? null,
    totalEvents: s.total_events ?? 0,
    eventsByType: s.events_by_type ?? {},
    eventsBySeverity: s.events_by_severity ?? {},
    eventsByEngine: s.events_by_engine ?? {},
    activeIncidentCount: s.active_incident_count ?? 0,
    criticalIncidentCount: s.critical_incident_count ?? 0,
    activeDeviceCount: s.active_device_count ?? 0,
    activeIncidents: Array.isArray(incidents) ? incidents : [],
    recentEvents: Array.isArray(events) ? events : [],
    safetyNote:
      s.safety_note ??
      "Offline demo replay — no device access, no packet capture, no command execution.",
    message: null,
  };
}

/** Full replayed event log (newest first) for the history section. */
export function loadEventHistory(streamingDir, limit = 200) {
  const logPath = join(streamingDir, "events.jsonl");
  if (!isFile(logPath)) return { available: false, events: [] };
  const events = readJsonl(logPath);
  events.sort((a, b) => (b.seq ?? 0) - (a.seq ?? 0));
  return { available: events.length > 0, events: events.slice(0, limit) };
}

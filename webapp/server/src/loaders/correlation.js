/**
 * Correlation-engine loader — incident runs under outputs/correlation/.
 *
 * A run directory contains correlation_summary.json, incidents.json,
 * signals.json and a markdown report. Loaders tolerate partial runs.
 */

import { readJson, readText, listDirs, join, isFile } from "../readers.js";

/** Run ids that contain a summary or incidents file, sorted. */
export function listCorrelationRuns(correlationDir) {
  return listDirs(correlationDir).filter(
    (name) =>
      isFile(join(correlationDir, name, "correlation_summary.json")) ||
      isFile(join(correlationDir, name, "incidents.json")),
  );
}

/** One correlation run's incidents, signals, summary and report text. */
export function loadCorrelation(correlationDir, correlationId) {
  const runDir = join(correlationDir, correlationId);
  const summary = readJson(join(runDir, "correlation_summary.json"));
  const incidents = readJson(join(runDir, "incidents.json"));
  const signals = readJson(join(runDir, "signals.json"));

  if (summary === null && incidents === null) {
    return {
      available: false,
      correlationId,
      incidents: [],
      signals: [],
      summary: {},
      message:
        "Correlation output not found. Run: python -m scripts.run_correlation",
    };
  }
  return {
    available: true,
    correlationId,
    incidents: Array.isArray(incidents) ? incidents : [],
    signals: Array.isArray(signals) ? signals : [],
    summary: summary && typeof summary === "object" ? summary : {},
    reportMarkdown: readText(join(runDir, "correlation_report.md")),
    message: null,
  };
}

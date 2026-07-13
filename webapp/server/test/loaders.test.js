/**
 * Loader tests — run against the repository's committed sample artefacts
 * plus synthetic fixtures for edge cases. No network, no live devices.
 */

import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { loadConfig } from "../src/config.js";
import { readJson, readJsonl } from "../src/readers.js";
import { loadLiveMonitor, loadEventHistory } from "../src/loaders/streaming.js";
import { loadExperiments, bestRunsMatrix } from "../src/loaders/training.js";
import {
  loadEngineA,
  loadEngineB,
  loadEngineC,
  listEngineCSnapshots,
} from "../src/loaders/engines.js";
import { loadCorrelation, listCorrelationRuns } from "../src/loaders/correlation.js";
import {
  humanizeTimestamp,
  labeledSnapshots,
  resolveDefault,
} from "../src/loaders/history.js";
import { computeOverview, buildExecutiveSummary } from "../src/loaders/overview.js";

const config = loadConfig();
const { dirs } = config;

// ------------------------------------------------------------------ readers

test("readJson returns null for a missing file", () => {
  assert.equal(readJson(path.join(os.tmpdir(), "netsen-nope.json")), null);
});

test("readJsonl skips malformed lines", () => {
  const file = path.join(os.tmpdir(), `netsen-test-${process.pid}.jsonl`);
  fs.writeFileSync(file, '{"a":1}\nnot json\n{"b":2}\n');
  try {
    assert.deepEqual(readJsonl(file), [{ a: 1 }, { b: 2 }]);
  } finally {
    fs.unlinkSync(file);
  }
});

// ------------------------------------------------------------------- config

test("config resolves absolute artefact directories", () => {
  assert.ok(path.isAbsolute(dirs.experiments));
  assert.ok(dirs.networkConfig.endsWith(path.join("outputs", "network_config")));
  assert.equal(typeof config.server.port, "number");
});

// ---------------------------------------------------------------- streaming

test("live monitor loads persisted streaming state when present", () => {
  const live = loadLiveMonitor(dirs.streaming);
  if (!live.available) {
    assert.match(live.message, /run_streaming_demo/);
    return;
  }
  assert.equal(typeof live.totalEvents, "number");
  assert.ok(Array.isArray(live.activeIncidents));
  assert.ok(live.safetyNote.length > 0);
});

test("live monitor degrades gracefully for a missing directory", () => {
  const live = loadLiveMonitor(path.join(os.tmpdir(), "netsen-absent"));
  assert.equal(live.available, false);
  assert.match(live.message, /streaming/i);
});

test("event history sorts newest first", () => {
  const history = loadEventHistory(dirs.streaming, 50);
  if (!history.available) return;
  const seqs = history.events.map((e) => e.seq ?? 0);
  const sorted = [...seqs].sort((a, b) => b - a);
  assert.deepEqual(seqs, sorted);
});

// ----------------------------------------------------------------- training

test("experiments walker finds runs with split metrics", () => {
  const experiments = loadExperiments(dirs.experiments);
  if (!experiments.available) return;
  assert.ok(experiments.runCount > 0);
  const run = experiments.runs[0];
  assert.ok(run.dataset && run.model && run.experimentId);
  assert.ok(run.splits.test || run.splits.train);
});

test("best-runs matrix keeps at most one run per dataset+model", () => {
  const experiments = loadExperiments(dirs.experiments);
  const matrix = bestRunsMatrix(experiments);
  const keys = matrix.map((r) => `${r.dataset}::${r.model}`);
  assert.equal(new Set(keys).size, keys.length);
  for (const row of matrix) assert.equal(typeof row.testF1, "number");
});

// ------------------------------------------------------------------ engines

test("Engine A loader reads the production registry", () => {
  const engineA = loadEngineA(dirs);
  if (!engineA.available) return;
  for (const model of engineA.models) {
    assert.ok(model.dataset);
    assert.ok(model.experimentId);
  }
});

test("Engine B loader computes an anomaly rate per dataset", () => {
  const engineB = loadEngineB(dirs.networkHealth);
  if (!engineB.available) return;
  for (const ds of engineB.datasets) {
    assert.ok(ds.anomalyRate >= 0 && ds.anomalyRate <= 1);
  }
});

test("Engine C snapshot listing excludes the diffs directory", () => {
  const snapshots = listEngineCSnapshots(dirs.networkConfig);
  assert.ok(!snapshots.includes("diffs"));
});

test("Engine C loader reports missing exports with guidance", () => {
  const missing = loadEngineC(dirs.networkConfig, "no_such_snapshot");
  assert.equal(missing.available, false);
  assert.match(missing.message, /export_network_config_dashboard/);
});

// -------------------------------------------------------------- correlation

test("correlation loader returns incidents for a sample run", () => {
  const runs = listCorrelationRuns(dirs.correlation);
  if (!runs.length) return;
  const data = loadCorrelation(dirs.correlation, runs[0]);
  assert.equal(data.available, true);
  assert.ok(Array.isArray(data.incidents));
  assert.ok(Array.isArray(data.signals));
});

// ------------------------------------------------------------------ history

test("humanizeTimestamp formats ISO input and rejects junk", () => {
  assert.equal(
    humanizeTimestamp("2026-07-07T20:16:45.738765+00:00"),
    "2026-07-07 20:16 UTC",
  );
  assert.equal(humanizeTimestamp("not-a-date"), null);
  assert.equal(humanizeTimestamp(null), null);
});

test("labeled snapshots mark exactly one latest", () => {
  const items = labeledSnapshots(dirs.networkConfig);
  if (!items.length) return;
  assert.equal(items.filter((i) => i.isLatest).length, 1);
  assert.match(items[0].label, /latest/);
});

test("resolveDefault prefers the latest run over the configured id", () => {
  assert.equal(
    resolveDefault([{ id: "b" }, { id: "a" }], "configured"),
    "b",
  );
  assert.equal(resolveDefault([], "configured"), "configured");
});

// ----------------------------------------------------------------- overview

test("executive summary flags critical incidents as attention", () => {
  const engineC = { views: {}, dryRunExecutedCount: 0 };
  const correlation = {
    incidents: [
      { incident_id: "I1", severity: "high", affected_devices: ["sw1"] },
      { incident_id: "I2", severity: "low" },
    ],
  };
  const summary = buildExecutiveSummary(engineC, correlation, { available: false });
  assert.equal(summary.networkStatusLevel, "attention");
  assert.equal(summary.criticalIncidentCount, 1);
  assert.deepEqual(summary.affectedDevices, ["sw1"]);
});

test("overview cards count high/critical incidents", () => {
  const cards = computeOverview(
    { views: { dashboard_summary: { finding_count: 3 } }, dryRunExecutedCount: 2 },
    { incidents: [{ severity: "critical" }, { severity: "info" }] },
    { productionModelCount: 3 },
    { anomalyStatus: "x: 1.0% anomalous (test)" },
  );
  assert.equal(cards.totalIncidents, 2);
  assert.equal(cards.highCriticalIncidents, 1);
  assert.equal(cards.engineCFindings, 3);
});

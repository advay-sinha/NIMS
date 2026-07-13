/**
 * API routes for the NetSentinel web frontend.
 *
 * Every route is read-only and artefact-driven. Responses share the
 * { available, ... } convention so the frontend can render a helpful
 * empty state (including the exact command to produce the artefact).
 */

import { Router } from "express";
import { loadLiveMonitor, loadEventHistory } from "./loaders/streaming.js";
import {
  loadExperiments,
  bestRunsMatrix,
  loadValidationReports,
  loadFeatureReports,
  experimentArtifacts,
} from "./loaders/training.js";
import {
  loadEngineA,
  loadEngineB,
  loadEngineC,
  listEngineCSnapshots,
} from "./loaders/engines.js";
import { loadCorrelation, listCorrelationRuns } from "./loaders/correlation.js";
import {
  labeledSnapshots,
  labeledCorrelationRuns,
  resolveDefault,
} from "./loaders/history.js";
import { computeOverview, buildExecutiveSummary } from "./loaders/overview.js";

/** Tiny TTL cache so bursts of UI requests do not re-read the disk. */
function makeCache(ttlSeconds) {
  const store = new Map();
  return (key, compute) => {
    const hit = store.get(key);
    const now = Date.now();
    if (hit && now - hit.at < ttlSeconds * 1000) return hit.value;
    const value = compute();
    store.set(key, { at: now, value });
    return value;
  };
}

/** Sanitise a run/snapshot id from the query string (path-safe names only). */
function safeId(value) {
  return typeof value === "string" && /^[\w.-]+$/.test(value) ? value : null;
}

/** Build the /api router bound to the loaded configuration. */
export function buildRouter(config) {
  const { dirs, sections, safety } = config;
  const cached = makeCache(config.server.cacheSeconds);
  const router = Router();

  const latestSnapshot = () =>
    resolveDefault(cached("snapshots", () => labeledSnapshots(dirs.networkConfig)));
  const latestCorrelation = () =>
    resolveDefault(
      cached("corr-runs", () => labeledCorrelationRuns(dirs.correlation)),
    );

  router.get("/meta", (_req, res) => {
    res.json({
      sections,
      safety,
      defaults: {
        engineCSnapshot: latestSnapshot(),
        correlationRun: latestCorrelation(),
      },
    });
  });

  router.get("/overview", (req, res) => {
    const snapshotId = safeId(req.query.snapshot) ?? latestSnapshot();
    const runId = safeId(req.query.run) ?? latestCorrelation();
    const engineC = snapshotId
      ? cached(`ec:${snapshotId}`, () => loadEngineC(dirs.networkConfig, snapshotId))
      : { available: false, views: {}, dryRunExecutedCount: 0 };
    const correlation = runId
      ? cached(`corr:${runId}`, () => loadCorrelation(dirs.correlation, runId))
      : { available: false, incidents: [], signals: [], summary: {} };
    const engineA = cached("engine-a", () => loadEngineA(dirs));
    const engineB = cached("engine-b", () => loadEngineB(dirs.networkHealth));
    res.json({
      snapshotId,
      correlationId: runId,
      cards: computeOverview(engineC, correlation, engineA, engineB),
      executive: buildExecutiveSummary(engineC, correlation, engineB),
      engineAvailability: {
        engineA: engineA.available,
        engineB: engineB.available,
        engineC: engineC.available,
        correlation: correlation.available,
      },
    });
  });

  router.get("/live", (_req, res) => {
    res.json(cached("live", () => loadLiveMonitor(dirs.streaming)));
  });

  router.get("/training", (_req, res) => {
    const experiments = cached("experiments", () => loadExperiments(dirs.experiments));
    res.json({
      ...experiments,
      bestRuns: bestRunsMatrix(experiments),
      validation: cached("validation", () => loadValidationReports(dirs.reports)),
      features: cached("features", () => loadFeatureReports(dirs.features)),
    });
  });

  router.get("/engine-a", (_req, res) => {
    const engineA = cached("engine-a", () => loadEngineA(dirs));
    const models = engineA.models.map((m) => ({
      ...m,
      artifacts: m.experimentId
        ? experimentArtifacts(dirs, m.experimentId)
        : null,
    }));
    res.json({ ...engineA, models });
  });

  router.get("/engine-b", (_req, res) => {
    res.json(cached("engine-b", () => loadEngineB(dirs.networkHealth)));
  });

  router.get("/engine-c", (req, res) => {
    const snapshotId = safeId(req.query.snapshot) ?? latestSnapshot();
    if (!snapshotId) {
      res.json({
        available: false,
        snapshots: [],
        message:
          "No Engine C snapshots found. Run: python -m scripts.analyze_network_config",
      });
      return;
    }
    const data = cached(`ec:${snapshotId}`, () =>
      loadEngineC(dirs.networkConfig, snapshotId),
    );
    res.json({
      ...data,
      snapshots: cached("snapshot-ids", () =>
        listEngineCSnapshots(dirs.networkConfig),
      ),
    });
  });

  router.get("/correlation", (req, res) => {
    const runId = safeId(req.query.run) ?? latestCorrelation();
    if (!runId) {
      res.json({
        available: false,
        runs: [],
        message:
          "No correlation runs found. Run: python -m scripts.run_correlation",
      });
      return;
    }
    const data = cached(`corr:${runId}`, () =>
      loadCorrelation(dirs.correlation, runId),
    );
    res.json({
      ...data,
      runs: cached("corr-run-ids", () => listCorrelationRuns(dirs.correlation)),
    });
  });

  router.get("/history", (_req, res) => {
    res.json({
      assessmentRuns: cached("snapshots", () => labeledSnapshots(dirs.networkConfig)),
      incidentRuns: cached("corr-runs", () =>
        labeledCorrelationRuns(dirs.correlation),
      ),
      eventHistory: cached("event-history", () => loadEventHistory(dirs.streaming)),
    });
  });

  router.get("/safety", (req, res) => {
    const snapshotId = safeId(req.query.snapshot) ?? latestSnapshot();
    const engineC = snapshotId
      ? cached(`ec:${snapshotId}`, () => loadEngineC(dirs.networkConfig, snapshotId))
      : { available: false, views: {}, dryRunExecutedCount: 0 };
    const audit = engineC.views?.action_audit_view ?? {};
    const live = cached("live", () => loadLiveMonitor(dirs.streaming));
    res.json({
      allowActions: safety.allowActions,
      showNoExecutionBanner: safety.showNoExecutionBanner,
      guarantees: {
        readOnly: true,
        offlineArtifactsOnly: true,
        noLiveDeviceAccess: true,
        noPacketCapture: true,
        noCommandExecution: true,
        confirmationRequiredForActions: true,
      },
      dryRun: {
        snapshotId: snapshotId ?? null,
        executedCount: Number(engineC.dryRunExecutedCount ?? 0),
        auditEntryCount: Number(
          audit.entry_count ?? audit.total_entries ?? audit.executed_count ?? 0,
        ),
      },
      safetyNotes: [
        engineC.views?.dashboard_summary?.safety_note,
        live.available ? live.safetyNote : null,
      ].filter(Boolean),
    });
  });

  return router;
}

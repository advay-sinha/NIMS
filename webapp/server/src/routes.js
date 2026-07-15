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
import {
  loadStatus as loadLiveStatus,
  loadSources as loadLiveSources,
  loadEvents as loadLiveEvents,
  getEvent as getLiveEvent,
  loadCheckpoints as loadLiveCheckpoints,
  loadReport as loadLiveReport,
  loadHealth as loadLiveHealth,
  loadReadiness as loadLiveReadiness,
  loadReadinessSource as loadLiveReadinessSource,
  appendRunOnceAudit,
  ALL_SOURCES as LIVE_SOURCES,
} from "./loaders/liveIngestion.js";

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

  // -------------------------------------------------- live ingestion (Phase 9)
  //
  // Read-only views of outputs/live_logging. Responses are sanitized (no
  // secrets, no credential values, no raw payloads). The optional run-once
  // control is DISABLED by default and never executes anything.

  const LIVE_SAFETY_TEXT =
    "Read-only live ingestion. NIMS collects and analyzes telemetry only. No " +
    "firewall or switch configuration is changed, and no remediation command " +
    "is executed.";

  const liveDir = dirs.liveLogging;

  router.get("/live-ingestion/status", (_req, res) => {
    res.json({
      ...cached("li-status", () => loadLiveStatus(liveDir)),
      safetyText: LIVE_SAFETY_TEXT,
    });
  });

  router.get("/live-ingestion/health", (_req, res) => {
    res.json({ ...cached("li-health", () => loadLiveHealth(liveDir)), safetyText: LIVE_SAFETY_TEXT });
  });

  router.get("/live-ingestion/sources", (_req, res) => {
    res.json(cached("li-sources", () => loadLiveSources(liveDir)));
  });

  router.get("/live-ingestion/events", (req, res) => {
    const filters = {
      vendor: safeFilter(req.query.vendor),
      source: safeFilter(req.query.source),
      severity: safeFilter(req.query.severity),
      device: safeFilter(req.query.device),
      category: safeFilter(req.query.category),
      engine: safeFilter(req.query.engine),
    };
    const limit = Math.min(Number(req.query.limit) || 200, 1000);
    res.json(loadLiveEvents(liveDir, filters, limit));
  });

  router.get("/live-ingestion/events/:id", (req, res) => {
    const event = safeId(req.params.id) ? getLiveEvent(liveDir, req.params.id) : null;
    if (!event) {
      res.status(404).json({ available: false, message: "Event not found." });
      return;
    }
    res.json({ available: true, event });
  });

  router.get("/live-ingestion/checkpoints", (_req, res) => {
    res.json(cached("li-checkpoints", () => loadLiveCheckpoints(liveDir)));
  });

  router.get("/live-ingestion/report", (_req, res) => {
    res.json(cached("li-report", () => loadLiveReport(liveDir)));
  });

  router.get("/live-ingestion/readiness", (_req, res) => {
    res.json({
      ...cached("li-readiness", () => loadLiveReadiness(liveDir)),
      allowRunOnce: Boolean(config.liveIngestion?.allowRunOnce),
      safetyText: LIVE_SAFETY_TEXT,
    });
  });

  router.get("/live-ingestion/readiness/:source", (req, res) => {
    if (!LIVE_SOURCES.includes(req.params.source)) {
      res.status(404).json({ available: false, message: "Unknown source." });
      return;
    }
    const data = loadLiveReadinessSource(liveDir, req.params.source);
    if (!data) {
      res.status(404).json({ available: false, message: "No readiness for source." });
      return;
    }
    res.json(data);
  });

  // Optional, DISABLED by default. Gated by config; validates the source
  // against a fixed allowlist; rejects arbitrary hosts/ports/OIDs/paths by
  // never accepting any; read-only; returns job metadata only (no execution).
  const runOnce = (req, res) => {
    const source = req.params.source ?? "all";
    if (!config.liveIngestion?.allowRunOnce) {
      appendRunOnceAudit(liveDir, { action: "run-once", source, outcome: "denied", status: 403 });
      res.status(403).json({
        accepted: false,
        reason:
          "Live-ingestion run-once is disabled. Enable it in configs/webapp.yaml " +
          "(live_ingestion.allow_run_once) after approval, and run the offline " +
          "CLI: python -m scripts.run_live_logger",
      });
      return;
    }
    if (req.params.source && !LIVE_SOURCES.includes(req.params.source)) {
      appendRunOnceAudit(liveDir, { action: "run-once", source, outcome: "rejected", status: 400 });
      res.status(400).json({ accepted: false, reason: "Unknown source." });
      return;
    }
    appendRunOnceAudit(liveDir, { action: "run-once", source, outcome: "accepted", status: 202 });
    res.status(202).json({
      accepted: true,
      readOnly: true,
      source: source ?? "all",
      note:
        "Acknowledged. Ingestion runs only through the offline CLI (python -m " +
        "scripts.run_live_logger); this endpoint records intent and never " +
        "executes commands or contacts a device.",
    });
  };
  router.post("/live-ingestion/run-once", runOnce);
  router.post("/live-ingestion/sources/:source/run-once", runOnce);

  return router;
}

/** Sanitise a free-text filter value (bounded, printable). */
function safeFilter(value) {
  if (typeof value !== "string") return null;
  const trimmed = value.trim().slice(0, 64);
  return /^[\w.\- :/]*$/.test(trimmed) ? trimmed : null;
}

/**
 * Engine A / B / C artefact loaders — direct port of src/dashboard/loader.py.
 *
 * Engine A: production registry + best-per-dataset metrics.
 * Engine B: latest network-health experiment per dataset.
 * Engine C: dashboard-export views for one snapshot.
 */

import { readJson, listDirs, join, isFile, isDir } from "../readers.js";

// Files written by scripts.export_network_config_dashboard (2 optional).
const ENGINE_C_VIEWS = [
  "dashboard_summary",
  "inventory_view",
  "topology_view",
  "findings_view",
  "remediation_view",
  "action_audit_view",
  "risk_timeline",
  "device_health_cards",
  "export_metadata",
];
const ENGINE_C_OPTIONAL = ["diff_view", "verification_view"];
const NON_SNAPSHOT_DIRS = new Set(["diffs"]);

// ------------------------------------------------------------------ Engine A

/** Production models from the registry with test F1 / ROC AUC attached. */
export function loadEngineA(dirs) {
  const production = readJson(join(dirs.registry, "production.json")) ?? {};
  const best = readJson(join(dirs.registry, "best_per_dataset.json")) ?? {};

  const models = [];
  for (const [dataset, entry] of Object.entries(production)) {
    if (!entry || typeof entry !== "object") continue;
    const bestEntry = best[dataset];
    models.push({
      dataset,
      modelType: entry.model_type ?? null,
      experimentId: entry.experiment_id ?? null,
      testF1: typeof bestEntry?.value === "number" ? bestEntry.value : null,
      rocAuc: engineARoc(dirs.experiments, dataset, entry),
      promotedAt: entry.promoted_at ?? null,
      reason: entry.reason ?? null,
    });
  }
  models.sort((a, b) => a.dataset.localeCompare(b.dataset));

  const reportPath = join(dirs.reports, "model_validation_report.md");
  return {
    available: models.length > 0,
    productionModelCount: models.length,
    models,
    validationReportAvailable: isFile(reportPath),
    message: models.length
      ? null
      : "No Engine A production models registered.",
  };
}

function engineARoc(experimentsDir, dataset, entry) {
  const expId = entry.experiment_id;
  const model = entry.model_type;
  if (!expId || !model) return null;
  const metrics = readJson(
    join(experimentsDir, dataset, model, expId, "metrics.json"),
  );
  const roc = metrics?.test?.roc_auc;
  return typeof roc === "number" ? roc : null;
}

// ------------------------------------------------------------------ Engine B

/** Latest network-health experiment metrics per dataset. */
export function loadEngineB(networkHealthDir) {
  const expRoot = join(networkHealthDir, "experiments");
  const datasets = [];
  for (const dataset of listDirs(expRoot)) {
    const entry = latestEngineBExperiment(join(expRoot, dataset), dataset);
    if (entry) datasets.push(entry);
  }
  let anomalyStatus = null;
  if (datasets.length) {
    const top = datasets.reduce((a, b) =>
      (b.anomalyRate ?? 0) > (a.anomalyRate ?? 0) ? b : a,
    );
    anomalyStatus = `${top.dataset}: ${(top.anomalyRate * 100).toFixed(1)}% anomalous (test)`;
  }
  return {
    available: datasets.length > 0,
    datasets,
    anomalyStatus,
    message: datasets.length
      ? null
      : "No Engine B network-health experiments found.",
  };
}

function latestEngineBExperiment(datasetDir, dataset) {
  // Runs live at <dataset>/<model>/<run_id>/metrics.json.
  const runs = [];
  for (const model of listDirs(datasetDir)) {
    for (const runId of listDirs(join(datasetDir, model))) {
      const runDir = join(datasetDir, model, runId);
      if (isFile(join(runDir, "metrics.json"))) runs.push({ runDir, runId });
    }
  }
  if (!runs.length) return null;
  runs.sort((a, b) => a.runId.localeCompare(b.runId));
  const { runDir, runId } = runs[runs.length - 1];
  const metrics = readJson(join(runDir, "metrics.json")) ?? {};
  const manifest = readJson(join(runDir, "manifest.json")) ?? {};
  const test = metrics.test && typeof metrics.test === "object" ? metrics.test : {};
  const nSamples = Number(test.n_samples ?? 0);
  const nPred = Number(test.n_anomalous_predicted ?? 0);
  return {
    dataset,
    experimentId: manifest.experiment_id ?? runId,
    modelName: manifest.model_name ?? null,
    labeled: Boolean(manifest.labeled),
    precision: test.precision ?? null,
    recall: test.recall ?? null,
    f1: test.f1 ?? null,
    rocAuc: test.roc_auc ?? null,
    nSamples,
    nAnomalousPredicted: nPred,
    anomalyRate: nSamples ? nPred / nSamples : 0,
  };
}

// ------------------------------------------------------------------ Engine C

/** Snapshot ids (dirs) under network_config/, ignoring non-snapshot dirs. */
export function listEngineCSnapshots(networkConfigDir) {
  return listDirs(networkConfigDir).filter(
    (name) =>
      !NON_SNAPSHOT_DIRS.has(name) &&
      (isDir(join(networkConfigDir, name, "dashboard")) ||
        isFile(join(networkConfigDir, name, "inventory.json"))),
  );
}

/** Dashboard-export views for one Engine C snapshot. */
export function loadEngineC(networkConfigDir, snapshotId) {
  const dashDir = join(networkConfigDir, snapshotId, "dashboard");
  if (!isDir(dashDir)) {
    return {
      available: false,
      snapshotId,
      views: {},
      missing: [...ENGINE_C_VIEWS],
      dryRunExecutedCount: 0,
      message:
        "Dashboard exports not found. Run: python -m scripts.export_network_config_dashboard " +
        `--snapshot-id ${snapshotId}`,
    };
  }
  const views = {};
  const missing = [];
  for (const name of ENGINE_C_VIEWS) {
    const data = readJson(join(dashDir, `${name}.json`));
    if (data === null) missing.push(name);
    else views[name] = data;
  }
  for (const name of ENGINE_C_OPTIONAL) {
    const data = readJson(join(dashDir, `${name}.json`));
    if (data !== null) views[name] = data;
  }
  const executed = Number(views.action_audit_view?.executed_count ?? 0);
  return {
    available: Object.keys(views).length > 0,
    snapshotId,
    views,
    missing,
    dryRunExecutedCount: executed,
    message: missing.length
      ? `Some Engine C dashboard views are missing (${missing.join(", ")}).`
      : null,
  };
}

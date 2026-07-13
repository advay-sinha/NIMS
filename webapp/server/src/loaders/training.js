/**
 * Training / testing / validation explorer loader.
 *
 * Walks outputs/experiments/<dataset>/<model>/<run_id>/ and summarises each
 * run's train / validation / test metrics plus manifest metadata (seed,
 * device, params). Also surfaces validation reports and feature reports.
 */

import {
  readJson,
  readText,
  listDirs,
  join,
  isFile,
  isDir,
} from "../readers.js";

const SPLITS = ["train", "validation", "test"];
const METRIC_KEYS = [
  "accuracy",
  "precision",
  "recall",
  "f1",
  "roc_auc",
  "false_positive_rate",
];

function splitSummary(split) {
  if (!split || typeof split !== "object") return null;
  const out = {};
  for (const key of METRIC_KEYS) {
    if (typeof split[key] === "number") out[key] = split[key];
  }
  if (typeof split.n_classes === "number") out.n_classes = split.n_classes;
  return Object.keys(out).length ? out : null;
}

function loadRun(runDir, dataset, model, runId) {
  const metrics = readJson(join(runDir, "metrics.json"));
  if (!metrics) return null;
  const manifest = readJson(join(runDir, "manifest.json")) ?? {};
  const splits = {};
  for (const split of SPLITS) {
    const summary = splitSummary(metrics[split]);
    if (summary) splits[split] = summary;
  }
  return {
    experimentId: manifest.experiment_id ?? runId,
    dataset,
    model,
    createdAt: manifest.created_at ?? null,
    seed: manifest.seed ?? null,
    device: manifest.model?.device ?? null,
    supervised: manifest.model?.supervised ?? null,
    params: manifest.model?.params ?? {},
    splits,
    testF1: splits.test?.f1 ?? null,
    testRocAuc: splits.test?.roc_auc ?? null,
  };
}

/** All experiment runs grouped by dataset and model, newest first per group. */
export function loadExperiments(experimentsDir) {
  const runs = [];
  for (const dataset of listDirs(experimentsDir)) {
    const datasetDir = join(experimentsDir, dataset);
    for (const model of listDirs(datasetDir)) {
      const modelDir = join(datasetDir, model);
      for (const runId of listDirs(modelDir)) {
        const run = loadRun(join(modelDir, runId), dataset, model, runId);
        if (run) runs.push(run);
      }
    }
  }
  runs.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
  return {
    available: runs.length > 0,
    runCount: runs.length,
    datasets: [...new Set(runs.map((r) => r.dataset))].sort(),
    models: [...new Set(runs.map((r) => r.model))].sort(),
    runs,
    message: runs.length
      ? null
      : "No experiments found. Run: python -m scripts.train_model",
  };
}

/** Best test-F1 run per dataset+model pair (the comparison matrix). */
export function bestRunsMatrix(experiments) {
  const best = new Map();
  for (const run of experiments.runs ?? []) {
    if (run.testF1 == null) continue;
    const key = `${run.dataset}::${run.model}`;
    const prev = best.get(key);
    if (!prev || run.testF1 > prev.testF1) best.set(key, run);
  }
  return [...best.values()].map((run) => ({
    dataset: run.dataset,
    model: run.model,
    experimentId: run.experimentId,
    testF1: run.testF1,
    testRocAuc: run.testRocAuc,
    testFpr: run.splits.test?.false_positive_rate ?? null,
  }));
}

/** Validation reports (per-dataset JSON + markdown report text). */
export function loadValidationReports(reportsDir) {
  const reports = [];
  for (const name of ["nsl_kdd", "unsw_nb15", "cicids2017"]) {
    const data = readJson(join(reportsDir, `${name}_validation.json`));
    if (data) reports.push({ dataset: name, ...summariseValidation(data) });
  }
  const reportMd = readText(join(reportsDir, "model_validation_report.md"));
  return {
    available: reports.length > 0 || Boolean(reportMd),
    reports,
    reportMarkdown: reportMd,
  };
}

function summariseValidation(data) {
  if (Array.isArray(data)) return { entries: data.length };
  if (data && typeof data === "object") {
    const out = { entries: Object.keys(data).length };
    for (const key of ["status", "passed", "generated_at"]) {
      if (data[key] !== undefined) out[key] = data[key];
    }
    return out;
  }
  return { entries: 0 };
}

/** Feature-engineering reports per dataset (selected/removed features). */
export function loadFeatureReports(featuresDir) {
  const datasets = [];
  for (const dataset of listDirs(featuresDir)) {
    const dir = join(featuresDir, dataset);
    if (!isDir(dir)) continue;
    const report = readJson(join(dir, "feature_report.json"));
    const selected = readJson(join(dir, "selected_features.json"));
    const removed = readJson(join(dir, "removed_features.json"));
    if (!report && !selected) continue;
    datasets.push({
      dataset,
      selectedCount: Array.isArray(selected) ? selected.length : null,
      removedCount: Array.isArray(removed) ? removed.length : null,
      hasReport: Boolean(report),
    });
  }
  return { available: datasets.length > 0, datasets };
}

/** Whether artefact directories exist for a given experiment id. */
export function experimentArtifacts(dirs, experimentId) {
  return {
    explainability: isDir(join(dirs.explainability, experimentId)),
    errorAnalysis: isDir(join(dirs.errorAnalysis, experimentId)),
    visualizations: isDir(join(dirs.visualizations, experimentId)),
  };
}

/** True when a metrics file exists for the run (used by health checks). */
export function runHasMetrics(experimentsDir, dataset, model, runId) {
  return isFile(join(experimentsDir, dataset, model, runId, "metrics.json"));
}

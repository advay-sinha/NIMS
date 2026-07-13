/**
 * Training · Testing · Validation — experiment explorer with filters,
 * best-run matrix, per-split metrics, validation and feature reports.
 */

import { useMemo, useState } from "react";
import { useApi } from "../api.js";
import { int, metric, ts, titleCase } from "../lib/format.js";
import {
  SectionHeader,
  StatTile,
  EmptyState,
  DataTable,
  InfoHover,
  Loader,
} from "../components/primitives.jsx";

export default function Training({ section }) {
  const state = useApi("/api/training");
  return (
    <>
      <SectionHeader section={section} />
      <Loader state={state}>
        {(data) =>
          data.available ? <TrainingBody data={data} /> : <EmptyState message={data.message} />
        }
      </Loader>
    </>
  );
}

function TrainingBody({ data }) {
  const [dataset, setDataset] = useState("all");
  const [model, setModel] = useState("all");

  const runs = useMemo(
    () =>
      data.runs.filter(
        (r) =>
          (dataset === "all" || r.dataset === dataset) &&
          (model === "all" || r.model === model),
      ),
    [data.runs, dataset, model],
  );

  return (
    <>
      <div className="grid tiles" style={{ marginBottom: 18 }}>
        <StatTile
          label="Experiments"
          value={int(data.runCount)}
          note="persisted runs, never overwritten"
          tip="Every training run keeps its own directory with metrics, manifest and config snapshot."
        />
        <StatTile
          label="Datasets"
          value={int(data.datasets.length)}
          note={data.datasets.join(" · ")}
          tip="Benchmark datasets with at least one persisted experiment."
        />
        <StatTile
          label="Model families"
          value={int(data.models.length)}
          note={data.models.join(" · ")}
          tip="Distinct model types trained across the experiment history."
        />
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <h3>
          Best run per dataset + model (test split)
          <InfoHover tip="For each dataset+model pair, the run with the highest test F1. FPR is the false-positive rate on test." />
        </h3>
        <DataTable
          columns={[
            { key: "dataset", label: "Dataset", render: (r) => titleCase(r.dataset) },
            { key: "model", label: "Model", render: (r) => titleCase(r.model) },
            { key: "testF1", label: "Test F1", num: true, render: (r) => metric(r.testF1) },
            {
              key: "testRocAuc",
              label: "ROC AUC",
              num: true,
              render: (r) => metric(r.testRocAuc),
            },
            { key: "testFpr", label: "FPR", num: true, render: (r) => metric(r.testFpr) },
            {
              key: "experimentId",
              label: "Experiment",
              render: (r) => <span className="mono">{r.experimentId}</span>,
            },
          ]}
          rows={[...data.bestRuns].sort(
            (a, b) =>
              a.dataset.localeCompare(b.dataset) || (b.testF1 ?? 0) - (a.testF1 ?? 0),
          )}
        />
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <h3>
          Experiment explorer
          <InfoHover tip="All persisted runs, newest first. Seed and device are recorded for reproducibility; metrics are reported per split, never accuracy alone." />
        </h3>
        <div className="select-row">
          <label>
            Dataset{" "}
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              <option value="all">all</option>
              {data.datasets.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          <label>
            Model{" "}
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="all">all</option>
              {data.models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <span className="muted">
            {runs.length} of {data.runCount} runs
          </span>
        </div>
        <DataTable
          columns={[
            { key: "createdAt", label: "Created", render: (r) => ts(r.createdAt) },
            { key: "dataset", label: "Dataset" },
            { key: "model", label: "Model" },
            {
              key: "val_f1",
              label: "Val F1",
              num: true,
              render: (r) => metric(r.splits.validation?.f1),
            },
            {
              key: "test_f1",
              label: "Test F1",
              num: true,
              render: (r) => metric(r.splits.test?.f1),
            },
            {
              key: "test_roc",
              label: "Test ROC AUC",
              num: true,
              render: (r) => metric(r.splits.test?.roc_auc),
            },
            {
              key: "test_fpr",
              label: "Test FPR",
              num: true,
              render: (r) => metric(r.splits.test?.false_positive_rate),
            },
            { key: "device", label: "Device" },
            { key: "seed", label: "Seed", num: true },
          ]}
          rows={runs.slice(0, 60)}
        />
        {runs.length > 60 && (
          <div className="chart-note">Showing the 60 most recent matching runs.</div>
        )}
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h3>
            Dataset validation
            <InfoHover tip="Validation artefacts produced by the reporting pipeline for each benchmark dataset." />
          </h3>
          <DataTable
            empty="No validation reports found."
            columns={[
              { key: "dataset", label: "Dataset", render: (r) => titleCase(r.dataset) },
              { key: "entries", label: "Entries", num: true },
            ]}
            rows={data.validation.reports}
          />
        </div>
        <div className="card">
          <h3>
            Feature engineering
            <InfoHover tip="Selected vs removed feature counts from the feature-engineering stage, per dataset." />
          </h3>
          <DataTable
            empty="No feature reports found."
            columns={[
              { key: "dataset", label: "Dataset", render: (r) => titleCase(r.dataset) },
              { key: "selectedCount", label: "Selected", num: true, render: (r) => int(r.selectedCount) },
              { key: "removedCount", label: "Removed", num: true, render: (r) => int(r.removedCount) },
            ]}
            rows={data.features.datasets}
          />
        </div>
      </div>

      {data.validation.reportMarkdown && (
        <div style={{ marginTop: 18 }}>
          <h2 style={{ fontSize: 15, margin: "0 0 8px" }}>Model validation report</h2>
          <pre className="report">{data.validation.reportMarkdown}</pre>
        </div>
      )}
    </>
  );
}

/**
 * History & Past Data — assessment runs, incident runs and the replayed
 * event log. Runs are immutable, so everything here stays reviewable.
 */

import { useApi } from "../api.js";
import { int, ts, titleCase } from "../lib/format.js";
import {
  SectionHeader,
  StatTile,
  SeverityBadge,
  DataTable,
  InfoHover,
  Loader,
} from "../components/primitives.jsx";

export default function History({ section }) {
  const state = useApi("/api/history");
  return (
    <>
      <SectionHeader section={section} />
      <Loader state={state}>{(data) => <HistoryBody data={data} />}</Loader>
    </>
  );
}

function HistoryBody({ data }) {
  const runColumns = [
    {
      key: "label",
      label: "Run",
      render: (r) => (
        <>
          {r.label}
          {r.isLatest && <span className="badge" style={{ color: "var(--accent)", marginLeft: 8 }}>latest</span>}
        </>
      ),
    },
    { key: "id", label: "Id", render: (r) => <span className="mono">{r.id}</span> },
    { key: "timestamp", label: "Generated", render: (r) => ts(r.timestamp) },
  ];

  return (
    <>
      <div className="grid tiles" style={{ marginBottom: 18 }}>
        <StatTile
          label="Assessment runs"
          value={int(data.assessmentRuns.length)}
          note="Engine C snapshots"
          tip="Every persisted network-configuration snapshot, newest first."
        />
        <StatTile
          label="Incident runs"
          value={int(data.incidentRuns.length)}
          note="correlation outputs"
          tip="Every persisted correlation run; open any of them in the Correlation section."
        />
        <StatTile
          label="Logged events"
          value={int(data.eventHistory.events.length)}
          note="streaming replay log"
          tip="Events persisted to the streaming event log (newest shown first)."
        />
      </div>

      <div className="grid cols-2" style={{ marginBottom: 18 }}>
        <div className="card">
          <h3>
            Assessment runs
            <InfoHover tip="Engine C snapshots under outputs/network_config — immutable directories, never overwritten." />
          </h3>
          <DataTable empty="No assessment runs." columns={runColumns} rows={data.assessmentRuns} />
        </div>
        <div className="card">
          <h3>
            Incident runs
            <InfoHover tip="Correlation runs under outputs/correlation — each keeps its incidents, signals and report." />
          </h3>
          <DataTable empty="No incident runs." columns={runColumns} rows={data.incidentRuns} />
        </div>
      </div>

      <div className="card">
        <h3>
          Event log
          <InfoHover tip="The full replayed streaming history, including events no longer visible in the live view." />
        </h3>
        <DataTable
          empty="No logged events."
          columns={[
            { key: "seq", label: "#", num: true },
            {
              key: "severity",
              label: "Severity",
              render: (r) => <SeverityBadge severity={r.severity} />,
            },
            { key: "source_engine", label: "Engine", render: (r) => titleCase(r.source_engine) },
            { key: "event_type", label: "Type", render: (r) => titleCase(r.event_type) },
            { key: "title", label: "Event" },
            { key: "emitted_at", label: "Emitted", render: (r) => ts(r.emitted_at) },
          ]}
          rows={data.eventHistory.events}
        />
      </div>
    </>
  );
}

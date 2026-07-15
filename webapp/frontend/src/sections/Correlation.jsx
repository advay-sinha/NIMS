/**
 * Correlation Engine — unified incidents combining Engine A/B/C evidence.
 */

import { useState } from "react";
import { useApi } from "../api.js";
import { int, metric, ts, titleCase, sortedEntries } from "../lib/format.js";
import {
  SectionHeader,
  StatTile,
  SeverityBadge,
  EmptyState,
  DataTable,
  InfoHover,
  Loader,
  SearchBox,
} from "../components/primitives.jsx";
import { BarChart } from "../components/charts.jsx";
import { filterByQuery } from "../lib/search.js";

export default function Correlation({ section }) {
  const [run, setRun] = useState(null);
  const state = useApi(
    run ? `/api/correlation?run=${encodeURIComponent(run)}` : "/api/correlation",
  );
  return (
    <>
      <SectionHeader section={section} />
      <Loader state={state}>
        {(data) => (
          <>
            {data.runs?.length > 0 && (
              <div className="select-row">
                <label>
                  Incident run{" "}
                  <select
                    value={run ?? data.correlationId ?? ""}
                    onChange={(e) => setRun(e.target.value)}
                  >
                    {data.runs.map((r) => (
                      <option key={r} value={r}>
                        {r}
                      </option>
                    ))}
                  </select>
                </label>
                <InfoHover tip="Each correlation run is an immutable directory under outputs/correlation. The latest is selected by default." />
              </div>
            )}
            {data.available ? (
              <CorrelationBody data={data} />
            ) : (
              <EmptyState message={data.message} />
            )}
          </>
        )}
      </Loader>
    </>
  );
}

function CorrelationBody({ data }) {
  const summary = data.summary;
  const [query, setQuery] = useState("");
  const incidents = filterByQuery(data.incidents, query, [
    "incident_id",
    "title",
    "rule_id",
    "affected_devices",
  ]);
  const signals = filterByQuery(data.signals, query, [
    "signal_id",
    "title",
    "summary",
    "device",
    "device_id",
    "engine",
    "source_engine",
  ]);
  return (
    <>
      <SearchBox
        value={query}
        onChange={setQuery}
        placeholder="Filter incidents & signals by id or device…"
        count={incidents.length + signals.length}
      />
      <div className="grid tiles" style={{ marginBottom: 18 }}>
        <StatTile
          label="Incidents"
          value={int(summary.total_incidents ?? data.incidents.length)}
          note={`${int(summary.multi_engine_incident_count)} multi-engine`}
          tip="Unified incidents produced by correlation rules over the combined signal set."
        />
        <StatTile
          label="Signals"
          value={int(summary.total_signals ?? data.signals.length)}
          note="across engines"
          tip="Per-engine inputs (cyber alerts, health anomalies, config findings) that fed the rules."
        />
        <StatTile
          label="Sources"
          value={ts(summary.timestamp)}
          note={`A: ${summary.engine_a_source ?? "—"} · B: ${summary.engine_b_source ?? "—"} · C: ${summary.engine_c_source ?? "—"}`}
          tip="The exact artefact runs each engine contributed to this correlation."
        />
      </div>

      <div className="grid cols-2" style={{ marginBottom: 18 }}>
        <div className="card">
          <h3>Signals by engine</h3>
          <BarChart entries={sortedEntries(summary.signals_by_engine)} palette="engine" />
        </div>
        <div className="card">
          <h3>Incidents by rule</h3>
          <BarChart entries={sortedEntries(summary.incidents_by_rule)} palette="single" />
        </div>
      </div>

      {incidents.map((inc) => (
        <div
          className={`card strip sev-${String(inc.severity ?? "info").toLowerCase()}`}
          key={inc.incident_id}
          style={{ marginBottom: 14 }}
        >
          <h3>
            <SeverityBadge severity={inc.severity} />
            <span
              style={{
                marginLeft: 8,
                textTransform: "none",
                fontSize: 14.5,
                fontFamily: '"IBM Plex Sans", sans-serif',
                fontWeight: 600,
                letterSpacing: 0,
                color: "var(--ink)",
              }}
            >
              {inc.title}
            </span>
          </h3>
          <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
            <span className="mono">{inc.incident_id}</span> · rule {inc.rule_id} ·
            confidence {metric(inc.confidence, 2)} · engines{" "}
            {(inc.engines ?? []).map(titleCase).join(", ")} · devices{" "}
            {(inc.affected_devices ?? []).join(", ") || "—"}
          </div>
          {inc.root_cause_hypothesis && (
            <div style={{ marginBottom: 8 }}>
              <strong>Root-cause hypothesis:</strong> {inc.root_cause_hypothesis}
            </div>
          )}
          {(inc.evidence ?? []).length > 0 && (
            <ul className="list-plain">
              {inc.evidence.map((ev, i) => (
                <li key={i}>
                  <strong>{titleCase(ev.engine)}:</strong> {ev.summary}
                </li>
              ))}
            </ul>
          )}
          {(inc.recommended_actions ?? []).length > 0 && (
            <div style={{ marginTop: 8 }}>
              <strong>Recommended:</strong>{" "}
              {inc.recommended_actions.map((a) => a.title).join(" · ")}
            </div>
          )}
        </div>
      ))}

      <div className="card">
        <h3>
          Signals
          <InfoHover tip="The raw evidence rows: one signal per engine observation, referenced by incident evidence." />
        </h3>
        <DataTable
          empty="No signals persisted."
          columns={[
            {
              key: "severity",
              label: "Severity",
              render: (r) => <SeverityBadge severity={r.severity} />,
            },
            { key: "engine", label: "Engine", render: (r) => titleCase(r.engine ?? r.source_engine) },
            { key: "title", label: "Signal", render: (r) => r.title ?? r.summary },
            { key: "device", label: "Device", render: (r) => r.device ?? r.device_id ?? "—" },
            {
              key: "signal_id",
              label: "Id",
              render: (r) => <span className="mono">{r.signal_id}</span>,
            },
          ]}
          rows={signals}
        />
      </div>
    </>
  );
}

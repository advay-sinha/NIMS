/**
 * Live Monitoring — current state of the offline streaming demo.
 */

import { useState } from "react";
import { useApi } from "../api.js";
import { int, ts, titleCase, sortedEntries } from "../lib/format.js";
import {
  SectionHeader,
  StatTile,
  SeverityBadge,
  EmptyState,
  DataTable,
  Loader,
  IncidentModal,
  SearchBox,
} from "../components/primitives.jsx";
import { BarChart } from "../components/charts.jsx";
import { filterByQuery } from "../lib/search.js";

export default function Live({ section }) {
  const state = useApi("/api/live");
  return (
    <>
      <SectionHeader section={section} />
      <Loader state={state}>
        {(data) =>
          data.available ? <LiveBody data={data} /> : <EmptyState message={data.message} />
        }
      </Loader>
    </>
  );
}

function LiveBody({ data }) {
  const [openIncident, setOpenIncident] = useState(null);
  const [query, setQuery] = useState("");
  const activeIncidents = filterByQuery(data.activeIncidents, query, [
    "incident_id",
    "title",
    "device_id",
    "severity",
  ]);
  const recentEvents = filterByQuery(data.recentEvents, query, [
    "title",
    "device_id",
    "source_engine",
    "event_type",
    "severity",
    "incident_id",
  ]);
  return (
    <>
      <SearchBox
        value={query}
        onChange={setQuery}
        placeholder="Filter incidents & events by id or device…"
        count={activeIncidents.length + recentEvents.length}
      />
      <div className="grid tiles" style={{ marginBottom: 18 }}>
        <StatTile
          label="Events replayed"
          value={int(data.totalEvents)}
          note={`last event ${ts(data.lastEventAt)}`}
          tip="Total events emitted by the offline stream replay since it started."
        />
        <StatTile
          label="Active incidents"
          value={int(data.activeIncidentCount)}
          note={`${int(data.criticalIncidentCount)} high/critical`}
          tip="Incidents currently open in the replayed stream."
        />
        <StatTile
          label="Devices involved"
          value={int(data.activeDeviceCount)}
          note="across active incidents"
          tip="Distinct devices referenced by currently active incidents."
        />
      </div>

      <div className="grid cols-2" style={{ marginBottom: 18 }}>
        <div className="card">
          <h3>Events by severity</h3>
          <BarChart
            entries={sortedEntries(data.eventsBySeverity)}
            palette="severity"
            note="Status colors are reserved for severity and always paired with a label."
          />
        </div>
        <div className="card">
          <h3>Events by engine</h3>
          <BarChart
            entries={sortedEntries(data.eventsByEngine)}
            palette="engine"
            note="Engine identity keeps a fixed color slot across the whole console."
          />
        </div>
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <h3>Active incidents</h3>
        <DataTable
          empty="No active incidents."
          onRowClick={(row) => setOpenIncident(row)}
          columns={[
            {
              key: "severity",
              label: "Severity",
              render: (r) => <SeverityBadge severity={r.severity} />,
            },
            { key: "title", label: "Incident" },
            { key: "device_id", label: "Device" },
            {
              key: "emitted_at",
              label: "Emitted",
              render: (r) => ts(r.emitted_at),
            },
            {
              key: "incident_id",
              label: "Id",
              render: (r) => <span className="mono">{r.incident_id}</span>,
            },
          ]}
          rows={activeIncidents}
        />
      </div>

      <div className="card">
        <h3>Recent events</h3>
        <DataTable
          empty="No recent events."
          columns={[
            {
              key: "severity",
              label: "Severity",
              render: (r) => <SeverityBadge severity={r.severity} />,
            },
            {
              key: "source_engine",
              label: "Engine",
              render: (r) => titleCase(r.source_engine),
            },
            {
              key: "event_type",
              label: "Type",
              render: (r) => titleCase(r.event_type),
            },
            { key: "title", label: "Event" },
            {
              key: "emitted_at",
              label: "Emitted",
              render: (r) => ts(r.emitted_at),
            },
          ]}
          rows={recentEvents}
        />
        <div className="chart-note">{data.safetyNote}</div>
      </div>

      <IncidentModal
        incident={openIncident}
        onClose={() => setOpenIncident(null)}
      />
    </>
  );
}

/**
 * Live Ingestion — Phase 9 live logging & telemetry ingestion (read-only).
 *
 * Shows the offline-first ingestion pipeline's state: overall health, per-source
 * status (Sophos Central / Firewall, Hirschmann SNMP / traps / config), recent
 * normalized events with filters, checkpoint freshness and failure/retry
 * summaries. Everything is read from persisted, sanitized artefacts through the
 * backend API — no device is contacted and nothing is executed from here.
 */

import { useEffect, useMemo, useState } from "react";
import { useApi } from "../api.js";
import { int, ts, titleCase } from "../lib/format.js";
import {
  SectionHeader,
  StatTile,
  SeverityBadge,
  EmptyState,
  DataTable,
  Loader,
  SearchBox,
  MarkdownReport,
} from "../components/primitives.jsx";
import { filterByQuery } from "../lib/search.js";

const SAFETY_TEXT =
  "Read-only live ingestion. NIMS collects and analyzes telemetry only. No " +
  "firewall or switch configuration is changed, and no remediation command is executed.";

export default function LiveIngestion({ section }) {
  const health = useApi("/api/live-ingestion/health");
  return (
    <>
      <SectionHeader section={section} />
      <div className="ingest-banner" role="note">
        <span className="ingest-banner-tag">read-only · offline</span>
        {SAFETY_TEXT}
      </div>
      <Loader state={health}>
        {(data) =>
          data.available ? (
            <IngestionBody health={data} />
          ) : (
            <EmptyState message={data.message} />
          )
        }
      </Loader>
    </>
  );
}

function IngestionBody({ health }) {
  return (
    <>
      <div className="grid tiles" style={{ marginBottom: 18 }}>
        <StatTile
          label="Ingestion health"
          value={health.healthy ? "Healthy" : "Degraded"}
          note={`mode ${health.mode} · ${health.readOnly ? "read-only" : "live"}`}
          tip="Whether the last ingestion run completed with no failed sources."
        />
        <StatTile
          label="Events ingested"
          value={int(health.totalEvents)}
          note={`last run ${ts(health.lastRunAt)}`}
          tip="Total normalized events persisted by the last ingestion run."
        />
        <StatTile
          label="Sources"
          value={`${health.okCount}/${health.sourceCount}`}
          note={`${health.failedCount} failed · ${health.disabledCount} disabled`}
          tip="Sources that completed cleanly out of all configured sources."
        />
      </div>

      <ReadinessCard />
      <SourcesCard />
      <EventsCard />

      <div className="grid cols-2" style={{ marginBottom: 18 }}>
        <CheckpointsCard />
        <FailuresCard health={health} />
      </div>

      <ReportCard />
    </>
  );
}

const READINESS_COLOR = {
  READY: "var(--status-good)",
  DISABLED: "var(--muted)",
  NOT_READY: "var(--sev-medium)",
  BLOCKED_BY_SAFETY: "var(--sev-critical)",
  MISSING_DEPENDENCY: "var(--sev-high)",
  MISSING_CONFIGURATION: "var(--sev-medium)",
  MISSING_CREDENTIALS: "var(--sev-medium)",
};

function ReadinessCard() {
  const state = useApi("/api/live-ingestion/readiness");
  return (
    <div className="card" style={{ marginBottom: 18 }}>
      <h3>Live readiness · preflight</h3>
      <Loader state={state}>
        {(data) =>
          data.available ? (
            <>
              <div className="readiness-grid">
                {data.sources.map((s) => (
                  <ReadinessTile key={s.source} s={s} />
                ))}
              </div>
              {!data.allowRunOnce && (
                <div className="chart-note">
                  Run-once via API is disabled (403). Live runs use the offline CLI after preflight
                  reports READY. Refresh with <span className="mono">python -m scripts.check_live_readiness --source all</span>.
                </div>
              )}
            </>
          ) : (
            <EmptyState message={data.message} />
          )
        }
      </Loader>
    </div>
  );
}

function ReadinessTile({ s }) {
  return (
    <div className="readiness-tile" style={{ borderLeftColor: READINESS_COLOR[s.status] ?? "var(--muted)" }}>
      <div className="readiness-head">
        <span className="readiness-name">{s.friendlyName}</span>
        <span className="badge" style={{ color: READINESS_COLOR[s.status] ?? "var(--muted)" }}>
          {s.status.replace(/_/g, " ").toLowerCase()}
        </span>
      </div>
      <div className="readiness-meta">
        mode {s.mode} · {titleCase(s.engineTarget)} ·{" "}
        {s.dependency ? `${s.dependency} ${s.dependencyOk ? "✓" : "✗"}` : "no dep"}
        {s.bindPort ? ` · udp ${s.bindPort}${s.bindPortAvailable === false ? " (in use)" : ""}` : ""}
      </div>
      {s.requiredEnvVars.length > 0 && (
        <div className="readiness-env">
          {s.requiredEnvVars.map((name) => (
            <span key={name} className={`env-chip ${s.envPresent[name] ? "set" : "unset"}`}>
              {name} {s.envPresent[name] ? "set" : "unset"}
            </span>
          ))}
        </div>
      )}
      {s.remainingSteps.length > 0 && (
        <ul className="readiness-steps">
          {s.remainingSteps.map((step, i) => (
            <li key={i}>{step}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SourcesCard() {
  const state = useApi("/api/live-ingestion/sources");
  return (
    <div className="card" style={{ marginBottom: 18 }}>
      <h3>Sources</h3>
      <Loader state={state}>
        {(data) => (
          <DataTable
            empty="No sources reported."
            columns={[
              { key: "label", label: "Source" },
              { key: "engineTarget", label: "Engine", render: (r) => titleCase(r.engineTarget) },
              {
                key: "status",
                label: "Status",
                render: (r) => <StatusBadge status={r.status} />,
              },
              { key: "mode", label: "Mode" },
              { key: "events", label: "Events", num: true },
              { key: "attempts", label: "Attempts", num: true },
              {
                key: "errorCategory",
                label: "Note",
                render: (r) => r.errorCategory ?? "—",
              },
            ]}
            rows={data.sources}
          />
        )}
      </Loader>
    </div>
  );
}

const FILTER_FIELDS = [
  ["vendor", "source_vendor"],
  ["source", "source_name"],
  ["severity", "severity"],
  ["category", "category"],
  ["engine", "engine_target"],
];

function EventsCard() {
  const state = useApi("/api/live-ingestion/events?limit=500");
  const [filters, setFilters] = useState({});
  const [device, setDevice] = useState("");
  const [open, setOpen] = useState(null);

  const events = state.data?.events ?? [];
  const options = useMemo(() => {
    const opt = {};
    for (const [name, key] of FILTER_FIELDS) {
      opt[name] = [...new Set(events.map((e) => e[key]).filter(Boolean))].sort();
    }
    return opt;
  }, [events]);

  let rows = events;
  for (const [name, key] of FILTER_FIELDS) {
    if (filters[name]) rows = rows.filter((e) => e[key] === filters[name]);
  }
  rows = filterByQuery(rows, device, ["device_id", "hostname", "device_ip"]);

  return (
    <div className="card" style={{ marginBottom: 18 }}>
      <h3>Recent events</h3>
      <Loader state={state}>
        {(data) =>
          data.available ? (
            <>
              <div className="ingest-filters">
                {FILTER_FIELDS.map(([name]) => (
                  <label key={name}>
                    {titleCase(name)}
                    <select
                      value={filters[name] ?? ""}
                      onChange={(e) =>
                        setFilters((f) => ({ ...f, [name]: e.target.value || undefined }))
                      }
                    >
                      <option value="">all</option>
                      {options[name].map((v) => (
                        <option key={v} value={v}>
                          {v}
                        </option>
                      ))}
                    </select>
                  </label>
                ))}
              </div>
              <SearchBox
                value={device}
                onChange={setDevice}
                placeholder="Filter by device id, hostname or ip…"
                count={rows.length}
              />
              <DataTable
                empty="No matching events."
                onRowClick={(row) => setOpen(row)}
                columns={[
                  { key: "timestamp", label: "Time", render: (r) => ts(r.timestamp) },
                  {
                    key: "severity",
                    label: "Severity",
                    render: (r) => <SeverityBadge severity={r.severity} />,
                  },
                  { key: "source_vendor", label: "Vendor", render: (r) => titleCase(r.source_vendor) },
                  { key: "category", label: "Category", render: (r) => titleCase(r.category) },
                  { key: "engine_target", label: "Engine", render: (r) => titleCase(r.engine_target) },
                  { key: "device_id", label: "Device", render: (r) => r.device_id ?? r.hostname ?? "—" },
                  { key: "message", label: "Message" },
                ]}
                rows={rows.slice(0, 200)}
              />
              <EventModal event={open} onClose={() => setOpen(null)} />
            </>
          ) : (
            <EmptyState message="No ingested events yet. Run: python -m scripts.run_live_logger" />
          )
        }
      </Loader>
    </div>
  );
}

function CheckpointsCard() {
  const state = useApi("/api/live-ingestion/checkpoints");
  return (
    <div className="card">
      <h3>Checkpoint freshness</h3>
      <Loader state={state}>
        {(data) => (
          <DataTable
            empty="No checkpoints yet."
            columns={[
              { key: "label", label: "Source" },
              { key: "updatedAt", label: "Updated", render: (r) => ts(r.updatedAt) },
              {
                key: "ageSeconds",
                label: "Age",
                render: (r) => (r.ageSeconds == null ? "—" : `${int(r.ageSeconds)}s`),
              },
              { key: "eventCount", label: "Events", num: true },
            ]}
            rows={data.checkpoints}
          />
        )}
      </Loader>
    </div>
  );
}

function FailuresCard({ health }) {
  return (
    <div className="card">
      <h3>Failures &amp; retries</h3>
      {health.failures?.length ? (
        <DataTable
          columns={[
            { key: "label", label: "Source" },
            { key: "status", label: "Status", render: (r) => <StatusBadge status={r.status} /> },
            { key: "category", label: "Category", render: (r) => r.category ?? "—" },
            { key: "attempts", label: "Attempts", num: true },
          ]}
          rows={health.failures}
        />
      ) : (
        <div className="muted">All sources completed cleanly — no failures or retries.</div>
      )}
    </div>
  );
}

function ReportCard() {
  const state = useApi("/api/live-ingestion/report");
  return (
    <div className="card" style={{ marginTop: 18 }}>
      <h3>Ingestion report</h3>
      <Loader state={state}>
        {(data) =>
          data.available ? (
            <MarkdownReport>{data.markdown}</MarkdownReport>
          ) : (
            <div className="muted">No ingestion report generated yet.</div>
          )
        }
      </Loader>
    </div>
  );
}

const STATUS_COLOR = {
  ok: "var(--status-good)",
  failed: "var(--sev-critical)",
  disabled: "var(--muted)",
  skipped: "var(--sev-info)",
};

function StatusBadge({ status }) {
  return (
    <span className="badge" style={{ color: STATUS_COLOR[status] ?? "var(--sev-info)" }}>
      {status}
    </span>
  );
}

/** Closable detail popup for one normalized event. */
function EventModal({ event, onClose }) {
  useEffect(() => {
    if (!event) return undefined;
    const onKey = (e) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [event, onClose]);

  if (!event) return null;
  const meta = [
    ["Event id", event.event_id],
    ["Timestamp", ts(event.timestamp)],
    ["Vendor", titleCase(event.source_vendor)],
    ["Product", event.source_product],
    ["Source type", event.source_type],
    ["Category", `${titleCase(event.category)}${event.subcategory ? ` · ${event.subcategory}` : ""}`],
    ["Engine target", titleCase(event.engine_target)],
    ["Device", event.device_id ?? event.hostname ?? "—"],
    ["Device ip", event.device_ip ?? "—"],
  ];
  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className={`modal strip sev-${String(event.severity).toLowerCase()}`}
        role="dialog"
        aria-modal="true"
        aria-label={`Event ${event.event_id}`}
        onClick={(e) => e.stopPropagation()}
      >
        <button className="modal-close" onClick={onClose} aria-label="Close">
          ×
        </button>
        <div className="modal-head">
          <SeverityBadge severity={event.severity} />
          <h2 className="modal-title" style={{ fontSize: 20, textTransform: "none" }}>
            {event.message}
          </h2>
        </div>
        <dl className="modal-meta">
          {meta.map(([k, v]) => (
            <div key={k}>
              <dt>{k}</dt>
              <dd className={k === "Event id" ? "mono" : undefined}>{v}</dd>
            </div>
          ))}
        </dl>
        {Object.keys(event.correlation_keys ?? {}).length > 0 && (
          <div className="modal-block">
            <h4>Correlation keys</h4>
            <pre className="report" style={{ maxHeight: 160 }}>
              {JSON.stringify(event.correlation_keys, null, 2)}
            </pre>
          </div>
        )}
        {Object.keys(event.normalized_fields ?? {}).length > 0 && (
          <div className="modal-block">
            <h4>Normalized fields</h4>
            <pre className="report" style={{ maxHeight: 160 }}>
              {JSON.stringify(event.normalized_fields, null, 2)}
            </pre>
          </div>
        )}
        <div className="modal-foot">Sanitized, read-only — no secrets, no execution.</div>
      </div>
    </div>
  );
}

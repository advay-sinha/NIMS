/**
 * Shared UI primitives: hover tooltips, section headers, stat tiles,
 * severity badges, empty states and tables.
 *
 * The hover layer implements the "small hovers" requirement: every section
 * feature is a chip that explains itself on hover/focus, and InfoHover adds
 * the same affordance next to individual widgets.
 */

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { metric, ts, titleCase } from "../lib/format.js";

/** Tooltip bubble shown on hover/focus of its wrapper. */
function useHover() {
  const [open, setOpen] = useState(false);
  return {
    open,
    bind: {
      onMouseEnter: () => setOpen(true),
      onMouseLeave: () => setOpen(false),
      onFocus: () => setOpen(true),
      onBlur: () => setOpen(false),
      tabIndex: 0,
    },
  };
}

/**
 * Search input for filtering incident-id / device tables in a section.
 * Controlled: pass `value` and `onChange`. Shows a live match count and a
 * clear button once a query is entered.
 */
export function SearchBox({ value, onChange, placeholder = "Filter…", count }) {
  return (
    <div className="search-box">
      <span className="search-ico" aria-hidden="true">
        ⌕
      </span>
      <input
        type="search"
        className="search-input"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        aria-label={placeholder}
      />
      {value ? (
        <button
          type="button"
          className="search-clear"
          onClick={() => onChange("")}
          aria-label="Clear filter"
        >
          ×
        </button>
      ) : null}
      {value && count != null ? (
        <span className="search-count">{count} match{count === 1 ? "" : "es"}</span>
      ) : null}
    </div>
  );
}

/** Small "?" affordance that reveals an explanation on hover. */
export function InfoHover({ title, tip }) {
  const { open, bind } = useHover();
  return (
    <span className="info-hover" role="note" aria-label={tip} {...bind}>
      ?
      {open && (
        <span className="hover-tip">
          {title && <strong>{title}</strong>}
          {tip}
        </span>
      )}
    </span>
  );
}

/** Feature chip with a hover tooltip describing how the feature works. */
export function FeatureChip({ name, tip }) {
  const { open, bind } = useHover();
  return (
    <span className="chip" role="note" aria-label={tip} {...bind}>
      {name}
      {open && (
        <span className="hover-tip">
          <strong>{name}</strong>
          {tip}
        </span>
      )}
    </span>
  );
}

/** Section header: designator eyebrow, display title, summary and chips. */
export function SectionHeader({ section }) {
  return (
    <>
      <div className="section-header">
        <div className="eyebrow">
          <b>{section.code}</b> · NetSentinel Operations Console
        </div>
        <h1>{section.title}</h1>
      </div>
      <p className="section-sub">{section.summary}</p>
      <div className="feature-chips">
        {section.features.map((f) => (
          <FeatureChip key={f.name} name={f.name} tip={f.tip} />
        ))}
      </div>
    </>
  );
}

/** Headline number card with an optional hover explanation. */
export function StatTile({ label, value, note, tip }) {
  return (
    <div className="card">
      <h3>
        {label}
        {tip && <InfoHover tip={tip} />}
      </h3>
      <div className="stat-value">{value}</div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  );
}

const SEVERITY_COLORS = {
  critical: "var(--sev-critical)",
  high: "var(--sev-high)",
  serious: "var(--sev-high)",
  medium: "var(--sev-medium)",
  warning: "var(--sev-medium)",
  low: "var(--sev-low)",
  good: "var(--sev-low)",
  info: "var(--sev-info)",
};

/** Severity badge — status color plus a visible text label (never color alone). */
export function SeverityBadge({ severity }) {
  const key = String(severity ?? "info").toLowerCase();
  const color = SEVERITY_COLORS[key] ?? "var(--sev-info)";
  return (
    <span className="badge" style={{ color }}>
      {key}
    </span>
  );
}

/** Empty-state panel that shows the exact command to produce the artefact. */
export function EmptyState({ message }) {
  const match = /Run:\s*(.+)$/s.exec(message ?? "");
  const text = match ? message.slice(0, match.index).trim() : message;
  return (
    <div className="empty">
      <div>{text || "No data available yet."}</div>
      {match && <code>{match[1].trim()}</code>}
    </div>
  );
}

/** Loading / error wrapper for API-backed sections. */
export function Loader({ state, children }) {
  if (state.loading) return <div className="muted">Loading…</div>;
  if (state.error)
    return (
      <EmptyState
        message={`Could not reach the NetSentinel API (${state.error}). Run: cd webapp/server && npm start`}
      />
    );
  return children(state.data);
}

/** Normalise an incident from either the overview (camelCase) or streaming
 * (snake_case) payloads into one shape the modal can render. */
function normalizeIncident(inc) {
  if (!inc) return null;
  const devices =
    inc.devices ??
    inc.affected_devices ??
    (inc.device_id ? [inc.device_id] : []);
  return {
    id: inc.incidentId ?? inc.incident_id ?? null,
    severity: inc.severity ?? "info",
    title: inc.title ?? "Incident",
    ruleId: inc.ruleId ?? inc.rule_id ?? null,
    devices,
    confidence: inc.confidence ?? null,
    engines: inc.engines ?? [],
    emittedAt: inc.emitted_at ?? inc.emittedAt ?? null,
    rootCause: inc.rootCauseHypothesis ?? inc.root_cause_hypothesis ?? null,
    evidence: inc.evidence ?? [],
    recommendedActions: inc.recommendedActions ?? inc.recommended_actions ?? [],
  };
}

/**
 * Closable incident-detail popup. Pass the raw incident object (either payload
 * shape) as `incident`; `onClose` dismisses it. Renders nothing when null.
 * Closes on the backdrop, the × button and the Escape key.
 */
export function IncidentModal({ incident, onClose }) {
  useEffect(() => {
    if (!incident) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [incident, onClose]);

  if (!incident) return null;
  const inc = normalizeIncident(incident);
  const meta = [
    inc.id && ["Id", inc.id],
    inc.ruleId && ["Rule", inc.ruleId],
    inc.confidence != null && ["Confidence", metric(inc.confidence, 2)],
    inc.engines.length && ["Engines", inc.engines.map(titleCase).join(", ")],
    inc.devices.length && ["Devices", inc.devices.join(", ")],
    inc.emittedAt && ["Emitted", ts(inc.emittedAt)],
  ].filter(Boolean);

  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onClick={onClose}
    >
      <div
        className={`modal strip sev-${String(inc.severity).toLowerCase()}`}
        role="dialog"
        aria-modal="true"
        aria-label={`Incident ${inc.title}`}
        onClick={(e) => e.stopPropagation()}
      >
        <button className="modal-close" onClick={onClose} aria-label="Close">
          ×
        </button>
        <div className="modal-head">
          <SeverityBadge severity={inc.severity} />
          <h2 className="modal-title">{inc.title}</h2>
        </div>

        {meta.length > 0 && (
          <dl className="modal-meta">
            {meta.map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd className={k === "Id" || k === "Rule" ? "mono" : undefined}>
                  {v}
                </dd>
              </div>
            ))}
          </dl>
        )}

        {inc.rootCause && (
          <div className="modal-block">
            <h4>Root-cause hypothesis</h4>
            <p>{inc.rootCause}</p>
          </div>
        )}

        {inc.evidence.length > 0 && (
          <div className="modal-block">
            <h4>Evidence trail</h4>
            <ul className="list-plain">
              {inc.evidence.map((ev, i) => (
                <li key={i}>
                  <strong>{titleCase(ev.engine ?? ev.source_engine ?? "")}:</strong>{" "}
                  {ev.summary ?? ev.title ?? ""}
                </li>
              ))}
            </ul>
          </div>
        )}

        {inc.recommendedActions.length > 0 && (
          <div className="modal-block">
            <h4>Recommended actions · advisory only</h4>
            <ul className="list-plain">
              {inc.recommendedActions.map((a, i) => (
                <li key={i}>
                  <strong>{a.title ?? a}</strong>
                  {a.detail ? ` — ${a.detail}` : ""}
                  {a.owner ? ` (${a.owner})` : ""}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="modal-foot">
          Nothing here is executed — details are read from persisted artefacts.
        </div>
      </div>
    </div>
  );
}

/** Rendered markdown preview (GFM tables/lists), themed to the console. */
export function MarkdownReport({ children }) {
  if (!children) return null;
  return (
    <div className="markdown-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}

/**
 * Plain data table; columns = [{key, label, num?, render?}].
 * When `onRowClick` is given, rows become clickable (pointer + keyboard).
 */
export function DataTable({ columns, rows, empty = "No rows.", onRowClick }) {
  if (!rows?.length) return <div className="muted">{empty}</div>;
  return (
    <div className="table-wrap">
      <table className={onRowClick ? "clickable" : undefined}>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.num ? "num" : undefined}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={row.id ?? i}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              tabIndex={onRowClick ? 0 : undefined}
              onKeyDown={
                onRowClick
                  ? (e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onRowClick(row);
                      }
                    }
                  : undefined
              }
            >
              {columns.map((c) => (
                <td key={c.key} className={c.num ? "num" : undefined}>
                  {c.render ? c.render(row) : (row[c.key] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

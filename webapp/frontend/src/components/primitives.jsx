/**
 * Shared UI primitives: hover tooltips, section headers, stat tiles,
 * severity badges, empty states and tables.
 *
 * The hover layer implements the "small hovers" requirement: every section
 * feature is a chip that explains itself on hover/focus, and InfoHover adds
 * the same affordance next to individual widgets.
 */

import { useState } from "react";

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

/** Plain data table; columns = [{key, label, num?, render?}]. */
export function DataTable({ columns, rows, empty = "No rows." }) {
  if (!rows?.length) return <div className="muted">{empty}</div>;
  return (
    <div className="table-wrap">
      <table>
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
            <tr key={row.id ?? i}>
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

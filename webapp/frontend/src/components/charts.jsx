/**
 * Chart components — thin marks, direct labels, hover tooltips.
 *
 * BarChart: horizontal bars with 4px rounded data-ends anchored to the
 * baseline, a visible value label on every bar (the relief rule for the
 * light palette) and a hover tooltip per mark. Identity is carried by the
 * label text, never by color alone.
 */

import { useState } from "react";
import { int, titleCase } from "../lib/format.js";

const SEVERITY_COLORS = {
  critical: "var(--sev-critical)",
  high: "var(--sev-high)",
  medium: "var(--sev-medium)",
  low: "var(--sev-low)",
  info: "var(--sev-info)",
};

const ENGINE_COLORS = {
  engine_a: "var(--engine-a)",
  engine_b: "var(--engine-b)",
  engine_c: "var(--engine-c)",
  correlation: "var(--correlation)",
  system: "var(--system)",
};

/** Color for a bar given the chart's palette role. */
function barColor(name, palette) {
  if (palette === "severity")
    return SEVERITY_COLORS[String(name).toLowerCase()] ?? "var(--sev-info)";
  if (palette === "engine")
    return ENGINE_COLORS[String(name).toLowerCase()] ?? "var(--system)";
  return "var(--accent)";
}

/**
 * Horizontal bar chart.
 *
 * entries: [[name, value], …]; palette: "severity" | "engine" | "single".
 */
export function BarChart({ entries, palette = "single", note }) {
  const [hover, setHover] = useState(null);
  if (!entries?.length) return <div className="muted">No data.</div>;
  const max = Math.max(...entries.map(([, v]) => v), 1);
  return (
    <div role="img" aria-label={entries.map(([n, v]) => `${n}: ${v}`).join(", ")}>
      {entries.map(([name, value]) => (
        <div
          className="bar-row"
          key={name}
          onMouseEnter={() => setHover(name)}
          onMouseLeave={() => setHover(null)}
          title={`${titleCase(name)}: ${int(value)}`}
        >
          <span className="bar-label">{titleCase(name)}</span>
          <span className="bar-track">
            <span
              className="bar-fill"
              style={{
                width: `${(value / max) * 100}%`,
                background: barColor(name, palette),
                outline: hover === name ? "2px solid var(--surface)" : "none",
              }}
            />
          </span>
          <span className="bar-value">{int(value)}</span>
        </div>
      ))}
      {note && <div className="chart-note">{note}</div>}
    </div>
  );
}

/**
 * Topology graph — nodes on a circle, straight edges, risk-tinted nodes.
 *
 * Every node keeps its visible text label; color encodes risk (status
 * palette) and is never the only carrier of meaning.
 */
export function TopologyGraph({ nodes, edges }) {
  const [hover, setHover] = useState(null);
  if (!nodes?.length) return <div className="muted">No topology nodes.</div>;
  const size = 420;
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 70;
  const pos = new Map(
    nodes.map((node, i) => {
      const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
      return [node.id, { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) }];
    }),
  );
  const riskColor = (score) =>
    score >= 70
      ? "var(--sev-critical)"
      : score >= 40
        ? "var(--sev-high)"
        : score > 0
          ? "var(--sev-medium)"
          : "var(--engine-b)";

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      style={{ width: "100%", maxWidth: 520, display: "block", margin: "0 auto" }}
      role="img"
      aria-label="Network topology graph"
    >
      {(edges ?? []).map((edge, i) => {
        const a = pos.get(edge.source ?? edge.from);
        const b = pos.get(edge.target ?? edge.to);
        if (!a || !b) return null;
        return (
          <line
            key={i}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            stroke="var(--baseline)"
            strokeWidth="1.5"
          />
        );
      })}
      {nodes.map((node) => {
        const p = pos.get(node.id);
        const active = hover === node.id;
        return (
          <g
            key={node.id}
            onMouseEnter={() => setHover(node.id)}
            onMouseLeave={() => setHover(null)}
          >
            <circle
              cx={p.x}
              cy={p.y}
              r={active ? 13 : 10}
              fill={riskColor(node.risk_score ?? 0)}
              stroke="var(--surface)"
              strokeWidth="2"
            />
            <text
              x={p.x}
              y={p.y + (p.y > cy ? 28 : -18)}
              textAnchor="middle"
              fontSize="11"
              fill="var(--ink-2)"
            >
              {node.label ?? node.id}
            </text>
            {active && (
              <text
                x={p.x}
                y={p.y + (p.y > cy ? 42 : -32)}
                textAnchor="middle"
                fontSize="10.5"
                fill="var(--muted)"
              >
                risk {node.risk_score ?? 0} · {node.finding_count ?? 0} finding(s)
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/**
 * Landing page — executive overview plus the console directory.
 *
 * Fulfils the "general summary on start of page" requirement: a status
 * banner, cross-engine headline cards, the assessor roll-up and a directory
 * card per section describing what it contains.
 */

import { useState } from "react";
import { useApi } from "../api.js";
import { SECTIONS } from "../lib/sections.js";
import { int } from "../lib/format.js";
import {
  SectionHeader,
  StatTile,
  SeverityBadge,
  Loader,
  IncidentModal,
} from "../components/primitives.jsx";

const STATUS_STRIP = {
  attention: "sev-critical",
  monitor: "sev-medium",
  stable: "sev-good",
};

const STATUS_WORD = {
  attention: "Attention required",
  monitor: "Monitoring",
  stable: "Stable",
};

export default function Overview({ section, onNavigate }) {
  const state = useApi("/api/overview");
  return (
    <>
      <SectionHeader section={section} />
      <Loader state={state}>
        {(data) => <OverviewBody data={data} onNavigate={onNavigate} />}
      </Loader>
    </>
  );
}

function OverviewBody({ data, onNavigate }) {
  const { cards, executive } = data;
  const [openIncident, setOpenIncident] = useState(null);
  return (
    <>
      <div
        className={`status-banner strip ${STATUS_STRIP[executive.networkStatusLevel] ?? "sev-info"}`}
      >
        <div>
          <div className="status-word">
            {STATUS_WORD[executive.networkStatusLevel] ?? "Status"}
          </div>
          <div className="status-detail">
            {executive.networkStatus.split("— ")[1] ?? executive.networkStatus}
          </div>
          <div className="status-meta">
            RUN {data.snapshotId ?? "—"} · COR {data.correlationId ?? "—"} ·
            READ-ONLY · ARTEFACT-DRIVEN
          </div>
        </div>
      </div>

      <div className="grid tiles" style={{ marginBottom: 18 }}>
        <StatTile
          label="Incidents"
          value={int(cards.totalIncidents)}
          note={`${int(cards.highCriticalIncidents)} high/critical`}
          tip="Unified incidents produced by the correlation engine for the selected run."
        />
        <StatTile
          label="Config findings"
          value={int(cards.engineCFindings)}
          note="Engine C rule engine"
          tip="Configuration problems detected by Engine C's YAML-driven rules in the selected snapshot."
        />
        <StatTile
          label="Remediation actions"
          value={int(cards.remediationActionsPlanned)}
          note={`${int(cards.dryRunExecutedCount)} executed (dry-run keeps this at 0)`}
          tip="Planned (never auto-executed) remediation actions with commands, rollback and verification."
        />
        <StatTile
          label="Production models"
          value={int(cards.engineAProductionModels)}
          note={cards.engineBAnomalyStatus}
          tip="Engine A intrusion-detection models currently promoted to production, one per dataset."
        />
      </div>

      <div className="grid cols-2" style={{ marginBottom: 18 }}>
        <div className="card">
          <h3>Critical incidents</h3>
          {executive.criticalIncidents.length ? (
            <ul className="list-plain incident-list">
              {executive.criticalIncidents.map((inc) => (
                <li key={inc.incidentId}>
                  <button
                    type="button"
                    className="incident-open"
                    onClick={() => setOpenIncident(inc)}
                  >
                    <SeverityBadge severity={inc.severity} /> {inc.title}{" "}
                    <span className="muted mono">
                      {inc.ruleId} · {inc.devices.join(", ")}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div className="muted">None in the selected runs.</div>
          )}
        </div>
        <div className="card">
          <h3>Likely root causes</h3>
          {executive.likelyRootCauses.length ? (
            <ul className="list-plain">
              {executive.likelyRootCauses.map((cause) => (
                <li key={cause}>{cause}</li>
              ))}
            </ul>
          ) : (
            <div className="muted">No root-cause hypotheses recorded.</div>
          )}
        </div>
      </div>

      {executive.recommendedActions.length > 0 && (
        <div className="card" style={{ marginBottom: 18 }}>
          <h3>Recommended actions (advisory — nothing is executed)</h3>
          <ul className="list-plain">
            {executive.recommendedActions.map((action) => (
              <li key={action.title}>
                <strong>{action.title}</strong>
                {action.detail ? ` — ${action.detail}` : ""}{" "}
                <span className="muted">({action.owner})</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <h2 style={{ fontSize: 16, margin: "24px 0 10px" }}>Console sections</h2>
      <div className="grid cols-2 overview-directory">
        {SECTIONS.filter((s) => s.id !== "overview").map((s) => (
          <div className="card" key={s.id}>
            <h3>
              <span
                className="nav-dot"
                style={{ background: s.color, marginRight: 8 }}
              />
              <a onClick={() => onNavigate(s.id)}>{s.title}</a>
            </h3>
            <div style={{ color: "var(--ink-2)", fontSize: 13 }}>{s.summary}</div>
          </div>
        ))}
      </div>

      <IncidentModal
        incident={openIncident}
        onClose={() => setOpenIncident(null)}
      />
    </>
  );
}

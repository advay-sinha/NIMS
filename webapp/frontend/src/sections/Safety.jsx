/**
 * Safety Posture — the guarantees the console operates under, plus the
 * latest dry-run audit evidence.
 */

import { useApi } from "../api.js";
import { int } from "../lib/format.js";
import {
  SectionHeader,
  StatTile,
  InfoHover,
  Loader,
} from "../components/primitives.jsx";

const GUARANTEE_LABELS = {
  readOnly: {
    label: "Read-only console",
    tip: "The web console and API only read persisted artefacts; no route mutates anything.",
  },
  offlineArtifactsOnly: {
    label: "Offline artefacts only",
    tip: "Every panel is rendered from files already on disk — nothing is recomputed or fetched live.",
  },
  noLiveDeviceAccess: {
    label: "No live device access",
    tip: "No SNMP polling, SSH sessions or vendor API calls are made from this application.",
  },
  noPacketCapture: {
    label: "No packet capture",
    tip: "The platform analyses saved datasets and configuration snapshots, never live traffic.",
  },
  noCommandExecution: {
    label: "No command execution",
    tip: "Remediation plans are dry-run documents; no command is ever sent to a device from here.",
  },
  confirmationRequiredForActions: {
    label: "Explicit confirmation required",
    tip: "Destructive actions (interface shutdown, PoE or VLAN changes) require explicit human confirmation in the separate CLI workflow, with rollback and verification commands.",
  },
};

export default function Safety({ section }) {
  const state = useApi("/api/safety");
  return (
    <>
      <SectionHeader section={section} />
      <Loader state={state}>{(data) => <SafetyBody data={data} />}</Loader>
    </>
  );
}

function SafetyBody({ data }) {
  return (
    <>
      <div
        className={`status-banner strip ${data.allowActions ? "sev-critical" : "sev-good"}`}
      >
        <div>
          <div className="status-word">
            Actions {data.allowActions ? "enabled" : "disabled"}
          </div>
          <div className="status-detail">
            This console is a viewer, never an actuator. Detection is separated from
            action execution; nothing on these pages can change device state.
          </div>
        </div>
      </div>

      <div className="grid tiles" style={{ marginBottom: 18 }}>
        {Object.entries(GUARANTEE_LABELS).map(([key, meta]) => (
          <div className="card" key={key}>
            <h3>
              {meta.label}
              <InfoHover tip={meta.tip} />
            </h3>
            <div
              className="stat-value"
              style={{
                fontSize: 20,
                color: data.guarantees[key] ? "var(--status-good)" : "var(--sev-critical)",
              }}
            >
              {data.guarantees[key] ? "guaranteed" : "off"}
            </div>
          </div>
        ))}
      </div>

      <div className="grid tiles" style={{ marginBottom: 18 }}>
        <StatTile
          label="Commands executed"
          value={int(data.dryRun.executedCount)}
          note="always 0 in dry-run"
          tip="Count of commands actually executed for the selected snapshot — the dry-run pipeline keeps this at zero."
        />
        <StatTile
          label="Audit entries"
          value={int(data.dryRun.auditEntryCount)}
          note={`snapshot ${data.dryRun.snapshotId ?? "—"}`}
          tip="Every validated (dry-run) action writes an audit-log entry with operator, timestamp and outcome."
        />
      </div>

      {data.safetyNotes.length > 0 && (
        <div className="card">
          <h3>Safety notes from artefacts</h3>
          <ul className="list-plain">
            {data.safetyNotes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      )}
    </>
  );
}

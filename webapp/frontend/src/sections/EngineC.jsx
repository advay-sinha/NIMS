/**
 * Engine C — network configuration intelligence for one snapshot:
 * summary tiles, findings, inventory, topology graph, device health and
 * the dry-run remediation plan.
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
  InfoHover,
  Loader,
  SearchBox,
} from "../components/primitives.jsx";
import { BarChart, TopologyGraph } from "../components/charts.jsx";
import { filterByQuery, rowMatches } from "../lib/search.js";

export default function EngineC({ section }) {
  const [snapshot, setSnapshot] = useState(null);
  const state = useApi(
    snapshot ? `/api/engine-c?snapshot=${encodeURIComponent(snapshot)}` : "/api/engine-c",
  );
  return (
    <>
      <SectionHeader section={section} />
      <Loader state={state}>
        {(data) => (
          <>
            {data.snapshots?.length > 0 && (
              <div className="select-row">
                <label>
                  Assessment run{" "}
                  <select
                    value={snapshot ?? data.snapshotId ?? ""}
                    onChange={(e) => setSnapshot(e.target.value)}
                  >
                    {data.snapshots.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </label>
                <InfoHover tip="Each assessment run is an immutable snapshot directory under outputs/network_config. The latest is selected by default." />
              </div>
            )}
            {data.available ? (
              <EngineCBody data={data} />
            ) : (
              <EmptyState message={data.message} />
            )}
          </>
        )}
      </Loader>
    </>
  );
}

function EngineCBody({ data }) {
  const views = data.views;
  const summary = views.dashboard_summary ?? {};
  const allFindings = views.findings_view?.findings ?? [];
  const topology = views.topology_view ?? {};
  const allActions = views.remediation_view?.actions ?? [];
  const allCards = views.device_health_cards?.cards ?? [];
  const inventory = views.inventory_view ?? {};

  const [query, setQuery] = useState("");
  const findings = filterByQuery(allFindings, query, [
    "finding_id",
    "title",
    "device",
    "interface",
    "category",
  ]);
  const actions = filterByQuery(allActions, query, [
    "action_id",
    "title",
    "device",
    "interface",
    "action_type",
  ]);
  const cards = filterByQuery(allCards, query, ["hostname"]);
  const interfacesByDevice = Object.entries(
    inventory.interfaces_by_device ?? {},
  ).filter(
    ([device, ifaces]) =>
      !query.trim() ||
      device.toLowerCase().includes(query.trim().toLowerCase()) ||
      (ifaces ?? []).some((i) => rowMatches(i, query, ["name", "description", "vlan"])),
  );
  const matchCount = findings.length + cards.length + actions.length;

  return (
    <>
      <SearchBox
        value={query}
        onChange={setQuery}
        placeholder="Filter findings, devices & actions by device or id…"
        count={matchCount}
      />
      <div className="grid tiles" style={{ marginBottom: 18 }}>
        <StatTile
          label="Devices"
          value={int(summary.device_count)}
          note={`${int(summary.interface_count)} interfaces · ${int(summary.vlan_count)} VLANs`}
          tip="Inventory parsed from saved device configurations and show-command output."
        />
        <StatTile
          label="Findings"
          value={int(summary.finding_count)}
          note="rule-engine detections"
          tip="Configuration problems detected by YAML-driven rules — severity and evidence below."
        />
        <StatTile
          label="Remediation actions"
          value={int(summary.remediation_action_count)}
          note={`${int(summary.command_action_count)} command · ${int(summary.investigation_action_count)} investigate`}
          tip="Planned actions with commands, rollback and verification. Dry-run only."
        />
        <StatTile
          label="Topology links"
          value={int(summary.topology_edge_count)}
          note="LLDP/CDP neighbor edges"
          tip="Device-to-device links discovered from saved LLDP/CDP neighbor tables."
        />
      </div>

      <div className="grid cols-2" style={{ marginBottom: 18 }}>
        <div className="card">
          <h3>Findings by severity</h3>
          <BarChart entries={sortedEntries(summary.findings_by_severity)} palette="severity" />
        </div>
        <div className="card">
          <h3>Findings by category</h3>
          <BarChart entries={sortedEntries(summary.findings_by_category)} palette="single" />
        </div>
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <h3>
          Findings
          <InfoHover tip="Every finding carries its rule id, evidence and a recommendation, so it is explainable back to source artefacts." />
        </h3>
        <DataTable
          empty="No findings in this snapshot."
          columns={[
            {
              key: "severity",
              label: "Severity",
              render: (r) => <SeverityBadge severity={r.severity} />,
            },
            { key: "title", label: "Finding" },
            { key: "category", label: "Category", render: (r) => titleCase(r.category) },
            { key: "device", label: "Device" },
            { key: "interface", label: "Interface" },
            { key: "evidence", label: "Evidence" },
            { key: "recommendation", label: "Recommendation" },
            { key: "confidence", label: "Confidence" },
          ]}
          rows={findings}
        />
      </div>

      <div className="grid cols-2" style={{ marginBottom: 18 }}>
        <div className="card">
          <h3>
            Topology
            <InfoHover tip="Node color encodes risk (status palette); hover a node for its risk score and finding count. Labels always stay visible." />
          </h3>
          <TopologyGraph nodes={topology.nodes} edges={topology.edges ?? topology.links} />
        </div>
        <div className="card">
          <h3>
            Device health
            <InfoHover tip="Per-device roll-up computed at export time from inventory, STP state, PoE and findings." />
          </h3>
          <DataTable
            empty="No device health cards."
            columns={[
              { key: "hostname", label: "Device" },
              {
                key: "highest_severity",
                label: "Worst",
                render: (r) => <SeverityBadge severity={r.highest_severity ?? "info"} />,
              },
              { key: "finding_count", label: "Findings", num: true },
              { key: "interface_count", label: "Ifaces", num: true },
              { key: "poe_port_count", label: "PoE", num: true },
              { key: "stp_blocked_count", label: "STP blocked", num: true },
              { key: "topology_neighbor_count", label: "Neighbors", num: true },
            ]}
            rows={cards}
          />
        </div>
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <h3>
          Interface inventory
          <InfoHover tip="Per-interface admin/oper status, mode, VLAN, speed/duplex and PoE state parsed from saved show-command output." />
        </h3>
        {interfacesByDevice.map(([device, ifaces]) => (
          <div key={device} style={{ marginBottom: 10 }}>
            <div style={{ fontWeight: 600, margin: "6px 0" }}>{device}</div>
            <DataTable
              columns={[
                { key: "name", label: "Interface" },
                { key: "status", label: "Status" },
                { key: "mode", label: "Mode" },
                { key: "vlan", label: "VLAN" },
                { key: "speed", label: "Speed" },
                { key: "duplex", label: "Duplex" },
                {
                  key: "poe_enabled",
                  label: "PoE",
                  render: (r) => (r.poe_enabled ? (r.poe_state ?? "on") : "off"),
                },
                { key: "description", label: "Description" },
              ]}
              rows={ifaces}
            />
          </div>
        ))}
      </div>

      <div className="card">
        <h3>
          Remediation plan (dry-run — requires explicit confirmation)
          <InfoHover tip="Each action lists exact commands, rollback and verification steps. Nothing here executes; the plan is applied only through the confirmed CLI workflow." />
        </h3>
        <DataTable
          empty="No remediation actions planned."
          columns={[
            {
              key: "severity",
              label: "Severity",
              render: (r) => <SeverityBadge severity={r.severity} />,
            },
            { key: "title", label: "Action" },
            { key: "action_type", label: "Type", render: (r) => titleCase(r.action_type) },
            { key: "device", label: "Device" },
            { key: "interface", label: "Interface" },
            {
              key: "commands",
              label: "Commands",
              render: (r) =>
                r.commands?.length ? (
                  <span className="mono">{r.commands.join(" ; ")}</span>
                ) : (
                  "investigation"
                ),
            },
            {
              key: "rollback",
              label: "Rollback",
              render: (r) =>
                r.rollback ? <span className="mono">{String(r.rollback)}</span> : "—",
            },
          ]}
          rows={actions}
        />
        <div className="chart-note">
          {summary.safety_note} Generated {ts(summary.generated_at)}.
        </div>
      </div>
    </>
  );
}

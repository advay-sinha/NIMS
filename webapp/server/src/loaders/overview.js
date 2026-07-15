/**
 * Overview loader — cross-engine executive summary for the landing page.
 *
 * Pure aggregation of already-loaded artefact sets (ports
 * compute_overview / build_executive_summary from src/dashboard/loader.py).
 * No recomputation, no IO.
 */

const HIGH = new Set(["high", "critical"]);

/** Top-level overview cards from the loaded artefact sets. */
export function computeOverview(engineC, correlation, engineA, engineB) {
  const incidents = correlation.incidents ?? [];
  const summary = engineC.views?.dashboard_summary ?? {};
  return {
    totalIncidents: incidents.length,
    highCriticalIncidents: incidents.filter((i) =>
      HIGH.has(String(i.severity ?? "").toLowerCase()),
    ).length,
    engineCFindings: Number(summary.finding_count ?? 0),
    remediationActionsPlanned: Number(summary.remediation_action_count ?? 0),
    dryRunExecutedCount: Number(engineC.dryRunExecutedCount ?? 0),
    engineBAnomalyStatus: engineB.anomalyStatus ?? "unavailable",
    engineAProductionModels: Number(engineA.productionModelCount ?? 0),
  };
}

/** Assessor-focused roll-up: status, critical incidents, causes, actions. */
export function buildExecutiveSummary(engineC, correlation, engineB) {
  const incidents = correlation.incidents ?? [];
  const critical = incidents.filter((i) =>
    HIGH.has(String(i.severity ?? "").toLowerCase()),
  );
  const focus = critical.length ? critical : incidents;
  const summary = engineC.views?.dashboard_summary ?? {};
  const findings = Number(summary.finding_count ?? 0);

  const [statusLevel, statusText] = networkStatus(critical, findings, engineB);

  const devices = [];
  for (const inc of focus) {
    for (const dev of inc.affected_devices ?? []) {
      if (!devices.includes(dev)) devices.push(dev);
    }
  }
  for (const dev of summary.top_risk_devices ?? []) {
    const name = dev && typeof dev === "object" ? dev.device : null;
    if (name && !devices.includes(name)) devices.push(name);
  }

  const causes = [];
  for (const inc of focus) {
    const hyp = inc.root_cause_hypothesis;
    if (hyp && !causes.includes(hyp)) causes.push(hyp);
  }

  const actions = [];
  const seen = new Set();
  for (const inc of focus) {
    for (const action of inc.recommended_actions ?? []) {
      const title = action.title ?? "";
      if (title && !seen.has(title)) {
        seen.add(title);
        actions.push({
          title,
          detail: action.detail ?? "",
          owner: action.owner ?? "network",
        });
      }
    }
  }

  return {
    networkStatus: statusText,
    networkStatusLevel: statusLevel,
    totalIncidents: incidents.length,
    criticalIncidentCount: critical.length,
    criticalIncidents: critical.map((i) => ({
      incidentId: i.incident_id,
      severity: i.severity,
      title: i.title,
      ruleId: i.rule_id,
      devices: i.affected_devices ?? [],
      confidence: i.confidence ?? null,
      engines: i.engines ?? [],
      rootCauseHypothesis: i.root_cause_hypothesis ?? null,
      evidence: i.evidence ?? [],
      recommendedActions: i.recommended_actions ?? [],
    })),
    affectedDevices: devices.slice(0, 12),
    likelyRootCauses: causes.slice(0, 5),
    recommendedActions: actions.slice(0, 6),
    safetyStatus: {
      offline: true,
      noCommandExecution: true,
      noLiveDeviceAccess: true,
      dryRunExecuted: Number(engineC.dryRunExecutedCount ?? 0),
    },
  };
}

function networkStatus(critical, findings, engineB) {
  if (critical.length) {
    return [
      "attention",
      `Attention required — ${critical.length} high/critical incident(s) ` +
        "correlated across engines.",
    ];
  }
  const anomaly = engineB.anomalyStatus;
  const anomalous =
    Boolean(engineB.available) && Boolean(anomaly) && !String(anomaly).includes("0.0%");
  if (findings || anomalous) {
    return [
      "monitor",
      "Monitoring — configuration findings or network-health anomalies are " +
        "present, but no critical incidents.",
    ];
  }
  return ["stable", "Stable — no critical incidents detected in the selected runs."];
}

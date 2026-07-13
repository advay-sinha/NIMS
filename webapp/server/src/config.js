/**
 * Configuration loading for the NetSentinel API server.
 *
 * All filesystem locations come from configs/paths.yaml and all webapp
 * behaviour from configs/webapp.yaml — nothing is hardcoded. The repository
 * root is resolved relative to this file (webapp/server/src -> repo root)
 * and can be overridden with the NETSENTINEL_ROOT environment variable.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import YAML from "yaml";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_ROOT = path.resolve(HERE, "..", "..", "..");

/** Resolve the repository root (env override for tests/deployments). */
export function repoRoot() {
  return process.env.NETSENTINEL_ROOT
    ? path.resolve(process.env.NETSENTINEL_ROOT)
    : DEFAULT_ROOT;
}

function readYaml(filePath) {
  try {
    return YAML.parse(fs.readFileSync(filePath, "utf-8")) ?? {};
  } catch {
    return {};
  }
}

/**
 * Load the full server configuration.
 *
 * Returns { root, server, sections, safety, dirs } where every entry in
 * `dirs` is an absolute path derived from configs/paths.yaml.
 */
export function loadConfig(root = repoRoot()) {
  const pathsCfg = readYaml(path.join(root, "configs", "paths.yaml"));
  const webCfg = readYaml(path.join(root, "configs", "webapp.yaml"));
  const p = pathsCfg.paths ?? {};

  const rel = (key, fallback) => path.join(root, p[key] ?? fallback);

  return {
    root,
    server: {
      host: webCfg.server?.host ?? "127.0.0.1",
      port: Number(webCfg.server?.port ?? 8050),
      cacheSeconds: Number(webCfg.server?.cache_seconds ?? 5),
    },
    sections: webCfg.sections ?? {},
    safety: {
      showNoExecutionBanner: webCfg.safety?.show_no_execution_banner ?? true,
      allowActions: webCfg.safety?.allow_actions ?? false,
    },
    dirs: {
      outputs: rel("outputs_dir", "outputs"),
      experiments: rel("experiments_dir", "outputs/experiments"),
      registry: rel("registry_dir", "outputs/registry"),
      reports: rel("reports_dir", "outputs/reports"),
      dataReports: rel("data_reports_dir", "outputs/data_reports"),
      features: rel("features_out_dir", "outputs/features"),
      errorAnalysis: rel("error_analysis_dir", "outputs/error_analysis"),
      explainability: rel("explainability_dir", "outputs/explainability"),
      visualizations: rel("visualizations_dir", "outputs/visualizations"),
      optimization: rel("optimization_dir", "outputs/optimization"),
      networkHealth: rel("network_health_dir", "outputs/network_health"),
      networkConfig: rel("network_config_dir", "outputs/network_config"),
      correlation: rel("correlation_dir", "outputs/correlation"),
      streaming: path.join(rel("outputs_dir", "outputs"), "streaming"),
    },
  };
}

/**
 * Application shell — sidebar navigation, theme toggle and section routing.
 *
 * Sections come from the catalogue in lib/sections.js and are filtered by
 * the server's /api/meta section toggles (configs/webapp.yaml).
 */

import { useEffect, useState } from "react";
import { useApi } from "./api.js";
import { SECTIONS, sectionById } from "./lib/sections.js";
import Overview from "./sections/Overview.jsx";
import Live from "./sections/Live.jsx";
import { EngineA, EngineB } from "./sections/Engines.jsx";
import EngineC from "./sections/EngineC.jsx";
import Correlation from "./sections/Correlation.jsx";
import LiveIngestion from "./sections/LiveIngestion.jsx";
import History from "./sections/History.jsx";
import Safety from "./sections/Safety.jsx";
import Training from "./sections/Training.jsx";

const VIEWS = {
  overview: Overview,
  live: Live,
  engine_a: EngineA,
  engine_b: EngineB,
  engine_c: EngineC,
  correlation: Correlation,
  live_ingestion: LiveIngestion,
  history: History,
  training: Training,
  safety: Safety,
};

function initialTheme() {
  return window.matchMedia?.("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
}

/* The "system voice" ticker — statements that hold regardless of the data on
 * screen, reinforcing the read-only/offline guarantees and the engine roster.
 * Duplicated once in the markup so the -50% keyframe loops seamlessly. */
const TICKER_LINES = [
  "Everything on screen is read from disk — nothing is executed, ever.",
  "Engine A serves intrusion-detection models across NSL-KDD, UNSW-NB15 and CICIDS2017.",
  "Engine B watches SNMP telemetry for degradation; Engine C reads saved network snapshots.",
  "Correlation groups cyber, health and configuration evidence into unified incidents.",
  "Remediation is dry-run only — plans ship with rollback and verification, never auto-applied.",
];

/** Short signage word drawn behind each section (first token of its label). */
function watermarkFor(section) {
  return (section.label ?? "").split(" ")[0].toUpperCase();
}

function Ticker() {
  const run = [...TICKER_LINES, ...TICKER_LINES];
  return (
    <div className="ticker">
      <div className="ticker-tag">
        <i />
        System voice
      </div>
      <div className="ticker-viewport">
        <div className="ticker-track">
          {run.map((line, i) => (
            <span key={i}>
              {line}
              <span className="dot">·</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [active, setActive] = useState(
    () => window.location.hash.replace("#", "") || "overview",
  );
  const [theme, setTheme] = useState(initialTheme);
  const meta = useApi("/api/meta");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    window.location.hash = active;
  }, [active]);

  const enabled = meta.data?.sections ?? {};
  const sections = SECTIONS.filter((s) => enabled[s.id] !== false);
  const section = sectionById(active);
  const View = VIEWS[section.id] ?? Overview;

  return (
    <div className="shell">
      <div className="grid-bg" aria-hidden="true" />
      <Ticker />
      <div className="app">
        <aside className="sidebar">
        <div className="brand">
          NetSentinel
          <small>Network Operations Console</small>
        </div>
        {sections.map((s) => (
          <button
            key={s.id}
            className={`nav-button ${s.id === section.id ? "active" : ""}`}
            onClick={() => setActive(s.id)}
          >
            <span className="nav-code">{s.code}</span>
            <span className="nav-dot" style={{ background: s.color }} />
            {s.label}
          </button>
        ))}
        <div className="sidebar-footer">
          <span className="readonly-pill">read-only · offline</span>
          <button
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? "Light theme" : "Dark theme"}
          </button>
        </div>
        </aside>
        <main className="main">
          <div className="watermark" aria-hidden="true">
            {watermarkFor(section)}
          </div>
          <div className="main-inner">
            <View section={section} onNavigate={setActive} />
          </div>
        </main>
      </div>
    </div>
  );
}

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
import Training from "./sections/Training.jsx";
import { EngineA, EngineB } from "./sections/Engines.jsx";
import EngineC from "./sections/EngineC.jsx";
import Correlation from "./sections/Correlation.jsx";
import History from "./sections/History.jsx";
import Safety from "./sections/Safety.jsx";

const VIEWS = {
  overview: Overview,
  live: Live,
  training: Training,
  engine_a: EngineA,
  engine_b: EngineB,
  engine_c: EngineC,
  correlation: Correlation,
  history: History,
  safety: Safety,
};

function initialTheme() {
  return window.matchMedia?.("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
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
        <View section={section} onNavigate={setActive} />
      </main>
    </div>
  );
}

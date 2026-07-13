/**
 * NetSentinel API server entry point.
 *
 * Serves the read-only artefact API under /api and, when a production
 * frontend build exists (webapp/frontend/dist), the static frontend too.
 * Usage: npm start (from webapp/server).
 */

import path from "node:path";
import fs from "node:fs";
import express from "express";
import { loadConfig } from "./config.js";
import { buildRouter } from "./routes.js";

const config = loadConfig();
const app = express();
app.disable("x-powered-by");

// Minimal structured request log (no print-style noise, one line per request).
app.use((req, res, next) => {
  const started = Date.now();
  res.on("finish", () => {
    process.stderr.write(
      JSON.stringify({
        level: "info",
        at: new Date().toISOString(),
        method: req.method,
        path: req.path,
        status: res.statusCode,
        ms: Date.now() - started,
      }) + "\n",
    );
  });
  next();
});

app.use("/api", buildRouter(config));

// Serve the built frontend when present (production mode).
const distDir = path.join(config.root, "webapp", "frontend", "dist");
if (fs.existsSync(distDir)) {
  app.use(express.static(distDir));
  app.get("*", (_req, res) => res.sendFile(path.join(distDir, "index.html")));
}

app.use((err, _req, res, _next) => {
  process.stderr.write(
    JSON.stringify({
      level: "error",
      at: new Date().toISOString(),
      message: String(err?.message ?? err),
    }) + "\n",
  );
  res.status(500).json({ error: "internal error" });
});

const { host, port } = config.server;
app.listen(port, host, () => {
  process.stderr.write(
    JSON.stringify({
      level: "info",
      at: new Date().toISOString(),
      message: `NetSentinel API listening on http://${host}:${port}`,
      root: config.root,
    }) + "\n",
  );
});

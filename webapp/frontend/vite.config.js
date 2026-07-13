import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api to the Node artefact server (webapp/server).
// Ports mirror configs/webapp.yaml (frontend.dev_port / server.port).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,
    proxy: {
      "/api": { target: "http://127.0.0.1:8050", changeOrigin: true },
    },
  },
});

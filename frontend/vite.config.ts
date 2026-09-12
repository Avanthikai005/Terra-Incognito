import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Single-page demo frontend. `npm run dev` serves public/ at the root, so the
// app can fetch /data/results.json and /assets/*.png with plain relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: true,
  },
});
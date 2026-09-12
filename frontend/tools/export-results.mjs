#!/usr/bin/env node
/**
 * Export real training results for the frontend.
 *
 *   npm run data        (run from frontend/)
 *
 * Reads ../results/{naive,joint,cl}_matrix.json (written by src/train.py) and
 * emits frontend/public/data/results.json in the schema the app consumes.
 * Also copies ../plots/domain_distance.png over the placeholder so the Domain
 * Distance page shows the real ablation.
 *
 * If the pipeline has not been run yet, this exits quietly and the app keeps
 * the bundled demo numbers (see src/data/results.ts).
 */
import { readFileSync, writeFileSync, copyFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const RESULTS_DIR = join(ROOT, "..", "results");
const PLOTS_DIR = join(ROOT, "..", "plots");
const OUT = join(ROOT, "public", "data", "results.json");

const METHODS = ["naive", "joint", "cl"];
const out = { matrices: {}, labels: null };
let wrote = false;

for (const mode of METHODS) {
  const path = join(RESULTS_DIR, `${mode}_matrix.json`);
  if (!existsSync(path)) {
    console.warn(`[data] missing ${path} — skipping (demo fallback stays active)`);
    continue;
  }
  const data = JSON.parse(readFileSync(path, "utf8"));
  const rows = (data.matrix ?? []).map((row) =>
    row.map((v) =>
      v === null || v === undefined || !Number.isFinite(v) ? null : Math.round(v * 100)
    )
  );
  out.matrices[mode] = rows;
  if (out.labels === null) out.labels = data.regions ?? [];
  wrote = true;
}

if (wrote) {
  writeFileSync(OUT, JSON.stringify(out, null, 2) + "\n");
  console.log(`[data] wrote ${OUT}`);

  const realPlot = join(PLOTS_DIR, "domain_distance.png");
  if (existsSync(realPlot)) {
    copyFileSync(realPlot, join(ROOT, "public", "assets", "domain_distance.png"));
    console.log("[data] copied real plots/domain_distance.png -> public/assets/");
  } else {
    console.warn("[data] plots/domain_distance.png not found; keeping placeholder");
  }
} else {
  console.log("[data] no pipeline results found; keeping bundled demo data");
}
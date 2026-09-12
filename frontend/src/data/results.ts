// =============================================================================
// Precomputed results + derived metrics for the simulator.
//
// REAL DATA (how to plug in the training pipeline's output):
//   1. Run the pipeline in the repo root:            bash run.sh
//   2. Export results for the UI:                    npm run data
//      (runs frontend/tools/export-results.mjs, which reads ../results/
//       <mode>_matrix.json and writes public/data/results.json)
//   3. Refresh the app. If public/data/results.json exists it wins; otherwise
//      the FALLBACK_MATRICES below are used so the demo always runs.
//
// To extend to more tasks/regions: add entries to REGIONS and add a row of
// the same length to each matrix below (or rerun export-results.mjs on a
// pipeline run with more regions).
// =============================================================================

export type MethodId = "naive" | "joint" | "cl";

export interface RegionInfo {
  id: string; // matches src/utils.py::DEFAULT_REGIONS
  label: string;
  short: string; // "Region 1" style label shown in charts
  desc: string;
}

export const REGIONS: RegionInfo[] = [
  {
    id: "hurricane-michael",
    label: "Hurricane",
    short: "Region 1",
    desc: "Hurricane Michael · Florida, US",
  },
  {
    id: "palu",
    label: "Tsunami",
    short: "Region 2",
    desc: "Palu tsunami · Sulawesi, ID",
  },
  {
    id: "santa-rosa-fire",
    label: "Wildfire",
    short: "Region 3",
    desc: "Santa Rosa fire · California, US",
  },
];

export interface MethodInfo {
  id: MethodId;
  label: string;
  tagline: string;
  color: string; // solid accent
  pale: string; // pale accent (for "lost" shading)
}

export const METHODS: MethodInfo[] = [
  {
    id: "naive",
    label: "Naive fine-tuning",
    tagline: "sequential fine-tune · unbuffered",
    color: "#b3342c",
    pale: "#e5bab0",
  },
  {
    id: "joint",
    label: "Joint training",
    tagline: "all regions pooled · upper bound",
    color: "#4a5568",
    pale: "#cdd3db",
  },
  {
    id: "cl",
    label: "Replay (ours)",
    tagline: "sequential + replay buffer",
    color: "#34703c",
    pale: "#bad5b4",
  },
];

// matrix[row][col] = accuracy (%) on region `col` measured AFTER training on
// task `row` (both 0-indexed). null = region not seen yet at that point.
// Conceptual demo numbers — replace with the real pipeline matrices.
export const FALLBACK_MATRICES: Record<MethodId, (number | null)[][]> = {
  naive: [
    [78, null, null],
    [62, 75, null],
    [55, 68, 74],
  ],
  // Joint sees every region in one run, so the row simply repeats — no drop.
  joint: [
    [75, 77, 76],
    [75, 77, 76],
    [75, 77, 76],
  ],
  cl: [
    [78, null, null],
    [72, 75, null],
    [70, 73, 75],
  ],
};

// ---------------------------------------------------------------------------
// Loader + normalization
// ---------------------------------------------------------------------------

export interface Results {
  matrices: Record<MethodId, (number | null)[][]>;
  labels: string[]; // region ids in matrix column order
}

export function buildResults(
  matrices: Record<MethodId, (number | null)[][]>
): Results {
  return { matrices, labels: REGIONS.map((r) => r.id) };
}

type RawJsonValue = unknown;

function isNan(v: RawJsonValue): boolean {
  // JSON round-trips NaN as null; be liberal anyway.
  return v === null || (typeof v === "number" && Number.isNaN(v));
}

function normalizeMatrix(raw: RawJsonValue): (number | null)[][] | null {
  if (!Array.isArray(raw)) return null;
  const rows = raw.map((row) => {
    if (!Array.isArray(row)) return null;
    return row.map((v) =>
      !isNan(v) && typeof v === "number" ? Math.round(v * 100) : null
    );
  });
  return rows.some((r) => r === null || r.some((v) => v !== null && v < 0))
    ? null
    : (rows as (number | null)[][]);
}

export async function loadResults(): Promise<Results> {
  try {
    const res = await fetch("/data/results.json");
    if (!res.ok) throw new Error("no results.json");
    const json = (await res.json()) as {
      matrices?: Record<string, RawJsonValue>;
      labels?: string[];
    };
    const matrices: Partial<Record<MethodId, (number | null)[][]>> = {};
    for (const m of METHODS) {
      const mtx = normalizeMatrix(json.matrices?.[m.id]);
      if (mtx) matrices[m.id] = mtx;
    }
    if (matrices.naive && matrices.joint && matrices.cl) {
      return buildResults(matrices as Record<MethodId, (number | null)[][]>);
    }
  } catch {
    /* fall through to bundled demo data */
  }
  return buildResults(FALLBACK_MATRICES);
}

// ---------------------------------------------------------------------------
// Derived metrics (mirror src/utils.py::average_forgetting semantics so the
// UI numbers match the report).
// ---------------------------------------------------------------------------

/** Best accuracy a region has shown since it was first evaluated. */
export function bestSoFar(
  matrix: (number | null)[][],
  regionIdx: number,
  taskIdx: number
): number | null {
  let best: number | null = null;
  for (let k = regionIdx; k <= taskIdx; k++) {
    const v = matrix[k]?.[regionIdx];
    if (v === null || v === undefined) continue;
    best = best === null || v > best ? v : best;
  }
  return best;
}

/** Mean accuracy over regions evaluated after the current task. */
export function averageAccuracy(
  matrix: (number | null)[][],
  taskIdx: number
): number | null {
  const row = matrix[taskIdx] ?? [];
  const vals = row.filter((v): v is number => v !== null);
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

/**
 * Average forgetting "so far": for every region evaluated repeatedly, mean of
 * (best-so-far − current). Same definition as src/utils.py::average_forgetting
 * but computed up to the current task.
 */
export function averageForgetting(
  matrix: (number | null)[][],
  taskIdx: number
): number | null {
  const drops: number[] = [];
  for (let regionIdx = 0; regionIdx < matrix.length; regionIdx++) {
    let seen = 0;
    let best: number | null = null;
    for (let k = regionIdx; k <= taskIdx; k++) {
      const v = matrix[k]?.[regionIdx];
      if (v === null || v === undefined) continue;
      seen += 1;
      best = best === null || v > best ? v : best;
    }
    if (seen >= 2 && best !== null) {
      const current = matrix[taskIdx]?.[regionIdx];
      if (current !== null && current !== undefined) drops.push(best - current);
    }
  }
  if (drops.length === 0) return null;
  return drops.reduce((a, b) => a + b, 0) / drops.length;
}

/** One-line narration of what just happened to earlier regions. */
export function narration(
  matrix: (number | null)[][],
  taskIdx: number,
  method: MethodId
): string {
  const current = REGIONS[taskIdx];
  if (taskIdx === 0) {
    return `The model was fine-tuned on the ${current.label.toLowerCase()} region only — nothing to forget yet. Advance the demo to see what happens next.`;
  }
  let worst: { region: RegionInfo; best: number; now: number } | null = null;
  for (let regionIdx = 0; regionIdx < taskIdx; regionIdx++) {
    const best = bestSoFar(matrix, regionIdx, taskIdx);
    const now = matrix[taskIdx]?.[regionIdx];
    if (best === null || now === null || now === undefined) continue;
    const drop = best - now;
    if (drop > 0.05 && (!worst || drop > worst.best - worst.now)) {
      worst = { region: REGIONS[regionIdx], best, now };
    }
  }
  const learned = `after learning ${current.label.toLowerCase()}`;
  if (!worst) {
    switch (method) {
      case "naive":
        return `So far no earlier region has dropped — but naive fine-tuning usually forgets as more regions arrive.`;
      case "joint":
        return `Joint training keeps every region stable — this is the upper bound we compare against.`;
      default:
        return `Experience replay keeps earlier regions stable so far. Step to the next disaster to keep testing.`;
    }
  }
  const pts = Math.round(worst.best - worst.now);
  const when = `despite ${pts} pts of reflex drop (${worst.best}% → ${worst.now}%) ${learned}.`;
  switch (method) {
    case "naive":
      return `Catastrophic forgetting: ${worst.region.label} fell to ${worst.now}% ${when}`;
    case "joint":
      return `Upper bound: joint training stays flat (${worst.region.label} at ${worst.now}%).`;
    default:
      return `Replay contains the drop: ${worst.region.label} is still at ${worst.now}% ${when}`;
  }
}
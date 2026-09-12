// =============================================================================
// Qualitative example images (pred vs true label per method).
//
// REAL DATA: the pipeline can emit a real grid (src/evaluate.py writes
// plots/qualitative_demo.png). Crop it into per-patch images and either
//  - drop a `public/data/qualitative.json` in the same schema below, or
//  - regenerate tiles + JSON with tools/make_assets.py.
//
// To extend: add more entries per region (or new region keys matching REGIONS).
// =============================================================================

import type { MethodId } from "./results";

export interface QualitativeExample {
  image: string; // path served from public/ (or an absolute URL)
  true: string; // undamaged | damaged | destroyed
  pred: Record<MethodId, string>;
}

export type QualitativeExamples = Record<string, QualitativeExample[]>;

// Keys match region ids (hurricane-michael, palu, santa-rosa-fire). Tiles are
// the synthetic placeholders rendered by tools/make_assets.py.
export const QUALITATIVE_FALLBACK: QualitativeExamples = {
  "hurricane-michael": [
    { image: "/assets/qualitative/r1_0.png", true: "damaged", pred: { naive: "destroyed", joint: "damaged", cl: "damaged" } },
    { image: "/assets/qualitative/r1_1.png", true: "undamaged", pred: { naive: "damaged", joint: "undamaged", cl: "undamaged" } },
    { image: "/assets/qualitative/r1_2.png", true: "destroyed", pred: { naive: "undamaged", joint: "destroyed", cl: "damaged" } },
    { image: "/assets/qualitative/r1_3.png", true: "destroyed", pred: { naive: "damaged", joint: "destroyed", cl: "destroyed" } },
    { image: "/assets/qualitative/r1_4.png", true: "damaged", pred: { naive: "destroyed", joint: "damaged", cl: "damaged" } },
    { image: "/assets/qualitative/r1_5.png", true: "undamaged", pred: { naive: "undamaged", joint: "undamaged", cl: "undamaged" } },
    { image: "/assets/qualitative/r1_6.png", true: "damaged", pred: { naive: "destroyed", joint: "damaged", cl: "damaged" } },
    { image: "/assets/qualitative/r1_7.png", true: "destroyed", pred: { naive: "undamaged", joint: "destroyed", cl: "destroyed" } },
  ],
  palu: [
    { image: "/assets/qualitative/r2_0.png", true: "damaged", pred: { naive: "damaged", joint: "damaged", cl: "damaged" } },
    { image: "/assets/qualitative/r2_1.png", true: "destroyed", pred: { naive: "damaged", joint: "destroyed", cl: "destroyed" } },
    { image: "/assets/qualitative/r2_2.png", true: "undamaged", pred: { naive: "damaged", joint: "undamaged", cl: "undamaged" } },
    { image: "/assets/qualitative/r2_3.png", true: "damaged", pred: { naive: "damaged", joint: "damaged", cl: "damaged" } },
    { image: "/assets/qualitative/r2_4.png", true: "undamaged", pred: { naive: "undamaged", joint: "undamaged", cl: "undamaged" } },
    { image: "/assets/qualitative/r2_5.png", true: "destroyed", pred: { naive: "destroyed", joint: "destroyed", cl: "damaged" } },
    { image: "/assets/qualitative/r2_6.png", true: "damaged", pred: { naive: "destroyed", joint: "damaged", cl: "damaged" } },
    { image: "/assets/qualitative/r2_7.png", true: "undamaged", pred: { naive: "damaged", joint: "undamaged", cl: "undamaged" } },
  ],
  "santa-rosa-fire": [
    { image: "/assets/qualitative/r3_0.png", true: "damaged", pred: { naive: "damaged", joint: "damaged", cl: "damaged" } },
    { image: "/assets/qualitative/r3_1.png", true: "undamaged", pred: { naive: "undamaged", joint: "undamaged", cl: "undamaged" } },
    { image: "/assets/qualitative/r3_2.png", true: "destroyed", pred: { naive: "destroyed", joint: "destroyed", cl: "destroyed" } },
    { image: "/assets/qualitative/r3_3.png", true: "damaged", pred: { naive: "damaged", joint: "damaged", cl: "damaged" } },
    { image: "/assets/qualitative/r3_4.png", true: "undamaged", pred: { naive: "undamaged", joint: "undamaged", cl: "undamaged" } },
    { image: "/assets/qualitative/r3_5.png", true: "destroyed", pred: { naive: "destroyed", joint: "damaged", cl: "destroyed" } },
    { image: "/assets/qualitative/r3_6.png", true: "damaged", pred: { naive: "damaged", joint: "damaged", cl: "damaged" } },
    { image: "/assets/qualitative/r3_7.png", true: "undamaged", pred: { naive: "undamaged", joint: "undamaged", cl: "undamaged" } },
  ],
};

async function loadJson<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(path);
    return res.ok ? ((await res.json()) as T) : null;
  } catch {
    return null;
  }
}

export async function loadQualitative(): Promise<QualitativeExamples> {
  const json = await loadJson<QualitativeExamples>("/data/qualitative.json");
  if (json && Object.keys(json).length > 0) return json;
  return QUALITATIVE_FALLBACK;
}
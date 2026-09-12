import type { CSSProperties } from "react";

export interface KpiItem {
  label: string;
  main: string;
  unit?: string;
  note?: string;
  accent?: string;
}

interface KpiStripProps {
  items: KpiItem[];
}

/**
 * A hairline-separated band of headline numbers. Renders as
 * many cells as given; the benchmark best reads when used with
 * exactly 3-4 crisp numbers.
 */
export default function KpiStrip({ items }: KpiStripProps) {
  return (
    <div
      className="kpi-band"
      style={{ "--k": items.length } as CSSProperties}
    >
      {items.map((it, i) => (
        <div className="kpi-cell" key={i}>
          <div className="kpi-label">{it.label}</div>
          <div
            className="kpi-main"
            style={it.accent ? { color: it.accent } : undefined}
          >
            {it.main}
            {it.unit && <span className="kpi-unit">{it.unit}</span>}
          </div>
          {it.note && <div className="kpi-note">{it.note}</div>}
        </div>
      ))}
    </div>
  );
}
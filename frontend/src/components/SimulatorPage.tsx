import { useState, type CSSProperties } from "react";
import {
  METHODS,
  REGIONS,
  averageAccuracy,
  averageForgetting,
  narration,
  type MethodId,
  type Results,
} from "../data/results";
import TaskStrip from "./TaskStrip";
import RegionChart from "./RegionChart";
import ForgettingCurve from "./ForgettingCurve";
import KpiStrip from "./KpiStrip";

interface SimulatorPageProps {
  results: Results;
  method: MethodId;
  onMethodChange: (m: MethodId) => void;
}

export default function SimulatorPage({ results, method, onMethodChange }: SimulatorPageProps) {
  const [taskIdx, setTaskIdx] = useState(0);
  const matrix = results.matrices[method];
  const meta = METHODS.find((m) => m.id === method)!;
  const last = taskIdx === REGIONS.length - 1;

  const blurb = narration(matrix, taskIdx, method);

  const acc = averageAccuracy(matrix, taskIdx);
  const forget = averageForgetting(matrix, taskIdx);
  const naiveForget = averageForgetting(results.matrices.naive, taskIdx);
  const delta =
    forget !== null && naiveForget !== null ? naiveForget - forget : null;

  // Per-method numbers for the ledger (at the current task step).
  const ledger = METHODS.map((m) => ({
    ...m,
    acc: averageAccuracy(results.matrices[m.id], taskIdx),
    forget: averageForgetting(results.matrices[m.id], taskIdx),
  }));

  return (
    <div className="page">
      <header className="page-head">
        <span className="kicker">Live demo console</span>
        <h1>Simulator</h1>
        <p className="lede">
          Step through Hurricane → Tsunami → Wildfire. Per region, per task, the
          console reports accuracy, the forgetting ledger, and the whole
          accuracy matrix for each regime.
        </p>
      </header>

      {/* method selection + run controls --------------------------------- */}
      <section className="controls">
        <div className="regime-tabs" role="tablist" aria-label="Training regime">
          {METHODS.map((m) => (
            <button
              key={m.id}
              className={`tab ${method === m.id ? "active" : ""}`}
              style={({ "--acc": m.color } as CSSProperties)}
              onClick={() => onMethodChange(m.id)}
            >
              <span className="tab-swatch" />
              <span className="tab-main">
                <span className="tab-label">{m.label}</span>
                <span className="tab-sub">{m.tagline}</span>
              </span>
            </button>
          ))}
        </div>

        <div className="ops">
          <span className="step-indicator">
            step <strong>{taskIdx + 1}/03</strong>
          </span>
          <button className="btn ghost" onClick={() => setTaskIdx(0)}>
            reset
          </button>
          <button
            className="btn primary"
            disabled={last}
            onClick={() => setTaskIdx((t) => Math.min(REGIONS.length - 1, t + 1))}
          >
            next task →
          </button>
        </div>
      </section>

      {/* rig readout ------------------------------------------------------- */}
      <div className="chips" aria-label="Simulation rig">
        <span className="chip">backbone · resnet-18 (pretrained)</span>
        <span className="chip">seed 42</span>
        <span className="chip">replay buffer · 300 / region</span>
        <span className="chip">eval · held-out test tiles</span>
      </div>

      {/* task strip ------------------------------------------------------- */}
      <TaskStrip taskIdx={taskIdx} matrix={matrix} accent={meta.color} />

      {/* headline numbers -------------------------------------------------- */}
      <KpiStrip
        items={[
          {
            label: "accuracy · regions evaluated",
            main: acc === null ? "—" : acc.toFixed(1),
            unit: "%",
            note: `${REGIONS[taskIdx].label} just learned — ${REGIONS[0].label} seen ${taskIdx + 1}×`,
          },
          {
            label: "forgetting · so far",
            main: forget === null ? "—" : forget.toFixed(1),
            unit: " pts",
            note: "mean drop vs each region's best score",
            accent: forget !== null && forget > 10 ? "var(--naive)" : undefined,
          },
          {
            label: "forgetting · vs naive",
            main: delta === null ? "—" : `−${delta.toFixed(1)}`,
            unit: " pts",
            note:
              delta !== null && delta > 0.05
                ? `${meta.label.toLowerCase()} holds on better than naive`
                : method === "naive"
                  ? "baseline this step"
                  : "no meaningful gap yet",
            accent: delta !== null && delta > 0.05 ? "var(--replay)" : undefined,
          },
          {
            label: "regime",
            main: meta.label[0].toUpperCase() + meta.label.slice(1),
            note: meta.tagline,
          },
        ]}
      />

      {/* chart + matrix --------------------------------------------------- */}
      <section className="data-grid">
        <div className="sheet fig chartbox">
          <span className="kicker">Fig. 2a · per-region accuracy</span>
          <div style={{ marginTop: 10 }}>
            <RegionChart taskIdx={taskIdx} matrix={matrix} color={meta.color} pale={meta.pale} />
          </div>
          <div className="chart-legend">
            <span><i style={{ background: meta.color }} />accuracy now</span>
            <span><i style={{ background: meta.pale }} />hollowed by forgetting</span>
            <span><i className="ghost-swatch" />not yet learned</span>
          </div>

          <div style={{ height: 22 }} />

          <span className="kicker">Fig. 2b · forgetting across tasks</span>
          <div style={{ marginTop: 10 }}>
            <ForgettingCurve results={results} method={method} />
          </div>
        </div>

        <div className="sheet" style={{ paddingTop: 20 }}>
          <table className="tbl">
            <caption>Table 1 — accuracy matrix, %<br />columns = regions, in task order</caption>
            <thead>
              <tr>
                <th>after task</th>
                {REGIONS.map((r) => (
                  <th key={r.id}>{r.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, i) => (
                <tr key={i} className={i === taskIdx ? "row-current" : ""}>
                  <td className="row-label">
                    {String(i + 1).padStart(2, "0")} · {REGIONS[i].label}
                  </td>
                  {row.map((v, j) => {
                    if (v === null) return <td key={j} className="missing">—</td>;
                    const best = bestSoFarAt(matrix, j, i);
                    const dropped = best !== null && best - v > 0.5;
                    return (
                      <td key={j} className={dropped ? "drop" : "hold"}>
                        {Math.round(v)}
                        {dropped ? " ↓" : ""}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* forgetting ledger -------------------------------------------------- */}
      <section className="ledger">
        <div className="ledger-title">
          <h2>Forgetting ledger</h2>
          <span className="note">
            at step {taskIdx + 1}/03 · active regime: {meta.label.toLowerCase()}
          </span>
        </div>
        <table className="tbl">
          <thead>
            <tr>
              <th>metric</th>
              {ledger.map((m) => (
                <th key={m.id} className={m.id === method ? "col-active" : ""}>
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="metric-name">
                average accuracy — regions evaluated so far
              </td>
              {ledger.map((m) => (
                <td key={m.id} className={m.id === method ? "col-active" : ""}>
                  {m.acc === null ? "—" : `${m.acc.toFixed(1)}%`}
                </td>
              ))}
            </tr>
            <tr>
              <td className="metric-name">average forgetting — so far</td>
              {ledger.map((m) => (
                <td key={m.id} className={m.id === method ? "col-active" : ""}>
                  {m.forget === null ? "—" : `${m.forget.toFixed(1)} pts`}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </section>

      <p className="logline">
        <span className="log-prompt">LOG&nbsp;›&nbsp;</span>
        {blurb}
      </p>

      <p className="page-foot">
        reading the bars: the pale sliver under a solid bar is the accuracy that
        region once held and has since lost — i.e. forgetting. hollow bars = not
        yet learned. figures and tables are precomputed; nothing trains here.
      </p>
    </div>
  );

  // Local helper: best accuracy a region (col j) has shown up to row i.
  function bestSoFarAt(m: (number | null)[][], j: number, i: number): number | null {
    let b: number | null = null;
    for (let k = j; k <= i; k++) {
      const v = m[k]?.[j];
      if (v === null || v === undefined) continue;
      b = b === null || v > b ? v : b;
    }
    return b;
  }
}
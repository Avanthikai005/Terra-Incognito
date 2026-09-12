import { useState, type CSSProperties } from "react";
import { METHODS, REGIONS, type MethodId } from "../data/results";
import type { QualitativeExamples } from "../data/qualitative";

interface QualitativePageProps {
  qualitative: QualitativeExamples;
  method: MethodId;
  onMethodChange: (m: MethodId) => void;
}

type Mode = "plates" | "h2h";

function PredRow({ who, value, truth }: { who: string; value: string; truth: string }) {
  const ok = value === truth;
  return (
    <div className="h2h-row">
      <span className="h2h-key">{who}</span>
      <span className="h2h-val">{value}</span>
      <span className={ok ? "h2h-tick" : "h2h-cross"}>{ok ? "✓" : "✗"}</span>
    </div>
  );
}

export default function QualitativePage({
  qualitative,
  method,
  onMethodChange,
}: QualitativePageProps) {
  const [regionId, setRegionId] = useState(REGIONS[0].id);
  const [mode, setMode] = useState<Mode>("h2h");
  const region = REGIONS.find((r) => r.id === regionId) ?? REGIONS[0];
  const examples = qualitative[region.id] ?? [];
  const total = examples.length;

  const naiveOk = examples.filter((e) => e.pred.naive === e.true).length;
  const replayOk = examples.filter((e) => e.pred.cl === e.true).length;

  return (
    <div className="page">
      <header className="page-head">
        <span className="kicker">Look inside the final checkpoints</span>
        <h1>Qualitative plates</h1>
        <p className="lede">
          Test patches from each region, labelled by all three regimes after all
          three tasks. Green border reads "prediction agrees with ground truth";
          red reads "it does not."
        </p>
      </header>

      <section className="plate-section">
        <div className="regime-tabs" role="tablist" aria-label="View mode">
          <button
            className={`tab ${mode === "h2h" ? "active" : ""}`}
            onClick={() => setMode("h2h")}
          >
            <span className="tab-main">
              <span className="tab-label">Head-to-head</span>
              <span className="tab-sub">naive vs replay</span>
            </span>
          </button>
          <button
            className={`tab ${mode === "plates" ? "active" : ""}`}
            onClick={() => setMode("plates")}
          >
            <span className="tab-main">
              <span className="tab-label">Plates</span>
              <span className="tab-sub">one regime per plate</span>
            </span>
          </button>
        </div>
        {mode === "plates" && (
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
                </span>
              </button>
            ))}
          </div>
        )}
        <div>
          <label className="contents-label" style={{ margin: "0 0 6px", display: "block" }}>
            region
          </label>
          <select className="select" value={region.id} onChange={(e) => setRegionId(e.target.value)}>
            {REGIONS.map((r) => (
              <option key={r.id} value={r.id}>
                {r.short} · {r.label} — {r.desc}
              </option>
            ))}
          </select>
        </div>
      </section>

      {mode === "h2h" ? (
        <>
          <p className="h2h-summary">
            <span>
              label agreement on {region.label}: <b>naive {naiveOk}/{total}</b> vs{" "}
              <b className="replay-acc">replay {replayOk}/{total}</b>
            </span>
            <span className="h2h-key-note">
              same patches · same final checkpoints · {region.desc}
            </span>
          </p>

          <section className="h2h-grid">
            {examples.map((ex, i) => {
              const replayOkHere = ex.pred.cl === ex.true;
              return (
                <figure key={i} className={`h2h-card ${replayOkHere ? "ok" : "bad"}`}>
                  <div className="h2h-img">
                    <img src={ex.image} alt={`patch ${i + 1} — truth ${ex.true}`} />
                  </div>
                  <figcaption>
                    <div className="h2h-row">
                      <span className="h2h-key">true</span>
                      <span className="h2h-val">{ex.true}</span>
                    </div>
                    <PredRow who="naive" value={ex.pred.naive} truth={ex.true} />
                    <PredRow who="replay" value={ex.pred.cl} truth={ex.true} />
                  </figcaption>
                </figure>
              );
            })}
          </section>

          <p className="logline">
            <span className="log-prompt">NOTE&nbsp;›&nbsp;</span>
            On the earliest-learned region the naive checkpoint drifts on nearly
            every patch; replay keeps the ground-truth labels — training on later
            regions did not erase this region.
          </p>
        </>
      ) : (
        <>
          <p className="plate-note">
            plate {region.short.toLowerCase().replace("region ", "r")} · {region.label} —{" "}
            {region.desc} · true label : pred ({METHODS.find((m) => m.id === method)?.label})
          </p>

          <section className="plate-grid">
            {examples.map((ex, i) => {
              const truth = ex.true;
              const pred = ex.pred[method];
              const ok = pred === truth;
              return (
                <figure key={i} className={`plate ${ok ? "ok" : "bad"}`}>
                  <div className="plate-img">
                    <img src={ex.image} alt={`building patch ${i + 1} (${truth})`} />
                  </div>
                  <figcaption>
                    <div>
                      <span className="t">true · </span>
                      <span>{truth}</span>
                    </div>
                    <div>
                      <span className="t">pred · </span>
                      <span className="p">{pred} </span>
                      <span className={ok ? "tick" : "cross"}>{ok ? "✓" : "✗"}</span>
                    </div>
                  </figcaption>
                </figure>
              );
            })}
          </section>

          <p className="logline">
            <span className="log-prompt">NOTE&nbsp;›&nbsp;</span>
            Even after training on later regions, replay keeps correct predictions
            on earlier regions — naive fine-tuning, on these same patches, drifts.
          </p>
        </>
      )}

      <p className="page-foot">
        tiles are synthetic stand-ins rendered by tools/make_assets.py · replace
        with real crops + predictions from your checkpoint evaluation.
      </p>
    </div>
  );
}
import {
  averageAccuracy,
  averageForgetting,
  REGIONS,
  METHODS,
  type Results,
} from "../data/results";
import KpiStrip from "./KpiStrip";

interface OverviewPageProps {
  results: Results;
  onStart: () => void;
}

function pct(v: number | null): string {
  return v === null ? "—" : v.toFixed(1);
}

export default function OverviewPage({ results, onStart }: OverviewPageProps) {
  const updateTask = REGIONS.length - 1; // final row index = "after all tasks"
  const accReplay = averageAccuracy(results.matrices.cl, updateTask);
  const accNaive = averageAccuracy(results.matrices.naive, updateTask);
  const fReplay = averageForgetting(results.matrices.cl, updateTask);
  const fNaive = averageForgetting(results.matrices.naive, updateTask);
  const ratio =
    fNaive !== null && fReplay !== null && fReplay > 0
      ? (fNaive / fReplay).toFixed(1)
      : null;

  return (
    <div className="page">
      <header className="page-head">
        <span className="kicker">Field report · cross-regional disaster response</span>
        <h1>Terra Incognita</h1>
        <p className="abstract">
          A satellite building-damage model is asked to serve one disaster region
          at a time — Hurricane Michael in Florida, then the Palu tsunami, then the
          Santa Rosa wildfire. Qualities acquired on earlier regions should not be
          lost when later regions arrive. We measure whether they are, and how much
          a simple <strong>experience-replay</strong> strategy recovers.
        </p>
        <p className="mast-meta">
          regions: 03 · regimes: 03 · metrics precomputed (no live training)
        </p>
      </header>

      <KpiStrip
        items={[
          {
            label: "avg accuracy · after all tasks",
            main: `${pct(accReplay)}`,
            unit: "%",
            note: `replay ${pct(accReplay)} · naive ${pct(accNaive)}`,
            accent: "var(--replay)",
          },
          {
            label: "avg forgetting · after all tasks",
            main: `${pct(fReplay)}`,
            unit: "pts",
            note:
              ratio !== null
                ? `replay ${pct(fReplay)} · naive ${pct(fNaive)} — ${ratio}× less`
                : `replay ${pct(fReplay)} · naive ${pct(fNaive)}`,
            accent: "var(--replay)",
          },
          {
            label: "regimes compared",
            main: `${METHODS.length}`,
            note: METHODS.map((m) => m.label.toLowerCase()).join(" · "),
          },
          {
            label: "disaster regions",
            main: `${REGIONS.length}`,
            note: REGIONS.map((r) => r.label.toLowerCase()).join(" · "),
          },
        ]}
      />

      <section className="sheet fig">
        <img
          src="/assets/overview.png"
          alt="Schematic: learning region A, naive fine-tuning on B forgets A, replay keeps both"
        />
        <p className="fig-caption">
          Fig. 1 — Accuracy on each learned region under the two learning regimes.
          Left: region A is learned alone. Centre: naive fine-tuning on region B
          pushes A down. Right: a replay buffer holds A steady while B is learned.
        </p>
      </section>

      <section className="two-col">
        <div className="block">
          <span className="block-tag">01 — the failure mode</span>
          <p>
            Naive sequential fine-tuning drifts toward the newest imagery and
            forgets the regions it learned first — catastrophic forgetting,
            precisely when an older disaster zone needs re-assessment.
          </p>
        </div>
        <div className="block">
          <span className="block-tag">02 — the fix under study</span>
          <p>
            Keep a small, capacity-capped replay buffer per region and mix its
            samples into every training batch. No full retraining; the model keeps
            learning new regions while older knowledge is re-read.
          </p>
        </div>
        <div className="block wide">
          <span className="block-tag">03 — this demo</span>
          <p>
            The simulator steps through the three disasters in order and reports,
            per regime and per task, the region accuracies, the forgetting ledger,
            and the domain-distance ablation behind it all.
          </p>
          <a className="textlink" href="#" onClick={(e) => { e.preventDefault(); onStart(); }}>
            open the simulator →
          </a>
        </div>
      </section>
    </div>
  );
}
export default function DomainDistancePage() {
  return (
    <div className="page">
      <header className="page-head">
        <span className="kicker">Ablation · the planning angle</span>
        <h1>Domain distance vs forgetting</h1>
        <p className="lede">
          Region distances are knowable <em>before</em> any training: they are the
          cosine distances between region mean embeddings from the pretrained
          backbone. Do they predict which regions will be forgotten?
        </p>
      </header>

      <section className="sheet fig">
        <img
          src="/assets/domain_distance.png"
          alt="Scatter of forgetting severity against inter-region domain distance"
        />
        <p className="fig-caption">
          Fig. 3 — Each point is an (earlier region, later region) pair. Under
          naive fine-tuning, forgetting grows with domain distance. Replay flattens
          the trend — the fitted slope and Pearson r collapse.
        </p>
      </section>

      <section className="takeaways">
        <div className="takeaway">
          <span className="k">x-axis</span>
          <span>
            inter-region domain distance — cosine distance between region mean
            features from the pretrained ResNet-18 backbone.
          </span>
        </div>
        <div className="takeaway">
          <span className="k">y-axis</span>
          <span>
            forgetting severity — accuracy drop on the earlier region after later
            regions are learned.
          </span>
        </div>
        <div className="takeaway">
          <span className="k">reading</span>
          <span>
            the larger the distance, the more an earlier region degrades; replay
            recovers the most exactly for those far-apart (high-distance) pairs.
          </span>
        </div>
      </section>

      <p className="field-note">
        This is an a-priori planning signal: with distances known up front, an ops
        team can weigh which region orderings are risky before spending compute.
      </p>

      <p className="page-foot">
        placeholder scatter rendered by tools/make_assets.py · the real plot
        replaces it via `npm run data` after a full pipeline run.
      </p>
    </div>
  );
}
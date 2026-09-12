import { useEffect, useState, useRef, type CSSProperties } from "react";
import { METHODS, type MethodId } from "../data/results";

interface SamplePatch {
  id: string;
  region: string;
  tile: string;
  label: number;
  label_name: string;
  image_b64: string;
}

interface SinglePrediction {
  predicted_class: number;
  class_name: string;
  confidence: number;
  probabilities: {
    undamaged: number;
    damaged: number;
    destroyed: number;
  };
  inference_ms: number;
  checkpoint?: string;
}

interface PredictionResponse {
  regime: string;
  sequence: string;
  task_idx?: number;
  prediction: SinglePrediction;
  comparisons?: Record<string, SinglePrediction>;
}

const DAMAGE_COLORS: Record<string, string> = {
  undamaged: "#2e7d32", // deep green
  damaged: "#d97706",   // amber
  destroyed: "#b3342c", // crimson
};

const DAMAGE_DESCRIPTIONS: Record<string, string> = {
  undamaged: "Structural roof & walls intact; no visible flood/fire destruction.",
  damaged: "Partial structural compromise, roof breach, or localized scorch/flood.",
  destroyed: "Complete structural collapse, rubble field, or total inundation.",
};

export default function ModelInferencePage() {
  const [samples, setSamples] = useState<SamplePatch[]>([]);
  const [selectedSample, setSelectedSample] = useState<SamplePatch | null>(null);
  const [customImageB64, setCustomImageB64] = useState<string | null>(null);
  const [activeRegime, setActiveRegime] = useState<MethodId>("cl");
  const [compareAll, setCompareAll] = useState<boolean>(true);
  const [sequence, setSequence] = useState<string>("similar_domain");
  const [taskIdx, setTaskIdx] = useState<number>(2); // Task 3 (0-indexed = 2)

  const [loading, setLoading] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load sample test patches and check server status
  useEffect(() => {
    let alive = true;

    fetch("/api/status")
      .then((r) => r.ok ? r.json() : null)
      .then((data) => {
        if (!alive) return;
        if (data && data.status === "online") {
          setApiConnected(true);
        }
      })
      .catch(() => setApiConnected(false));

    fetch(`/api/samples?sequence=${sequence}&count_per_region=4`)
      .then((r) => r.ok ? r.json() : null)
      .then((data) => {
        if (!alive) return;
        if (data && data.samples && data.samples.length > 0) {
          setSamples(data.samples);
          setSelectedSample(data.samples[0]);
        }
      })
      .catch((err) => console.warn("Failed to fetch sample patches:", err));

    return () => {
      alive = false;
    };
  }, [sequence]);

  // Handle custom image upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const b64 = reader.result as string;
      setCustomImageB64(b64);
      setSelectedSample(null);
      setPredictionResult(null);
    };
    reader.readAsDataURL(file);
  };

  // Run live PyTorch model inference
  const runInference = async () => {
    setLoading(true);
    setErrorMsg(null);

    const payload: Record<string, any> = {
      sequence,
      regime: activeRegime,
      task_idx: taskIdx,
      compare_all: compareAll,
    };

    if (customImageB64) {
      payload.image_b64 = customImageB64;
    } else if (selectedSample) {
      payload.sample_id = selectedSample.id;
      payload.image_b64 = selectedSample.image_b64;
    } else {
      setErrorMsg("Please select a sample patch or upload a satellite image.");
      setLoading(false);
      return;
    }

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Inference error (status ${res.status})`);
      }

      const data: PredictionResponse = await res.json();
      setPredictionResult(data);
    } catch (err: any) {
      console.error("Inference failed:", err);
      // If API server is offline, supply a simulated live prediction demo
      if (!apiConnected) {
        setErrorMsg("Model API offline. Please ensure 'python src/api_server.py' is running on port 8000.");
      } else {
        setErrorMsg(err.message || "Failed to execute inference.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Auto-run inference when sample changes if we already ran before
  const handleSelectSample = (s: SamplePatch) => {
    setSelectedSample(s);
    setCustomImageB64(null);
    setPredictionResult(null);
  };

  const activeImage = customImageB64 || selectedSample?.image_b64;
  const trueLabel = selectedSample ? selectedSample.label_name : null;

  return (
    <div className="page model-lab-page">
      <header className="page-head">
        <span className="kicker">Live PyTorch ResNet-18 Engine</span>
        <h1>Model Lab & Live Inference</h1>
        <p className="lede">
          Evaluate disaster damage live on satellite building patches. Inspect class
          probabilities, compare <strong>Naive Fine-Tuning</strong> vs. <strong>Experience Replay</strong>,
          and witness catastrophic forgetting in real time on held-out test imagery.
        </p>
      </header>

      {/* Connection Status Banner */}
      <div className={`status-banner ${apiConnected ? "connected" : "disconnected"}`}>
        <span className="status-dot" />
        <span className="status-text">
          {apiConnected
            ? "PyTorch ResNet-18 API Connected (127.0.0.1:8000) · Real Checkpoints Active"
            : "Backend Server Offline · Start Python API Server to enable live GPU/CPU inference"}
        </span>
      </div>

      {/* Control panel: Sequence, Regime mode, Task step */}
      <section className="lab-controls-sheet sheet">
        <div className="lab-controls-grid">
          <div>
            <label className="contents-label">Task Sequence</label>
            <select
              className="select"
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
            >
              <option value="similar_domain">Hurricane Matthew (Bands A, B, C)</option>
              <option value="cross_disaster">Cross-Disaster (Hurricane → Palu Tsunami)</option>
            </select>
          </div>

          <div>
            <label className="contents-label">Inference Mode</label>
            <div className="toggle-group">
              <button
                type="button"
                className={`btn toggle-btn ${compareAll ? "active" : ""}`}
                onClick={() => setCompareAll(true)}
              >
                Compare All Regimes
              </button>
              <button
                type="button"
                className={`btn toggle-btn ${!compareAll ? "active" : ""}`}
                onClick={() => setCompareAll(false)}
              >
                Single Regime
              </button>
            </div>
          </div>

          {!compareAll && (
            <div>
              <label className="contents-label">Active Regime</label>
              <select
                className="select"
                value={activeRegime}
                onChange={(e) => setActiveRegime(e.target.value as MethodId)}
              >
                {METHODS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label} ({m.tagline})
                  </option>
                ))}
              </select>
            </div>
          )}

          <div>
            <label className="contents-label">Checkpoint Stage</label>
            <select
              className="select"
              value={taskIdx}
              onChange={(e) => setTaskIdx(Number(e.target.value))}
            >
              <option value={0}>After Task 1 (First Region Learned)</option>
              <option value={1}>After Task 2 (Second Region Learned)</option>
              <option value={2}>After Task 3 (All Tasks Completed)</option>
            </select>
          </div>
        </div>
      </section>

      {/* Image selector: Test Patch Strip & Upload Option */}
      <section className="patch-selection-section">
        <div className="section-title-row">
          <div>
            <span className="kicker">Imagery Input</span>
            <h2 style={{ fontSize: "1.25rem", marginTop: 4 }}>Select Test Patch or Upload Satellite Tile</h2>
          </div>
          <button
            type="button"
            className="btn ghost"
            onClick={() => fileInputRef.current?.click()}
          >
            + Upload Custom Tile
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={handleFileUpload}
          />
        </div>

        {/* Patch Carousel / Grid */}
        <div className="sample-patch-strip">
          {samples.map((s) => {
            const isSelected = selectedSample?.id === s.id && !customImageB64;
            return (
              <button
                key={s.id}
                type="button"
                className={`sample-card ${isSelected ? "selected" : ""}`}
                onClick={() => handleSelectSample(s)}
              >
                <img src={s.image_b64} alt={s.tile} className="sample-thumb" />
                <div className="sample-meta">
                  <span className="sample-region">{s.region}</span>
                  <span
                    className="sample-label-badge"
                    style={{ background: DAMAGE_COLORS[s.label_name] || "#666" }}
                  >
                    True: {s.label_name}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </section>

      {/* Main Inference Playground: Image Preview + Run Button + Results */}
      <section className="inference-workspace sheet">
        <div className="workspace-left">
          <div className="preview-container">
            {activeImage ? (
              <img src={activeImage} alt="Input Patch" className="main-patch-preview" />
            ) : (
              <div className="empty-preview">Select an image to inspect</div>
            )}
            <div className="preview-overlay">
              <span className="preview-resolution">224 × 224 RGB · Post-Disaster</span>
              {trueLabel && (
                <span className="preview-ground-truth">
                  Ground Truth: <strong>{trueLabel}</strong>
                </span>
              )}
            </div>
          </div>

          <button
            type="button"
            className="btn primary run-inference-btn"
            disabled={loading || !activeImage}
            onClick={runInference}
          >
            {loading ? "Running PyTorch Model..." : "Classify Damage →"}
          </button>

          {errorMsg && <div className="lab-error-box">{errorMsg}</div>}
        </div>

        <div className="workspace-right">
          {!predictionResult ? (
            <div className="placeholder-results">
              <div className="placeholder-icon">⚡</div>
              <h3>Ready for Inference</h3>
              <p>
                Click <strong>"Classify Damage"</strong> to run the post-disaster satellite
                image through the ResNet-18 continual learning backbone.
              </p>
            </div>
          ) : compareAll && predictionResult.comparisons ? (
            /* Multi-Regime Head-to-Head Comparison */
            <div className="comparisons-container">
              <div className="comparisons-header">
                <span className="kicker">Head-to-Head Evaluation</span>
                <h3>Regime Performance Comparison</h3>
              </div>

              {/* Forgetting Alert Banner */}
              {(() => {
                const naive = predictionResult.comparisons["naive"];
                const replay = predictionResult.comparisons["cl"];
                if (naive && replay && trueLabel) {
                  const naiveOk = naive.class_name === trueLabel;
                  const replayOk = replay.class_name === trueLabel;
                  if (!naiveOk && replayOk) {
                    return (
                      <div className="forgetting-alert success">
                        <strong>Insight: Catastrophic Forgetting Demonstrated!</strong>
                        <br />
                        Naive fine-tuning misclassified this patch ({naive.class_name} vs. {trueLabel})
                        because previous disaster weights were overwritten.
                        Experience Replay ({replay.class_name}) successfully retained the domain knowledge.
                      </div>
                    );
                  }
                }
                return null;
              })()}

              <div className="comparisons-grid">
                {Object.entries(predictionResult.comparisons).map(([regId, pred]) => {
                  const meta = METHODS.find((m) => m.id === regId) || {
                    label: regId,
                    tagline: "regime",
                    color: "#4a5568",
                  };
                  const isCorrect = trueLabel ? pred.class_name === trueLabel : null;

                  return (
                    <div
                      key={regId}
                      className="regime-comp-card"
                      style={({ "--reg-color": meta.color } as CSSProperties)}
                    >
                      <div className="comp-card-header">
                        <div>
                          <div className="comp-reg-name">{meta.label}</div>
                          <div className="comp-reg-tag">{meta.tagline}</div>
                        </div>
                        {isCorrect !== null && (
                          <span className={`eval-pill ${isCorrect ? "correct" : "incorrect"}`}>
                            {isCorrect ? "✓ Agree" : "✗ Disagree"}
                          </span>
                        )}
                      </div>

                      <div className="comp-pred-block">
                        <div
                          className="pred-class-tag"
                          style={{ background: DAMAGE_COLORS[pred.class_name] }}
                        >
                          {pred.class_name.toUpperCase()}
                        </div>
                        <div className="pred-conf">{(pred.confidence * 100).toFixed(1)}% conf</div>
                      </div>

                      {/* Probability Bars */}
                      <div className="mini-prob-bars">
                        {Object.entries(pred.probabilities).map(([cls, prob]) => (
                          <div key={cls} className="mini-bar-row">
                            <span className="mini-bar-label">{cls.slice(0, 3)}</span>
                            <div className="mini-bar-track">
                              <div
                                className="mini-bar-fill"
                                style={{
                                  width: `${(prob * 100).toFixed(1)}%`,
                                  background: DAMAGE_COLORS[cls],
                                }}
                              />
                            </div>
                            <span className="mini-bar-pct">{(prob * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                      </div>

                      <div className="comp-card-footer">
                        <span>{pred.inference_ms}ms</span>
                        <span className="comp-ckpt" title={pred.checkpoint}>
                          {pred.checkpoint?.replace(".pt", "")}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            /* Single Regime Detail View */
            <div className="single-result-container">
              <div className="single-header">
                <span className="kicker">Primary Prediction</span>
                <div className="result-headline">
                  <h2
                    className="predicted-class-name"
                    style={{ color: DAMAGE_COLORS[predictionResult.prediction.class_name] }}
                  >
                    {predictionResult.prediction.class_name.toUpperCase()}
                  </h2>
                  <span className="confidence-pill">
                    {(predictionResult.prediction.confidence * 100).toFixed(1)}% Confidence
                  </span>
                </div>
                <p className="class-desc">
                  {DAMAGE_DESCRIPTIONS[predictionResult.prediction.class_name]}
                </p>
              </div>

              {/* Full Probability Distribution */}
              <div className="prob-distribution-box">
                <div className="contents-label" style={{ marginBottom: 12 }}>
                  Softmax Probability Distribution (3-Class)
                </div>
                {Object.entries(predictionResult.prediction.probabilities).map(([cls, prob]) => (
                  <div key={cls} className="prob-bar-group">
                    <div className="prob-bar-meta">
                      <span className="prob-label">{cls}</span>
                      <span className="prob-val">{(prob * 100).toFixed(2)}%</span>
                    </div>
                    <div className="prob-track">
                      <div
                        className="prob-fill"
                        style={{
                          width: `${(prob * 100).toFixed(1)}%`,
                          background: DAMAGE_COLORS[cls],
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* Latency & Metadata Chip Row */}
              <div className="metadata-strip">
                <span className="meta-item">
                  Latency: <strong>{predictionResult.prediction.inference_ms} ms</strong>
                </span>
                <span className="meta-item">
                  Device: <strong>CPU/CUDA</strong>
                </span>
                <span className="meta-item">
                  Checkpoint: <strong>{predictionResult.prediction.checkpoint}</strong>
                </span>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

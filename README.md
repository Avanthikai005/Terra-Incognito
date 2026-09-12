# Terra Incognita — Continual Learning for Cross‑Regional Disaster Response

Satellite‑based building‑damage assessment (xBD/xView2) trained **sequentially across
disasters/regions**: a Hurricane (`hurricane-michael`), a Tsunami (`palu`), and a
Wildfire (`santa-rosa-fire`). We compare three regimes, measure catastrophic forgetting,
and — our originality anchor — correlate forgetting severity with a cheap
**domain‑distance** computed from pretrained‑backbone features.

## Run (one command)

```bash
bash run.sh [--subset N]     # N = max patches/region for a fast smoke test
```

Installs deps into `venv/`, prepares patches, trains **naive → joint → cl**, evaluates,
and plots. If xBD is **not** downloaded under `data/xbd/`, a learnable synthetic
fallback kicks in so the full pipeline still runs end‑to‑end.

### UI demo (`frontend/`)

Interactive training‑timeline simulator for presenting the project — no Streamlit.

```bash
cd frontend
bash run.sh            # deps + placeholder assets + real data export, then `npm run dev`
# or directly:  npm install && npm run dev
```

Open the printed URL (default http://localhost:5173). Four pages:

- **Overview** — the problem in 3 bullets + schematic.
- **Simulator** — step through Hurricane → Tsunami → Wildfire; pick a regime and
  read the per‑region accuracy chart + full accuracy matrix and the forgetting
  ledger (naive vs joint vs replay all reported side by side).
- **Domain Distance** — forgetting vs inter‑region feature distance (key ablation).
- **Qualitative** — per‑patch pred vs true labels for each method.

Bundled demo numbers keep the app running out of the box. To plug in **real
pipeline results**: run `bash run.sh` at the repo root, then
`npm run data` in `frontend/` — this reads `results/{naive,joint,cl}_matrix.json`
into `frontend/public/data/results.json` and swaps in `plots/domain_distance.png`.
Placeholder assets are rendered by `frontend/tools/make_assets.py` (overview
schematic, distance scatter, synthetic qualitative tiles).

### Real xBD
Place the raw data so that `data/xbd/<disaster>/images/*.png` and
`data/xbd/<disaster>/labels/*.json` exist, then run `bash run.sh`.
`data/prepare_xbd.py` crops building polygons (padded bbox, 224×224) from the
**post**‑disaster image and collapses the 4‑class damage into
`undamaged / damaged / destroyed`.

## Regimes & tasks

| Regime | Behaviour |
|---|---|
| `naive` | Sequential fine‑tune on A → B → C. Lower bound. |
| `joint` | All regions pooled, single training run. Upper bound. |
| `cl` | Sequential + **experience replay** (per‑region buffer, mixed into every batch). Our method. |

Eval happens on a held‑out test split we never train on. **Splits are done by tile** so
buildings from the same tile are never scattered across train/test.

## Outputs

- `report/results.md` — accuracy matrices (rows: after task i; cols: per‑region acc) +
  per‑method **average forgetting** (`src/utils.py::average_forgetting`).
- `plots/domain_distance.png` — forgetting severity vs inter‑region cosine distance
  of pretrained ResNet‑18 mean features (`src/domain_distance.py`).
- `plots/qualitative_demo.png` — early‑region test patches, pred vs true, from the
  **final cl** model (shows what is retained).
- `results/` — JSON matrices, checkpoints.

## Repo layout

```
├── run.sh
├── requirements.txt
├── data/prepare_xbd.py      # polygon→patch; real xBD + synthetic fallback
├── src/
│   ├── model.py             # pretrained ResNet‑18 + 3‑class head
│   ├── train.py             # naive / joint / cl driver (seed=42)
│   ├── continual_methods.py # ReplayBuffer + batch mixing
│   ├── evaluate.py          # matrices, forgetting, demo grid
│   ├── domain_distance.py   # feature distances vs forgetting
│   └── utils.py             # seeding, loaders, split, metrics
└── report/report.md         # ≤4‑page write‑up
```

## Citations

- xBD dataset: Gupta, S. et al., *xBD: A Dataset for Assessing Building Damage
  from Satellite Imagery* (NeurIPS 2019). https://xview2.org
- Pretrained backbone: He, K. et al., *Deep Residual Learning* (CVPR 2016),
  torchvision `ResNet18_Weights.IMAGENET1K_V1`.
- Experience replay for continual learning: e.g., Rolnick et al., *Experience
  Replay for Continual Learning* (NeurIPS 2019); Chaudhry et al., *On Tiny
  Episodic Memories in Continual Learning* (arXiv:1902.10486).
- (Optional context) EWC: Kirkpatrick et al., *Overcoming catastrophic
  forgetting* (PNAS 2017); LwF: Li & Hoiem, *Learning without Forgetting*
  (ECCV 2016).

Everything seeded with `seed=42`.
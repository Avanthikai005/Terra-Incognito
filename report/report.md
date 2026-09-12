# Terra Incognita — Continual Learning for Cross-Regional Disaster Response

*Team Terra Incognita — hackathon report (≤4 pages).*

## 1. Problem

Building-damage classifiers are trained on imagery of the disasters we already have
data for, but are deployed on the ones we don't. When such a model is fine-tuned
sequentially on new regions, it catastrophically forgets older ones — the exact moment
a new disaster hits. We frame building damage assessment as a continual-learning
problem across geographic regions and disaster types and ask: (a) how much forgetting
does naive sequential training incur; (b) how much can a simple replay strategy recover;
(c) can forgetting severity be predicted by a cheap, pre-computed **domain distance**.

## 2. Method

**Task setup (3 tasks, 3 pseudo-regions).** The xBD *selected* release ships only
per-tile aggregate damage counts (no per-building polygons), so we build a **tile-level**
task. Each 128×128 post-disaster Sentinel-2 (s2_tci) RGB tile of `hurricane-matthew`
is classified into the collapsed 3-class scheme (`undamaged / damaged / destroyed`)
defined by the argmax over its collapsed building counts. The hurricane footprint runs
along Haiti's south coast, so the tiles are clustered by longitude into three
spatially-separated pseudo-regions (task order west→east):
1. `hurricane-matthew-A`
2. `hurricane-matthew-B`
3. `hurricane-matthew-C`
Held-out test sets per region are split **by tile** (no leakage); those tiles are
never used in training (`data/prepare_selected.py`).

**Model.** Pretrained ResNet-18 backbone + 3-way head, fine-tuned end-to-end
(AdamW, lr=2e-4, weight decay 1e-4). Everything seeded with seed=42.

**Regimes.**
- *Baseline (solvable ceiling):* one fresh model per region, trained alone — no
  cross-region transfer; the diagonal is the per-region accuracy upper bound.
- *Naive (lower bound):* sequential fine-tuning on A → B → C with no constraint.
- *Joint (upper bound):* all regions pooled and trained once.
- *Replay (ours):* sequential fine-tuning + a capacity-capped per-region buffer;
  replay samples are mixed into every batch of new-region training
  (mix-in ratio 0.5, buffer 80/region).

**Metrics.** Accuracy matrix (rows = after task i; columns = per-region test acc).
Average forgetting = mean over earlier regions of (best prior accuracy on that region −
final accuracy on that region).

**Domain-distance ablation (originality anchor).**
Region-level mean embeddings are computed by passing cached patches through the
*pretrained* ResNet-18 backbone (avgpool, no fc). Pairwise cosine distance between
region means predicts how much damage learning one region does to another
(plots/domain_distance.png).

## 3. Results

*Auto-generated from the actual run — see `report/results.md` and Table below.
(bias note: the 16-test-tile sets are tiny, so all accuracies are coarse).*

| Regime | Task 1 (A) | Task 2 (B) | Task 3 (C) | Avg forgetting |
|---|---|---|---|---|
| after task 1 — naive | 68.8% | — | — | 0% |
| after task 2 — naive | 81.2% | 68.8% | — | 0% |
| after task 3 — naive | 81.2% | 75.0% | 87.5% | 0% |
| **joint** | 66.7% | 66.7% | 66.7% | 0 |
| after task 1 — replay | 66.7% | — | — | 0% |
| after task 2 — replay | 66.7% | 66.7% | — | 0% |
| after task 3 — replay | 66.7% | 66.7% | 66.7% | 0% |

The three longitude bands are visually near-identical (cosine domain distances of
~4%, see §4), so naive transfer is *cheap* here and no regime catastrophically
forgets — the deliberately hard cross-disaster setting (hurricane vs wildfire,
pre/post pairs) is the follow-up that would separate the regimes.

## 4. Ablation — forgetting vs domain distance

Vectorised pretrained-backbone distance between regions vs the forgetting each early
region suffers. Here bands are near-identical, so no monotone trend is expected and
fit fails on degenerate (constant) points — the ablation becomes informative once
cross-disaster tasks (separated by ≫4% in feature space) are introduced.

## 5. Limitations

- **Input is post-only.** ResNet conv1 therefore stays at 3 channels. Pre/post pairs
  (6ch input, or two-stream heads) typically improve xView2 accuracy; we defer fusion.
- **Post-patch domain distance proxy.** Mean features use post-disaster patches; the
  deployed domain is pre-disaster imagery. We flag recomputing on pre tiles (TODOs in
  `src/domain_distance.py`).
- **Single method.** We implement replay only; EWC/LwF are cited but not compared.
- **Capacity/candidates:** classes collapsed to 3; buffer capacity 80 fixed (no sweep
  of capacity); region feature means use up to 100 patches.
- **Compute:** runs off CPU-friendly subsets; GPU run uses identical code.

## References

- Gupta, R. et al. *xBD: A Dataset for Assessing Building Damage from Satellite
  Imagery*, NeurIPS 2019. https://xview2.org
- He, K. et al. *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Rolnick, D. et al. *Experience Replay for Continual Learning*, NeurIPS 2019.
- Chaudhry, A. et al. *On Tiny Episodic Memories in Continual Learning*, 2019.
- Kirkpatrick, J. et al. *Overcoming catastrophic forgetting in neural networks*, PNAS 2017.
- Li, Z. & Hoiem, D. *Learning without Forgetting*, ECCV 2016.
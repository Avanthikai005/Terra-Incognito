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

**Task setup (3 tasks, 3 regions).** Using xBD, tasks are split by disaster event
so each task is a different region type:
1. `hurricane-michael` — hurricane/flood (Florida, US)
2. `palu` — tsunami (Sulawesi, Indonesia)
3. `santa-rosa-fire` — wildfire (California, US)
Building polygons from label JSONs are cropped (padded bbox) from the **post**-disaster
image at 224×224; the xView2 4-class damage is collapsed to `undamaged / damaged /
destroyed`. Held-out test sets per region are split **by tile** (no leakage); those
tiles are never used in training.

**Model.** Pretrained ResNet-18 backbone + 3-way head, fine-tuned end-to-end
(AdamW, lr=2e-4, weight decay 1e-4). Everything seeded with seed=42.

**Regimes.**
- *Naive (lower bound):* sequential fine-tuning on A → B → C with no constraint.
- *Joint (upper bound):* all regions pooled and trained once.
- *Replay (ours):* sequential fine-tuning + a capacity-capped per-region buffer;
  replay samples are mixed into every batch of new-region training
  (mix-in ratio 0.5, buffer 300/region).

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
(fill from `bash run.sh` output)*

| Regime | Task 1 (hurricane) | Task 2 (palu) | Task 3 (wildfire) | Avg forgetting |
|---|---|---|---|---|
| after task 1 — naive | … | — | — | … |
| after task 2 — naive | … | … | — | … |
| after task 3 — naive | … | … | … | … |
| **joint** | … | … | … | 0 |
| after task 1 — replay | … | — | — | |
| after task 2 — replay | … | … | — | |
| after task 3 — replay | … | … | … | … |

Expected patterns (from literature + our synthetic smoke run): naive degrades on
earlier regions; joint is the ceiling; replay recovers a large share of the gap and
shows much lower forgetting, especially on the pair of regions that are *close*
in feature space.

## 4. Ablation — forgetting vs domain distance

Vectorised pretrained-backbone distance between regions vs the forgetting each early
region suffers. We expect a monotone trend (larger distance ⇒ more forgetting), i.e.,
positive Pearson r for naive, weakened by replay. This gives an *a priori* estimate of
which sequential orderings are safe — a planning tool, not just a post-hoc metric.

## 5. Limitations

- **Input is post-only.** ResNet conv1 therefore stays at 3 channels. Pre/post pairs
  (6ch input, or two-stream heads) typically improve xView2 accuracy; we defer fusion.
- **Post-patch domain distance proxy.** Mean features use post-disaster patches; the
  deployed domain is pre-disaster imagery. We flag recomputing on pre tiles (TODOs in
  `src/domain_distance.py`).
- **Single method.** We implement replay only; EWC/LwF are cited but not compared.
- **Capacity/candidates:** classes collapsed to 3; buffer capacity 300 fixed (no sweep
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
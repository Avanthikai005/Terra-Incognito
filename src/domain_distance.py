# Terra Incognita — Domain-distance analysis ("originality anchor").
#
# 1. Region-wise mean feature vector from the PRETRAINED ResNet-18 backbone
#    (avgpool, no fc) on POST-disaster s2-RGB tiles (default), i.e. the
#    damage-domain shift that is actually task-relevant for the classifier.
#    Measured on each region's TRAIN split: every image used is legitimately
#    available when the adaptive decision is made (no test leakage).
#    --features pre instead reads the pre-disaster s2-RGB cache from prep.
# 2. Pairwise cosine distance between region mean vectors.
# 3. Per-sequence outputs:
#      results/<seq>_domain_distance.npy          (matrix)
#      results/<seq>_domain_distances.json        (dict — consumed by cl_adaptive)
#      plots/domain_distance_<seq>.png            (heatmap)
#      plots/domain_distance_forgetting_<seq>.png (scatter vs forgetting, needs matrices)
#
# Usage:
#   python -m src.domain_distance --sequence similar_domain
#   python -m src.domain_distance --sequence cross_disaster --features post
#   python -m src.domain_distance --sequence all
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from src.utils import (RESULTS_ROOT, PATCHES_ROOT, PRE_ROOT, load_sequence,
                       load_task_sequences, load_json, save_json)

OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")


def get_backbone(device):
    backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity()  # features only
    backbone.eval().to(device)
    return backbone


@torch.no_grad()
def region_mean_features(regions: list, device, features="post", max_patches=200):
    """Per-region mean embedding.
    post: train-split (post-disaster) RGB — the task-relevant damage domain.
    pre : pre-disaster s2-RGB cache (alternate; geography-dominated).
    """
    backbone = get_backbone(device)
    means = {}
    for region in regions:
        if features == "pre":
            path = os.path.join(PRE_ROOT, f"{region}.pt")
        else:
            path = os.path.join(PATCHES_ROOT, region, "train.pt")
        if not os.path.exists(path):
            path = os.path.join(PATCHES_ROOT, region, "test.pt")
        if not os.path.exists(path):
            print(f"[dist] WARNING: no features for {region}; skipping")
            continue
        entries = torch.load(path, map_location="cpu", weights_only=False)[:max_patches]
        feats = [backbone(e["img"].unsqueeze(0).to(device)).cpu().flatten() for e in entries]
        means[region] = torch.stack(feats).mean(0)
        print(f"[dist] {region}: mean norm={means[region].norm():.2f} "
              f"({features}, {len(entries)} tiles)")
    return means


def cosine_distance(a, b):
    return float(1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0))


def pairwise_distances(means, regions):
    D = np.zeros((len(regions), len(regions)))
    for i, ri in enumerate(regions):
        for j, rj in enumerate(regions):
            D[i, j] = cosine_distance(means[ri], means[rj])
    return D


def per_pair_forgetting(matrix):
    """[(i, j, f)] pairs: forgetting f on earlier region j by the time later
    task i is trained."""
    matrix = np.asarray(matrix, dtype=float)
    T, R = matrix.shape
    pairs = []
    for j in range(R):
        for i in range(j + 1, T):
            seen = [matrix[k][j] for k in range(j, i + 1) if not np.isnan(matrix[k][j])]
            if len(seen) < 2:
                continue
            pairs.append((i, j, max(seen) - seen[-1]))
    return pairs


def plot_heatmap(D, regions, seq_name):
    os.makedirs(OUT_PNG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(D, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(regions)))
    ax.set_yticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(regions, fontsize=8)
    for i in range(len(regions)):
        for j in range(len(regions)):
            ax.text(j, i, f"{D[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="cosine distance")
    ax.set_title(f"Pre-disaster domain distance ({seq_name})")
    fig.tight_layout()
    out = os.path.join(OUT_PNG, f"domain_distance_{seq_name}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[dist] heatmap -> {out}")


def plot_forgetting_vs_distance(D, regions, matrices, seq_name):
    os.makedirs(OUT_PNG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for mode, matrix in matrices.items():
        xs, ys = [], []
        for i, j, f in per_pair_forgetting(matrix):
            xs.append(D[i, j])
            ys.append(f)
        xs, ys = np.asarray(xs), np.asarray(ys)
        ax.scatter(xs, ys, marker="o", label=mode, s=40)
        if len(xs) > 2 and xs.std() > 1e-9 and ys.std() > 1e-9:
            m, b = np.polyfit(xs, ys, 1)
            r = np.corrcoef(xs, ys)[0, 1]
            xl = np.linspace(xs.min(), xs.max(), 10)
            ax.plot(xl, m * xl + b, ls="--", alpha=0.6)
            ax.annotate(f"{mode}: r={r:.2f}, slope={m:.2f}",
                        xy=(0.02, 0.95 - 0.09 * list(matrices).index(mode)),
                        xycoords="axes fraction", fontsize=8)
    pairing = ", ".join(f"{regions[j]}↔{regions[i]}" for i, j, _ in
                        per_pair_forgetting(next(iter(matrices.values()))))
    ax.set_xlabel("domain distance (cosine, pretrained ResNet-18 mean pre features)")
    ax.set_ylabel("forgetting on earlier region (acc. drop)")
    ax.set_title(f"Forgetting severity vs domain distance — {seq_name}\n{pairing}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_PNG, f"domain_distance_forgetting_{seq_name}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[dist] scatter -> {out}")


def run_sequence(seq_name, device, features="post"):
    regions = load_sequence(seq_name)
    print(f"\n[dist] ==== {seq_name}: {regions} (features={features}) ====")
    means = region_mean_features(regions, device, features=features)
    if len(means) < 2:
        print(f"[dist] not enough regions for distances; skipping {seq_name}")
        return
    D = pairwise_distances(means, regions)

    np.save(os.path.join(RESULTS_ROOT, f"{seq_name}_domain_distance.npy"), D)
    dist_dict = {ri: {rj: float(D[i, j]) for j, rj in enumerate(regions)}
                 for i, ri in enumerate(regions)}
    save_json(dist_dict, os.path.join(RESULTS_ROOT, f"{seq_name}_domain_distances.json"))

    print("[dist] pairwise cosine distances:")
    print("        " + "  ".join(f"{r[:8]:>9}" for r in regions))
    for i, ri in enumerate(regions):
        print(f"{ri[:8]:>9} " + "  ".join(f"{100*D[i, j]:6.1f}%" for j in range(len(regions))))

    plot_heatmap(D, regions, seq_name)

    matrices = {}
    for mode in ["baseline", "naive", "joint", "cl", "cl_adaptive"]:
        path = os.path.join(RESULTS_ROOT, f"{seq_name}_{mode}_matrix.json")
        if os.path.exists(path):
            matrices[mode] = load_json(path)["matrix"]
    if matrices:
        plot_forgetting_vs_distance(D, regions, matrices, seq_name)
    else:
        print("[dist] no matrices yet; scatter postponed until after training")


def main():
    ap = argparse.ArgumentParser(description="Domain-distance analysis.")
    ap.add_argument("--sequence", default="similar_domain",
                    help="configs/task_sequences.json key, or 'all' for every sequence.")
    ap.add_argument("--features", choices=["post", "pre"], default="post",
                    help="post-disaster train RGB (default) or pre-disaster s2-RGB cache.")
    args = ap.parse_args()

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(PATCHES_ROOT, exist_ok=True)
    os.makedirs(PRE_ROOT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.sequence == "all":
        for seq_name in load_task_sequences():
            run_sequence(seq_name, device, features=args.features)
    else:
        run_sequence(args.sequence, device, features=args.features)


if __name__ == "__main__":
    main()
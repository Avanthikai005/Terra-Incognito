# Terra Incognita — Domain-distance ablation ("originality anchor").
#
# 1. Region-wise mean feature vector: forward cached patches (post-disaster,
#    used as a cheap proxy for a region's visual domain) through the PRETRAINED
#    ResNet-18 backbone (avgpool, no fc) and average the embeddings.
# 2. Pairwise cosine distance between region mean vectors.
# 3. For each method, plot per-pair forgetting on the earlier region vs the
#    domain distance to the later region (one point per earlier/later region
#    pair), fit a line, and report Pearson r.
#
# TODO(team): for the real xBD run, recompute mean features on PRE-disaster
# tiles (pre_t2, pre_t3) to measure catastrophic-distribution distance in the
# deployed domain — the post-patch proxy slightly overestimates damage-domain
# difference. Swap by pointing REGION_FEATURES at xbd images when available.
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from src.utils import PATCHES_ROOT, RESULTS_ROOT, average_forgetting, load_json

OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
REGIONS = ["hurricane-michael", "palu", "santa-rosa-fire"]

PRETRAINED_WTS = dict(pretrained=True)


@torch.no_grad()
def region_mean_features(device="cpu", max_patches=100):
    """Per-region mean embedding from the pretrained backbone."""
    backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity()  # keep features only
    backbone.eval().to(device)

    means = {}
    for region in REGIONS:
        path = os.path.join(PATCHES_ROOT, region, "test.pt")
        entries = torch.load(path, map_location="cpu", weights_only=False)[:max_patches]
        feats = []
        for e in entries:
            img = e["img"].unsqueeze(0).to(device)
            feats.append(backbone(img).cpu().flatten())
        means[region] = torch.stack(feats).mean(0)
        print(f"[dist] {region}: mean feature norm={means[region].norm():.2f}")
    return means


def cosine_distance(a, b):
    return float(1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0))


def pairwise_distances(means):
    D = np.zeros((len(REGIONS), len(REGIONS)))
    for i, ri in enumerate(REGIONS):
        for j, rj in enumerate(REGIONS):
            D[i, j] = cosine_distance(means[ri], means[rj])
    return D


def per_pair_forgetting(matrix):
    """[(d_ij, forgetting_on_j that accumulates by the time region i is learned)]."""
    matrix = np.asarray(matrix, dtype=float)
    T, R = matrix.shape
    pairs = []
    for j in range(R):                      # earlier region j
        for i in range(j + 1, T):           # later region i
            seen = [matrix[k][j] for k in range(j, i + 1) if not np.isnan(matrix[k][j])]
            if len(seen) < 2:
                continue
            f = max(seen) - matrix[-1][j]   # drop vs final accuracy
            pairs.append((i, j, f))
    return pairs


def plot_forgetting_vs_distance(D, matrices, means):
    os.makedirs(OUT_PNG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for mode, matrix in matrices.items():
        xs, ys = [], []
        for i, j, f in per_pair_forgetting(matrix):
            xs.append(D[i, j])
            ys.append(f)
        xs, ys = np.asarray(xs), np.asarray(ys)
        ax.scatter(xs, ys, marker="o", label=mode)
        if len(xs) > 2:
            m, b = np.polyfit(xs, ys, 1)
            r = np.corrcoef(xs, ys)[0, 1]
            xl = np.linspace(xs.min(), xs.max(), 10)
            ax.plot(xl, m * xl + b, ls="--", alpha=0.6)
            ax.annotate(f"{mode}: r = {r:.2f}, slope={m:.2f}",
                        xy=(0.02, 0.5 - 0.08 * list(matrices).index(mode)), xycoords="axes fraction",
                        fontsize=8)
    pairing = ", ".join(f"{REGIONS[j]}↔{REGIONS[i]}" for i, j in per_pair_forgetting(matrix))
    ax.set_xlabel("domain distance (cosine, pretrained ResNet-18 mean features)")
    ax.set_ylabel("forgetting on earlier region (acc. drop)")
    ax.set_title(f"Forgetting severity vs domain distance\n{pairing}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_PNG, "domain_distance.png"), dpi=120)
    plt.close(fig)
    print(f"[dist] plot -> plots/domain_distance.png")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    means = region_mean_features(device)
    D = pairwise_distances(means)
    np.save(os.path.join(RESULTS_ROOT, "region_distance_matrix.npy"), D)
    print("[dist] pairwise cosine distances:")
    print("        " + "  ".join(f"{r[:8]:>9}" for r in REGIONS))
    for i, ri in enumerate(REGIONS):
        print(f"{ri[:8]:>9} " + "  ".join(f"{100*D[i, j]:6.1f}%" for j in range(len(REGIONS))))

    matrices = {}
    for mode in ["naive", "cl"]:
        path = os.path.join(RESULTS_ROOT, f"{mode}_matrix.json")
        if os.path.exists(path):
            matrices[mode] = load_json(path)["matrix"]

    if matrices:
        plot_forgetting_vs_distance(D, matrices, means)
    else:
        print("[dist] WARNING: no matrices found; run src/train.py first")


if __name__ == "__main__":
    main()
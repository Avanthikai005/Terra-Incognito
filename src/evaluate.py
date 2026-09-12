# Terra Incognita — offline evaluation.
#   * loads results/<mode>_matrix.json for naive/joint/cl
#   * prints accuracy matrices + per-method average forgetting
#   * writes report/results.md (consumed by report/report.md)
#   * renders a qualitative demo grid: early-region patches, pred vs true,
#     using the FINAL cl checkpoint (shows retained knowledge).
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model import make_model
from src.utils import RESULTS_ROOT, PATCHES_ROOT, average_forgetting, load_json

MODES = ["naive", "joint", "cl"]
OUT_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "results.md")
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")


def cell(v):
    return "  -  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100*v:.1f}%"


def fmt_row(row):
    return "| " + " | ".join(cell(v) for v in row) + " |"


def make_matrix_table(mode, matrix, regions, forgetting):
    header = "| after task | " + " | ".join(regions) + f" | avg forgetting |\n"
    sep = "|---|---" * (len(regions) + 2) + "|\n"
    lines = [header, sep]
    for i, row in enumerate(matrix):
        lines.append(f"| task {i+1} | " + " | ".join(cell(v) for v in row) +
                     f" | {forgetting:.2%} |\n")
    return "".join(lines)


@torch.no_grad()
def qualitative_demo(checkpoint_path, region, out_png, k=8, device="cpu"):
    """Grid of test patches from an EARLY region labelled by the final model."""
    from src.utils import PatchDataset
    if not os.path.exists(checkpoint_path):
        print("[eval] no cl checkpoint; skipping qualitative demo")
        return
    model = make_model(pretrained=True, device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    entries = torch.load(os.path.join(PATCHES_ROOT, region, "test.pt"),
                         map_location="cpu", weights_only=False)
    import random
    random.seed(42)
    random.shuffle(entries)
    entries = entries[:k]

    fig, axes = plt.subplots(2, k // 2, figsize=(2.2 * k, 4.6))
    axes = np.atleast_2d(axes).ravel()
    names = {0: "undamaged", 1: "damaged", 2: "destroyed"}
    for ax, e in zip(axes, entries):
        img = e["img"].unsqueeze(0).to(device)
        logit = model(img)
        pred = int(logit.argmax(1))
        true = e["label"]
        ax.imshow(e["img"].permute(1, 2, 0))
        ok = "✓" if pred == true else "✗"
        ax.set_title(f"pred={names[pred]}\ntrue={names[true]} {ok}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Qualitative demo — {region} test patches, FINAL {os.path.basename(checkpoint_path)} model", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[eval] demo grid -> {out_png}")


def main():
    os.makedirs(OUT_PNG, exist_ok=True)
    summaries = []
    for mode in MODES:
        path = os.path.join(RESULTS_ROOT, f"{mode}_matrix.json")
        if not os.path.exists(path):
            print(f"[eval] WARNING: missing {path} (run src/train.py first)")
            continue
        data = load_json(path)
        matrix, regions = data["matrix"], data["regions"]
        forgetting = average_forgetting(matrix)
        summaries.append((mode, matrix, regions, forgetting))
        print(f"\n=== {mode.upper()} === accuracy matrix (cols: regions in task order)")
        print("| after task | " + " | ".join(regions) + " |")
        for i, row in enumerate(matrix):
            print(f"| task {i+1} | " + " | ".join(cell(v) for v in row) + " |")
        print(f"average forgetting: {forgetting:.2%}")

    if summaries:
        md = ["# Results\n", 
              "- Forgetting := (best earlier accuracy on a region) − (final accuracy on that region).\n\n"]
        for mode, matrix, regions, forgetting in summaries:
            md.append(f"## {mode.upper()}\n")
            md.append(make_matrix_table(mode, matrix, regions, forgetting))
            md.append("\n")
        os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
        with open(OUT_MD, "w") as f:
            f.writelines(md)
        print(f"\n[eval] wrote report/results.md")

    # qualitative demo: early region, final cl model
    qualitative_demo(os.path.join(RESULTS_ROOT, "ckpt_cl_task2.pt"), "hurricane-michael",
                     os.path.join(OUT_PNG, "qualitative_demo.png"))


if __name__ == "__main__":
    main()
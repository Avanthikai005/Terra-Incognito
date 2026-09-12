# Terra Incognita — offline evaluation.
#   * loads results/<seq>_<mode>_matrix.json for each sequence × mode
#   * prints accuracy matrices + per-method average forgetting
#   * writes per-sequence report/results_<seq>.md and a combined
#     report/benchmark_results.md (methods × sequences forgetting table)
#   * qualitative demo grid of the FIRST region's test patches under the final
#     cl checkpoint of that sequence.
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model import make_model
from src.utils import (RESULTS_ROOT, PATCHES_ROOT, load_task_sequences,
                       average_forgetting, load_json)

MODES = ["baseline", "naive", "joint", "cl", "cl_adaptive"]
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report")
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")


def cell(v):
    return "  -  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100*v:.1f}%"


def make_matrix_table(matrix, regions, forgetting):
    header = "| after task | " + " | ".join(regions) + " | avg forgetting |\n"
    sep = "|---|---" * (len(regions) + 2) + "|\n"
    lines = [header, sep]
    for i, row in enumerate(matrix):
        fcell = f"{forgetting:.2%}" if forgetting is not None else "n/a"
        lines.append(f"| task {i+1} | " + " | ".join(cell(v) for v in row) +
                     f" | {fcell} |\n")
    return "".join(lines)


@torch.no_grad()
def qualitative_demo(checkpoint_path, region, out_png, k=8, device="cpu"):
    """Grid of test patches from the FIRST region labelled by the final model."""
    if not os.path.exists(checkpoint_path):
        print(f"[eval] no checkpoint {checkpoint_path}; skipping qualitative demo")
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
        pred = int(model(img).argmax(1))
        true = e["label"]
        ax.imshow(e["img"].permute(1, 2, 0))
        ok = "✓" if pred == true else "✗"
        ax.set_title(f"pred={names[pred]}\ntrue={names[true]} {ok}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Qualitative demo — {region} test, {os.path.basename(checkpoint_path)}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[eval] demo grid -> {out_png}")


def evaluate_sequence(seq_name, regions, device):
    """Load matrices for one sequence; print + return (mode, matrix, forgetting)."""
    summaries = []
    for mode in MODES:
        path = os.path.join(RESULTS_ROOT, f"{seq_name}_{mode}_matrix.json")
        if not os.path.exists(path):
            print(f"[eval] WARNING: missing {path} (run src/train.py first)")
            continue
        data = load_json(path)
        matrix = data["matrix"]
        forgetting = None if mode == "baseline" else average_forgetting(matrix)
        summaries.append((mode, matrix, forgetting))
        print(f"\n=== {seq_name.upper()} / {mode.upper()} === (cols: regions in task order)")
        print("| after task | " + " | ".join(regions) + " |")
        for i, row in enumerate(matrix):
            print(f"| task {i+1} | " + " | ".join(cell(v) for v in row) + " |")
        if forgetting is None:
            print("(independent per-region models: diagonal = per-region ceiling)")
        else:
            print(f"average forgetting: {forgetting:.2%}")
    return summaries


def main():
    os.makedirs(OUT_PNG, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    seqs = load_task_sequences()
    all_summaries = {}

    for seq_name, seq_info in seqs.items():
        regions = seq_info["regions"]
        summaries = evaluate_sequence(seq_name, regions, "cpu")
        all_summaries[seq_name] = summaries

        if summaries:
            md = [f"# Results — {seq_name}\n\n",
                  "- Forgetting := (best earlier accuracy on a region) − "
                  "(final accuracy on that region).\n\n"]
            for mode, matrix, forgetting in summaries:
                md.append(f"## {mode.upper()}\n")
                md.append(make_matrix_table(matrix, regions, forgetting))
                md.append("\n")
            with open(os.path.join(REPORT_DIR, f"results_{seq_name}.md"), "w") as f:
                f.writelines(md)

            ckpt = os.path.join(RESULTS_ROOT,
                                f"ckpt_{seq_name}_cl_task{len(regions)-1}.pt")
            qualitative_demo(ckpt, regions[0],
                             os.path.join(OUT_PNG, f"qualitative_demo_{seq_name}.png"))

    # combined benchmark summary: methods × sequences, avg forgetting
    md = ["# Benchmark Results — Terra Incognita\n\n",
          "Disaster-type-sequential continual-learning benchmark on xBD-selected.\n\n",
          "## Methods × sequences (average forgetting)\n\n",
          "| Method | " + " | ".join(seqs.keys()) + " |\n",
          "|---|" + "---|" * len(seqs) + "\n"]
    for mode in MODES:
        cells = []
        for seq_name in seqs:
            found = [s for s in all_summaries.get(seq_name, []) if s[0] == mode]
            cells.append(f"{found[0][2]:.2%}" if found and found[0][2] is not None
                         else ("n/a" if found else "  -  "))
        md.append(f"| {mode} | " + " | ".join(cells) + " |\n")

    md.append("\n## Per-sequence matrices\n\n")
    for seq_name in seqs:
        md.append(f"### {seq_name} — "
                  f"{', '.join(seqs[seq_name]['regions'])}\n\n"
                  f"*{seqs[seq_name]['description']}*\n\n")
        for mode, matrix, forgetting in all_summaries.get(seq_name, []):
            md.append(f"#### {mode.upper()}\n")
            md.append(make_matrix_table(matrix, seqs[seq_name]["regions"], forgetting))
            md.append("\n")

    with open(os.path.join(REPORT_DIR, "benchmark_results.md"), "w") as f:
        f.writelines(md)
    print(f"\n[eval] wrote report/benchmark_results.md")


if __name__ == "__main__":
    main()
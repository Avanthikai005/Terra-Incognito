#!/usr/bin/env python3
"""Generate placeholder demo assets for the Terra Incognita frontend.

Writes into frontend/public/:
  assets/overview.png                 three-step schematic (Overview page)
  assets/domain_distance.png          scatter placeholder (Domain Distance page)
  assets/qualitative/r{N}_{i}.png     synthetic building patches (Qualitative page)
  data/qualitative.json               pred-vs-true labels for those patches

REAL PIPELINE: after `bash run.sh` in the repo root, the real plots can replace
these (node tools/export-results.mjs already copies plots/domain_distance.png).
Qualitative grids can be re-rendered from results/ checkpoints with
src/evaluate.py::qualitative_demo.

Requires: numpy, PIL, matplotlib (all present in the repo venv/).
"""
import json
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(HERE, "..", "public")
ASSETS = os.path.join(PUBLIC, "assets")
QA = os.path.join(ASSETS, "qualitative")
DATA = os.path.join(PUBLIC, "data")

REGIONS = ["hurricane-michael", "palu", "santa-rosa-fire"]

# ---------------------------------------------------------------------------
# Qualitative labels (must mirror src/data/qualitative.ts fallback)
# ---------------------------------------------------------------------------
LABELS = {
    "hurricane-michael": [
        ("damaged", {"naive": "destroyed", "joint": "damaged", "cl": "damaged"}),
        ("undamaged", {"naive": "damaged", "joint": "undamaged", "cl": "undamaged"}),
        ("destroyed", {"naive": "undamaged", "joint": "destroyed", "cl": "damaged"}),
        ("destroyed", {"naive": "damaged", "joint": "destroyed", "cl": "destroyed"}),
        ("damaged", {"naive": "destroyed", "joint": "damaged", "cl": "damaged"}),
        ("undamaged", {"naive": "undamaged", "joint": "undamaged", "cl": "undamaged"}),
        ("damaged", {"naive": "destroyed", "joint": "damaged", "cl": "damaged"}),
        ("destroyed", {"naive": "undamaged", "joint": "destroyed", "cl": "destroyed"}),
    ],
    "palu": [
        ("damaged", {"naive": "damaged", "joint": "damaged", "cl": "damaged"}),
        ("destroyed", {"naive": "damaged", "joint": "destroyed", "cl": "destroyed"}),
        ("undamaged", {"naive": "damaged", "joint": "undamaged", "cl": "undamaged"}),
        ("damaged", {"naive": "damaged", "joint": "damaged", "cl": "damaged"}),
        ("undamaged", {"naive": "undamaged", "joint": "undamaged", "cl": "undamaged"}),
        ("destroyed", {"naive": "destroyed", "joint": "destroyed", "cl": "damaged"}),
        ("damaged", {"naive": "destroyed", "joint": "damaged", "cl": "damaged"}),
        ("undamaged", {"naive": "damaged", "joint": "undamaged", "cl": "undamaged"}),
    ],
    "santa-rosa-fire": [
        ("damaged", {"naive": "damaged", "joint": "damaged", "cl": "damaged"}),
        ("undamaged", {"naive": "undamaged", "joint": "undamaged", "cl": "undamaged"}),
        ("destroyed", {"naive": "destroyed", "joint": "destroyed", "cl": "destroyed"}),
        ("damaged", {"naive": "damaged", "joint": "damaged", "cl": "damaged"}),
        ("undamaged", {"naive": "undamaged", "joint": "undamaged", "cl": "undamaged"}),
        ("destroyed", {"naive": "destroyed", "joint": "damaged", "cl": "destroyed"}),
        ("damaged", {"naive": "damaged", "joint": "damaged", "cl": "damaged"}),
        ("undamaged", {"naive": "undamaged", "joint": "undamaged", "cl": "undamaged"}),
    ],
}


def _roof(roof_center, w, h, angle_deg):
    """Rotate a rectangle center+half-size around its center; polygon points."""
    import math

    cx, cy = roof_center
    a = math.radians(angle_deg)
    cos, sin = math.cos(a), math.sin(a)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [
        (cx + x * cos - y * sin, cy + x * sin + y * cos) for x, y in pts
    ]


def _make_tile(region_idx: int, idx: int, label: str):
    """240x240 synthetic aerial building patch."""

    def import_pil():
        from PIL import Image, ImageDraw, ImageFilter
        return Image, ImageDraw, ImageFilter

    Image, ImageDraw, ImageFilter = import_pil()

    rng = random.Random(region_idx * 131 + idx * 17 + 7)
    S = 240

    # --- region-tinted low-frequency background -------------------------
    tints = [
        (122, 133, 148),   # hurricane: slate urban
        (168, 141, 103),   # palu: sandy coast
        (122, 84, 60),     # wildfire: scorched earth
    ]
    tint = tints[region_idx]
    noise = np.random.RandomState(region_idx * 101 + idx * 13 + 3)
    big = noise.randint(0, 60, (12, 12)).astype(np.uint8)
    im = Image.fromarray(big, "L").resize((S, S), Image.BILINEAR)
    colorized = Image.new("RGB", (S, S), tint)
    im = Image.composite(im, colorized, im.point(lambda v: 120))  # brightness ramp
    base = Image.new("RGB", (S, S))
    base.paste(colorized)
    # blend noise over tint to keep textures subtle
    overlay = Image.new("RGB", (S, S), (40, 44, 52))
    base = Image.blend(base, overlay, 0.12)

    dr = ImageDraw.Draw(base, "RGBA")

    # --- ground texture: roads / bare soil strips -----------------------
    for _ in range(2):
        ry = rng.randint(10, S - 30)
        dr.rectangle(
            [rng.randint(0, 20), ry, S - rng.randint(0, 20), ry + rng.randint(4, 8)],
            fill=(rng.randint(110, 160), rng.randint(105, 150), rng.randint(90, 135), 110),
        )

    # --- shadow ----------------------------------------------------------
    bx, by = S / 2 + rng.randint(-26, 26), S / 2 + rng.randint(-22, 22)
    if label == "destroyed":
        bx, by = S / 2 + rng.randint(-10, 10), S / 2 + rng.randint(-10, 10)
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sdr = ImageDraw.Draw(shadow)
    sdr.ellipse([bx - 58, by - 52, bx + 72, by + 62], fill=(20, 16, 12, 70))
    base = Image.alpha_composite(base.convert("RGBA"), shadow)
    dr = ImageDraw.Draw(base, "RGBA")

    # --- the building ------------------------------------------------------
    roof_color = {
        "undamaged": (221, 222, 226, 255),
        "damaged": (160, 165, 172, 255),
        "destroyed": (88, 82, 76, 255),
    }[label]
    ridge = {
        "undamaged": (114, 118, 128, 255),
        "damaged": (80, 82, 88, 255),
        "destroyed": (52, 48, 44, 255),
    }[label]

    if label == "destroyed":
        # rubble: a cluster of dark broken blocks + charred patches
        for _ in range(rng.randint(4, 7)):
            rx, ry = bx + rng.randint(-34, 34), by + rng.randint(-30, 30)
            dr.polygon(
                _roof((rx, ry), rng.randint(14, 30), rng.randint(10, 22),
                      rng.randint(0, 60)),
                fill=(rng.randint(70, 95), rng.randint(62, 84), rng.randint(54, 72), 255),
            )
        for _ in range(24):
            px, py = bx + rng.randint(-40, 40), by + rng.randint(-36, 36)
            dr.rectangle([px, py, px + rng.randint(2, 5), py + rng.randint(1, 4)],
                         fill=(70, 64, 58, 200))
        dr.polygon(_roof((bx, by), 70, 60, rng.randint(-6, 6)), fill=roof_color)
    else:
        dr.polygon(_roof((bx, by), 96, 78, rng.randint(-6, 6)), fill=roof_color)
        # roof ridge line
        a = rng.uniform(-6, 6)
        import math
        rad = math.radians(a)
        L = 88
        cxr, cyr = bx, by - 8
        dr.line(
            [
                (cxr - L / 2 * math.cos(rad), cyr - L / 2 * math.sin(rad)),
                (cxr + L / 2 * math.cos(rad), cyr + L / 2 * math.sin(rad)),
            ],
            fill=ridge, width=rng.randint(4, 6),
        )

        if label == "damaged":
            # visible crack + debris
            crack = [(bx - 40, by + 22), (bx - 8, by - 6),
                     (bx + 14, by + 30), (bx + 34, by + 4)]
            dr.line(crack, fill=(60, 58, 60, 230), width=4)
            for _ in range(14):
                px, py = bx + rng.randint(-46, 46), by + rng.randint(-38, 38)
                dr.rectangle([px, py, px + rng.randint(3, 7), py + rng.randint(2, 5)],
                             fill=(110, 112, 118, 190))
        else:
            # neat yard: shrubs around an intact roof
            for _ in range(6):
                sx, sy = bx + rng.randint(-70, 70), by + rng.randint(-70, 70)
                if abs(sx - bx) < 55 and abs(sy - by) < 48:
                    continue
                dr.ellipse([sx, sy, sx + rng.randint(7, 13), sy + rng.randint(7, 13)],
                           fill=(rng.randint(70, 110), rng.randint(120, 155),
                                 rng.randint(60, 95), 220))

    # --- vignette ----------------------------------------------------------
    vig = Image.new("L", (S, S), 0)
    vd = ImageDraw.Draw(vig)
    vd.ellipse([-80, -80, S + 80, S + 80], fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(30))
    dark = Image.new("RGBA", (S, S), (10, 10, 14, 150))
    base = Image.composite(base, dark, vig)
    return base.convert("RGB")


def make_qualitative():
    os.makedirs(QA, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    out = {}
    for r, region in enumerate(REGIONS):
        entries = []
        for i, (true, pred) in enumerate(LABELS[region]):
            name = f"r{r + 1}_{i}.png"
            path = os.path.join(QA, name)
            _make_tile(r, i, true).save(path)
            entries.append({"image": f"/assets/qualitative/{name}", "true": true, "pred": pred})
        out[region] = entries
    with open(os.path.join(DATA, "qualitative.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[assets] wrote {len(REGIONS) * 8} qualitative tiles + data/qualitative.json")


def make_overview(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13.2, 4.4), dpi=140)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [  # (x0, title, sub, r1, r2, note)
        (0.3, "Step 1 · Learn Region A", "train on Hurricane", 78, None, None),
        (4.15, "Step 2 · Fine-tune on Region B", "naive sequential update", 55, 75, ("A drops 78 → 55", "bad")),
        (8.0, "Replay (ours)", "keeps A while learning B", 70, 75, ("A stays 70%", "good")),
    ]
    colors = {"good": "#16a34a", "bad": "#e11d48", "seen": "#0f766e"}

    for (x0, title, sub, r1, r2, note) in boxes:
        box = FancyBboxPatch((x0, 0.32), 3.35, 3.4,
                             boxstyle="round,pad=0.05,rounding_size=0.16",
                             fc="#ffffff", ec="#e7e2d8", lw=1.2)
        ax.add_patch(box)
        ax.text(x0 + 1.67, 3.5, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="#0f172a")
        ax.text(x0 + 1.67, 3.12, sub, ha="center", va="center",
                fontsize=8.6, color="#64748b")

        ybase = 1.75  # bar baseline
        for col, rid, val in [("A", "R1 Hurricane", r1), ("B", "R2 Tsunami", r2)]:
            xc = x0 + (0.95 if col == "A" else 2.4)
            if val is None:
                ax.bar(xc, 2.0, width=0.62, bottom=0.55, color="#f1f5f9",
                       edgecolor="#e2e8f0", linewidth=1.2)
                ax.text(xc, 0.4, rid, ha="center", fontsize=7.5, color="#94a3b8")
                ax.text(xc, 2.7, "—", ha="center", fontsize=10, color="#cbd5e1")
            else:
                # Region A bar: green if retained, red if it collapsed.
                color = colors["good"] if col == "A" and val >= 70 else colors["bad"] if col == "A" else colors["seen"]
                ax.bar(xc, val / 100 * 2.0, width=0.62, bottom=0.55, color=color,
                       edgecolor="none", alpha=0.94)
                ax.text(xc, 2.7, f"{val}%", ha="center", fontsize=10, fontweight="bold",
                        color=color)
                ax.text(xc, 0.4, rid, ha="center", fontsize=7.5, color="#475569")

        if note:
            text, kind = note
            ax.text(x0 + 1.67, 2.92, text, ha="center", fontsize=9.5,
                    fontweight="bold", color=colors[kind])

    # arrows between panels
    for xa, xb in [(3.72, 4.05), (7.57, 7.9)]:
        ax.annotate("", xy=(xb, 2.5), xytext=(xa, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.6,
                                    mutation_scale=16))
        ax.text((xa + xb) / 2, 2.95, "next\ndisaster", ha="center", va="center",
                fontsize=7.5, color="#94a3b8")

    ax.text(0.3, 3.86, "Accuracy on each learned region", fontsize=8.2,
            color="#94a3b8")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[assets] wrote {path}")


def make_distance_plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [0.35, 0.18, 0.24]  # later/earlier region-pair cosine distances
    naive = [23, 23, 7]
    replay = [8, 8, 2]

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=140)
    for xs, ys, name, color, marker in [
        (x, naive, "Naive fine-tuning", "#e11d48", "o"),
        (x, replay, "Replay (ours)", "#16a34a", "^"),
    ]:
        ax.scatter(xs, ys, s=64, color=color, zorder=3, edgecolor="white",
                   linewidth=0.8, marker=marker, label=name)
        m, b = np.polyfit(xs, ys, 1)
        r = np.corrcoef(xs, ys)[0, 1]
        xl = np.linspace(min(xs) - 0.02, max(xs) + 0.02, 10)
        ax.plot(xl, m * xl + b, ls="--", color=color, alpha=0.55, lw=1.4)
        ax.annotate(f"{name}: r = {r:.2f}, slope {m:.2f}",
                    xy=(0.02, 0.9 - 0.08 * (xs is x)), xycoords="axes fraction",
                    fontsize=9, color=color, fontweight="bold")
    ax.set_xlabel("domain distance (cosine, pretrained ResNet-18 features)")
    ax.set_ylabel("forgetting on earlier region (acc. drop, pts)")
    ax.set_title("Forgetting severity vs. inter-region domain distance",
                 fontsize=12, fontweight="bold", color="#0f172a")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"[assets] wrote {path}")


def main():
    os.makedirs(ASSETS, exist_ok=True)
    make_overview(os.path.join(ASSETS, "overview.png"))
    make_distance_plot(os.path.join(ASSETS, "domain_distance.png"))
    make_qualitative()


if __name__ == "__main__":
    main()
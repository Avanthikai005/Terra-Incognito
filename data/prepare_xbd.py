# Terra Incognita — data preparation.
#
# Turns raw xBD (xView2) tiles + label JSONs into a per-building patch cache:
#   data/patches/<region>/{train,val,test}.pt
#
# Real-xBD path:
#   xbd_root/<disaster>/images/<tile>_pre_disaster.png  + post
#   xbd_root/<disaster>/labels/<tile>_post_disaster.json
#   JSON: features[] -> {feature_type: "building", properties: {damage, uid}}
#   -> crop building bbox (padded) from the POST image, resize 224, classify.
#
# Synthetic fallback (no xBD downloaded): generates learnable patches with a
# per-region "visual signature" (hue) that shifts between disasters. This lets
# the full pipeline run end-to-end and demonstrates catastrophic forgetting +
# replay on a toy problem. Classes are derived deterministically from the
# per-region signature histogram so all three regimes produce a real matrix.

import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES_ROOT = os.path.join(ROOT, "data", "patches")
IMG_SIZE = 224
PAD = 15

# xView2 4-class -> our 3-class scheme.
DAMAGE_MAP = {
    "no-damage": 0,       # undamaged
    "minor-damage": 1,    # damaged
    "major-damage": 1,    # damaged
    "destroyed": 2,       # destroyed
    "un-classified": None,
}


# ---------------------------------------------------------------------------
# Synthetic fallback (offline-proof pipeline)
# ---------------------------------------------------------------------------

def _region_hue(region: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + abs(hash(region)) % 1000)
    return rng.uniform(0, 1, size=3)


def _synthetic_region(region: str, n: int, seed: int) -> list:
    """n patches; label = bin of the patch's dominant intensity per channel."""
    rng = np.random.default_rng(seed + len(region))
    hue = _region_hue(region, seed)
    entries = []
    for i in range(n):
        base = np.random.uniform(0.15, 0.5, size=(IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
        tint = (np.random.rand(3) * 0.5 + 0.5) * hue * 1.5
        img = np.clip(base * tint[None, None, :], 0, 1).astype(np.float32)
        # label <- quantised "brightness signature" of the patch
        sig = img.mean(axis=(0, 1))                     # 3-vector, ~ region hue
        label = int((sig.sum() % 1) * 3)                # deterministic per patch
        entries.append({"img": torch.from_numpy(img.transpose(2, 0, 1)).clone(),
                        "label": label, "tile": f"{region}_syn_t{i}"})
    return entries


# ---------------------------------------------------------------------------
# Real xBD: polygon -> patch crop
# ---------------------------------------------------------------------------

def _load_real_region(disaster_dir: str, seed: int, max_buildings: int) -> list:
    img_dir = os.path.join(disaster_dir, "images")
    lbl_dir = os.path.join(disaster_dir, "labels")
    post_files = sorted(glob.glob(os.path.join(img_dir, "*_post_disaster*.png")))
    if not post_files:
        return []

    rng = np.random.RandomState(seed)
    rng.shuffle(post_files)
    entries, count = [], 0
    for post in post_files:
        label = os.path.join(lbl_dir, os.path.basename(post).replace("_post_disaster", "_post_disaster"))
        label = label.replace(".png", ".json")
        if not os.path.exists(label):
            continue
        tile = os.path.basename(post)
        with open(label) as f:
            ann = json.load(f)
        img_post = Image.open(post).convert("RGB")
        for feat in ann.get("features", []):
            if feat.get("feature_type") != "building":
                continue
            dmg = feat.get("properties", {}).get("damage")
            cls = DAMAGE_MAP.get(dmg)
            if cls is None:
                continue
            poly = feat["properties"].get("polygon_xy") or feat.get("geometry", {}).get("coordinates")
            if not poly:
                continue
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            x1, y1, x2, y2 = max(0, min(xs) - PAD), max(0, min(ys) - PAD), min(img_post.width, max(xs) + PAD), min(img_post.height, max(ys) + PAD)
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue  # degenerate polygon
            crop = T.Resize((IMG_SIZE, IMG_SIZE), antialias=True)(
                T.ToTensor()(img_post.crop((int(x1), int(y1), int(x2), int(y2)))))
            entries.append({"img": crop, "label": cls, "tile": tile})
            count += 1
            if max_buildings and count >= max_buildings:
                return entries
    return entries


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def split_by_tile(entries, ratios=(0.7, 0.1, 0.2), seed=42):
    """Split by TILE (no building-level leakage)."""
    tiles = sorted({e["tile"] for e in entries})
    rng = np.random.RandomState(seed)
    rng.shuffle(tiles)
    n1, n2 = int(len(tiles) * ratios[0]), int(len(tiles) * (ratios[0] + ratios[1]))
    tile2split = {}
    for i, t in enumerate(tiles):
        tile2split[t] = "train" if i < n1 else ("val" if i < n2 else "test")
    out = {"train": [], "val": [], "test": []}
    for e in entries:
        out[tile2split[e["tile"]]].append(e)
    return out


def collate_region(disaster: str, xbd_root: str, subset: int, seed: int) -> dict:
    src = os.path.join(xbd_root, disaster)
    entries = _load_real_region(src, seed, subset) if os.path.isdir(src) else []
    if entries:
        print(f"[prep] {disaster}: {len(entries)} real building patches (xBD)")
    else:
        n = subset if subset else 400
        entries = _synthetic_region(disaster, n, seed)
        print(f"[prep] {disaster}: {len(entries)} SYNTHETIC fallback patches "
              f"(xBD not found at {src})")
    return split_by_tile(entries, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xbd-root", default=os.path.join(ROOT, "data", "xbd"))
    ap.add_argument("--regions", nargs="+", default=["hurricane-michael", "palu", "santa-rosa-fire"])
    ap.add_argument("--subset", type=int, default=0, help="max building patches per region (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for region in args.regions:
        splits = collate_region(region, args.xbd_root, args.subset, args.seed)
        out_dir = os.path.join(PATCHES_ROOT, region)
        os.makedirs(out_dir, exist_ok=True)
        for split, entries in splits.items():
            torch.save(entries, os.path.join(out_dir, f"{split}.pt"))
            labels = [e["label"] for e in entries]
            counts = np.bincount(labels, minlength=3)
            print(f"    {region}/{split}: n={len(entries)}  "
                  f"undamaged={counts[0]} damaged={counts[1]} destroyed={counts[2]}")
    print("[prep] done.")


if __name__ == "__main__":
    main()
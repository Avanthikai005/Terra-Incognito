# Terra Incognita — preprocessing for the xBD "selected" download.
#
# The classic per-building polygon labels are NOT shipped in the selected
# release: only a metadata.csv / metadata.geojson with per-tile aggregate
# damage counts (N_intact, N_minor, N_major, N_destroyed) plus Sentinel
# imagery. We build a tile-level task (3-class, collapsed from xView2):
#   0 = undamaged, 1 = damaged, 2 = destroyed  (argmax over collapsed counts).
#
# What this script produces (cache contract consumed by src/ and configs/):
#
#  * similar_domain sequence: hurricane-matthew tiles split by longitude into
#    THREE spatially-separated pseudo-regions (A/B/C) — the earlier benchmark.
#      data/patches/hurricane-matthew-{A,B,C}/{train,val,test}.pt
#      data/patches/region_order.txt  (band order A->C, task order)
#      data/patches/region_map.json   (uid -> band region)
#    Imagery source: sentinel-2 true colour (s2_tci, 128x128 RGB).
#
#  * cross_disaster sequence:
#      data/patches/hurricane-matthew/{train,val,test}.pt   (full event, s2_tci RGB)
#      data/patches/palu-tsunami/{train,val,test}.pt        (s2 12-band -> RGB)
#    palu has NO s2_tci release; its RGB is derived from bands B04/B03/B02
#    of the 12-band s2 GeoTIFF (validated: r>0.99 vs s2_tci on hurricane).
#
#  * Pre-disaster RGB cache (ALTERNATE domain-distance features):
#      data/patches/_pre/<region>.pt    list of {"img": [3,128,128] 0..1, "tile": str}
#    Built from PRE s2 (12-band -> RGB) so hurricane & palu share one modality.
#    domain_distance.py defaults to POST-disaster train RGB (measured damage
#    domain, no test leakage); --features pre uses this cache instead.
#
# Each entry of a train/val/test .pt: {"img": [3,H,W] float 0..1, "label": int, "tile": str}
# Splits are BY TILE (no leakage), ratios 70/10/20, seeded (default 42).

import argparse
import csv
import glob
import json
import os
import struct

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES_ROOT = os.path.join(ROOT, "data", "patches")
PRE_ROOT = os.path.join(PATCHES_ROOT, "_pre")
N_REGIONS = 3

# xView2 4-class -> our 3-class scheme.
DAMAGE_COUNTS = ["N_intact", "N_minor", "N_major", "N_destroyed"]
DAMAGE_MAP = {0: 0, 1: 1, 2: 1, 3: 2}  # intact/minor/major/destroyed -> 3 class
SPLIT_RATIOS = (0.7, 0.1, 0.2)


# ---------------------------------------------------------------------------
# Sentinel-2 12-band GeoTIFF -> RGB (no external deps; uncompressed uint16 chunky)
# ---------------------------------------------------------------------------

def read_s2_tiff_rgb(path: str) -> torch.Tensor:
    """Read a 128x128 12-sample uint16 GeoTIFF into an [3,128,128] float tensor.

    Band layout is S2 L2A order (index 1=B02 blue, 2=B03 green, 3=B04 red), so
    the true-colour composite reads indices [3,2,1]. Values are scaled
    reflectance (~1k-7k of 0-10000); we normalise to 0..1.
    """
    d = open(path, "rb").read()
    bo = "<" if d[:2] == b"II" else ">"
    ifd = struct.unpack(bo + "I", d[4:8])[0]
    n = struct.unpack(bo + "H", d[ifd:ifd + 2])[0]
    tags = {}
    for i in range(n):
        e = d[ifd + 2 + 12 * i:ifd + 2 + 12 * (i + 1)]
        tag, typ, cnt, val = struct.unpack(bo + "HHII", e)
        tags[tag] = (typ, cnt, val)
    w, h, spp = tags[256][2], tags[257][2], tags[277][2]

    def _values(tno):
        typ, cnt, val = tags[tno]
        if cnt == 1:
            return [val]
        if typ == 4:
            return list(np.frombuffer(d[val:val + 4 * cnt], dtype=bo + "u4"))
        if typ == 3:
            return list(np.frombuffer(d[val:val + 2 * cnt], dtype=bo + "u2"))
        raise ValueError(f"unsupported TIFF tag type {typ}")

    offsets, bytecounts = _values(273), _values(279)
    dt = np.uint16
    arr = np.zeros((h, w, spp), dtype=np.uint16)
    row = 0
    nbytes = w * spp * dt().itemsize
    for off, by in zip(offsets, bytecounts):
        rows = by // nbytes
        arr[row:row + rows] = np.frombuffer(d[off:off + by], dt).reshape(rows, w, spp)
        row += rows
    # RGB <- bands [4,3,2] (1-indexed) i.e. 0-based indices [3,2,1]
    img = arr[..., [3, 2, 1]].astype(np.float32) / 10000.0
    return torch.from_numpy(np.ascontiguousarray(img).transpose(2, 0, 1)).float()


def _load_rgb(post_dir: str, uid: str, kind: str) -> torch.Tensor:
    """kind in {'tci','s2'}: returns [3,128,128] float 0..1 tensor."""
    if kind == "tci":
        path = os.path.join(post_dir, f"{uid}_post_disaster_s2_tci.tif")
        img = T.ToTensor()(Image.open(path).convert("RGB"))
    else:
        path = os.path.join(post_dir, f"{uid}_post_disaster_s2.tif")
        img = read_s2_tiff_rgb(path)
    return img


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def find_selected_root(xbd_root: str) -> str:
    if os.path.isdir(os.path.join(xbd_root, "xbd_selected")):
        return os.path.join(xbd_root, "xbd_selected")
    hits = sorted(glob.glob(os.path.join(xbd_root, "xbd_selected*", "xbd_selected")))
    return hits[0] if hits else ""


def load_tile_centroids(selected_root: str) -> dict:
    path = os.path.join(selected_root, "metadata.geojson")
    with open(path) as f:
        fc = json.load(f)
    out = {}
    for feat in fc.get("features", []):
        uid = feat.get("properties", {}).get("xbd_uid")
        geom = feat.get("geometry", {})
        if geom.get("type") == "Polygon":
            ring = geom["coordinates"][0]
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            out[uid] = (float(np.mean(xs)), float(np.mean(ys)))
    return out


def tile_label(counts: dict):
    collapsed = np.array([counts["N_intact"],
                          counts["N_minor"] + counts["N_major"],
                          counts["N_destroyed"]], dtype=float)
    if collapsed.sum() == 0:
        return None, 0
    return int(collapsed.argmax()), int(collapsed.sum())


def longitude_bands(entries: dict, k: int) -> np.ndarray:
    """Split tiles into k contiguous longitude bands of (near-)equal size."""
    uids = sorted(entries.keys())
    lon = np.array([entries[u]["coords"][0] for u in uids])
    order = np.argsort(lon)
    assign = np.empty(len(uids), dtype=int)
    n_per = np.full(k, len(uids) // k, dtype=int)
    n_per[: len(uids) % k] += 1
    start = 0
    for c, n in enumerate(n_per):
        assign[order[start:start + n]] = c
        start += n
    return assign


# ---------------------------------------------------------------------------
# Split + write
# ---------------------------------------------------------------------------

def split_by_tile(entries, ratios=SPLIT_RATIOS, seed=42):
    n_tiles = len(entries)
    if n_tiles < 4:
        return {"train": entries, "val": [], "test": []}
    tiles = sorted(entries.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(tiles)
    n1, n2 = int(n_tiles * ratios[0]), int(n_tiles * (ratios[0] + ratios[1]))
    tile2split = {}
    for i, t in enumerate(tiles):
        tile2split[t] = "train" if i < n1 else ("val" if i < n2 else "test")
    out = {"train": [], "val": [], "test": []}
    for t in tiles:
        out[tile2split[t]] += entries[t]
    return out


def write_region(region: str, entries: list, seed: int) -> None:
    """entries: list of {"img","label","tile"} -> splits .pt per region."""
    by_tile = {}
    for e in entries:
        by_tile.setdefault(e["tile"], []).append(
            {"img": e["img"], "label": e["label"], "tile": e["tile"]})
    out_dir = os.path.join(PATCHES_ROOT, region)
    os.makedirs(out_dir, exist_ok=True)
    for split, items in split_by_tile(by_tile, seed=seed).items():
        torch.save(items, os.path.join(out_dir, f"{split}.pt"))
        labels = [e["label"] for e in items]
        counts = np.bincount(labels, minlength=3)
        print(f"    {region}/{split}: n={len(items)}  "
              f"undamaged={counts[0]} damaged={counts[1]} destroyed={counts[2]}")


def write_pre_cache(region: str, uids: list, selected_root: str) -> None:
    """Pre-disaster s2-RGB tensors for a region (used by domain_distance only)."""
    pre_dir = os.path.join(selected_root, "s2")
    items = []
    for uid in uids:
        path = os.path.join(pre_dir, f"{uid}_pre_disaster_s2.tif")
        if not os.path.exists(path):
            continue
        items.append({"img": read_s2_tiff_rgb(path), "tile": uid})
    if not items:
        print(f"    [pre] {region}: no pre s2 tiles, skipping cache")
        return
    os.makedirs(PRE_ROOT, exist_ok=True)
    torch.save(items, os.path.join(PRE_ROOT, f"{region}.pt"))
    print(f"    [pre] {region}: {len(items)} pre s2-tiles cached")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build patch cache for both task sequences.")
    ap.add_argument("--xbd-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--build", choices=["similar", "cross", "all"], default="all",
                    help="which sequence's regions to build (all=default)")
    args = ap.parse_args()

    selected_root = find_selected_root(args.xbd_root)
    if not selected_root:
        raise SystemExit(f"[prep] xBD_selected not found under {args.xbd_root}")
    print(f"[prep] selected root: {selected_root}")

    centroids = load_tile_centroids(selected_root)
    meta_rows = {r["xbd_uid"]: r for r in csv.DictReader(open(os.path.join(selected_root, "metadata.csv")))}
    tci_dir = os.path.join(selected_root, "s2_tci")
    s2_dir = os.path.join(selected_root, "s2")

    def gather(disaster: str, kind: str):
        """usable uid -> {"img","label","tile","coords"} for post tiles."""
        img_dir = tci_dir if kind == "tci" else s2_dir
        usable = {}
        for uid, row in meta_rows.items():
            if row["disaster"] != disaster:
                continue
            label, _ = tile_label(row)
            if label is None or uid not in centroids:
                continue
            suffix = "_s2_tci.tif" if kind == "tci" else "_s2.tif"
            if not os.path.exists(os.path.join(img_dir, f"{uid}_post_disaster{suffix}")):
                continue
            usable[uid] = {"coords": centroids[uid]}
        return usable

    # ---- 1) similar_domain: hurricane-matthew s2_tci -> A/B/C longitude bands
    if args.build in ("similar", "all"):
        print("[prep] similar_domain (hurricane-matthew longitude bands, s2_tci RGB)")
        usable_tci = gather("hurricane-matthew", "tci")
        if len(usable_tci) < 12:
            raise SystemExit(f"[prep] too few s2_tci tiles ({len(usable_tci)}) for bands")
        uids = sorted(usable_tci)
        assign = longitude_bands(usable_tci, N_REGIONS)
        name_by_band = {c: f"hurricane-matthew-{'ABC'[c]}" for c in range(N_REGIONS)}
        region_map = {u: name_by_band[c] for u, c in zip(uids, assign.tolist())}
        os.makedirs(PATCHES_ROOT, exist_ok=True)
        with open(os.path.join(PATCHES_ROOT, "region_map.json"), "w") as f:
            json.dump(region_map, f, indent=2)
        region_order = []
        for c in range(N_REGIONS):
            region = name_by_band[c]
            region_order.append(region)
            member = [u for u, cu in zip(uids, assign.tolist()) if cu == c]
            entries = [{"img": _load_rgb(tci_dir, u, "tci"),
                        "label": tile_label(meta_rows[u])[0], "tile": u}
                       for u in member]
            write_region(region, entries, args.seed)
            write_pre_cache(region, member, selected_root)
        with open(os.path.join(PATCHES_ROOT, "region_order.txt"), "w") as f:
            f.write("\n".join(region_order) + "\n")
        print(f"[prep] similar_domain task order: {', '.join(region_order)}")

    # ---- 2/3) cross_disaster: full hurricane-matthew + palu-tsunami
    if args.build in ("cross", "all"):
        print("[prep] cross_disaster (hurricane-matthew -> palu-tsunami)")
        # full hurricane-matthew: s2_tci RGB (single region, not banded)
        uso = gather("hurricane-matthew", "tci")
        entries = [{"img": _load_rgb(tci_dir, u, "tci"),
                    "label": tile_label(meta_rows[u])[0], "tile": u}
                   for u in sorted(uso)]
        write_region("hurricane-matthew", entries, args.seed)
        write_pre_cache("hurricane-matthew", sorted(uso), selected_root)
        print(f"[prep] hurricane-matthew: {len(entries)} post s2_tci tiles (full-event region)")

        # palu-tsunami: s2_tci absent -> derive RGB from 12-band s2
        usp = gather("palu-tsunami", "s2")
        if len(usp) < 12:
            raise SystemExit(f"[prep] too few palu-tsunami s2 tiles ({len(usp)})")
        entries = [{"img": read_s2_tiff_rgb(os.path.join(s2_dir, f"{u}_post_disaster_s2.tif")),
                    "label": tile_label(meta_rows[u])[0], "tile": u}
                   for u in sorted(usp)]
        write_region("palu-tsunami", entries, args.seed)
        write_pre_cache("palu-tsunami", sorted(usp), selected_root)
        print(f"[prep] palu-tsunami: {len(entries)} post s2 tiles (RGB derived from 12-band)")

    print("[prep] done.")


if __name__ == "__main__":
    main()
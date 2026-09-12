# Terra Incognita — shared utilities: seeding, dataloaders, metrics.
import json
import random
import os

import numpy as np
import torch
import torchvision.transforms as T

# Ordering used everywhere (accuracy-matrix column order = task order).
DEFAULT_REGIONS = ["hurricane-michael", "palu", "santa-rosa-fire"]

PATCHES_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "patches")
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def set_seed(seed: int = 42) -> None:
    """Seed python/numpy/torch (+cuda) so every regime is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transform(train: bool):
    """224x224 patches; light augmentation for train, center-crop for eval."""
    base = [T.Resize((224, 224), antialias=True)]
    if train:
        base += [T.RandomHorizontalFlip(), T.RandomRotation(10)]
    return T.Compose(base)


# ---------------------------------------------------------------------------
# Data loading. Expects the patch cache written by data/prepare_xbd.py, per
# region: data/patches/<region>/{train,val,test}.pt
# Each entry: {"img": [3,H,W] float tensor 0..1, "label": int, "tile": str}
# ---------------------------------------------------------------------------

def _load_region_part(region: str, split: str, subset: int = 0):
    path = os.path.join(PATCHES_ROOT, region, f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing patch cache {path}. Run `python data/prepare_xbd.py` first "
            f"(run.sh does this automatically)."
        )
    entries = torch.load(path, map_location="cpu", weights_only=False)
    if subset and len(entries) > subset:
        entries = entries[:subset]
    return entries


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, region: str, split: str, train: bool, subset: int = 0, seed: int = 42):
        set_seed(seed)  # deterministic truncation
        self.entries = _load_region_part(region, split, subset)
        self.transform = build_transform(train)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        img = self.transform(e["img"])
        return img, e["label"], e["tile"]


def make_loader(region: str, split: str, train: bool, batch_size: int = 32,
                num_workers: int = 2, subset: int = 0, seed: int = 42):
    ds = PatchDataset(region, split, train=train, subset=subset, seed=seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train,
                                       num_workers=num_workers, pin_memory=False)


class MergedDataset(torch.utils.data.Dataset):
    """Concatenates several regions' datasets for JOINT training."""

    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, i):
        for d in self.datasets:
            if i < len(d):
                return d[i]
            i -= len(d)
        raise IndexError


def make_joint_loader(regions, split, train, **kw):
    dss = [PatchDataset(r, split, train=train, **kw) for r in regions]
    ds = MergedDataset(dss)
    return torch.utils.data.DataLoader(ds, batch_size=kw.get("batch_size", 32),
                                       shuffle=train, num_workers=kw.get("num_workers", 2))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def average_forgetting(matrix):
    """Mean backward transfer.

    matrix[i][j] = accuracy on region j's test set AFTER training on task i
    (NaN where region j was not yet seen at step i). Forgetting on region j is
    (best accuracy seen on j before/at its own task) - (final accuracy on j).
    """
    matrix = np.asarray(matrix, dtype=float)
    T, R = matrix.shape
    forget = []
    for j in range(R):
        seen = [matrix[i][j] for i in range(j, T) if not np.isnan(matrix[i][j])]
        if len(seen) < 2:
            continue
        forget.append(max(seen) - seen[-1])
    return float(np.mean(forget)) if forget else 0.0


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
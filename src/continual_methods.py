# Terra Incognita — continual learning methods.
#
# Chosen CL method: EXPERIENCE REPLAY (a strong, simple baseline for
# domain-incremental learning). A small per-region buffer of raw training
# patches is kept, and its samples are mixed into every batch of later-task
# training. EWC / LwF are deliberately NOT implemented — one method, done well.

import torch


class ReplayBuffer:
    """Per-region reservoir of (img, label) training examples, capacity-capped."""

    def __init__(self, capacity_per_region: int = 300):
        self.capacity = capacity_per_region
        self.imgs = []        # list of [C,H,W] float tensors
        self.labels = []      # list of ints
        self.origin = []      # region tag of each sample (for stratified sampling / stats)

    def set_capacity(self, new_capacity: int):
        """Resize buffer; drop excess samples uniformly at random."""
        self.capacity = new_capacity
        if new_capacity and len(self.imgs) > new_capacity:
            keep = sorted(torch.randperm(len(self.imgs))[:new_capacity].tolist())
            self.imgs = [self.imgs[i] for i in keep]
            self.labels = [self.labels[i] for i in keep]
            self.origin = [self.origin[i] for i in keep]

    # ------------------------------------------------------------------
    def add(self, imgs, labels, region: str):
        """Add samples from one batch; drop uniformly at random past capacity."""
        # TODO(h3): switch to reservoir sampling keeping per-region balance.
        for img, lab in zip(imgs, labels):
            self.imgs.append(img.detach().cpu())
            self.labels.append(int(lab))
            self.origin.append(region)
        if self.capacity and len(self.imgs) > self.capacity:
            # drop uniform-random oldest-half -> keeps buffer bounded
            keep = sorted(torch.randperm(len(self.imgs))[: self.capacity].tolist())
            self.imgs = [self.imgs[i] for i in keep]
            self.labels = [self.labels[i] for i in keep]
            self.origin = [self.origin[i] for i in keep]

    # ------------------------------------------------------------------
    def sample(self, n: int):
        """Random stratified batch of n (img, label) pairs."""
        if len(self) == 0:
            return None
        idx = torch.randperm(len(self))[: min(n, len(self))]
        imgs = torch.stack([self.imgs[i] for i in idx])
        labels = torch.tensor([self.labels[i] for i in idx])
        return imgs, labels

    @property
    def size(self) -> int:
        return len(self.imgs)

    def __len__(self) -> int:
        return len(self.imgs)

    @property
    def regions_seen(self) -> set:
        return set(self.origin)


def mix_batch(batch_imgs, batch_labels, replay: "ReplayBuffer | None",
              ratio_replay: float = 0.5):
    """Return (imgs, labels) with a fraction of samples drawn from replay."""
    if replay is None or replay.size == 0 or ratio_replay <= 0:
        return batch_imgs, batch_labels

    n_replay = max(1, int(ratio_replay * batch_imgs.shape[0]))
    samp = replay.sample(n_replay)
    if samp is None:
        return batch_imgs, batch_labels
    r_imgs, r_labels = samp

    imgs = torch.cat([batch_imgs, r_imgs])
    labels = torch.cat([batch_labels, r_labels])
    perm = torch.randperm(imgs.shape[0])
    return imgs[perm], labels[perm]
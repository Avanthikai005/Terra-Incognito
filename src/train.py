# Terra Incognita — training driver.
#
# Three regimes, one entry point:
#   --mode naive   sequential fine-tuning, no replay        (lower bound)
#   --mode joint   all regions pooled, single training run  (upper bound)
#   --mode cl      sequential + experience replay buffer    (our method)
#
# Produces per mode:
#   results/<mode>_matrix.json   rows: after task i; cols: accuracy on each region
#   results/ckpt_<mode>_task<i>.pt
import argparse
import os

import torch
import torch.nn as nn

from src.continual_methods import ReplayBuffer, mix_batch
from src.model import make_model
from src.utils import (RESULTS_ROOT, accuracy, average_forgetting, make_joint_loader,
                       make_loader, save_json, set_seed)


def train_one_epoch(model, loader, replay, opt, crit, device, ratio_replay=0.5):
    model.train()
    total, correct = 0.0, 0.0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if replay is not None:
            imgs, labels = mix_batch(imgs, labels, replay, ratio_replay)
        opt.zero_grad()
        out = model(imgs)
        loss = crit(out, labels)
        loss.backward()
        opt.step()
        total += labels.numel()
        correct += (out.argmax(1) == labels).sum().item()
    return correct / total


@torch.no_grad()
def evaluate(model, loaders_eval, device):
    model.eval()
    accs = {}
    for region, loader in loaders_eval.items():
        total, correct = 0.0, 0.0
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += labels.numel()
        accs[region] = correct / total if total else float("nan")
    return accs


def select_critic():
    return nn.CrossEntropyLoss()


def refresh_optimizer(model, lr=2e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["naive", "joint", "cl"], required=True)
    ap.add_argument("--regions", nargs="+", default=["hurricane-michael", "palu", "santa-rosa-fire"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--replay-cap", type=int, default=300, help="per-region replay capacity (cl only)")
    ap.add_argument("--ratio-replay", type=float, default=0.5)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    print(f"[train] mode={args.mode} device={args.device} regions={args.regions} "
          f"epochs={args.epochs} seed={args.seed}")
    device = torch.device(args.device)

    regions = args.regions
    model = make_model(pretrained=True, device=device)
    crit = select_critic()

    # shared evaluation loaders (held-out test per region, never trained on)
    eval_loaders = {r: make_loader(r, "test", train=False, batch_size=32,
                                   subset=args.subset, seed=args.seed) for r in regions}

    matrix = [[float("nan")] * len(regions) for _ in regions]

    if args.mode == "joint":
        # one training stage on ALL regions; repeat its row for matrix shape
        train_loader = make_joint_loader(regions, "train", train=True,
                                         batch_size=32, subset=args.subset, seed=args.seed)
        opt = refresh_optimizer(model)
        print(f"[train] JOINT: {len(train_loader.dataset)} samples pooled")
        for ep in range(1, args.epochs + 1):
            acc = train_one_epoch(model, train_loader, None, opt, crit, device)
            print(f"    ep {ep}/{args.epochs} train_acc={acc:.4f}")
        accs = evaluate(model, eval_loaders, device)
        row = [accs[r] for r in regions]
        for i in range(len(regions)):
            matrix[i] = row[:]
    else:
        replay = ReplayBuffer(args.replay_cap) if args.mode == "cl" else None
        # tasks arrive in given order; "seen" grows with each regime step
        for t, region in enumerate(regions):
            train_loader = make_loader(region, "train", train=True, batch_size=32,
                                       subset=args.subset, seed=args.seed)
            opt = refresh_optimizer(model)
            print(f"[train] task {t+1}/{len(regions)} on {region} "
                  f"({len(train_loader.dataset)} samples, replay={replay.size if replay else 0})")
            for ep in range(1, args.epochs + 1):
                acc = train_one_epoch(model, train_loader, replay, opt, crit, device,
                                      ratio_replay=args.ratio_replay)
                print(f"    ep {ep}/{args.epochs} train_acc={acc:.4f}")
            if replay is not None:
                # buffer new region's patches for future tasks
                for imgs, labels, _ in train_loader:
                    replay.add(imgs, labels, region)
                print(f"    replay size now {replay.size}")
            # eval all regions seen so far
            accs = evaluate(model, eval_loaders, device)
            for j in range(t + 1):
                matrix[t][j] = accs[regions[j]]
            torch.save(model.state_dict(), os.path.join(RESULTS_ROOT, f"ckpt_{args.mode}_task{t}.pt"))

    save_json({"mode": args.mode, "regions": regions, "matrix": matrix,
               "avg_forgetting": average_forgetting(matrix)}, 
              os.path.join(RESULTS_ROOT, f"{args.mode}_matrix.json"))
    print(f"[train] {args.mode} avg_forgetting={average_forgetting(matrix):.4f}")
    print(f"[train] saved results/{args.mode}_matrix.json")


if __name__ == "__main__":
    main()
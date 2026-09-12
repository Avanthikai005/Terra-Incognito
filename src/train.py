# Terra Incognita — training driver.
#
# Five regimes, one entry point:
#   --mode baseline    one fresh model per region, no transfer   (solvable ceiling)
#   --mode naive       sequential fine-tuning, no replay        (lower bound)
#   --mode joint       all regions pooled, single training run  (upper bound)
#   --mode cl          sequential + experience replay buffer     (our method)
#   --mode cl_adaptive adaptive replay: buffer/ratio scale with domain distance
#
# Sequence-scoped outputs:
#   results/<seq>_<mode>_matrix.json   rows: after task i; cols: accuracy on each region
#   results/ckpt_<seq>_<mode>_task<i>.pt
#
# Usage:
#   python -m src.train --mode cl --sequence similar_domain
#   python -m src.train --mode cl_adaptive --sequence cross_disaster
#   python -m src.train --mode cl --sequence similar_domain --mini
import argparse
import os

import torch
import torch.nn as nn

from src.continual_methods import ReplayBuffer, mix_batch
from src.model import make_model
from src.utils import (RESULTS_ROOT, load_sequence, average_forgetting,
                       make_joint_loader, make_loader, save_json, load_json, set_seed)

ADAPTIVE_DEFAULTS = dict(threshold=0.10, cap_high=300, cap_low=80,
                         ratio_high=0.5, ratio_low=0.3)


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
    ap = argparse.ArgumentParser(description="Terra Incognita training driver.")
    ap.add_argument("--mode", choices=["naive", "joint", "cl", "cl_adaptive", "baseline"],
                    required=True)
    ap.add_argument("--sequence", default="similar_domain",
                    help="task sequence key in configs/task_sequences.json")
    ap.add_argument("--regions", nargs="+", default=None,
                    help="override region list (ignores --sequence when given)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--replay-cap", type=int, default=300,
                    help="per-region replay capacity (cl only, non-adaptive)")
    ap.add_argument("--ratio-replay", type=float, default=0.5,
                    help="replay mix ratio (cl only, non-adaptive)")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mini", action="store_true",
                    help="quick mode: 1 epoch, 20 samples")
    # adaptive-replay knobs (cl_adaptive)
    ap.add_argument("--adapt-threshold", type=float, default=ADAPTIVE_DEFAULTS["threshold"],
                    help="avg distance(new, seen) > threshold => high replay")
    ap.add_argument("--adapt-cap-high", type=int, default=ADAPTIVE_DEFAULTS["cap_high"])
    ap.add_argument("--adapt-cap-low", type=int, default=ADAPTIVE_DEFAULTS["cap_low"])
    ap.add_argument("--adapt-ratio-high", type=float, default=ADAPTIVE_DEFAULTS["ratio_high"])
    ap.add_argument("--adapt-ratio-low", type=float, default=ADAPTIVE_DEFAULTS["ratio_low"])
    args = ap.parse_args()

    if args.mini:
        args.epochs, args.subset = 1, 20
        print("[train] MINI mode: epochs=1, subset=20")

    set_seed(args.seed)
    seq_name = args.sequence
    regions = args.regions if args.regions else load_sequence(seq_name)
    print(f"[train] mode={args.mode} sequence={seq_name} device={args.device} "
          f"regions={regions} epochs={args.epochs} seed={args.seed}")
    device = torch.device(args.device)

    def matrix_path():
        return os.path.join(RESULTS_ROOT, f"{seq_name}_{args.mode}_matrix.json")

    def ckpt_path(t):
        return os.path.join(RESULTS_ROOT, f"ckpt_{seq_name}_{args.mode}_task{t}.pt")

    crit = select_critic()
    eval_loaders = {r: make_loader(r, "test", train=False, batch_size=32,
                                   subset=args.subset, seed=args.seed) for r in regions}
    matrix = [[float("nan")] * len(regions) for _ in regions]

    if args.mode == "baseline":
        for t, region in enumerate(regions):
            model = make_model(pretrained=True, device=device)
            opt = refresh_optimizer(model)
            train_loader = make_loader(region, "train", train=True, batch_size=32,
                                       subset=args.subset, seed=args.seed)
            print(f"[train] BASELINE task {t+1}/{len(regions)} on {region} "
                  f"({len(train_loader.dataset)} samples, no transfer)")
            for ep in range(1, args.epochs + 1):
                acc = train_one_epoch(model, train_loader, None, opt, crit, device)
                print(f"    ep {ep}/{args.epochs} train_acc={acc:.4f}")
            accs = evaluate(model, eval_loaders, device)
            matrix[t] = [accs[r] for r in regions]
            torch.save(model.state_dict(), ckpt_path(t))

    elif args.mode == "joint":
        model = make_model(pretrained=True, device=device)
        train_loader = make_joint_loader(regions, "train", train=True,
                                         batch_size=32, subset=args.subset, seed=args.seed)
        opt = refresh_optimizer(model)
        print(f"[train] JOINT: {len(train_loader.dataset)} samples pooled")
        for ep in range(1, args.epochs + 1):
            acc = train_one_epoch(model, train_loader, None, opt, crit, device)
            print(f"    ep {ep}/{args.epochs} train_acc={acc:.4f}")
        accs = evaluate(model, eval_loaders, device)
        for i in range(len(regions)):
            matrix[i] = [accs[r] for r in regions][:]

    else:
        # sequential regimes: naive / cl / cl_adaptive
        model = make_model(pretrained=True, device=device)
        is_adaptive = args.mode == "cl_adaptive"
        use_replay = args.mode in ("cl", "cl_adaptive")
        replay = ReplayBuffer(args.replay_cap) if use_replay else None
        ratio_replay = args.ratio_replay

        dist_dict = None
        if is_adaptive:
            dist_path = os.path.join(RESULTS_ROOT, f"{seq_name}_domain_distances.json")
            if os.path.exists(dist_path):
                dist_dict = load_json(dist_path)
                print(f"[train] adaptive: loaded distances from {dist_path}")
            else:
                print(f"[train] WARNING: {dist_path} missing; adaptive uses HIGH config "
                      "on every task start (over-conservative fallback)")

        for t, region in enumerate(regions):
            if is_adaptive and dist_dict is not None:
                if t == 0:
                    print(f"[train] adaptive: task 1 -> warm-up with low config")
                    replay.set_capacity(args.adapt_cap_low)
                    ratio_replay = args.adapt_ratio_low
                else:
                    seen = regions[:t]
                    avg_dist = sum(dist_dict[region].get(r, 0.0) for r in seen) / len(seen)
                    high = avg_dist > args.adapt_threshold
                    replay.set_capacity(args.adapt_cap_high if high else args.adapt_cap_low)
                    ratio_replay = args.adapt_ratio_high if high else args.adapt_ratio_low
                    print(f"[train] adaptive: new={region} avg_dist={avg_dist:.4f} "
                          f"-> {'HIGH' if high else 'LOW'} (cap={replay.capacity}, "
                          f"ratio={ratio_replay})")

            train_loader = make_loader(region, "train", train=True, batch_size=32,
                                       subset=args.subset, seed=args.seed)
            opt = refresh_optimizer(model)
            buf = replay.size if replay else 0
            print(f"[train] task {t+1}/{len(regions)} on {region} "
                  f"({len(train_loader.dataset)} samples, replay_buf={buf})")
            for ep in range(1, args.epochs + 1):
                acc = train_one_epoch(model, train_loader, replay, opt, crit, device,
                                      ratio_replay=ratio_replay)
                print(f"    ep {ep}/{args.epochs} train_acc={acc:.4f}")
            if replay is not None:
                for imgs, labels, _ in train_loader:
                    replay.add(imgs, labels, region)
                print(f"    replay size now {replay.size} / {replay.capacity}")
            accs = evaluate(model, eval_loaders, device)
            for j in range(t + 1):
                matrix[t][j] = accs[regions[j]]
            torch.save(model.state_dict(), ckpt_path(t))

    forgetting = None if args.mode == "baseline" else average_forgetting(matrix)
    save_json({"mode": args.mode, "sequence": seq_name, "regions": regions,
               "matrix": matrix, "avg_forgetting": forgetting}, matrix_path())
    forget_msg = "n/a (independent models)" if forgetting is None else f"{forgetting:.4f}"
    print(f"[train] {args.mode} ({seq_name}) avg_forgetting={forget_msg}")
    print(f"[train] saved {matrix_path()}")


if __name__ == "__main__":
    main()
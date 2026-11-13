"""Evaluate trained models under L2 PGD attacks and record the attacked generalization gap.

Reads the runs listed in <out_dir>/clean.csv, loads their checkpoints, and appends one row per (run, eps) to
<out_dir>/attack.csv with attacked_gap = attacked_val_loss - memorization_loss (clean training loss).
"""
import argparse
import csv
import os

import torch
from tqdm import tqdm

from pe_icl.attack import attacked_loss
from pe_icl.data import TaskDataset
from pe_icl.model import TransformerModel, load_checkpoint
from train import ckpt_path

FIELDS = ["pe", "t", "seed", "eps", "attacked_val_loss", "memorization_loss", "attacked_gap"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--val-data", default="data/val_d5.pt")
    p.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.2, 0.3])
    p.add_argument("--pe", nargs="+", default=None, help="only attack runs with these PE types")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=309)
    p.add_argument("--n-dims", type=int, default=5)
    p.add_argument("--n-embd", type=int, default=1024)
    p.add_argument("--n-layer", type=int, default=1)
    p.add_argument("--n-head", type=int, default=1)
    args = p.parse_args()

    with open(os.path.join(args.out_dir, "clean.csv")) as f:
        runs = [r for r in csv.DictReader(f) if args.pe is None or r["pe"] in args.pe]
    val_set = TaskDataset(args.val_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = os.path.join(args.out_dir, "attack.csv")
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for run in tqdm(runs, desc="models"):
            pe, t, seed = run["pe"], int(run["t"]), int(run["seed"])
            model = TransformerModel(args.n_dims, t, args.n_embd, args.n_layer, args.n_head, pe=pe).to(device)
            load_checkpoint(model, ckpt_path(args.out_dir, pe, t, seed), map_location=device)
            mem = float(run["memorization_loss"])
            for eps in args.eps:
                loss = attacked_loss(model, val_set, t, eps, args.batch_size, args.steps)
                writer.writerow({"pe": pe, "t": t, "seed": seed, "eps": eps, "attacked_val_loss": loss,
                                 "memorization_loss": mem, "attacked_gap": loss - mem})
                f.flush()


if __name__ == "__main__":
    main()

"""Train single-layer Transformers on a fixed set of ICL regression prompts and measure the generalization gap.

For every (context length t, seed) the script trains one model on one fixed length-t prompt per training task,
then records
  memorization_loss  query MSE on the fixed training prompts,
  train_loss         query MSE on fresh evaluation prompts of the training tasks,
  val_loss           query MSE on evaluation prompts of the held-out tasks,
  gap                val_loss - memorization_loss,
appending one row per run to <out_dir>/clean.csv and saving weights to <out_dir>/checkpoints/.
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pe_icl.data import TaskDataset, evaluate, generate_tasks, query_mse, train_prompts
from pe_icl.model import PE_TYPES, TransformerModel

FIELDS = ["pe", "t", "seed", "memorization_loss", "train_loss", "val_loss", "gap"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ckpt_path(out_dir, pe, t, seed):
    return os.path.join(out_dir, "checkpoints", f"{pe}_t{t}_seed{seed}.pt")


def train_one(args, pe, t, seed, train_set, val_set, device):
    set_seed(seed)
    model = TransformerModel(args.n_dims, t, args.n_embd, args.n_layer, args.n_head, pe=pe, pe_scale=args.pe_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    prompts = TensorDataset(*train_prompts(train_set, t))
    loader = DataLoader(prompts, batch_size=args.batch_size, shuffle=True, drop_last=True)
    pbar = tqdm(range(args.epochs), desc=f"pe={pe} t={t} seed={seed}")
    for _ in pbar:
        model.train()
        for xs, ys in loader:
            optimizer.zero_grad()
            loss = query_mse(model, xs.to(device), ys.to(device))
            loss.backward()
            optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        mem = [query_mse(model, xs.to(device), ys.to(device)).item()
               for xs, ys in DataLoader(prompts, batch_size=args.batch_size, shuffle=False)]
    row = {"pe": pe, "t": t, "seed": seed, "memorization_loss": sum(mem) / len(mem),
           "train_loss": evaluate(model, train_set, t, args.batch_size),
           "val_loss": evaluate(model, val_set, t, args.batch_size)}
    row["gap"] = row["val_loss"] - row["memorization_loss"]
    return model, row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pe", choices=PE_TYPES, nargs="+", default=["none", "learned"])
    p.add_argument("--context-lens", type=int, nargs="+", default=[6, 7, 8, 9, 10, 12, 15, 20, 25, 30])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 256, 567, 777, 2024])
    p.add_argument("--train-data", default="data/train_d5.pt")
    p.add_argument("--val-data", default="data/val_d5.pt")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=309, help="309 = full training set (full-batch training)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--n-dims", type=int, default=5)
    p.add_argument("--n-embd", type=int, default=1024)
    p.add_argument("--n-layer", type=int, default=1)
    p.add_argument("--n-head", type=int, default=1)
    p.add_argument("--pe-scale", type=float, default=25.0, help="init scale of the learned PE relative to GPT-2's default")
    p.add_argument("--no-save", action="store_true", help="do not save model weights")
    args = p.parse_args()

    if not (os.path.exists(args.train_data) and os.path.exists(args.val_data)):
        print("Dataset not found; generating a new one (results will differ slightly from the paper's data).")
        generate_tasks(args.train_data, args.val_data, n_dims=args.n_dims)
    train_set, val_set = TaskDataset(args.train_data), TaskDataset(args.val_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    csv_path = os.path.join(args.out_dir, "clean.csv")
    new_file = not os.path.exists(csv_path)

    for pe in args.pe:
        for t in args.context_lens:
            for seed in args.seeds:
                model, row = train_one(args, pe, t, seed, train_set, val_set, device)
                print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
                if not args.no_save:
                    torch.save(model.state_dict(), ckpt_path(args.out_dir, pe, t, seed))
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS)
                    if new_file:
                        writer.writeheader()
                        new_file = False
                    writer.writerow(row)


if __name__ == "__main__":
    main()

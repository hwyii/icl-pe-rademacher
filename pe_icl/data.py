"""Linear-regression ICL tasks: w ~ N(0, I/d), x ~ N(0, I), y = w^T x.

Each task stores a pool of points plus a fixed random permutation of the pool, so that prompts are
deterministic given the task:
  * training prompt of length t:   points perm[:t]   (the t-th point is the query);
  * evaluation prompt of length t: context perm[:t-1] and query perm[-1] (same query for every t).
"""
import math

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def generate_tasks(train_path, val_path, num_tasks=10300, val_fraction=0.97, num_points=100, n_dims=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(num_tasks, n_dims, generator=g) / math.sqrt(n_dims)
    x = torch.randn(num_tasks, num_points, n_dims, generator=g)
    y = torch.einsum("nd,npd->np", w, x)
    perm = torch.stack([torch.randperm(num_points, generator=g) for _ in tqdm(range(num_tasks), desc="tasks")])
    data = {"w": w, "x": x, "y": y, "shuffled_indices": perm}

    order = torch.randperm(num_tasks, generator=g)
    num_val = int(num_tasks * val_fraction)
    torch.save({k: v[order[num_val:]] for k, v in data.items()}, train_path)
    torch.save({k: v[order[:num_val]] for k, v in data.items()}, val_path)


class TaskDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path, weights_only=True)

    def __len__(self):
        return self.data["w"].shape[0]

    def __getitem__(self, idx):
        return self.data["x"][idx], self.data["y"][idx], self.data["shuffled_indices"][idx]


def gather_points(x, y, idx):
    xs = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    ys = torch.gather(y, 1, idx)
    return xs, ys


def train_prompts(dataset, prompt_len):
    """Fixed training set: one prompt of length t per training task."""
    d = dataset.data
    return gather_points(d["x"], d["y"], d["shuffled_indices"][:, :prompt_len])


def eval_prompts(x, y, perm, prompt_len):
    idx = torch.cat((perm[:, : prompt_len - 1], perm[:, -1:]), dim=1)
    return gather_points(x, y, idx)


def query_mse(model, xs, ys):
    return torch.nn.functional.mse_loss(model(xs, ys)[:, -1], ys[:, -1])


@torch.no_grad()
def evaluate(model, dataset, prompt_len, batch_size):
    """Mean over batches of the query-point MSE on evaluation prompts."""
    device = next(model.parameters()).device
    model.eval()
    losses = []
    for x, y, perm in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        xs, ys = eval_prompts(x.to(device), y.to(device), perm.to(device), prompt_len)
        losses.append(query_mse(model, xs, ys).item())
    return sum(losses) / len(losses)

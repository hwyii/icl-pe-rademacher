"""L2 PGD attack on the inputs x of an ICL prompt (Eq. 5 of the paper).

All inputs of the prompt (context x_1..x_t and query x_q) are perturbed jointly, with the perturbation of the
whole prompt constrained to ||X' - X||_2 <= eps. Each step moves along the L2-normalised gradient with step size
alpha = eps / 10, followed by projection back onto the eps-ball.
"""
import torch
from torch.utils.data import DataLoader

from pe_icl.data import eval_prompts, query_mse


def pgd_attack(model, xs, ys, eps, alpha, num_iter):
    xs_orig = xs.detach()
    xs_adv = xs_orig.clone()
    for _ in range(num_iter):
        xs_adv.requires_grad_(True)
        loss = query_mse(model, xs_adv, ys)
        grad, = torch.autograd.grad(loss, xs_adv)
        grad_norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-8
        xs_adv = xs_adv.detach() + alpha * grad / grad_norm

        delta = xs_adv - xs_orig
        delta_norm = delta.flatten(1).norm(dim=1).view(-1, 1, 1)
        delta = delta * torch.clamp(eps / delta_norm, max=1.0)
        xs_adv = (xs_orig + delta).detach()
    return xs_adv


def attacked_loss(model, dataset, prompt_len, eps, batch_size=309, num_iter=40):
    """Mean over batches of the query MSE on attacked evaluation prompts."""
    device = next(model.parameters()).device
    model.eval()
    losses = []
    for x, y, perm in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        xs, ys = eval_prompts(x.to(device), y.to(device), perm.to(device), prompt_len)
        xs_adv = pgd_attack(model, xs, ys, eps=eps, alpha=eps / 10, num_iter=num_iter)
        with torch.no_grad():
            losses.append(query_mse(model, xs_adv, ys).item())
    return sum(losses) / len(losses)

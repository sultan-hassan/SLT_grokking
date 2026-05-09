"""
Local Learning Coefficient (LLC) estimator via localised SGLD.

The LLC λ̂ estimates the Real Log Canonical Threshold (RLCT) from
Singular Learning Theory.  A lower λ̂ indicates a flatter, more degenerate
minimum with a stronger implicit Bayesian prior.

Estimator:
    λ̂ = β · n · E_{SGLD}[L(w) − L(w*)]
    β = 1 / log(n),   spring prior: (γ/2)‖w − w*‖²

The spring constant γ controls what the chain probes:
  - Large γ → chain stays near w*, measures local flatness.
  - Small γ → chain drifts between equivalent solutions, measures global multiplicity.
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def estimate_llc(model: nn.Module, criterion: nn.Module,
                 train_loader: DataLoader,
                 n_steps: int = 200, step_size: float = 3e-5,
                 localization: float = 100.0,
                 burnin: int = 50, device: str = "cpu") -> float:
    """
    Estimate the LLC at the current model weights.

    Parameters
    ----------
    model        : trained model at w*
    criterion    : cross-entropy loss (reduction='mean')
    train_loader : training dataloader
    n_steps      : SGLD steps to average after burn-in
    step_size    : SGLD step size ε
    localization : spring constant γ (keeps chain near w*)
    burnin       : initial steps discarded for mixing
    device       : 'cpu' | 'cuda' | 'mps'

    Returns
    -------
    llc : scalar LLC estimate λ̂
    """
    n = len(train_loader.dataset)
    beta = 1.0 / math.log(n)

    sgld = copy.deepcopy(model).to(device)
    w_star = {name: p.data.clone() for name, p in sgld.named_parameters()}

    sgld.train()
    data_iter = _cycle(train_loader)
    energies = []

    for step in range(n_steps + burnin):
        a, b, c = next(data_iter)
        a, b, c = a.to(device), b.to(device), c.to(device)

        sgld.zero_grad()
        criterion(sgld(a, b), c).backward()

        scale = beta * n / len(a)
        with torch.no_grad():
            for name, param in sgld.named_parameters():
                if param.grad is None:
                    continue
                spring = localization * (param.data - w_star[name])
                noise  = math.sqrt(2.0 * step_size) * torch.randn_like(param)
                param.data -= step_size * (scale * param.grad + spring) - noise

        if step >= burnin:
            with torch.no_grad():
                sgld.eval()
                # Per-batch baseline so mini-batch variance cancels
                saved = {n: p.data.clone() for n, p in sgld.named_parameters()}
                for nm, p in sgld.named_parameters():
                    p.data.copy_(w_star[nm])
                L_star = criterion(sgld(a, b), c).item()
                for nm, p in sgld.named_parameters():
                    p.data.copy_(saved[nm])
                L_w = criterion(sgld(a, b), c).item()
                sgld.train()
                energies.append(max(0.0, L_w - L_star))

    return beta * n * float(np.mean(energies))


def _cycle(loader: DataLoader):
    while True:
        yield from loader

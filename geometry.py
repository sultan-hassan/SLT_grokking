"""
Hessian geometry and Fourier circuit analysis.

hessian_trace     — Hutchinson finite-difference estimator of tr(H)
top_eigenvalue    — power iteration estimate of lambda_max(H)
fourier_amplitudes — DFT amplitude spectrum of the token-embedding matrix
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.cpu().flatten() for p in model.parameters()])


def _load_flat(model: nn.Module, flat: torch.Tensor):
    device = next(model.parameters()).device
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].view(p.shape).to(device))
        offset += n


@torch.no_grad()
def _full_loss(model, criterion, loader, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for a, b, c in loader:
        a, b, c = a.to(device), b.to(device), c.to(device)
        total += criterion(model(a, b), c).item() * len(a)
        count += len(a)
    return total / count


def _grad_flat(model, criterion, loader, device) -> torch.Tensor:
    model.train()
    total_grad, n_batches = None, 0
    for a, b, c in loader:
        a, b, c = a.to(device), b.to(device), c.to(device)
        model.zero_grad()
        criterion(model(a, b), c).backward()
        g = torch.cat([p.grad.cpu().flatten() for p in model.parameters()
                       if p.grad is not None])
        total_grad = g if total_grad is None else total_grad + g
        n_batches += 1
    model.zero_grad()
    return total_grad / n_batches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hessian_trace(model: nn.Module, criterion: nn.Module, loader: DataLoader,
                  n_samples: int = 20, delta: float = 1e-3,
                  device: str = "cpu") -> float:
    """
    Estimate tr(H) via the Hutchinson finite-difference trick.
    trace(H) ≈ E_v[(L(w+δv) − 2L(w) + L(w−δv)) / δ²],  v ~ N(0, I)
    """
    w0 = _flat(model)
    L0 = _full_loss(model, criterion, loader, device)
    estimates = []
    for _ in range(n_samples):
        v = torch.randn_like(w0)
        _load_flat(model, w0 + delta * v)
        Lp = _full_loss(model, criterion, loader, device)
        _load_flat(model, w0 - delta * v)
        Lm = _full_loss(model, criterion, loader, device)
        estimates.append((Lp + Lm - 2.0 * L0) / delta ** 2)
    _load_flat(model, w0)
    return float(np.mean(estimates))


def top_eigenvalue(model: nn.Module, criterion: nn.Module, loader: DataLoader,
                   n_iter: int = 20, delta: float = 1e-3,
                   device: str = "cpu") -> float:
    """
    Estimate λ_max(H) via power iteration using gradient finite differences.
    Hv ≈ (∇L(w + δv) − ∇L(w − δv)) / (2δ)
    """
    w0 = _flat(model)
    v = torch.randn(w0.numel())
    v = v / v.norm()
    eigenvalue = 0.0
    for _ in range(n_iter):
        _load_flat(model, w0 + delta * v)
        gp = _grad_flat(model, criterion, loader, device)
        _load_flat(model, w0 - delta * v)
        gm = _grad_flat(model, criterion, loader, device)
        Hv = (gp - gm) / (2.0 * delta)
        eigenvalue = Hv.norm().item()
        v = Hv / (eigenvalue + 1e-12)
    _load_flat(model, w0)
    return eigenvalue


@torch.no_grad()
def fourier_amplitudes(model: nn.Module) -> np.ndarray:
    """
    Fourier amplitude spectrum of the token-embedding matrix W_E.
    A_k = (Σ_d |Ŵ_{k,d}|²)^{1/2} / p  for k = 1 … p//2.
    A large A_k means frequency k is actively used in the embedding.
    """
    p = model.p
    W = model.embed.weight[:p].detach().cpu().float().numpy()   # (p, d_model)
    W_hat = np.fft.fft(W, axis=0)                               # (p, d_model) complex
    amps = (np.abs(W_hat) ** 2).sum(axis=1) ** 0.5 / p         # (p,) real
    return amps[1: p // 2 + 1].astype(float)                   # drop DC + mirror

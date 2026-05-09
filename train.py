"""
Train the grokking transformer across multiple seeds and save geometry metrics.

Each seed controls both model initialisation and train/test split, giving
independent runs for robustness analysis.  Results are saved to
results/run_{seed}.json and then read by plot.py to generate the paper figures.

Usage:
    python train.py                          # 5 seeds, full run (~25 min/seed on CPU)
    python train.py --seeds 0 1 2           # 3 seeds
    python train.py --seeds 0 --steps 3000  # quick single-seed test

    # Step-size ablation (computes LLC at multiple epsilon values in one pass):
    python train.py --seeds 0 --llc_step_sizes 3e-6 3e-5 3e-4

Saved metrics per seed (one entry per checkpoint step):
    steps        : list of ints — checkpoint steps
    train_loss   : list of floats
    test_loss    : list of floats
    train_acc    : list of floats
    test_acc     : list of floats
    llc          : list of floats — SGLD-estimated LLC (primary step size)
    hessian_llc  : list of floats — tr(H) / (2 * lambda_max)
    htrace       : list of floats — Hessian trace tr(H)
    lambda_max   : list of floats — largest Hessian eigenvalue
    fourier_amps : list of lists  — (p//2,) embedding Fourier amplitudes
    llc_ablation : dict  — {"3e-05": [...], "3e-06": [...], ...} one list per
                           step size (only present when --llc_step_sizes has
                           more than one value)
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model    import ModularTransformer, count_params
from data     import make_dataloaders
from llc      import estimate_llc
from geometry import hessian_trace, top_eigenvalue, fourier_amplitudes


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for a, b, c in loader:
        a, b, c = a.to(device), b.to(device), c.to(device)
        logits = model(a, b)
        total_loss += criterion(logits, c).item() * len(a)
        correct    += (logits.argmax(-1) == c).sum().item()
        count      += len(a)
    model.train()
    return total_loss / count, correct / count


# ---------------------------------------------------------------------------
# Single-seed training run
# ---------------------------------------------------------------------------

def run_seed(seed: int, cfg: argparse.Namespace, device: str) -> dict:
    """Train one model and return a metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = make_dataloaders(
        p=cfg.p, train_frac=cfg.train_frac,
        batch_size=cfg.batch_size, seed=seed,
    )

    model = ModularTransformer(
        p=cfg.p, d_model=cfg.d_model,
        n_heads=cfg.n_heads, n_layers=cfg.n_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98),
    )
    criterion = nn.CrossEntropyLoss()

    step_sizes = cfg.llc_step_sizes
    metrics = dict(
        seed=seed,
        steps=[], train_loss=[], test_loss=[], train_acc=[], test_acc=[],
        llc=[], htrace=[], lambda_max=[], hessian_llc=[], fourier_amps=[],
    )
    if len(step_sizes) > 1:
        metrics["llc_ablation"] = {f"{s:.0e}": [] for s in step_sizes}

    train_iter = _cycle(train_loader)

    for step in range(cfg.steps + 1):
        # ── checkpoint ──────────────────────────────────────────────────────
        if step % cfg.ckpt_every == 0:
            tr_loss, tr_acc = evaluate(model, train_loader, criterion, device)
            te_loss, te_acc = evaluate(model, test_loader,  criterion, device)
            metrics["steps"].append(step)
            metrics["train_loss"].append(tr_loss)
            metrics["test_loss"].append(te_loss)
            metrics["train_acc"].append(tr_acc)
            metrics["test_acc"].append(te_acc)

            # SGLD LLC — primary step size (first in list)
            llc = estimate_llc(
                model, criterion, train_loader,
                n_steps=cfg.llc_steps, step_size=step_sizes[0],
                localization=cfg.llc_localization,
                burnin=cfg.llc_burnin, device=device,
            )
            metrics["llc"].append(llc)

            # LLC ablation: additional step sizes (skipped if only one)
            if len(step_sizes) > 1:
                for s in step_sizes:
                    val = estimate_llc(
                        model, criterion, train_loader,
                        n_steps=cfg.llc_steps, step_size=s,
                        localization=cfg.llc_localization,
                        burnin=cfg.llc_burnin, device=device,
                    ) if s != step_sizes[0] else llc
                    metrics["llc_ablation"][f"{s:.0e}"].append(val)

            # Hessian geometry
            ht = hessian_trace(
                model, criterion, train_loader,
                n_samples=cfg.htrace_samples, delta=cfg.htrace_delta,
                device=device,
            )
            lm = top_eigenvalue(
                model, criterion, train_loader,
                n_iter=cfg.eig_iter, delta=cfg.htrace_delta,
                device=device,
            )
            h_llc = ht / (2.0 * lm) if lm > 0 else 0.0
            metrics["htrace"].append(ht)
            metrics["lambda_max"].append(lm)
            metrics["hessian_llc"].append(h_llc)

            # Fourier amplitude spectrum
            fa = fourier_amplitudes(model)
            metrics["fourier_amps"].append(fa.tolist())

            print(f"  seed={seed} step={step:>5d} | "
                  f"train {tr_loss:.3f}/{tr_acc:.1%} | "
                  f"test {te_loss:.3f}/{te_acc:.1%} | "
                  f"llc={llc:.1f}  h_llc={h_llc:.1f}  "
                  f"tr(H)={ht:.0f}  λ_max={lm:.1f}")

        # ── training step ────────────────────────────────────────────────────
        if step == cfg.steps:
            break
        a, b, c = next(train_iter)
        a, b, c = a.to(device), b.to(device), c.to(device)
        optimizer.zero_grad()
        criterion(model(a, b), c).backward()
        optimizer.step()

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _cycle(loader):
    while True:
        yield from loader


def main():
    parser = argparse.ArgumentParser(
        description="Train grokking transformer across multiple seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── experiment ──────────────────────────────────────────────────────────
    parser.add_argument("--seeds",     type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="list of random seeds to run")
    parser.add_argument("--p",         type=int,   default=97,
                        help="prime modulus for (a+b) mod p")
    # ── model ───────────────────────────────────────────────────────────────
    parser.add_argument("--d_model",   type=int,   default=128)
    parser.add_argument("--n_heads",   type=int,   default=4)
    parser.add_argument("--n_layers",  type=int,   default=1)
    # ── training ────────────────────────────────────────────────────────────
    parser.add_argument("--train_frac",type=float, default=0.30)
    parser.add_argument("--steps",     type=int,   default=5000)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--batch_size",type=int,   default=128)
    parser.add_argument("--ckpt_every",type=int,   default=250,
                        help="checkpoint (and measure geometry) every N steps")
    # ── LLC ─────────────────────────────────────────────────────────────────
    parser.add_argument("--llc_steps", type=int,   default=200)
    parser.add_argument("--llc_step_sizes", type=float, nargs="+", default=[3e-5],
                        help="one or more SGLD step sizes; first is the primary LLC; "
                             "extras trigger the ablation stored in llc_ablation")
    parser.add_argument("--llc_localization", type=float, default=100.0)
    parser.add_argument("--llc_burnin",type=int,   default=50)
    # ── Hessian ─────────────────────────────────────────────────────────────
    parser.add_argument("--htrace_samples", type=int,   default=20)
    parser.add_argument("--htrace_delta",   type=float, default=1e-3)
    parser.add_argument("--eig_iter",       type=int,   default=20)
    # ── output ──────────────────────────────────────────────────────────────
    parser.add_argument("--results_dir", type=str, default="results")

    cfg = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(exist_ok=True)

    print(f"\nSLT × Grokking  —  p={cfg.p}, d={cfg.d_model}, "
          f"layers={cfg.n_layers}, device={device}")
    print(f"Seeds: {cfg.seeds}  |  {cfg.steps} steps  |  "
          f"checkpoint every {cfg.ckpt_every} steps\n")

    for seed in cfg.seeds:
        print(f"\n{'─'*60}")
        print(f"Seed {seed}")
        print(f"{'─'*60}")
        t0 = time.time()
        metrics = run_seed(seed, cfg, device)
        elapsed = time.time() - t0

        out_path = results_dir / f"run_{seed}.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f)
        print(f"  → saved {out_path}  ({elapsed/60:.1f} min)")

    print(f"\nAll seeds done. Results in '{cfg.results_dir}/'")
    print("Run  python plot.py  to generate the three paper figures.")


if __name__ == "__main__":
    main()

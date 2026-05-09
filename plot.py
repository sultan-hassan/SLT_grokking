"""
Generate the three paper figures from saved results.

Loads all results/run_*.json files, computes mean ± std across seeds,
and saves three publication-quality figures to figures/.

Usage:
    python plot.py
    python plot.py --results_dir results --figures_dir figures
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        13,
    "axes.labelsize":   15,
    "axes.titlesize":   15,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "legend.fontsize":  11,
    "legend.framealpha": 0.85,
    "lines.linewidth":  2.2,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── colour palette ────────────────────────────────────────────────────────────
C_TRAIN = "#2471A3"   # blue
C_TEST  = "#C0392B"   # red
C_LLC   = "#7D3C98"   # purple  (SGLD-LLC)
C_HLLC  = "#17A589"   # teal    (Hessian-LLC)
C_FREQ  = "#D35400"   # orange  (active frequency count)
C_TRACE = "#1A6A9A"   # deep blue — tr(H)

# Phase background shading
PH_MEM  = "#85C1E9"   # medium blue   — memorisation
PH_PLAT = "#F9E79F"   # medium yellow — plateau
PH_GROK = "#82E0AA"   # medium green  — grokking


# ---------------------------------------------------------------------------
# Data loading & alignment
# ---------------------------------------------------------------------------

def load_runs(results_dir: Path) -> list[dict]:
    paths = sorted(results_dir.glob("run_*.json"))
    if not paths:
        raise FileNotFoundError(f"No run_*.json files found in {results_dir}")
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    print(f"Loaded {len(runs)} run(s): seeds {[r['seed'] for r in runs]}")
    return runs


def align(runs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps  = np.array(runs[0]["steps"])
    matrix = np.array([r[key] for r in runs])   # (n_seeds, n_steps)
    return steps, matrix.mean(axis=0), matrix.std(axis=0)


def detect_transition(runs: list[dict], threshold: float = 0.5) -> tuple[float, float]:
    transition_steps = []
    for r in runs:
        steps = np.array(r["steps"])
        accs  = np.array(r["test_acc"])
        idx = np.where(accs >= threshold)[0]
        if len(idx):
            transition_steps.append(steps[idx[0]])
    arr = np.array(transition_steps, dtype=float)
    return float(arr.mean()), float(arr.std()) if len(arr) > 1 else 0.0


# ---------------------------------------------------------------------------
# Shared phase-shading helper
# ---------------------------------------------------------------------------

def add_phase_shading(ax, mem_end, grok_start, grok_end):
    ax.axvspan(0,          mem_end,    alpha=0.30, color=PH_MEM,  zorder=0)
    ax.axvspan(mem_end,    grok_start, alpha=0.30, color=PH_PLAT, zorder=0)
    ax.axvspan(grok_start, grok_end,   alpha=0.42, color=PH_GROK, zorder=0)


# ---------------------------------------------------------------------------
# Figure 1 — Training dynamics
# ---------------------------------------------------------------------------

def fig_training(runs, figures_dir, phase_steps):
    mem_end, grok_start, grok_end = phase_steps
    steps, mean_tr_loss, std_tr_loss = align(runs, "train_loss")
    _,     mean_te_loss, std_te_loss = align(runs, "test_loss")
    _,     mean_tr_acc,  std_tr_acc  = align(runs, "train_acc")
    _,     mean_te_acc,  std_te_acc  = align(runs, "test_acc")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    xmax = steps[-1]

    # — Loss panel (log scale) ————————————————————————————————————————————
    add_phase_shading(ax1, mem_end, grok_start, grok_end)

    ax1.semilogy(steps, mean_tr_loss, color=C_TRAIN, label="Train loss")
    ax1.fill_between(steps,
                     np.maximum(mean_tr_loss - std_tr_loss, 1e-4),
                     mean_tr_loss + std_tr_loss,
                     color=C_TRAIN, alpha=0.18)

    ax1.semilogy(steps, mean_te_loss, color=C_TEST, linestyle="--", label="Test loss")
    ax1.fill_between(steps,
                     np.maximum(mean_te_loss - std_te_loss, 1e-4),
                     mean_te_loss + std_te_loss,
                     color=C_TEST, alpha=0.18)

    # Random-chance baseline: CE loss for uniform over p=97 classes = log(97)
    ax1.axhline(np.log(97), color="gray", linestyle=":", linewidth=1.5,
                label=r"Random chance ($\ln 97 \approx 4.6$)")

    ax1.set_ylim(bottom=1e-3)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Cross-entropy loss (nats)")
    ax1.set_title("Loss")
    ax1.legend(loc="lower left", fontsize=11)

    # Phase labels
    ylo, yhi = ax1.get_ylim()
    label_y = 10 ** (0.88 * (np.log10(yhi) - np.log10(ylo)) + np.log10(ylo))
    ax1.text(mem_end / 2,                        label_y*2.2, "Memorization",
             ha="center", fontsize=11, color="#1A5276")
    ax1.text((mem_end + grok_start) / 2,         label_y*2.2, "Plateau",
             ha="center", fontsize=11, color="#7D6608")
    ax1.text((grok_start + min(grok_end+200, xmax)) / 2, label_y*2.2, "Grokking",
             ha="center", fontsize=11, color="#145A32")

    # — Accuracy panel ———————————————————————————————————————————————————
    add_phase_shading(ax2, mem_end, grok_start, grok_end)

    ax2.plot(steps, 100 * mean_tr_acc, color=C_TRAIN, label="Train accuracy")
    ax2.fill_between(steps,
                     np.clip(100 * (mean_tr_acc - std_tr_acc), 0, 100),
                     np.clip(100 * (mean_tr_acc + std_tr_acc), 0, 100),
                     color=C_TRAIN, alpha=0.18)

    ax2.plot(steps, 100 * mean_te_acc, color=C_TEST, linestyle="--",
             label="Test accuracy")
    ax2.fill_between(steps,
                     np.clip(100 * (mean_te_acc - std_te_acc), 0, 100),
                     np.clip(100 * (mean_te_acc + std_te_acc), 0, 100),
                     color=C_TEST, alpha=0.18)

    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.set_ylim(-3, 105)
    ax2.axhline(100, color=C_TRAIN, linestyle=":", linewidth=1, alpha=0.4)
    ax2.legend(loc="center right")

    n = len(runs)
    fig.suptitle(
        f"Grokking arc: memorization → plateau → generalization  (mean ± std, n={n} seeds)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    out = figures_dir / "fig1_training.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 — tr(H) collapses; SGLD-LLC does not
# ---------------------------------------------------------------------------

def fig_dissociation(runs, figures_dir, phase_steps):
    mem_end, grok_start, grok_end = phase_steps
    steps = np.array(runs[0]["steps"])
    xmax  = steps[-1]

    _, mean_ht,  std_ht  = align(runs, "htrace")
    _, mean_llc, std_llc = align(runs, "llc")
    _, mean_te_acc, _    = align(runs, "test_acc")

    plat_mask = (mean_te_acc > 0.05) & (mean_te_acc < 0.50)
    post_mask = mean_te_acc > 0.95
    ht_ratio  = mean_ht[plat_mask].mean() / (mean_ht[post_mask].mean() + 1e-9)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    add_phase_shading(ax, mem_end, grok_start, grok_end)

    # tr(H) on left log-axis
    ax.semilogy(steps, mean_ht, color=C_TRACE, linewidth=2.5,
                label=r"$\mathrm{tr}(H)$  (total curvature)")
    ax.fill_between(steps,
                    np.maximum(mean_ht - std_ht, 1.0),
                    mean_ht + std_ht,
                    color=C_TRACE, alpha=0.18)
    ax.set_ylabel(r"Hessian trace  $\mathrm{tr}(H)$", color=C_TRACE, fontsize=13)
    ax.tick_params(axis="y", labelcolor=C_TRACE)

    # Annotate the collapse: plateau mean → post-grokking mean
    post_i = np.where(steps >= grok_end)[0]
    if len(post_i):
        pi = post_i[0]
        ax.annotate(
            f"÷{ht_ratio:.1f}×",
            xy=(steps[pi], mean_ht[pi]),
            xytext=(steps[pi] + xmax * 0.06, mean_ht[pi] * 4.5),
            fontsize=12, color=C_TRACE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_TRACE, lw=1.5),
        )

    # SGLD-LLC on right log-axis
    axb = ax.twinx()
    axb.semilogy(steps, mean_llc, color=C_LLC, linewidth=2.5,
                 label="SGLD-LLC  (global multiplicity)")
    axb.fill_between(steps,
                     np.maximum(mean_llc - std_llc, 1.0),
                     mean_llc + std_llc,
                     color=C_LLC, alpha=0.18)
    axb.set_ylabel("SGLD-LLC", color=C_LLC, fontsize=13)
    axb.tick_params(axis="y", labelcolor=C_LLC)
    axb.spines["right"].set_visible(True)

    ax.set_xlabel("Training step")
    ax.set_title(
        r"$\mathrm{tr}(H)$ collapses $\sim\!4\times$ at grokking; SGLD-LLC unchanged"
        f"  (n={len(runs)} seeds)"
    )

    handles1, labels1 = ax.get_legend_handles_labels()
    handleb, labelb   = axb.get_legend_handles_labels()
    ax.legend(handles1 + handleb, labels1 + labels1 + labelb,
              loc="lower left", fontsize=10.5)

    # Fix: merge both legend sets properly
    ax.get_legend().remove()
    ax.legend(handles1 + handleb, labels1 + labelb, loc="lower right", fontsize=10.5)

    fig.tight_layout()
    out = figures_dir / "fig2_dissociation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 — Circuit Imprint
# ---------------------------------------------------------------------------

def fig_circuit_imprint(runs, figures_dir, phase_steps):
    mem_end, grok_start, grok_end = phase_steps
    # Heatmap uses a single seed (averaged amplitudes blur individual freq activations).
    # All line plots use mean ± std across seeds.
    ref_run    = runs[0]
    amp_matrix = np.array(ref_run["fourier_amps"])   # (n_steps, p//2)
    steps      = np.array(ref_run["steps"])
    # Number of final checkpoints used as the settled reference for both panels.
    n_ref = 5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    xmax = steps[-1]

    # — Left: Fourier spectrum heatmap ————————————————————————————————————
    n_freq = amp_matrix.shape[1]
    im = ax1.imshow(
        amp_matrix.T,                               # (p//2, n_steps)
        aspect="auto",
        origin="lower",
        extent=[steps[0], steps[-1], 0.5, n_freq + 0.5],
        cmap="inferno",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax1, label="Amplitude", fraction=0.04, pad=0.02)

    # Mark active frequencies using the last 5 checkpoints of the reference seed
    # — same window used for the right-panel reference lines — so both panels
    # show a consistent count.
    if len(amp_matrix) >= n_ref:
        mean_post = amp_matrix[-n_ref:].mean(axis=0)    # (p//2,)
        active_k  = np.where(mean_post > 2.0 * mean_post.mean())[0]
        for k in active_k:
            line = ax1.axhline(k + 1, color="white", linewidth=1.2,
                               linestyle="--", alpha=0.9)
            line.set_path_effects([
                pe.withStroke(linewidth=3.0, foreground="black", alpha=0.55)
            ])
        #if len(active_k):
        #    txt = ax1.text(steps[-1] * 0.97, active_k.max() + 2,
        #                   f"{len(active_k)} active\nfreq.", color="white",
        #                   fontsize=9, ha="right", va="bottom")
        #    txt.set_path_effects([
        #        pe.withStroke(linewidth=2.5, foreground="black", alpha=0.6)
        #    ])

    ax1.axvline(mem_end,    color="dodgerblue", linewidth=1.5,
                linestyle="--", alpha=0.8)
    ax1.axvline(grok_start, color="limegreen",  linewidth=2.0,
                linestyle="--", alpha=0.9, label="Grokking onset")

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Fourier frequency  k")
    ax1.set_title("Token-embedding frequency spectrum\n"
                  "(bright = high amplitude; single seed)")
    ax1.legend(loc="upper left", fontsize=11)

    # — Right: active freq count (left axis) and hLLC (right axis) ————————
    # Per-step active-frequency count: frequencies above 2× the step-mean amplitude.
    active_all = []
    for r in runs:
        amps     = np.array(r["fourier_amps"])
        mean_amp = amps.mean(axis=1, keepdims=True)
        active_all.append((amps > 2.0 * mean_amp).sum(axis=1).astype(float))
    active_arr  = np.array(active_all)
    mean_active = active_arr.mean(axis=0)
    std_active  = active_arr.std(axis=0)

    _, mean_hllc, std_hllc = align(runs, "hessian_llc")

    # Use the last 5 checkpoints as the settled reference — the count is still
    # rising in early post-grokking steps (transient), so averaging over all
    # post-grokking steps underestimates the settled value.
    n_ref = 5
    settled_active = mean_active[-n_ref:].mean()
    settled_hllc   = mean_hllc[-n_ref:].mean()

    add_phase_shading(ax2, mem_end, grok_start, grok_end)

    # Active frequency count — left axis
    ax2.plot(steps, mean_active, color=C_FREQ, linewidth=2.5,
             label="Active Fourier freq.")
    ax2.fill_between(steps,
                     np.maximum(mean_active - std_active, 0),
                     mean_active + std_active,
                     color=C_FREQ, alpha=0.22)
    ax2.axhline(settled_active, color=C_FREQ, linestyle=":", linewidth=1.5, alpha=0.7)
    ax2.text(xmax * 0.97, settled_active + 1.6,
             f"~{settled_active:.0f} active freq.", ha="right",
             fontsize=10, color=C_FREQ)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Active Fourier frequencies", color=C_FREQ, fontsize=13)
    ax2.tick_params(axis="y", labelcolor=C_FREQ)
    ax2.set_ylim(0, max(mean_active.max(), 15) * 1.3)

    # hLLC — right axis (linear scale; represents effective constrained dimensions)
    ax2b = ax2.twinx()
    ax2b.plot(steps, mean_hllc, color=C_HLLC, linewidth=2.5, linestyle="--",
              label=r"$\hat\lambda_H = \mathrm{tr}(H)/2\lambda_{\max}$")
    ax2b.fill_between(steps,
                      np.maximum(mean_hllc - std_hllc, 0),
                      mean_hllc + std_hllc,
                      color=C_HLLC, alpha=0.18)
    ax2b.axhline(settled_hllc, color=C_HLLC, linestyle=":", linewidth=1.5, alpha=0.7)
    ax2b.text(xmax * 0.97, settled_hllc - 6.,
              rf"$\hat\lambda_H \approx {settled_hllc:.0f}$"
              rf"$\,\approx\,{settled_active:.0f}\times 2$",
              ha="right", fontsize=10, color=C_HLLC)
    ax2b.set_ylabel(r"Hessian-LLC  $\hat\lambda_H$", color=C_HLLC, fontsize=13)
    ax2b.tick_params(axis="y", labelcolor=C_HLLC)
    ax2b.set_ylim(0, max(mean_hllc.max(), 30) * 1.3)
    ax2b.spines["right"].set_visible(True)

    # Annotate the grokking-onset step where both signals transition
    grok_i = np.where(steps >= grok_start)[0][0]
    ax2.annotate(
        "Both transition\nhere",
        xy=(grok_start, mean_active[grok_i]),
        xytext=(grok_start - xmax * 0.30, mean_active[grok_i] + 3),
        fontsize=10.5,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )

    handles2, labels2 = ax2.get_legend_handles_labels()
    handleb, labelb   = ax2b.get_legend_handles_labels()
    ax2.legend(handles2 + handleb, labels2 + labelb, loc="lower right", fontsize=11)

    ax2.set_title(
        "Circuit Imprint: Hessian geometry recovers circuit dimensionality\n"
        rf"($\hat\lambda_H \approx {settled_hllc:.0f} \approx {settled_active:.0f}"
        r"\,\mathrm{freq} \times 2\,\mathrm{params}$)"
    )

    n = len(runs)
    fig.suptitle(
        "Fourier amplitude and Hessian curvature agree with no shared information"
        f"  (n={n} seeds)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = figures_dir / "fig3_circuit_imprint.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 — LLC step-size ablation (only generated when llc_ablation exists)
# ---------------------------------------------------------------------------

def fig_llc_ablation(runs, figures_dir, phase_steps):
    """
    Overlay tr(H) and LLC at multiple SGLD step sizes, all normalized to their
    plateau mean, so relative changes at grokking are directly comparable.
    Only runs that contain 'llc_ablation' data are used.
    """
    ablation_runs = [r for r in runs if "llc_ablation" in r]
    if not ablation_runs:
        print("  (no llc_ablation data found — skipping fig4)")
        return

    mem_end, grok_start, grok_end = phase_steps
    steps = np.array(ablation_runs[0]["steps"])
    _, mean_te_acc, _ = align(ablation_runs, "test_acc")
    plat_mask = (mean_te_acc > 0.05) & (mean_te_acc < 0.50)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    add_phase_shading(ax, mem_end, grok_start, grok_end)

    # --- tr(H): normalize to plateau mean ---
    _, mean_ht, std_ht = align(ablation_runs, "htrace")
    norm_ht = mean_ht / (mean_ht[plat_mask].mean() + 1e-9)
    ax.plot(steps, norm_ht, color=C_TRACE, linewidth=2.5,
            label=r"$\mathrm{tr}(H)$  (Hessian trace)")
    ax.fill_between(steps,
                    np.maximum(norm_ht - std_ht / (mean_ht[plat_mask].mean() + 1e-9), 0),
                    norm_ht + std_ht / (mean_ht[plat_mask].mean() + 1e-9),
                    color=C_TRACE, alpha=0.15)

    # --- LLC at each step size ---
    # Collect all step-size keys; include primary "llc" mapped to its key
    step_size_keys = sorted(ablation_runs[0]["llc_ablation"].keys(),
                            key=lambda s: float(s))
    colors = ["#7D3C98", "#C0392B", "#27AE60", "#E67E22"]   # purple, red, green, orange
    linestyles = ["--", "-.", ":", (0, (5, 1))]

    for i, key in enumerate(step_size_keys):
        llc_mat = np.array([r["llc_ablation"][key] for r in ablation_runs])
        mean_llc = llc_mat.mean(axis=0)
        std_llc  = llc_mat.std(axis=0)
        norm_llc = mean_llc / (mean_llc[plat_mask].mean() + 1e-9)
        color = colors[i % len(colors)]
        ls    = linestyles[i % len(linestyles)]
        eps_label = float(key)
        ax.plot(steps, norm_llc, color=color, linewidth=2.2, linestyle=ls,
                label=rf"SGLD-LLC  $\varepsilon={eps_label:.0e}$")
        ax.fill_between(steps,
                        np.maximum(norm_llc - std_llc / (mean_llc[plat_mask].mean() + 1e-9), 0),
                        norm_llc + std_llc / (mean_llc[plat_mask].mean() + 1e-9),
                        color=color, alpha=0.12)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.6,
               label="Plateau baseline (1.0)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Value normalized to plateau mean")
    ax.set_title(
        r"SGLD-LLC is flat at grokking across all step sizes; $\mathrm{tr}(H)$ collapses"
        f"  (n={len(ablation_runs)} seed{'s' if len(ablation_runs) > 1 else ''})"
    )
    ax.legend(loc="upper right", fontsize=10.5)

    fig.tight_layout()
    out = figures_dir / "fig4_llc_ablation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate the three paper figures from saved results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--figures_dir", default="figures")
    parser.add_argument("--mem_end",    type=int, default=500)
    parser.add_argument("--grok_start", type=int, default=1900)
    parser.add_argument("--grok_end",   type=int, default=2600)
    cfg = parser.parse_args()

    results_dir = Path(cfg.results_dir)
    figures_dir = Path(cfg.figures_dir)
    figures_dir.mkdir(exist_ok=True)

    runs = load_runs(results_dir)
    phase_steps = (cfg.mem_end, cfg.grok_start, cfg.grok_end)

    if len(runs) == 1:
        t_mean, _ = detect_transition(runs, threshold=0.50)
        phase_steps = (500, int(t_mean) - 100, int(t_mean) + 300)

    print(f"\nPhase boundaries: mem_end={phase_steps[0]}, "
          f"grok_start={phase_steps[1]}, grok_end={phase_steps[2]}")
    print(f"Generating figures → {figures_dir}/\n")

    fig_training(runs,        figures_dir, phase_steps)
    fig_dissociation(runs,    figures_dir, phase_steps)
    fig_circuit_imprint(runs, figures_dir, phase_steps)
    fig_llc_ablation(runs,    figures_dir, phase_steps)

    print("\nDone!  Figures saved:")
    print("  figures/fig1_training.png")
    print("  figures/fig2_dissociation.png")
    print("  figures/fig3_circuit_imprint.png")
    print("  figures/fig4_llc_ablation.png  (if llc_ablation data present)")


if __name__ == "__main__":
    main()

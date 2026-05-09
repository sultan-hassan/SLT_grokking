# SLT × Grokking — Reproducibility Code

Code to reproduce the three figures in  
**"Circuit Imprint"** paper submitted for review. 

---

## What this reproduces

| Figure | File | Description |
|--------|------|-------------|
| Fig. 1 | `figures/fig1_training.png` | Three-phase grokking arc (loss & accuracy) |
| Fig. 2 | `figures/fig2_dissociation.png` | Local flatness vs. global multiplicity diverge |
| Fig. 3 | `figures/fig3_circuit_imprint.png` | Circuit Imprint: geometry recovers circuit dimensionality |

All figures show **mean ± std across multiple random seeds**, demonstrating that the results are robust to initialisation and train/test split.

---

## Setup

```bash
pip install -r requirements.txt
```

No special hardware required; CPU works fine.  
GPU/MPS accelerates each seed from ~25 min → ~5 min.

---

## Reproduce

**Step 1 — Train across seeds** (saves `results/run_{seed}.json` for each seed):

```bash
python train.py --seeds 0 1 2 3 4
```

Optional arguments:
```
--seeds          list of random seeds            [default: 0 1 2 3 4]
--steps          training steps per seed         [default: 5000]
--ckpt_every     checkpoint interval (steps)     [default: 250]
--p              prime modulus                   [default: 97]
--results_dir    where to save JSON results      [default: results]
```

**Step 2 — Generate figures** (reads `results/`, writes `figures/`):

```bash
python plot.py
```

---

## Key hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | 1-layer transformer, d=128, 4 heads | ~224k parameters |
| Task | (a+b) mod 97, 30% train split | 2,821 training pairs |
| Optimiser | AdamW, lr=1e-3, weight_decay=1.0 | High wd drives grokking |
| SGLD steps | 200 (+ 50 burn-in) | step size 3e-5, γ=100 |
| Hessian | 20 Hutchinson probes + 20 power-iter steps | δ=1e-3 |

---

## File structure

```
SLT_Grokking/
├── model.py        # Transformer architecture
├── data.py         # Modular arithmetic dataset
├── geometry.py     # Hessian trace, top eigenvalue, Fourier amplitudes
├── llc.py          # SGLD-based LLC estimator
├── train.py        # Multi-seed training (saves results/)
├── plot.py         # Generate 3 paper figures (reads results/, writes figures/)
├── requirements.txt
└── results/        # Created by train.py
```

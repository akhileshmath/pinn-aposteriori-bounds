# PINN Error Bounds

Validated a posteriori error estimation for physics-informed neural networks solving coercive elliptic Dirichlet problems with soft boundary conditions.

## Project Motive

The mathematical claim behind this repository is simple:

> PINNs produce approximate PDE solutions, but standard training gives no certificate of accuracy. This project aims to provide that certificate.

A trained PINN gives a neural approximation `u_hat`, but a small training loss does not by itself tell you how far `u_hat` is from the exact solution `u` in a mathematically meaningful norm. The purpose of this repository is to compute a rigorous, reproducible upper bound on that error.

## Project Goal

The core estimator used in this repository is:

\[
\|u - \hat{u}\|_{H^1_0(\Omega)} \le \frac{1}{\alpha}\|R\|_{H^{-1}(\Omega)} + \|\nabla w\|_{L^2(\Omega)},
\]

where:
- `R = f + div(a grad u_hat)` is the interior residual
- `w` is the harmonic lifting of the boundary mismatch, solving `-Δw = 0` in `Ω` with `w = g - u_hat` on `∂Ω`

This is the only supported estimator in the refactored pipeline.

## What This Repository Tries To Achieve

The research objective is to build a publication-grade pipeline with four pillars:

1. Theoretical rigor  
   The estimator must come from a correct coercive elliptic error decomposition, not from training heuristics.

2. Reliable effectivity  
   The effectivity index
   \[
   \eta = \frac{\text{estimated error}}{\text{true error}}
   \]
   must satisfy `η >= 1`, and ideally remain moderate.

3. Benchmark validation  
   The code should verify the estimator numerically on representative PDE problems.

4. Reproducible code  
   A reviewer should be able to clone the repository, run one command, regenerate the JSON outputs, and reproduce the figures.

## Current Validated Scope

The standard supported pipeline is intentionally narrower than the full research vision.

The validated code currently supports:
- Poisson on the unit square
- Variable-coefficient diffusion on the unit square
- L-shaped Laplace problem with a re-entrant corner singularity

Legacy exploratory convection-diffusion code has been archived under [legacy](D:/Work/PHD-2026/Project/project-2/pinn-error-bounds/legacy) because it is not part of the mathematically validated coercive elliptic pipeline.

The historical experiment log at [exp_log.md](D:/Work/PHD-2026/Project/project-2/pinn-error-bounds/exp_log.md) is still useful, but it should be read as legacy evidence:
- it records where the old pipeline failed
- it shows why residual-only or inconsistently validated estimates were not sufficient
- it motivates the current refactor toward a stricter, reviewable setup

For the current project state and research tracking, use:
- [docs/CURRENT_RESEARCH_STATUS.md](D:/Work/PHD-2026/Project/project-2/pinn-error-bounds/docs/CURRENT_RESEARCH_STATUS.md)
- [docs/EXPERIMENT_TRACK.md](D:/Work/PHD-2026/Project/project-2/pinn-error-bounds/docs/EXPERIMENT_TRACK.md)
- [docs/PROJECT_MOTIVE_GOAL_AND_READY_TO_WRITE.md](D:/Work/PHD-2026/Project/project-2/pinn-error-bounds/docs/PROJECT_MOTIVE_GOAL_AND_READY_TO_WRITE.md)

## Ready To Write Criteria

The project is only truly ready for paper writing when the following are simultaneously true:

- the error decomposition is proved cleanly
- the constants in the bound are explicit
- the numerical estimator satisfies `η >= 1` on the target benchmark suite
- the effectivity is not uselessly large
- all paper figures and tables are reproducible from the code
- the estimator can be explained and defended without appealing to training loss

A longer research note with the full motive, four-benchmark target, writing checklist, and strategic value is available at [docs/PROJECT_MOTIVE_GOAL_AND_READY_TO_WRITE.md](D:/Work/PHD-2026/Project/project-2/pinn-error-bounds/docs/PROJECT_MOTIVE_GOAL_AND_READY_TO_WRITE.md).

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the lightweight validation tests:

```bash
python tests.py
```

Run one benchmark first:

```bash
python experiments/run.py --benchmark poisson
```

Run the full validated pipeline:

```bash
python experiments/run.py
```

Results are written to `results/validated_results.json`. Figures are written to `figures/`.

For a minimal research workflow:

1. Run `python tests.py`
2. Run `python experiments/run.py --benchmark poisson`
3. Inspect `results/validated_results.json`
4. Run `python experiments/run.py`
5. Update the experiment checklist in `docs/EXPERIMENT_TRACK.md`

## Repository Layout

```text
pinn-error-bounds/
├── src/
│   ├── pinn/
│   ├── estimator/
│   └── benchmarks/
├── experiments/
│   └── run.py
├── results/
├── figures/
├── paper/
├── docs/
├── legacy/
├── README.md
├── requirements.txt
└── setup.py
```

## What Gets Logged

Each benchmark stores:
- true `H^1_0` seminorm error
- estimated error
- residual contribution
- boundary lifting contribution
- effectivity index `η`
- training loss
- estimator mesh size
- stabilization settings

## Expected Interpretation

Good signs:
- tests pass
- the run finishes without solver failures
- `η >= 1`
- `η` stays reasonably close to `1`
- residual and boundary contributions are both visible in the decomposition

Bad signs:
- `η < 1`
- repeated instability in the dual solve
- boundary lifting stays artificially zero under soft boundary conditions
- results change materially under fixed seeds

## Supported Workflow

Use the refactored pipeline only:

```bash
python experiments/run.py
```

Do not use archived legacy runners for new figures or new claims.

## Research Setup

This repository is now organized to support a reviewer-facing numerical analysis workflow:
- `src/` contains the actual supported implementation
- `experiments/run.py` is the single reproducible entry point
- `results/` contains machine-readable outputs
- `figures/` contains regenerated plots
- `docs/` records motive, current status, and experiment tracking
- `legacy/` preserves older code and logs for provenance only

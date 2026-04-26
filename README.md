# PINN Error Bounds

Validated a posteriori error estimation for physics-informed neural networks solving coercive elliptic Dirichlet problems with soft boundary conditions.

## Project Motive

The central claim of this repository is:

> PINNs produce approximate PDE solutions, but standard training does not provide a certificate of accuracy. This project computes that certificate.

A trained PINN gives a neural approximation `u_hat`, but a small training loss does not by itself quantify how far `u_hat` is from the exact solution `u` in a mathematically meaningful norm. The goal of this repository is to compute a rigorous, reproducible upper bound on that error.

## Project Goal

The supported estimator is

\[
\|u - \hat{u}\|_{H^1_0(\Omega)} \le \frac{1}{\alpha}\|R\|_{H^{-1}(\Omega)} + \|\nabla w\|_{L^2(\Omega)},
\]

where:
- `R = f + div(a grad u_hat)` is the interior residual
- `w` is the harmonic lifting of the boundary mismatch, solving `-Delta w = 0` in `Omega` with `w = g - u_hat` on `partial Omega`

This is the only estimator supported by the validated pipeline.

## Current Validated Scope

The validated repository covers three coercive elliptic benchmarks:
- Poisson on the unit square
- variable-coefficient diffusion on the unit square
- the L-shaped Laplace problem with a re-entrant corner singularity

The historical run log in [exp_log.md](./exp_log.md) remains useful as background evidence, but it should be read as pre-refactor history rather than as part of the current validated pipeline.

For current status and research tracking, use:
- [docs/CURRENT_RESEARCH_STATUS.md](./docs/CURRENT_RESEARCH_STATUS.md)
- [docs/EXPERIMENT_TRACK.md](./docs/EXPERIMENT_TRACK.md)
- [docs/PROJECT_MOTIVE_GOAL_AND_READY_TO_WRITE.md](./docs/PROJECT_MOTIVE_GOAL_AND_READY_TO_WRITE.md)

## Validated Results

The current validated benchmark suite achieves effectivities in the range `1.05-1.31`:
- Poisson: `eta = 1.0501`
- Variable coefficient diffusion: `eta = 1.0676`
- L-shaped singularity: `eta = 1.3056`

These values are stored in [results/validated_results.json](./results/validated_results.json).

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

Generate the paper artifacts:

```bash
python experiments/paper_artifacts.py
```

## Repository Layout

```text
pinn-error-bounds/
|- src/
|  |- pinn/
|  |- estimator/
|  `- benchmarks/
|- experiments/
|  |- run.py
|  `- paper_artifacts.py
|- results/
|- paper/
|- docs/
|- tests/
|- README.md
|- requirements.txt
`- setup.py
```

## Output Structure

- `results/validated_results.json`: frozen validated benchmark outputs
- `results/paper/`: machine-readable paper artifact JSON
- `paper/figures/`: submission figures
- `paper/tables/`: submission tables
- `paper/`: self-contained manuscript source

## Supported Workflow

Use the refactored entry points only:

```bash
python experiments/run.py
python experiments/paper_artifacts.py
```

Do not use exploratory output files as the basis for new claims unless they are explicitly regenerated through these scripts.

## What Gets Logged

Each benchmark stores:
- true energy error
- estimated energy error
- residual contribution
- boundary lifting contribution
- effectivity index `eta`
- training loss
- estimator mesh size
- stabilization settings

## Expected Interpretation

Good signs:
- tests pass
- the runs finish without solver failures
- `eta >= 1`
- `eta` stays reasonably close to `1`
- residual and boundary contributions are both visible in the decomposition

Bad signs:
- `eta < 1`
- repeated instability in the dual solve
- boundary lifting stays artificially zero under soft boundary conditions
- materially different results under fixed seeds

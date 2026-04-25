# Current Research Status

## Why This File Exists

This document separates three things that were previously mixed together:
- the long-term paper ambition
- the legacy experiment evidence
- the current validated repository scope

That separation is important if the project is to look like a serious research codebase.

## What The Legacy Experiment Log Shows

The historical run log in [exp_log.md](D:/Work/PHD-2026/Project/project-2/pinn-error-bounds/exp_log.md) captures the main reason the repository needed refactoring.

The old pipeline showed:
- smooth Poisson and variable-coefficient cases with effectivities below `1`
- one acceptable L-shaped result and one catastrophic one
- convection-dominated cases with completely invalid effectivities

In other words, the old code demonstrated the research problem clearly, but it did not provide a stable publication-grade estimator pipeline.

## What The Refactored Codebase Now Validates

The current supported repository validates the following estimator structure:

\[
\|u - \hat{u}\|_{H^1_0(\Omega)} \le \frac{1}{\alpha}\|R\|_{H^{-1}(\Omega)} + \|\nabla w\|_{L^2(\Omega)}.
\]

The current codebase provides:
- a single supported entry point: `python experiments/run.py`
- a single supported implementation path under `src/`
- explicit effectivity validation with failure on `η < 1`
- deterministic seeds
- JSON outputs for reproducibility
- figure generation from those outputs

## What Is Currently In Scope

Validated benchmark scope:
- Poisson on the unit square
- Variable-coefficient coercive diffusion
- L-shaped coercive Laplace benchmark

Archived from the validated scope:
- convection-diffusion and other unsupported operator classes

## What Is Still Not Fully Settled

The repository is cleaner and more defensible now, but some research milestones still remain:
- full end-to-end validated runs across the supported benchmark suite must be collected and preserved
- the theory note in the paper must explain the energy norm and boundary lifting cleanly
- the numerical safety factor used for conservative validation must be documented explicitly in the paper
- the supported benchmark suite must be evaluated enough times to judge robustness, not just single smoke runs

## Practical Interpretation

The repository is currently in this state:

- good enough to demonstrate a serious research direction
- good enough to justify the refactor and the new workflow
- not yet enough, by itself, to claim that the full paper is numerically closed

That is not a weakness. It is the correct status for a serious in-progress numerical analysis project.

## Recommended Next Milestones

1. Run the full supported suite with the validated entry point.
2. Record the benchmark outcomes in `docs/EXPERIMENT_TRACK.md`.
3. Decide whether the L-shaped benchmark is stable enough for the main paper.
4. Keep convection-diffusion explicitly outside the validated scope unless a separate theorem is proved.

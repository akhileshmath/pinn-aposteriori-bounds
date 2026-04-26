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

The validated baseline is now in place, but several paper-preparation tasks still remain:

- the manuscript must stay consistent with the validated energy-error interpretation
- figures and tables must be regenerated through the paper-artifact script
- the local TeX environment must compile the self-contained SIAM manuscript cleanly
- additional experiments beyond the supported benchmark suite remain future work

## Practical Interpretation

The repository is currently in this state:

- good enough to demonstrate a serious research direction
- good enough to justify the refactor and the new workflow
- numerically validated on the supported benchmark suite, with remaining work concentrated in presentation and submission cleanup

That is not a weakness. It is the correct status for a serious in-progress numerical analysis project.

## Recommended Next Milestones

1. Keep `results/validated_results.json` as the frozen benchmark baseline.
2. Regenerate figures and tables through `python experiments/paper_artifacts.py`.
3. Compile the SIAM manuscript once the local TeX environment is fixed.
4. Keep convection-diffusion explicitly outside the validated scope unless a separate theorem is proved.

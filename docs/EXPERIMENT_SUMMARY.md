# Experiment Summary

## Validated baseline

The validated benchmark suite achieves effectivities in the range `1.0501-1.3056`.

Benchmark outcomes:

* Poisson: `eta = 1.0501`
* Variable coefficient diffusion: `eta = 1.0676`
* L-shaped singularity: `eta = 1.3056`

## Ablation interpretation

Across all validated benchmarks, the lifting term is the dominant contribution to the final estimate, confirming that soft Dirichlet enforcement introduces a boundary-driven error component that is invisible to residual-only certification.

The strongest boundary-driven case in the current suite is `Poisson (smooth)`.
The largest relative residual contribution appears in `L-shaped domain singularity`, but even there the lifting term remains essential for reliability.

## Convergence note

The paper artifact pipeline evaluates each trained solution on a four-level mesh ladder:

* Poisson (smooth): estimated error changes from `1.2935e-03` to `1.2257e-03` across the four-level mesh ladder.
* Variable coefficient diffusion: estimated error changes from `2.7544e-03` to `2.5807e-03` across the four-level mesh ladder.
* L-shaped domain singularity: estimated error changes from `8.6505e-02` to `6.2433e-02` across the four-level mesh ladder.

## Paper-ready takeaway

> The full estimator remains reliable and reasonably sharp across all supported coercive elliptic benchmarks. Residual-only estimates systematically underrepresent the total error, while the harmonic lifting term captures the boundary-driven component induced by soft boundary enforcement. The estimator is sharpest on smooth problems and remains moderately efficient on the singular L-shaped domain.

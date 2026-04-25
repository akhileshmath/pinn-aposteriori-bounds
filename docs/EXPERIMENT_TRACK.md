# Experiment Track

## Purpose

This file records validated numerical results for the residual-based a posteriori estimator across all supported benchmarks.

It serves as:

* a reproducibility log,
* a validation checkpoint before submission,
* a clean separation from exploratory/debug runs.

Only validated runs satisfying `eta >= 1` are recorded here.

---

## How to Run

Full benchmark suite:

```bash
python experiments/run.py
```

Single benchmark:

```bash
python experiments/run.py --benchmark poisson
python experiments/run.py --benchmark variable_coefficient
python experiments/run.py --benchmark l_shaped
```

---

## Acceptance Criteria

* `eta < 1` -> reject: estimator not reliable
* `1 <= eta <= 2` -> accept: reliable and reasonably sharp
* `eta > 2` -> accept with warning: bound is reliable but loose

---

## Validated Run

Command:

```bash
python experiments/run.py
```

Outcome:

* Poisson: validated
* Variable coefficient diffusion: validated
* L-shaped singularity: validated

---

## Results

### Poisson

* Status: PASS
* True energy error: `1.1830e-03`
* Estimated energy error: `1.2423e-03`
* Residual contribution: `3.2535e-05`
* Boundary contribution: `1.2098e-03`
* Effectivity `eta`: `1.0501`
* Notes:
  * Smooth baseline benchmark.
  * Bound is reliable and sharp.
  * Boundary contribution dominates because the PINN uses soft Dirichlet enforcement.

---

### Variable Coefficient

* Status: PASS
* True energy error: `2.9830e-03`
* Estimated energy error: `3.1848e-03`
* Residual contribution: `1.0700e-04`
* Boundary contribution: `3.0778e-03`
* Effectivity `eta`: `1.0676`
* Notes:
  * Confirms the estimator remains reliable with heterogeneous diffusion.
  * Coercivity-dependent scaling behaves correctly in practice.
  * Bound remains sharp.

---

### L-Shaped

* Status: PASS
* True energy error: `4.4128e-02`
* Estimated energy error: `5.7611e-02`
* Residual contribution: `9.2933e-03`
* Boundary contribution: `4.8318e-02`
* Effectivity `eta`: `1.3056`
* Notes:
  * Non-convex domain with a re-entrant corner singularity.
  * True energy is evaluated on the fitted triangular mesh for geometric consistency.
  * The bound is reliable and moderately sharp.
  * Residual contribution shows mild non-monotonicity under refinement and is handled by the conservative refinement fallback.

---

## Summary

Validated effectivities:

* Poisson: `eta = 1.0501`
* Variable coefficient diffusion: `eta = 1.0676`
* L-shaped singularity: `eta = 1.3056`

Overall assessment:

* the estimator is reliable across all supported benchmarks
* the smooth and variable-coefficient cases are very sharp
* the singular benchmark is also validated and remains reasonably efficient

---

## Paper Interpretation

Recommended experiment-section interpretation:

> The validated benchmark suite shows that the estimator is both reliable and practically sharp across the supported coercive elliptic problems. On the smooth Poisson and variable-coefficient diffusion benchmarks, the effectivity indices remain close to one (`1.0501` and `1.0676`), indicating that the computable bound closely tracks the true energy error. On the L-shaped re-entrant corner benchmark, the effectivity increases to `1.3056`, which is expected for a singular geometry, but the estimator remains reliable and moderately tight. In all three cases, the boundary lifting term is the dominant contribution, confirming that soft boundary enforcement introduces a boundary-driven error component that must be accounted for explicitly in any rigorous a posteriori bound for PINNs.

Short form:

> Across all supported benchmarks, the estimator achieves validated effectivities in the range `1.05-1.31`, with the sharpest performance on smooth problems and a moderate but still reliable loss of efficiency on the singular L-shaped domain.

---

## Legacy Context

Earlier versions exhibited:

* `eta < 1` reliability failures,
* unstable L-shaped dual solves,
* inconsistent L-shaped geometry handling,
* overly crude validation margins.

These issues are now resolved for the supported benchmark suite.

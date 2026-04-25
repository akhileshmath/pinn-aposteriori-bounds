# Project Motive, Goal, and "Ready to Write" Criteria

## 1. What Is This Project Actually About?

The core intellectual claim of this project is:

> **PINNs produce approximate PDE solutions, but they come with no certificate of accuracy. This paper provides one.**

A PINN trains a neural network `u_hat` to minimize a residual loss, but after training you only know the loss is small, not how far `u_hat` is from the true solution `u` in a mathematically meaningful norm. This project aims to construct a computable and reliable upper bound on that error.

In the long-form research vision, the target statement is:

\[
\|u - \hat{u}\|_{H^1(\Omega)} \le \eta(\hat{u}),
\]

with a reliability condition that the estimator should dominate the true error.

In the current validated repository, the implemented and enforced form is the energy-seminorm version

\[
\|u - \hat{u}\|_{H^1_0(\Omega)} \le \frac{1}{\alpha}\|R\|_{H^{-1}(\Omega)} + \|\nabla w\|_{L^2(\Omega)},
\]

where:
- `R = f + div(a grad u_hat)`
- `w` solves `-Δw = 0` in `Ω` with `w = g - u_hat` on `∂Ω`

This is the a posteriori error estimation problem, classical in FEM, adapted here to PINNs with penalty-enforced boundary conditions.

## 2. What Must You Achieve?

There are four pillars.

### Pillar 1 — Theoretical Rigor

The estimator must be derived from first principles. The full coercive elliptic bound is:

\[
\|u - \hat{u}\|_{H^1_0} \le \underbrace{\frac{1}{\alpha}\|R\|_{H^{-1}(\Omega)}}_{\text{interior residual term}} + \underbrace{\|\nabla w\|_{L^2(\Omega)}}_{\text{BC lifting term}}.
\]

This requires a proof that can withstand scrutiny from a numerical analysis referee.

### Pillar 2 — Effectivity Index \(\eta \in [1, C]\)

Define:

\[
\text{effectivity index} = \frac{\eta(\hat{u})}{\|u - \hat{u}\|}.
\]

Interpretation:
- `>= 1`: reliability
- moderate size: efficiency

The codebase now enforces `η >= 1` in the validated pipeline.

### Pillar 3 — Benchmark Validation

The broad research vision includes four benchmark types:

| Benchmark | Challenge | What It Proves |
|---|---|---|
| Poisson on unit square | Baseline | Bound works in a clean setting |
| Variable-coefficient diffusion | non-constant `a(x)` | Bound handles coercivity constants correctly |
| L-shaped domain, `r^{2/3}` singularity | low regularity | Bound does not rely on global `H^2` smoothness |
| Convection-dominated regime | near-singular perturbation | Tracks parameter dependence correctly |

Important current status:
- the validated repository currently supports the first three coercive elliptic cases
- convection-dominated experiments are archived as legacy exploratory work, not part of the supported publication pipeline

### Pillar 4 — Clean, Reproducible Codebase

Every table and figure should be regeneratable from one entry point. That is a research-maturity requirement, not just a software convenience.

## 3. When Is the Project "Ready to Write"?

Do not treat the project as paper-ready until the following are satisfied.

### Theory
- [ ] The decomposition `e = e_0 + w` is proved clearly
- [ ] The bound with explicit constants is written cleanly
- [ ] The regularity assumptions are fully stated
- [ ] The `H^{-1}` computation and normalization are justified

### Numerics
- [ ] `η >= 1` on the intended benchmark suite
- [ ] Effectivity is stable as the PINN improves
- [ ] The L-shaped benchmark is robust
- [ ] If convection-diffusion is claimed, it has its own mathematically justified estimator

### Code
- [x] Single entry point exists: `experiments/run.py`
- [x] Lightweight validation tests exist: `tests.py`
- [x] Results are saved to JSON
- [x] Publication figures are generated from the same results file
- [ ] All constants appearing in the paper text are documented cleanly

### Paper readiness signal

You should be able to explain, without notes:
- why `η >= 1` matters
- why a residual-only estimator fails under soft boundary conditions
- why the harmonic lifting fixes that gap

## 4. What Does the Paper Argue, in One Paragraph?

One defensible summary is:

> We derive a rigorous a posteriori error bound for soft-BC PINN approximations of coercive second-order elliptic PDEs in the energy norm. The main point is that penalty-based boundary enforcement produces a boundary error component that is invisible to interior residual estimators. By decomposing the error via harmonic lifting and bounding the interior and boundary parts separately, we obtain a computable reliability estimate. The estimator is implemented in a reproducible code pipeline and validated on representative elliptic benchmarks, including a smooth Poisson problem, a variable-coefficient problem, and an L-shaped singular benchmark.

That is the current publication-aligned statement for this repository.

## 5. Strategic Value for PhD Applications

The value of this project is that it demonstrates:
- theorem-driven numerical analysis, not just empirical ML
- implementation of a mathematically meaningful estimator
- numerical verification of a falsifiable claim
- the ability to connect PDE analysis, scientific computing, and reproducible research software

The boundary-lifting insight is the main conceptual contribution:
- interior residuals alone are insufficient under soft BC training
- the missing boundary contribution must be estimated separately
- the decomposition is mathematically nontrivial and computationally useful

## 6. Reference Status Note

This document records the full research motive and long-term writing checklist.

The current validated repository standard is narrower:
- single supported entry point: `python experiments/run.py`
- supported benchmark suite: coercive elliptic problems only
- enforced reliability guard: `η >= 1`

That narrower scope is intentional and is what makes the present repository reviewable and trustworthy.

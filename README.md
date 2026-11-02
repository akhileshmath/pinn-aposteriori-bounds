# Residual-Based A Posteriori Error Estimation for PINNs in the Energy Seminorm

Code and numerical experiments for the manuscript *Residual-Based A Posteriori Error Bounds for Physics-Informed Neural Networks in the Energy Seminorm* (A. Yadav, in preparation).

The repository computes, for a trained PINN $\hat u$ with **soft** Dirichlet boundary enforcement, the quantities that enter an energy-seminorm error estimate, and compares them with the exact error on three benchmarks with known solutions.

---

## 1. Problem and decomposition

We consider the coercive elliptic problem

$$
-\nabla\cdot\big(a\,\nabla u\big)=f \ \text{ in } \Omega,
\qquad u=g \ \text{ on } \partial\Omega,
\qquad 0<\alpha\le a\le\beta .
$$

A PINN with soft boundary enforcement satisfies the boundary condition only approximately, so the error $e=u-\hat u$ is **not** in $H^1_0(\Omega)$. We write $e=e_0+w$, where $w$ is the Laplace-harmonic lifting of the boundary mismatch $g-\hat u$ (defined weakly) and $e_0\in H^1_0(\Omega)$.

Notation: $R=f+\nabla\cdot(a\nabla\hat u)$ is the interior residual, $\|R\|_{H^{-1}}$ its dual norm, and $\|\cdot\|$ the $L^2$ norm.

### What is proved (see the manuscript)

| Case | Statement | Status |
|---|---|---|
| Constant $a=a_0$ | $\|\nabla e\|^2=a_0^{-2}\|R\|_{H^{-1}}^2+\|\nabla w\|^2$ (exact identity) | Proved (Theorem 3.1) |
| Constant $a=a_0$ | Sum form $a_0^{-1}\|R\|_{H^{-1}}+\|\nabla w\|$ is between $1$ and $\sqrt2$ times $\|\nabla e\|$ | Follows from the identity |
| Variable $a$ | Upper bound with an additional factor $\sqrt{\beta/\alpha}$ on the lifting term | Stated in the manuscript (Theorem 3.3) |
| Variable $a$ | The sum form with factor $1$ on the lifting term | **Not proved.** Reported only as a numerical observation. |

These statements concern **exact** norms. The values computed by this code are discretization estimates, not certified bounds.

### Estimators computed by the code

| Name in output | Definition | Meaning |
|---|---|---|
| `eta_oplus` (sum form) | $a_0^{-1}\|R\|_{H^{-1}}+\|\nabla w\|$ | Upper-bound form for constant $a$ |
| `eta_perp` (Pythagoras) | $\big(a_0^{-2}\|R\|_{H^{-1}}^2+\|\nabla w\|^2\big)^{1/2}$ | Equals $\|\nabla e\|$ in the continuum for constant $a$ |
| `eta_a` (a-weighted) | residual term combined with the $a$-weighted lifting energy | Numerical diagnostic for variable $a$; no theorem |

The error is measured in the unweighted seminorm $\|\nabla e\|$. The effectivity index of an estimator $\eta$ is $\eta/\|\nabla e\|$.

---

## 2. Benchmarks

| Key | Problem | Discretization of the norms |
|---|---|---|
| `poisson` | Smooth Poisson problem on the unit square | 5-point finite-difference Laplacian solved by a sine transform (coincides with the P1 stiffness matrix on the right-triangle mesh) |
| `variable_coefficient` | Variable diffusion on the unit square, $\alpha=1$, $\beta=1.5$ | Same, with the coefficient evaluated at triangle centroids for the weighted lifting energy |
| `l_shaped` | Laplace problem on the L-shaped domain, re-entrant-corner singularity | Conforming P1 finite elements; exact error by 3-point Gauss quadrature on a finer reference mesh |

The mesh ladder is fixed before any evaluation and does not use the exact solution. For the unit square it is 96, 192, 384; for the L-shaped domain it is 80, 160, 320. All ladder values are stored in `results/validated_results.json` under `metrics/mesh_history`.

---

## 3. Results (seed 42, finest mesh level)

One trained network per benchmark. Full numbers: `results/validated_results.json`.

| Benchmark | Training loss | $\|\nabla e\|$ | $\eta_\perp/\|\nabla e\|$ | $\eta_\oplus/\|\nabla e\|$ | residual term / lifting term |
|---|---|---|---|---|---|
| Poisson | 1.117e-06 | 1.1830e-03 | 1.0003 | 1.0275 | 0.027 |
| Variable coefficient | 4.117e-06 | 2.4432e-03 | 1.0002 | 1.0229 | 0.023 |
| L-shaped | 5.944e-05 | 5.7498e-02 | 1.0441 | 1.0858 | 0.041 |

For the variable-coefficient benchmark $\eta_a/\|\nabla e\|=1.0143$. This is a numerical observation, not a proven bound.

Observations supported by these runs:

- The lifting term of the boundary mismatch is by far the larger of the two terms in all three runs, although the boundary-condition loss is small. A small training loss therefore does not imply a small energy error.
- On the unit square the computed estimators are stable under mesh refinement.
- On the L-shaped domain the effectivity decreases under refinement: $\eta_\perp/\|\nabla e\|=1.2198,\ 1.1012,\ 1.0441$ at mesh sizes 80, 160, 320. The value at the finest level still contains discretization error of the lifting norm and should not be read as a property of the estimator. The L-shaped network was still improving when training stopped.

Limitations: one seed and one trained network per benchmark; no statement is made about variability across seeds. Norms are computed numerically, so all effectivities are estimates.

---

## 4. Repository layout

```
src/
  pinn/          network, sampling, Adam + L-BFGS training
  estimator/     dual norm, harmonic lifting, P1 energy, quadrature, estimator
  benchmarks/    problem definitions and exact solutions
experiments/
  run.py               train the benchmarks and evaluate the estimators
  paper_artifacts.py   regenerate all paper tables and figures from one set of networks
configs/         benchmark configuration
tests/           smoke tests and identity tests
results/
  validated_results.json   (+ .meta.json)   output of run.py
  weights/                 trained network weights, one .pt per benchmark
  paper/                   JSON written by paper_artifacts.py
docs/EXPERIMENT_SUMMARY.md  generated summary of the latest run
```

`paper/figures/` and `paper/tables/` are written by `paper_artifacts.py`.

---

## 5. Reproducing the results

Python 3.10 with the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Tests (run from the repository root):

```bash
python tests/test_identities.py
python -m tests.smoke
python -m tests.test_result_schema
python -m tests.test_manufactured_l_shaped
```

Experiments (all three benchmarks took about 13 minutes on a laptop CPU):

```bash
python experiments/run.py --benchmark all --seed 42 --weights-dir results/weights
python experiments/paper_artifacts.py --seed 42
```

`run.py` writes `results/validated_results.json`, `results/validated_results.meta.json`, `results/figures/` and the weights. `paper_artifacts.py` retrains with the same seed and writes `results/paper/`, `paper/figures/` and `paper/tables/`. Both scripts produce identical training losses for the same seed.

---

## 6. Scope

The code covers coercive elliptic problems with soft Dirichlet enforcement. It makes no claim for convection-dominated problems, non-coercive problems or hyperbolic systems. A proof of the sum-form bound for variable coefficients is not given.

---

## 7. Citation

```bibtex
@misc{yadav2026aposteriori,
  title  = {Residual-Based A Posteriori Error Bounds for Physics-Informed Neural Networks in the Energy Seminorm},
  author = {Yadav, Akhilesh},
  year   = {2026},
  note   = {Manuscript in preparation. DOI or repository link to be added after posting.}
}

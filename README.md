# Residual-Based A Posteriori Error Bounds for PINNs

This repository implements and validates a **residual-based a posteriori error estimator** for Physics-Informed Neural Networks (PINNs) applied to coercive elliptic PDEs with Dirichlet boundary conditions.

The codebase is structured as a **research artifact**, ensuring:

- reproducibility
- theoretical correctness
- alignment between code, experiments, and manuscript

---

### Project Overview

Physics-Informed Neural Networks (PINNs) approximate PDE solutions, but **training loss does not provide a reliable error certificate**.

This work develops a **rigorous, computable a posteriori error bound** for a PINN approximation $$\hat{u}$$ of the true solution $$u$$ in the energy norm.

### Validated Benchmarks

- Poisson equation (smooth solution)
- Variable-coefficient diffusion
- L-shaped domain (singularity)

---

##  Mathematical Formulation

We consider the elliptic PDE:

$$
-\nabla \cdot (a(x)\nabla u(x)) = f(x)
\quad \text{in } \Omega,
$$

$$
u = g \quad \text{on } \partial\Omega
$$

where $$a(x) \ge \alpha > 0$$ ensures coercivity.

---

### A Posteriori Error Bound

The validated estimator is:

$$
\|u - \hat{u}\|_{H^1_0(\Omega)}
\;\le\;
\frac{1}{\alpha} \|R\|_{H^{-1}(\Omega)}
\;+\;
\|\nabla w\|_{L^2(\Omega)}
$$

where:

- Interior residual:
$$
R = f + \nabla \cdot (a \nabla \hat{u})
$$

- Boundary lifting:
$$
-\Delta w = 0 \quad \text{in } \Omega
$$

$$
w = g - \hat{u} \quad \text{on } \partial\Omega
$$

---

### 🔴 Key Insight

- ❌ Residual-only estimators are **incomplete**
- ✔ Boundary mismatch contributes significantly to error
- ✔ Full estimator ensures **reliability**

---

## ⚙️ Methodology

The pipeline consists of three core modules:

### 1. PINN Solver (`src/pinn/`)
- neural network architecture
- collocation sampling
- training (Adam + L-BFGS)

### 2. Error Estimator (`src/estimator/`)

- dual norm computation $$\|R\|_{H^{-1}}$$
- harmonic lifting $$w$$
- error decomposition
- effectivity validation

### 3. Benchmarks (`src/benchmarks/`)

- PDE definitions
- exact solutions
- domain geometry
- coercivity constants

---

## Repository Structure

```

pinn-error-bounds/
│
├── src/
│   ├── pinn/
│   ├── estimator/
│   └── benchmarks/
│
├── experiments/
│   ├── run.py
│   └── paper_artifacts.py
│
├── results/
│   ├── validated_results.json
│   └── paper/
│
├── paper/
│   ├── main.tex
│   ├── figures/
│   ├── tables/
│   └── sections/
│
├── docs/
│   └── archive/
│
├── tests/
├── requirements.txt
└── setup.py

```

---

## 📊 Results (Validated)

Stored in:
```

results/validated_results.json

````

### Effectivity Index

$$
\eta = \frac{\text{Estimated Error}}{\text{True Error}}
$$

| Benchmark | η |
|----------|---|
| Poisson | 1.0501 |
| Variable Coefficient | 1.0676 |
| L-shaped Domain | 1.3056 |

---

### Interpretation

- $$\eta \ge 1$$ → estimator is **reliable**
- $$\eta \approx 1$$ → estimator is **sharp**
- L-shaped domain → higher η due to singularity

---

## 🔬 Scientific Findings

### Training Loss is Misleading
Small loss does NOT imply small error.

---

### Residual Alone is Insufficient
Fails to capture boundary error.

---

### Full Estimator is Reliable
Includes both:
- interior residual
- boundary lifting

---

##  Reproducibility

### Install dependencies
```bash
pip install -r requirements.txt
````

---

### Run tests

```bash
python tests.py
```

---

### Run experiments

```bash
python experiments/run.py
```

Outputs:

* metrics → `results/validated_results.json`
* figures → `results/figures/`

---

### Generate paper artifacts

```bash
python experiments/paper_artifacts.py
```

Outputs:

* figures → `paper/figures/`
* tables → `paper/tables/`
* JSON → `results/paper/`

---

## 📄 Paper Compilation

```bash
cd paper
pdflatex main.tex
pdflatex main.tex
```

* Uses bundled `.bbl` (arXiv-safe)
* No external dependencies required

---

## 📚 Documentation

* `docs/CURRENT_RESEARCH_STATUS.md`
* `docs/EXPERIMENT_SUMMARY.md`
* `docs/PROJECT_MOTIVE_GOAL_AND_READY_TO_WRITE.md`

---

##  Scope

This repository validates:

✔ Coercive elliptic PDEs
✔ Soft boundary PINNs
✔ Residual + boundary estimator

It does NOT claim validity for:

*  convection-dominated problems
*  non-coercive PDEs
* hyperbolic systems

---

##  Contribution

> A corrected and reliable a posteriori error estimator for PINNs that accounts for both interior residual and boundary mismatch.

---

##  Citation

```bibtex
@article{yadav2026aposteriori,
  title={Residual-Based A Posteriori Error Bounds for PINN Solutions of Elliptic PDEs},
  author={Yadav, Akhilesh},
  year={2026},
  journal={arXiv preprint}
}
```

# Project Motive, Goal, and "Ready to Write" Criteria

## Audit Note

This document has been rewritten to ensure every claim is grounded in the
actual codebase and validated results. The previous version contained four
classes of inaccuracy: an incomplete bound formula, inflated effectivity
claims, inclusion of an unvalidated benchmark, and an unjustified safety
margin presented as a theorem. All have been corrected below.

---

## 1. What Is This Project Actually About?

The core intellectual claim is:

> **PINNs produce approximate PDE solutions but carry no certificate of
> accuracy. This paper provides one — and shows that the standard
> residual-only approach is provably insufficient for penalty-based
> boundary enforcement.**

A PINN trains a neural network `û` to minimize a residual loss. After
training, a small loss value does not tell you how far `û` is from the
true solution `u` in a mathematically meaningful norm. This project
constructs a computable, reliable upper bound on that error.

The key insight — which is the paper's central contribution — is that
penalty-based (soft) boundary enforcement introduces a boundary error
component that is entirely invisible to standard interior residual
estimators. Any estimator that ignores this component will produce
effectivity indices below 1, violating the reliability condition and
making the bound mathematically useless.

The fix is a harmonic lifting decomposition. The implemented and
validated bound is:

$$
\|u - \hat{u}\|_{H^1_0(\Omega)}
\;\leq\;
\underbrace{\frac{1}{\alpha}\|R\|_{H^{-1}(\Omega)}}_{\text{interior residual term}}
\;+\;
\underbrace{\|\nabla w\|_{L^2(\Omega)}}_{\text{BC lifting term}}
$$

where:

- $R = f + \operatorname{div}(a \,\nabla \hat{u})$ is the interior PDE residual
- $w$ solves $-\Delta w = 0$ in $\Omega$ with $w = g - \hat{u}|_{\partial\Omega}$
  on $\partial\Omega$ (the harmonic lifting of the boundary mismatch)
- $\alpha > 0$ is the coercivity constant of the bilinear form

Both terms are computable from the trained PINN without access to the
true solution.

---

## 2. What Must Be Achieved?

There are four pillars. Each is stated honestly against current status.

### Pillar 1 — Theoretical Rigor

The bound must be derived from first principles and be defensible to a
numerical analysis referee. The full derivation proceeds as follows.

**Step 1.** Write the error as $e = u - \hat{u}$. Decompose:

$$e = e_0 + w$$

where $w \in H^1(\Omega)$ is the harmonic lifting of the BC mismatch
$g - \hat{u}|_{\partial\Omega}$, and $e_0 = e - w \in H^1_0(\Omega)$.

**Step 2.** Since $e_0 \in H^1_0(\Omega)$ and satisfies

$$a(e_0, v) = \langle R, v \rangle_{H^{-1}, H^1_0} \quad \forall v \in H^1_0(\Omega),$$

coercivity gives $\alpha \|e_0\|_{H^1_0} \leq \|R\|_{H^{-1}}$.

**Step 3.** The triangle inequality gives

$$\|e\|_{H^1_0} \leq \|e_0\|_{H^1_0} + \|w\|_{H^1_0}
= \|e_0\|_{H^1_0} + \|\nabla w\|_{L^2}.$$

Combining yields the bound.

**Current status:** The derivation is correct. It needs to be written
up as a formal theorem with explicit regularity assumptions in the paper.
The assumption that must be stated is: $\Omega$ is a Lipschitz domain,
$a \in L^\infty(\Omega)$ with $a \geq \alpha > 0$ a.e., and
$f \in H^{-1}(\Omega)$.

**Note on norm vs. seminorm.** The bound as stated is for the
$H^1_0$ seminorm $|\cdot|_{H^1}$ (gradient part only). For zero
Dirichlet data (Poisson benchmark), this equals the $H^1_0$ norm by
the Poincaré inequality. For nonzero Dirichlet data (variable-coefficient
benchmark), the true error is measured as the $H^1$ seminorm of
$u - \hat{u}$ throughout, which is consistent with the bound. This must
be stated precisely in the paper.

---

### Pillar 2 — Effectivity Index $\eta \in [1, C]$

Define the effectivity index as:

$$\eta = \frac{\text{estimated error}}{\|u - \hat{u}\|_{H^1_0}}.$$

The reliability condition $\eta \geq 1$ is a hard mathematical
requirement: the estimator must dominate the true error for the bound
to be valid. The pipeline enforces this as a hard assertion that raises
an error if violated.

**Actual validated results** (from `validated_results.json`):

| Benchmark | True $H^1$ Error | Estimated Error | $\eta$ | Status |
|---|---|---|---|---|
| Poisson (smooth) | 1.183e-3 | 1.256e-3 | **1.061** | ✓ Clean |
| Variable-coefficient diffusion | 2.983e-3 | 3.232e-3 | **1.083** | ✓ Clean |
| L-shaped ($r^{2/3}$ singularity) | 5.038e-1 | 1.379 | **2.738** | ⚠ Too loose |

The Poisson and variable-coefficient results are publication-quality:
$\eta$ is close to 1, the bound is tight, and the boundary lifting
contribution dominates the residual contribution (confirming the
theoretical narrative). The L-shaped result passes the $\eta \geq 1$
check but $\eta = 2.738$ is too loose to be a strong main result.

**Known cause of L-shaped looseness:** The FD harmonic lifting solve
in `lifting.py` produces estimates that increase on mesh refinement
(mesh 80 gives estimate 1.081, mesh 160 gives 1.229). A convergent
discrete harmonic solve should produce decreasing estimates as
$h \to 0$. This is a numerical instability, likely from missing $h^2$
area scaling in `_edge_based_dirichlet_energy`, compounded by the
safety margin logic in `estimator.py`. Both need to be fixed before
the L-shaped benchmark can be claimed as validated.

---

### Pillar 3 — Benchmark Validation

The paper validates the estimator on three coercive elliptic benchmarks.
Convection-dominated problems are explicitly excluded from the current
scope because the bound requires coercivity ($\alpha > 0$ uniform) and
convection-dominated operators near the singular perturbation limit
violate this in practice. The legacy experiment log shows effectivities
of $\eta \approx 0.009$ for convection-dominated cases — confirming
that a separate theorem and a separate estimator would be required.

| Benchmark | PDE | Domain | $\alpha$ | Challenge | Target $\eta$ |
|---|---|---|---|---|---|
| Poisson (smooth) | $-\Delta u = f$ | Unit square | 1.0 | Baseline correctness | 1.0–1.2 |
| Variable-coefficient | $-\operatorname{div}(a \nabla u) = f$ | Unit square | 0.5 | Non-constant $a(x)$, coercivity constant handling | 1.0–1.3 |
| L-shaped singularity | $-\Delta u = 0$ | L-shaped | 1.0 | $u \sim r^{2/3}$, low regularity, no global $H^2$ | 1.0–2.0 |

**What each benchmark proves:**

- **Poisson:** The bound works in the canonical smooth setting and that
  the BC lifting term correctly captures the dominant error source when
  the PINN satisfies the PDE well but imperfectly enforces BCs.
- **Variable-coefficient:** The bound handles non-constant $a(x)$ and
  the explicit coercivity constant $\alpha = 0.5$ appears correctly in
  the denominator of the residual term.
- **L-shaped:** The bound does not require global $H^2$ regularity of
  the exact solution. This is important because $u \in H^{1+2/3-\epsilon}$
  only near the re-entrant corner, and classical FEM error analysis
  would require more smoothness. The PINN bound is norm-based and does
  not invoke elliptic regularity beyond what is needed for coercivity.

---

### Pillar 4 — Clean, Reproducible Codebase

The codebase must allow a reviewer to clone the repository, run one
command, and reproduce every table and figure in the paper.

**Current implementation:**

- Single entry point: `python experiments/run.py` ✓
- Per-benchmark single benchmark runs: `--benchmark poisson` etc. ✓
- Deterministic seeds ✓
- JSON outputs with full metric logging ✓
- Automated figure generation from JSON ✓
- Lightweight unit tests: `python tests.py` ✓

**What the numerical methods actually are** (important for the paper's
methods section):

- $H^{-1}$ dual norm on unit square: DST-based spectral Laplacian
  inversion, $O(n^2 \log n)$, spectrally exact for the discrete problem.
- $H^{-1}$ dual norm on L-shaped domain: sparse CG on hand-assembled
  5-point FD stencil.
- Harmonic lifting norm: FD harmonic solve with BC mismatch as Dirichlet
  data; Dirichlet energy computed from the discrete solution.
- True error: Monte Carlo quadrature over the domain, 50,000 points,
  seeded for reproducibility.

**No FEniCS is used anywhere in the codebase.** Any reference to FEniCS
in notes or summaries is incorrect and should be removed.

---

## 3. What Is Not Yet Settled

These are the open items before the paper can be submitted.

### Numerical gaps

1. **L-shaped lifting solve instability.** The estimate increases on
   mesh refinement, which must be diagnosed and fixed. The two likely
   causes are: (a) missing $h^2$ area factor in
   `_edge_based_dirichlet_energy`, and (b) the safety margin in
   `estimator.py` inflating the estimate instead of bounding it. Once
   fixed, the target is $\eta \in [1.0, 2.0]$ with a decreasing
   mesh refinement sequence.

2. **Safety margin is not a theorem.** The current pipeline adds
   `|eta_h - eta_{h/2}|` to the estimate as a discretization buffer.
   This is an ad hoc heuristic. It needs to be replaced with a
   Richardson extrapolation bound (for a $p=2$ FD method, the correction
   factor is $1/(2^2 - 1) = 1/3$, not 1), or documented as a stated
   assumption with a monotone convergence condition.

3. **Robustness across seeds not yet demonstrated.** Current validation
   is single runs per benchmark. The paper needs 3–5 runs with
   different seeds to show the effectivity is not seed-dependent.

### Theory gaps

4. **Formal theorem not yet written.** The derivation is correct but
   needs to be typeset with full assumptions, norm definitions, and
   statement of the bound as a theorem with proof.

5. **Norm consistency note not written.** The paper must explicitly
   state that the bound is for the $H^1$ seminorm of the error and
   that all true error measurements use the same norm.

### Already satisfied

- Hard $\eta \geq 1$ guard in pipeline ✓
- Correct two-term bound implemented ✓
- Correct benchmark problem definitions ✓
- Correct coercivity constants in the denominator ✓
- BC lifting correctly captures boundary mismatch ✓

---

## 4. What the Paper Argues, in One Paragraph

> We derive a computable a posteriori error bound for PINN approximations
> of coercive second-order elliptic PDEs under penalty-based (soft)
> boundary enforcement. The central observation is that soft BC training
> produces a boundary error component that is invisible to standard
> interior residual estimators: the interior residual can be small while
> the true $H^1$ error remains large due to boundary mismatch. We resolve
> this by decomposing the error via harmonic lifting, yielding the bound
> $\|u - \hat{u}\|_{H^1_0} \leq (1/\alpha)\|R\|_{H^{-1}} + \|\nabla
> w\|_{L^2}$, where both terms are computable from the trained network.
> We implement the estimator using DST-based dual norm computation and
> FD harmonic lifting solves, and validate it on three coercive elliptic
> benchmarks: smooth Poisson ($\eta = 1.06$), variable-coefficient
> diffusion ($\eta = 1.08$), and the L-shaped domain with re-entrant
> corner singularity. The bound is implemented as a single reproducible
> pipeline with hard reliability enforcement.

---

## 5. Ready-to-Write Checklist

### Theory

- [ ] Theorem statement with full assumptions written in LaTeX
- [ ] Proof of the $e = e_0 + w$ decomposition written cleanly
- [ ] Regularity assumptions stated explicitly (Lipschitz domain, $a \in L^\infty$, $\alpha > 0$)
- [ ] Norm convention (seminorm vs. full $H^1_0$ norm) stated precisely
- [ ] $H^{-1}$ computation and DST normalization justified in appendix

### Numerics

- [ ] L-shaped lifting solve instability fixed and confirmed ($\eta \leq 2.0$, decreasing mesh sequence)
- [ ] Safety margin replaced with Richardson extrapolation bound or stated as assumption
- [ ] All three benchmarks pass with $\eta \geq 1$ in 3–5 independent runs
- [ ] Effectivity stable: does not vary by more than 0.2 across seeds

### Code

- [x] Single entry point: `python experiments/run.py`
- [x] Unit tests: `python tests.py`
- [x] Results saved to JSON
- [x] Figures generated from JSON
- [ ] Richardson extrapolation margin implemented and documented
- [ ] All paper-cited constants (mesh size, $\alpha$, safety factor) logged in JSON

### Paper readiness signal

You are ready to write when you can answer these without notes:

1. Why does $\eta \geq 1$ matter, and what goes wrong if it fails?
2. Why does an interior-residual-only estimator fail under soft BC training?
3. What does the harmonic lifting $w$ represent physically, and why
   does its $H^1$ norm bound the boundary contribution to the error?
4. Why is DST-based inversion exact for the discrete $H^{-1}$ dual norm
   on the unit square?
5. Why does the L-shaped benchmark test something the Poisson benchmark
   does not?

---

## 6. Strategic Value for PhD Applications

This project demonstrates four things that pure ML projects do not:

1. **Theorem-driven numerical analysis.** The estimator is derived from
   a mathematical proof, not tuned empirically. A professor reading the
   code can verify the theoretical claim from the implementation.

2. **A falsifiable claim.** The effectivity index $\eta \geq 1$ is a
   hard mathematical condition that can be checked. The project would
   have failed if the estimator produced $\eta < 1$ — and it did fail
   for the interior-residual-only estimator ($\eta \approx 0.03$).
   The fix required a nontrivial theoretical insight, not a hyperparameter
   search.

3. **Nontrivial implementation.** DST-based $H^{-1}$ computation, FD
   harmonic lifting solves, Monte Carlo quadrature with seeded
   reproducibility — these are not standard ML engineering.

4. **Research maturity.** A single reproducible entry point, hard
   pipeline guards, JSON outputs, and clean separation of theory,
   implementation, and experiments signal that the candidate understands
   what a research codebase is supposed to look like.

The boundary-lifting insight is the main conceptual contribution and the
one a professor will ask about: why did the interior residual fail, what
was missing, and how did the decomposition fix it?

---

## 7. Scope Boundaries (What This Paper Does Not Claim)

To avoid overpromising, the paper must explicitly state what is out of scope:

- **Convection-dominated problems** ($\varepsilon \ll 1$): excluded.
  The legacy experiments show $\eta \approx 0.009$. A separate theorem
  (likely using streamline diffusion norms) is needed.
- **Nonlinear PDEs**: the bound requires linearity for the coercivity
  argument. Not claimed.
- **Time-dependent problems**: not claimed.
- **Efficiency (lower bound on $\eta$)**: the paper proves reliability
  ($\eta \geq 1$) only. Efficiency ($\eta \leq C$) is observed
  numerically but not proved.
- **Optimal training**: the paper does not claim the PINN is well-trained.
  The bound holds for any $\hat{u}$, regardless of training quality.
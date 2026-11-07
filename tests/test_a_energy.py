"""Torch-free tests of the a-energy estimator (Theorem A).  Run: ``python -m pytest tests/test_a_energy.py``
or ``python tests/test_a_energy.py``."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.a_energy import (  # noqa: E402
    a_energy_estimator, assemble_stiffness, build_mesh, estimator_from_vectors,
    true_error_a_quadrature, unit_square_mesh,
)

PI = np.pi


# ---------------------------------------------------------------- helpers
def _manufactured():
    import sympy as sp

    x, y = sp.symbols("x y")
    a = 1 + sp.Rational(1, 2) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    u = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) * (1 + x**2 * y)
    f = -(sp.diff(a * sp.diff(u, x), x) + sp.diff(a * sp.diff(u, y), y))
    lam = lambda e: sp.lambdify((x, y), e, "numpy")  # noqa: E731
    a_n, u_n, f_n = lam(a), lam(u), lam(f)
    gx, gy = lam(sp.diff(u, x)), lam(sp.diff(u, y))
    return (
        lambda p: a_n(p[:, 0], p[:, 1]) * np.ones(len(p)),
        lambda p: u_n(p[:, 0], p[:, 1]),
        lambda p: f_n(p[:, 0], p[:, 1]),
        lambda p: np.stack([gx(p[:, 0], p[:, 1]), gy(p[:, 0], p[:, 1])], axis=1),
    )


def _perturbed(u, grad_u, pert, grad_pert):
    """uhat = u + pert (so e = u - uhat = -pert)."""
    return lambda p: (u(p) + pert(p), grad_u(p) + grad_pert(p))


# ---------------------------------------------------------------- tests
def test_discrete_pythagoras_is_exact_on_random_data():
    rng = np.random.default_rng(0)
    for domain in ("unit_square", "l_shaped"):
        mesh = build_mesh(domain, 12)
        a_cell = rng.uniform(0.5, 3.0, mesh.tris.shape[0])   # rough coefficient: no smoothness needed
        K = assemble_stiffness(mesh, a_cell)
        e = rng.standard_normal(mesh.n_nodes)                  # arbitrary nodal error
        dual, lift, w = estimator_from_vectors(K, K @ e, e, mesh)
        total = float(e @ (K @ e))
        assert abs(total - (dual**2 + lift**2)) <= 1e-12 * total, (domain, total, dual**2 + lift**2)
        # w is a_h-harmonic and e - w is in V_h0 and a_h-orthogonal to w
        e0 = e - w
        assert np.max(np.abs(e0[mesh.boundary])) < 1e-14
        assert abs(float(e0 @ (K @ w))) <= 1e-12 * total


def test_affine_exact_solution_gives_eta_equal_error():
    # u = x, uhat = 0, a = 2, f = 0: R = 0, e = x, ||e||_a^2 = 2.
    for domain in ("unit_square", "l_shaped"):
        mesh = build_mesh(domain, 8)
        res = a_energy_estimator(
            mesh, lambda p: 2.0 * np.ones(len(p)), lambda p: np.zeros(len(p)), lambda p: p[:, 0],
            lambda p: (np.zeros(len(p)), np.zeros((len(p), 2))),
        )
        area = 1.0 if domain == "unit_square" else 3.0
        assert abs(res.dual_norm) < 1e-12
        assert abs(res.eta - np.sqrt(2.0 * area)) < 1e-12


def test_variable_a_converges_and_dual_norm_is_monotone_lower_bound():
    a, u, f, gu = _manufactured()
    pert = lambda p: 0.1 * np.cos(2 * p[:, 0] + p[:, 1])                     # nonzero on the boundary
    gpert = lambda p: 0.1 * np.stack([-2 * np.sin(2 * p[:, 0] + p[:, 1]), -np.sin(2 * p[:, 0] + p[:, 1])], axis=1)
    uhat = _perturbed(u, gu, pert, gpert)
    true = true_error_a_quadrature(unit_square_mesh(64), a, gu, uhat, order=5)
    errs, duals = [], []
    for n in (8, 16, 32, 64):
        r = a_energy_estimator(unit_square_mesh(n), a, f, u, uhat, quad_order=4)
        errs.append(abs(r.eta / true - 1.0))
        duals.append(r.dual_norm)
    assert all(d2 >= d1 - 1e-12 for d1, d2 in zip(duals, duals[1:])), duals   # nested spaces
    assert errs[-1] < 1e-3, errs
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert rates[-1] > 1.6, (errs, rates)                                      # ~ O(h^2)


def test_hard_bc_case_lifting_vanishes_and_eta_is_dual_norm():
    a, u, f, gu = _manufactured()
    bub = lambda p: p[:, 0] * (1 - p[:, 0]) * p[:, 1] * (1 - p[:, 1])
    gbub = lambda p: np.stack([(1 - 2 * p[:, 0]) * p[:, 1] * (1 - p[:, 1]),
                               p[:, 0] * (1 - p[:, 0]) * (1 - 2 * p[:, 1])], axis=1)
    pert = lambda p: 0.2 * bub(p) * np.cos(3 * p[:, 0])
    gpert = lambda p: 0.2 * (gbub(p) * np.cos(3 * p[:, 0])[:, None]
                             + bub(p)[:, None] * np.stack([-3 * np.sin(3 * p[:, 0]), 0 * p[:, 0]], axis=1))
    uhat = _perturbed(u, gu, pert, gpert)
    true = true_error_a_quadrature(unit_square_mesh(64), a, gu, uhat, order=5)
    r = a_energy_estimator(unit_square_mesh(64), a, f, u, uhat, quad_order=4)
    assert r.lifting_norm < 1e-12
    assert abs(r.dual_norm / true - 1.0) < 1e-3


def test_noncontrollability_family_loss_to_zero_but_energy_error_stays():
    """uhat_k = -c k^{-1/2} sin(k pi x) sinh(k pi (1-y)) / sinh(k pi), f = 0, g = 0, u = 0.

    Harmonic => R = 0; boundary mismatch lives on y = 0 with L2(Gamma)^2 = c^2/(2k) -> 0, while
    ||grad(u - uhat_k)||^2 = (c^2 pi / 2) coth(k pi) does not tend to 0."""
    c = 1.0
    for k in (1, 2, 4):
        s = k * PI

        def uhat(p, s=s, k=k):
            x, y = p[:, 0], p[:, 1]
            sh = np.sinh(s * (1 - y)) / np.sinh(s)
            ch = np.cosh(s * (1 - y)) / np.sinh(s)
            u = -c * k**-0.5 * np.sin(s * x) * sh
            gx = -c * k**-0.5 * s * np.cos(s * x) * sh
            gy = c * k**-0.5 * np.sin(s * x) * s * ch
            return u, np.stack([gx, gy], axis=1)

        exact_sq = 0.5 * c**2 * PI / np.tanh(s)
        r = a_energy_estimator(unit_square_mesh(128), lambda p: np.ones(len(p)), lambda p: np.zeros(len(p)),
                               lambda p: np.zeros(len(p)), uhat, quad_order=4)
        assert r.dual_norm < 2e-3, (k, r.dual_norm)
        assert abs(r.eta**2 / exact_sq - 1.0) < 0.03, (k, r.eta**2, exact_sq)
        quad = true_error_a_quadrature(unit_square_mesh(128), lambda p: np.ones(len(p)),
                                       lambda p: np.zeros((len(p), 2)), uhat, order=5)
        assert abs(quad**2 / exact_sq - 1.0) < 1e-4, (k, quad**2, exact_sq)
        assert c**2 / (2 * k) <= 0.5 / k + 1e-15   # boundary L2^2 -> 0 while the energy stays ~ c^2 pi/2


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)


if __name__ == "__main__":
    _run_all()

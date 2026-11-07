"""Torch-free tests of reference errors, boundary diagnostics and the one-call evaluator."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.a_energy import (  # noqa: E402
    boundary_norms, evaluate_from_callables, lshape_grad_u, lshape_true_error_boundary_integral,
    lshape_true_error_graded, lshape_u,
)

PI = np.pi


def _smooth_uhat(p):
    """A smooth, non-harmonic, boundary-violating 'network output' on the L-shape."""
    x, y = p[:, 0], p[:, 1]
    u = 0.3 * np.sin(1.3 * x + 0.4) * np.cos(0.9 * y) + 0.2 * x * y + 0.1 * (x**2 - y)
    gx = 0.3 * 1.3 * np.cos(1.3 * x + 0.4) * np.cos(0.9 * y) + 0.2 * y + 0.2 * x
    gy = -0.3 * 0.9 * np.sin(1.3 * x + 0.4) * np.sin(0.9 * y) + 0.2 * x - 0.1
    return u, np.stack([gx, gy], axis=1)


def test_lshape_exact_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1, 1, (400, 2))
    pts = pts[~((pts[:, 0] > 0) & (pts[:, 1] < 0))]
    pts = pts[(np.hypot(pts[:, 0], pts[:, 1]) > 0.1) & (np.abs(pts[:, 1]) > 0.05)]
    h = 1e-6
    fd = np.stack([(lshape_u(pts + [h, 0]) - lshape_u(pts - [h, 0])) / (2 * h),
                   (lshape_u(pts + [0, h]) - lshape_u(pts - [0, h])) / (2 * h)], axis=1)
    assert np.max(np.abs(fd - lshape_grad_u(pts))) < 1e-6


def test_two_independent_L_shape_references_agree():
    graded = lshape_true_error_graded(_smooth_uhat, None, 48, 32)
    graded2 = lshape_true_error_graded(_smooth_uhat, None, 96, 64)
    bi = lshape_true_error_boundary_integral(_smooth_uhat, n_ref=128)
    assert abs(graded - graded2) / graded2 < 1e-10          # graded rule is converged
    assert abs(bi - graded2) / graded2 < 1e-6, (bi, graded2)  # independent Green's-formula value


def test_graded_rule_matches_green_formula_on_singular_integrand():
    # int_L |grad u|^2 with uhat = 0; compare with the boundary integral (Green) value.
    zero = lambda p: (np.zeros(len(p)), np.zeros((len(p), 2)))  # noqa: E731
    graded = lshape_true_error_graded(zero, None, 96, 64)
    bi = lshape_true_error_boundary_integral(zero)
    assert abs(graded - bi) / bi < 1e-9, (graded, bi)


def test_boundary_norms_of_noncontrollability_family():
    c = 1.0
    for k in (1, 3):
        s = k * PI

        def uhat(p, s=s, k=k):
            x, y = p[:, 0], p[:, 1]
            sh, ch = np.sinh(s * (1 - y)) / np.sinh(s), np.cosh(s * (1 - y)) / np.sinh(s)
            return (-c * k**-0.5 * np.sin(s * x) * sh,
                    np.stack([-c * k**-0.5 * s * np.cos(s * x) * sh, c * k**-0.5 * np.sin(s * x) * s * ch], axis=1))

        b = boundary_norms("unit_square", lambda p: np.zeros(len(p)), None, uhat)
        assert abs(b["l2"] ** 2 - c**2 / (2 * k)) < 1e-12
        assert abs(b["tangential_h1"] ** 2 - c**2 * k * PI**2 / 2) < 1e-9


def test_evaluate_from_callables_lshape_and_square():
    # L-shape, f = 0, a = 1: boundary-violating smooth uhat.  eta approaches the graded true error.
    zero = lambda p: np.zeros(len(p))  # noqa: E731
    one = lambda p: np.ones(len(p))    # noqa: E731
    res = evaluate_from_callables("l_shaped", one, zero, lshape_u, lshape_grad_u, None, _smooth_uhat,
                                  ladder=(16, 32, 64), ref_cells=64)
    ratios = [lv["eta_over_true"] for lv in res["levels"]]
    assert res["ref_method"] == "graded_polar" and res["ref_self_convergence_rel"] < 1e-9
    assert abs(ratios[-1] - 1) < 5e-3 and abs(ratios[-1] - 1) < abs(ratios[0] - 1), ratios
    lo, hi = res["unweighted_interval_finest"]
    assert lo <= res["true_error_grad"] * (1 + 5e-3) and res["true_error_grad"] <= hi * (1 + 5e-3)
    assert res["boundary"]["l2"] > 0


def _run_all():
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("PASS", k)


if __name__ == "__main__":
    _run_all()

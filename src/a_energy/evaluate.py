
"""One-call evaluation of the a-energy estimator on a mesh ladder (torch-free)."""

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .estimator import (
    a_energy_estimator,
    true_error_a_quadrature,
)
from .fem import UhatEval
from .mesh import build_mesh, unit_square_mesh
from .references import (
    boundary_norms,
    lshape_grad_u,
    lshape_true_error_graded,
)

Scalar = Callable[[np.ndarray], np.ndarray]


def _one(p: np.ndarray) -> np.ndarray:
    return np.ones(len(p), dtype=np.float64)


def reference_errors(
    domain: str,
    a_fun: Scalar,
    grad_u_exact: Callable,
    uhat_eval: UhatEval,
    ref_cells: int = 128,
    order: int = 5,
) -> Dict:
    """Compute true errors and a reference self-convergence diagnostic."""

    out = {}

    if domain == "unit_square":

        def run(n, a):
            return true_error_a_quadrature(
                unit_square_mesh(n),
                a,
                grad_u_exact,
                uhat_eval,
                order,
            )

        fine_a = run(ref_cells, a_fun)
        coarse_a = run(max(ref_cells // 2, 8), a_fun)

        out.update(
            true_error_a=fine_a,
            true_error_grad=run(ref_cells, _one),
            ref_method="gauss_mesh",
            ref_self_convergence_rel=(
                abs(fine_a - coarse_a) / max(fine_a, 1e-300)
            ),
        )

    elif domain == "l_shaped":

        coarse = lshape_true_error_graded(
            uhat_eval, a_fun, 48, 32
        )
        fine = lshape_true_error_graded(
            uhat_eval, a_fun, 96, 64
        )

        out.update(
            true_error_a=fine,
            true_error_grad=lshape_true_error_graded(
                uhat_eval, None, 96, 64
            ),
            ref_method="graded_polar",
            ref_self_convergence_rel=(
                abs(fine - coarse) / max(fine, 1e-300)
            ),
        )

    else:
        raise ValueError(f"Unsupported domain '{domain}'")

    return out


def evaluate_from_callables(
    domain: str,
    a_fun: Scalar,
    f_fun: Scalar,
    g_fun: Scalar,
    grad_g_fun: Optional[Callable],
    grad_u_exact: Optional[Callable],
    uhat_eval: UhatEval,
    alpha: float = 1.0,
    beta: float = 1.0,
    ladder: Sequence[int] = (32, 64, 128),
    quad_order: int = 4,
    ref_cells: int = 128,
) -> Dict:
    """Evaluate estimator, reference errors and boundary diagnostics."""

    if not ladder:
        raise ValueError("Mesh ladder cannot be empty.")

    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive.")

    if domain == "l_shaped" and grad_u_exact is None:
        grad_u_exact = lshape_grad_u

    levels = []

    for n in ladder:
        r = a_energy_estimator(
            build_mesh(domain, n),
            a_fun,
            f_fun,
            g_fun,
            uhat_eval,
            quad_order,
        )
        levels.append(r.to_dict())

    ref = (
        reference_errors(
            domain,
            a_fun,
            grad_u_exact,
            uhat_eval,
            ref_cells,
        )
        if grad_u_exact is not None
        else {}
    )

    true_a = ref.get("true_error_a")

    for lv in levels:
        lv["eta_over_true"] = (
            lv["eta"] / true_a
            if true_a is not None and true_a > 0
            else None
        )

    finest = levels[-1]

    interval = sorted([
        finest["eta"] / np.sqrt(alpha),
        finest["eta"] / np.sqrt(beta),
    ])

    out = {
        "levels": levels,
        "alpha": float(alpha),
        "beta": float(beta),
        "eta_a_finest": finest["eta"],
        "unweighted_interval_finest": interval,
        "boundary": boundary_norms(
            domain, g_fun, grad_g_fun, uhat_eval
        ),
        **ref,
    }

    if true_a is not None:
        out["pythagoras_gap_finest"] = float(
            true_a**2 - finest["eta"]**2
        )

    return out

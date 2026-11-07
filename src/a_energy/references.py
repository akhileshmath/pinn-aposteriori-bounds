"""Reference (true) errors and boundary diagnostics.  NumPy only, float64.

L-shaped domain.  u = r^{2/3} sin(2 theta/3) has |grad u|^2 ~ r^{-2/3}, so a plain mesh quadrature of
|grad u - grad uhat|^2 converges slowly (this is why the earlier reference error was untrusted).
Two independent evaluations are provided:

* ``lshape_true_error_graded``:   int_L a |grad u - grad uhat|^2 over six triangles with apex at
  the corner, collapsed with the grading rho = s^q (q = 3) so that the integrand becomes analytic.
  Works for any smooth a and any smooth uhat; no cancellation.
* ``lshape_true_error_boundary_integral``:  for f = 0, a = 1 (u harmonic), Green's formula gives
      ||grad e||^2 = ||grad uhat||^2 + oint (u - 2 uhat) d_n u ds,
  where the two edges touching the corner are integrated with Gauss-Jacobi (weight r^{-1/3}).
  Used as a cross-check; it suffers cancellation when ||grad e|| << ||grad uhat||.
"""
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.special import roots_jacobi

from .estimator import true_error_a_quadrature
from .fem import UhatEval
from .mesh import l_shaped_mesh

Scalar = Callable[[np.ndarray], np.ndarray]


# ------------------------------------------------------------------ L-shape exact solution
def lshape_u(p: np.ndarray) -> np.ndarray:
    r = np.hypot(p[:, 0], p[:, 1])
    th = np.arctan2(p[:, 1], p[:, 0])
    th = np.where(th < 0, th + 2 * np.pi, th)
    return r ** (2.0 / 3.0) * np.sin(2.0 * th / 3.0)


def lshape_grad_u(p: np.ndarray) -> np.ndarray:
    r = np.hypot(p[:, 0], p[:, 1])
    th = np.arctan2(p[:, 1], p[:, 0])
    th = np.where(th < 0, th + 2 * np.pi, th)
    dr = (2.0 / 3.0) * r ** (-1.0 / 3.0) * np.sin(2.0 * th / 3.0)
    dth = (2.0 / 3.0) * r ** (2.0 / 3.0) * np.cos(2.0 * th / 3.0)
    c, s = np.cos(th), np.sin(th)
    return np.stack([c * dr - s / r * dth, s * dr + c / r * dth], axis=1)


# ------------------------------------------------------------------ graded polar quadrature
_L_TRIANGLES = [  # six triangles with apex O = (0,0) covering the L-shaped domain
    ((1, 0), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (-1, 1)),
    ((-1, 1), (-1, 0)), ((-1, 0), (-1, -1)), ((-1, -1), (0, -1)),
]


def lshape_graded_rule(n_s: int = 48, n_t: int = 32, q: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    xs, ws = np.polynomial.legendre.leggauss(n_s)
    xt, wt = np.polynomial.legendre.leggauss(n_t)
    xs, ws, xt, wt = 0.5 * (xs + 1), 0.5 * ws, 0.5 * (xt + 1), 0.5 * wt
    S, T = np.meshgrid(xs, xt, indexing="ij")
    W0 = (ws[:, None] * wt[None, :])
    rho = S**q
    drho = q * S ** (q - 1.0)
    pts, wts = [], []
    for p1, p2 in _L_TRIANGLES:
        p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        cross = abs(p1[0] * p2[1] - p1[1] * p2[0])
        pos = rho[..., None] * ((1 - T)[..., None] * p1 + T[..., None] * p2)
        pts.append(pos.reshape(-1, 2))
        wts.append((W0 * drho * rho * cross).ravel())
    return np.concatenate(pts), np.concatenate(wts)


def lshape_true_error_graded(
    uhat_eval: UhatEval, a_fun: Scalar = None, n_s: int = 48, n_t: int = 32
) -> float:
    pts, w = lshape_graded_rule(n_s, n_t)
    _, g_hat = uhat_eval(pts)
    diff = lshape_grad_u(pts) - np.asarray(g_hat)
    a = 1.0 if a_fun is None else np.asarray(a_fun(pts), dtype=np.float64).reshape(-1)
    return float(np.sqrt(max(np.sum(w * a * np.sum(diff**2, axis=1)), 0.0)))


# ------------------------------------------------------------------ boundary-integral cross-check
_L_EDGES = [  # (start, end, outward normal, touches_corner)
    ((-1.0, -1.0), (0.0, -1.0), (0.0, -1.0), False),
    ((0.0, -1.0), (0.0, 0.0), (1.0, 0.0), True),
    ((0.0, 0.0), (1.0, 0.0), (0.0, -1.0), True),
    ((1.0, 0.0), (1.0, 1.0), (1.0, 0.0), False),
    ((1.0, 1.0), (-1.0, 1.0), (0.0, 1.0), False),
    ((-1.0, 1.0), (-1.0, -1.0), (-1.0, 0.0), False),
]
_SQUARE_EDGES = [
    ((0.0, 0.0), (1.0, 0.0), (0.0, -1.0), False),
    ((1.0, 0.0), (1.0, 1.0), (1.0, 0.0), False),
    ((1.0, 1.0), (0.0, 1.0), (0.0, 1.0), False),
    ((0.0, 1.0), (0.0, 0.0), (-1.0, 0.0), False),
]


def _edge_rule(p0, p1, panels: int = 16, order: int = 10):
    """Composite Gauss-Legendre points, weights (arc length) and unit tangent on one edge."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    length = np.linalg.norm(p1 - p0)
    tau = (p1 - p0) / length
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0.0, 1.0, panels + 1)
    t = (0.5 * (edges[1:] - edges[:-1])[:, None] * (x[None, :] + 1) + edges[:-1, None]).ravel()
    wt = (0.5 * (edges[1:] - edges[:-1])[:, None] * w[None, :]).ravel() * length
    return p0[None, :] + t[:, None] * (p1 - p0)[None, :], wt, tau


def lshape_true_error_boundary_integral(uhat_eval: UhatEval, n_ref: int = 128, order: int = 5,
                                        n_gj: int = 48) -> float:
    """||grad(u - uhat)|| for the L-shape benchmark (f = 0, a = 1) via Green's formula."""
    mesh = l_shaped_mesh(n_ref)
    zero = lambda p: np.zeros((len(p), 2))  # noqa: E731
    # ||grad uhat||^2 = true_error_a_quadrature with grad u := 0
    grad_hat_sq = true_error_a_quadrature(mesh, lambda p: np.ones(len(p)), zero, uhat_eval, order) ** 2
    total = 0.0
    xg, wg = roots_jacobi(n_gj, 0.0, -1.0 / 3.0)       # weight (1+x)^(-1/3) on [-1,1]
    for p0, p1, nrm, corner in _L_EDGES:
        if corner:
            # u = 0 on these edges and d_n u = -(2/3) r^{-1/3}  =>  integrand (4/3) uhat r^{-1/3}
            origin = np.zeros(2)
            far = np.asarray(p0 if np.allclose(p1, 0.0) else p1, float)   # the end that is not the corner
            r = 0.5 * (xg + 1.0)
            pts = origin[None, :] + r[:, None] * far[None, :]
            uh = uhat_eval(pts)[0]
            total += 2.0 ** (-2.0 / 3.0) * float(np.sum(wg * (4.0 / 3.0) * uh))
        else:
            pts, w, _ = _edge_rule(p0, p1)
            dn = lshape_grad_u(pts) @ np.asarray(nrm)
            total += float(np.sum(w * (lshape_u(pts) - 2.0 * uhat_eval(pts)[0]) * dn))
    return float(np.sqrt(max(grad_hat_sq + total, 0.0)))


# ------------------------------------------------------------------ boundary diagnostics
def boundary_norms(
    domain: str,
    g_fun: Scalar,
    grad_g_fun: Callable[[np.ndarray], np.ndarray],
    uhat_eval: UhatEval,
    panels: int = 16,
    order: int = 10,
) -> Dict[str, float]:
    """||g - uhat||_{L2(Gamma)} and the tangential seminorm ||d_tau (g - uhat)||_{L2(Gamma)}
    (sum over the straight edges).  ``grad_g_fun`` may be None when g == 0."""
    edges = _SQUARE_EDGES if domain == "unit_square" else _L_EDGES if domain == "l_shaped" else None
    if edges is None:
        raise ValueError(f"Unsupported domain '{domain}'")
    l2_sq = tan_sq = 0.0
    for p0, p1, _, _ in edges:
        pts, w, tau = _edge_rule(p0, p1, panels, order)
        u_hat, g_hat = uhat_eval(pts)
        d = np.asarray(g_fun(pts), dtype=np.float64).reshape(-1) - u_hat
        dg = 0.0 if grad_g_fun is None else np.asarray(grad_g_fun(pts)) @ tau
        dtau = dg - np.asarray(g_hat) @ tau
        l2_sq += float(np.sum(w * d**2))
        tan_sq += float(np.sum(w * dtau**2))
    return {"l2": float(np.sqrt(l2_sq)), "tangential_h1": float(np.sqrt(tan_sq))}

"""P1 finite-element building blocks for the a-energy estimator (NumPy/SciPy only).

Everything is evaluated in float64.  The discrete bilinear form is

    a_h(v, w) = sum_T abar_T |T| grad v . grad w,      abar_T = cell average of a,

for P1 functions v, w.  The *same* form is used for the dual norm and for the lifting, which is
what makes the discrete Pythagorean identity exact (see ``estimator.py``).
"""
from typing import Callable, Tuple

import numpy as np
from scipy.sparse import coo_matrix

from .mesh import TriMesh

Scalar = Callable[[np.ndarray], np.ndarray]                      # (M,2) -> (M,)
UhatEval = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]  # (M,2) -> (u (M,), grad (M,2))


def triangle_rule(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Collapsed Gauss-Legendre rule on the reference triangle.

    Returns barycentric coordinates ``lam`` (Q, 3) and weights ``w`` (Q,) with sum(w) = 1/2.
    ``order`` Gauss points per direction -> Q = order**2, exact for total degree <= 2*order - 2.
    """
    xg, wg = np.polynomial.legendre.leggauss(int(order))
    xg, wg = 0.5 * (xg + 1.0), 0.5 * wg
    xi, eta = np.meshgrid(xg, xg, indexing="ij")
    w = (wg[:, None] * wg[None, :]) * (1.0 - xi)
    l2 = xi.ravel()
    l3 = (eta * (1.0 - xi)).ravel()
    lam = np.stack([1.0 - l2 - l3, l2, l3], axis=1)
    return lam, w.ravel()


def p1_geometry(mesh: TriMesh) -> Tuple[np.ndarray, np.ndarray]:
    """Areas (T,) and constant gradients of the three hat functions, shape (T, 3, 2)."""
    P = mesh.nodes[mesh.tris]
    x1, y1 = P[:, 0, 0], P[:, 0, 1]
    x2, y2 = P[:, 1, 0], P[:, 1, 1]
    x3, y3 = P[:, 2, 0], P[:, 2, 1]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if np.any(det <= 0.0):
        raise ValueError("Mesh has degenerate or clockwise triangles.")
    grads = np.stack(
        [
            np.stack([y2 - y3, x3 - x2], axis=1),
            np.stack([y3 - y1, x1 - x3], axis=1),
            np.stack([y1 - y2, x2 - x1], axis=1),
        ],
        axis=1,
    ) / det[:, None, None]
    return 0.5 * det, grads


def _chunks(n: int, size: int):
    for s in range(0, n, size):
        yield s, min(s + size, n)


def cell_average(mesh: TriMesh, fun: Scalar, order: int = 4, chunk: int = 20000) -> np.ndarray:
    """Cell averages (1/|T|) int_T fun dx by Gauss quadrature, shape (T,)."""
    lam, w = triangle_rule(order)
    out = np.empty(mesh.tris.shape[0])
    for s, e in _chunks(out.size, chunk):
        P = mesh.nodes[mesh.tris[s:e]]
        pts = np.einsum("qk,ckd->cqd", lam, P).reshape(-1, 2)
        vals = np.broadcast_to(np.asarray(fun(pts), dtype=np.float64).reshape(-1), (pts.shape[0],))
        out[s:e] = (vals.reshape(e - s, -1) * w[None, :]).sum(axis=1) / w.sum()
    return out


def assemble_stiffness(mesh: TriMesh, a_cell: np.ndarray):
    """Sparse CSR matrix of a_h(phi_i, phi_j)."""
    area, grads = p1_geometry(mesh)
    local = (a_cell * area)[:, None, None] * np.einsum("tid,tjd->tij", grads, grads)
    rows = np.repeat(mesh.tris[:, :, None], 3, axis=2).ravel()
    cols = np.repeat(mesh.tris[:, None, :], 3, axis=1).ravel()
    n = mesh.n_nodes
    return coo_matrix((local.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def residual_vector(
    mesh: TriMesh,
    a_fun: Scalar,
    f_fun: Scalar,
    uhat_eval: UhatEval,
    order: int = 4,
    chunk: int = 20000,
) -> np.ndarray:
    """Weak residual r_i = int f phi_i - int a grad(uhat).grad(phi_i)  (all nodes i).

    Only the FIRST derivative of uhat is needed (no second derivatives, no strong-form residual).
    Entries at boundary nodes are computed but are not used by the dual norm.
    """
    lam, w = triangle_rule(order)
    area, grads = p1_geometry(mesh)
    r = np.zeros(mesh.n_nodes)
    n_q = w.size
    for s, e in _chunks(mesh.tris.shape[0], chunk):
        tri = mesh.tris[s:e]
        c = e - s
        pts = np.einsum("qk,ckd->cqd", lam, mesh.nodes[tri]).reshape(-1, 2)
        W = 2.0 * area[s:e, None] * w[None, :]                       # (c, Q) physical weights
        _, g = uhat_eval(pts)
        f = np.broadcast_to(np.asarray(f_fun(pts), dtype=np.float64).reshape(-1), (pts.shape[0],))
        a = np.broadcast_to(np.asarray(a_fun(pts), dtype=np.float64).reshape(-1), (pts.shape[0],))
        load = np.einsum("cq,qk->ck", W * f.reshape(c, n_q), lam)
        flux = np.einsum("cq,cqd->cd", W * a.reshape(c, n_q), np.asarray(g).reshape(c, n_q, 2))
        stiff = np.einsum("ckd,cd->ck", grads[s:e], flux)
        r += np.bincount(tri.ravel(), weights=(load - stiff).ravel(), minlength=mesh.n_nodes)
    return r

"""a-energy estimator:  ||e||_a^2 = ||R||_{*,a}^2 + ||w_a||_a^2   (Theorem A).

Discrete version on a P1 mesh.  With K the stiffness matrix of a_h, I the interior nodes and
B the boundary nodes:

    dual_sq   = r_I^T K_II^{-1} r_I                     (restriction of the sup to V_h0)
    w         : w_B = d_B,   K_II w_I = -K_IB d_B       (a_h-harmonic lifting, d = g - uhat)
    lift_sq   = w^T K w

If r = K e for some nodal vector e with e_B = d_B, then e^T K e = dual_sq + lift_sq holds
*exactly in exact arithmetic* (it is the discrete Pythagoras theorem); the unit test checks this
to round-off.  For the true continuum quantities, dual_sq is a rigorous LOWER bound of
||R||_{*,a}^2 (up to quadrature error of r), and it increases monotonically under nested
refinement.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from scipy.sparse.linalg import splu

from .fem import UhatEval, assemble_stiffness, cell_average, residual_vector
from .mesh import TriMesh

Scalar = Callable[[np.ndarray], np.ndarray]


@dataclass
class AEnergyResult:
    n_cells: int
    n_nodes: int
    dual_norm: float          # ||R||_{*,a,h}
    lifting_norm: float       # ||w_h||_a
    eta: float                # sqrt(dual^2 + lifting^2)
    residual_vector_norm: float

    def to_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def estimator_from_vectors(K, r: np.ndarray, d_boundary_full: np.ndarray, mesh: TriMesh):
    """Pure linear algebra.  ``d_boundary_full`` is a length-N vector whose boundary entries
    hold g - uhat at the boundary nodes (interior entries are ignored).
    Returns (dual_norm, lifting_norm, w) with w the full nodal lifting vector."""
    I, B = mesh.interior, mesh.boundary
    K_II = K[I][:, I].tocsc()
    K_IB = K[I][:, B]
    lu = splu(K_II)
    r_I = r[I]
    dual_sq = float(r_I @ lu.solve(r_I))
    w = np.zeros(mesh.n_nodes)
    w[B] = d_boundary_full[B]
    w[I] = lu.solve(-(K_IB @ w[B]))
    lift_sq = float(w @ (K @ w))
    return np.sqrt(max(dual_sq, 0.0)), np.sqrt(max(lift_sq, 0.0)), w


def a_energy_estimator(
    mesh: TriMesh,
    a_fun: Scalar,
    f_fun: Scalar,
    g_fun: Scalar,
    uhat_eval: UhatEval,
    quad_order: int = 4,
) -> AEnergyResult:
    a_cell = cell_average(mesh, a_fun, quad_order)
    K = assemble_stiffness(mesh, a_cell)
    r = residual_vector(mesh, a_fun, f_fun, uhat_eval, quad_order)
    xb = mesh.nodes[mesh.boundary]
    d = np.zeros(mesh.n_nodes)
    d[mesh.boundary] = np.asarray(g_fun(xb), dtype=np.float64).reshape(-1) - uhat_eval(xb)[0]
    dual, lift, _ = estimator_from_vectors(K, r, d, mesh)
    return AEnergyResult(
        n_cells=mesh.n_cells,
        n_nodes=mesh.n_nodes,
        dual_norm=float(dual),
        lifting_norm=float(lift),
        eta=float(np.hypot(dual, lift)),
        residual_vector_norm=float(np.linalg.norm(r[mesh.interior])),
    )


def true_error_a_quadrature(
    mesh_ref: TriMesh,
    a_fun: Scalar,
    grad_u_exact: Callable[[np.ndarray], np.ndarray],
    uhat_eval: UhatEval,
    order: int = 5,
    chunk: int = 20000,
) -> float:
    """||u - uhat||_a by Gauss quadrature of a|grad u - grad uhat|^2 (smooth u only)."""
    from .fem import p1_geometry, triangle_rule

    lam, w = triangle_rule(order)
    area, _ = p1_geometry(mesh_ref)
    total = 0.0
    n_q = w.size
    for s in range(0, mesh_ref.tris.shape[0], chunk):
        e = min(s + chunk, mesh_ref.tris.shape[0])
        P = mesh_ref.nodes[mesh_ref.tris[s:e]]
        pts = np.einsum("qk,ckd->cqd", lam, P).reshape(-1, 2)
        W = 2.0 * area[s:e, None] * w[None, :]
        _, g_hat = uhat_eval(pts)
        diff = np.asarray(grad_u_exact(pts)) - np.asarray(g_hat)
        a = np.broadcast_to(np.asarray(a_fun(pts), dtype=np.float64).reshape(-1), (pts.shape[0],))
        total += float(np.sum(W * (a * np.sum(diff**2, axis=1)).reshape(e - s, n_q)))
    return float(np.sqrt(max(total, 0.0)))

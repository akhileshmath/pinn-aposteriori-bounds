from typing import Dict

import numpy as np
import torch
from scipy.fft import dstn, idstn
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .lifting import _build_l_shaped_triangular_mesh, _triangle_shape_gradients, _triangle_stiffness


def compute_dual_norm(
    solver,
    domain: str,
    n_mesh: int,
    stabilization_eps: float,
) -> Dict[str, float]:
    n_mesh = max(int(n_mesh), 32)
    if domain == "unit_square":
        dual = _dual_norm_unit_square(solver, n_mesh, stabilization_eps)
    elif domain == "l_shaped":
        dual = _dual_norm_l_shaped(solver, n_mesh, stabilization_eps)
    else:
        raise ValueError(f"Unsupported domain '{domain}'")

    return {
        "dual_norm": float(dual),
        "mesh_size": n_mesh,
        "stabilization_eps": float(stabilization_eps),
    }


def _evaluate_residual(solver, points: np.ndarray) -> np.ndarray:
    x_torch = torch.tensor(points, dtype=torch.float32, device=solver.device)
    residual = solver.compute_pde_residual(x_torch)
    return residual.detach().cpu().numpy().reshape(-1)


def _dual_norm_unit_square(solver, n_mesh: int, stabilization_eps: float) -> float:
    h = 1.0 / (n_mesh + 1)
    x = np.linspace(h, 1.0 - h, n_mesh)
    y = np.linspace(h, 1.0 - h, n_mesh)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    residual_grid = _evaluate_residual(solver, points).reshape(n_mesh, n_mesh)

    j = np.arange(1, n_mesh + 1)
    k = np.arange(1, n_mesh + 1)
    J, K = np.meshgrid(j, k)
    eigenvalues = (2.0 / h) ** 2 * (
        np.sin(J * np.pi * h / 2.0) ** 2 + np.sin(K * np.pi * h / 2.0) ** 2
    )
    eigenvalues = np.maximum(eigenvalues, stabilization_eps / (h * h))

    residual_hat = dstn(residual_grid, type=1, norm="ortho")
    psi_hat = residual_hat / eigenvalues
    psi_grid = idstn(psi_hat, type=1, norm="ortho")
    energy = h**2 * np.sum(residual_grid * psi_grid)
    return np.sqrt(max(float(energy), 0.0))


def _dual_norm_l_shaped(solver, n_mesh: int, stabilization_eps: float) -> float:
    nodes, triangles, boundary_nodes = _build_l_shaped_triangular_mesh(n_mesh)
    residual_values = _evaluate_residual(solver, nodes)

    n_nodes = nodes.shape[0]
    boundary_mask = np.zeros(n_nodes, dtype=bool)
    boundary_mask[boundary_nodes] = True
    interior_nodes = np.flatnonzero(~boundary_mask)

    rows = []
    cols = []
    vals = []
    load = np.zeros(n_nodes, dtype=np.float64)

    mass_template = np.array(
        [
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )

    for tri in triangles:
        tri_nodes = nodes[tri]
        element_stiffness = _triangle_stiffness(tri_nodes)
        area, _ = _triangle_shape_gradients(tri_nodes)
        element_mass = (area / 12.0) * mass_template
        element_load = element_mass @ residual_values[tri]

        for a in range(3):
            load[tri[a]] += element_load[a]
            for b in range(3):
                rows.append(tri[a])
                cols.append(tri[b])
                vals.append(element_stiffness[a, b])

    system = csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    interior_matrix = system[interior_nodes][:, interior_nodes].tolil()
    interior_matrix.setdiag(interior_matrix.diagonal() + stabilization_eps)
    rhs = load[interior_nodes]

    psi = spsolve(interior_matrix.tocsr(), rhs)
    if not np.all(np.isfinite(psi)):
        raise RuntimeError("L-shaped dual solve produced non-finite values.")

    energy = np.dot(rhs, psi)
    return np.sqrt(max(float(energy), 0.0))

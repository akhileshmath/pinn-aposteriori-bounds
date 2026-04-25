from typing import Dict

import numpy as np
import torch
from scipy.fft import dstn, idstn
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg


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
    h = 2.0 / (n_mesh + 1)
    x = np.linspace(-1.0 + h, 1.0 - h, n_mesh)
    y = np.linspace(-1.0 + h, 1.0 - h, n_mesh)
    X, Y = np.meshgrid(x, y)
    mask = ~((X > 0.0) & (Y < 0.0))
    interior_idx = np.where(mask.ravel())[0]

    mapping = -np.ones(n_mesh * n_mesh, dtype=int)
    mapping[interior_idx] = np.arange(len(interior_idx))

    rows = []
    cols = []
    vals = []
    for local_idx, global_idx in enumerate(interior_idx):
        i = global_idx // n_mesh
        j = global_idx % n_mesh
        rows.append(local_idx)
        cols.append(local_idx)
        vals.append(4.0 / h**2 + stabilization_eps)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni = i + di
            nj = j + dj
            if 0 <= ni < n_mesh and 0 <= nj < n_mesh:
                neighbor_global = ni * n_mesh + nj
                neighbor_local = mapping[neighbor_global]
                if neighbor_local >= 0:
                    rows.append(local_idx)
                    cols.append(neighbor_local)
                    vals.append(-1.0 / h**2)

    system = csr_matrix((vals, (rows, cols)), shape=(len(interior_idx), len(interior_idx)))
    points = np.stack([X.ravel()[interior_idx], Y.ravel()[interior_idx]], axis=1)
    residual_values = _evaluate_residual(solver, points)
    psi, info = cg(system, residual_values, rtol=1e-10, atol=0.0, maxiter=5000)
    if info != 0:
        raise RuntimeError(f"L-shaped dual solve did not converge (info={info})")

    energy = np.dot(residual_values, psi) * h**2
    return np.sqrt(max(float(energy), 0.0))

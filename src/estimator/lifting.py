import numpy as np
import torch
from scipy.fft import dstn, idstn
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def compute_boundary_lifting_norm(
    solver,
    benchmark,
    n_mesh: int,
    stabilization_eps: float,
) -> float:
    if benchmark.domain == "unit_square":
        return _lifting_unit_square(solver, benchmark, n_mesh, stabilization_eps)
    if benchmark.domain == "l_shaped":
        return _lifting_l_shaped(solver, benchmark, n_mesh, stabilization_eps)
    raise ValueError(f"Unsupported domain '{benchmark.domain}'")


def _boundary_mismatch(solver, benchmark, x_value: float, y_value: float) -> float:
    point = torch.tensor([[x_value, y_value]], dtype=torch.float32, device=solver.device)
    exact = benchmark.boundary_condition(point).item()
    pred = solver.predict(point).item()
    return float(exact - pred)


def _lifting_unit_square(solver, benchmark, n_mesh: int, stabilization_eps: float) -> float:
    h = 1.0 / (n_mesh + 1)
    x_full = np.linspace(0.0, 1.0, n_mesh + 2)
    y_full = np.linspace(0.0, 1.0, n_mesh + 2)

    traces = {}
    for name, coords in [
        ("bottom", np.stack([x_full, np.zeros_like(x_full)], axis=1)),
        ("top", np.stack([x_full, np.ones_like(x_full)], axis=1)),
        ("left", np.stack([np.zeros_like(y_full), y_full], axis=1)),
        ("right", np.stack([np.ones_like(y_full), y_full], axis=1)),
    ]:
        x_torch = torch.tensor(coords, dtype=torch.float32, device=solver.device)
        exact = benchmark.boundary_condition(x_torch).detach().cpu().numpy().reshape(-1)
        pred = solver.predict(x_torch).detach().cpu().numpy().reshape(-1)
        traces[name] = exact - pred

    rhs = np.zeros((n_mesh, n_mesh))
    rhs[0, :] += traces["bottom"][1:-1] / h**2
    rhs[-1, :] += traces["top"][1:-1] / h**2
    rhs[:, 0] += traces["left"][1:-1] / h**2
    rhs[:, -1] += traces["right"][1:-1] / h**2

    j = np.arange(1, n_mesh + 1)
    k = np.arange(1, n_mesh + 1)
    J, K = np.meshgrid(j, k)
    eigenvalues = (2.0 / h) ** 2 * (
        np.sin(J * np.pi * h / 2.0) ** 2 + np.sin(K * np.pi * h / 2.0) ** 2
    )
    eigenvalues = np.maximum(eigenvalues, stabilization_eps / (h * h))

    rhs_hat = dstn(rhs, type=1, norm="ortho")
    interior = idstn(rhs_hat / eigenvalues, type=1, norm="ortho")

    full_grid = np.zeros((n_mesh + 2, n_mesh + 2))
    full_grid[1:-1, 1:-1] = interior
    full_grid[0, :] = traces["bottom"]
    full_grid[-1, :] = traces["top"]
    full_grid[:, 0] = traces["left"]
    full_grid[:, -1] = traces["right"]

    energy = _edge_based_dirichlet_energy(full_grid, np.ones((n_mesh + 2, n_mesh + 2), dtype=bool))
    return float(np.sqrt(max(float(energy), 0.0)))


def _lifting_l_shaped(solver, benchmark, n_mesh: int, stabilization_eps: float) -> float:
    h = 2.0 / (n_mesh + 1)
    x = np.linspace(-1.0 + h, 1.0 - h, n_mesh)
    y = np.linspace(-1.0 + h, 1.0 - h, n_mesh)
    X, Y = np.meshgrid(x, y)
    mask = ~((X > 0.0) & (Y < 0.0))
    interior_idx = np.where(mask.ravel())[0]
    mapping = -np.ones(n_mesh * n_mesh, dtype=int)
    mapping[interior_idx] = np.arange(len(interior_idx))

    system = lil_matrix((len(interior_idx), len(interior_idx)))
    rhs = np.zeros(len(interior_idx))
    x_full = np.linspace(-1.0, 1.0, n_mesh + 2)
    y_full = np.linspace(-1.0, 1.0, n_mesh + 2)

    for local_idx, global_idx in enumerate(interior_idx):
        i = global_idx // n_mesh
        j = global_idx % n_mesh
        system[local_idx, local_idx] = 4.0 / h**2 + stabilization_eps
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni = i + di
            nj = j + dj
            if 0 <= ni < n_mesh and 0 <= nj < n_mesh:
                neighbor_global = ni * n_mesh + nj
                neighbor_local = mapping[neighbor_global]
                if neighbor_local >= 0:
                    system[local_idx, neighbor_local] = -1.0 / h**2
                else:
                    rhs[local_idx] += _boundary_mismatch(
                        solver, benchmark, x_full[nj + 1], y_full[ni + 1]
                    ) / h**2
            else:
                rhs[local_idx] += _boundary_mismatch(
                    solver, benchmark, x_full[nj + 1], y_full[ni + 1]
                ) / h**2

    interior = spsolve(system.tocsr(), rhs)
    full_grid = np.zeros((n_mesh + 2, n_mesh + 2))
    for local_idx, global_idx in enumerate(interior_idx):
        i = global_idx // n_mesh
        j = global_idx % n_mesh
        full_grid[i + 1, j + 1] = interior[local_idx]

    for i in range(n_mesh + 2):
        for j in range(n_mesh + 2):
            x_val = x_full[j]
            y_val = y_full[i]
            on_box_boundary = i in (0, n_mesh + 1) or j in (0, n_mesh + 1)
            on_reentrant_edge = (
                (abs(x_val) < 1e-14 and -1.0 <= y_val <= 0.0)
                or (abs(y_val) < 1e-14 and 0.0 <= x_val <= 1.0)
            )
            if (on_box_boundary or on_reentrant_edge) and not (x_val > 0.0 and y_val < 0.0):
                full_grid[i, j] = _boundary_mismatch(solver, benchmark, x_val, y_val)

    full_mask = np.ones((n_mesh + 2, n_mesh + 2), dtype=bool)
    for i in range(n_mesh + 2):
        for j in range(n_mesh + 2):
            x_val = x_full[j]
            y_val = y_full[i]
            if x_val > 0.0 and y_val < 0.0:
                full_mask[i, j] = False

    energy = _edge_based_dirichlet_energy(full_grid, full_mask)
    return float(np.sqrt(max(float(energy), 0.0)))


def _edge_based_dirichlet_energy(full_grid: np.ndarray, valid_mask: np.ndarray) -> float:
    energy = 0.0

    # Horizontal edges.
    for i in range(full_grid.shape[0]):
        for j in range(full_grid.shape[1] - 1):
            if valid_mask[i, j] and valid_mask[i, j + 1]:
                diff = full_grid[i, j + 1] - full_grid[i, j]
                energy += diff**2

    # Vertical edges.
    for i in range(full_grid.shape[0] - 1):
        for j in range(full_grid.shape[1]):
            if valid_mask[i, j] and valid_mask[i + 1, j]:
                diff = full_grid[i + 1, j] - full_grid[i, j]
                energy += diff**2

    return energy

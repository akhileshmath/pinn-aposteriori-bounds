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
    nodes, triangles, boundary_nodes = _build_l_shaped_triangular_mesh(n_mesh)
    boundary_values = np.array(
        [_boundary_mismatch(solver, benchmark, x_val, y_val) for x_val, y_val in nodes],
        dtype=np.float64,
    )

    stiffness = lil_matrix((nodes.shape[0], nodes.shape[0]), dtype=np.float64)
    for tri in triangles:
        tri_nodes = nodes[tri]
        element = _triangle_stiffness(tri_nodes)
        for a in range(3):
            for b in range(3):
                stiffness[tri[a], tri[b]] += element[a, b]

    stiffness = stiffness.tocsr()
    interior_nodes = np.setdiff1d(np.arange(nodes.shape[0]), boundary_nodes, assume_unique=True)

    if interior_nodes.size == 0:
        solution = boundary_values
    else:
        solution = boundary_values.copy()
        rhs = -stiffness[interior_nodes][:, boundary_nodes] @ boundary_values[boundary_nodes]
        interior_matrix = stiffness[interior_nodes][:, interior_nodes].tolil()
        interior_matrix.setdiag(interior_matrix.diagonal() + stabilization_eps)
        solution[interior_nodes] = spsolve(interior_matrix.tocsr(), rhs)

    energy = 0.0
    for tri in triangles:
        tri_nodes = nodes[tri]
        tri_values = solution[tri]
        area, gradients = _triangle_shape_gradients(tri_nodes)
        gradient = tri_values @ gradients
        energy += area * float(np.dot(gradient, gradient))

    return float(np.sqrt(max(energy, 0.0)))


def _build_l_shaped_triangular_mesh(n_mesh: int):
    n_div = max(int(n_mesh), 16)
    if n_div % 2 != 0:
        n_div += 1

    x = np.linspace(-1.0, 1.0, n_div + 1)
    y = np.linspace(-1.0, 1.0, n_div + 1)
    index_map = -np.ones((n_div + 1, n_div + 1), dtype=int)
    nodes = []

    for i, y_val in enumerate(y):
        for j, x_val in enumerate(x):
            if x_val > 0.0 and y_val < 0.0:
                continue
            index_map[i, j] = len(nodes)
            nodes.append((x_val, y_val))

    nodes = np.asarray(nodes, dtype=np.float64)
    triangles = []

    for i in range(n_div):
        for j in range(n_div):
            x_center = 0.5 * (x[j] + x[j + 1])
            y_center = 0.5 * (y[i] + y[i + 1])
            if x_center > 0.0 and y_center < 0.0:
                continue

            bl = index_map[i, j]
            br = index_map[i, j + 1]
            tl = index_map[i + 1, j]
            tr = index_map[i + 1, j + 1]

            if min(bl, br, tl, tr) < 0:
                continue

            triangles.append((bl, br, tr))
            triangles.append((bl, tr, tl))

    boundary_nodes = []
    tol = 1e-12
    for idx, (x_val, y_val) in enumerate(nodes):
        on_outer_boundary = (
            abs(x_val + 1.0) < tol
            or abs(x_val - 1.0) < tol
            or abs(y_val + 1.0) < tol
            or abs(y_val - 1.0) < tol
        )
        on_reentrant_boundary = (
            (abs(x_val) < tol and y_val <= tol)
            or (abs(y_val) < tol and x_val >= -tol)
        )
        if on_outer_boundary or on_reentrant_boundary:
            boundary_nodes.append(idx)

    return nodes, np.asarray(triangles, dtype=np.int64), np.asarray(boundary_nodes, dtype=np.int64)


def _triangle_shape_gradients(triangle_nodes: np.ndarray):
    x1, y1 = triangle_nodes[0]
    x2, y2 = triangle_nodes[1]
    x3, y3 = triangle_nodes[2]
    det_j = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * abs(det_j)
    if area <= 0.0:
        raise ValueError("Degenerate triangle encountered in L-shaped lifting mesh.")

    gradients = np.array(
        [
            [y2 - y3, x3 - x2],
            [y3 - y1, x1 - x3],
            [y1 - y2, x2 - x1],
        ],
        dtype=np.float64,
    ) / det_j
    return area, gradients


def _triangle_stiffness(triangle_nodes: np.ndarray) -> np.ndarray:
    area, gradients = _triangle_shape_gradients(triangle_nodes)
    return area * (gradients @ gradients.T)


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

"""Structured P1 triangulations (NumPy only, fully vectorised).

Triangle convention (same as ``src/estimator/p1_energy.py``): every grid cell is split into
(bl, br, tr) and (bl, tr, tl), both counter-clockwise.  For even ``n`` the meshes with ``n`` and
``2n`` cells per side are nested, so the P1 spaces are nested: V_n is a subspace of V_{2n}.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TriMesh:
    nodes: np.ndarray        # (N, 2) float64
    tris: np.ndarray         # (T, 3) int64, counter-clockwise
    is_boundary: np.ndarray  # (N,) bool
    n_cells: int             # cells per side of the bounding square
    domain: str

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def interior(self) -> np.ndarray:
        return np.flatnonzero(~self.is_boundary)

    @property
    def boundary(self) -> np.ndarray:
        return np.flatnonzero(self.is_boundary)


def _boundary_flags(n_nodes: int, tris: np.ndarray) -> np.ndarray:
    """A node is on the boundary iff it lies on an edge that belongs to exactly one triangle."""
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    flags = np.zeros(n_nodes, dtype=bool)
    flags[uniq[counts == 1].ravel()] = True
    return flags


def _cells_to_tris(idx: np.ndarray, keep: np.ndarray) -> np.ndarray:
    n = idx.shape[0] - 1
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ii, jj = ii[keep], jj[keep]
    bl, br = idx[ii, jj], idx[ii, jj + 1]
    tl, tr = idx[ii + 1, jj], idx[ii + 1, jj + 1]
    lower = np.stack([bl, br, tr], axis=1)
    upper = np.stack([bl, tr, tl], axis=1)
    return np.concatenate([lower, upper], axis=0).astype(np.int64)


def unit_square_mesh(n_cells: int) -> TriMesh:
    n = int(n_cells)
    if n < 2:
        raise ValueError("n_cells must be >= 2")
    coords = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(coords, coords)  # row index = y, column index = x
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    tris = _cells_to_tris(idx, np.ones((n, n), dtype=bool))
    return TriMesh(nodes, tris, _boundary_flags(nodes.shape[0], tris), n, "unit_square")


def l_shaped_mesh(n_cells: int) -> TriMesh:
    """L-shaped domain (-1,1)^2 minus the closed quadrant {x>0, y<0}; ``n_cells`` must be even."""
    n = int(n_cells)
    if n < 2 or n % 2:
        raise ValueError("n_cells must be even and >= 2 so that the re-entrant corner is a node")
    coords = np.linspace(-1.0, 1.0, n + 1)
    coords[n // 2] = 0.0  # exact zero for the corner
    cc = 0.5 * (coords[:-1] + coords[1:])
    CX, CY = np.meshgrid(cc, cc)  # CX[i, j] = x-centre of cell (i, j)
    keep = ~((CX > 0.0) & (CY < 0.0))
    full_idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    tris_full = _cells_to_tris(full_idx, keep)
    used = np.unique(tris_full)
    renumber = -np.ones((n + 1) ** 2, dtype=np.int64)
    renumber[used] = np.arange(used.size)
    X, Y = np.meshgrid(coords, coords)
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)[used]
    tris = renumber[tris_full]
    return TriMesh(nodes, tris, _boundary_flags(nodes.shape[0], tris), n, "l_shaped")


def build_mesh(domain: str, n_cells: int) -> TriMesh:
    if domain == "unit_square":
        return unit_square_mesh(n_cells)
    if domain == "l_shaped":
        return l_shaped_mesh(n_cells)
    raise ValueError(f"Unsupported domain '{domain}'")

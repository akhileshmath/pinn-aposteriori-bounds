"""Exact P1 Dirichlet energy on the uniform right-triangle mesh of the unit square.

NumPy only (no torch), so it can be unit-tested without the rest of the pipeline.

Mesh convention (matches the five-point finite-difference Laplacian used by the DST solver):
every grid cell is split into the triangles (bl, br, tr) and (bl, tr, tl).  On such a mesh the
element stiffness matrix gives

    int_T |grad v|^2 dx = 1/2 * [ (v_br - v_bl)^2 + (v_tr - v_br)^2 ]   for (bl, br, tr),
    int_T |grad v|^2 dx = 1/2 * [ (v_tl - v_bl)^2 + (v_tr - v_tl)^2 ]   for (bl, tr, tl),

independently of h (in two dimensions).  The diagonal edge carries no stiffness.  Every grid edge
shared by two triangles therefore contributes (dv)^2 and every edge on the boundary of the
square, which belongs to one triangle only, contributes (dv)^2 / 2.
"""
from typing import Callable, Optional

import numpy as np


def p1_energy_square(
    full_grid: np.ndarray,
    a_fun: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> float:
    """Return int_Omega a |grad v_h|^2 dx for the P1 interpolant of ``full_grid``.

    ``full_grid[i, j]`` is the nodal value at (x_j, y_i) with x_j = j h, y_i = i h and
    h = 1 / (full_grid.shape[0] - 1).  If ``a_fun`` is None the weight is 1.  Otherwise
    ``a_fun(x, y)`` is evaluated at the centroid of each triangle (vectorised).
    """
    grid = np.asarray(full_grid, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1] or grid.shape[0] < 2:
        raise ValueError("full_grid must be a square array with at least 2 nodes per side.")

    bl, br = grid[:-1, :-1], grid[:-1, 1:]
    tl, tr = grid[1:, :-1], grid[1:, 1:]
    lower = 0.5 * ((br - bl) ** 2 + (tr - br) ** 2)
    upper = 0.5 * ((tl - bl) ** 2 + (tr - tl) ** 2)

    if a_fun is None:
        return float(np.sum(lower + upper))

    h = 1.0 / (grid.shape[0] - 1)
    coords = np.arange(grid.shape[1] - 1) * h
    x, y = np.meshgrid(coords, coords)
    a_lower = np.asarray(a_fun(x + 2.0 * h / 3.0, y + h / 3.0), dtype=np.float64)
    a_upper = np.asarray(a_fun(x + h / 3.0, y + 2.0 * h / 3.0), dtype=np.float64)
    return float(np.sum(a_lower * lower + a_upper * upper))
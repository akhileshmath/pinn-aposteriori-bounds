from typing import Optional, Tuple

import numpy as np


def get_quadrature_points(
    domain: str,
    n_points: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if rng is None:
        rng = np.random.default_rng()

    if domain == "unit_square":
        return _structured_unit_square(max(int(n_points), 16))

    if domain == "l_shaped":
        return _structured_l_shaped(max(int(n_points), 16))

    raise ValueError(f"Unsupported domain '{domain}'")


def _structured_unit_square(n_points: int) -> Tuple[np.ndarray, np.ndarray, float]:
    n_side = max(int(np.ceil(np.sqrt(n_points))), 4)
    h = 1.0 / n_side
    coords = (np.arange(n_side) + 0.5) * h
    X, Y = np.meshgrid(coords, coords)
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    weights = np.ones(points.shape[0]) / points.shape[0]
    return points, weights, 1.0


def _structured_l_shaped(n_points: int) -> Tuple[np.ndarray, np.ndarray, float]:
    # 3/4 of the bounding box is active, so scale the side count accordingly.
    n_side = max(int(np.ceil(np.sqrt((4.0 / 3.0) * n_points))), 6)
    h = 2.0 / n_side
    coords = -1.0 + (np.arange(n_side) + 0.5) * h
    X, Y = np.meshgrid(coords, coords)
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    mask = ~((points[:, 0] > 0.0) & (points[:, 1] < 0.0))
    valid_points = points[mask]
    weights = np.ones(valid_points.shape[0]) / valid_points.shape[0]
    return valid_points, weights, 3.0

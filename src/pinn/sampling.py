import math
from typing import Callable, Dict, Optional, Tuple

import torch


def unit_square_interior_sampler(n: int) -> torch.Tensor:
    return torch.rand(n, 2)


def unit_square_boundary_sampler(n: int) -> torch.Tensor:
    n_per_side = max(n // 4, 1)
    points = []

    t = torch.rand(n_per_side, 1)
    points.append(torch.cat([t, torch.zeros_like(t)], dim=1))

    t = torch.rand(n_per_side, 1)
    points.append(torch.cat([t, torch.ones_like(t)], dim=1))

    t = torch.rand(n_per_side, 1)
    points.append(torch.cat([torch.zeros_like(t), t], dim=1))

    t = torch.rand(n_per_side, 1)
    points.append(torch.cat([torch.ones_like(t), t], dim=1))

    return torch.cat(points, dim=0)[:n]


def _uniform_l_shaped_points(n: int) -> torch.Tensor:
    chunks = []
    total = 0
    while total < n:
        proposal = torch.rand(2 * max(n, 1), 2) * 2.0 - 1.0
        mask = ~((proposal[:, 0] > 0.0) & (proposal[:, 1] < 0.0))
        valid = proposal[mask]
        chunks.append(valid)
        total += valid.shape[0]
    return torch.cat(chunks, dim=0)[:n]


def l_shaped_interior_sampler(
    n: int,
    corner_fraction: float = 0.0,
    corner_radius: float = 0.35,
    corner_power: float = 2.0,
) -> torch.Tensor:
    n_corner = min(max(int(round(n * corner_fraction)), 0), n)
    n_uniform = n - n_corner

    points = []
    if n_uniform > 0:
        points.append(_uniform_l_shaped_points(n_uniform))

    if n_corner > 0:
        theta = torch.rand(n_corner, 1) * (1.5 * math.pi)
        radius = corner_radius * torch.rand(n_corner, 1).pow(corner_power)
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        points.append(torch.cat([x, y], dim=1))

    if not points:
        return torch.empty(0, 2)
    return torch.cat(points, dim=0)[:n]


def l_shaped_boundary_sampler(
    n: int,
    corner_fraction: float = 0.0,
    corner_power: float = 2.0,
) -> torch.Tensor:
    segments = [
        ((-1.0, -1.0), (0.0, -1.0), 1.0),
        ((0.0, -1.0), (0.0, 0.0), 1.0),
        ((0.0, 0.0), (1.0, 0.0), 1.0),
        ((1.0, 0.0), (1.0, 1.0), 1.0),
        ((1.0, 1.0), (-1.0, 1.0), 2.0),
        ((-1.0, 1.0), (-1.0, -1.0), 2.0),
    ]
    total_length = sum(length for _, _, length in segments)
    points = []
    n_corner = min(max(int(round(n * corner_fraction)), 0), n)
    n_base = max(n - n_corner, 0)

    for index, (start, end, length) in enumerate(segments):
        n_segment = max(int(n_base * length / total_length), 1)
        t = torch.rand(n_segment, 1)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        points.append(torch.cat([x, y], dim=1))

    if n_corner > 0:
        n_vertical = n_corner // 2
        n_horizontal = n_corner - n_vertical

        if n_vertical > 0:
            t = 1.0 - torch.rand(n_vertical, 1).pow(corner_power)
            x = torch.zeros_like(t)
            y = -1.0 + t
            points.append(torch.cat([x, y], dim=1))

        if n_horizontal > 0:
            t = torch.rand(n_horizontal, 1).pow(corner_power)
            x = t
            y = torch.zeros_like(t)
            points.append(torch.cat([x, y], dim=1))

    return torch.cat(points, dim=0)[:n]


def get_domain_samplers(
    domain: str,
    sampler_config: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[Callable, Callable]:
    sampler_config = sampler_config or {}
    interior_kwargs = sampler_config.get("interior", {})
    boundary_kwargs = sampler_config.get("boundary", {})

    if domain == "unit_square":
        return unit_square_interior_sampler, unit_square_boundary_sampler
    if domain == "l_shaped":
        return (
            lambda n: l_shaped_interior_sampler(n, **interior_kwargs),
            lambda n: l_shaped_boundary_sampler(n, **boundary_kwargs),
        )
    raise ValueError(f"Unknown domain '{domain}'")

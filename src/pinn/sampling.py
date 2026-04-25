from typing import Callable, Tuple

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


def l_shaped_interior_sampler(n: int) -> torch.Tensor:
    chunks = []
    total = 0
    while total < n:
        proposal = torch.rand(2 * n, 2) * 2.0 - 1.0
        mask = ~((proposal[:, 0] > 0.0) & (proposal[:, 1] < 0.0))
        valid = proposal[mask]
        chunks.append(valid)
        total += valid.shape[0]
    return torch.cat(chunks, dim=0)[:n]


def l_shaped_boundary_sampler(n: int) -> torch.Tensor:
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
    for start, end, length in segments:
        n_segment = max(int(n * length / total_length), 1)
        t = torch.rand(n_segment, 1)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        points.append(torch.cat([x, y], dim=1))
    return torch.cat(points, dim=0)[:n]


def get_domain_samplers(domain: str) -> Tuple[Callable, Callable]:
    if domain == "unit_square":
        return unit_square_interior_sampler, unit_square_boundary_sampler
    if domain == "l_shaped":
        return l_shaped_interior_sampler, l_shaped_boundary_sampler
    raise ValueError(f"Unknown domain '{domain}'")

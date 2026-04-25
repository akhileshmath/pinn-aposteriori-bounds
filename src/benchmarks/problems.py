from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch


@dataclass
class BenchmarkProblem:
    name: str
    key: str
    domain: str
    diffusion_coeff: Callable
    source_term: Callable
    boundary_condition: Callable
    exact_solution: Optional[Callable]
    exact_gradient: Optional[Callable]
    description: str
    coercivity_constant: float


def _poisson_problem() -> BenchmarkProblem:
    def a(x):
        return torch.ones(x.shape[0], 1, device=x.device)

    def f(x):
        return 2 * np.pi**2 * torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

    def g(x):
        return torch.zeros(x.shape[0], 1, device=x.device)

    def u(x):
        return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

    def grad_u(x):
        dudx = np.pi * torch.cos(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])
        dudy = np.pi * torch.sin(np.pi * x[:, 0:1]) * torch.cos(np.pi * x[:, 1:2])
        return torch.cat([dudx, dudy], dim=1)

    return BenchmarkProblem(
        name="Poisson (smooth)",
        key="poisson",
        domain="unit_square",
        diffusion_coeff=a,
        source_term=f,
        boundary_condition=g,
        exact_solution=u,
        exact_gradient=grad_u,
        description="Smooth Dirichlet Poisson benchmark on the unit square.",
        coercivity_constant=1.0,
    )


def _variable_coefficient_problem() -> BenchmarkProblem:
    def a(x):
        return 1.0 + 0.5 * torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

    def u(x):
        return (
            torch.sin(np.pi * x[:, 0:1])
            * torch.sin(np.pi * x[:, 1:2])
            * (1.0 + x[:, 0:1] ** 2 * x[:, 1:2])
        )

    def grad_u(x):
        sx = torch.sin(np.pi * x[:, 0:1])
        sy = torch.sin(np.pi * x[:, 1:2])
        cx = torch.cos(np.pi * x[:, 0:1])
        cy = torch.cos(np.pi * x[:, 1:2])
        xx = x[:, 0:1]
        yy = x[:, 1:2]
        dudx = np.pi * cx * sy * (1 + xx**2 * yy) + sx * sy * (2 * xx * yy)
        dudy = np.pi * sx * cy * (1 + xx**2 * yy) + sx * sy * xx**2
        return torch.cat([dudx, dudy], dim=1)

    def f(x):
        x_ad = x.clone().requires_grad_(True)
        u_val = u(x_ad)
        grad = torch.autograd.grad(u_val.sum(), x_ad, create_graph=True)[0]
        a_val = a(x_ad)
        divergence = torch.zeros(x.shape[0], 1, device=x.device)
        for i in range(2):
            flux_i = a_val * grad[:, i : i + 1]
            d_flux = torch.autograd.grad(flux_i.sum(), x_ad, create_graph=True)[0][:, i : i + 1]
            divergence = divergence + d_flux
        return (-divergence).detach()

    def g(x):
        return u(x)

    return BenchmarkProblem(
        name="Variable coefficient diffusion",
        key="variable_coefficient",
        domain="unit_square",
        diffusion_coeff=a,
        source_term=f,
        boundary_condition=g,
        exact_solution=u,
        exact_gradient=grad_u,
        description="Coercive diffusion with spatially varying coefficient a(x).",
        coercivity_constant=0.5,
    )


def _l_shaped_problem() -> BenchmarkProblem:
    def a(x):
        return torch.ones(x.shape[0], 1, device=x.device)

    def polar(x):
        r = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + 1e-12)
        theta = torch.atan2(x[:, 1:2], x[:, 0:1])
        theta = torch.where(theta < 0, theta + 2 * np.pi, theta)
        return r, theta

    def u(x):
        r, theta = polar(x)
        return r ** (2.0 / 3.0) * torch.sin(2.0 * theta / 3.0)

    def grad_u(x):
        r, theta = polar(x)
        dudr = (2.0 / 3.0) * r ** (-1.0 / 3.0) * torch.sin(2.0 * theta / 3.0)
        dudtheta = (2.0 / 3.0) * r ** (2.0 / 3.0) * torch.cos(2.0 * theta / 3.0)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        dudx = cos_t * dudr - (sin_t / r) * dudtheta
        dudy = sin_t * dudr + (cos_t / r) * dudtheta
        return torch.cat([dudx, dudy], dim=1)

    def f(x):
        return torch.zeros(x.shape[0], 1, device=x.device)

    def g(x):
        return u(x)

    return BenchmarkProblem(
        name="L-shaped domain singularity",
        key="l_shaped",
        domain="l_shaped",
        diffusion_coeff=a,
        source_term=f,
        boundary_condition=g,
        exact_solution=u,
        exact_gradient=grad_u,
        description="Laplace benchmark with the classical re-entrant corner singularity.",
        coercivity_constant=1.0,
    )


def get_supported_benchmarks() -> Dict[str, BenchmarkProblem]:
    return {
        "poisson": _poisson_problem(),
        "variable_coefficient": _variable_coefficient_problem(),
        "l_shaped": _l_shaped_problem(),
    }

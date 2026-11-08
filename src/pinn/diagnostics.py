"""Training-independent loss diagnostics.

``fixed_sample_losses`` evaluates the three loss components on a FIXED, UNIFORM sample that does
not depend on how the network was trained (in particular not on corner-refined collocation).  This
is what makes the 'loss is blind to the corner' comparison fair: two networks trained with
different collocation measures are scored by the same, independent measure.
"""
from typing import Dict

import torch

from .sampling import get_domain_samplers


def fixed_sample_losses(solver, benchmark, n_interior: int = 20000, n_boundary: int = 4000,
                        seed: int = 0, dtype=torch.float64) -> Dict[str, float]:
    state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        interior, boundary = get_domain_samplers(benchmark.domain)  # default samplers = uniform
        x_r = interior(n_interior).to(dtype)
        x_b = boundary(n_boundary).to(dtype)
    finally:
        torch.set_rng_state(state)
    net = solver.net
    old_dtype = next(net.parameters()).dtype
    net.to(dtype)
    try:
        res = solver.compute_pde_residual(x_r).detach()
        bnd = (net(x_b) - benchmark.boundary_condition(x_b)).detach()
        tan = solver.compute_tangential_loss(x_b).detach()
        return {
            "residual_loss": float(torch.mean(res**2)),
            "boundary_loss": float(torch.mean(bnd**2)),
            "tangential_loss": float(tan),
            "residual_l2": float(torch.sqrt(torch.mean(res**2))),
        }
    finally:
        net.to(old_dtype)

"""Build and train a PINN for a benchmark, optionally in float64 and/or with a hard-BC ansatz.

``train_pinn`` is the single entry point used by the a-energy experiments.  With the default
arguments (float32, no hard BC, w_tangent = 0) it performs exactly the same operations, in the same
order, as ``experiments/run.py::run_single_experiment`` for the same seed.
"""
import contextlib
from typing import Any, Dict, Optional

import numpy as np
import torch

from .model import PINNNetwork
from .sampling import get_domain_samplers
from .solver import PINNSolver
from .training import Trainer, TrainingConfig


@contextlib.contextmanager
def default_dtype(dtype):
    """Temporarily set torch's default dtype (restored on exit, also on error)."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _check_hard_bc(benchmark, network_cfg: Dict[str, Any]) -> None:
    if network_cfg.get("hard_bc") != "unit_square_zero":
        return
    if benchmark.domain != "unit_square":
        raise ValueError("hard_bc='unit_square_zero' requires the unit-square domain.")
    _, boundary_sampler = get_domain_samplers(benchmark.domain)
    x = boundary_sampler(400).to(torch.float64)
    if float(benchmark.boundary_condition(x).abs().max()) > 1e-10:
        raise ValueError("hard_bc='unit_square_zero' requires homogeneous Dirichlet data g = 0.")


def train_pinn(
    benchmark,
    network_cfg: Dict[str, Any],
    training_cfg: TrainingConfig,
    sampler_cfg: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    device: str = "cpu",
    dtype: str = "float32",
):
    """Returns (solver, history).  ``dtype`` in {'float32', 'float64'}."""
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    _check_hard_bc(benchmark, network_cfg)
    with default_dtype(getattr(torch, dtype)):
        set_reproducibility(seed)
        interior_sampler, boundary_sampler = get_domain_samplers(benchmark.domain, sampler_cfg)
        network = PINNNetwork(**network_cfg)
        solver = PINNSolver(
            network=network,
            diffusion_coeff=benchmark.diffusion_coeff,
            source_term=benchmark.source_term,
            boundary_condition=benchmark.boundary_condition,
            domain_sampler=interior_sampler,
            boundary_sampler=boundary_sampler,
            device=device,
            domain=benchmark.domain,
        )
        history = Trainer(solver, training_cfg).train()
    return solver, history

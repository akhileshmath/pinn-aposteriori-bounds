"""Bridge from a trained PINN / benchmark (torch) to the torch-free a-energy core.

The network is deep-copied to CPU float64 so that every quantity entering the estimator is
evaluated in double precision, independently of the precision used for training.  Only FIRST
derivatives of uhat are needed (weak-form residual), so the second-derivative autograd of the old
strong-form pipeline is not used here.
"""
import copy
from typing import Callable, Dict, Sequence

import numpy as np

from .evaluate import evaluate_from_callables
from .references import lshape_grad_u


def _batched_scalar(fn: Callable, chunk: int = 200_000) -> Callable[[np.ndarray], np.ndarray]:
    import torch

    def call(pts: np.ndarray) -> np.ndarray:
        out = []
        for s in range(0, len(pts), chunk):
            x = torch.tensor(pts[s:s + chunk], dtype=torch.float64)
            out.append(fn(x).detach().cpu().numpy().astype(np.float64).reshape(-1))
        return np.concatenate(out)

    return call


def _batched_vector(fn: Callable, chunk: int = 200_000) -> Callable[[np.ndarray], np.ndarray]:
    import torch

    def call(pts: np.ndarray) -> np.ndarray:
        out = []
        for s in range(0, len(pts), chunk):
            x = torch.tensor(pts[s:s + chunk], dtype=torch.float64)
            out.append(fn(x).detach().cpu().numpy().astype(np.float64).reshape(-1, 2))
        return np.concatenate(out, axis=0)

    return call


def make_uhat_eval(net, chunk: int = 100_000):
    """(M,2) float64 points -> (uhat (M,), grad uhat (M,2)) from a float64 copy of ``net``."""
    import torch

    net64 = copy.deepcopy(net).to("cpu").double().eval()
    for p in net64.parameters():
        p.requires_grad_(False)

    def uhat_eval(pts: np.ndarray):
        us, gs = [], []
        for s in range(0, len(pts), chunk):
            x = torch.tensor(pts[s:s + chunk], dtype=torch.float64, requires_grad=True)
            u = net64(x)
            (g,) = torch.autograd.grad(u.sum(), x)
            us.append(u.detach().numpy().reshape(-1))
            gs.append(g.detach().numpy())
        return np.concatenate(us), np.concatenate(gs, axis=0)

    return uhat_eval


def make_callables(solver_or_net, benchmark) -> Dict[str, Callable]:
    net = getattr(solver_or_net, "net", solver_or_net)
    is_l = benchmark.domain == "l_shaped"
    return {
        "a_fun": _batched_scalar(benchmark.diffusion_coeff),
        "f_fun": _batched_scalar(benchmark.source_term),
        "g_fun": _batched_scalar(benchmark.boundary_condition),
        # g = u|_Gamma for all benchmarks, so grad g . tau = grad u . tau on the boundary
        "grad_g_fun": lshape_grad_u if is_l else _batched_vector(benchmark.exact_gradient),
        "grad_u_exact": None if is_l else _batched_vector(benchmark.exact_gradient),
        "uhat_eval": make_uhat_eval(net),
    }


def evaluate_for_solver(
    solver_or_net,
    benchmark,
    ladder: Sequence[int] = (32, 64, 128),
    quad_order: int = 4,
    ref_cells: int = 128,
) -> Dict:
    c = make_callables(solver_or_net, benchmark)
    return evaluate_from_callables(
        benchmark.domain, c["a_fun"], c["f_fun"], c["g_fun"], c["grad_g_fun"], c["grad_u_exact"],
        c["uhat_eval"],
        alpha=float(benchmark.coercivity_constant),
        beta=float(getattr(benchmark, "coefficient_upper_bound", benchmark.coercivity_constant)),
        ladder=ladder, quad_order=quad_order, ref_cells=ref_cells,
    )

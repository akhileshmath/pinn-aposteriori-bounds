"""Tests that need torch: bridge between PINN/benchmarks and the a-energy core.
Run:  python -m pytest tests/test_a_energy_bridge.py -v"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.a_energy import lshape_grad_u  # noqa: E402
from src.a_energy.torch_bridge import evaluate_for_solver, make_uhat_eval  # noqa: E402
from src.benchmarks import get_supported_benchmarks  # noqa: E402
from src.pinn import PINNNetwork  # noqa: E402


class _ExactPoisson(torch.nn.Module):
    def forward(self, x):
        return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])


def test_exact_poisson_network_has_tiny_estimator_and_error():
    res = evaluate_for_solver(_ExactPoisson(), get_supported_benchmarks()["poisson"], ladder=(16, 32), ref_cells=64)
    assert res["true_error_a"] < 1e-9
    assert res["eta_a_finest"] < 1e-3          # weak residual / lifting: quadrature + P1 level only
    assert res["boundary"]["l2"] < 1e-12


def test_numpy_L_shape_gradient_matches_torch_benchmark():
    bm = get_supported_benchmarks()["l_shaped"]
    rng = np.random.default_rng(0)
    p = rng.uniform(-1, 1, (500, 2))
    p = p[~((p[:, 0] > 0) & (p[:, 1] < 0))]
    p = p[np.hypot(p[:, 0], p[:, 1]) > 0.05]
    ref = bm.exact_gradient(torch.tensor(p, dtype=torch.float64)).numpy()
    assert np.max(np.abs(ref - lshape_grad_u(p))) < 1e-8


def test_float64_copy_reproduces_float32_network():
    torch.manual_seed(0)
    net = PINNNetwork(hidden_dims=[16, 16])
    ev = make_uhat_eval(net)
    pts = np.random.default_rng(0).uniform(0, 1, (50, 2))
    u64, _ = ev(pts)
    u32 = net(torch.tensor(pts, dtype=torch.float32)).detach().numpy().reshape(-1)
    assert np.max(np.abs(u64 - u32)) < 1e-5


@pytest.mark.parametrize("key", ["poisson", "variable_coefficient", "l_shaped"])
def test_untrained_network_runs_and_theorem_A_ratio_is_near_one(key):
    torch.manual_seed(1)
    net = PINNNetwork(hidden_dims=[16, 16])
    res = evaluate_for_solver(net, get_supported_benchmarks()[key], ladder=(16, 32), ref_cells=64)
    r = res["levels"][-1]["eta_over_true"]
    assert np.isfinite(r) and 0.7 < r < 1.3, r
    assert res["ref_self_convergence_rel"] < 1e-3

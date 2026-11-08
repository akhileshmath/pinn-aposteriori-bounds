"""Step-3 tests: hard-BC ansatz, tangential penalty, float64 training, corner-refined sampling,
fixed-sample diagnostics.  Run:  python -m pytest tests/test_pinn_step3.py -v"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks import get_supported_benchmarks  # noqa: E402
from src.pinn import (  # noqa: E402
    PINNNetwork, PINNSolver, TrainingConfig, fixed_sample_losses, get_domain_samplers, train_pinn,
)
from src.pinn.sampling import boundary_tangent  # noqa: E402

BM = get_supported_benchmarks()
TINY = {"hidden_dims": [16, 16]}


def _cfg(**kw):
    base = dict(adam_epochs=30, lbfgs_epochs=0, n_collocation=200, n_boundary=80,
                resample_every=0, print_every=10_000)
    base.update(kw)
    return TrainingConfig(**base)


class _Analytic(torch.nn.Module):
    """u(x) = sin(x1) x2^2 + 0.3 x1 (parameter-free, for exact-value checks)."""

    def forward(self, x):
        return torch.sin(x[:, 0:1]) * x[:, 1:2] ** 2 + 0.3 * x[:, 0:1]


def test_defaults_are_off():
    c = TrainingConfig()
    assert c.w_tangent == 0.0
    assert PINNNetwork().hard_bc is None


def test_hard_bc_vanishes_exactly_on_unit_square_boundary_float64():
    torch.manual_seed(0)
    with torch.no_grad():
        net = PINNNetwork(hidden_dims=[16, 16], hard_bc="unit_square_zero").double()
        _, bsamp = get_domain_samplers("unit_square")
        xb = bsamp(400).double()
        assert float(net(xb).abs().max()) < 1e-14
        xi = torch.rand(100, 2, dtype=torch.float64) * 0.8 + 0.1
        assert float(net(xi).abs().min()) > 0.0                       # not identically zero inside


def test_hard_bc_guards():
    cfg = {**TINY, "hard_bc": "unit_square_zero"}
    with pytest.raises(ValueError):
        train_pinn(BM["l_shaped"], cfg, _cfg())                        # wrong domain
    class _NonzeroG:                                                   # g != 0
        domain = "unit_square"
        boundary_condition = staticmethod(lambda x: x[:, 0:1] + 1.0)
    with pytest.raises(ValueError):
        train_pinn(_NonzeroG, cfg, _cfg())
    with pytest.raises(ValueError):
        PINNNetwork(hard_bc="nonsense")


def test_hard_bc_training_has_zero_boundary_loss():
    solver, hist = train_pinn(BM["poisson"], {**TINY, "hard_bc": "unit_square_zero"}, _cfg(), dtype="float64")
    assert max(hist.boundary_loss) < 1e-25
    assert hist.total_loss[-1] < hist.total_loss[0]


def test_boundary_tangents_unit_norm_and_on_edge_direction():
    for domain in ("unit_square", "l_shaped"):
        _, bsamp = get_domain_samplers(domain, {"boundary": {"corner_fraction": 0.3}} if domain == "l_shaped" else None)
        xb = bsamp(2000)
        tau = boundary_tangent(domain, xb)
        assert torch.allclose(tau.norm(dim=1), torch.ones(len(xb)))
        horiz = tau[:, 0].abs() > 0.5
        if domain == "unit_square":
            on_h = (xb[:, 1] == 0) | (xb[:, 1] == 1)
        else:
            on_h = (xb[:, 1].abs() == 1) | ((xb[:, 1] == 0) & (xb[:, 0] >= 0))
        # every sampled point is on a straight edge; tangent must be along that edge
        assert torch.equal(horiz, on_h)
        if domain == "l_shaped":   # re-entrant vertical edge x = 0, y < 0 has a vertical tangent
            m = (xb[:, 0] == 0) & (xb[:, 1] < 0)
            assert m.any() and bool((tau[m][:, 1].abs() == 1).all())


def test_tangential_loss_matches_closed_form():
    # poisson (g = 0) on the unit square: d_tau uhat = d_x uhat on y in {0,1}, d_y uhat on x in {0,1}
    bm = BM["poisson"]
    interior, boundary = get_domain_samplers(bm.domain)
    solver = PINNSolver(_Analytic(), bm.diffusion_coeff, bm.source_term, bm.boundary_condition,
                        interior, boundary, "cpu", domain="unit_square")
    torch.manual_seed(0)
    xb = boundary(500).double()
    got = float(solver.compute_tangential_loss(xb).detach())
    x, y = xb[:, 0], xb[:, 1]
    dx = torch.cos(x) * y**2 + 0.3
    dy = 2 * torch.sin(x) * y
    horiz = (y == 0) | (y == 1)
    want = float(torch.mean(torch.where(horiz, dx, dy) ** 2))
    assert abs(got - want) < 1e-12 * max(1.0, want)


def test_tangential_loss_subtracts_boundary_data_gradient():
    # L-shape: g = r^{2/3} sin(2 theta/3); on the re-entrant edges g == 0 so d_tau g == 0, on the
    # outer edges d_tau g is nonzero.  Check against the analytic exact_gradient . tau.
    bm = BM["l_shaped"]
    interior, boundary = get_domain_samplers(bm.domain)
    solver = PINNSolver(_Analytic(), bm.diffusion_coeff, bm.source_term, bm.boundary_condition,
                        interior, boundary, "cpu", domain="l_shaped")
    torch.manual_seed(1)
    xb = boundary(800).double()
    xb = xb[(xb.norm(dim=1) > 0.05)]                                   # avoid the corner singularity
    tau = boundary_tangent("l_shaped", xb)
    gu = bm.exact_gradient(xb)
    gh = torch.stack([torch.cos(xb[:, 0]) * xb[:, 1] ** 2 + 0.3, 2 * torch.sin(xb[:, 0]) * xb[:, 1]], dim=1)
    want = float(torch.mean(((gh - gu) * tau).sum(1) ** 2))
    got = float(solver.compute_tangential_loss(xb).detach())
    assert abs(got - want) < 1e-8 * max(1.0, want), (got, want)


def test_tangential_penalty_is_off_by_default_and_requires_domain():
    bm = BM["poisson"]
    interior, boundary = get_domain_samplers(bm.domain)
    solver = PINNSolver(PINNNetwork(**TINY), bm.diffusion_coeff, bm.source_term, bm.boundary_condition,
                        interior, boundary)                            # no domain given
    xr, xb = interior(50), boundary(20)
    t1, r1, b1 = solver.compute_total_loss(xr, xb, 1.0, 10.0)
    assert torch.allclose(t1, r1 + 10.0 * b1)
    with pytest.raises(ValueError):
        solver.compute_total_loss(xr, xb, 1.0, 10.0, 1.0)


def test_tangential_penalty_reduces_tangential_loss_and_gets_gradients():
    base, _ = train_pinn(BM["poisson"], TINY, _cfg(adam_epochs=300), seed=3, dtype="float64")
    pen, hist = train_pinn(BM["poisson"], TINY, _cfg(adam_epochs=300, w_tangent=10.0), seed=3, dtype="float64")
    tb = fixed_sample_losses(base, BM["poisson"])["tangential_loss"]
    tp = fixed_sample_losses(pen, BM["poisson"])["tangential_loss"]
    assert tp < tb, (tp, tb)
    assert len(hist.tangential_loss) == len(hist.total_loss) and hist.tangential_loss[-1] > 0


def test_float64_training_and_dtype_restored():
    assert torch.get_default_dtype() == torch.float32
    solver, hist = train_pinn(BM["poisson"], TINY, _cfg(), dtype="float64")
    assert all(p.dtype == torch.float64 for p in solver.net.parameters())
    assert torch.get_default_dtype() == torch.float32                  # restored
    assert np.isfinite(hist.total_loss).all()
    s32, _ = train_pinn(BM["poisson"], TINY, _cfg(), dtype="float32")
    assert all(p.dtype == torch.float32 for p in s32.net.parameters())


def test_corner_refined_sampler_concentrates_points():
    n = 4000
    torch.manual_seed(0)
    uni, _ = get_domain_samplers("l_shaped")
    ref, _ = get_domain_samplers("l_shaped", {"interior": {"corner_fraction": 0.5}})
    f_uni = float((uni(n).norm(dim=1) < 0.35).float().mean())
    f_ref = float((ref(n).norm(dim=1) < 0.35).float().mean())
    assert f_uni < 0.15 and f_ref > 0.5, (f_uni, f_ref)
    xr = ref(n)
    assert bool((~((xr[:, 0] > 0) & (xr[:, 1] < 0))).all())            # still inside the L-shape


def test_fixed_sample_losses_deterministic_and_rng_neutral():
    torch.manual_seed(0)
    solver, _ = train_pinn(BM["poisson"], TINY, _cfg(), dtype="float64")
    torch.manual_seed(123)
    a = fixed_sample_losses(solver, BM["poisson"], 2000, 500, seed=5)
    state_after = torch.get_rng_state()
    torch.manual_seed(123)
    assert torch.equal(state_after, torch.get_rng_state())            # global RNG untouched
    torch.manual_seed(999)                                             # different global state ...
    b = fixed_sample_losses(solver, BM["poisson"], 2000, 500, seed=5)
    assert a == b                                                      # ... same fixed sample


def _run_all():
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("PASS", k)


if __name__ == "__main__":
    _run_all()

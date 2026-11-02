"""Identity, known-answer and regression tests for the energy-error estimator.

Run with either
    python tests/test_identities.py      (no pytest needed)
    pytest tests/test_identities.py

Layer 1 needs only numpy/scipy.  Layer 2 needs torch.  Layer 3 needs the results/ and
paper/tables/ files and is skipped if they are absent.

Every tolerance below is justified by the recorded value in the comment next to it.
"""
import importlib.util
import json
import math
import os
import re
import sys

import numpy as np
from scipy.fft import dstn, idstn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PI = math.pi


class _Skip(Exception):
    """Raised by a test that cannot run in this environment (pytest turns it into a skip)."""


def _skip(message):
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest

        pytest.skip(message)
    raise _Skip(message)


def _load_p1_energy():
    """Import src/estimator/p1_energy.py WITHOUT importing the torch-dependent package."""
    path = os.path.join(ROOT, "src", "estimator", "p1_energy.py")
    spec = importlib.util.spec_from_file_location("p1_energy_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.p1_energy_square


p1_energy_square = _load_p1_energy()


# --------------------------------------------------------------------------- #
# Layer 1: numpy-only
# --------------------------------------------------------------------------- #
def _grid(n, fun):
    s = np.linspace(0.0, 1.0, n + 2)
    X, Y = np.meshgrid(s, s)
    return fun(X, Y)


def _harmonic_grid(n, tau):
    """Discrete harmonic extension of the boundary function tau (same algorithm as the repo)."""
    h = 1.0 / (n + 1)
    j = np.arange(1, n + 1)
    J, K = np.meshgrid(j, j)
    eig = (2.0 / h) ** 2 * (np.sin(J * PI * h / 2) ** 2 + np.sin(K * PI * h / 2) ** 2)
    s = np.linspace(0.0, 1.0, n + 2)
    z, o = np.zeros_like(s), np.ones_like(s)
    tb, tt, tl, tr = tau(s, z), tau(s, o), tau(z, s), tau(o, s)
    rhs = np.zeros((n, n))
    rhs[0, :] += tb[1:-1] / h**2
    rhs[-1, :] += tt[1:-1] / h**2
    rhs[:, 0] += tl[1:-1] / h**2
    rhs[:, -1] += tr[1:-1] / h**2
    inner = idstn(dstn(rhs, type=1, norm="ortho") / eig, type=1, norm="ortho")
    grid = np.zeros((n + 2, n + 2))
    grid[1:-1, 1:-1] = inner
    grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1] = tb, tt, tl, tr
    return grid


def _dual_norm(n, rfun):
    h = 1.0 / (n + 1)
    j = np.arange(1, n + 1)
    J, K = np.meshgrid(j, j)
    eig = (2.0 / h) ** 2 * (np.sin(J * PI * h / 2) ** 2 + np.sin(K * PI * h / 2) ** 2)
    x = np.linspace(h, 1 - h, n)
    X, Y = np.meshgrid(x, x)
    r = rfun(X, Y)
    psi = idstn(dstn(r, type=1, norm="ortho") / eig, type=1, norm="ortho")
    return math.sqrt(h * h * float(np.sum(r * psi)))


def test_p1_energy_linear_functions_are_exact():
    # Boundary edges must carry weight 1/2: a linear function has exact energy for every n.
    for n in (8, 31, 96):
        assert abs(p1_energy_square(_grid(n, lambda X, Y: X)) - 1.0) < 1e-12
        assert abs(p1_energy_square(_grid(n, lambda X, Y: X + 2 * Y)) - 5.0) < 1e-12
    # The original all-edges-weight-1 sum gives 1 + h at n = 96; documented here so that a
    # regression to the old formula cannot pass the test above.
    g = _grid(96, lambda X, Y: X)
    old = np.sum(np.diff(g, axis=0) ** 2) + np.sum(np.diff(g, axis=1) ** 2)
    assert abs(old - (1.0 + 1.0 / 97.0)) < 1e-12


def test_p1_energy_quadratic_is_second_order():
    # w = x^2 - y^2 is harmonic with exact energy 8/3 (recorded: errors 8.5e-5, 2.2e-5, 5.5e-6, 1.4e-6
    # for ||grad w|| at n = 48, 96, 192, 384, i.e. ratio 4 = second order).
    exact = math.sqrt(8.0 / 3.0)
    err = {}
    for n in (96, 192):
        energy = p1_energy_square(_harmonic_grid(n, lambda x, y: x**2 - y**2))
        err[n] = abs(math.sqrt(energy) - exact)
    assert err[192] < 1e-5
    assert 3.5 < err[96] / err[192] < 4.5


def test_pythagoras_identity_constant_coefficient():
    # u = sin sin (a = 1), u_hat = u - eps e0 - w with e0 = sin sin, w = x^2 - y^2:
    # ||R||_{H^-1} = eps * pi / sqrt(2) exactly, ||grad w||^2 = 8/3, and the error is the
    # Pythagorean combination.  Recorded ratios sqrt(R^2 + W^2) / exact at n = 48 .. 384:
    # 0.999949, 0.999987, 0.999997, 0.9999992.
    eps = 0.05
    exact_r = eps * PI / math.sqrt(2.0)
    exact_err = math.sqrt(exact_r**2 + 8.0 / 3.0)
    n = 96
    r = _dual_norm(n, lambda X, Y: 2 * PI**2 * eps * np.sin(PI * X) * np.sin(PI * Y))
    w = math.sqrt(p1_energy_square(_harmonic_grid(n, lambda x, y: x**2 - y**2)))
    assert abs(r / exact_r - 1.0) < 1e-4                       # recorded 4.4e-5
    assert abs(math.hypot(r, w) / exact_err - 1.0) < 1e-4      # recorded 1.3e-5
    # the sum form lies in [1, sqrt 2] times the error
    ratio = (r + w) / exact_err
    assert 1.0 <= ratio <= math.sqrt(2.0)


def test_variable_coefficient_bound_holds_and_is_not_vacuous():
    # a = 1 + 0.5 sin sin (alpha = 1, beta = 3/2); error e = d = c (x + 0.5 sin sin + 0.3 x^2 y).
    # Theorem 3.3 with the a-weighted lifting energy: bound/true recorded 1.1411 (n = 96).
    c = 0.01
    a = lambda x, y: 1.0 + 0.5 * np.sin(PI * x) * np.sin(PI * y)

    def rvar(X, Y):
        ax = 0.5 * PI * np.cos(PI * X) * np.sin(PI * Y)
        ay = 0.5 * PI * np.sin(PI * X) * np.cos(PI * Y)
        dx = c * (1 + 0.5 * PI * np.cos(PI * X) * np.sin(PI * Y) + 0.6 * X * Y)
        dy = c * (0.5 * PI * np.sin(PI * X) * np.cos(PI * Y) + 0.3 * X**2)
        lap = c * (-(PI**2) * np.sin(PI * X) * np.sin(PI * Y) + 0.6 * Y)
        return -(ax * dx + ay * dy + a(X, Y) * lap)

    m = 1000
    xs = (np.arange(m) + 0.5) / m
    X, Y = np.meshgrid(xs, xs)
    dx = c * (1 + 0.5 * PI * np.cos(PI * X) * np.sin(PI * Y) + 0.6 * X * Y)
    dy = c * (0.5 * PI * np.sin(PI * X) * np.cos(PI * Y) + 0.3 * X**2)
    true = math.sqrt(float(np.mean(dx**2 + dy**2)))

    n = 96
    rn = _dual_norm(n, rvar)
    grid = _harmonic_grid(n, lambda x, y: c * (x + 0.3 * x**2 * y))
    w_a2 = p1_energy_square(grid, a)
    w_2 = p1_energy_square(grid)
    alpha, beta = 1.0, 1.5
    bound_a = math.sqrt(rn**2 / alpha**2 + w_a2 / alpha)
    bound_sum = rn / alpha + math.sqrt(beta / alpha) * math.sqrt(w_2)
    assert 1.0 <= bound_a / true <= 1.2        # recorded 1.1411
    assert bound_a <= bound_sum + 1e-15        # sqrt(x^2 + y^2) <= x + y
    assert bound_sum / true >= 1.0             # recorded 1.7
    # Proposition 3.4 (lower bounds, any a): ||grad w|| <= ||grad e||, ||R|| <= beta ||grad e||
    assert math.sqrt(w_2) <= true
    assert rn <= beta * true


# --------------------------------------------------------------------------- #
# Layer 2: needs torch (real or the sandbox stand-in)
# --------------------------------------------------------------------------- #
def _torch_or_none():
    try:
        import torch  # noqa: F401

        return torch
    except Exception:
        return None


def _make_repo_objects(torch):
    class Solver:
        device = "cpu"

        def predict(self, x):
            return torch.zeros((x.shape[0], 1), dtype=torch.float32)

        def predict_with_gradient(self, x):
            return (
                torch.zeros((x.shape[0], 1), dtype=torch.float32),
                torch.zeros((x.shape[0], 2), dtype=torch.float32),
            )

        def compute_pde_residual(self, x):
            return torch.zeros((x.shape[0], 1), dtype=torch.float32)

    class Linear:
        """u = x + 2y with a = 2: error energy sqrt(a) * sqrt(5) = sqrt(10) in the a-norm."""

        domain = "unit_square"
        coercivity_constant = 2.0
        coefficient_upper_bound = 2.0

        @staticmethod
        def diffusion_coeff(x):
            return 2.0 * torch.ones((x.shape[0], 1), dtype=x.dtype)

        @staticmethod
        def boundary_condition(x):
            return x[:, 0:1] + 2.0 * x[:, 1:2]

        @staticmethod
        def exact_solution(x):
            return x[:, 0:1] + 2.0 * x[:, 1:2]

        @staticmethod
        def exact_gradient(x):
            g = torch.cat([torch.ones((x.shape[0], 1)), 2.0 * torch.ones((x.shape[0], 1))], dim=1)
            return g

    return Solver, Linear


def test_repo_lifting_is_exact_for_linear_trace():
    torch = _torch_or_none()
    if torch is None:
        _skip("torch not installed")
    from src.estimator.lifting import compute_boundary_lifting_norm

    Solver, Linear = _make_repo_objects(torch)
    for n in (48, 96):
        plain = compute_boundary_lifting_norm(Solver(), Linear, n, 1e-12)
        weighted = compute_boundary_lifting_norm(Solver(), Linear, n, 1e-12, weighted=True)
        assert abs(plain - math.sqrt(5.0)) < 1e-7          # recorded: exact to 1e-10 (float32 trace)
        assert abs(weighted - math.sqrt(10.0)) < 1e-7      # a = 2


def test_repo_estimator_does_not_use_exact_solution_to_choose_mesh():
    torch = _torch_or_none()
    if torch is None:
        _skip("torch not installed")
    from src.estimator import ValidatedEstimator

    Solver, Linear = _make_repo_objects(torch)

    class Scaled(Linear):
        # same boundary data and residual, 100x larger "true" error: only the ratios may change
        @staticmethod
        def exact_gradient(x):
            return 100.0 * Linear.exact_gradient(x)

    ref = ValidatedEstimator(Solver(), Linear, fem_mesh_size=48, max_mesh_size=192).evaluate(0.0)
    alt = ValidatedEstimator(Solver(), Scaled, fem_mesh_size=48, max_mesh_size=192).evaluate(0.0)
    assert [m["mesh_size"] for m in ref.mesh_history] == [48.0, 96.0, 192.0]
    assert ref.mesh_history == alt.mesh_history
    assert ref.mesh_size_used == alt.mesh_size_used == 192
    assert ref.discretization_margin == 0.0


# --------------------------------------------------------------------------- #
# Layer 3: every number printed in a paper table must exist in the results JSON
# --------------------------------------------------------------------------- #
_FLOAT = re.compile(r"\d\.\d{4}e[+-]\d{2}|\d\.\d{4}(?![\de])")


def _leaves(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _leaves(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _leaves(value)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield float(obj)


def test_table_numbers_match_results_json():
    results_path = os.path.join(ROOT, "results", "validated_results.json")
    table_dir = os.path.join(ROOT, "paper", "tables")
    if not (os.path.exists(results_path) and os.path.isdir(table_dir)):
        _skip("results or tables missing")
    with open(results_path, encoding="utf-8") as handle:
        results = json.load(handle)
    pool = [v for item in results for v in _leaves(item["metrics"])]
    ratios = []
    for item in results:
        m = item["metrics"]
        ratios += [
            m["residual_contribution"] / m["true_error_energy"],      # eta_r
            m["boundary_lifting_norm"] / m["true_error_energy"],
        ]
    pool += ratios
    for name in sorted(os.listdir(table_dir)):
        if not name.endswith(".tex"):
            continue
        with open(os.path.join(table_dir, name), encoding="utf-8") as handle:
            text = handle.read()
        for token in _FLOAT.findall(text):
            value = float(token)
            ok = any(abs(value - p) <= 6e-5 * max(abs(p), 1e-300) + 5e-5 * (0 if "e" in token else 1) for p in pool)
            assert ok, f"{name}: {token} is not reproduced by results/validated_results.json"


def _run_all():
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except _Skip as exc:
                print(f"[SKIP] {name}: {exc}")
            except AssertionError as exc:
                failures += 1
                print(f"[FAIL] {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
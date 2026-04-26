#!/usr/bin/env python3
"""Lightweight regression checks for the validated estimator pipeline."""

import numpy as np
import torch

from src.benchmarks import get_supported_benchmarks
from src.estimator import ValidatedEstimator


class _DummyNet:
    def to(self, device):
        return self

    def eval(self):
        return self


class _DummySolver:
    def __init__(self):
        self.device = "cpu"
        self.net = _DummyNet()

    def compute_pde_residual(self, x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)

    def predict(self, x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)

    def predict_with_gradient(self, x):
        u = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)
        grad = torch.zeros((x.shape[0], 2), dtype=torch.float32, device=x.device)
        return u, grad


class _UnitSquareHarmonicBenchmark:
    name = "harmonic-x"
    key = "harmonic_x"
    domain = "unit_square"
    coercivity_constant = 1.0
    description = "Synthetic harmonic lifting validation benchmark."

    @staticmethod
    def boundary_condition(x):
        return x[:, 0:1]

    @staticmethod
    def exact_solution(x):
        return x[:, 0:1]

    @staticmethod
    def exact_gradient(x):
        return torch.cat(
            [
                torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device),
                torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device),
            ],
            dim=1,
        )


class _LShapedHarmonicBenchmark:
    name = "lshaped-harmonic-x"
    key = "lshaped_harmonic_x"
    domain = "l_shaped"
    coercivity_constant = 1.0
    description = "Synthetic harmonic lifting validation benchmark on the L-shaped domain."

    @staticmethod
    def boundary_condition(x):
        return x[:, 0:1]

    @staticmethod
    def exact_solution(x):
        return x[:, 0:1]

    @staticmethod
    def exact_gradient(x):
        return torch.cat(
            [
                torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device),
                torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device),
            ],
            dim=1,
        )


def _check(name, condition, detail=""):
    if condition:
        print(f"[PASS] {name}")
        return True
    print(f"[FAIL] {name}: {detail}")
    return False


def main() -> int:
    passed = 0
    failed = 0

    print("Validating supported benchmark registry")
    supported = get_supported_benchmarks()
    if _check("Three supported benchmarks", set(supported.keys()) == {"poisson", "variable_coefficient", "l_shaped"}):
        passed += 1
    else:
        failed += 1

    print("\nValidating unit-square harmonic lifting benchmark")
    estimator = ValidatedEstimator(
        solver=_DummySolver(),
        benchmark=_UnitSquareHarmonicBenchmark(),
        fem_mesh_size=96,
        max_mesh_size=192,
        eval_seed=7,
    )
    result = estimator.evaluate(training_loss=0.0)
    checks = [
        ("Effectivity >= 1", result.effectivity >= 1.0 - 1e-6, f"{result.effectivity:.6f}"),
        ("Boundary lifting close to one", abs(result.boundary_lifting_norm - 1.0) < 5e-2, f"{result.boundary_lifting_norm:.6f}"),
        ("Residual contribution zero", abs(result.residual_contribution) < 1e-12, f"{result.residual_contribution:.3e}"),
        ("True energy near one", abs(result.true_error_energy - 1.0) < 1e-6, f"{result.true_error_energy:.6f}"),
    ]
    for name, condition, detail in checks:
        if _check(name, condition, detail):
            passed += 1
        else:
            failed += 1

    print("\nValidating L-shaped harmonic lifting benchmark")
    lshape_estimator = ValidatedEstimator(
        solver=_DummySolver(),
        benchmark=_LShapedHarmonicBenchmark(),
        fem_mesh_size=80,
        max_mesh_size=160,
        eval_seed=11,
    )
    lshape_result = lshape_estimator.evaluate(training_loss=0.0)
    checks = [
        ("L-shaped effectivity >= 1", lshape_result.effectivity >= 1.0 - 1e-6, f"{lshape_result.effectivity:.6f}"),
        (
            "L-shaped boundary lifting close to sqrt(3)",
            abs(lshape_result.boundary_lifting_norm - np.sqrt(3.0)) < 5e-2,
            f"{lshape_result.boundary_lifting_norm:.6f}",
        ),
        ("L-shaped residual contribution zero", abs(lshape_result.residual_contribution) < 1e-12, f"{lshape_result.residual_contribution:.3e}"),
        (
            "L-shaped true energy near sqrt(3)",
            abs(lshape_result.true_error_energy - np.sqrt(3.0)) < 5e-2,
            f"{lshape_result.true_error_energy:.6f}",
        ),
    ]
    for name, condition, detail in checks:
        if _check(name, condition, detail):
            passed += 1
        else:
            failed += 1

    print("\nSummary")
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

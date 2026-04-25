#!/usr/bin/env python3
import numpy as np
import torch

from src.benchmarks import get_supported_benchmarks
from src.estimator import ValidatedEstimator


PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name}: {detail}")


class DummyNet:
    def to(self, device):
        return self

    def eval(self):
        return self


class DummySolver:
    def __init__(self):
        self.device = "cpu"
        self.net = DummyNet()

    def compute_pde_residual(self, x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)

    def predict(self, x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)

    def predict_with_gradient(self, x):
        u = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)
        grad = torch.zeros((x.shape[0], 2), dtype=torch.float32, device=x.device)
        return u, grad


class HarmonicBenchmark:
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


class LShapedHarmonicBenchmark:
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


print("Validating supported benchmark registry")
supported = get_supported_benchmarks()
check("Three supported benchmarks", set(supported.keys()) == {"poisson", "variable_coefficient", "l_shaped"})

print("\nValidating harmonic lifting benchmark")
estimator = ValidatedEstimator(
    solver=DummySolver(),
    benchmark=HarmonicBenchmark(),
    fem_mesh_size=96,
    max_mesh_size=192,
    eval_seed=7,
)
result = estimator.evaluate(training_loss=0.0)
check("Effectivity >= 1", result.effectivity >= 1.0 - 1e-6, f"{result.effectivity:.6f}")
check("Boundary lifting close to one", abs(result.boundary_lifting_norm - 1.0) < 5e-2, f"{result.boundary_lifting_norm:.6f}")
check("Residual contribution zero", abs(result.residual_contribution) < 1e-12, f"{result.residual_contribution:.3e}")
check("True energy near one", abs(result.true_error_energy - 1.0) < 1e-6, f"{result.true_error_energy:.6f}")

print("\nValidating L-shaped harmonic lifting benchmark")
lshape_estimator = ValidatedEstimator(
    solver=DummySolver(),
    benchmark=LShapedHarmonicBenchmark(),
    fem_mesh_size=80,
    max_mesh_size=160,
    eval_seed=11,
)
lshape_result = lshape_estimator.evaluate(training_loss=0.0)
check("L-shaped effectivity >= 1", lshape_result.effectivity >= 1.0 - 1e-6, f"{lshape_result.effectivity:.6f}")
check(
    "L-shaped boundary lifting close to sqrt(3)",
    abs(lshape_result.boundary_lifting_norm - np.sqrt(3.0)) < 5e-2,
    f"{lshape_result.boundary_lifting_norm:.6f}",
)
check("L-shaped residual contribution zero", abs(lshape_result.residual_contribution) < 1e-12, f"{lshape_result.residual_contribution:.3e}")
check(
    "L-shaped true energy near sqrt(3)",
    abs(lshape_result.true_error_energy - np.sqrt(3.0)) < 5e-2,
    f"{lshape_result.true_error_energy:.6f}",
)

print("\nSummary")
print(f"{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)

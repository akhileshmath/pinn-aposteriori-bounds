#!/usr/bin/env python3
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.benchmarks import get_supported_benchmarks
from src.estimator import ValidatedEstimator


torch.set_default_dtype(torch.float32)


class AnalyticSolver:
    def __init__(self, trial_function: Callable[[torch.Tensor], torch.Tensor]):
        self.device = "cpu"
        self.trial_function = trial_function
        self.net = self

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.trial_function(x)

    def predict_with_gradient(self, x: torch.Tensor):
        x_req = x.clone().requires_grad_(True)
        u = self.trial_function(x_req)
        if not u.requires_grad:
            u = u + 0.0 * x_req[:, 0:1]
        grad = torch.autograd.grad(
            u,
            x_req,
            grad_outputs=torch.ones_like(u),
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]
        return u.detach(), grad.detach()

    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        x_req = x.clone().requires_grad_(True)
        u = self.trial_function(x_req)
        if not u.requires_grad:
            u = u + 0.0 * x_req[:, 0:1]
        grad = torch.autograd.grad(
            u,
            x_req,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        divergence = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        for axis in range(x.shape[1]):
            grad_axis = grad[:, axis : axis + 1]
            if not grad_axis.requires_grad:
                second = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
            else:
                second = torch.autograd.grad(
                    grad_axis,
                    x_req,
                    grad_outputs=torch.ones_like(grad_axis),
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if second is None:
                    second = torch.zeros((x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
                second = second[:, axis : axis + 1]
            divergence = divergence + second
        return divergence.detach()


@dataclass
class DiagnosticCase:
    name: str
    trial_function: Callable[[torch.Tensor], torch.Tensor]
    expected_mode: str


def _bubble(x: torch.Tensor) -> torch.Tensor:
    xx = x[:, 0:1]
    yy = x[:, 1:2]
    return xx * yy * (1.0 - xx * xx) * (1.0 - yy * yy)


def _smooth_unit_square_mode(x: torch.Tensor) -> torch.Tensor:
    xx = 0.5 * (x[:, 0:1] + 1.0)
    yy = 0.5 * (x[:, 1:2] + 1.0)
    return torch.sin(math.pi * xx) * torch.sin(math.pi * yy)


def _build_cases(benchmark) -> List[DiagnosticCase]:
    return [
        DiagnosticCase(
            name="zero_state",
            trial_function=lambda x: torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device),
            expected_mode="mixed_large",
        ),
        DiagnosticCase(
            name="harmonic_x",
            trial_function=lambda x: 0.1 * x[:, 0:1],
            expected_mode="boundary_dominated",
        ),
        DiagnosticCase(
            name="harmonic_x_plus_y",
            trial_function=lambda x: 0.05 * (x[:, 0:1] + x[:, 1:2]),
            expected_mode="boundary_dominated",
        ),
        DiagnosticCase(
            name="interior_bubble_only",
            trial_function=lambda x: 0.05 * _bubble(x),
            expected_mode="residual_dominated",
        ),
        DiagnosticCase(
            name="mixed_smooth_mode",
            trial_function=lambda x: 0.02 * x[:, 0:1] + 0.03 * _bubble(x) * _smooth_unit_square_mode(x),
            expected_mode="mixed_large",
        ),
        DiagnosticCase(
            name="bubble_times_mode",
            trial_function=lambda x: 0.03 * _bubble(x) * _smooth_unit_square_mode(x),
            expected_mode="residual_dominated",
        ),
    ]


def _evaluate_case(benchmark, case: DiagnosticCase) -> Dict:
    solver = AnalyticSolver(case.trial_function)
    estimator = ValidatedEstimator(
        solver=solver,
        benchmark=benchmark,
        fem_mesh_size=40,
        max_mesh_size=160,
        eval_seed=17,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = estimator.evaluate(training_loss=0.0)
            error = None
        except Exception as exc:
            result = None
            error = str(exc)

    return {
        "case": case,
        "result": result,
        "error": error,
        "warnings": [str(item.message) for item in caught],
    }


def _format_mesh_history(mesh_history: List[Dict[str, float]]) -> str:
    pieces = []
    for item in mesh_history:
        pieces.append(
            "h={mesh:.0f}: total={total:.4e}, res={res:.4e}, bc={bc:.4e}".format(
                mesh=item["mesh_size"],
                total=item["estimate"],
                res=item["residual_contribution"],
                bc=item["boundary_contribution"],
            )
        )
    return " | ".join(pieces)


def _ascii(text: str) -> str:
    return text.replace("η", "eta")


def _check_case(case: DiagnosticCase, result) -> List[str]:
    issues = []
    mesh_history = result.mesh_history
    totals = [item["estimate"] for item in mesh_history]
    residual = result.residual_contribution
    boundary = result.boundary_lifting_norm

    if result.effectivity < 1.0:
        issues.append(f"effectivity below one: {result.effectivity:.6f}")

    if len(totals) >= 2 and totals[-1] > totals[0] + 1e-8:
        issues.append(
            f"estimate increased under refinement: {totals[0]:.4e} -> {totals[-1]:.4e}"
        )

    if case.expected_mode == "boundary_dominated" and not (boundary > 5.0 * max(residual, 1e-14)):
        issues.append(
            f"boundary-dominated case not boundary dominated: res={residual:.4e}, bc={boundary:.4e}"
        )

    if case.expected_mode == "residual_dominated" and not (residual > 5.0 * max(boundary, 1e-14)):
        issues.append(
            f"residual-dominated case not residual dominated: res={residual:.4e}, bc={boundary:.4e}"
        )

    return issues


def main() -> int:
    benchmark = get_supported_benchmarks()["l_shaped"]
    cases = _build_cases(benchmark)
    overall_issues = []

    print("L-shaped Manufactured Diagnostics")
    print("=" * 72)
    for case in cases:
        print(f"\nCase: {case.name} ({case.expected_mode})")
        record = _evaluate_case(benchmark, case)
        result = record["result"]
        if result is None:
            issue = _ascii(record["error"] or "estimator evaluation failed")
            overall_issues.append(f"{case.name}: {issue}")
            print(f"  evaluation failed      : {issue}")
            if record["warnings"]:
                print("  warnings               :")
                for message in record["warnings"]:
                    print(f"    - {_ascii(message)}")
            continue

        issues = _check_case(case, result)
        overall_issues.extend([f"{case.name}: {issue}" for issue in issues])

        print(f"  true energy error      : {result.true_error_energy:.6e}")
        print(f"  estimated energy error : {result.estimated_error_energy:.6e}")
        print(f"  residual contribution  : {result.residual_contribution:.6e}")
        print(f"  boundary contribution  : {result.boundary_lifting_norm:.6e}")
        print(f"  effectivity            : {result.effectivity:.6f}")
        print(f"  discretization margin  : {result.discretization_margin:.6e}")
        print(f"  mesh history           : {_format_mesh_history(result.mesh_history)}")
        if record["warnings"]:
            print("  warnings               :")
            for message in record["warnings"]:
                print(f"    - {_ascii(message)}")
        if issues:
            print("  diagnostic issues      :")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  diagnostic issues      : none")

    print("\nSummary")
    print("-" * 72)
    if overall_issues:
        for issue in overall_issues:
            print(f"- {_ascii(issue)}")
        return 1

    print("All manufactured diagnostics behaved as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

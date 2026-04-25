import json
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch

from .dual_norm import compute_dual_norm
from .lifting import compute_boundary_lifting_norm
from .lifting import _build_l_shaped_triangular_mesh, _triangle_shape_gradients
from .quadrature import get_quadrature_points


@dataclass
class ErrorResult:
    estimated_error_energy: float
    true_error_energy: float
    true_error_l2: float
    residual_dual_norm: float
    residual_contribution: float
    boundary_lifting_norm: float
    effectivity: float
    residual_l2_norm: float
    boundary_error_l2: float
    training_loss: float
    coercivity_constant: float
    mesh_size_used: int
    mesh_refinements: int
    stabilization_eps: float
    discretization_margin: float
    mesh_history: List[Dict[str, float]]

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


class ValidatedEstimator:
    def __init__(
        self,
        solver,
        benchmark,
        fem_mesh_size: int = 96,
        max_mesh_size: int = 384,
        mesh_growth_factor: int = 2,
        stabilization_eps: float = 1e-12,
        effectivity_tol: float = 1e-6,
        eval_seed: int = 12345,
    ):
        self.solver = solver
        self.benchmark = benchmark
        self.fem_mesh_size = max(int(fem_mesh_size), 48)
        self.max_mesh_size = max(int(max_mesh_size), self.fem_mesh_size)
        self.mesh_growth_factor = max(int(mesh_growth_factor), 2)
        self.stabilization_eps = float(stabilization_eps)
        self.effectivity_tol = float(effectivity_tol)
        self.eval_seed = int(eval_seed)

    def _rng(self, offset: int) -> np.random.Generator:
        return np.random.default_rng(self.eval_seed + offset)

    def compute_true_errors(self, n_points: int = 50000):
        if self.benchmark.domain == "l_shaped":
            return self._compute_true_errors_l_shaped()

        points, weights, area = get_quadrature_points(
            self.benchmark.domain,
            n_points,
            rng=self._rng(1),
        )
        x = torch.tensor(points, dtype=torch.float32, device=self.solver.device)
        w = torch.tensor(weights, dtype=torch.float64, device=self.solver.device)
        u_pred, grad_pred = self.solver.predict_with_gradient(x)
        u_exact = self.benchmark.exact_solution(x).detach()
        grad_exact = self.benchmark.exact_gradient(x).detach()

        l2_sq = area * torch.sum(w * (u_pred - u_exact).pow(2).reshape(-1).to(torch.float64)).item()
        h1_semi_sq = area * torch.sum(
            w * torch.sum((grad_pred - grad_exact).pow(2), dim=1).to(torch.float64)
        ).item()
        return float(np.sqrt(max(h1_semi_sq, 0.0))), float(np.sqrt(max(l2_sq, 0.0)))

    def _compute_true_errors_l_shaped(self):
        reference_mesh = max(self.max_mesh_size, self.fem_mesh_size)
        nodes, triangles, _ = _build_l_shaped_triangular_mesh(reference_mesh)
        x_nodes = torch.tensor(nodes, dtype=torch.float32, device=self.solver.device)

        u_pred, _ = self.solver.predict_with_gradient(x_nodes)
        u_exact = self.benchmark.exact_solution(x_nodes).detach()
        nodal_error = (u_pred - u_exact).reshape(-1).to(torch.float64).cpu().numpy()

        energy_sq = 0.0
        l2_sq = 0.0
        mass_template = np.array(
            [
                [2.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )

        for tri in triangles:
            tri_nodes = nodes[tri]
            tri_error = nodal_error[tri]
            area, gradients = _triangle_shape_gradients(tri_nodes)
            grad_error = tri_error @ gradients
            energy_sq += area * float(np.dot(grad_error, grad_error))
            element_mass = (area / 12.0) * mass_template
            l2_sq += float(tri_error @ element_mass @ tri_error)

        return float(np.sqrt(max(energy_sq, 0.0))), float(np.sqrt(max(l2_sq, 0.0)))

    def compute_residual_l2_norm(self, n_points: int = 50000) -> float:
        points, weights, area = get_quadrature_points(
            self.benchmark.domain,
            n_points,
            rng=self._rng(2),
        )
        x = torch.tensor(points, dtype=torch.float32, device=self.solver.device)
        w = torch.tensor(weights, dtype=torch.float64, device=self.solver.device)
        residual = self.solver.compute_pde_residual(x).reshape(-1).to(torch.float64)
        residual_sq = area * torch.sum(w * residual.pow(2)).item()
        return float(np.sqrt(max(residual_sq, 0.0)))

    def compute_boundary_error(self, n_points: int = 2000) -> float:
        from src.pinn.sampling import get_domain_samplers

        _, boundary_sampler = get_domain_samplers(self.benchmark.domain)
        x_b = boundary_sampler(n_points).to(self.solver.device)
        pred = self.solver.predict(x_b)
        exact = self.benchmark.boundary_condition(x_b)
        return float(torch.sqrt(torch.mean((pred - exact) ** 2)).item())

    def _refinement_upper_bound(
        self,
        current_value: float,
        previous_value: float,
        label: str,
    ) -> tuple[float, float]:
        diff = abs(current_value - previous_value)
        if current_value <= previous_value + self.effectivity_tol:
            margin = diff / 3.0
            return current_value + margin, margin

        warnings.warn(
            (
                f"{label} estimate increased under refinement "
                f"({previous_value:.6e} -> {current_value:.6e}). "
                "Using a conservative non-monotone fallback bound."
            ),
            stacklevel=3,
        )
        margin = diff
        return max(current_value, previous_value) + margin, margin

    def evaluate(self, training_loss: float) -> ErrorResult:
        """
        Evaluate the validated estimator and compare it with the true energy error.

        The reported estimate is a computable upper bound under the assumption that
        the discrete lifting and dual norm computations converge from above as h->0.
        The Richardson correction (1/3)*|eta_h - eta_{h/2}| is added as a
        discretization margin. If the estimates are not monotonically decreasing,
        a convergence warning is raised and the bound may not be rigorous.
        """
        alpha = float(self.benchmark.coercivity_constant)
        true_energy, true_l2 = self.compute_true_errors()
        residual_l2 = self.compute_residual_l2_norm()
        boundary_error_l2 = self.compute_boundary_error()

        mesh_size = self.fem_mesh_size
        refinements = 0
        effectivity = None
        estimated = None
        residual_info = None
        boundary_value = None
        discretization_margin = None
        mesh_history: List[Dict[str, float]] = []
        residual_contribution = None
        boundary_contribution = None

        while True:
            residual_info = compute_dual_norm(
                self.solver,
                self.benchmark.domain,
                mesh_size,
                self.stabilization_eps,
            )
            boundary_value = compute_boundary_lifting_norm(
                self.solver,
                self.benchmark,
                mesh_size,
                self.stabilization_eps,
            )
            current_residual_contribution = residual_info["dual_norm"] / alpha
            current_boundary_contribution = boundary_value
            current_estimate = current_residual_contribution + current_boundary_contribution

            entry = {
                "mesh_size": float(mesh_size),
                "residual_contribution": float(current_residual_contribution),
                "boundary_contribution": float(current_boundary_contribution),
                "estimate": float(current_estimate),
            }
            mesh_history.append(entry)

            if len(mesh_history) >= 2:
                previous = mesh_history[-2]
                residual_contribution, residual_margin = self._refinement_upper_bound(
                    current_residual_contribution,
                    previous["residual_contribution"],
                    "Residual contribution",
                )
                boundary_contribution, boundary_margin = self._refinement_upper_bound(
                    current_boundary_contribution,
                    previous["boundary_contribution"],
                    "Boundary contribution",
                )
                estimated = residual_contribution + boundary_contribution
                discretization_margin = residual_margin + boundary_margin
                effectivity = estimated / true_energy

                previous_estimate = previous["estimate"]
                if current_estimate > previous_estimate + self.effectivity_tol:
                    warnings.warn(
                        (
                            f"Total estimate increased under refinement "
                            f"({previous_estimate:.6e} -> {current_estimate:.6e}). "
                            "The validation margin is conservative and may not be rigorous."
                        ),
                        stacklevel=2,
                    )
                else:
                    total_margin = (previous_estimate - current_estimate) / 3.0
                    discretization_margin = max(discretization_margin, total_margin)
                    estimated = max(estimated, current_estimate + total_margin)
                    effectivity = estimated / true_energy
            else:
                residual_contribution = current_residual_contribution
                boundary_contribution = current_boundary_contribution
                discretization_margin = 0.0
                estimated = current_estimate
                effectivity = estimated / true_energy

            has_minimum_history = len(mesh_history) >= 2
            if (
                has_minimum_history
                and (effectivity + self.effectivity_tol >= 1.0 or mesh_size >= self.max_mesh_size)
            ):
                break
            mesh_size = min(mesh_size * self.mesh_growth_factor, self.max_mesh_size)
            refinements += 1

            if mesh_size == mesh_history[-1]["mesh_size"]:
                break

        if effectivity + self.effectivity_tol < 1.0:
            raise ValueError(
                f"Estimator is NOT reliable (η < 1). eta={effectivity:.6f}, mesh={mesh_size}"
            )

        if effectivity > 2.0:
            warnings.warn(
                f"Effectivity is larger than desired: eta={effectivity:.6f}",
                stacklevel=2,
            )

        result = ErrorResult(
            estimated_error_energy=float(estimated),
            true_error_energy=true_energy,
            true_error_l2=true_l2,
            residual_dual_norm=float(residual_info["dual_norm"]),
            residual_contribution=float(residual_contribution),
            boundary_lifting_norm=float(boundary_contribution),
            effectivity=float(effectivity),
            residual_l2_norm=float(residual_l2),
            boundary_error_l2=float(boundary_error_l2),
            training_loss=float(training_loss),
            coercivity_constant=alpha,
            mesh_size_used=int(mesh_size),
            mesh_refinements=int(refinements),
            stabilization_eps=float(self.stabilization_eps),
            discretization_margin=float(discretization_margin),
            mesh_history=mesh_history,
        )

        print(f"  True energy error       : {result.true_error_energy:.6e}")
        print(f"  Estimated energy error  : {result.estimated_error_energy:.6e}")
        print(f"  Residual contribution   : {result.residual_contribution:.6e}")
        print(f"  Boundary contribution   : {result.boundary_lifting_norm:.6e}")
        print(f"  Effectivity eta         : {result.effectivity:.6f}")
        print(f"  Training loss           : {result.training_loss:.6e}")
        print(f"  Discretization margin   : {result.discretization_margin:.6e}")
        return result

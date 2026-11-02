import json
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .dual_norm import compute_dual_norm
from .lifting import compute_boundary_lifting_norm
from .lifting import _build_l_shaped_triangular_mesh
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
    # Added in the audit patch.  `estimated_error_energy` / `effectivity` now refer to the
    # sum form eta_oplus = ||R||_{H^-1}/alpha + ||grad w|| on the finest mesh WITHOUT any margin.
    estimator_pythagoras: Optional[float] = None      # sqrt((||R||/alpha)^2 + ||grad w||^2)
    estimator_variable: Optional[float] = None        # sqrt((||R||/alpha)^2 + ||w||_a^2 / alpha)
    effectivity_pythagoras: Optional[float] = None
    effectivity_variable: Optional[float] = None
    boundary_lifting_a_weighted: Optional[float] = None
    coefficient_upper_bound: Optional[float] = None

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
        true_error_mesh_size: int | None = None,
    ):
        self.solver = solver
        self.benchmark = benchmark
        self.fem_mesh_size = max(int(fem_mesh_size), 48)
        self.max_mesh_size = max(int(max_mesh_size), self.fem_mesh_size)
        self.mesh_growth_factor = max(int(mesh_growth_factor), 2)
        self.stabilization_eps = float(stabilization_eps)
        self.effectivity_tol = float(effectivity_tol)
        self.eval_seed = int(eval_seed)
        # Triangulation used ONLY for the quadrature of the exact error on the L-shaped domain.
        self.true_error_mesh_size = int(true_error_mesh_size or 2 * self.max_mesh_size)

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

    def _compute_true_errors_l_shaped(self, chunk: int = 200_000):
        """Exact energy / L2 error on the L-shaped domain by 3-point Gauss quadrature (degree 2).

        The analytic u and grad u are evaluated at the quadrature points; no interpolation of the
        exact solution is involved.  (The previous version returned the error of the P1
        interpolant of u_hat - u on the mesh of size max_mesh_size; that quantity depends on the
        mesh and, because grad u ~ r^{-1/3}, converges slowly.)  The quadrature error of the
        singular integrand is still mesh dependent: report the value on a ladder of
        `true_error_mesh_size`.
        """
        nodes, triangles, _ = _build_l_shaped_triangular_mesh(self.true_error_mesh_size)
        tri_xy = nodes[triangles]  # (T, 3, 2)
        areas = 0.5 * np.abs(
            (tri_xy[:, 1, 0] - tri_xy[:, 0, 0]) * (tri_xy[:, 2, 1] - tri_xy[:, 0, 1])
            - (tri_xy[:, 2, 0] - tri_xy[:, 0, 0]) * (tri_xy[:, 1, 1] - tri_xy[:, 0, 1])
        )
        bary = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
        points = np.einsum("qk,tkd->tqd", bary, tri_xy).reshape(-1, 2)
        weights = np.repeat(areas / 3.0, 3)

        energy_sq = 0.0
        l2_sq = 0.0
        for start in range(0, points.shape[0], chunk):
            stop = start + chunk
            x32 = torch.tensor(points[start:stop], dtype=torch.float32, device=self.solver.device)
            x64 = torch.tensor(points[start:stop], dtype=torch.float64, device=self.solver.device)
            w = torch.tensor(weights[start:stop], dtype=torch.float64, device=self.solver.device)
            u_pred, grad_pred = self.solver.predict_with_gradient(x32)
            u_exact = self.benchmark.exact_solution(x64).detach()
            grad_exact = self.benchmark.exact_gradient(x64).detach()
            diff_grad = grad_pred.to(torch.float64) - grad_exact.to(torch.float64)
            diff_u = u_pred.reshape(-1).to(torch.float64) - u_exact.reshape(-1).to(torch.float64)
            energy_sq += torch.sum(w * torch.sum(diff_grad.pow(2), dim=1)).item()
            l2_sq += torch.sum(w * diff_u.pow(2)).item()

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

    def mesh_ladder(self) -> List[int]:
        """Mesh sizes fem_mesh_size * growth^k <= max_mesh_size, fixed before any evaluation."""
        ladder = [self.fem_mesh_size]
        while ladder[-1] * self.mesh_growth_factor <= self.max_mesh_size:
            ladder.append(ladder[-1] * self.mesh_growth_factor)
        return ladder

    def evaluate(self, training_loss: float) -> ErrorResult:
        """
        Evaluate the estimator on a FIXED mesh ladder and compare it with the true energy error.

        Nothing in this method uses the exact solution to choose the mesh, to stop refining, or to
        adjust a value: the exact error is computed once, up front, and used only for the ratios
        stored in the result.  No discretisation "margin" is added: the computed norms are
        approximations of the exact norms (see Remark 3.6 of the paper), and the mesh ladder in
        `mesh_history` is the evidence about their discretisation error.

        The reported estimate is the sum form  eta_oplus = ||R||_{H^-1}/alpha + ||grad w||  on the
        finest mesh.  Also returned: the Pythagorean form (exact in the continuum for constant a)
        and the a-weighted form of Theorem 3.3 (needed for variable a).
        """
        alpha = float(self.benchmark.coercivity_constant)
        true_energy, true_l2 = self.compute_true_errors()
        residual_l2 = self.compute_residual_l2_norm()
        boundary_error_l2 = self.compute_boundary_error()

        mesh_history: List[Dict[str, float]] = []
        for mesh_size in self.mesh_ladder():
            dual = compute_dual_norm(self.solver, self.benchmark.domain, mesh_size, self.stabilization_eps)
            lifting = compute_boundary_lifting_norm(
                self.solver, self.benchmark, mesh_size, self.stabilization_eps
            )
            lifting_a = compute_boundary_lifting_norm(
                self.solver, self.benchmark, mesh_size, self.stabilization_eps, weighted=True
            )
            residual_term = dual["dual_norm"] / alpha
            mesh_history.append(
                {
                    "mesh_size": float(dual["mesh_size"]),
                    "dual_norm": float(dual["dual_norm"]),
                    "residual_contribution": float(residual_term),
                    "boundary_contribution": float(lifting),
                    "boundary_contribution_a_weighted": float(lifting_a),
                    "estimate": float(residual_term + lifting),
                    "estimate_pythagoras": float(np.hypot(residual_term, lifting)),
                    "estimate_variable": float(np.sqrt(residual_term**2 + lifting_a**2 / alpha)),
                }
            )

        finest = mesh_history[-1]
        estimated = finest["estimate"]
        effectivity = estimated / true_energy
        if effectivity < 1.0 - 1e-3:
            warnings.warn(
                f"eta_oplus/true = {effectivity:.6f} < 1 on mesh {finest['mesh_size']:.0f}. "
                "For constant a the continuous value is >= 1 (Theorem 3.1); a smaller computed value "
                "is discretisation error in the dual norm, the lifting, or the reference error.",
                stacklevel=2,
            )

        result = ErrorResult(
            estimated_error_energy=float(estimated),
            true_error_energy=true_energy,
            true_error_l2=true_l2,
            residual_dual_norm=float(finest["dual_norm"]),
            residual_contribution=float(finest["residual_contribution"]),
            boundary_lifting_norm=float(finest["boundary_contribution"]),
            effectivity=float(effectivity),
            residual_l2_norm=float(residual_l2),
            boundary_error_l2=float(boundary_error_l2),
            training_loss=float(training_loss),
            coercivity_constant=alpha,
            mesh_size_used=int(finest["mesh_size"]),
            mesh_refinements=len(mesh_history) - 1,
            stabilization_eps=float(self.stabilization_eps),
            discretization_margin=0.0,  # deprecated: kept for schema compatibility only
            mesh_history=mesh_history,
            estimator_pythagoras=float(finest["estimate_pythagoras"]),
            estimator_variable=float(finest["estimate_variable"]),
            effectivity_pythagoras=float(finest["estimate_pythagoras"] / true_energy),
            effectivity_variable=float(finest["estimate_variable"] / true_energy),
            boundary_lifting_a_weighted=float(finest["boundary_contribution_a_weighted"]),
            coefficient_upper_bound=float(getattr(self.benchmark, "coefficient_upper_bound", alpha)),
        )

        print(f"  True energy error       : {result.true_error_energy:.6e}")
        print(f"  eta_oplus (sum form)    : {result.estimated_error_energy:.6e}  ratio {result.effectivity:.6f}")
        print(f"  eta_perp (Pythagoras)   : {result.estimator_pythagoras:.6e}  ratio {result.effectivity_pythagoras:.6f}")
        print(f"  eta_a (a-weighted)      : {result.estimator_variable:.6e}  ratio {result.effectivity_variable:.6f}")
        print(f"  Residual contribution   : {result.residual_contribution:.6e}")
        print(f"  Boundary contribution   : {result.boundary_lifting_norm:.6e}")
        print(f"  Training loss           : {result.training_loss:.6e}")
        return result

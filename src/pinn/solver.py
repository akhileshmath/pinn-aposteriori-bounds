from typing import Callable, Dict, Optional, Tuple

import torch


class PINNSolver:
    def __init__(
        self,
        network,
        diffusion_coeff: Callable,
        source_term: Callable,
        boundary_condition: Callable,
        domain_sampler: Callable,
        boundary_sampler: Callable,
        device: str = "cpu",
        domain: Optional[str] = None,
    ):
        self.net = network.to(device)
        self.a = diffusion_coeff
        self.f = source_term
        self.g = boundary_condition
        self.domain_sampler = domain_sampler
        self.boundary_sampler = boundary_sampler
        self.device = device
        self.domain = domain  # only needed for the tangential-derivative penalty

    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().requires_grad_(True)
        u = self.net(x)
        grad_u = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        a_value = self.a(x)
        divergence = torch.zeros(x.shape[0], 1, device=self.device)
        for i in range(x.shape[1]):
            flux_i = a_value * grad_u[:, i : i + 1]
            d_flux_i = torch.autograd.grad(
                flux_i,
                x,
                grad_outputs=torch.ones_like(flux_i),
                create_graph=True,
                retain_graph=True,
            )[0][:, i : i + 1]
            divergence = divergence + d_flux_i

        return self.f(x) + divergence

    def compute_tangential_loss(self, x_boundary: torch.Tensor) -> torch.Tensor:
        """mean over boundary points of (d_tau (uhat - g))^2, d_tau the derivative along the edge."""
        from .sampling import boundary_tangent

        if self.domain is None:
            raise ValueError("PINNSolver(domain=...) is required for the tangential penalty.")
        x = x_boundary.clone().requires_grad_(True)
        u = self.net(x)
        grad_u = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]
        xg = x_boundary.detach().clone().requires_grad_(True)
        g_val = self.g(xg)
        if g_val.requires_grad:
            grad_g = torch.autograd.grad(g_val.sum(), xg)[0].detach()
        else:  # g independent of x (e.g. g == 0)
            grad_g = torch.zeros_like(grad_u)
        tau = boundary_tangent(self.domain, x_boundary.detach())
        d_tau = torch.sum((grad_u - grad_g) * tau, dim=1, keepdim=True)
        return torch.mean(d_tau**2)

    def compute_loss_terms(
        self,
        x_residual: torch.Tensor,
        x_boundary: torch.Tensor,
        w_r: float = 1.0,
        w_b: float = 10.0,
        w_t: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        residual = self.compute_pde_residual(x_residual)
        residual_loss = torch.mean(residual**2)

        boundary_pred = self.net(x_boundary)
        boundary_exact = self.g(x_boundary)
        boundary_loss = torch.mean((boundary_pred - boundary_exact) ** 2)

        total = w_r * residual_loss + w_b * boundary_loss
        tangential_loss = None
        if w_t > 0.0:  # skipped entirely (no extra autograd, no RNG) when the penalty is off
            tangential_loss = self.compute_tangential_loss(x_boundary)
            total = total + w_t * tangential_loss
        return {
            "total": total,
            "residual": residual_loss,
            "boundary": boundary_loss,
            "tangential": tangential_loss,
        }

    def compute_total_loss(
        self,
        x_residual: torch.Tensor,
        x_boundary: torch.Tensor,
        w_r: float = 1.0,
        w_b: float = 10.0,
        w_t: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        terms = self.compute_loss_terms(x_residual, x_boundary, w_r, w_b, w_t)
        return terms["total"], terms["residual"], terms["boundary"]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            return self.net(x)

    def predict_with_gradient(self, x: torch.Tensor):
        self.net.eval()
        x = x.clone().requires_grad_(True)
        u = self.net(x)
        grad_u = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=False,
            retain_graph=False,
        )[0]
        return u.detach(), grad_u.detach()

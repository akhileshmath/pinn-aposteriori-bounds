from typing import Callable, Tuple

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
    ):
        self.net = network.to(device)
        self.a = diffusion_coeff
        self.f = source_term
        self.g = boundary_condition
        self.domain_sampler = domain_sampler
        self.boundary_sampler = boundary_sampler
        self.device = device

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

    def compute_total_loss(
        self,
        x_residual: torch.Tensor,
        x_boundary: torch.Tensor,
        w_r: float = 1.0,
        w_b: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self.compute_pde_residual(x_residual)
        residual_loss = torch.mean(residual**2)

        boundary_pred = self.net(x_boundary)
        boundary_exact = self.g(x_boundary)
        boundary_loss = torch.mean((boundary_pred - boundary_exact) ** 2)

        total = w_r * residual_loss + w_b * boundary_loss
        return total, residual_loss, boundary_loss

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

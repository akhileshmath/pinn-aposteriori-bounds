import time
from dataclasses import dataclass
from typing import List

import torch.optim as optim


@dataclass
class TrainingConfig:
    adam_lr: float = 1e-3
    adam_epochs: int = 5000
    lbfgs_epochs: int = 200
    n_collocation: int = 2000
    n_boundary: int = 500
    w_r: float = 1.0
    w_b: float = 10.0
    resample_every: int = 1000
    print_every: int = 1000


class TrainingHistory:
    def __init__(self):
        self.epochs: List[int] = []
        self.total_loss: List[float] = []
        self.residual_loss: List[float] = []
        self.boundary_loss: List[float] = []
        self.time_elapsed: List[float] = []


class Trainer:
    def __init__(self, solver, config: TrainingConfig):
        self.solver = solver
        self.config = config

    def train(self) -> TrainingHistory:
        history = TrainingHistory()
        x_r = self.solver.domain_sampler(self.config.n_collocation).to(self.solver.device)
        x_b = self.solver.boundary_sampler(self.config.n_boundary).to(self.solver.device)

        optimizer = optim.Adam(self.solver.net.parameters(), lr=self.config.adam_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.adam_epochs
        )

        start = time.time()
        self.solver.net.train()

        for epoch in range(1, self.config.adam_epochs + 1):
            if self.config.resample_every > 0 and epoch % self.config.resample_every == 0:
                x_r = self.solver.domain_sampler(self.config.n_collocation).to(self.solver.device)
                x_b = self.solver.boundary_sampler(self.config.n_boundary).to(self.solver.device)

            optimizer.zero_grad()
            total, residual, boundary = self.solver.compute_total_loss(
                x_r, x_b, self.config.w_r, self.config.w_b
            )
            total.backward()
            optimizer.step()
            scheduler.step()

            history.epochs.append(epoch)
            history.total_loss.append(total.item())
            history.residual_loss.append(residual.item())
            history.boundary_loss.append(boundary.item())
            history.time_elapsed.append(time.time() - start)

            if epoch % self.config.print_every == 0:
                print(
                    f"  Adam {epoch:6d} | L={total.item():.3e} | "
                    f"Res={residual.item():.3e} | BC={boundary.item():.3e}"
                )

        if self.config.lbfgs_epochs > 0:
            optimizer_lbfgs = optim.LBFGS(
                self.solver.net.parameters(),
                lr=1.0,
                max_iter=20,
                history_size=50,
                tolerance_grad=1e-9,
                tolerance_change=1e-11,
                line_search_fn="strong_wolfe",
            )

            for step in range(1, self.config.lbfgs_epochs + 1):
                def closure():
                    optimizer_lbfgs.zero_grad()
                    total, _, _ = self.solver.compute_total_loss(
                        x_r, x_b, self.config.w_r, self.config.w_b
                    )
                    total.backward()
                    return total

                optimizer_lbfgs.step(closure)
                total, residual, boundary = self.solver.compute_total_loss(
                    x_r, x_b, self.config.w_r, self.config.w_b
                )
                history.epochs.append(self.config.adam_epochs + step)
                history.total_loss.append(total.item())
                history.residual_loss.append(residual.item())
                history.boundary_loss.append(boundary.item())
                history.time_elapsed.append(time.time() - start)

                if step % max(self.config.lbfgs_epochs // 5, 1) == 0:
                    print(
                        f"  LBFGS {step:4d} | L={total.item():.3e} | "
                        f"Res={residual.item():.3e} | BC={boundary.item():.3e}"
                    )

        print(f"Training complete. Final loss: {history.total_loss[-1]:.3e}")
        return history

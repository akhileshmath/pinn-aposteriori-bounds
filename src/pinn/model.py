import numpy as np
import torch
import torch.nn as nn


class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, sigma: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("B", torch.randn(embed_dim, input_dim) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = 2.0 * np.pi * x @ self.B.T
        return torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)

    @property
    def output_dim(self) -> int:
        return 2 * self.embed_dim


class SinActivation(nn.Module):
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class PINNNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims=None,
        activation: str = "tanh",
        use_fourier: bool = False,
        fourier_dim: int = 64,
        fourier_sigma: float = 1.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]

        activations = {
            "tanh": nn.Tanh(),
            "sin": SinActivation(),
            "swish": nn.SiLU(),
            "gelu": nn.GELU(),
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation '{activation}'")

        self.activation = activations[activation]
        self.use_fourier = use_fourier

        if use_fourier:
            self.fourier = FourierFeatureEmbedding(input_dim, fourier_dim, fourier_sigma)
            current_dim = self.fourier.output_dim
        else:
            current_dim = input_dim

        layers = []
        for width in hidden_dims:
            layers.append(nn.Linear(current_dim, width))
            current_dim = width
        layers.append(nn.Linear(current_dim, 1))
        self.layers = nn.ModuleList(layers)

        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fourier(x) if self.use_fourier else x
        for layer in self.layers[:-1]:
            hidden = self.activation(layer(hidden))
        return self.layers[-1](hidden)

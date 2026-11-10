"""Training-statistic z-score normalization, whitening PCA, and local projection."""

import torch
from torch import nn


class FeatureNodeEmbedder(nn.Module):
    """Project standardized 20-D descriptors and optional local CNN vectors."""

    def __init__(
        self, input_dim: int = 20, output_dim: int = 64, local_dim: int = 0
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.local_dim = local_dim
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("scale", torch.ones(input_dim))
        self.register_buffer("components", torch.eye(input_dim))
        self.register_buffer("eigen_scale", torch.ones(input_dim))
        self.register_buffer("fitted", torch.tensor(False))
        self.feature_projection = nn.Linear(input_dim, output_dim)
        self.local_projection = nn.Linear(local_dim, output_dim) if local_dim else None
        self.norm = nn.LayerNorm(output_dim)

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> "FeatureNodeEmbedder":
        """Fit z-score and PCA whitening using training nodes only."""
        values = torch.as_tensor(features, dtype=torch.float64, device=self.mean.device)
        if values.ndim != 2 or values.shape[1] != self.input_dim or values.shape[0] < 2:
            raise ValueError(
                f"fit expects at least two rows with {self.input_dim} features"
            )
        if not torch.isfinite(values).all():
            raise ValueError("Training feature matrix contains non-finite values")
        mean = values.mean(dim=0)
        scale = values.std(dim=0, unbiased=True).clamp_min(1e-6)
        standardized = (values - mean) / scale
        covariance = standardized.T @ standardized / max(1, values.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(1e-6)
        eigenvectors = eigenvectors[:, order]
        self.mean.copy_(mean.to(self.mean.dtype))
        self.scale.copy_(scale.to(self.scale.dtype))
        self.components.copy_(eigenvectors.to(self.components.dtype))
        self.eigen_scale.copy_(eigenvalues.rsqrt().to(self.eigen_scale.dtype))
        self.fitted.fill_(True)
        return self

    def forward(
        self, features: torch.Tensor, local_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not bool(self.fitted):
            raise RuntimeError("FeatureNodeEmbedder.fit must be called before forward")
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape [M, {self.input_dim}]")
        normalized = (features - self.mean) / self.scale
        whitened = (normalized @ self.components) * self.eigen_scale
        embedding = self.feature_projection(whitened)
        if self.local_projection is not None:
            if local_features is None or local_features.shape != (
                features.shape[0],
                self.local_dim,
            ):
                raise ValueError(
                    f"local_features must have shape [M, {self.local_dim}]"
                )
            embedding = embedding + self.local_projection(local_features)
        elif local_features is not None:
            raise ValueError("This embedder was created without local feature inputs")
        return self.norm(embedding)

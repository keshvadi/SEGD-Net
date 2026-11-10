"""Dynamic multi-scale graph filtration and normalized graph operators."""

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.data import Data


@dataclass
class GraphScale:
    """Weighted graph operators and topology statistics for one filtration scale."""

    radius: float
    affinity: torch.Tensor
    degree: torch.Tensor
    clustering: torch.Tensor
    laplacian: torch.Tensor
    propagation: torch.Tensor
    pyg_data: Data


class MultiScaleGraphBuilder(nn.Module):
    """Build K radius-expanded graphs using current latent features at each step."""

    def __init__(
        self,
        scales: int = 4,
        base_radius: float = 35.0,
        radius_growth: float = 1.6,
        spatial_bandwidth: float = 0.5,
        feature_bandwidth: float = 1.0,
        similarity_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if scales < 1 or base_radius <= 0 or radius_growth <= 1:
            raise ValueError(
                "scales and radius parameters must define an expanding sequence"
            )
        if spatial_bandwidth <= 0 or feature_bandwidth <= 0:
            raise ValueError("bandwidths must be positive")
        self.scales = scales
        self.base_radius = base_radius
        self.radius_growth = radius_growth
        self.spatial_bandwidth = spatial_bandwidth
        self.feature_bandwidth = feature_bandwidth
        self.similarity_threshold = similarity_threshold

    @property
    def radii(self) -> tuple[float, ...]:
        return tuple(
            self.base_radius * self.radius_growth**index for index in range(self.scales)
        )

    def forward(
        self, positions: torch.Tensor, hidden: torch.Tensor, step: int
    ) -> GraphScale:
        """Construct one undirected affinity matrix, degree, clustering, and P."""
        if not 0 <= step < self.scales:
            raise IndexError(f"step must be in [0, {self.scales})")
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape [M, 2]")
        if hidden.ndim != 2 or hidden.shape[0] != positions.shape[0]:
            raise ValueError("hidden must have shape [M, D] matching positions")
        nodes = positions.shape[0]
        if nodes == 0:
            raise ValueError("Cannot build a graph with zero nuclei")
        radius = self.radii[step]
        spatial_distance = torch.cdist(positions, positions, p=2)
        feature_distance = torch.cdist(hidden, hidden, p=2)
        spatial = torch.exp(
            -(spatial_distance**2) / (2.0 * self.spatial_bandwidth**2 * radius**2)
        )
        feature = torch.exp(-(feature_distance**2) / (2.0 * self.feature_bandwidth**2))
        affinity = spatial * feature
        edge_mask = (spatial_distance <= radius) & (
            affinity >= self.similarity_threshold
        )
        edge_mask.fill_diagonal_(False)
        affinity = affinity * edge_mask.to(affinity.dtype)
        affinity = 0.5 * (affinity + affinity.T)
        degree_values = affinity.sum(dim=1)
        degree_matrix = torch.diag(degree_values)

        normalized = affinity / affinity.max().clamp_min(1e-12)
        root = normalized.clamp_min(0).pow(1.0 / 3.0)
        triangle_sum = torch.diagonal(root @ root @ root)
        neighbor_count = edge_mask.sum(dim=1).to(hidden.dtype)
        clustering = (
            2.0
            * triangle_sum
            / (neighbor_count * (neighbor_count - 1.0)).clamp_min(1.0)
        )
        self_looped = affinity + torch.eye(
            nodes, device=hidden.device, dtype=hidden.dtype
        )
        self_degree = self_looped.sum(dim=1).clamp_min(1e-12)
        inv_sqrt = self_degree.rsqrt()
        propagation = inv_sqrt[:, None] * self_looped * inv_sqrt[None, :]
        laplacian = (
            torch.eye(nodes, device=hidden.device, dtype=hidden.dtype) - propagation
        )

        edge_index = edge_mask.nonzero(as_tuple=False).T.contiguous()
        edge_weight = affinity[edge_index[0], edge_index[1]]
        graph_data = Data(
            x=hidden,
            pos=positions,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        return GraphScale(
            radius=radius,
            affinity=affinity,
            degree=degree_matrix,
            clustering=clustering,
            laplacian=laplacian,
            propagation=propagation,
            pyg_data=graph_data,
        )


def filtration_radii(
    scales: int = 4, base_radius: float = 35.0, growth: float = 1.6
) -> tuple[float, ...]:
    """Return radii r_t=r_0 gamma^(t-1); default values are 35, 56, 89.6, 143.36."""
    if scales < 1 or base_radius <= 0 or growth <= 1:
        raise ValueError("invalid filtration parameters")
    return tuple(base_radius * growth**index for index in range(scales))

"""Chebyshev-style graph propagation with gated recurrent scale evolution."""

import torch
from torch import nn
from torch.nn import functional as F

from .temporal_graph import MultiScaleGraphBuilder


class SpatioTemporalGNN(nn.Module):
    """Apply Eq. 10-12 to an individual patch's nuclei and positions."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 2,
        scales: int = 4,
        base_radius: float = 35.0,
        radius_growth: float = 1.6,
        spatial_bandwidth: float = 0.5,
        feature_bandwidth: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.builder = MultiScaleGraphBuilder(
            scales,
            base_radius,
            radius_growth,
            spatial_bandwidth,
            feature_bandwidth,
        )
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.spatial_weights = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(scales)
        )
        self.topology_projection = nn.Linear(2, hidden_dim)
        self.update_input = nn.Linear(hidden_dim, hidden_dim)
        self.update_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reset_input = nn.Linear(hidden_dim, hidden_dim)
        self.reset_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_input = nn.Linear(hidden_dim, hidden_dim)
        self.candidate_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.readout_projection = nn.Linear(hidden_dim, hidden_dim)
        self.readout_query = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.readout_query, std=hidden_dim**-0.5)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        return_details: bool = False,
    ):
        if node_features.shape[0] == 0:
            raise ValueError("SpatioTemporalGNN requires at least one node")
        hidden = self.input_projection(node_features)
        readouts, graph_scales = [], []
        for step in range(self.builder.scales):
            graph = self.builder(positions, hidden, step)
            topology = torch.stack((graph.degree.diagonal(), graph.clustering), dim=-1)
            message = F.elu(
                graph.propagation @ self.spatial_weights[step](hidden)
                + self.topology_projection(topology)
            )
            update = torch.sigmoid(
                self.update_input(message) + self.update_hidden(hidden)
            )
            reset = torch.sigmoid(self.reset_input(message) + self.reset_hidden(hidden))
            candidate = torch.tanh(
                self.candidate_input(message) + self.candidate_hidden(reset * hidden)
            )
            hidden = (1.0 - update) * hidden + update * candidate
            mean_pool = hidden.mean(dim=0)
            max_pool = hidden.max(dim=0).values
            readouts.append(torch.cat((mean_pool, max_pool), dim=-1))
            graph_scales.append(graph)
        means = torch.stack([state[: self.hidden_dim] for state in readouts])
        horizon_scores = torch.tanh(self.readout_projection(means)) @ self.readout_query
        attention = torch.softmax(horizon_scores, dim=0)
        graph_embedding = (attention[:, None] * torch.stack(readouts)).sum(dim=0)
        logits = self.classifier(graph_embedding)
        if return_details:
            return logits, graph_embedding, graph_scales, attention, hidden
        return logits

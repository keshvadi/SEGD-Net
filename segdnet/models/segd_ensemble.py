"""Three-branch SEGD-Net ensemble and adaptive reliability consensus."""

import torch
from torch import nn

from ..features.node_embedder import FeatureNodeEmbedder
from .cnn_branches import DenseNetBranch, ResNetBranch
from .st_gnn_branch import SpatioTemporalGNN


def _probability_statistics(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits, dim=-1)
    confidence = probabilities.max(dim=-1).values
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)
    return confidence, entropy


def adaptive_branch_weights(
    logits: torch.Tensor,
    validation_errors: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Compute pi_m proportional to inverse validation error, confidence, exp(-H)."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [branches, batch, classes]")
    if validation_errors.shape != (logits.shape[0],):
        raise ValueError("validation_errors must contain one value per branch")
    if torch.any(validation_errors < 0) or torch.any(validation_errors > 1):
        raise ValueError("validation error rates must be in [0, 1]")
    confidence, entropy = _probability_statistics(logits)
    validation_reliability = 1.0 / (validation_errors[:, None] + epsilon)
    scores = validation_reliability * confidence * torch.exp(-entropy)
    return scores / scores.sum(dim=0, keepdim=True).clamp_min(1e-12)


class SEGDEnsemble(nn.Module):
    """Spatio-temporal GNN, ResNet-50, and DenseNet-121 co-distillation ensemble."""

    def __init__(
        self,
        num_classes: int = 2,
        graph_input_dim: int = 64,
        raw_feature_dim: int = 20,
        graph_hidden_dim: int = 128,
        scales: int = 4,
        pretrained_cnn: bool = False,
    ) -> None:
        super().__init__()
        self.graph = SpatioTemporalGNN(
            input_dim=graph_input_dim,
            hidden_dim=graph_hidden_dim,
            num_classes=num_classes,
            scales=scales,
        )
        self.node_embedder = FeatureNodeEmbedder(
            input_dim=raw_feature_dim, output_dim=graph_input_dim
        )
        self.resnet = ResNetBranch(num_classes, pretrained_cnn)
        self.densenet = DenseNetBranch(num_classes, pretrained_cnn)
        self.register_buffer("validation_errors", torch.full((3,), 0.5))

    @torch.no_grad()
    def set_validation_errors(self, errors: torch.Tensor) -> None:
        errors = torch.as_tensor(errors, dtype=self.validation_errors.dtype)
        if errors.shape != (3,) or not torch.isfinite(errors).all():
            raise ValueError("errors must be a finite vector of length 3")
        self.validation_errors.copy_(
            errors.clamp(0, 1).to(self.validation_errors.device)
        )

    def forward(
        self,
        images: torch.Tensor,
        graphs: list[tuple[torch.Tensor, torch.Tensor]],
        temperature: float = 2.5,
    ) -> dict[str, torch.Tensor]:
        if len(graphs) != images.shape[0]:
            raise ValueError(
                "one (node_features, positions) graph is required per image"
            )
        graph_logits_list = []
        for features, positions in graphs:
            if features.shape[-1] == self.node_embedder.input_dim:
                features = self.node_embedder(features)
            elif features.shape[-1] != self.graph.input_projection.in_features:
                raise ValueError(
                    "graph node features must be either raw 20-D descriptors or "
                    f"embedded {self.graph.input_projection.in_features}-D vectors"
                )
            graph_logits_list.append(self.graph(features, positions))
        graph_logits = torch.stack(graph_logits_list, dim=0)
        resnet_logits, _ = self.resnet(images)
        densenet_logits, _ = self.densenet(images)
        logits = torch.stack((graph_logits, resnet_logits, densenet_logits), dim=0)
        weights = adaptive_branch_weights(logits, self.validation_errors)
        soft_probabilities = torch.softmax(logits / temperature, dim=-1)
        teacher = (weights[..., None] * soft_probabilities).sum(dim=0)
        return {
            "logits": logits,
            "weights": weights.transpose(0, 1),
            "teacher": teacher,
            "probabilities": torch.softmax(logits, dim=-1),
        }

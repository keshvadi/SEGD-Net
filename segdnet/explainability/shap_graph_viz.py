"""Integrated-gradient approximations to graph-node and feature attributions."""

import numpy as np
import torch


def integrated_gradient_node_saliency(
    model,
    node_features: torch.Tensor,
    positions: torch.Tensor,
    target_class: int,
    steps: int = 32,
) -> dict[str, torch.Tensor]:
    """Integrate gradients from a zero-feature baseline to each node's attributes."""
    if node_features.ndim != 2 or node_features.shape[0] == 0:
        raise ValueError("node_features must be a non-empty [nodes, features] matrix")
    if positions.shape != (node_features.shape[0], 2):
        raise ValueError("positions must have shape [nodes, 2]")
    if steps < 1:
        raise ValueError("steps must be positive")
    baseline = torch.zeros_like(node_features)
    total_gradient = torch.zeros_like(node_features)
    for fraction in torch.linspace(
        1.0 / steps, 1.0, steps, device=node_features.device
    ):
        interpolated = (baseline + fraction * (node_features - baseline)).detach()
        interpolated.requires_grad_(True)
        logits = model(interpolated, positions)
        if logits.ndim != 1 or not 0 <= target_class < logits.shape[0]:
            raise ValueError("target_class is outside the graph model output")
        gradient = torch.autograd.grad(logits[target_class], interpolated)[0]
        total_gradient += gradient
    attributions = (node_features - baseline) * total_gradient / steps
    return {
        "feature_attributions": attributions,
        "node_saliency": attributions.abs().sum(dim=-1),
    }


def plot_node_saliency(
    positions: np.ndarray,
    saliency: np.ndarray,
    image_shape: tuple[int, int],
    axis=None,
):
    """Draw node attribution as a spatial point overlay and return its axis."""
    import matplotlib.pyplot as plt

    if axis is None:
        _, axis = plt.subplots()
    points = np.asarray(positions)
    values = np.asarray(saliency)
    if points.ndim != 2 or points.shape[1] != 2 or values.shape != (len(points),):
        raise ValueError("positions must be [nodes,2] and saliency [nodes]")
    axis.scatter(
        points[:, 1],
        points[:, 0],
        c=values,
        s=18 + 80 * values / max(float(values.max(initial=0)), 1e-12),
        cmap="magma",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.25,
    )
    axis.set_xlim(0, image_shape[1])
    axis.set_ylim(image_shape[0], 0)
    axis.set_axis_off()
    return axis

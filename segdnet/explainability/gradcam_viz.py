"""Grad-CAM maps for convolutional classifier branches."""

import cv2
import numpy as np
import torch


class GradCAM:
    """Compute class-specific Grad-CAM for a chosen convolutional module."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inputs, output):
        self.activations = output

    def _save_gradient(self, _module, _grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0]

    def __call__(
        self, image: torch.Tensor, class_index: int | None = None
    ) -> np.ndarray:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("GradCAM expects one image with shape [1,C,H,W]")
        self.model.zero_grad(set_to_none=True)
        output = self.model(image)
        logits = output[0] if isinstance(output, tuple) else output
        if logits.ndim == 2:
            logits = logits[0]
        selected = int(logits.argmax()) if class_index is None else int(class_index)
        logits[selected].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("target layer did not expose gradients and activations")
        weights = self.gradients.mean(dim=(-2, -1), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = torch.nn.functional.interpolate(
            heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        heatmap = heatmap.detach().cpu().numpy()
        return heatmap / max(float(heatmap.max()), 1e-12)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def overlay_heatmap(
    rgb_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Blend a JET color map into an RGB image."""
    image = np.asarray(rgb_image).astype(np.uint8)
    resized = cv2.resize(
        np.asarray(heatmap, dtype=np.float32), (image.shape[1], image.shape[0])
    )
    colored = cv2.applyColorMap(
        np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)

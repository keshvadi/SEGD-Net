"""Lightweight U-Net/IFCM nuclear segmentation with marker watershed splitting."""

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from torch import nn


class ConvBlock(nn.Module):
    """Residual two-convolution U-Net block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.block(inputs) + self.skip(inputs))


class NuclearUNet(nn.Module):
    """Compact U-Net producing a single-channel nuclear probability map."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.mid = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = self.enc1(inputs)
        second = self.enc2(self.pool(first))
        middle = self.mid(self.pool(second))
        up_second = self.up2(middle)
        up_second = torch.cat((up_second, second), dim=1)
        decoded_second = self.dec2(up_second)
        up_first = self.up1(decoded_second)
        up_first = torch.cat((up_first, first), dim=1)
        return torch.sigmoid(self.out(self.dec1(up_first)))


def intuitionistic_membership(
    intensity: np.ndarray, clusters: int = 2, fuzziness: float = 2.0, max_iter: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute fuzzy memberships, non-memberships, and hesitation per pixel."""
    values = np.asarray(intensity, dtype=np.float64).reshape(-1)
    if clusters < 2 or fuzziness <= 1:
        raise ValueError("clusters must be >=2 and fuzziness must be >1")
    centers = np.percentile(values, np.linspace(10, 90, clusters))
    distances = np.abs(values[:, None] - centers[None, :]) + 1e-8
    memberships = np.full((len(values), clusters), 1.0 / clusters)
    exponent = 2.0 / (fuzziness - 1.0)
    for _ in range(max_iter):
        ratios = (distances[:, :, None] / distances[:, None, :]) ** exponent
        updated = 1.0 / ratios.sum(axis=2)
        weights = updated**fuzziness
        new_centers = (weights * values[:, None]).sum(axis=0) / np.maximum(
            weights.sum(axis=0), 1e-12
        )
        if np.max(np.abs(updated - memberships)) < 1e-5:
            memberships = updated
            centers = new_centers
            break
        memberships, centers = updated, new_centers
        distances = np.abs(values[:, None] - centers[None, :]) + 1e-8
    non_memberships = (1.0 - memberships) / (1.0 + memberships)
    hesitation = np.clip(1.0 - memberships - non_memberships, 0.0, 1.0)
    return memberships, non_memberships, hesitation


def _unet_probability(hematoxylin: np.ndarray, model: NuclearUNet | None) -> np.ndarray:
    normalized = hematoxylin.astype(np.float32) / 255.0
    if model is not None:
        device = next(model.parameters()).device
        tensor = torch.from_numpy(normalized)[None, None].to(device)
        model.eval()
        with torch.no_grad():
            return model(tensor)[0, 0].cpu().numpy()
    smooth = cv2.GaussianBlur(normalized, (0, 0), 1.2)
    threshold = np.percentile(smooth, 58.0)
    return np.clip((smooth - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)


def segment_nuclei(
    image: np.ndarray,
    unet: NuclearUNet | None = None,
    probability_threshold: float = 0.35,
    min_area: int = 50,
) -> np.ndarray:
    """Return integer instance labels after U-Net/IFCM fusion and watershed."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("image must be an HxWx3 RGB array")
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hematoxylin = 255 - gray
    probability = _unet_probability(hematoxylin, unet)
    membership, _, hesitation = intuitionistic_membership(hematoxylin)
    intensity = hematoxylin.astype(np.float64).reshape(-1)
    cluster_means = (membership * intensity[:, None]).sum(axis=0) / np.maximum(
        membership.sum(axis=0), 1e-12
    )
    nuclear_cluster = int(np.argmax(cluster_means))
    fuzzy_score = membership[:, nuclear_cluster] * (1 - hesitation[:, nuclear_cluster])
    fuzzy_score = fuzzy_score.reshape(gray.shape)
    fused = 0.5 * probability + 0.5 * fuzzy_score
    mask = (fused >= probability_threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    distance = ndi.distance_transform_edt(mask)
    coordinates = peak_local_max(
        distance, labels=mask, min_distance=3, exclude_border=False
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    for marker_id, (row, column) in enumerate(coordinates, start=1):
        markers[row, column] = marker_id
    if not markers.any() and mask.any():
        markers, _ = ndi.label(mask)
    instances = watershed(-distance, markers, mask=mask)
    output = np.zeros(instances.shape, dtype=np.int32)
    next_id = 1
    for region_id in np.unique(instances):
        if region_id == 0:
            continue
        region = instances == region_id
        if int(region.sum()) >= min_area:
            output[region] = next_id
            next_id += 1
    return output

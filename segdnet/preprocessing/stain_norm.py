"""Macenko optical-density normalization and hematoxylin-channel CLAHE."""

import cv2
import numpy as np

REFERENCE_STAIN_MATRIX = np.asarray(
    [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]], dtype=np.float64
)


def rgb_to_od(image: np.ndarray) -> np.ndarray:
    """Beer-Lambert optical density conversion, with the specified intensity floor."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("image must be an HxWx3 RGB array")
    return -np.log10(np.maximum(rgb.astype(np.float64), 1.0) / 255.0)


def estimate_stain_matrix(
    od: np.ndarray, tissue_threshold: float = 0.15, angular_percentile: float = 1.0
) -> np.ndarray:
    """Estimate H&E stain vectors from tissue OD vectors using SVD."""
    pixels = od.reshape(-1, 3)
    tissue = pixels[np.linalg.norm(pixels, axis=1) > tissue_threshold]
    if tissue.shape[0] < 3:
        raise ValueError("Insufficient tissue pixels for Macenko stain estimation")
    _, singular_values, right_vectors = np.linalg.svd(tissue, full_matrices=False)
    if singular_values[1] <= np.finfo(float).eps:
        raise ValueError("Tissue OD vectors do not span a stable two-dimensional plane")
    basis = right_vectors[:2].T
    projected = tissue @ basis
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low, high = np.percentile(angles, [angular_percentile, 100.0 - angular_percentile])
    stain_a = basis @ np.asarray([np.cos(low), np.sin(low)])
    stain_b = basis @ np.asarray([np.cos(high), np.sin(high)])
    matrix = np.column_stack((stain_a, stain_b))
    matrix /= np.maximum(np.linalg.norm(matrix, axis=0, keepdims=True), 1e-12)
    if matrix[0, 0] < matrix[0, 1]:
        matrix = matrix[:, ::-1]
    return matrix


def macenko_normalize(
    image: np.ndarray,
    reference_matrix: np.ndarray = REFERENCE_STAIN_MATRIX,
    concentration_percentile: float = 99.0,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Normalize RGB stain concentrations and apply CLAHE to hematoxylin OD."""
    rgb = np.asarray(image)
    od = rgb_to_od(rgb)
    source_matrix = estimate_stain_matrix(od)
    concentrations = np.linalg.lstsq(source_matrix, od.reshape(-1, 3).T, rcond=None)[0]
    source_max = np.percentile(concentrations, concentration_percentile, axis=1)
    if np.any(source_max <= np.finfo(float).eps):
        raise ValueError("Estimated stain concentrations have a zero percentile")
    concentrations *= (np.asarray([1.0, 1.0]) / source_max)[:, None]
    normalized_od = (np.asarray(reference_matrix) @ concentrations).T.reshape(od.shape)
    normalized = np.clip(255.0 * np.power(10.0, -normalized_od), 0, 255).astype(
        np.uint8
    )

    hematoxylin = np.clip(
        255.0 * concentrations[0].reshape(rgb.shape[:2]), 0, 255
    ).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
    enhanced_h = clahe.apply(hematoxylin).astype(np.float64) / 255.0
    h_od = enhanced_h * float(np.percentile(normalized_od[..., 0], 99.0))
    normalized_od[..., 0] = h_od
    normalized = np.clip(255.0 * np.power(10.0, -normalized_od), 0, 255).astype(
        np.uint8
    )
    return normalized

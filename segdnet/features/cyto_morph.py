"""Sixteen explicit geometric descriptors for each segmented nucleus."""

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError

MORPH_FEATURES = (
    "fractal_dimension",
    "lacunarity",
    "cell_area",
    "convex_hull_area",
    "density",
    "cell_perimeter",
    "convex_hull_perimeter",
    "boundary_roughness",
    "convex_span_ratio",
    "nuclear_circularity",
    "hull_circularity",
    "compactness",
    "minimum_enclosing_circle_diameter",
    "maximum_convex_chord",
    "hull_radius_ratio",
    "mean_centroid_radius",
)


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def _box_counts(
    binary: np.ndarray, sizes: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    boundary = _boundary(binary)
    counts = []
    lacunarities = []
    for size in sizes:
        height, width = boundary.shape
        pad_h = (-height) % size
        pad_w = (-width) % size
        padded = np.pad(boundary, ((0, pad_h), (0, pad_w)))
        blocks = padded.reshape(
            padded.shape[0] // size, size, padded.shape[1] // size, size
        )
        occupied = blocks.any(axis=(1, 3))
        counts.append(max(1, int(occupied.sum())))

        mass = ndi.uniform_filter(binary.astype(np.float64), size=size, mode="constant")
        observed_mass = mass[binary]
        mean_mass = float(observed_mass.mean()) if observed_mass.size else 0.0
        variance = float(observed_mass.var()) if observed_mass.size else 0.0
        lacunarities.append(1.0 + variance / max(mean_mass**2, 1e-12))
    return np.asarray(counts, dtype=np.float64), np.asarray(lacunarities)


def _hull_geometry(
    rows_cols: np.ndarray, centroid: np.ndarray
) -> tuple[float, float, float, float, float]:
    try:
        hull = ConvexHull(rows_cols)
    except QhullError:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    vertices = rows_cols[hull.vertices]
    deltas = np.roll(vertices, -1, axis=0) - vertices
    hull_perimeter = float(np.linalg.norm(deltas, axis=1).sum())
    hull_area = float(hull.volume)
    pairwise = vertices[:, None, :] - vertices[None, :, :]
    chord_lengths = np.linalg.norm(pairwise, axis=2)
    max_chord = float(chord_lengths.max(initial=0.0))
    edge_angles = np.arctan2(deltas[:, 0], deltas[:, 1])
    widths = []
    for angle in edge_angles:
        direction = np.asarray([np.cos(angle), np.sin(angle)])
        projections = vertices @ direction
        widths.append(float(projections.max() - projections.min()))
    min_feret = max(min(widths, default=0.0), 1e-12)
    radii = np.linalg.norm(vertices - centroid, axis=1)
    edge_lengths = np.sum(deltas**2, axis=1)
    offsets = centroid - vertices
    segment_positions = np.clip(
        np.sum(offsets * deltas, axis=1) / np.maximum(edge_lengths, 1e-12), 0.0, 1.0
    )
    nearest_points = vertices + segment_positions[:, None] * deltas
    minimum_radius = float(np.linalg.norm(centroid - nearest_points, axis=1).min())
    radius_ratio = float(radii.max() / max(minimum_radius, 1e-12))
    return hull_area, hull_perimeter, max_chord / min_feret, max_chord, radius_ratio


def extract_morph_features(mask: np.ndarray) -> np.ndarray:
    """Return FD, lacunarity, shape ratios, and radial extents in documented order."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("mask must be a non-empty 2-D nucleus mask")
    area = float(binary.sum())
    rows_cols = np.argwhere(binary).astype(np.float64)
    centroid = rows_cols.mean(axis=0)
    contour = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )[0]
    if not contour:
        raise ValueError("Could not extract a contour from the nucleus mask")
    perimeter = float(cv2.arcLength(max(contour, key=cv2.contourArea), True))
    (
        hull_area,
        hull_perimeter,
        span_ratio,
        max_chord,
        hull_radius_ratio,
    ) = _hull_geometry(rows_cols, centroid)
    hull_area = max(hull_area, area)
    hull_perimeter = max(hull_perimeter, perimeter, 1e-12)
    counts, lacunarities = _box_counts(binary, (2, 4, 8, 16, 32))
    log_inverse_size = np.log(1.0 / np.asarray((2, 4, 8, 16, 32), dtype=np.float64))
    fractal_dimension = float(np.polyfit(log_inverse_size, np.log(counts), 1)[0])

    _, enclosing_radius = cv2.minEnclosingCircle(
        rows_cols[:, ::-1].astype(np.float32).reshape(-1, 1, 2)
    )
    distances = np.linalg.norm(rows_cols - centroid, axis=1)
    circularity = 4.0 * np.pi * area / max(perimeter**2, 1e-12)
    hull_circularity = 4.0 * np.pi * hull_area / max(hull_perimeter**2, 1e-12)
    return np.asarray(
        [
            fractal_dimension,
            float(lacunarities.mean()),
            area,
            hull_area,
            area / hull_area,
            perimeter,
            hull_perimeter,
            perimeter / hull_perimeter,
            span_ratio,
            circularity,
            hull_circularity,
            perimeter**2 / area,
            2.0 * float(enclosing_radius),
            max_chord,
            hull_radius_ratio,
            float(distances.mean()),
        ],
        dtype=np.float32,
    )


def extract_feature_matrix(
    instance_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 16 morphology values and centroid positions for each labeled instance."""
    labels = np.asarray(instance_labels)
    if labels.ndim != 2:
        raise ValueError("instance_labels must be a 2-D integer label image")
    features, positions = [], []
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = labels == label
        features.append(extract_morph_features(mask))
        positions.append(np.argwhere(mask).mean(axis=0))
    if not features:
        return np.empty((0, len(MORPH_FEATURES)), np.float32), np.empty(
            (0, 2), np.float32
        )
    return np.stack(features), np.asarray(positions, dtype=np.float32)

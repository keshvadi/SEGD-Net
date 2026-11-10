"""Four rotationally averaged Haralick GLCM descriptors."""

import numpy as np

GLCM_FEATURES = ("contrast", "homogeneity", "energy", "entropy")
GLCM_OFFSETS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))


def extract_glcm_features(
    gray_image: np.ndarray,
    mask: np.ndarray,
    levels: int = 64,
    distance: int = 1,
) -> np.ndarray:
    """Calculate normalized GLCM features inside the tight nuclear bounding box."""
    gray = np.asarray(gray_image)
    binary = np.asarray(mask, dtype=bool)
    if gray.ndim != 2 or binary.shape != gray.shape:
        raise ValueError("gray_image and mask must be 2-D arrays of identical shape")
    if not binary.any() or levels < 2 or distance < 1:
        raise ValueError("mask must be non-empty, levels >=2, and distance >=1")
    y, x = np.where(binary)
    crop = gray[y.min() : y.max() + 1, x.min() : x.max() + 1].astype(np.float64)
    crop_mask = binary[y.min() : y.max() + 1, x.min() : x.max() + 1]
    minimum, maximum = float(crop.min()), float(crop.max())
    if maximum > minimum:
        quantized = np.floor((crop - minimum) * (levels - 1) / (maximum - minimum))
    else:
        quantized = np.zeros_like(crop)
    quantized = np.clip(quantized, 0, levels - 1).astype(np.uint8)
    quantized[~crop_mask] = 0
    matrices = []
    height, width = quantized.shape
    for row_offset, column_offset in GLCM_OFFSETS:
        row_offset *= distance
        column_offset *= distance
        row_start = max(0, -row_offset)
        row_end = min(height, height - row_offset)
        column_start = max(0, -column_offset)
        column_end = min(width, width - column_offset)
        first = quantized[row_start:row_end, column_start:column_end]
        second = quantized[
            row_start + row_offset : row_end + row_offset,
            column_start + column_offset : column_end + column_offset,
        ]
        first_mask = crop_mask[row_start:row_end, column_start:column_end]
        second_mask = crop_mask[
            row_start + row_offset : row_end + row_offset,
            column_start + column_offset : column_end + column_offset,
        ]
        valid = first_mask & second_mask
        matrix = np.zeros((levels, levels), dtype=np.float64)
        if valid.any():
            np.add.at(matrix, (first[valid], second[valid]), 1.0)
            matrix += matrix.T
            matrix /= max(float(matrix.sum()), 1e-12)
        matrices.append(matrix)
    matrix = np.mean(matrices, axis=0)
    matrix /= max(float(matrix.sum()), 1e-12)
    rows, columns = np.indices(matrix.shape)
    difference = (rows - columns).astype(np.float64)
    contrast = float((difference**2 * matrix).sum())
    homogeneity = float((matrix / (1.0 + difference**2)).sum())
    energy = float(np.square(matrix).sum())
    entropy = float(-(matrix * np.log(matrix + 1e-12)).sum())
    return np.asarray([contrast, homogeneity, energy, entropy], dtype=np.float32)


def extract_texture_matrix(
    gray_image: np.ndarray, instance_labels: np.ndarray, levels: int = 64
) -> np.ndarray:
    """Extract four GLCM values per non-background instance label."""
    labels = np.asarray(instance_labels)
    values = [
        extract_glcm_features(gray_image, labels == label, levels=levels)
        for label in np.unique(labels)
        if label != 0
    ]
    return np.stack(values) if values else np.empty((0, 4), dtype=np.float32)


def extract_node_feature_matrix(
    gray_image: np.ndarray, instance_labels: np.ndarray, levels: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Return all 20 descriptors (16 shape + 4 texture) and node centroids."""
    from .cyto_morph import MORPH_FEATURES, extract_feature_matrix

    labels = np.asarray(instance_labels)
    morphology, positions = extract_feature_matrix(labels)
    texture = extract_texture_matrix(gray_image, labels, levels=levels)
    if morphology.shape[0] != texture.shape[0]:
        raise RuntimeError(
            "morphology and texture extraction produced different node counts"
        )
    if morphology.shape[1] != len(MORPH_FEATURES) or texture.shape[1] != len(
        GLCM_FEATURES
    ):
        raise RuntimeError("unexpected feature dimensionality")
    return np.concatenate((morphology, texture), axis=1), positions

#!/usr/bin/env python3
"""Generate a clearly labeled synthetic, non-model-output XAI illustration."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def synthetic_patch(
    malignant: bool, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw H&E-like synthetic tissue, an illustrative saliency field, and node locations."""
    rng = np.random.default_rng(seed)
    height, width = 300, 360
    tissue = np.zeros((height, width, 3), dtype=np.uint8)
    tissue[:] = (242, 207, 220)
    noise = rng.normal(0, 4, tissue.shape).astype(np.int16)
    tissue = np.clip(tissue.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    nodes = []
    count = 70 if not malignant else 150
    for _ in range(count):
        if malignant:
            center = (
                int(rng.integers(10, width - 10)),
                int(rng.integers(10, height - 10)),
            )
            axes = (int(rng.integers(3, 9)), int(rng.integers(3, 11)))
            angle = int(rng.integers(0, 180))
        else:
            center = (
                int(rng.integers(20, width - 20)),
                int(rng.integers(20, height - 20)),
            )
            axes = (int(rng.integers(4, 7)), int(rng.integers(4, 7)))
            angle = 0
        cv2.ellipse(tissue, center, axes, angle, 0, 360, (83, 54, 112), -1)
        nodes.append((center[0], center[1]))
    if not malignant:
        cv2.ellipse(
            tissue, (width // 2, height // 2), (112, 78), 0, 0, 360, (231, 173, 194), 5
        )
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            x = int(width // 2 + 88 * np.cos(angle))
            y = int(height // 2 + 58 * np.sin(angle))
            cv2.circle(tissue, (x, y), 5, (81, 52, 110), -1)
    else:
        cv2.polylines(
            tissue,
            [
                np.asarray(
                    [[20, 220], [95, 175], [170, 190], [240, 115], [340, 70]], np.int32
                )
            ],
            False,
            (118, 76, 134),
            12,
        )
    centers = (
        [(width // 2, height // 2), (width // 2 + 55, height // 2 - 20)]
        if not malignant
        else [(90, 195), (170, 175), (250, 105)]
    )
    saliency = np.zeros((height, width), dtype=np.float32)
    for x, y in centers:
        cv2.circle(saliency, (x, y), 34 if not malignant else 48, 1.0, -1)
    saliency = cv2.GaussianBlur(saliency, (0, 0), 22)
    saliency /= max(float(saliency.max()), 1e-12)
    return tissue, saliency, np.asarray(nodes, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("../images/Fig8.png"))
    args = parser.parse_args()
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row, malignant in enumerate((False, True)):
        patch, heatmap, nodes = synthetic_patch(malignant, seed=41 + row)
        title = "Malignant-like" if malignant else "Benign-like"
        axes[row, 0].imshow(patch)
        axes[row, 0].set_title(f"{title} synthetic H&E patch")
        axes[row, 1].imshow(patch)
        axes[row, 1].imshow(heatmap, cmap="jet", alpha=0.48, vmin=0, vmax=1)
        axes[row, 1].set_title("Illustrative Grad-CAM field")
        node_values = np.linspace(0.1, 1.0, len(nodes))
        points = axes[row, 2].scatter(
            nodes[:, 0],
            nodes[:, 1],
            c=node_values,
            s=13 + 34 * node_values,
            cmap="magma",
            alpha=0.84,
            edgecolors="white",
            linewidths=0.2,
        )
        axes[row, 2].imshow(patch, alpha=0.35)
        axes[row, 2].set_title("Illustrative graph-node saliency")
        for axis in axes[row]:
            axis.set_axis_off()
    figure.suptitle(
        "Synthetic illustration only — not a pathology image or model explanation",
        fontsize=14,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.02, right=0.91, bottom=0.02, top=0.90, hspace=0.22, wspace=0.08
    )
    colorbar_axis = figure.add_axes((0.93, 0.22, 0.015, 0.56))
    figure.colorbar(points, cax=colorbar_axis, label="Illustrative saliency")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote synthetic illustration: {args.output.resolve()}")


if __name__ == "__main__":
    main()

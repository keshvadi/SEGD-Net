"""Synthetic integration checks for preprocessing, graphs, training, and XAI."""

import subprocess
import sys

import cv2
import numpy as np
import torch

from segdnet.evaluation.metrics import classification_metrics
from segdnet.explainability.gradcam_viz import GradCAM
from segdnet.explainability.shap_graph_viz import integrated_gradient_node_saliency
from segdnet.features.glcm_texture import extract_node_feature_matrix
from segdnet.features.node_embedder import FeatureNodeEmbedder
from segdnet.models.segd_ensemble import SEGDEnsemble
from segdnet.models.temporal_graph import MultiScaleGraphBuilder
from segdnet.preprocessing.segmentation import segment_nuclei
from segdnet.preprocessing.stain_norm import macenko_normalize
from segdnet.training.nf_module import NoiseFilteringModule
from segdnet.training.trainer import SEGDTrainer


def _synthetic_patch() -> np.ndarray:
    rng = np.random.default_rng(2026)
    image = np.full((112, 112, 3), (236, 192, 211), dtype=np.uint8)
    image = np.clip(
        image.astype(np.int16) + rng.normal(0, 3, image.shape), 0, 255
    ).astype(np.uint8)
    for row in range(10, 106, 16):
        for column in range(10, 106, 16):
            axes = (int(rng.integers(5, 8)), int(rng.integers(5, 8)))
            cv2.ellipse(
                image,
                (column, row),
                axes,
                int(rng.integers(0, 180)),
                0,
                360,
                (73, 48, 105),
                -1,
            )
    return image


def test_smoke_pipeline(tmp_path):
    patch = _synthetic_patch()
    normalized = macenko_normalize(patch)
    assert normalized.shape == patch.shape
    assert normalized.dtype == np.uint8

    labels = segment_nuclei(normalized)
    assert labels.shape == normalized.shape[:2]
    assert labels.max() > 0
    gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
    raw_features, positions = extract_node_feature_matrix(gray, labels)
    assert raw_features.shape == (int(labels.max()), 20)
    assert positions.shape == (raw_features.shape[0], 2)

    embedder = FeatureNodeEmbedder(20, 64)
    embedder.fit(torch.from_numpy(np.concatenate([raw_features, raw_features], axis=0)))
    embeddings = embedder(torch.from_numpy(raw_features))
    assert embeddings.shape == (raw_features.shape[0], 64)

    graph_builder = MultiScaleGraphBuilder()
    scale_graphs = [
        graph_builder(torch.from_numpy(positions), embeddings, step)
        for step in range(4)
    ]
    assert [round(graph.radius, 2) for graph in scale_graphs] == [
        35.0,
        56.0,
        89.6,
        143.36,
    ]
    assert torch.allclose(
        scale_graphs[-1].propagation, scale_graphs[-1].propagation.T, atol=1e-6
    )

    gate = NoiseFilteringModule()
    selected = gate(
        torch.tensor([[0.96, 0.04], [0.03, 0.97]]),
        torch.tensor([[0.94, 0.06], [0.02, 0.98]]),
    )
    assert selected["selected"].all()

    model = SEGDEnsemble(graph_input_dim=64, graph_hidden_dim=32, scales=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    graphs = [(embeddings.detach(), torch.from_numpy(positions))] * 2
    image_batch = torch.randn(2, 3, 64, 64)
    trainer = SEGDTrainer(
        model,
        optimizer,
        warmup_epochs=2,
        patience=2,
        nf_module=gate,
    )
    labeled_batch = {
        "image": image_batch,
        "label": torch.tensor([0, 1]),
        "graphs": graphs,
    }
    unlabeled_batch = {"image": image_batch, "graphs": graphs}
    validation_batch = {
        "image": image_batch,
        "label": torch.tensor([0, 1]),
        "graphs": graphs,
    }
    history = trainer.fit(
        [labeled_batch],
        [unlabeled_batch],
        [validation_batch],
        epochs=2,
    )
    assert len(history.train_loss) == 2
    assert np.isfinite(history.train_loss).all()

    gradcam = GradCAM(model.resnet, model.resnet.backbone[7])
    heatmap = gradcam(image_batch[:1], class_index=1)
    gradcam.close()
    assert heatmap.shape == (64, 64)
    assert np.isfinite(heatmap).all()

    attribution = integrated_gradient_node_saliency(
        model.graph,
        embeddings.detach(),
        torch.from_numpy(positions),
        target_class=1,
        steps=4,
    )
    assert attribution["node_saliency"].shape == (raw_features.shape[0],)
    assert torch.isfinite(attribution["node_saliency"]).all()

    generated = tmp_path / "Fig8.png"
    script = (
        __import__("pathlib").Path(__file__).parents[1]
        / "scripts"
        / "generate_xai_fig8.py"
    )
    subprocess.run(
        [sys.executable, str(script), "--output", str(generated)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert generated.is_file() and generated.stat().st_size > 10_000
    assert classification_metrics(np.array([0, 1]), np.array([0, 1]))["accuracy"] == 1.0

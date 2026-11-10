#!/usr/bin/env python3
"""Train and evaluate SEGD-Net with patient/slide-grouped five-fold splits."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from segdnet.data.augmentations import HistologyAugment, HistologyEvaluationTransform
from segdnet.data.datasets import (
    HistologyImageDataset,
    build_cross_validation_splits,
    collate_histology,
    discover_images,
)
from segdnet.data.ssl_noise_split import create_ssl_partition
from segdnet.evaluation.metrics import classification_metrics, fold_summary
from segdnet.features.glcm_texture import extract_node_feature_matrix
from segdnet.models.segd_ensemble import SEGDEnsemble
from segdnet.preprocessing.segmentation import segment_nuclei
from segdnet.preprocessing.stain_norm import macenko_normalize
from segdnet.training.nf_module import NoiseFilteringModule
from segdnet.training.trainer import SEGDTrainer

ABLATIONS = {
    "full": (True, True, 4, True),
    "without_cyto_features": (False, True, 4, True),
    "without_temporal_graph": (True, True, 1, True),
    "without_noise_filter": (True, False, 4, True),
    "without_weighted_kd": (True, True, 4, False),
}


class GraphCache:
    """Cache per-image raw morphology/texture features and nuclear centroids."""

    def __init__(self, maximum_side: int = 512, include_cytomorphology: bool = True):
        self.maximum_side = maximum_side
        self.include_cytomorphology = include_cytomorphology
        self.values = {}

    def __call__(self, image: Image.Image, record):
        key = str(record.path)
        if key in self.values:
            return self.values[key]
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        factor = min(1.0, self.maximum_side / max(height, width))
        if factor < 1:
            rgb = cv2.resize(
                rgb,
                (max(1, round(width * factor)), max(1, round(height * factor))),
                interpolation=cv2.INTER_AREA,
            )
        normalized = macenko_normalize(rgb)
        instances = segment_nuclei(normalized)
        gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
        features, positions = extract_node_feature_matrix(gray, instances)
        if not features.shape[0]:
            raise RuntimeError(f"No nuclei >=50 px found in image: {record.path}")
        if not self.include_cytomorphology:
            features[:, :16] = 0.0
        self.values[key] = (features, positions)
        return self.values[key]


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("breakhis", "bach"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-rate", type=float)
    parser.add_argument("--noise-mode", choices=("symmetric", "asymmetric"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _evaluate(model, loader, device):
    model.eval()
    targets, predictions = [], []
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["image"].to(device),
                [
                    (features.to(device), positions.to(device))
                    for features, positions in batch["graphs"]
                ],
            )
            labels = batch["label"].to(device)
            targets.extend(labels.cpu().tolist())
            predictions.extend(
                output["logits"].mean(dim=0).argmax(dim=-1).cpu().tolist()
            )
    return np.asarray(targets), np.asarray(predictions)


def _graph_for_record(cache, record):
    with Image.open(record.path) as source:
        return cache(source.convert("RGB"), record)


def main():
    args = _arguments()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["dataset"] != args.dataset:
        raise ValueError("--dataset must match the selected configuration")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    records = discover_images(args.data_root, args.dataset)
    splits = build_cross_validation_splits(records, args.folds, seed)
    args.output.mkdir(parents=True, exist_ok=True)
    include_cyto, enable_nf, graph_scales, enable_kd = ABLATIONS[args.ablation]
    cache = GraphCache(include_cytomorphology=include_cyto)
    train_transform = HistologyAugment(config["image_size"])
    eval_transform = HistologyEvaluationTransform(config["image_size"])
    noise_rate = config["label_noise"] if args.noise_rate is None else args.noise_rate
    noise_mode = config["noise_mode"] if args.noise_mode is None else args.noise_mode
    fold_results = []

    for fold_number, (train_indices, validation_indices, test_indices) in enumerate(
        splits, start=1
    ):
        train_labels = np.asarray([records[index].label for index in train_indices])
        partition = create_ssl_partition(
            train_labels,
            labeled_fraction=float(config["labeled_fraction"]),
            noise_rate=noise_rate,
            noise_mode=noise_mode,
            seed=seed + fold_number,
        )
        label_overrides = {
            int(train_indices[local]): int(partition.observed_labels[local])
            for local in partition.labeled_indices
        }
        labeled_indices = train_indices[partition.labeled_indices]
        unlabeled_indices = train_indices[partition.unlabeled_indices]

        labeled_dataset = HistologyImageDataset(
            records, train_transform, labeled_indices, cache, label_overrides
        )
        unlabeled_dataset = (
            HistologyImageDataset(records, train_transform, unlabeled_indices, cache)
            if len(unlabeled_indices)
            else None
        )
        validation_dataset = HistologyImageDataset(
            records, eval_transform, validation_indices, cache
        )
        test_dataset = HistologyImageDataset(
            records, eval_transform, test_indices, cache
        )
        loader_options = {
            "batch_size": int(config["batch_size"]),
            "num_workers": 0,
            "collate_fn": collate_histology,
        }
        labeled_loader = DataLoader(labeled_dataset, shuffle=True, **loader_options)
        unlabeled_loader = (
            DataLoader(unlabeled_dataset, shuffle=True, **loader_options)
            if unlabeled_dataset is not None
            else None
        )
        validation_loader = DataLoader(
            validation_dataset, shuffle=False, **loader_options
        )
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

        training_graphs = [
            _graph_for_record(cache, records[i])[0] for i in train_indices
        ]
        model = SEGDEnsemble(
            num_classes=int(config["num_classes"]),
            graph_input_dim=int(config["graph"]["input_dim"]),
            graph_hidden_dim=int(config["graph"]["hidden_dim"]),
            scales=graph_scales,
        )
        model.node_embedder.fit(
            torch.from_numpy(np.concatenate(training_graphs, axis=0))
        )
        device = torch.device(args.device)
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        nf_module = NoiseFilteringModule(
            confidence_threshold=float(config["confidence_threshold"]),
            entropy_threshold=float(config["entropy_threshold"]),
            relabel_threshold=float(config["relabel_threshold"]),
        )
        trainer = SEGDTrainer(
            model,
            optimizer,
            device=device,
            num_classes=int(config["num_classes"]),
            warmup_epochs=int(config["warmup_epochs"]),
            alpha=float(config["alpha"]),
            lambda_kd=float(config["lambda_kd"]) if enable_kd else 0.0,
            temperature=float(config["temperature"]),
            nf_module=nf_module,
            enable_noise_filter=enable_nf,
        )
        history = trainer.fit(
            labeled_loader,
            unlabeled_loader,
            validation_loader,
            epochs=args.epochs or int(config["epochs"]),
        )
        targets, predictions = _evaluate(model, test_loader, device)
        metrics = classification_metrics(targets, predictions)
        fold_results.append(metrics)
        torch.save(
            {
                "model": model.state_dict(),
                "best_epoch": history.best_epoch,
                "metrics": metrics,
                "config": config,
                "ablation": args.ablation,
            },
            args.output / f"fold_{fold_number}.pt",
        )
        print(
            f"fold={fold_number} epoch={history.best_epoch} "
            f"test_accuracy={metrics['accuracy']:.4f}"
        )

    result = {
        "dataset": args.dataset,
        "noise_rate": noise_rate,
        "ablation": args.ablation,
        "fold_metrics": fold_results,
        "summary": fold_summary(fold_results),
    }
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

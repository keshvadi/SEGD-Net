"""Algorithm 1 semi-supervised co-distillation trainer."""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .losses import co_distillation_loss
from .nf_module import NoiseFilteringModule


def linear_warmup_ramp(epoch: int, warmup_epochs: int = 25) -> float:
    """Linearly increase unsupervised loss weight to one during warm-up."""
    if warmup_epochs < 1:
        raise ValueError("warmup_epochs must be positive")
    return min(1.0, max(0.0, (epoch + 1) / warmup_epochs))


def _move_graphs(graphs, device):
    return [
        (features.to(device), positions.to(device)) for features, positions in graphs
    ]


@dataclass
class TrainingHistory:
    train_loss: list[float]
    validation_loss: list[float]
    validation_accuracy: list[float]
    best_epoch: int


class SEGDTrainer:
    """Train and validate a SEGDEnsemble with dual-criterion pseudo-labeling."""

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str = "cpu",
        num_classes: int = 2,
        warmup_epochs: int = 25,
        alpha: float = 0.60,
        lambda_kd: float = 0.40,
        temperature: float = 2.5,
        patience: int = 15,
        nf_module: NoiseFilteringModule | None = None,
        enable_noise_filter: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be positive")
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.lambda_kd = lambda_kd
        self.temperature = temperature
        self.patience = patience
        self.nf = nf_module or NoiseFilteringModule()
        self.enable_noise_filter = enable_noise_filter
        self.teacher_weights = (0.5, 0.5)
        self.model.to(self.device)

    def _forward_batch(self, batch: dict) -> dict:
        images = batch["image"].to(self.device)
        graphs = _move_graphs(batch["graphs"], self.device)
        return self.model(images, graphs, temperature=self.temperature)

    def train_epoch(
        self,
        labeled_loader: Iterable[dict],
        unlabeled_loader: Iterable[dict] | None,
        epoch: int,
    ) -> float:
        self.model.train()
        unlabeled_iterator = (
            iter(unlabeled_loader) if unlabeled_loader is not None else None
        )
        losses = []
        ramp = linear_warmup_ramp(epoch, self.warmup_epochs)
        for labeled_batch in labeled_loader:
            labeled_batch = dict(labeled_batch)
            labeled_output = self._forward_batch(labeled_batch)
            labels = labeled_batch["label"].to(self.device).long()
            teacher_one = labeled_output["probabilities"][0]
            teacher_two = labeled_output["probabilities"][1]
            labeled_filter = self.nf(teacher_one, teacher_two, self.teacher_weights)
            if not self.enable_noise_filter:
                labeled_filter["reliability"] = torch.ones_like(
                    labeled_filter["reliability"]
                )
            labeled_targets, _ = self.nf.refurbish(labels, labeled_filter)
            if not self.enable_noise_filter:
                labeled_targets = F.one_hot(labels, num_classes=self.num_classes).to(
                    labeled_output["logits"].dtype
                )
            labeled_weights = labeled_filter["reliability"].clamp_min(0.05)

            unlabeled_output = None
            pseudo_targets = None
            unlabeled_weights = None
            unlabeled_teacher = None
            if unlabeled_iterator is not None:
                try:
                    unlabeled_batch = next(unlabeled_iterator)
                except StopIteration:
                    unlabeled_iterator = iter(unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iterator)
                unlabeled_output = self._forward_batch(dict(unlabeled_batch))
                gate = self.nf(
                    unlabeled_output["probabilities"][0],
                    unlabeled_output["probabilities"][1],
                    self.teacher_weights,
                )
                if not self.enable_noise_filter:
                    gate["selected"] = torch.ones_like(
                        gate["selected"], dtype=torch.bool
                    )
                    gate["reliability"] = torch.ones_like(gate["reliability"])
                    gate["pseudo_labels"] = (
                        unlabeled_output["logits"].mean(dim=0).argmax(dim=-1)
                    )
                accepted = gate["selected"]
                if accepted.any():
                    pseudo_targets = gate["pseudo_labels"][accepted]
                    unlabeled_weights = gate["reliability"][accepted].clamp_min(0.05)
                    unlabeled_logits = unlabeled_output["logits"][:, accepted]
                    unlabeled_teacher = unlabeled_output["teacher"][accepted]
                else:
                    unlabeled_logits = None
            else:
                unlabeled_logits = None
            objective = co_distillation_loss(
                labeled_output["logits"],
                labeled_targets,
                labeled_weights,
                unlabeled_logits=unlabeled_logits,
                pseudo_targets=pseudo_targets,
                unlabeled_weights=unlabeled_weights,
                alpha=self.alpha,
                lambda_ramp=ramp,
                lambda_kd=self.lambda_kd,
                temperature=self.temperature,
                labeled_teacher=labeled_output["teacher"],
                unlabeled_teacher=unlabeled_teacher,
            )
            self.optimizer.zero_grad(set_to_none=True)
            objective["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            losses.append(float(objective["total"].detach().cpu()))
        if not losses:
            raise ValueError("labeled_loader yielded no batches")
        return float(np.mean(losses))

    @torch.no_grad()
    def validate(
        self, validation_loader: Iterable[dict]
    ) -> tuple[float, float, list[float]]:
        self.model.eval()
        total_loss, total_correct, total_count = 0.0, 0, 0
        errors = torch.zeros(3, dtype=torch.float64)
        confusion = torch.zeros(
            3,
            self.num_classes,
            self.num_classes,
            dtype=torch.float64,
        )
        for batch in validation_loader:
            output = self._forward_batch(dict(batch))
            labels = batch["label"].to(self.device).long()
            logits = output["logits"]
            total_loss += float(F.cross_entropy(logits.mean(dim=0), labels).cpu())
            predictions = logits.argmax(dim=-1)
            errors += (predictions != labels.unsqueeze(0)).sum(dim=1).cpu()
            for branch in range(3):
                counts = torch.bincount(
                    labels * self.num_classes + predictions[branch],
                    minlength=self.num_classes**2,
                ).reshape(self.num_classes, self.num_classes)
                confusion[branch] += counts.cpu()
            total_correct += int((logits.mean(dim=0).argmax(dim=-1) == labels).sum())
            total_count += labels.numel()
        if total_count == 0:
            raise ValueError("validation_loader yielded no samples")
        rates = (errors / total_count).tolist()
        true_positive = confusion.diagonal(dim1=1, dim2=2)
        false_positive = confusion.sum(dim=1) - true_positive
        false_negative = confusion.sum(dim=2) - true_positive
        branch_f1 = (
            2
            * true_positive
            / (2 * true_positive + false_positive + false_negative).clamp_min(1)
        ).mean(dim=1)
        teacher_f1 = branch_f1[:2]
        self.teacher_weights = tuple(
            (teacher_f1 / teacher_f1.sum().clamp_min(1e-12)).tolist()
        )
        self.model.set_validation_errors(torch.tensor(rates, dtype=torch.float32))
        return (
            total_loss / max(1, len(validation_loader)),
            total_correct / total_count,
            rates,
        )

    def fit(
        self,
        labeled_loader: Iterable[dict],
        unlabeled_loader: Iterable[dict] | None,
        validation_loader: Iterable[dict],
        epochs: int = 100,
    ) -> TrainingHistory:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        train_losses, validation_losses, validation_accuracies = [], [], []
        best_state = deepcopy(self.model.state_dict())
        best_loss = float("inf")
        best_epoch, stale_epochs = 0, 0
        for epoch in range(epochs):
            train_losses.append(
                self.train_epoch(labeled_loader, unlabeled_loader, epoch)
            )
            val_loss, val_accuracy, _ = self.validate(validation_loader)
            validation_losses.append(val_loss)
            validation_accuracies.append(val_accuracy)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_state = deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.patience:
                break
        self.model.load_state_dict(best_state)
        return TrainingHistory(
            train_losses, validation_losses, validation_accuracies, best_epoch
        )

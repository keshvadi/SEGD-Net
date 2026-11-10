"""Dual-confidence/entropy pseudo-label filter and noisy-label refurbishment."""

import math

import torch


class NoiseFilteringModule:
    """Consensus and Eq. 13-16 selection logic for two teacher distributions."""

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        entropy_threshold: float = 0.50,
        relabel_threshold: float = 0.90,
        epsilon: float = 1e-12,
    ) -> None:
        if not 0.5 < confidence_threshold < 1:
            raise ValueError("confidence_threshold must be in (0.5, 1)")
        if entropy_threshold <= 0 or relabel_threshold <= confidence_threshold:
            raise ValueError(
                "entropy threshold must be positive and relabel threshold larger"
            )
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.relabel_threshold = relabel_threshold
        self.epsilon = epsilon

    def forward(
        self,
        teacher_one: torch.Tensor,
        teacher_two: torch.Tensor,
        validation_weights: tuple[float, float] = (0.5, 0.5),
    ) -> dict[str, torch.Tensor]:
        if teacher_one.shape != teacher_two.shape or teacher_one.ndim != 2:
            raise ValueError(
                "teacher distributions must have matching [batch, classes] shapes"
            )
        weights = torch.as_tensor(
            validation_weights, dtype=teacher_one.dtype, device=teacher_one.device
        )
        if (
            weights.shape != (2,)
            or torch.any(weights < 0)
            or not torch.isfinite(weights).all()
        ):
            raise ValueError(
                "validation_weights must be two finite non-negative values"
            )
        weights = weights / weights.sum().clamp_min(self.epsilon)
        consensus = weights[0] * teacher_one + weights[1] * teacher_two
        consensus = consensus / consensus.sum(dim=-1, keepdim=True).clamp_min(
            self.epsilon
        )
        confidence, pseudo_labels = consensus.max(dim=-1)
        entropy = -(consensus * torch.log(consensus.clamp_min(self.epsilon))).sum(
            dim=-1
        )
        agreement = teacher_one.argmax(dim=-1) == teacher_two.argmax(dim=-1)
        selected = (
            (confidence >= self.confidence_threshold)
            & (entropy <= self.entropy_threshold)
            & agreement
        )
        classes = consensus.shape[-1]
        reliability = confidence * (1.0 - entropy / math.log(max(classes, 2)))
        return {
            "consensus": consensus,
            "pseudo_labels": pseudo_labels,
            "confidence": confidence,
            "entropy": entropy,
            "agreement": agreement,
            "selected": selected,
            "reliability": reliability.clamp(0, 1),
        }

    __call__ = forward

    def refurbish(
        self,
        observed_labels: torch.Tensor,
        result: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return soft targets, replacing only selected high-confidence conflicts."""
        num_classes = result["consensus"].shape[-1]
        targets = torch.nn.functional.one_hot(
            observed_labels.long(), num_classes=num_classes
        ).to(result["consensus"].dtype)
        replace = (
            result["selected"]
            & (result["pseudo_labels"] != observed_labels)
            & (result["confidence"] > self.relabel_threshold)
        )
        targets[replace] = result["consensus"][replace]
        return targets, replace

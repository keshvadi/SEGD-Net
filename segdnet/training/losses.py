"""Reliability-weighted cross entropy and temperature-scaled co-distillation."""

import math

import torch
from torch.nn import functional as F


def sample_reliability(probabilities: torch.Tensor) -> torch.Tensor:
    """Compute omega=c(1-H/log(C)) from class probabilities."""
    if probabilities.ndim != 2 or probabilities.shape[-1] < 2:
        raise ValueError("probabilities must have shape [batch, at least 2 classes]")
    normalized = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        1e-12
    )
    confidence = normalized.max(dim=-1).values
    entropy = -(normalized * torch.log(normalized.clamp_min(1e-12))).sum(dim=-1)
    return (confidence * (1.0 - entropy / math.log(normalized.shape[-1]))).clamp(0, 1)


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy for hard integer labels or soft probability targets."""
    if targets.ndim == 1:
        per_sample = F.cross_entropy(logits, targets.long(), reduction="none")
    elif targets.shape == logits.shape:
        per_sample = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    else:
        raise ValueError(
            "targets must be integer labels or a logits-shaped probability matrix"
        )
    if sample_weights.shape != per_sample.shape:
        raise ValueError("sample_weights must contain one weight per sample")
    return (per_sample * sample_weights).mean()


def reliability_weighted_kl(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    reliability: torch.Tensor,
    temperature: float = 2.5,
) -> torch.Tensor:
    """Temperature-compensated KL divergence to detached consensus targets."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if logits.ndim != 3 or teacher.shape != logits.shape[1:]:
        raise ValueError(
            "logits must be [branches,batch,classes], teacher [batch,classes]"
        )
    if reliability.shape != (logits.shape[1],):
        raise ValueError("reliability must contain one value per batch sample")
    target = teacher.detach().clamp_min(1e-12)
    target = target / target.sum(dim=-1, keepdim=True)
    log_student = F.log_softmax(logits / temperature, dim=-1)
    per_sample = F.kl_div(
        log_student,
        target.unsqueeze(0).expand_as(log_student),
        reduction="none",
    ).sum(dim=-1)
    return temperature**2 * (per_sample * reliability.unsqueeze(0)).mean()


def co_distillation_loss(
    labeled_logits: torch.Tensor,
    labeled_targets: torch.Tensor,
    labeled_weights: torch.Tensor,
    unlabeled_logits: torch.Tensor | None = None,
    pseudo_targets: torch.Tensor | None = None,
    unlabeled_weights: torch.Tensor | None = None,
    alpha: float = 0.60,
    lambda_ramp: float = 1.0,
    lambda_kd: float = 0.40,
    temperature: float = 2.5,
    labeled_teacher: torch.Tensor | None = None,
    unlabeled_teacher: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Combine Eq. 20-22 objectives across all three heterogeneous branches."""
    if labeled_logits.ndim != 3:
        raise ValueError("labeled_logits must be [branches,batch,classes]")
    if not 0 <= alpha <= 1 or lambda_ramp < 0 or lambda_kd < 0:
        raise ValueError("loss coefficients must be non-negative and alpha in [0,1]")
    branch_losses = []
    labeled_ce_terms = []
    unlabeled_ce_terms = []
    for branch in range(labeled_logits.shape[0]):
        labeled_ce = weighted_cross_entropy(
            labeled_logits[branch], labeled_targets, labeled_weights
        )
        unlabeled_ce = labeled_ce.new_zeros(())
        if unlabeled_logits is not None:
            if pseudo_targets is None or unlabeled_weights is None:
                raise ValueError(
                    "unlabeled targets and weights are required with unlabeled logits"
                )
            unlabeled_ce = weighted_cross_entropy(
                unlabeled_logits[branch], pseudo_targets, unlabeled_weights
            )
        labeled_ce_terms.append(labeled_ce)
        unlabeled_ce_terms.append(unlabeled_ce)
        branch_losses.append(
            alpha * labeled_ce + (1 - alpha) * lambda_ramp * unlabeled_ce
        )

    all_logits = (
        labeled_logits
        if unlabeled_logits is None
        else torch.cat((labeled_logits, unlabeled_logits), dim=1)
    )
    if labeled_teacher is None:
        labeled_teacher = torch.softmax(
            labeled_logits.detach() / temperature, dim=-1
        ).mean(0)
    else:
        labeled_teacher = labeled_teacher.detach()
    if unlabeled_logits is None:
        teacher = labeled_teacher
        reliability = labeled_weights
    else:
        if unlabeled_teacher is None:
            unlabeled_teacher = torch.softmax(
                unlabeled_logits.detach() / temperature, dim=-1
            ).mean(0)
        else:
            unlabeled_teacher = unlabeled_teacher.detach()
        teacher = torch.cat((labeled_teacher, unlabeled_teacher), dim=0)
        reliability = torch.cat((labeled_weights, unlabeled_weights), dim=0)
    kd = reliability_weighted_kl(all_logits, teacher, reliability, temperature)
    supervised = torch.stack(branch_losses).sum()
    total = supervised + lambda_kd * kd
    return {
        "total": total,
        "labeled_ce": torch.stack(labeled_ce_terms).mean(),
        "unlabeled_ce": torch.stack(unlabeled_ce_terms).mean(),
        "kl": kd,
    }

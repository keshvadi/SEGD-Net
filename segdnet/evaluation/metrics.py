"""Binary and multiclass classification metrics with fold summaries."""

import numpy as np


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray, positive_label: int = 1
) -> dict[str, float]:
    """Compute accuracy, sensitivity, specificity, precision, and F1."""
    truth = np.asarray(targets)
    predicted = np.asarray(predictions)
    if truth.ndim != 1 or predicted.shape != truth.shape or truth.size == 0:
        raise ValueError("targets and predictions must be non-empty matching vectors")
    classes = np.unique(np.concatenate((truth, predicted)))
    if classes.size == 2:
        positive = truth == positive_label
        predicted_positive = predicted == positive_label
        tp = int(np.sum(positive & predicted_positive))
        tn = int(np.sum(~positive & ~predicted_positive))
        fp = int(np.sum(~positive & predicted_positive))
        fn = int(np.sum(positive & ~predicted_positive))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
        return {
            "accuracy": float(np.mean(truth == predicted)),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
        }

    recalls, precisions, f1s = [], [], []
    for label in classes:
        tp = np.sum((truth == label) & (predicted == label))
        fp = np.sum((truth != label) & (predicted == label))
        fn = np.sum((truth == label) & (predicted != label))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
    return {
        "accuracy": float(np.mean(truth == predicted)),
        "sensitivity": float(np.mean(recalls)),
        "specificity": float("nan"),
        "precision": float(np.mean(precisions)),
        "f1": float(np.mean(f1s)),
    }


def fold_summary(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Calculate unweighted mean and sample SD for each metric over folds."""
    if not fold_metrics:
        raise ValueError("fold_metrics must contain at least one fold")
    keys = set(fold_metrics[0])
    if any(set(fold) != keys for fold in fold_metrics):
        raise ValueError("all fold metric dictionaries must have identical keys")
    summary = {}
    for key in sorted(keys):
        values = np.asarray([fold[key] for fold in fold_metrics], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary

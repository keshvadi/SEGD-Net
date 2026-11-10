"""Run the five manuscript ablation variants using a caller-supplied evaluator."""

from collections.abc import Callable

ABLATIONS = {
    "full": {
        "cyto_features": True,
        "temporal_graph": True,
        "noise_filter": True,
        "weighted_kd": True,
    },
    "without_cyto_features": {
        "cyto_features": False,
        "temporal_graph": True,
        "noise_filter": True,
        "weighted_kd": True,
    },
    "without_temporal_graph": {
        "cyto_features": True,
        "temporal_graph": False,
        "noise_filter": True,
        "weighted_kd": True,
    },
    "without_noise_filter": {
        "cyto_features": True,
        "temporal_graph": True,
        "noise_filter": False,
        "weighted_kd": True,
    },
    "without_weighted_kd": {
        "cyto_features": True,
        "temporal_graph": True,
        "noise_filter": True,
        "weighted_kd": False,
    },
}


def run_ablations(
    evaluator: Callable[[str, dict[str, bool]], dict], names: list[str] | None = None
) -> dict[str, dict]:
    """Invoke evaluator for each ablation and return its measurements by name."""
    selected = list(ABLATIONS) if names is None else names
    unknown = set(selected) - set(ABLATIONS)
    if unknown:
        raise ValueError(f"Unknown ablation(s): {', '.join(sorted(unknown))}")
    return {name: evaluator(name, ABLATIONS[name].copy()) for name in selected}

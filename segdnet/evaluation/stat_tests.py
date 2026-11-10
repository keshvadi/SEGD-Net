"""Paired two-sided t-test and Wilcoxon signed-rank comparison."""

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def paired_significance(baseline: np.ndarray, proposed: np.ndarray) -> dict[str, float]:
    """Return two-sided paired t-test and Wilcoxon p-values for matched folds."""
    baseline = np.asarray(baseline, dtype=np.float64)
    proposed = np.asarray(proposed, dtype=np.float64)
    if baseline.ndim != 1 or proposed.shape != baseline.shape or baseline.size < 2:
        raise ValueError(
            "paired results must be matching vectors with at least two folds"
        )
    if not np.isfinite(baseline).all() or not np.isfinite(proposed).all():
        raise ValueError("paired results must be finite")
    t_result = ttest_rel(proposed, baseline, alternative="two-sided")
    difference = proposed - baseline
    wilcoxon_p = (
        1.0
        if np.allclose(difference, 0)
        else float(wilcoxon(proposed, baseline, alternative="two-sided").pvalue)
    )
    return {
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_pvalue": float(t_result.pvalue),
        "wilcoxon_statistic": (
            0.0
            if np.allclose(difference, 0)
            else float(wilcoxon(proposed, baseline).statistic)
        ),
        "wilcoxon_pvalue": wilcoxon_p,
    }

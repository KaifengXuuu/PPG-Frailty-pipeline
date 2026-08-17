"""参与者指标的 canonical facade / Canonical participant-metrics facade.

中文：BA、macro/per-class P/R/F1、worst class、Brier、ECE、confusion 与
coverage 全部委托 training.evaluator，避免两套公式随时间漂移。
English: Every metric delegates to training.evaluator so no duplicate formula drifts.
"""

from __future__ import annotations

import numpy as np

from ..training.evaluator import (
    EvaluationMetrics,
    PairedDeltaSummary,
    PerClassMetrics,
    RepeatMetricSummary,
    evaluate_predictions,
    paired_fold_seed_deltas,
    summarize_repeat_metric,
)


# English/中文：Compatibility type name is an alias to the sole metrics container.
ParticipantMetrics = EvaluationMetrics


def evaluate_participant_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    retained_mask: np.ndarray | None = None,
    class_names: tuple[str, ...] = ("Pre-Frail", "Robust/Non-Frail", "Young"),
    ece_bins: int = 10,
) -> EvaluationMetrics:
    """适配旧签名并委托唯一公式 / Adapt the old signature to the sole formulas."""

    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must declare at least two unique classes")
    return evaluate_predictions(
        np.asarray(labels),
        np.asarray(probabilities),
        class_order=tuple(range(len(class_names))),
        retained_mask=retained_mask,
        ece_bins=ece_bins,
    )


__all__ = [
    "EvaluationMetrics",
    "PairedDeltaSummary",
    "ParticipantMetrics",
    "PerClassMetrics",
    "RepeatMetricSummary",
    "evaluate_participant_probabilities",
    "evaluate_predictions",
    "paired_fold_seed_deltas",
    "summarize_repeat_metric",
]

"""一对一事件匹配 / One-to-one event matching for heartbeat benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventMatchMetrics:
    """事件匹配量化结果 / Quantitative event-matching result."""

    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    timing_mae_s: float | None


def match_events(reference_s: np.ndarray, predicted_s: np.ndarray, *, tolerance_s: float) -> EventMatchMetrics:
    """按时间排序贪心一对一匹配 / Greedy chronological one-to-one matching."""

    reference = np.sort(np.asarray(reference_s, dtype=np.float64))
    predicted = np.sort(np.asarray(predicted_s, dtype=np.float64))
    if reference.ndim != 1 or predicted.ndim != 1 or tolerance_s <= 0.0:
        raise ValueError("events must be one-dimensional and tolerance_s positive")
    if not np.isfinite(reference).all() or not np.isfinite(predicted).all():
        raise ValueError("event timestamps must be finite")
    used = np.zeros(predicted.size, dtype=bool)
    errors: list[float] = []
    for event in reference:
        candidates = np.flatnonzero((~used) & (np.abs(predicted - event) <= tolerance_s))
        if candidates.size:
            chosen = int(candidates[np.argmin(np.abs(predicted[candidates] - event))])
            used[chosen] = True
            errors.append(abs(float(predicted[chosen] - event)))
    tp = len(errors)
    fp = int(predicted.size - tp)
    fn = int(reference.size - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-15)
    return EventMatchMetrics(tp, fp, fn, float(precision), float(recall), float(f1), float(np.mean(errors)) if errors else None)


__all__ = ["EventMatchMetrics", "match_events"]

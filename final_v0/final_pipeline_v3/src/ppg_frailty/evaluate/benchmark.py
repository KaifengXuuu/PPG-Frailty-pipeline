"""Repeat 汇总与配对差值 / Repeat summaries and paired deltas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.stats import t


@dataclass(frozen=True)
class PairedMetricDelta:
    """相同 fold/seed 单位的配对差 / Paired identical-unit delta."""

    keys: tuple[str, ...]
    deltas: tuple[float, ...]
    mean_delta: float
    sd_delta: float
    ci95: tuple[float, float]


def _summary(values: np.ndarray) -> dict[str, float | list[float]]:
    """均值、样本 SD 与 t-CI / Mean, sample SD, and t confidence interval."""

    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    half = float(t.ppf(0.975, values.size - 1) * sd / np.sqrt(values.size)) if values.size > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95": [mean - half, mean + half], "n": int(values.size)}


def summarize_repeats(metrics_by_repeat: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float | list[float]]]:
    """逐指标汇总完整 repeat / Summarise complete repeats per metric."""

    if not metrics_by_repeat:
        raise ValueError("at least one repeat metric row is required")
    names = set(next(iter(metrics_by_repeat.values())))
    if any(set(row) != names for row in metrics_by_repeat.values()):
        raise ValueError("repeat metric schemas differ")
    return {
        name: _summary(np.asarray([metrics_by_repeat[key][name] for key in sorted(metrics_by_repeat)], dtype=np.float64))
        for name in sorted(names)
    }


def paired_metric_delta(reference: Mapping[str, float], candidate: Mapping[str, float]) -> PairedMetricDelta:
    """要求完全相同 fold/seed key / Require identical fold/seed keys."""

    if set(reference) != set(candidate) or not reference:
        raise ValueError("paired deltas require identical non-empty keys")
    keys = tuple(sorted(reference))
    deltas = np.asarray([candidate[key] - reference[key] for key in keys], dtype=np.float64)
    if not np.isfinite(deltas).all():
        raise ValueError("paired deltas must be finite")
    summary = _summary(deltas)
    return PairedMetricDelta(keys, tuple(float(value) for value in deltas), float(summary["mean"]), float(summary["sd"]), tuple(float(value) for value in summary["ci95"]))


__all__ = ["PairedMetricDelta", "paired_metric_delta", "summarize_repeats"]

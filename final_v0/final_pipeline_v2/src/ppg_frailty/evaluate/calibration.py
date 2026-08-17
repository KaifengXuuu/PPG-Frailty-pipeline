"""仅 outer-train 可拟合的温度校准 / Outer-train-only temperature calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ..provenance import assert_training_only


@dataclass(frozen=True)
class TemperatureCalibrator:
    """一个正温度及拟合成员 / One positive temperature and fit membership."""

    temperature: float
    fitted_on_participant_ids: tuple[str, ...]

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """转换为校准概率 / Convert logits to calibrated probabilities."""

        values = np.asarray(logits, dtype=np.float64) / self.temperature
        values -= logsumexp(values, axis=1, keepdims=True)
        return np.exp(values)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    participant_ids: tuple[str, ...],
    *,
    outer_train_participant_ids: tuple[str, ...],
    outer_oof_participant_ids: tuple[str, ...],
) -> TemperatureCalibrator:
    """只用声明 train participant 拟合 / Fit only declared training participants."""

    fitted = assert_training_only(participant_ids, outer_train_participant_ids, outer_oof_participant_ids)
    values = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != y.size or values.shape[1] != 3:
        raise ValueError("temperature fit requires logits [sample,3] and labels")

    def loss(log_temperature: float) -> float:
        scaled = values / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(y.size), y]))

    result = minimize_scalar(loss, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError("temperature optimization failed")
    return TemperatureCalibrator(float(np.exp(result.x)), fitted)


__all__ = ["TemperatureCalibrator", "fit_temperature"]

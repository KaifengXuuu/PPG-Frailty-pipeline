"""Recording 级 fail-closed QC / Fail-closed recording-level quality control.

中文：QC 只决定 recording 是否可安全进入后续预处理，不等同于 SQI 或分类标签。
阈值必须由调用方显式提供；本模块不以零替换整条缺失通道，也不吞掉解析错误。

English: QC decides whether a recording can safely enter preprocessing; it is not
SQI and never a class label. Callers provide every threshold explicitly. Entirely
missing channels are never replaced with zeros and parse failures remain visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .schema import CANONICAL_CHANNEL_SCHEMA, QCReason, QCStatus


@dataclass(frozen=True)
class QCThresholds:
    """全部显式 recording QC 阈值 / Fully explicit recording QC thresholds."""

    minimum_duration_s: float
    maximum_nonfinite_gap_s: float
    flatline_std_floor_by_channel: Mapping[str, float]
    robust_span_floor_by_channel: Mapping[str, float]
    absolute_limit_by_channel: Mapping[str, float]
    saturation_fraction_limit: float
    timestamp_relative_tolerance: float
    timestamps_required: bool

    def validate(self) -> None:
        """拒绝不完整或无效阈值 / Reject incomplete or invalid thresholds."""

        expected = set(CANONICAL_CHANNEL_SCHEMA)
        mappings = (
            self.flatline_std_floor_by_channel,
            self.robust_span_floor_by_channel,
            self.absolute_limit_by_channel,
        )
        if any(set(mapping) != expected for mapping in mappings):
            raise ValueError("QC channel thresholds must cover exact channel schema")
        if self.minimum_duration_s < 0.0 or self.maximum_nonfinite_gap_s < 0.0:
            raise ValueError("duration/gap thresholds cannot be negative")
        if not 0.0 <= self.saturation_fraction_limit <= 1.0:
            raise ValueError("saturation_fraction_limit must be in [0,1]")
        if self.timestamp_relative_tolerance < 0.0:
            raise ValueError("timestamp tolerance cannot be negative")


@dataclass(frozen=True)
class QCAssessment:
    """可序列化 recording QC 结果 / Serializable recording QC result."""

    status: QCStatus
    reasons: tuple[str, ...]
    metrics: dict[str, float | int | str | bool | None]


def _longest_true_run(mask: np.ndarray) -> int:
    """计算最长连续缺失段 / Return the longest consecutive true run."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0 or not bool(np.any(values)):
        return 0
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return int(np.max(stops - starts))


def parse_failure_assessment(message: str) -> QCAssessment:
    """显式表示解析失败 / Represent a parse failure without dropping it."""

    return QCAssessment(
        status=QCStatus.FAIL,
        reasons=(QCReason.PARSE_FAILURE.value,),
        metrics={"parse_error": str(message)},
    )


def assess_numeric_record(
    values: np.ndarray,
    channel_names: Sequence[str],
    *,
    fs: float,
    thresholds: QCThresholds,
    timestamps_s: np.ndarray | None,
) -> QCAssessment:
    """对已解析数组执行 recording QC / Assess one parsed numeric recording."""

    thresholds.validate()
    reasons: set[str] = set()
    metrics: dict[str, float | int | str | bool | None] = {}
    matrix = np.asarray(values, dtype=np.float64)
    channels = tuple(str(value) for value in channel_names)
    if matrix.ndim != 2:
        return QCAssessment(
            QCStatus.FAIL,
            (QCReason.PARSE_FAILURE.value,),
            {"observed_ndim": int(matrix.ndim)},
        )
    if channels != CANONICAL_CHANNEL_SCHEMA or matrix.shape[1] != len(channels):
        reasons.add(QCReason.MISSING_REQUIRED_CHANNEL.value)
    if fs <= 0.0:
        reasons.add(QCReason.TIMESTAMP_FAILURE.value)
        duration_s = 0.0
    else:
        duration_s = float(matrix.shape[0] / fs)
    metrics.update({"n_samples": int(matrix.shape[0]), "duration_s": duration_s})
    if duration_s < thresholds.minimum_duration_s:
        reasons.add(QCReason.INSUFFICIENT_DURATION.value)

    # 中文：列数错误时仍审计现有列，但绝不伪造缺失列。
    # English: Audit present columns after a shape error, but never fabricate channels.
    for index, channel in enumerate(channels[: matrix.shape[1]]):
        if channel not in CANONICAL_CHANNEL_SCHEMA:
            # 中文：未知列已经由 missing_required_channel 表示；不得因查询一个
            # 不存在的阈值 key 而中断整条 recording 的可见 QC 结果。
            # English: The schema error is already explicit; do not abort the
            # visible recording-level result by indexing an unknown threshold key.
            continue
        column = matrix[:, index]
        finite = np.isfinite(column)
        finite_count = int(np.sum(finite))
        metrics[f"{channel}_finite_fraction"] = (
            float(finite_count / max(1, column.size))
        )
        if finite_count == 0:
            reasons.add(QCReason.ALL_NONFINITE_CHANNEL.value)
            continue
        gap = _longest_true_run(~finite)
        metrics[f"{channel}_max_nonfinite_gap_samples"] = gap
        if fs > 0.0 and gap / fs > thresholds.maximum_nonfinite_gap_s:
            reasons.add(QCReason.EXCESSIVE_NONFINITE_GAP.value)
        observed = column[finite]
        std = float(np.std(observed, ddof=0))
        low, high = np.percentile(observed, [1.0, 99.0])
        robust_span = float(high - low)
        max_abs = float(np.max(np.abs(observed)))
        metrics[f"{channel}_std"] = std
        metrics[f"{channel}_robust_span"] = robust_span
        metrics[f"{channel}_max_abs"] = max_abs
        if (
            std <= float(thresholds.flatline_std_floor_by_channel[channel])
            or robust_span
            <= float(thresholds.robust_span_floor_by_channel[channel])
        ):
            reasons.add(QCReason.FLATLINE.value)
        if max_abs > float(thresholds.absolute_limit_by_channel[channel]):
            reasons.add(QCReason.IMPLAUSIBLE_SCALE.value)
            reasons.add(QCReason.CLIPPING.value)
        minimum = float(np.min(observed))
        maximum = float(np.max(observed))
        extreme_fraction = float(
            np.mean((observed == minimum) | (observed == maximum))
        )
        metrics[f"{channel}_extreme_fraction"] = extreme_fraction
        if extreme_fraction > thresholds.saturation_fraction_limit:
            reasons.add(QCReason.SATURATION.value)

    if timestamps_s is None:
        metrics["timestamps_present"] = False
        if thresholds.timestamps_required:
            reasons.add(QCReason.TIMESTAMP_FAILURE.value)
    else:
        timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
        metrics["timestamps_present"] = True
        if timestamps.size != matrix.shape[0] or not np.isfinite(timestamps).all():
            reasons.add(QCReason.TIMESTAMP_FAILURE.value)
        elif timestamps.size >= 2:
            differences = np.diff(timestamps)
            expected = 1.0 / fs if fs > 0.0 else np.nan
            if (
                np.any(differences <= 0.0)
                or not np.isfinite(expected)
                or np.max(np.abs(differences - expected))
                > thresholds.timestamp_relative_tolerance * expected
            ):
                reasons.add(QCReason.TIMESTAMP_FAILURE.value)
                reasons.add(QCReason.SYNCHRONY_FAILURE.value)

    ordered = tuple(sorted(reasons))
    return QCAssessment(
        status=QCStatus.FAIL if ordered else QCStatus.PASS,
        reasons=ordered,
        metrics=metrics,
    )


__all__ = [
    "QCAssessment",
    "QCThresholds",
    "assess_numeric_record",
    "parse_failure_assessment",
]

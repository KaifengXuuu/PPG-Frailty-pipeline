"""Recording 级 fail-closed QC / Fail-closed recording-level quality control.

中文：QC 只决定 recording 是否可安全进入后续预处理，不等同于 SQI 或分类标签。
阈值必须由调用方显式提供；本模块不以零替换整条缺失通道，也不吞掉解析错误。

English: QC decides whether a recording can safely enter preprocessing; it is not
SQI and never a class label. Callers provide every threshold explicitly. Entirely
missing channels are never replaced with zeros and parse failures remain visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import ManifestRow
from .schema import CANONICAL_CHANNEL_SCHEMA, QCReason, QCStatus


@dataclass(frozen=True)
class QCThresholds:
    """全部显式 recording QC 阈值 / Fully explicit recording QC thresholds."""

    minimum_duration_s: float
    maximum_nonfinite_gap_s: float
    flatline_std_floor_by_channel: Mapping[str, float]
    robust_span_floor_by_channel: Mapping[str, float]
    absolute_limit_by_channel: Mapping[str, float | None]
    saturation_fraction_limit: float | None
    timestamp_relative_tolerance: float | None
    timestamps_required: bool
    device_limits_verified: bool = False

    def validate(self, *, timestamps_present: bool | None = None) -> None:
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
        absolute_limits = tuple(self.absolute_limit_by_channel.values())
        if self.device_limits_verified:
            if any(value is None or not np.isfinite(float(value)) or float(value) <= 0.0 for value in absolute_limits):
                raise ValueError("verified device limits must be finite positive values")
            if (self.saturation_fraction_limit is None or not np.isfinite(float(self.saturation_fraction_limit))
                    or not 0.0 <= float(self.saturation_fraction_limit) <= 1.0):
                raise ValueError("verified saturation limit must be finite in [0,1]")
        elif any(value is not None for value in absolute_limits):
            raise ValueError("deferred device limits must use explicit None for every channel")
        elif self.saturation_fraction_limit is not None:
            raise ValueError("deferred saturation limit must be explicit None")
        needs_timestamp_tolerance = self.timestamps_required or timestamps_present is True
        if needs_timestamp_tolerance:
            if (self.timestamp_relative_tolerance is None or not np.isfinite(float(self.timestamp_relative_tolerance))
                    or float(self.timestamp_relative_tolerance) < 0.0):
                raise ValueError("timestamped QC requires a finite nonnegative tolerance")
        elif self.timestamp_relative_tolerance is not None:
            raise ValueError("non-timestamped QC tolerance must be explicit None")


@dataclass(frozen=True)
class QCAssessment:
    """可序列化 recording QC 结果 / Serializable recording QC result."""

    status: QCStatus
    reasons: tuple[str, ...]
    metrics: dict[str, float | int | str | bool | None]


@dataclass(frozen=True)
class RecordingQCAdmission:
    """Manifest-bound admission result for the formal recording loader."""

    record_id: str
    admitted: bool
    assessment: QCAssessment
    evidence: Mapping[str, Any]

    def validate(self) -> None:
        if not self.record_id:
            raise ValueError("recording QC admission requires record_id")
        if self.admitted != (self.assessment.status is QCStatus.PASS):
            raise ValueError("recording QC admission/status mismatch")
        if self.evidence.get("record_id") != self.record_id:
            raise ValueError("recording QC evidence identity drift")
        if bool(self.evidence.get("admitted")) != self.admitted:
            raise ValueError("recording QC evidence admission drift")


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
        reasons=(QCReason.PARSE_FAILURE.value, ),
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

    thresholds.validate(timestamps_present=timestamps_s is not None)
    reasons: set[str] = set()
    metrics: dict[str, float | int | str | bool | None] = {
        "device_dependent_qc_status":
        ("enabled_verified_device_limits" if thresholds.device_limits_verified else "deferred_missing_device_metadata")
    }
    matrix = np.asarray(values, dtype=np.float64)
    channels = tuple(str(value) for value in channel_names)
    if matrix.ndim != 2:
        return QCAssessment(
            QCStatus.FAIL,
            (QCReason.PARSE_FAILURE.value, ),
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
    for index, channel in enumerate(channels[:matrix.shape[1]]):
        if channel not in CANONICAL_CHANNEL_SCHEMA:
            # 中文：未知列已经由 missing_required_channel 表示；不得因查询一个
            # 不存在的阈值 key 而中断整条 recording 的可见 QC 结果。
            # English: The schema error is already explicit; do not abort the
            # visible recording-level result by indexing an unknown threshold key.
            continue
        column = matrix[:, index]
        finite = np.isfinite(column)
        finite_count = int(np.sum(finite))
        metrics[f"{channel}_finite_fraction"] = float(finite_count / max(1, column.size))
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
        if std <= float(thresholds.flatline_std_floor_by_channel[channel]) or robust_span <= float(
                thresholds.robust_span_floor_by_channel[channel]):
            reasons.add(QCReason.FLATLINE.value)
        if thresholds.device_limits_verified and max_abs > float(thresholds.absolute_limit_by_channel[channel]):
            reasons.add(QCReason.IMPLAUSIBLE_SCALE.value)
            reasons.add(QCReason.CLIPPING.value)
        minimum = float(np.min(observed))
        maximum = float(np.max(observed))
        extreme_fraction = float(np.mean((observed == minimum) | (observed == maximum)))
        metrics[f"{channel}_extreme_fraction"] = extreme_fraction
        if (thresholds.device_limits_verified and thresholds.saturation_fraction_limit is not None
                and extreme_fraction > thresholds.saturation_fraction_limit):
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
            if (np.any(differences <= 0.0) or not np.isfinite(expected) or
                    np.max(np.abs(differences - expected)) > float(thresholds.timestamp_relative_tolerance) * expected):
                reasons.add(QCReason.TIMESTAMP_FAILURE.value)
                reasons.add(QCReason.SYNCHRONY_FAILURE.value)

    ordered = tuple(sorted(reasons))
    return QCAssessment(
        status=QCStatus.FAIL if ordered else QCStatus.PASS,
        reasons=ordered,
        metrics=metrics,
    )


def assess_manifest_record(
    row: ManifestRow,
    values: np.ndarray,
    *,
    observed_channel_names: Sequence[str],
    observed_fs: float,
    thresholds: QCThresholds,
    timestamps_s: np.ndarray | None,
) -> RecordingQCAdmission:
    """Bind physical recording QC to one frozen manifest row.

    This adapter is deliberately separate from SQI. Device-dependent absolute
    rails, clipping and saturation remain deferred when device limits are not
    verified; duration, finite channels, flatline, bounded missing runs, schema,
    sample count, sampling rate and synchrony are still enforced directly.
    """

    matrix = np.asarray(values, dtype=np.float64)
    base = assess_numeric_record(
        matrix,
        observed_channel_names,
        fs=float(observed_fs),
        thresholds=thresholds,
        timestamps_s=timestamps_s,
    )
    reasons = set(base.reasons)
    metrics = dict(base.metrics)
    metrics.update({
        "manifest_record_id": row.record_id,
        "manifest_n_samples": int(row.n_samples),
        "observed_n_samples": int(matrix.shape[0]) if matrix.ndim >= 1 else 0,
        "manifest_fs_hz": float(row.fs),
        "observed_fs_hz": float(observed_fs),
        "device_dependent_checks_executed": bool(thresholds.device_limits_verified),
    })
    if tuple(observed_channel_names) != tuple(row.channel_schema):
        reasons.add("manifest_channel_schema_mismatch")
    if matrix.ndim != 2 or matrix.shape[0] != int(row.n_samples):
        reasons.add("manifest_sample_count_mismatch")
    if not np.isclose(float(observed_fs), float(row.fs), rtol=0.0, atol=0.0):
        reasons.add("manifest_sampling_rate_mismatch")
    ordered = tuple(sorted(reasons))
    assessment = QCAssessment(
        status=QCStatus.FAIL if ordered else QCStatus.PASS,
        reasons=ordered,
        metrics=metrics,
    )
    admitted = assessment.status is QCStatus.PASS
    evidence: dict[str, Any] = {
        "schema_version": "ppg_frailty.recording_qc_admission.v2",
        "record_id": row.record_id,
        "manifest_version": row.manifest_version,
        "source_hash": row.source_hash,
        "admitted": admitted,
        "status": assessment.status.value,
        "reasons": list(assessment.reasons),
        "metrics": assessment.metrics,
        "device_dependent_qc_status": assessment.metrics.get("device_dependent_qc_status"),
        "sqi_or_classifier_effect": "none_recording_safety_admission_only",
    }
    result = RecordingQCAdmission(row.record_id, admitted, assessment, evidence)
    result.validate()
    return result


def require_recording_qc_pass(admission: RecordingQCAdmission) -> None:
    """Fail closed at the formal loader boundary without inventing repair."""

    admission.validate()
    if not admission.admitted:
        raise ValueError("recording failed manifest-bound QC: " + ";".join(admission.assessment.reasons))


def physical_recording_qc_thresholds_v2() -> QCThresholds:
    """Return the frozen non-device physical admission thresholds."""

    channels = tuple(CANONICAL_CHANNEL_SCHEMA)
    return QCThresholds(
        minimum_duration_s=5.0,
        maximum_nonfinite_gap_s=0.0,
        flatline_std_floor_by_channel={channel: 0.0
                                       for channel in channels},
        robust_span_floor_by_channel={channel: 0.0
                                      for channel in channels},
        absolute_limit_by_channel={channel: None
                                   for channel in channels},
        saturation_fraction_limit=None,
        timestamp_relative_tolerance=None,
        timestamps_required=False,
        device_limits_verified=False,
    )


def physical_recording_qc_profile_v2() -> dict[str, Any]:
    """Serialize the exact physical gate and its deferred device boundary."""

    thresholds = physical_recording_qc_thresholds_v2()
    return {
        "profile_id": "physical_recording_qc_thresholds_v2",
        "minimum_duration_s": thresholds.minimum_duration_s,
        "minimum_duration_authority": "shortest_formal_window_raw_dl_5s",
        "maximum_nonfinite_gap_s": thresholds.maximum_nonfinite_gap_s,
        "flatline_std_floor_by_channel": dict(thresholds.flatline_std_floor_by_channel),
        "robust_span_floor_by_channel": dict(thresholds.robust_span_floor_by_channel),
        "timestamps_required": thresholds.timestamps_required,
        "timestamp_grid_authority": "manifest_fs_csv_has_no_timestamp_column",
        "device_limits_verified": thresholds.device_limits_verified,
        "absolute_limit_by_channel": dict(thresholds.absolute_limit_by_channel),
        "saturation_fraction_limit": thresholds.saturation_fraction_limit,
        "device_dependent_checks": "deferred_not_executed_v2_006",
        "sqi_or_classifier_effect": "none_recording_safety_admission_only",
    }


__all__ = [
    "QCAssessment",
    "QCThresholds",
    "RecordingQCAdmission",
    "assess_manifest_record",
    "assess_numeric_record",
    "parse_failure_assessment",
    "physical_recording_qc_profile_v2",
    "physical_recording_qc_thresholds_v2",
    "require_recording_qc_pass",
]

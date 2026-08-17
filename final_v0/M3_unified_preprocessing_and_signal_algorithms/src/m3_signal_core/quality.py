"""输入质量检查与有限缺口修复 / Input quality checks and bounded-gap repair.

中文：检查发生在滤波和缩放之前。只有被有限规则允许的内部短缺口可以插值；
边界缺口、长缺口、全无效、过量非有限值和平线都产生显式非成功状态。

English: Inspection runs before filtering or scaling. Only bounded internal short gaps
may be interpolated. Boundary gaps, long gaps, all-invalid channels, excessive
non-finite values, and flatlines produce explicit non-success states.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

import numpy as np

from .contracts import ProcessingStatus, QualityAssessment, QualityIssue


def validate_timestamp_grid(
    timestamps_s: np.ndarray | None,
    fs_hz: float,
    expected_length: int,
    *,
    previous_timestamp_s: float | None = None,
    jitter_tolerance_fraction: float = 0.05,
) -> list[QualityIssue]:
    """验证显式时间轴；None 表示已登记的隐式均匀网格。

    Validate an explicit time grid. None means that the caller is using a
    separately registered implicit uniform grid, as is the case for Frailty3.
    """

    if timestamps_s is None:
        return []
    time = np.asarray(timestamps_s, dtype=np.float64).ravel()
    issues: list[QualityIssue] = []
    if time.size != int(expected_length):
        return [
            QualityIssue(
                "timestamp_length_mismatch",
                "fatal",
                f"timestamps={time.size}, samples={expected_length}",
            )
        ]
    if not np.isfinite(time).all():
        return [QualityIssue("timestamp_nonfinite", "fatal", "Timestamps must be finite.")]
    if time.size < 2:
        return []
    intervals = np.diff(time)
    if np.any(intervals <= 0.0):
        issues.append(
            QualityIssue(
                "timestamp_not_strictly_increasing",
                "fatal",
                "Timestamp differences must all be positive.",
            )
        )
        return issues
    expected_dt = 1.0 / float(fs_hz)
    relative_error = np.abs(intervals - expected_dt) / expected_dt
    if float(np.percentile(relative_error, 99.0)) > float(jitter_tolerance_fraction):
        issues.append(
            QualityIssue(
                "timestamp_jitter_exceeds_tolerance",
                "fatal",
                (
                    f"p99_relative_error={np.percentile(relative_error, 99.0):.6f}, "
                    f"limit={jitter_tolerance_fraction:.6f}"
                ),
            )
        )
    if previous_timestamp_s is not None:
        boundary_dt = float(time[0] - previous_timestamp_s)
        boundary_error = abs(boundary_dt - expected_dt) / expected_dt
        if boundary_dt <= 0.0 or boundary_error > float(jitter_tolerance_fraction):
            issues.append(
                QualityIssue(
                    "timestamp_chunk_boundary_gap",
                    "fatal",
                    f"boundary_dt={boundary_dt:.9f}, expected_dt={expected_dt:.9f}",
                )
            )
    return issues


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """返回布尔 mask 的右开连续区间 / Return right-open runs of a Boolean mask."""

    values = np.asarray(mask, dtype=bool).ravel()
    if values.size == 0:
        return []
    padded = np.concatenate(([False], values, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _longest_low_change_run(signal: np.ndarray, tolerance: float) -> int:
    """计算近似平线的最长样本数 / Compute the longest approximately-flat run."""

    values = np.asarray(signal, dtype=np.float64).ravel()
    if values.size == 0:
        return 0
    if values.size == 1:
        return 1
    low_change = np.abs(np.diff(values)) <= float(tolerance)
    runs = _true_runs(low_change)
    return max((stop - start + 1 for start, stop in runs), default=1)


def validate_channel_contract(
    observed: Sequence[str],
    required: Sequence[str],
    *,
    allow_extra: bool = False,
) -> list[QualityIssue]:
    """验证通道顺序和缺失 / Validate channel order and missing channels."""

    observed_list = list(observed)
    required_list = list(required)
    issues: list[QualityIssue] = []
    missing = [name for name in required_list if name not in observed_list]
    if missing:
        issues.append(
            QualityIssue(
                code="missing_channel",
                severity="fatal",
                detail="Missing required channels: " + ",".join(missing),
            )
        )
    if not missing and observed_list[: len(required_list)] != required_list:
        issues.append(
            QualityIssue(
                code="channel_order_mismatch",
                severity="fatal",
                detail=f"Expected {required_list}, observed {observed_list}",
            )
        )
    if not allow_extra and len(observed_list) != len(required_list):
        issues.append(
            QualityIssue(
                code="unexpected_channel_count",
                severity="fatal",
                detail=f"Expected {len(required_list)}, observed {len(observed_list)}",
            )
        )
    return issues


def inspect_and_repair_signal(
    signal: np.ndarray,
    fs_hz: float,
    *,
    channel_names: Sequence[str] | None = None,
    profile_id: str = "corrected_v1",
    max_gap_sec: float = 0.25,
    max_nonfinite_fraction: float = 0.01,
    min_duration_sec: float = 0.0,
    flatline_channels: Iterable[str] | None = None,
    flatline_sec: float = 1.0,
    clipping_occupancy: float = 0.02,
    clipping_plateau_sec: float = 0.25,
) -> QualityAssessment:
    """检查并修复一维或二维信号 / Inspect and repair a 1-D or 2-D signal.

    中文：二维输入约定为 samples × channels。clipping 仅产生 heuristic warning，
    因为 M2 未获得设备 ADC 量程；平线和不可修复缺口则是 fatal。

    English: Two-dimensional input is samples × channels. Clipping only emits a
    heuristic warning because M2 found no authoritative ADC range; flatlines and
    irreparable gaps are fatal.
    """

    values = np.asarray(signal, dtype=np.float64)
    original_ndim = values.ndim
    if original_ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        empty = np.asarray(signal, dtype=np.float64)
        return QualityAssessment(
            status=ProcessingStatus.INVALID,
            signal=empty,
            valid_mask=np.zeros_like(empty, dtype=bool),
            repair_mask=np.zeros_like(empty, dtype=bool),
            issues=[QualityIssue("invalid_shape", "fatal", f"shape={empty.shape}")],
            metrics={},
            profile_id=profile_id,
        )
    if not np.isfinite(fs_hz) or float(fs_hz) <= 0:
        return QualityAssessment(
            status=ProcessingStatus.INVALID,
            signal=values[:, 0] if original_ndim == 1 else values,
            valid_mask=np.isfinite(values),
            repair_mask=np.zeros_like(values, dtype=bool),
            issues=[QualityIssue("invalid_sampling_rate", "fatal", f"fs_hz={fs_hz}")],
            metrics={},
            profile_id=profile_id,
        )

    n_samples, n_channels = values.shape
    names = list(channel_names) if channel_names is not None else [
        f"ch{index}" for index in range(n_channels)
    ]
    if len(names) != n_channels:
        return QualityAssessment(
            status=ProcessingStatus.INVALID,
            signal=values[:, 0] if original_ndim == 1 else values,
            valid_mask=np.isfinite(values),
            repair_mask=np.zeros_like(values, dtype=bool),
            issues=[
                QualityIssue(
                    "channel_name_count_mismatch",
                    "fatal",
                    f"names={len(names)}, channels={n_channels}",
                )
            ],
            metrics={"n_samples": int(n_samples), "n_channels": int(n_channels)},
            profile_id=profile_id,
        )
    flatline_set = set(flatline_channels or [])
    original_valid = np.isfinite(values)
    repaired = values.copy()
    repair_mask = np.zeros_like(values, dtype=bool)
    issues: list[QualityIssue] = []
    metrics: dict[str, object] = {
        "n_samples": int(n_samples),
        "n_channels": int(n_channels),
        "duration_sec": float(n_samples / float(fs_hz)),
    }

    if n_samples == 0:
        issues.append(QualityIssue("empty_signal", "fatal", "Signal contains no samples."))
    elif n_samples / float(fs_hz) < float(min_duration_sec):
        issues.append(
            QualityIssue(
                "too_short",
                "insufficient",
                f"duration={n_samples / float(fs_hz):.6f}s < {min_duration_sec:.6f}s",
            )
        )

    max_gap_samples = max(0, int(round(float(max_gap_sec) * float(fs_hz))))
    flatline_samples = max(1, int(round(float(flatline_sec) * float(fs_hz))))
    clipping_plateau_samples = max(
        1, int(round(float(clipping_plateau_sec) * float(fs_hz)))
    )
    channel_metrics: dict[str, dict[str, float | int]] = {}

    for channel_index, channel_name in enumerate(names):
        raw = values[:, channel_index]
        finite = np.isfinite(raw)
        # 中文：直接用计数相除，避免 1-mean 在恰好 1% 边界产生消减误差。
        # English: Count directly to avoid cancellation at the exact one-percent gate.
        nonfinite_fraction = (
            float(np.count_nonzero(~finite) / raw.size) if raw.size else 1.0
        )
        channel_info: dict[str, float | int] = {
            "nonfinite_fraction": nonfinite_fraction,
            "repaired_samples": 0,
            "longest_flat_run_samples": 0,
            "min_occupancy": 0.0,
            "max_occupancy": 0.0,
        }
        if not finite.any():
            issues.append(
                QualityIssue(
                    "all_nonfinite",
                    "fatal",
                    "No finite samples are available.",
                    channel=channel_name,
                )
            )
            channel_metrics[channel_name] = channel_info
            continue
        if nonfinite_fraction > float(max_nonfinite_fraction):
            issues.append(
                QualityIssue(
                    "excessive_nonfinite",
                    "fatal",
                    f"fraction={nonfinite_fraction:.6f} > {max_nonfinite_fraction:.6f}",
                    channel=channel_name,
                )
            )

        for start, stop in _true_runs(~finite):
            gap_length = stop - start
            internal = start > 0 and stop < raw.size
            if internal and gap_length <= max_gap_samples:
                left = repaired[start - 1, channel_index]
                right = repaired[stop, channel_index]
                repaired[start:stop, channel_index] = np.linspace(
                    left, right, gap_length + 2, dtype=np.float64
                )[1:-1]
                repair_mask[start:stop, channel_index] = True
                channel_info["repaired_samples"] = int(
                    channel_info["repaired_samples"]
                ) + gap_length
                issues.append(
                    QualityIssue(
                        "short_gap_interpolated",
                        "repair",
                        f"Interpolated {gap_length} samples.",
                        channel=channel_name,
                        start_index=start,
                        stop_index=stop,
                    )
                )
            else:
                code = "boundary_gap" if not internal else "excessive_gap"
                issues.append(
                    QualityIssue(
                        code,
                        "fatal",
                        f"gap_samples={gap_length}, limit={max_gap_samples}",
                        channel=channel_name,
                        start_index=start,
                        stop_index=stop,
                    )
                )

        cleaned = repaired[:, channel_index]
        if not np.isfinite(cleaned).all():
            channel_metrics[channel_name] = channel_info
            continue
        q25, q75 = np.percentile(cleaned, [25.0, 75.0])
        robust_span = float(q75 - q25)
        longest_flat = _longest_low_change_run(cleaned, 0.0)
        channel_info["longest_flat_run_samples"] = int(longest_flat)
        if channel_name in flatline_set and longest_flat >= flatline_samples:
            issues.append(
                QualityIssue(
                    "flatline",
                    "fatal",
                    f"run_samples={longest_flat}, threshold={flatline_samples}",
                    channel=channel_name,
                )
            )

        minimum = float(np.min(cleaned))
        maximum = float(np.max(cleaned))
        min_mask = cleaned == minimum
        max_mask = cleaned == maximum
        min_occupancy = float(np.mean(min_mask))
        max_occupancy = float(np.mean(max_mask))
        channel_info["min_occupancy"] = min_occupancy
        channel_info["max_occupancy"] = max_occupancy
        longest_extreme = max(
            max((stop - start for start, stop in _true_runs(min_mask)), default=0),
            max((stop - start for start, stop in _true_runs(max_mask)), default=0),
        )
        if (
            min_occupancy >= float(clipping_occupancy)
            or max_occupancy >= float(clipping_occupancy)
            or longest_extreme >= clipping_plateau_samples
        ):
            issues.append(
                QualityIssue(
                    "clipping_suspected_heuristic",
                    "warning",
                    (
                        f"min_occ={min_occupancy:.6f}, max_occ={max_occupancy:.6f}, "
                        f"extreme_run={longest_extreme}"
                    ),
                    channel=channel_name,
                )
            )
        channel_metrics[channel_name] = channel_info

    metrics["channels"] = channel_metrics
    metrics["valid_mask_semantics"] = "source_finite_before_repair"
    metrics["repair_mask_semantics"] = "samples_reconstructed_by_bounded_interpolation"
    severities = {issue.severity for issue in issues}
    if "fatal" in severities:
        status = ProcessingStatus.INVALID
    elif "insufficient" in severities:
        status = ProcessingStatus.INSUFFICIENT
    elif repair_mask.any():
        status = ProcessingStatus.REPAIRED
    else:
        status = ProcessingStatus.VALID
    output_signal = repaired[:, 0] if original_ndim == 1 else repaired
    output_valid = original_valid[:, 0] if original_ndim == 1 else original_valid
    output_repair = repair_mask[:, 0] if original_ndim == 1 else repair_mask
    return QualityAssessment(
        status=status,
        signal=output_signal,
        valid_mask=output_valid,
        repair_mask=output_repair,
        issues=issues,
        metrics=metrics,
        profile_id=profile_id,
    )


def with_contract_issues(
    assessment: QualityAssessment,
    contract_issues: Sequence[QualityIssue],
) -> QualityAssessment:
    """合并通道合同问题 / Merge channel-contract issues into an assessment."""

    issues = [*assessment.issues, *contract_issues]
    status = (
        ProcessingStatus.INVALID
        if any(issue.severity == "fatal" for issue in issues)
        else assessment.status
    )
    return replace(assessment, status=status, issues=issues)

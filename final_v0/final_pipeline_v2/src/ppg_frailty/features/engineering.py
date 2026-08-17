"""10 s/5 s 工程特征与 fold-local 变换 / Engineering features and fold-local transform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import integrate, signal, stats

from ..contracts import EngineeringFeatureSequence, SignalRoute
from ..data.windows import WindowPlan
from ..provenance import assert_training_only
from ..signal.views import (
    CANONICAL_FS_HZ,
    CanonicalSignalViews,
)


TIME_STATISTICS = (
    "mean", "population_sd", "rms", "iqr", "mad", "skew_bias_corrected",
    "pearson_kurtosis",
)
PPG_SPECTRAL_STATISTICS = (
    "total_power",
    "normalized_spectral_entropy",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
)
PPG_BANDS = ((0.2, 0.5), (0.5, 3.0), (3.0, 8.0))
IMU_BANDS = ((0.1, 0.5), (0.5, 3.0), (3.0, 8.0), (8.0, 20.0))
ENGINEERING_SCHEMA_VERSION = "engineering_10s_hop5s_thesis_115_v2"
WELCH_WINDOW = "hann"
WELCH_MAX_SEGMENT_SAMPLES = 2048
WELCH_MIN_SEGMENT_SAMPLES = 64
WELCH_SECONDS = 4.0


@dataclass(frozen=True)
class EngineeringExtraction:
    """现有 sequence 外加逐值 validity / Sequence plus per-value validity."""

    sequence: EngineeringFeatureSequence
    value_validity: np.ndarray
    route: SignalRoute
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FoldFeatureTransform:
    """仅 outer-train 拟合的 robust center/scale / Train-only robust transform."""

    center: np.ndarray
    scale: np.ndarray
    feature_names: tuple[str, ...]
    fitted_on_participant_ids: tuple[str, ...]


def validate_engineering_extraction(
    extraction: EngineeringExtraction,
    *,
    fold_transformed: bool,
) -> None:
    """Reject stale or structurally inconsistent engineering sequences."""

    if not isinstance(extraction, EngineeringExtraction):
        raise TypeError("engineering input must be an EngineeringExtraction")
    sequence = extraction.sequence
    values = np.asarray(sequence.values)
    validity = np.asarray(extraction.value_validity)
    starts = np.asarray(sequence.start_samples)
    row_mask = np.asarray(sequence.valid_row_mask)
    names = engineering_feature_names()
    expected_version = ENGINEERING_SCHEMA_VERSION + (
        "+fold_robust_v1" if fold_transformed else ""
    )
    if (
        tuple(sequence.channel_schema) != names
        or sequence.schema_version != expected_version
        or values.ndim != 2
        or values.shape[1] != len(names)
        or validity.shape != values.shape
        or starts.shape != (values.shape[0],)
        or row_mask.shape != (values.shape[0],)
    ):
        raise ValueError("engineering extraction schema/shape is stale or inconsistent")


def engineering_welch_parameters(sample_count: int, fs_hz: float) -> tuple[int, int]:
    """Return the frozen Welch segment and overlap sizes."""

    count = int(sample_count)
    sampling_rate = float(fs_hz)
    if count <= 0 or not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("Welch sample count and sampling rate must be positive")
    nperseg = min(
        count,
        max(
            WELCH_MIN_SEGMENT_SAMPLES,
            min(
                WELCH_MAX_SEGMENT_SAMPLES,
                int(round(WELCH_SECONDS * sampling_rate)),
            ),
        ),
    )
    return nperseg, nperseg // 2


def _entropy(power: np.ndarray) -> float:
    """Shannon entropy / log(bin count) / Normalized Shannon spectral entropy."""

    total = float(np.sum(power))
    if total <= 0.0 or power.size < 2:
        return float("nan")
    distribution = power / total
    positive = distribution > 0.0
    return float(
        -np.sum(distribution[positive] * np.log(distribution[positive]))
        / np.log(distribution.size)
    )


def _band_power(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    """Welch band integral / Welch 频带积分。"""

    mask = (frequencies >= low) & (frequencies <= high)
    return (
        float(integrate.trapezoid(power[mask], frequencies[mask]))
        if np.count_nonzero(mask) >= 2
        else float("nan")
    )


def _one_channel_features(
    values: np.ndarray,
    *,
    fs_hz: float,
    bands: tuple[tuple[float, float], ...],
    include_spectral_summary: bool,
) -> tuple[list[float], list[bool]]:
    """提取 time statistics + Welch bands / Extract one channel descriptor."""

    source = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(source)
    if np.mean(finite) < 0.80:
        count = (
            len(TIME_STATISTICS)
            + (len(PPG_SPECTRAL_STATISTICS) if include_spectral_summary else 0)
            + len(bands)
        )
        return [float("nan")] * count, [False] * count
    x = source[finite]
    q25, q75 = np.percentile(x, [25.0, 75.0])
    median = float(np.median(x))
    # 中文：Welch 只用最长连续有限 run，绝不跨内部 invalid gap 拼接。
    # English: Welch uses the longest contiguous finite run and never bridges a gap.
    padded = np.concatenate(([False], finite, [False]))
    changes = np.diff(padded.astype(np.int8))
    runs = list(
        zip(
            np.flatnonzero(changes == 1),
            np.flatnonzero(changes == -1),
        )
    )
    start, stop = max(runs, key=lambda bounds: bounds[1] - bounds[0])
    spectral_x = source[start:stop]
    nperseg, noverlap = engineering_welch_parameters(spectral_x.size, fs_hz)
    frequencies, power = signal.welch(
        spectral_x,
        fs=fs_hz,
        window=WELCH_WINDOW,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=True,
    )
    if np.any(power < 0.0):
        raise RuntimeError("Welch returned a negative one-sided PSD")
    features = [
        float(np.mean(x)),
        float(np.std(x, ddof=0)),
        float(np.sqrt(np.mean(np.square(x)))),
        float(q75 - q25),
        float(np.median(np.abs(x - median))),
        float(stats.skew(x, bias=False)),
        float(stats.kurtosis(x, fisher=False, bias=False)),
    ]
    if include_spectral_summary:
        total_power = (
            float(integrate.trapezoid(power, frequencies))
            if frequencies.size >= 2
            else float("nan")
        )
        positive_power = (
            power.size > 1
            and np.isfinite(power).all()
            and float(np.sum(power)) > 0.0
        )
        dominant_frequency = (
            float(frequencies[int(np.argmax(power))])
            if positive_power
            else float("nan")
        )
        spectral_centroid = (
            float(
                np.sum(frequencies * power)
                / (float(np.sum(power)) + np.finfo(np.float64).eps)
            )
            if positive_power
            else float("nan")
        )
        features.extend(
            (
                total_power,
                _entropy(power),
                dominant_frequency,
                spectral_centroid,
            )
        )
    features.extend(_band_power(frequencies, power, low, high) for low, high in bands)
    return features, [bool(np.isfinite(value)) for value in features]


def engineering_feature_names() -> tuple[str, ...]:
    """返回冻结有序工程 schema / Return the frozen ordered engineering schema."""

    names: list[str] = []
    for channel in ("ppg_red", "ppg_ir"):
        names.extend(f"{channel}.{statistic}" for statistic in TIME_STATISTICS)
        names.extend(
            f"{channel}.{statistic}" for statistic in PPG_SPECTRAL_STATISTICS
        )
        names.extend(f"{channel}.bandpower_{low:g}_{high:g}_hz" for low, high in PPG_BANDS)
    for channel in ("acc_magnitude", "angular_rate_magnitude", "jerk_magnitude"):
        names.extend(f"{channel}.{statistic}" for statistic in TIME_STATISTICS)
        names.extend(
            f"{channel}.{statistic}" for statistic in PPG_SPECTRAL_STATISTICS
        )
        names.extend(f"{channel}.bandpower_{low:g}_{high:g}_hz" for low, high in IMU_BANDS)
    for channel in (
        "acc_dynamic_x", "acc_dynamic_y", "acc_dynamic_z",
        "gyro_x", "gyro_y", "gyro_z",
    ):
        names.extend(f"{channel}.{statistic}" for statistic in TIME_STATISTICS)
    frozen = tuple(names)
    if len(frozen) != 115 or len(set(frozen)) != 115:
        raise RuntimeError("canonical engineering schema must contain 115 unique columns")
    return frozen


def _imu_columns(views: CanonicalSignalViews) -> list[tuple[np.ndarray, bool]]:
    """取 A/Omega/J 及六轴 / Read A/Omega/J followed by six measured axes."""

    required = (
        "dynamic_acc_mps2",
        "gyro_rads",
        "dynamic_magnitude",
        "gyro_magnitude",
        "jerk_magnitude",
    )
    missing = [key for key in required if key not in views.imu_processed]
    if missing:
        raise ValueError("missing processed IMU fields: " + ",".join(missing))
    dynamic = np.asarray(views.imu_processed["dynamic_acc_mps2"], dtype=np.float64)
    gyro = np.asarray(views.imu_processed["gyro_rads"], dtype=np.float64)
    acc_magnitude = np.asarray(
        views.imu_processed["dynamic_magnitude"], dtype=np.float64
    )
    gyro_magnitude = np.asarray(
        views.imu_processed["gyro_magnitude"], dtype=np.float64
    )
    jerk_magnitude = np.asarray(
        views.imu_processed["jerk_magnitude"], dtype=np.float64
    )
    if dynamic.shape != gyro.shape or dynamic.shape != (views.x_filter.shape[0], 3):
        raise ValueError("processed IMU axes lost PPG alignment")
    expected_scalar_shape = (views.x_filter.shape[0],)
    if any(
        item.shape != expected_scalar_shape
        for item in (acc_magnitude, gyro_magnitude, jerk_magnitude)
    ):
        raise ValueError("processed IMU magnitude signals lost PPG alignment")
    if not np.allclose(
        acc_magnitude,
        np.linalg.norm(dynamic, axis=1),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError("canonical acceleration magnitude differs from dynamic axes")
    if not np.allclose(
        gyro_magnitude,
        np.linalg.norm(gyro, axis=1),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError("canonical angular-rate magnitude differs from gyro axes")
    return [
        (acc_magnitude, True),
        (gyro_magnitude, True),
        (jerk_magnitude, True),
        (dynamic[:, 0], False),
        (dynamic[:, 1], False),
        (dynamic[:, 2], False),
        (gyro[:, 0], False),
        (gyro[:, 1], False),
        (gyro[:, 2], False),
    ]


def extract_engineering_features(
    views: CanonicalSignalViews,
    *,
    plan: WindowPlan,
) -> EngineeringExtraction:
    """按共享计划提取工程 rows / Extract chronological engineering rows.

    非恒等 `x_ar` 的 PPG engineering predictor slots 明确为 NaN/false；只保留 IMU
    predictors。For a non-identity route, PPG slots are unavailable rather than copied.
    """

    views.validate()
    selected_plan = plan
    expected_plan = (
        10.0,
        5.0,
        "start",
        "reject",
        False,
        None,
        "not_applicable",
    )
    observed_plan = (
        float(selected_plan.window_seconds),
        float(selected_plan.hop_seconds),
        selected_plan.end_alignment,
        selected_plan.short_record_action,
        selected_plan.include_padded_tail,
        selected_plan.max_windows,
        selected_plan.cap_policy,
    )
    if observed_plan != expected_plan:
        raise ValueError(
            "engineering extraction requires the frozen complete-window "
            "10 s / 5 s-hop WindowPlan"
        )
    record_id = str(views.metadata.get("record_id", ""))
    if not record_id or selected_plan.source_record_id != record_id:
        raise ValueError("WindowPlan source_record_id must exactly match signal metadata")
    windows = selected_plan.plan(views.x_filter.shape[0], CANONICAL_FS_HZ)
    names = engineering_feature_names()
    rows: list[list[float]] = []
    validities: list[list[bool]] = []
    starts: list[int] = []
    imu_columns = _imu_columns(views)
    reasons: list[str] = []
    for item in windows:
        if item.valid_length != item.window_length or any(item.padding_mask):
            # 中文：engineering reference 明确只接受 complete window。
            # English: The engineering reference explicitly rejects padded windows.
            continue
        start, stop = item.start_sample, item.end_sample
        row: list[float] = []
        row_validity: list[bool] = []
        if views.route in {SignalRoute.DIRECT, SignalRoute.IDENTITY}:
            for channel in range(2):
                values, validity = _one_channel_features(
                    views.x_filter[start:stop, channel],
                    fs_hz=CANONICAL_FS_HZ,
                    bands=PPG_BANDS,
                    include_spectral_summary=True,
                )
                row.extend(values)
                row_validity.extend(validity)
        else:
            unavailable = (
                len(TIME_STATISTICS)
                + len(PPG_SPECTRAL_STATISTICS)
                + len(PPG_BANDS)
            )
            row.extend([float("nan")] * (2 * unavailable))
            row_validity.extend([False] * (2 * unavailable))
            reasons.append("non_identity_ppg_engineering_unavailable")
        for channel, include_spectral_summary in imu_columns:
            values, validity = _one_channel_features(
                channel[start:stop],
                fs_hz=CANONICAL_FS_HZ,
                bands=IMU_BANDS if include_spectral_summary else (),
                include_spectral_summary=include_spectral_summary,
            )
            row.extend(values)
            row_validity.extend(validity)
        if len(row) != len(names):
            raise RuntimeError("engineering schema length mismatch")
        rows.append(row)
        validities.append(row_validity)
        starts.append(start)
    values_array = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(names))
    validity_array = np.asarray(validities, dtype=bool).reshape(len(rows), len(names))
    sequence = EngineeringFeatureSequence(
        values=values_array,
        start_samples=np.asarray(starts, dtype=np.int64),
        valid_row_mask=np.ones(len(rows), dtype=bool),
        channel_schema=names,
        schema_version=ENGINEERING_SCHEMA_VERSION,
    )
    return EngineeringExtraction(
        sequence=sequence,
        value_validity=validity_array,
        route=views.route,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def fit_fold_feature_transform(
    extractions: Iterable[EngineeringExtraction],
    *,
    fitted_on_participant_ids: Iterable[str],
    outer_train_participant_ids: Iterable[str],
    outer_oof_participant_ids: Iterable[str],
) -> FoldFeatureTransform:
    """仅 train rows 拟合 median/IQR / Fit median/IQR on outer-train rows only."""

    fitted = assert_training_only(
        fitted_on_participant_ids, outer_train_participant_ids, outer_oof_participant_ids
    )
    items = tuple(extractions)
    if not items:
        raise ValueError("at least one train extraction is required")
    for item in items:
        validate_engineering_extraction(item, fold_transformed=False)
    names = items[0].sequence.channel_schema
    if any(item.sequence.channel_schema != names for item in items):
        raise ValueError("engineering schemas differ across train records")
    matrix = np.vstack([item.sequence.values for item in items])
    validity = np.vstack([item.value_validity for item in items])
    center = np.zeros(matrix.shape[1], dtype=np.float64)
    scale = np.ones(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        selected = matrix[:, column][validity[:, column] & np.isfinite(matrix[:, column])]
        if selected.size:
            center[column] = np.median(selected)
            q25, q75 = np.percentile(selected, [25.0, 75.0])
            scale[column] = max(float(q75 - q25), 1e-8)
    return FoldFeatureTransform(center, scale, names, fitted)


def transform_engineering(
    extraction: EngineeringExtraction,
    transform: FoldFeatureTransform,
) -> EngineeringExtraction:
    """应用 train-only 变换，保持 unavailable=NaN / Apply without imputing unavailable slots."""

    validate_engineering_extraction(extraction, fold_transformed=False)
    expected_names = engineering_feature_names()
    if (
        extraction.sequence.channel_schema != transform.feature_names
        or transform.feature_names != expected_names
        or np.asarray(transform.center).shape != (len(expected_names),)
        or np.asarray(transform.scale).shape != (len(expected_names),)
        or not np.isfinite(transform.center).all()
        or not np.isfinite(transform.scale).all()
        or np.any(np.asarray(transform.scale) <= 0.0)
    ):
        raise ValueError("transform schema differs from engineering sequence")
    values = np.asarray(extraction.sequence.values, dtype=np.float64).copy()
    valid = np.asarray(extraction.value_validity, dtype=bool)
    transformed = (values - transform.center) / transform.scale
    transformed[~valid] = np.nan
    sequence = EngineeringFeatureSequence(
        values=transformed,
        start_samples=extraction.sequence.start_samples.copy(),
        valid_row_mask=extraction.sequence.valid_row_mask.copy(),
        channel_schema=extraction.sequence.channel_schema,
        schema_version=extraction.sequence.schema_version + "+fold_robust_v1",
    )
    return EngineeringExtraction(sequence, valid.copy(), extraction.route, extraction.reasons)

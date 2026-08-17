"""版本化 PPG 清洗、滤波与幅值保留 / Versioned PPG cleaning and filtering.

中文：离线零相位和移动端因果模式显式分版；函数从不在短输入上偷偷切换相位模式。
原始 DC/AC 指标在任何标准化之前计算并保留。

English: Offline zero-phase and mobile causal modes are explicitly versioned. The
function never silently changes phase mode for short inputs. Raw DC/AC descriptors are
computed and retained before any normalization.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
from scipy import signal, stats

from .contracts import (
    ExternalResampleResult,
    PpgPreprocessResult,
    ProcessingStatus,
    QualityIssue,
)
from .quality import (
    inspect_and_repair_signal,
    validate_timestamp_grid,
    with_contract_issues,
)
from .registry import get_profile, registry_sha256


def design_ppg_sos(
    fs_hz: float,
    low_hz: float,
    high_hz: float,
    *,
    order: int = 3,
) -> np.ndarray:
    """设计固定 SOS 带通 / Design the frozen SOS band-pass."""

    fs_value = float(fs_hz)
    if not np.isfinite(fs_value) or fs_value <= 0:
        raise ValueError("fs_hz must be positive and finite")
    if not 0 < float(low_hz) < float(high_hz) < fs_value / 2:
        raise ValueError("band edges must satisfy 0 < low < high < Nyquist")
    return signal.butter(
        int(order),
        [float(low_hz), float(high_hz)],
        btype="bandpass",
        fs=fs_value,
        output="sos",
    )


def raw_ppg_metrics(values: np.ndarray) -> dict[str, float]:
    """保留 DC、AC、灌注代理和振幅 / Preserve DC, AC, perfusion proxy, and span."""

    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0 or not np.isfinite(x).all():
        return {
            "dc_median": float("nan"),
            "ac_std": float("nan"),
            "robust_peak_to_peak": float("nan"),
            "perfusion_index_proxy": float("nan"),
        }
    dc = float(np.median(x))
    ac = float(np.std(x))
    q05, q95 = np.percentile(x, [5.0, 95.0])
    span = float(q95 - q05)
    return {
        "dc_median": dc,
        "ac_std": ac,
        "robust_peak_to_peak": span,
        "perfusion_index_proxy": float(ac / max(abs(dc), 1e-12)),
    }


def _source_and_repaired_metrics(
    source_values: np.ndarray,
    repaired_values: np.ndarray,
) -> dict[str, float]:
    """分开记录 source/repaired 描述量 / Separate source and repaired descriptors."""

    source = np.asarray(source_values, dtype=np.float64).ravel()
    finite_source = source[np.isfinite(source)]
    source_metrics = raw_ppg_metrics(finite_source)
    repaired_metrics = raw_ppg_metrics(repaired_values)
    # 中文：保留旧无前缀字段为 source-finite 口径，同时提供无歧义前缀。
    # English: Keep legacy unprefixed keys as source-finite metrics and add explicit views.
    return {
        **source_metrics,
        **{f"source_{key}": value for key, value in source_metrics.items()},
        **{f"repaired_{key}": value for key, value in repaired_metrics.items()},
        "source_nonfinite_fraction": (
            float(1.0 - finite_source.size / source.size) if source.size else 1.0
        ),
    }


def dual_ppg_raw_metrics(
    red_values: np.ndarray,
    infrared_values: np.ndarray,
) -> dict[str, float]:
    """保留双通道幅值与比例 / Preserve dual-channel amplitude and ratios.

    中文：比例只在归一化前的 repaired raw view 上计算；它们不能从逐窗
    median/IQR 标准化后的 shape view 反推。

    English: Ratios are computed from the repaired raw view before normalization.
    They cannot be reconstructed from the per-window shape-normalized view.
    """

    red = raw_ppg_metrics(red_values)
    infrared = raw_ppg_metrics(infrared_values)

    def safe_ratio(left: float, right: float) -> float:
        """有限分母比例 / Ratio with an explicit finite denominator."""

        return float(left / right) if np.isfinite(left) and np.isfinite(right) and abs(right) > 1e-12 else float("nan")

    return {
        **{f"red_{key}": value for key, value in red.items()},
        **{f"ir_{key}": value for key, value in infrared.items()},
        "red_ir_dc_ratio": safe_ratio(red["dc_median"], infrared["dc_median"]),
        "red_ir_ac_ratio": safe_ratio(red["ac_std"], infrared["ac_std"]),
        "red_ir_pulse_amplitude_ratio": safe_ratio(
            red["robust_peak_to_peak"], infrared["robust_peak_to_peak"]
        ),
        "red_ir_perfusion_proxy_ratio": safe_ratio(
            red["perfusion_index_proxy"], infrared["perfusion_index_proxy"]
        ),
    }


def preprocess_ppg(
    values: np.ndarray,
    fs_hz: float,
    *,
    profile_id: str,
    initial_filter_state: np.ndarray | None = None,
    timestamps_s: np.ndarray | None = None,
) -> PpgPreprocessResult:
    """严格按注册 profile 清洗并带通 / Process only by a registered profile."""

    profile = get_profile(profile_id)
    allowed_purposes = {
        "static_preprocessing",
        "motion_preprocessing",
        "peak_detection_input",
        "denoiser_input",
    }
    if (
        profile.get("status") != "future_active"
        or profile.get("modality") != "ppg"
        or profile.get("purpose") not in allowed_purposes
        or profile.get("resampling") != "no_resample"
    ):
        raise ValueError(f"profile_mismatch:not_future_ppg_preprocessing:{profile_id}")
    expected_fs = float(profile["sampling_rate_hz"])
    if not np.isclose(float(fs_hz), expected_fs, rtol=0.0, atol=1e-12):
        raise ValueError(f"profile_mismatch:fs_hz={fs_hz}, expected={expected_fs}")
    low_hz, high_hz = (float(value) for value in profile["bandpass_hz"])
    order = int(profile["butterworth_order"])
    phase_mode = str(profile["phase_mode"])
    detrend_mode = str(profile["detrend"])
    if initial_filter_state is not None and phase_mode != "causal_stateful":
        raise ValueError("profile_mismatch:offline_profile_rejects_filter_state")

    source_raw = np.asarray(values, dtype=np.float64).ravel().copy()
    quality = inspect_and_repair_signal(
        source_raw,
        fs_hz,
        channel_names=["PPG"],
        profile_id=profile_id,
        max_gap_sec=0.25,
        max_nonfinite_fraction=0.01,
        # 中文：streaming chunk 不是分析窗口；最终 peak/feature 门另行要求 8 s。
        # English: A streaming chunk is not an analysis window; peak/features gate at 8 s.
        min_duration_sec=0.0 if phase_mode == "causal_stateful" else 3.0,
        flatline_channels=["PPG"],
    )
    quality = with_contract_issues(
        quality,
        validate_timestamp_grid(timestamps_s, fs_hz, np.asarray(values).size),
    )
    metadata: dict[str, Any] = {
        "profile_id": profile_id,
        "fs_hz": float(fs_hz),
        "low_hz": float(low_hz),
        "high_hz": float(high_hz),
        "order": int(order),
        "phase_mode": phase_mode,
        "detrend_mode": detrend_mode,
        "notch": "disabled",
        "registry_id": "m3_preprocessing_profiles_corrected_v1",
        "registry_sha256": registry_sha256(),
    }
    repaired_raw = np.asarray(quality.signal, dtype=np.float64).ravel().copy()
    raw_metrics = _source_and_repaired_metrics(source_raw, repaired_raw)
    if quality.status in {ProcessingStatus.INVALID, ProcessingStatus.INSUFFICIENT}:
        return PpgPreprocessResult(
            quality.status,
            None,
            quality,
            raw_metrics,
            metadata,
            None,
            source_raw,
            repaired_raw,
        )

    cleaned = np.asarray(quality.signal, dtype=np.float64).ravel()
    if phase_mode == "offline_zero_phase":
        if detrend_mode == "linear":
            working = signal.detrend(cleaned, type="linear")
        elif detrend_mode == "none":
            working = cleaned.copy()
        else:
            raise ValueError(f"Unsupported detrend_mode: {detrend_mode}")
    elif phase_mode == "causal_stateful":
        if detrend_mode != "none":
            raise ValueError("causal_stateful requires detrend_mode='none'")
        working = cleaned.copy()
    else:
        raise ValueError(f"Unsupported phase_mode: {phase_mode}")

    sos = design_ppg_sos(fs_hz, low_hz, high_hz, order=order)
    if phase_mode == "offline_zero_phase":
        try:
            filtered = signal.sosfiltfilt(sos, working)
        except ValueError as exc:
            quality.issues.append(
                QualityIssue("filter_pad_too_short", "insufficient", str(exc), channel="PPG")
            )
            quality.status = ProcessingStatus.INSUFFICIENT
            return PpgPreprocessResult(
                quality.status,
                None,
                quality,
                raw_metrics,
                metadata,
                None,
                source_raw,
                repaired_raw,
            )
        final_state = None
    else:
        if initial_filter_state is None:
            initial_filter_state = signal.sosfilt_zi(sos) * float(working[0])
        filtered, final_state = signal.sosfilt(
            sos, working, zi=np.asarray(initial_filter_state, dtype=np.float64)
        )
    # 中文：明确区分 source raw spread 与带通后的 AC/pulse amplitude。
    # English: Separate source spread from filtered AC and pulse amplitude.
    filtered_q05, filtered_q95 = np.percentile(filtered, [5.0, 95.0])
    raw_metrics["source_std_including_drift"] = raw_metrics["ac_std"]
    raw_metrics["filtered_ac_std"] = float(np.std(filtered))
    raw_metrics["filtered_pulse_amplitude_p95_p05"] = float(
        filtered_q95 - filtered_q05
    )
    raw_metrics["filtered_ac_to_abs_dc_proxy"] = float(
        raw_metrics["filtered_ac_std"] / max(abs(raw_metrics["dc_median"]), 1e-12)
    )
    return PpgPreprocessResult(
        quality.status,
        np.asarray(filtered, dtype=np.float64),
        quality,
        raw_metrics,
        metadata,
        None if final_state is None else np.asarray(final_state, dtype=np.float64),
        source_raw,
        repaired_raw,
    )


def resample_poly_explicit(
    values: np.ndarray,
    source_fs_hz: float,
    target_fs_hz: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """显式 polyphase 重采样 / Explicit polyphase resampling with provenance."""

    source = float(source_fs_hz)
    target = float(target_fs_hz)
    if source <= 0 or target <= 0:
        raise ValueError("sampling rates must be positive")
    ratio = Fraction(target / source).limit_denominator(10000)
    output = signal.resample_poly(
        np.asarray(values, dtype=np.float64),
        ratio.numerator,
        ratio.denominator,
        axis=0,
    )
    metadata: dict[str, int | float] = {
        "source_fs_hz": source,
        "target_fs_hz": target,
        "up": int(ratio.numerator),
        "down": int(ratio.denominator),
    }
    return np.asarray(output, dtype=np.float64), metadata


def resample_external_ppg_to_400(
    values: np.ndarray,
    source_fs_hz: float,
    *,
    timestamps_s: np.ndarray,
    valid_mask: np.ndarray,
    peak_annotations: np.ndarray | None = None,
    profile_id: str = "external_ppg_to_400_polyphase_v1",
) -> ExternalResampleResult:
    """按唯一注册 profile 同步映射外部 PPG payload / Resample all payloads.

    中文：波形使用 SciPy polyphase anti-alias 重采样；时间、mask 和离散峰事件
    使用同一 source↔target sample-coordinate 比例。输入时间轴必须显式、有限且
    符合来源采样率；不允许用 profile 名掩盖 125 Hz 或其他未登记来源。

    English: The waveform uses SciPy polyphase anti-alias resampling. Time, masks,
    and discrete peak events use the same source-to-target sample-coordinate ratio.
    The source time axis must be explicit, finite, and consistent with source rate;
    a profile ID cannot disguise an unregistered source such as 125 Hz.
    """

    profile = get_profile(profile_id)
    if (
        profile.get("status") != "future_active"
        or profile.get("modality") != "ppg_resampling"
        or profile.get("purpose") != "external_resampling"
        or profile.get("resampling") != "scipy_resample_poly"
    ):
        raise ValueError(f"profile_mismatch:not_future_external_resampling:{profile_id}")
    source_fs = float(source_fs_hz)
    allowed_source_rates = np.asarray(
        profile["source_sampling_rate_hz_allowed"], dtype=np.float64
    )
    if not np.any(np.isclose(source_fs, allowed_source_rates, rtol=0.0, atol=1e-12)):
        raise ValueError(
            f"profile_mismatch:source_fs_hz={source_fs_hz}, "
            f"allowed={allowed_source_rates.tolist()}"
        )
    target_fs = float(profile["target_sampling_rate_hz"])
    if not np.isclose(
        target_fs, float(profile["sampling_rate_hz"]), rtol=0.0, atol=1e-12
    ):
        raise ValueError("profile_mismatch:target_sampling_rate_fields_disagree")

    source_values = np.asarray(values, dtype=np.float64)
    if source_values.ndim not in {1, 2} or source_values.shape[0] < 2:
        raise ValueError(f"external_resampling_invalid_signal_shape:{source_values.shape}")
    if not np.isfinite(source_values).all():
        raise ValueError("external_resampling_signal_must_be_finite_before_filtering")
    source_length = int(source_values.shape[0])
    source_time = np.asarray(timestamps_s, dtype=np.float64).ravel()
    timestamp_issues = validate_timestamp_grid(source_time, source_fs, source_length)
    if timestamp_issues:
        raise ValueError(
            "external_resampling_" + ",".join(issue.code for issue in timestamp_issues)
        )
    source_valid = np.asarray(valid_mask, dtype=bool).ravel()
    if source_valid.size != source_length:
        raise ValueError(
            f"external_resampling_valid_mask_length_mismatch:"
            f"{source_valid.size}!={source_length}"
        )

    source_peaks = (
        np.empty(0, dtype=np.int64)
        if peak_annotations is None
        else np.asarray(peak_annotations)
    )
    if source_peaks.ndim != 1 or (
        source_peaks.size and not np.issubdtype(source_peaks.dtype, np.integer)
    ):
        raise ValueError("external_resampling_peak_annotations_must_be_integer_vector")
    source_peaks = np.asarray(source_peaks, dtype=np.int64)
    if source_peaks.size and (
        np.any(source_peaks < 0)
        or np.any(source_peaks >= source_length)
        or np.any(np.diff(source_peaks) <= 0)
    ):
        raise ValueError(
            "external_resampling_peak_annotations_must_be_sorted_unique_in_bounds"
        )

    output, polyphase = resample_poly_explicit(source_values, source_fs, target_fs)
    target_length = int(output.shape[0])
    # 中文：离散 payload 共用以下 coordinate mapping；峰走 source→target，
    # mask 走 target→nearest-source，时间轴直接位于相同 target coordinates。
    # English: Every discrete payload uses this coordinate map. Peaks map
    # source→target, masks target→nearest source, and time uses target coordinates.
    target_coordinates = np.arange(target_length, dtype=np.float64)
    nearest_source = np.rint(target_coordinates * source_fs / target_fs).astype(
        np.int64
    )
    nearest_source = np.clip(nearest_source, 0, source_length - 1)
    target_valid = source_valid[nearest_source]
    mapped_peaks = np.rint(
        source_peaks.astype(np.float64) * target_fs / source_fs
    ).astype(np.int64)
    mapped_peaks = np.unique(np.clip(mapped_peaks, 0, target_length - 1))
    target_time = float(source_time[0]) + target_coordinates / target_fs
    reasons = [] if bool(np.all(target_valid)) else ["source_valid_mask_partial"]
    status = ProcessingStatus.VALID if not reasons else ProcessingStatus.PARTIAL
    metadata: dict[str, Any] = {
        **polyphase,
        "profile_id": profile_id,
        "registry_id": "m3_preprocessing_profiles_corrected_v1",
        "registry_sha256": registry_sha256(),
        "source_sample_count": source_length,
        "target_sample_count": target_length,
        "time_origin_s": float(source_time[0]),
        "mapping_rule": (
            "target_j_time=t0+j/target_fs;"
            "nearest_source_i=round(j*source_fs/target_fs);"
            "target_peak_j=round(source_peak_i*target_fs/source_fs)"
        ),
        "shared_index_mapping": True,
    }
    return ExternalResampleResult(
        status=status,
        signal=np.asarray(output, dtype=np.float64),
        timestamps_s=np.asarray(target_time, dtype=np.float64),
        valid_mask=np.asarray(target_valid, dtype=bool),
        peak_annotations=np.asarray(mapped_peaks, dtype=np.int64),
        reason_codes=reasons,
        profile_id=profile_id,
        metadata=metadata,
    )


def normalized_spectral_entropy(
    values: np.ndarray,
    fs_hz: float,
    *,
    low_hz: float = 0.4,
    high_hz: float = 8.0,
) -> float:
    """计算 0–1 归一化谱熵 / Compute normalized 0-to-1 spectral entropy."""

    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size < 8 or not np.isfinite(x).all() or np.std(x) <= 1e-12:
        return float("nan")
    frequencies, power = signal.welch(
        x,
        fs=float(fs_hz),
        window="hann",
        nperseg=min(x.size, max(64, int(round(4.0 * float(fs_hz))))),
    )
    keep = (frequencies >= float(low_hz)) & (frequencies <= float(high_hz))
    selected = np.maximum(power[keep], 0.0)
    if selected.size < 2 or float(np.sum(selected)) <= 0:
        return float("nan")
    probability = selected / np.sum(selected)
    entropy = -float(np.sum(probability * np.log(probability + 1e-18)))
    return float(entropy / np.log(selected.size))


def ppg_statistical_features(values: np.ndarray, fs_hz: float) -> dict[str, float]:
    """生成 SQI 可复用统计量 / Produce reusable SQI statistical primitives."""

    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size < 4 or not np.isfinite(x).all():
        return {
            "skew": float("nan"),
            "kurtosis_pearson": float("nan"),
            "normalized_spectral_entropy": float("nan"),
        }
    return {
        "skew": float(stats.skew(x, bias=False)),
        "kurtosis_pearson": float(stats.kurtosis(x, fisher=False, bias=False)),
        "normalized_spectral_entropy": normalized_spectral_entropy(x, fs_hz),
    }

"""统一 peak/PPI/HR/PPG-derived PRV 后端 / Shared physiology backend.

中文：corrected_v1 分离峰事件和 PPI validity；异常间期不删除原峰。RED/IR 各自
检测，由 SQI 选择主通道，不生成或移动共识峰。PPG 变异性明确命名为 PRV。

English: corrected_v1 separates peak events from PPI validity, so invalid intervals
never delete source peaks. RED and IR are detected independently; SQI selects the
primary channel without generating or shifting consensus peaks. PPG variability is
explicitly named PRV.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from .contracts import HrvResult, PeakResult, ProcessingStatus
from .registry import get_profile


PPI_MIN_SEC = 0.30
PPI_MAX_SEC = 2.00


def _window_bounds(length: int, fs_hz: float) -> list[tuple[int, int]]:
    """生成 10 s / 5 s hop 窗口 / Build frozen local detection windows."""

    minimum = int(round(8.0 * fs_hz))
    window = int(round(10.0 * fs_hz))
    hop = int(round(5.0 * fs_hz))
    if length < minimum:
        return []
    if length <= window:
        return [(0, length)]
    bounds = [
        (start, min(start + window, length))
        for start in range(0, length - minimum + 1, hop)
    ]
    if bounds[-1][1] < length:
        bounds.append((max(0, length - window), length))
    return sorted(set(bounds))


def _merge_events(
    indices: np.ndarray,
    confidences: np.ndarray,
    fs_hz: float,
    *,
    radius_sec: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """聚类重叠窗事件并保留最高置信既有峰 / Merge without shifting peaks."""

    if indices.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    order = np.argsort(indices, kind="stable")
    locations = np.asarray(indices, dtype=np.int64)[order]
    scores = np.asarray(confidences, dtype=np.float64)[order]
    radius = max(1, int(round(float(radius_sec) * float(fs_hz))))
    merged_locations: list[int] = []
    merged_scores: list[float] = []
    start = 0
    while start < locations.size:
        stop = start + 1
        while stop < locations.size and locations[stop] - locations[start] <= radius:
            stop += 1
        best = start + int(np.argmax(scores[start:stop]))
        merged_locations.append(int(locations[best]))
        merged_scores.append(float(scores[best]))
        start = stop
    return np.asarray(merged_locations), np.asarray(merged_scores)


def derive_ppi(peaks: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """保留全部 PPI 并返回 0.30–2.00 s mask / Preserve PPI and validity mask."""

    locations = np.asarray(peaks, dtype=np.int64)
    if locations.size < 2:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)
    raw = np.diff(locations).astype(np.float64) / float(fs_hz)
    return raw, (raw >= PPI_MIN_SEC) & (raw <= PPI_MAX_SEC)


def _polarity_score(peaks: np.ndarray, fs_hz: float) -> float:
    """冻结极性评分 / Frozen polarity score."""

    raw, mask = derive_ppi(peaks, fs_hz)
    valid = raw[mask]
    if valid.size == 0:
        return float(peaks.size) * 0.001
    duration = max((peaks[-1] - peaks[0]) / float(fs_hz), 1e-12)
    coverage = float(np.sum(valid) / duration)
    variability = float(np.std(valid) / max(np.mean(valid), 1e-12))
    return float(valid.size + 0.5 * min(coverage, 1.0) - min(variability, 2.0))


def detect_peaks_corrected(
    filtered_ppg: np.ndarray,
    fs_hz: float,
    *,
    profile_id: str = "frailty3_peak_ppg_400_offline_v1",
) -> PeakResult:
    """在 canonical 0.4–8 Hz 波形上双极性检测 / Detect canonical PPG peaks."""

    profile = get_profile(profile_id)
    if (
        profile.get("status") != "future_active"
        or profile.get("modality") != "ppg"
        or profile.get("purpose") != "peak_detection_input"
    ):
        raise ValueError(f"profile_mismatch:not_future_peak_input:{profile_id}")
    expected_fs = float(profile["sampling_rate_hz"])
    if not np.isclose(float(fs_hz), expected_fs, rtol=0.0, atol=1e-12):
        raise ValueError(f"profile_mismatch:fs_hz={fs_hz}, expected={expected_fs}")
    if (
        [float(value) for value in profile.get("bandpass_hz", [])] != [0.4, 8.0]
        or int(profile.get("butterworth_order", -1)) != 3
        or profile.get("notch") != "disabled"
    ):
        raise ValueError(f"profile_mismatch:peak_filter_contract:{profile_id}")
    values = np.asarray(filtered_ppg, dtype=np.float64).ravel()
    if values.size < int(round(8.0 * float(fs_hz))):
        return PeakResult(
            ProcessingStatus.INSUFFICIENT,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            0,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            float("-inf"),
            ["DURATION_LT_8S"],
            profile_id=profile_id,
        )
    if not np.isfinite(values).all() or float(np.std(values)) <= 1e-12:
        return PeakResult(
            ProcessingStatus.INVALID,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            0,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            float("-inf"),
            ["PEAK_INPUT_INVALID"],
            profile_id=profile_id,
        )

    distance = max(1, int(np.floor(PPI_MIN_SEC * float(fs_hz))))
    best: PeakResult | None = None
    for polarity in (1, -1):
        candidate_locations: list[int] = []
        candidate_scores: list[float] = []
        oriented = values * float(polarity)
        for start, stop in _window_bounds(values.size, fs_hz):
            segment = oriented[start:stop]
            q25, q75 = np.percentile(segment, [25.0, 75.0])
            scale = max(
                float((q75 - q25) / 1.349),
                float(np.std(segment)),
                1e-12,
            )
            peaks, properties = signal.find_peaks(
                segment,
                distance=distance,
                prominence=0.25 * scale,
            )
            prominences = np.asarray(
                properties.get("prominences", np.ones(peaks.size)),
                dtype=np.float64,
            )
            candidate_locations.extend((peaks + start).astype(int).tolist())
            # 中文：1-exp(-x) 保持排序并把置信度严格压到 [0,1)。
            # English: 1-exp(-x) preserves rank while bounding confidence in [0,1).
            normalized_confidence = 1.0 - np.exp(-np.maximum(prominences / scale, 0.0))
            candidate_scores.extend(normalized_confidence.tolist())
        merged, confidence = _merge_events(
            np.asarray(candidate_locations, dtype=np.int64),
            np.asarray(candidate_scores, dtype=np.float64),
            fs_hz,
        )
        raw_ppi, valid_mask = derive_ppi(merged, fs_hz)
        valid_ppi = raw_ppi[valid_mask]
        status = (
            ProcessingStatus.VALID
            if merged.size >= 5 and valid_ppi.size >= 4
            else ProcessingStatus.INSUFFICIENT
        )
        current = PeakResult(
            status,
            merged,
            confidence,
            int(polarity),
            raw_ppi,
            valid_mask,
            valid_ppi,
            valid_ppi.copy(),
            _polarity_score(merged, fs_hz),
            [],
            profile_id=profile_id,
        )
        if best is None or current.score > best.score:
            best = current
    assert best is not None
    return best


def _contiguous_valid_differences(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """只跨共享中间峰计算相邻差 / Only compare truly adjacent valid PPI."""

    if raw.size < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(raw)[mask[:-1] & mask[1:]]


def _longest_valid_run(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """取最长连续 valid PPI run / Return the longest continuous valid PPI run."""

    best = np.empty(0, dtype=np.float64)
    index = 0
    while index < mask.size:
        while index < mask.size and not mask[index]:
            index += 1
        stop = index
        while stop < mask.size and mask[stop]:
            stop += 1
        if stop - index > best.size:
            best = raw[index:stop]
        index = stop + 1
    return best


def compute_prv(
    peak_result: PeakResult,
    observation_duration_sec: float,
) -> HrvResult:
    """计算 HR 与分层 PPG-derived PRV / Compute HR and tiered PRV."""

    metrics: dict[str, Any] = {
        "hr_median_ppi_bpm": None,
        "hr_mean_instantaneous_bpm": None,
        "ppi_mean_ms": None,
        "ppi_median_ms": None,
        "sdnn_ms": None,
        "rmssd_ms": None,
        "sdsd_ms": None,
        "nn50": None,
        "pnn50_fraction": None,
        "lf_power_ms2": None,
        "hf_power_ms2": None,
        "lf_hf_ratio": None,
        "frequency_tier": None,
        "valid_ppi_coverage": 0.0,
    }
    valid_ppi = np.asarray(peak_result.valid_ppi_sec, dtype=np.float64)
    reasons: list[str] = []
    if peak_result.status == ProcessingStatus.INVALID:
        return HrvResult(
            ProcessingStatus.INVALID,
            metrics,
            ["PEAK_RESULT_INVALID", *peak_result.reason_codes],
            float(observation_duration_sec),
            0,
            source_peak_algorithm_id=peak_result.algorithm_id,
            source_profile_id=peak_result.profile_id,
        )
    if observation_duration_sec < 8.0:
        reasons.append("DURATION_LT_8S")
    if peak_result.peaks.size < 5:
        reasons.append("PEAK_COUNT_LT_5")
    if valid_ppi.size < 4:
        reasons.append("VALID_PPI_COUNT_LT_4")
    if reasons:
        return HrvResult(
            ProcessingStatus.INSUFFICIENT,
            metrics,
            reasons,
            float(observation_duration_sec),
            int(valid_ppi.size + 1 if valid_ppi.size else 0),
            source_peak_algorithm_id=peak_result.algorithm_id,
            source_profile_id=peak_result.profile_id,
        )

    metrics["hr_median_ppi_bpm"] = float(60.0 / np.median(valid_ppi))
    metrics["hr_mean_instantaneous_bpm"] = float(np.mean(60.0 / valid_ppi))
    metrics["ppi_mean_ms"] = float(np.mean(valid_ppi) * 1000.0)
    metrics["ppi_median_ms"] = float(np.median(valid_ppi) * 1000.0)
    metrics["valid_ppi_coverage"] = float(
        np.sum(valid_ppi) / max(float(observation_duration_sec), 1e-12)
    )
    if metrics["valid_ppi_coverage"] > 1.0 + 1e-9:
        return HrvResult(
            ProcessingStatus.INVALID,
            metrics,
            ["OBSERVATION_DURATION_INCONSISTENT"],
            float(observation_duration_sec),
            0,
            source_peak_algorithm_id=peak_result.algorithm_id,
            source_profile_id=peak_result.profile_id,
        )

    raw = np.asarray(peak_result.raw_ppi_sec, dtype=np.float64)
    mask = np.asarray(peak_result.ppi_valid_mask, dtype=bool)
    time_domain_coverage_ok = metrics["valid_ppi_coverage"] >= 0.80
    if observation_duration_sec >= 60.0 and valid_ppi.size >= 5 and time_domain_coverage_ok:
        ppi_ms = valid_ppi * 1000.0
        differences_ms = _contiguous_valid_differences(raw, mask) * 1000.0
        metrics["sdnn_ms"] = float(np.std(ppi_ms, ddof=1))
        if differences_ms.size:
            metrics["rmssd_ms"] = float(np.sqrt(np.mean(differences_ms**2)))
            metrics["sdsd_ms"] = (
                float(np.std(differences_ms, ddof=1))
                if differences_ms.size > 1
                else None
            )
            metrics["nn50"] = int(np.sum(np.abs(differences_ms) > 50.0))
            metrics["pnn50_fraction"] = float(
                np.mean(np.abs(differences_ms) > 50.0)
            )
    else:
        reasons.append(
            "TIME_DOMAIN_PRV_COVERAGE_LT_0P80"
            if observation_duration_sec >= 60.0 and not time_domain_coverage_ok
            else "TIME_DOMAIN_PRV_REQUIRES_60S"
        )

    continuous = _longest_valid_run(raw, mask)
    continuous_duration = float(np.sum(continuous))
    if continuous_duration >= 120.0 and continuous.size >= 10:
        beat_time = np.cumsum(continuous)
        grid = np.arange(beat_time[0], beat_time[-1], 0.25)
        if grid.size >= 16:
            tachogram = np.interp(grid, beat_time, continuous * 1000.0)
            tachogram = signal.detrend(tachogram, type="linear")
            frequencies, power = signal.welch(
                tachogram,
                fs=4.0,
                window="hann",
                nperseg=min(256, tachogram.size),
            )

            def band_power(low: float, high: float, include_high: bool = False) -> float:
                """积分一个 PRV 频段 / Integrate one PRV frequency band."""

                keep = (frequencies >= low) & (
                    frequencies <= high if include_high else frequencies < high
                )
                return (
                    float(np.trapz(power[keep], frequencies[keep]))
                    if np.any(keep)
                    else 0.0
                )

            lf = band_power(0.04, 0.15)
            hf = band_power(0.15, 0.40, include_high=True)
            metrics["lf_power_ms2"] = lf
            metrics["hf_power_ms2"] = hf
            metrics["lf_hf_ratio"] = None if hf <= 1e-12 else float(lf / hf)
            metrics["frequency_tier"] = (
                "confirmatory_300s"
                if continuous_duration >= 300.0
                else "exploratory_120s"
            )
    else:
        reasons.append("FREQUENCY_PRV_REQUIRES_CONTIGUOUS_120S")
    participating_peaks = np.zeros(mask.size + 1, dtype=bool)
    participating_peaks[:-1] |= mask
    participating_peaks[1:] |= mask
    return HrvResult(
        ProcessingStatus.PARTIAL if reasons else ProcessingStatus.VALID,
        metrics,
        reasons,
        float(observation_duration_sec),
        int(np.sum(participating_peaks)),
        source_peak_algorithm_id=peak_result.algorithm_id,
        source_profile_id=peak_result.profile_id,
    )


def _match_events(
    reference: np.ndarray, candidate: np.ndarray, tolerance_samples: int
) -> tuple[int, list[int]]:
    """单调一对一匹配 / Greedy monotonic one-to-one matching."""

    left = np.asarray(reference, dtype=np.int64)
    right = np.asarray(candidate, dtype=np.int64)
    i = 0
    j = 0
    differences: list[int] = []
    while i < left.size and j < right.size:
        delta = int(right[j] - left[i])
        if abs(delta) <= tolerance_samples:
            differences.append(delta)
            i += 1
            j += 1
        elif right[j] < left[i]:
            j += 1
        else:
            i += 1
    return len(differences), differences


def dual_channel_agreement(
    red: PeakResult, infrared: PeakResult, fs_hz: float
) -> dict[str, Any]:
    """报告多容差 RED/IR agreement / Report agreement without shifting peaks."""

    output: dict[str, Any] = {}
    for tolerance_ms in (20, 50, 100):
        tolerance = int(round(tolerance_ms * float(fs_hz) / 1000.0))
        matched, differences = _match_events(red.peaks, infrared.peaks, tolerance)
        precision = matched / infrared.peaks.size if infrared.peaks.size else 0.0
        recall = matched / red.peaks.size if red.peaks.size else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        output[f"f1_{tolerance_ms}ms"] = float(f1)
        output[f"matched_{tolerance_ms}ms"] = int(matched)
        if tolerance_ms == 50:
            output["median_signed_lag_ms"] = (
                float(np.median(differences) * 1000.0 / float(fs_hz))
                if differences
                else None
            )
    return output


def choose_primary_channel(
    red: PeakResult,
    infrared: PeakResult,
    *,
    red_sqi: float,
    infrared_sqi: float,
    fs_hz: float,
) -> dict[str, Any]:
    """由 SQI 选通道，平局 RED / Select by SQI with deterministic RED tie-break."""

    red_sqi_valid = bool(np.isfinite(red_sqi) and 0.0 <= float(red_sqi) <= 1.0)
    infrared_sqi_valid = bool(
        np.isfinite(infrared_sqi) and 0.0 <= float(infrared_sqi) <= 1.0
    )
    red_eligible = red.status == ProcessingStatus.VALID and red_sqi_valid
    infrared_eligible = infrared.status == ProcessingStatus.VALID and infrared_sqi_valid
    if not red_eligible and not infrared_eligible:
        selected: str | None = None
        reason = "no_valid_channel"
    elif red_eligible and not infrared_eligible:
        selected = "RED"
        reason = "only_red_valid"
    elif infrared_eligible and not red_eligible:
        selected = "IR"
        reason = "only_ir_valid"
    else:
        red_score = float(red_sqi) if np.isfinite(red_sqi) else float("-inf")
        ir_score = float(infrared_sqi) if np.isfinite(infrared_sqi) else float("-inf")
        selected = "RED" if red_score >= ir_score else "IR"
        reason = "sqi_tie_red_order" if red_score == ir_score else "higher_sqi"
    return {
        "selected_channel": selected,
        "selection_reason": reason,
        "red_sqi": float(red_sqi) if np.isfinite(red_sqi) else None,
        "ir_sqi": float(infrared_sqi) if np.isfinite(infrared_sqi) else None,
        "red_eligible": red_eligible,
        "ir_eligible": infrared_eligible,
        "red_sqi_valid": red_sqi_valid,
        "ir_sqi_valid": infrared_sqi_valid,
        "agreement": dual_channel_agreement(red, infrared, fs_hz),
        "consensus_peak_generation": False,
    }


def autocorrelation_periodicity(filtered_ppg: np.ndarray, fs_hz: float) -> float:
    """计算 30–200 bpm 自相关强度 / Autocorrelation periodicity strength."""

    values = np.asarray(filtered_ppg, dtype=np.float64).ravel()
    centered = values - np.mean(values)
    energy = float(np.dot(centered, centered))
    if values.size < 3 or energy <= 1e-12:
        return float("nan")
    correlation = signal.correlate(centered, centered, mode="full", method="fft")
    correlation = correlation[values.size - 1 :] / energy
    low = max(1, int(np.floor(PPI_MIN_SEC * float(fs_hz))))
    high = min(values.size - 1, int(np.ceil(PPI_MAX_SEC * float(fs_hz))))
    return float(np.max(correlation[low : high + 1])) if high >= low else float("nan")


def template_correlation(
    filtered_ppg: np.ndarray, peaks: np.ndarray, *, template_samples: int = 100
) -> float:
    """重采样周期并计算 median-template correlation / Beat-template correlation."""

    values = np.asarray(filtered_ppg, dtype=np.float64).ravel()
    locations = np.asarray(peaks, dtype=np.int64)
    beats: list[np.ndarray] = []
    for left, right in zip(locations[:-1], locations[1:]):
        if right - left < 3:
            continue
        source = np.linspace(0.0, 1.0, right - left, endpoint=False)
        target = np.linspace(0.0, 1.0, int(template_samples), endpoint=False)
        beat = np.interp(target, source, values[left:right])
        scale = float(np.std(beat))
        if scale > 1e-12:
            beats.append((beat - np.mean(beat)) / scale)
    if len(beats) < 3:
        return float("nan")
    matrix = np.vstack(beats)
    template = np.median(matrix, axis=0)
    scale = float(np.std(template))
    if scale <= 1e-12:
        return float("nan")
    template = (template - np.mean(template)) / scale
    return float(
        np.median([np.corrcoef(row, template)[0, 1] for row in matrix])
    )

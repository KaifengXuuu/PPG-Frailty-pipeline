"""仅 direct 的双波长 AC/DC、PI 与一致性 / Direct-only dual optical features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..contracts import PulseResult, SignalRoute
from .morphology import require_direct_route
from .views import CANONICAL_FS_HZ


@dataclass(frozen=True)
class OpticalFeatureResult:
    """逐搏 optical 值与全局一致性 / Beatwise optical values and global agreement."""

    beat_values: dict[str, np.ndarray]
    beat_validity: dict[str, np.ndarray]
    aggregate_values: dict[str, float]
    aggregate_validity: dict[str, bool]
    reasons: tuple[str, ...]


def _safe_ratio(numerator: float, denominator: float) -> float:
    """有限、非零分母比例 / Ratio with a finite non-zero denominator."""

    return float(numerator / denominator) if np.isfinite(numerator) and np.isfinite(denominator) and abs(denominator) > 1e-12 else float("nan")


def _normalized_xcorr(red: np.ndarray, infrared: np.ndarray, max_lag: int) -> tuple[float, int]:
    """计算限 lag 归一化互相关 / Compute normalized cross-correlation within a lag bound."""

    left = red - np.mean(red)
    right = infrared - np.mean(infrared)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return float("nan"), 0
    correlation = signal.correlate(left, right, mode="full", method="fft") / denominator
    lags = signal.correlation_lags(left.size, right.size, mode="full")
    selected = np.abs(lags) <= max_lag
    index = int(np.argmax(correlation[selected]))
    return float(correlation[selected][index]), int(lags[selected][index])


def extract_dual_optical(
    x_native: np.ndarray,
    x_filter: np.ndarray,
    pulse: PulseResult,
    *,
    route: SignalRoute,
    fs_hz: float = CANONICAL_FS_HZ,
) -> OpticalFeatureResult:
    """提取逐搏 AC/DC/PI/R 与 RED/IR 一致性 / Extract dual-wavelength descriptors."""

    require_direct_route(route)
    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("optical extraction requires exactly 400 Hz")
    native = np.asarray(x_native, dtype=np.float64)
    filtered = np.asarray(x_filter, dtype=np.float64)
    if native.ndim != 2 or native.shape[1] != 2 or filtered.shape != native.shape:
        raise ValueError("native/filter must share shape samples-by-[RED,IR]")
    if not np.isfinite(native).all() or not np.isfinite(filtered).all():
        raise ValueError("direct optical inputs must be finite")
    peaks = np.asarray(pulse.peaks, dtype=np.int64)
    accepted = np.asarray(pulse.accepted_peak_mask, dtype=bool)
    names = (
        "red_ac", "ir_ac", "red_dc", "ir_dc", "red_pi", "ir_pi",
        "red_ir_ac_ratio", "red_ir_dc_ratio", "ratio_of_ratios",
    )
    beat_values = {name: np.full(peaks.size, np.nan, dtype=np.float64) for name in names}
    beat_validity = {name: np.zeros(peaks.size, dtype=bool) for name in names}
    for ordinal in range(1, peaks.size - 1):
        if not accepted[ordinal]:
            continue
        peak = int(peaks[ordinal])
        left_bound = int((peaks[ordinal - 1] + peak) // 2)
        right_bound = int((peak + peaks[ordinal + 1]) // 2)
        if not (0 <= left_bound < peak < right_bound < native.shape[0]):
            continue
        # 中文：每个波长均在相同 beat 边界内寻找自身 valley，并把 valley-to-valley
        # 线性基线对齐到 accepted peak。AC 来自滤波波形的 peak-baseline；DC 来自
        # acquisition-preserving native 波形在同一 valley 索引处的线性基线。
        # English: Each wavelength finds its own valleys within the shared beat.
        # AC is filtered peak minus its valley baseline; DC is the aligned native
        # valley-to-valley baseline evaluated at the same accepted peak.
        polarity = -1.0 if "polarity=-1" in str(pulse.detector_version) else 1.0
        ac = np.full(2, np.nan, dtype=np.float64)
        dc = np.full(2, np.nan, dtype=np.float64)
        for channel in range(2):
            oriented = polarity * filtered[:, channel]
            left_valley = left_bound + int(np.argmin(oriented[left_bound : peak + 1]))
            right_valley = peak + int(np.argmin(oriented[peak : right_bound + 1]))
            if not left_valley < peak < right_valley:
                continue
            filtered_baseline = float(
                np.interp(
                    float(peak),
                    [float(left_valley), float(right_valley)],
                    [float(oriented[left_valley]), float(oriented[right_valley])],
                )
            )
            native_baseline = float(
                np.interp(
                    float(peak),
                    [float(left_valley), float(right_valley)],
                    [float(native[left_valley, channel]), float(native[right_valley, channel])],
                )
            )
            candidate_ac = float(oriented[peak] - filtered_baseline)
            if np.isfinite(candidate_ac) and candidate_ac > 0.0 and np.isfinite(native_baseline):
                ac[channel] = candidate_ac
                dc[channel] = native_baseline
        red_pi = _safe_ratio(float(ac[0]), abs(float(dc[0])))
        ir_pi = _safe_ratio(float(ac[1]), abs(float(dc[1])))
        red_ac_over_dc = _safe_ratio(float(ac[0]), float(dc[0]))
        ir_ac_over_dc = _safe_ratio(float(ac[1]), float(dc[1]))
        current = {
            "red_ac": float(ac[0]), "ir_ac": float(ac[1]),
            "red_dc": float(dc[0]), "ir_dc": float(dc[1]),
            "red_pi": red_pi, "ir_pi": ir_pi,
            "red_ir_ac_ratio": _safe_ratio(float(ac[0]), float(ac[1])),
            "red_ir_dc_ratio": _safe_ratio(float(dc[0]), float(dc[1])),
            "ratio_of_ratios": _safe_ratio(red_ac_over_dc, ir_ac_over_dc),
        }
        for name, value in current.items():
            beat_values[name][ordinal] = value
            beat_validity[name][ordinal] = bool(np.isfinite(value))

    aggregate: dict[str, float] = {}
    aggregate_validity: dict[str, bool] = {}
    for name in names:
        selected = beat_values[name][beat_validity[name]]
        key = f"{name}_median"
        aggregate[key] = float(np.median(selected)) if selected.size else float("nan")
        aggregate_validity[key] = bool(selected.size >= 3)
    red = filtered[:, 0]
    infrared = filtered[:, 1]
    zero_corr = float(np.corrcoef(red, infrared)[0, 1]) if np.std(red) > 0 and np.std(infrared) > 0 else float("nan")
    max_corr, lag = _normalized_xcorr(red, infrared, int(round(0.25 * fs_hz)))
    frequencies, coherence = signal.coherence(
        red, infrared, fs=fs_hz, nperseg=min(2048, red.size)
    )
    cardiac = (frequencies >= 0.5) & (frequencies <= 3.0)
    coherence_value = float(np.mean(coherence[cardiac])) if np.any(cardiac) else float("nan")
    global_values = {
        "red_ir_zero_lag_correlation": zero_corr,
        "red_ir_max_xcorr": max_corr,
        "red_ir_xcorr_lag_s": float(lag / fs_hz),
        "red_ir_cardiac_coherence": coherence_value,
    }
    aggregate.update(global_values)
    aggregate_validity.update({name: np.isfinite(value) for name, value in global_values.items()})
    reasons = () if any(aggregate_validity.values()) else ("dual_optical_unavailable",)
    return OpticalFeatureResult(
        beat_values=beat_values,
        beat_validity=beat_validity,
        aggregate_values=aggregate,
        aggregate_validity=aggregate_validity,
        reasons=reasons,
    )

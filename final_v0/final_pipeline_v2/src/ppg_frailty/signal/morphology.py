"""仅 direct/identity 可用的逐搏形态 / Beat morphology for direct/identity only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import PulseResult, SignalRoute
from .views import CANONICAL_FS_HZ


MORPHOLOGY_NAMES = (
    "amplitude", "width_half_s", "rise_s", "decay_s",
    "rise_slope_per_s", "decay_slope_per_s", "positive_area",
)


@dataclass(frozen=True)
class MorphologyResult:
    """逐搏数组、validity 与 robust 汇总 / Beat arrays, validity, and robust summaries."""

    beat_values: dict[str, np.ndarray]
    beat_validity: dict[str, np.ndarray]
    aggregate_values: dict[str, float]
    aggregate_validity: dict[str, bool]
    reasons: tuple[str, ...]


def require_direct_route(route: SignalRoute) -> None:
    """在读取波形前阻止 rate-only 形态调用 / Block rate-only morphology before access."""

    if route not in {SignalRoute.DIRECT, SignalRoute.IDENTITY}:
        raise PermissionError(
            "morphology is forbidden for non-identity x_ar; Q_morph is not_applicable"
        )


def _polarity(pulse: PulseResult) -> float:
    """从版本 provenance 读取检测极性 / Read detector polarity from provenance."""

    marker = "polarity=-1"
    return -1.0 if marker in str(pulse.detector_version) else 1.0


def _crossing_time(
    samples: np.ndarray, values: np.ndarray, target: float, *, first: bool
) -> float | None:
    """线性插值阈值交点 / Linearly interpolate a threshold crossing."""

    if samples.size < 2:
        return None
    products = (values[:-1] - target) * (values[1:] - target)
    candidates = np.flatnonzero(products <= 0.0)
    if not candidates.size:
        return None
    index = int(candidates[0] if first else candidates[-1])
    left, right = float(values[index]), float(values[index + 1])
    fraction = 0.0 if right == left else (target - left) / (right - left)
    return float(samples[index] + np.clip(fraction, 0.0, 1.0))


def extract_morphology(
    x_filter: np.ndarray,
    pulse: PulseResult,
    *,
    route: SignalRoute,
    fs_hz: float = CANONICAL_FS_HZ,
) -> MorphologyResult:
    """提取幅值、宽度、rise/decay、坡度和面积 / Extract beat morphology.

    每搏用左右 valley 间的线性 baseline，减少 residual drift 对 amplitude/area 的偏差。
    Each beat uses a linear valley-to-valley baseline to reduce residual-drift bias.
    """

    require_direct_route(route)
    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("morphology requires the 400 Hz acquisition grid")
    matrix = np.asarray(x_filter, dtype=np.float64)
    if matrix.ndim == 1:
        values = matrix
    elif matrix.ndim == 2 and matrix.shape[1] == 2:
        channel = 0 if pulse.wavelength.upper() == "RED" else 1
        values = matrix[:, channel]
    else:
        raise ValueError("x_filter must be one channel or samples-by-[RED,IR]")
    if not np.isfinite(values).all():
        raise ValueError("morphology input must be finite")
    peaks = np.asarray(pulse.peaks, dtype=np.int64)
    accepted = np.asarray(pulse.accepted_peak_mask, dtype=bool)
    if peaks.shape != accepted.shape:
        raise ValueError("peak and accepted masks must share shape")
    oriented = values * _polarity(pulse)
    beat_values = {name: np.full(peaks.size, np.nan, dtype=np.float64) for name in MORPHOLOGY_NAMES}
    beat_validity = {name: np.zeros(peaks.size, dtype=bool) for name in MORPHOLOGY_NAMES}
    reasons: list[str] = []

    for ordinal in range(1, peaks.size - 1):
        if not accepted[ordinal]:
            continue
        peak = int(peaks[ordinal])
        left_bound = int((peaks[ordinal - 1] + peak) // 2)
        right_bound = int((peak + peaks[ordinal + 1]) // 2)
        if not (0 <= left_bound < peak < right_bound < oriented.size):
            continue
        left_valley = left_bound + int(np.argmin(oriented[left_bound : peak + 1]))
        right_valley = peak + int(np.argmin(oriented[peak : right_bound + 1]))
        if not left_valley < peak < right_valley:
            continue
        segment_samples = np.arange(left_valley, right_valley + 1, dtype=np.float64)
        baseline = np.interp(
            segment_samples,
            [float(left_valley), float(right_valley)],
            [float(oriented[left_valley]), float(oriented[right_valley])],
        )
        detrended = oriented[left_valley : right_valley + 1] - baseline
        peak_offset = peak - left_valley
        amplitude = float(detrended[peak_offset])
        if not np.isfinite(amplitude) or amplitude <= 0.0:
            continue
        half = amplitude / 2.0
        left_cross = _crossing_time(
            segment_samples[: peak_offset + 1], detrended[: peak_offset + 1], half, first=False
        )
        right_cross = _crossing_time(
            segment_samples[peak_offset:], detrended[peak_offset:], half, first=True
        )
        rise = (peak - left_valley) / fs_hz
        decay = (right_valley - peak) / fs_hz
        area = float(np.trapz(np.maximum(detrended, 0.0), dx=1.0 / fs_hz))
        current = {
            "amplitude": amplitude,
            "width_half_s": (
                float((right_cross - left_cross) / fs_hz)
                if left_cross is not None and right_cross is not None and right_cross > left_cross
                else float("nan")
            ),
            "rise_s": float(rise),
            "decay_s": float(decay),
            "rise_slope_per_s": float(amplitude / rise) if rise > 0.0 else float("nan"),
            "decay_slope_per_s": float(-amplitude / decay) if decay > 0.0 else float("nan"),
            "positive_area": area,
        }
        for name, value in current.items():
            beat_values[name][ordinal] = value
            beat_validity[name][ordinal] = bool(np.isfinite(value))

    aggregate: dict[str, float] = {}
    aggregate_validity: dict[str, bool] = {}
    for name in MORPHOLOGY_NAMES:
        selected = beat_values[name][beat_validity[name]]
        for statistic, value in (
            ("median", float(np.median(selected)) if selected.size else float("nan")),
            (
                "mad",
                float(np.median(np.abs(selected - np.median(selected))))
                if selected.size
                else float("nan"),
            ),
        ):
            key = f"{name}_{statistic}"
            aggregate[key] = value
            aggregate_validity[key] = bool(selected.size >= 3 and np.isfinite(value))
    if not any(aggregate_validity.values()):
        reasons.append("insufficient_valid_beats_for_morphology")
    return MorphologyResult(
        beat_values=beat_values,
        beat_validity=beat_validity,
        aggregate_values=aggregate,
        aggregate_validity=aggregate_validity,
        reasons=tuple(reasons),
    )

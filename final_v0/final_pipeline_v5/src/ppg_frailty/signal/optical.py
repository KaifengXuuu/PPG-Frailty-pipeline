"""Canonical paired dual-wavelength AC/DC, PI, and waveform agreement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contracts import PulseResult, SignalRoute
from ..peaks.pairing import (
    BeatPairAudit,
    BeatPairingResult,
    pair_dual_wavelength_beats,
)
from .morphology import require_direct_route
from .views import CANONICAL_FS_HZ


OPTICAL_SCHEMA_VERSION = "dual_optical_paired_absolute_dc_v2"
OPTICAL_RATIO_EPSILON = 1e-12
OPTICAL_MINIMUM_PAIRED_BEATS = 3
XCORR_MAX_LAG_SECONDS = 0.5

@dataclass(frozen=True)
class OpticalBeatAudit:
    """Pairing plus wavelength-local valley eligibility for one audit row."""

    pairing: BeatPairAudit
    red_left_valley_sample: int | None
    red_right_valley_sample: int | None
    ir_left_valley_sample: int | None
    ir_right_valley_sample: int | None
    optical_valid: bool
    reason_codes: tuple[str, ...]

@dataclass(frozen=True)
class OpticalFeatureResult:
    """Paired-beat values, canonical aggregates, and non-predictor diagnostics."""

    beat_values: dict[str, np.ndarray]
    beat_validity: dict[str, np.ndarray]
    aggregate_values: dict[str, float]
    aggregate_validity: dict[str, bool]
    pairing: BeatPairingResult
    beat_audit: tuple[OpticalBeatAudit, ...]
    diagnostics: dict[str, Any]
    reasons: tuple[str, ...]
    schema_version: str = OPTICAL_SCHEMA_VERSION

def _safe_ratio(numerator: float, denominator: float) -> float:
    """Apply the existing finite/non-zero denominator policy plus epsilon."""

    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= OPTICAL_RATIO_EPSILON:
        return float("nan")
    return float(numerator / (denominator + OPTICAL_RATIO_EPSILON))

def _safe_absolute_denominator_ratio(numerator: float, denominator: float) -> float:
    """Return numerator/(abs(denominator)+epsilon) after validity checks."""

    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= OPTICAL_RATIO_EPSILON:
        return float("nan")
    return float(numerator / (abs(denominator) + OPTICAL_RATIO_EPSILON))

def _standardize(values: np.ndarray) -> np.ndarray:
    """Population-z-standardize one finite analysis channel."""

    centered = values - float(np.mean(values))
    scale = float(np.std(centered, ddof=0))
    if not np.isfinite(scale) or scale <= OPTICAL_RATIO_EPSILON:
        raise ValueError("waveform agreement requires non-constant channels")
    return centered / scale

def _standardized_waveform_agreement(
    red: np.ndarray,
    infrared: np.ndarray,
    *,
    max_lag_samples: int,
) -> tuple[float, float, int | None]:
    """Return zero-lag rho and bounded normalized xcorr.

    Positive lag means the IR waveform occurs later than RED. The full channels
    are standardized once, then every overlapping lag uses its own vector norms.
    """

    if red.ndim != 1 or infrared.ndim != 1 or red.shape != infrared.shape:
        raise ValueError("waveform agreement requires aligned one-dimensional channels")
    if max_lag_samples < 0 or max_lag_samples >= red.size:
        raise ValueError("max_lag_samples must be within the aligned recording")
    try:
        standardized_red = _standardize(red)
        standardized_ir = _standardize(infrared)
    except ValueError:
        return float("nan"), float("nan"), None
    zero_lag = float(np.dot(standardized_red, standardized_ir) / standardized_red.size)
    lags = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=np.int64)
    correlations = np.full(lags.size, np.nan, dtype=np.float64)
    for index, lag in enumerate(lags.tolist()):
        if lag >= 0:
            left = standardized_red[: standardized_red.size - lag]
            right = standardized_ir[lag:]
        else:
            left = standardized_red[-lag:]
            right = standardized_ir[: standardized_ir.size + lag]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator > OPTICAL_RATIO_EPSILON:
            correlations[index] = float(np.dot(left, right) / denominator)
    if not np.isfinite(correlations).any():
        return zero_lag, float("nan"), None
    maximum = float(np.nanmax(correlations))
    tied = np.flatnonzero(np.isfinite(correlations) & (correlations == maximum))
    selected_index = min(
        tied.tolist(),
        key=lambda index: (abs(int(lags[index])), int(lags[index])),
    )
    return zero_lag, float(correlations[selected_index]), int(lags[selected_index])

def _pulse_positions_by_ordinal(pulse: PulseResult) -> dict[int, int]:
    """Map persisted peak ordinals back to PulseResult vector positions."""

    ordinals = np.asarray(pulse.peak_ordinals, dtype=np.int64)
    peaks = np.asarray(pulse.peaks, dtype=np.int64)
    if ordinals.shape != peaks.shape or np.unique(ordinals).size != ordinals.size:
        raise ValueError("PulseResult peak_ordinals must be aligned and unique")
    return {int(ordinal): position for position, ordinal in enumerate(ordinals)}

def _wavelength_local_ac_dc(
    *,
    native: np.ndarray,
    filtered: np.ndarray,
    pulse: PulseResult,
    peak_ordinal: int,
) -> tuple[float, float, int | None, int | None, str | None]:
    """Calculate AC/DC at one wavelength's own peak and own boundary valleys."""

    ordinal_positions = _pulse_positions_by_ordinal(pulse)
    if peak_ordinal not in ordinal_positions:
        return float("nan"), float("nan"), None, None, "peak_ordinal_not_found"
    position = ordinal_positions[peak_ordinal]
    peaks = np.asarray(pulse.peaks, dtype=np.int64)
    accepted = np.asarray(pulse.accepted_peak_mask, dtype=bool)
    if not accepted[position]:
        return float("nan"), float("nan"), None, None, "paired_peak_not_accepted"
    if position == 0 or position == peaks.size - 1:
        return float("nan"), float("nan"), None, None, "paired_peak_has_no_boundary_neighbors"
    peak = int(peaks[position])
    left_bound = int((int(peaks[position - 1]) + peak) // 2)
    right_bound = int((peak + int(peaks[position + 1])) // 2)
    if not (0 <= left_bound < peak < right_bound < filtered.size):
        return float("nan"), float("nan"), None, None, "invalid_local_beat_bounds"
    polarity = int(pulse.selected_polarity)
    if polarity not in {-1, 1}:
        raise ValueError("PulseResult selected_polarity must be exactly -1 or +1")
    oriented = float(polarity) * filtered
    left_valley = left_bound + int(np.argmin(oriented[left_bound : peak + 1]))
    right_valley = peak + int(np.argmin(oriented[peak : right_bound + 1]))
    if not left_valley < peak < right_valley:
        return (
            float("nan"),
            float("nan"),
            left_valley,
            right_valley,
            "invalid_local_boundary_valleys",
        )
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
            [float(native[left_valley]), float(native[right_valley])],
        )
    )
    ac = float(oriented[peak] - filtered_baseline)
    if not np.isfinite(ac) or ac <= 0.0 or not np.isfinite(native_baseline):
        return (
            float("nan"),
            float("nan"),
            left_valley,
            right_valley,
            "nonpositive_ac_or_nonfinite_dc",
        )
    return ac, native_baseline, left_valley, right_valley, None

def extract_dual_optical(
    x_native: np.ndarray,
    x_filter: np.ndarray,
    pulses: Mapping[str, PulseResult],
    *,
    route: SignalRoute,
    fs_hz: float = CANONICAL_FS_HZ,
) -> OpticalFeatureResult:
    """Extract canonical optical features from independently detected RED/IR beats."""

    require_direct_route(route)
    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("optical extraction requires exactly 400 Hz")
    native = np.asarray(x_native, dtype=np.float64)
    filtered = np.asarray(x_filter, dtype=np.float64)
    if native.ndim != 2 or native.shape[1] != 2 or filtered.shape != native.shape:
        raise ValueError("native/filter must share shape samples-by-[RED,IR]")
    if not np.isfinite(native).all() or not np.isfinite(filtered).all():
        raise ValueError("direct optical inputs must be finite")
    if not isinstance(pulses, Mapping):
        raise TypeError("optical extraction requires independent RED and IR PulseResults")
    pairing = pair_dual_wavelength_beats(pulses, fs_hz=fs_hz)
    normalized_pulses = {str(name).upper(): pulse for name, pulse in pulses.items()}
    if any(SignalRoute(pulse.source_route) is not route for pulse in normalized_pulses.values()):
        raise ValueError("optical route disagrees with RED/IR PulseResult source route")

    names = (
        "red_ac",
        "ir_ac",
        "red_dc",
        "ir_dc",
        "red_pi",
        "ir_pi",
        "red_ir_ac_ratio",
        "red_ir_dc_ratio",
        "ratio_of_ratios",
    )
    beat_values = {name: np.full(len(pairing.rows), np.nan, dtype=np.float64) for name in names}
    beat_validity = {name: np.zeros(len(pairing.rows), dtype=bool) for name in names}
    beat_audit: list[OpticalBeatAudit] = []
    common_valid_positions: list[int] = []

    for index, pair in enumerate(pairing.rows):
        if not pair.pair_valid:
            beat_audit.append(
                OpticalBeatAudit(
                    pairing=pair,
                    red_left_valley_sample=None,
                    red_right_valley_sample=None,
                    ir_left_valley_sample=None,
                    ir_right_valley_sample=None,
                    optical_valid=False,
                    reason_codes=pair.reason_codes,
                )
            )
            continue
        red = _wavelength_local_ac_dc(
            native=native[:, 0],
            filtered=filtered[:, 0],
            pulse=normalized_pulses["RED"],
            peak_ordinal=int(pair.red_peak_ordinal),
        )
        infrared = _wavelength_local_ac_dc(
            native=native[:, 1],
            filtered=filtered[:, 1],
            pulse=normalized_pulses["IR"],
            peak_ordinal=int(pair.ir_peak_ordinal),
        )
        red_ac, red_dc, red_left, red_right, red_reason = red
        ir_ac, ir_dc, ir_left, ir_right, ir_reason = infrared
        optical_valid = red_reason is None and ir_reason is None
        reason_codes = list(pair.reason_codes)
        if red_reason is not None:
            reason_codes.append(f"red:{red_reason}")
        if ir_reason is not None:
            reason_codes.append(f"ir:{ir_reason}")
        if optical_valid:
            common_valid_positions.append(index)
            red_pi = _safe_absolute_denominator_ratio(red_ac, red_dc)
            ir_pi = _safe_absolute_denominator_ratio(ir_ac, ir_dc)
            ratio_of_ratios = (
                float(red_pi / ir_pi)
                if (np.isfinite(red_pi) and np.isfinite(ir_pi) and abs(ir_pi) > OPTICAL_RATIO_EPSILON)
                else float("nan")
            )
            current = {
                "red_ac": red_ac,
                "ir_ac": ir_ac,
                "red_dc": red_dc,
                "ir_dc": ir_dc,
                "red_pi": red_pi,
                "ir_pi": ir_pi,
                "red_ir_ac_ratio": _safe_ratio(red_ac, ir_ac),
                "red_ir_dc_ratio": _safe_absolute_denominator_ratio(abs(red_dc), ir_dc),
                "ratio_of_ratios": ratio_of_ratios,
            }
            for name, value in current.items():
                beat_values[name][index] = value
                beat_validity[name][index] = bool(np.isfinite(value))
        beat_audit.append(
            OpticalBeatAudit(
                pairing=pair,
                red_left_valley_sample=red_left,
                red_right_valley_sample=red_right,
                ir_left_valley_sample=ir_left,
                ir_right_valley_sample=ir_right,
                optical_valid=optical_valid,
                reason_codes=tuple(reason_codes),
            )
        )

    common = np.asarray(common_valid_positions, dtype=np.int64)
    support_valid = common.size >= OPTICAL_MINIMUM_PAIRED_BEATS
    recording_medians = {
        name: (float(np.median(beat_values[name][common])) if common.size else float("nan"))
        for name in ("red_ac", "ir_ac", "red_dc", "ir_dc")
    }
    aggregate = {f"{name}_median": value for name, value in recording_medians.items()}
    aggregate_validity = {
        f"{name}_median": bool(support_valid and np.isfinite(value)) for name, value in recording_medians.items()
    }
    red_ac = recording_medians["red_ac"]
    ir_ac = recording_medians["ir_ac"]
    red_dc = recording_medians["red_dc"]
    ir_dc = recording_medians["ir_dc"]
    derived_ratios = {
        "red_pi_median": _safe_absolute_denominator_ratio(red_ac, red_dc),
        "ir_pi_median": _safe_absolute_denominator_ratio(ir_ac, ir_dc),
        "red_ir_ac_ratio_median": _safe_ratio(red_ac, ir_ac),
        "red_ir_dc_ratio_median": _safe_absolute_denominator_ratio(abs(red_dc), ir_dc),
    }
    red_pi = derived_ratios["red_pi_median"]
    ir_pi = derived_ratios["ir_pi_median"]
    derived_ratios["ratio_of_ratios_median"] = (
        float(red_pi / ir_pi)
        if (np.isfinite(red_pi) and np.isfinite(ir_pi) and abs(ir_pi) > OPTICAL_RATIO_EPSILON)
        else float("nan")
    )
    aggregate.update(derived_ratios)
    aggregate_validity.update(
        {name: bool(support_valid and np.isfinite(value)) for name, value in derived_ratios.items()}
    )

    max_lag_samples = int(round(XCORR_MAX_LAG_SECONDS * fs_hz))
    zero_corr, max_corr, lag = _standardized_waveform_agreement(
        filtered[:, 0],
        filtered[:, 1],
        max_lag_samples=max_lag_samples,
    )
    waveform_values = {
        "red_ir_zero_lag_correlation": zero_corr,
        "red_ir_max_xcorr": max_corr,
        "red_ir_xcorr_lag_s": (float(lag / fs_hz) if lag is not None else float("nan")),
    }
    aggregate.update(waveform_values)
    aggregate_validity.update({name: bool(np.isfinite(value)) for name, value in waveform_values.items()})
    diagnostics: dict[str, Any] = {
        "affects_prediction": False,
        "beatwise_ratios_affect_prediction": False,
        "canonical_ratios_from_recording_median_ac_dc": True,
        "common_paired_valid_beat_count": int(common.size),
        "waveform_agreement": {
            "standardization": "per_channel_population_zscore",
            "normalization": "overlap_vector_norm_by_lag",
            "lag_sign": "positive_when_ir_occurs_after_red",
            "max_lag_samples": max_lag_samples,
            "max_lag_seconds": XCORR_MAX_LAG_SECONDS,
            "search_bounds_inclusive_samples": (
                -max_lag_samples,
                max_lag_samples,
            ),
            "tie_break": "highest_rho_then_smallest_abs_lag_then_smallest_signed_lag",
        },
        "coherence": {
            "computed": False,
            "affects_prediction": False,
            "reason": "not_in_thesis_table_8_contract",
        },
    }
    reasons = () if any(aggregate_validity.values()) else ("dual_optical_unavailable",)
    return OpticalFeatureResult(
        beat_values=beat_values,
        beat_validity=beat_validity,
        aggregate_values=aggregate,
        aggregate_validity=aggregate_validity,
        pairing=pairing,
        beat_audit=tuple(beat_audit),
        diagnostics=diagnostics,
        reasons=reasons,
    )


__all__ = [
    "OPTICAL_MINIMUM_PAIRED_BEATS",
    "OPTICAL_RATIO_EPSILON",
    "OPTICAL_SCHEMA_VERSION",
    "OpticalBeatAudit",
    "OpticalFeatureResult",
    "XCORR_MAX_LAG_SECONDS",
    "extract_dual_optical",
]

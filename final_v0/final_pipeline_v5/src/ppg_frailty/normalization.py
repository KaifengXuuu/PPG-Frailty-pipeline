"""Runtime-selectable raw PPG and IMU normalization strategies.

The effective mapping is deliberately small and JSON serializable so the same
resolved values can be validated, hashed, executed, and written to provenance.
Historical names remain aliases; they never select the separate motion tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

PPG_ROBUST = "per_window_robust"
PPG_STANDARD_ZSCORE = "per_window_standard_zscore"
PPG_NONE = "none"

IMU_ROBUST = "outer_train_robust"
IMU_MEAN_STD = "outer_train_mean_std"
IMU_NONE = "none"

FALLBACK_STANDARD_DEVIATION = "standard_deviation_then_finite_one"
FALLBACK_MAD = "median_absolute_deviation_then_finite_one"
FALLBACK_ONE = "finite_one"

_PPG_ALIASES = {
    PPG_ROBUST: PPG_ROBUST,
    "robust": PPG_ROBUST,
    "per_window_median_iqr": PPG_ROBUST,
    "per_window_median_iqr_over_1p349_sd_finite": PPG_ROBUST,
    PPG_STANDARD_ZSCORE: PPG_STANDARD_ZSCORE,
    "standard_zscore": PPG_STANDARD_ZSCORE,
    "per_window_mean_std": PPG_STANDARD_ZSCORE,
    PPG_NONE: PPG_NONE,
    "identity": PPG_NONE,
}

_IMU_ALIASES = {
    IMU_ROBUST: IMU_ROBUST,
    "robust": IMU_ROBUST,
    "outer_train_fold_robust_scaler": IMU_ROBUST,
    ("outer_training_participant_only_median_iqr_over_1p349_"
     "population_sd_then_one_axes6"): IMU_ROBUST,
    "outer_training_participant_only_robust_scaler_axes6": IMU_ROBUST,
    IMU_MEAN_STD: IMU_MEAN_STD,
    "standard_zscore": IMU_MEAN_STD,
    "outer_training_participant_only_mean_std_axes6": IMU_MEAN_STD,
    IMU_NONE: IMU_NONE,
    "identity": IMU_NONE,
}

_FALLBACK_ALIASES = {
    FALLBACK_STANDARD_DEVIATION: FALLBACK_STANDARD_DEVIATION,
    "standard_deviation": FALLBACK_STANDARD_DEVIATION,
    FALLBACK_MAD: FALLBACK_MAD,
    "median_absolute_deviation_then_one": FALLBACK_MAD,
    "mad_then_one": FALLBACK_MAD,
    FALLBACK_ONE: FALLBACK_ONE,
    "one": FALLBACK_ONE,
}


def _strategy(value: Any, aliases: Mapping[str, str], *, name: str) -> str:
    key = str(value).strip().lower()
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"{name} must select one of {sorted(set(aliases.values()))}; "
                         "motion-derived channels are a separate model input module") from exc


@dataclass(frozen=True)
class RawNormalizationConfig:
    """Effective parameters for the isolated DL-window tensor.

    ``raw_ppg`` is the compatibility name for the per-window transform applied
    to all eight DL channels. ``raw_imu`` is only an optional legacy/ablation
    post-transform; ordinary execution does not alter physical IMU views.
    """

    raw_ppg: str = PPG_ROBUST
    raw_imu: str = IMU_NONE
    iqr_fallback: str = FALLBACK_STANDARD_DEVIATION
    clip_after_scale: tuple[float, float] | None = (-8.0, 8.0)
    robust_iqr_divisor: float = 1.349
    mad_consistency_divisor: float = 0.6744897501960817
    scale_epsilon: float = 1e-8
    standard_ddof: int = 0

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
    ) -> "RawNormalizationConfig":
        """Resolve aliases and omitted values to one hashable effective form."""

        if mapping is None:
            declared: dict[str, Any] = {}
        elif isinstance(mapping, Mapping) and all(isinstance(key, str) for key in mapping):
            declared = dict(mapping)
        else:
            raise ValueError("signal.normalization must be a string-keyed mapping")
        allowed = {
            "raw_ppg",
            "raw_imu",
            "iqr_fallback",
            "clip_after_scale",
            "robust_iqr_divisor",
            "mad_consistency_divisor",
            "scale_epsilon",
            "standard_ddof",
        }
        unknown = sorted(set(declared) - allowed)
        if unknown:
            raise ValueError(f"signal.normalization contains unknown fields: {unknown}")
        defaults = cls()
        clip_raw = declared.get("clip_after_scale", defaults.clip_after_scale)
        if clip_raw is None:
            clip: tuple[float, float] | None = None
        elif isinstance(clip_raw, (str, bytes)) or not isinstance(clip_raw, (list, tuple)) or len(clip_raw) != 2:
            raise ValueError("signal.normalization.clip_after_scale must be null or two bounds")
        else:
            if any(isinstance(value, bool) for value in clip_raw):
                raise ValueError("signal.normalization.clip_after_scale bounds must be numeric")
            try:
                clip = (float(clip_raw[0]), float(clip_raw[1]))
            except (TypeError, ValueError) as exc:
                raise ValueError("signal.normalization.clip_after_scale bounds must be numeric") from exc
            if not np.isfinite(clip).all() or not clip[0] < clip[1]:
                raise ValueError("signal.normalization.clip_after_scale bounds must be finite "
                                 "and strictly increasing")
        scalar_values = (
            declared.get("robust_iqr_divisor", defaults.robust_iqr_divisor),
            declared.get("mad_consistency_divisor", defaults.mad_consistency_divisor),
            declared.get("scale_epsilon", defaults.scale_epsilon),
        )
        if any(isinstance(value, bool) for value in scalar_values):
            raise ValueError("normalization scale parameters must be numeric")
        try:
            iqr_divisor = float(scalar_values[0])
            mad_divisor = float(scalar_values[1])
            epsilon = float(scalar_values[2])
        except (TypeError, ValueError) as exc:
            raise ValueError("normalization scale parameters must be numeric") from exc
        if not np.isfinite([iqr_divisor, mad_divisor, epsilon]).all() or not (iqr_divisor > 0.0 and mad_divisor > 0.0
                                                                              and epsilon > 0.0):
            raise ValueError("normalization scale parameters must be finite and positive")
        ddof = declared.get("standard_ddof", defaults.standard_ddof)
        if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or int(ddof) < 0:
            raise ValueError("signal.normalization.standard_ddof must be a non-negative integer")
        result = cls(
            raw_ppg=_strategy(
                declared.get("raw_ppg", defaults.raw_ppg),
                _PPG_ALIASES,
                name="signal.normalization.raw_ppg",
            ),
            raw_imu=_strategy(
                declared.get("raw_imu", defaults.raw_imu),
                _IMU_ALIASES,
                name="signal.normalization.raw_imu",
            ),
            iqr_fallback=_strategy(
                declared.get("iqr_fallback", defaults.iqr_fallback),
                _FALLBACK_ALIASES,
                name="signal.normalization.iqr_fallback",
            ),
            clip_after_scale=clip,
            robust_iqr_divisor=iqr_divisor,
            mad_consistency_divisor=mad_divisor,
            scale_epsilon=epsilon,
            standard_ddof=int(ddof),
        )
        any_robust = result.raw_ppg == PPG_ROBUST or result.raw_imu == IMU_ROBUST
        consumes_standard_ddof = (result.raw_ppg == PPG_STANDARD_ZSCORE or result.raw_imu == IMU_MEAN_STD
                                  or (any_robust and result.iqr_fallback == FALLBACK_STANDARD_DEVIATION))
        consumed = {
            "iqr_fallback": any_robust,
            "clip_after_scale": result.raw_ppg != PPG_NONE,
            "robust_iqr_divisor": any_robust,
            "mad_consistency_divisor": (any_robust and result.iqr_fallback == FALLBACK_MAD),
            "scale_epsilon": (result.raw_ppg != PPG_NONE or result.raw_imu != IMU_NONE),
            "standard_ddof": consumes_standard_ddof,
        }
        resolved_values = {
            "iqr_fallback": result.iqr_fallback,
            "clip_after_scale": result.clip_after_scale,
            "robust_iqr_divisor": result.robust_iqr_divisor,
            "mad_consistency_divisor": result.mad_consistency_divisor,
            "scale_epsilon": result.scale_epsilon,
            "standard_ddof": result.standard_ddof,
        }
        default_values = {name: getattr(defaults, name) for name in resolved_values}
        for name, is_consumed in consumed.items():
            if name in declared and not is_consumed and resolved_values[name] != default_values[name]:
                raise ValueError(f"signal.normalization.{name} has no runtime consumer for "
                                 f"raw_ppg={result.raw_ppg}, raw_imu={result.raw_imu}")
        return result

    def to_mapping(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible form used for hashing."""

        return {
            "raw_ppg":
            self.raw_ppg,
            "raw_imu":
            self.raw_imu,
            "iqr_fallback":
            self.iqr_fallback,
            "clip_after_scale":
            (None if self.clip_after_scale is None else [float(value) for value in self.clip_after_scale]),
            "robust_iqr_divisor":
            float(self.robust_iqr_divisor),
            "mad_consistency_divisor":
            float(self.mad_consistency_divisor),
            "scale_epsilon":
            float(self.scale_epsilon),
            "standard_ddof":
            int(self.standard_ddof),
        }


__all__ = [
    "FALLBACK_MAD",
    "FALLBACK_ONE",
    "FALLBACK_STANDARD_DEVIATION",
    "IMU_MEAN_STD",
    "IMU_NONE",
    "IMU_ROBUST",
    "PPG_NONE",
    "PPG_ROBUST",
    "PPG_STANDARD_ZSCORE",
    "RawNormalizationConfig",
]

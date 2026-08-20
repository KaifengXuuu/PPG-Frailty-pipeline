"""Endpoint-aware SQI：Q_rate 与 Q_morph / Endpoint-aware signal quality.

组件覆盖 cardiac concentration、自相关、模板相关、偏度、峰度、归一化谱熵、
完整 PPI 合理性、RED/IR 一致性、motion energy、coverage/flatline。

Components cover cardiac concentration, autocorrelation, template correlation,
skewness, kurtosis, normalized spectral entropy, full PPI plausibility, RED/IR
agreement, motion energy, coverage, and flatline evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import signal, stats

from ..contracts import (
    PulseResult,
    QualityComponent,
    QualityEndpoint,
    QualityResult,
    QualityState,
    SignalRoute,
)
from ..provenance import assert_training_only
from .views import CANONICAL_FS_HZ, CanonicalSignalViews


RATE_COMPONENT_NAMES = (
    "cardiac_concentration",
    "autocorrelation_periodicity",
    "normalized_spectral_entropy",
    "peak_density_bpm",
    "ppi_physiological_fraction",
    "ppi_stability",
    "red_ir_agreement",
    "motion_energy_rms",
    "nonflat_scale",
    "source_coverage",
    "flatline",
    "clipping",
    "saturation",
    "long_gap",
)
MORPH_COMPONENT_NAMES = (
    "template_correlation",
    "skewness",
    "pearson_kurtosis",
    "red_ir_agreement",
    "cardiac_concentration",
    "nonflat_scale",
    "source_coverage",
    "flatline",
    "clipping",
    "saturation",
    "long_gap",
)


def _default_rate_component_weights() -> dict[str, float]:
    return {
        "cardiac_concentration": 0.20,
        "autocorrelation_periodicity": 0.15,
        "normalized_spectral_entropy": 0.10,
        "peak_density_bpm": 0.08,
        "ppi_physiological_fraction": 0.15,
        "ppi_stability": 0.12,
        "red_ir_agreement": 0.08,
        "motion_energy_rms": 0.05,
        "nonflat_scale": 0.02,
        "source_coverage": 0.04,
        "flatline": 0.02,
        "clipping": 0.015,
        "saturation": 0.015,
        "long_gap": 0.01,
    }


def _default_morph_component_weights() -> dict[str, float]:
    return {
        "template_correlation": 0.30,
        "skewness": 0.08,
        "pearson_kurtosis": 0.08,
        "red_ir_agreement": 0.18,
        "cardiac_concentration": 0.16,
        "nonflat_scale": 0.04,
        "source_coverage": 0.12,
        "flatline": 0.03,
        "clipping": 0.02,
        "saturation": 0.02,
        "long_gap": 0.02,
    }


def _value_or_default(
    mapping: Mapping[str, object], name: str, default: Any
) -> Any:
    value = mapping.get(name, default)
    return default if value is None else value


def _pair(
    mapping: Mapping[str, object],
    name: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    raw = _value_or_default(mapping, name, default)
    if (
        isinstance(raw, (str, bytes))
        or not isinstance(raw, (list, tuple))
        or len(raw) != 2
    ):
        raise ValueError(f"quality.{name} must contain two values")
    return float(raw[0]), float(raw[1])


def _weights(
    mapping: Mapping[str, object],
    name: str,
    defaults: Mapping[str, float],
) -> dict[str, float]:
    raw = mapping.get(name, {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"quality.{name} must be a mapping")
    unknown = set(raw) - set(defaults)
    if unknown:
        raise ValueError(f"quality.{name} has unknown components: {sorted(unknown)}")
    return {
        component: float(raw.get(component, default))
        for component, default in defaults.items()
    }


@dataclass(frozen=True)
class SqiConfig:
    """显式 SQI profile / Explicit SQI profile."""

    q_rate_threshold: float = 0.50
    q_morph_threshold: float = 0.65
    minimum_coverage: float = 0.80
    cardiac_low_hz: float = 0.5
    cardiac_high_hz: float = 3.0
    peak_density_min_bpm: float = 30.0
    peak_density_max_bpm: float = 200.0
    ppi_min_s: float = 0.3
    ppi_max_s: float = 2.0
    long_gap_max_samples: int = 100
    flatline_duration_s: float = 1.0
    calibrator: str = "fixed_formula_thresholds_v1"
    calibrator_lower_quantile: float = 0.10
    calibrator_upper_quantile: float = 0.90
    cardiac_concentration_reference: float = 0.65
    autocorrelation_reference: float = 0.70
    ppi_cv_scale: float = 0.20
    motion_rms_scale: float = 4.0
    nonflat_std_threshold: float = 1e-10
    clipping_fraction_reference: float = 0.02
    saturation_fraction_reference: float = 0.02
    morph_skewness_scale: float = 3.0
    morph_kurtosis_center: float = 3.0
    morph_kurtosis_scale: float = 5.0
    component_pass_threshold: float = 0.50
    template_half_width_s: float = 0.30
    spectral_analysis_low_hz: float = 0.20
    spectral_analysis_high_hz: float = 8.0
    welch_max_nperseg: int = 2048
    template_min_peaks: int = 5
    template_min_beats: int = 3
    template_resample_points: int = 101
    ppi_stability_min_intervals: int = 3
    rate_component_weights: Mapping[str, float] = field(
        default_factory=_default_rate_component_weights
    )
    morph_component_weights: Mapping[str, float] = field(
        default_factory=_default_morph_component_weights
    )

    def validate(self) -> None:
        """校验所有 SQI 阈值 / Validate every explicit SQI threshold."""

        finite_scalars = {
            "q_rate_threshold": self.q_rate_threshold,
            "q_morph_threshold": self.q_morph_threshold,
            "minimum_coverage": self.minimum_coverage,
            "cardiac_low_hz": self.cardiac_low_hz,
            "cardiac_high_hz": self.cardiac_high_hz,
            "peak_density_min_bpm": self.peak_density_min_bpm,
            "peak_density_max_bpm": self.peak_density_max_bpm,
            "ppi_min_s": self.ppi_min_s,
            "ppi_max_s": self.ppi_max_s,
            "flatline_duration_s": self.flatline_duration_s,
            "calibrator_lower_quantile": self.calibrator_lower_quantile,
            "calibrator_upper_quantile": self.calibrator_upper_quantile,
            "cardiac_concentration_reference": self.cardiac_concentration_reference,
            "autocorrelation_reference": self.autocorrelation_reference,
            "ppi_cv_scale": self.ppi_cv_scale,
            "motion_rms_scale": self.motion_rms_scale,
            "nonflat_std_threshold": self.nonflat_std_threshold,
            "clipping_fraction_reference": self.clipping_fraction_reference,
            "saturation_fraction_reference": self.saturation_fraction_reference,
            "morph_skewness_scale": self.morph_skewness_scale,
            "morph_kurtosis_center": self.morph_kurtosis_center,
            "morph_kurtosis_scale": self.morph_kurtosis_scale,
            "component_pass_threshold": self.component_pass_threshold,
            "template_half_width_s": self.template_half_width_s,
            "spectral_analysis_low_hz": self.spectral_analysis_low_hz,
            "spectral_analysis_high_hz": self.spectral_analysis_high_hz,
        }
        if not np.isfinite(list(finite_scalars.values())).all():
            raise ValueError("all SQI numerical parameters must be finite")
        if not 0.0 <= self.q_rate_threshold <= 1.0 or not 0.0 <= self.q_morph_threshold <= 1.0:
            raise ValueError("SQI endpoint thresholds must lie in [0,1]")
        if not 0.0 <= self.component_pass_threshold <= 1.0:
            raise ValueError("component_pass_threshold must lie in [0,1]")
        if not 0.0 < self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must lie in (0,1]")
        if not 0.0 < self.cardiac_low_hz < self.cardiac_high_hz < CANONICAL_FS_HZ / 2.0:
            raise ValueError("invalid SQI cardiac band")
        if not (
            0.0 <= self.spectral_analysis_low_hz
            <= self.cardiac_low_hz
            < self.cardiac_high_hz
            <= self.spectral_analysis_high_hz
            < CANONICAL_FS_HZ / 2.0
        ):
            raise ValueError(
                "SQI spectral analysis band must contain the cardiac band within Nyquist"
            )
        if not 0.0 < self.peak_density_min_bpm < self.peak_density_max_bpm:
            raise ValueError("invalid peak-density range")
        if not 0.0 < self.ppi_min_s < self.ppi_max_s:
            raise ValueError("invalid PPI range")
        if (
            isinstance(self.long_gap_max_samples, bool)
            or not isinstance(self.long_gap_max_samples, (int, np.integer))
            or self.long_gap_max_samples < 0
            or self.flatline_duration_s <= 0.0
        ):
            raise ValueError("invalid flatline/long-gap SQI thresholds")
        if not 0.0 <= self.calibrator_lower_quantile < self.calibrator_upper_quantile <= 1.0:
            raise ValueError("calibrator quantiles must be ordered in [0,1]")
        positive_scales = {
            "cardiac_concentration_reference": self.cardiac_concentration_reference,
            "autocorrelation_reference": self.autocorrelation_reference,
            "ppi_cv_scale": self.ppi_cv_scale,
            "motion_rms_scale": self.motion_rms_scale,
            "nonflat_std_threshold": self.nonflat_std_threshold,
            "clipping_fraction_reference": self.clipping_fraction_reference,
            "saturation_fraction_reference": self.saturation_fraction_reference,
            "morph_skewness_scale": self.morph_skewness_scale,
            "morph_kurtosis_scale": self.morph_kurtosis_scale,
            "template_half_width_s": self.template_half_width_s,
        }
        if any(value <= 0.0 for value in positive_scales.values()):
            raise ValueError("SQI normalization scales must be positive")
        if self.calibrator not in {
            "outer_train_empirical_quantiles_v1",
            "fixed_formula_thresholds_v1",
        }:
            raise ValueError("unsupported SQI calibrator profile")
        for name, value, minimum in (
            ("welch_max_nperseg", self.welch_max_nperseg, 2),
            ("template_min_peaks", self.template_min_peaks, 2),
            ("template_min_beats", self.template_min_beats, 2),
            ("template_resample_points", self.template_resample_points, 3),
            ("ppi_stability_min_intervals", self.ppi_stability_min_intervals, 2),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < minimum
            ):
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if self.template_min_peaks < self.template_min_beats:
            raise ValueError("template_min_peaks must be at least template_min_beats")
        if self.template_half_width_s * CANONICAL_FS_HZ < 1.0:
            raise ValueError("template_half_width_s must span at least one sample")
        for name, values, expected in (
            ("rate_component_weights", self.rate_component_weights, RATE_COMPONENT_NAMES),
            ("morph_component_weights", self.morph_component_weights, MORPH_COMPONENT_NAMES),
        ):
            if set(values) != set(expected):
                raise ValueError(f"{name} must resolve every registered component")
            weights = np.asarray([values[item] for item in expected], dtype=np.float64)
            if (
                not np.isfinite(weights).all()
                or np.any(weights < 0.0)
                or not np.any(weights > 0.0)
            ):
                raise ValueError(f"{name} must contain finite nonnegative weights with positive mass")

    @classmethod
    def from_quality_mapping(cls, quality: Mapping[str, object] | None) -> "SqiConfig":
        """Resolve a partial public quality section to the exact runtime values."""

        if quality is None:
            quality = {}
        if not isinstance(quality, Mapping):
            raise ValueError("quality must be a mapping")
        allowed_fields = {
            "mode",
            "fit_scope",
            "components",
            "high_quality_rule",
            "failure_action",
            "rate_threshold",
            "morph_threshold",
            "minimum_coverage",
            "cardiac_band_hz",
            "peak_density_bpm_range",
            "ppi_range_s",
            "long_gap_max_samples",
            "flatline_duration_s",
            "calibrator",
            "calibrator_quantiles",
            "rate_component_weights",
            "morph_component_weights",
            "component_normalization",
            "spectral_analysis_band_hz",
            "welch_max_nperseg",
            "template_min_peaks",
            "template_min_beats",
            "template_resample_points",
            "ppi_stability_min_intervals",
        }
        unknown_fields = set(quality) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f"quality has unknown fields: {sorted(unknown_fields)}"
            )
        defaults = cls()
        band = _pair(
            quality,
            "cardiac_band_hz",
            (defaults.cardiac_low_hz, defaults.cardiac_high_hz),
        )
        density = _pair(
            quality,
            "peak_density_bpm_range",
            (defaults.peak_density_min_bpm, defaults.peak_density_max_bpm),
        )
        ppi = _pair(
            quality,
            "ppi_range_s",
            (defaults.ppi_min_s, defaults.ppi_max_s),
        )
        spectral_analysis = _pair(
            quality,
            "spectral_analysis_band_hz",
            (
                defaults.spectral_analysis_low_hz,
                defaults.spectral_analysis_high_hz,
            ),
        )
        quantiles = _pair(
            quality,
            "calibrator_quantiles",
            (
                defaults.calibrator_lower_quantile,
                defaults.calibrator_upper_quantile,
            ),
        )
        normalization_raw = quality.get("component_normalization", {})
        if normalization_raw is None:
            normalization_raw = {}
        if not isinstance(normalization_raw, Mapping):
            raise ValueError("quality.component_normalization must be a mapping")
        normalization_names = {
            "cardiac_concentration_reference",
            "autocorrelation_reference",
            "ppi_cv_scale",
            "motion_rms_scale",
            "nonflat_std_threshold",
            "clipping_fraction_reference",
            "saturation_fraction_reference",
            "morph_skewness_scale",
            "morph_kurtosis_center",
            "morph_kurtosis_scale",
            "component_pass_threshold",
            "template_half_width_s",
        }
        unknown_normalization = set(normalization_raw) - normalization_names
        if unknown_normalization:
            raise ValueError(
                "quality.component_normalization has unknown fields: "
                f"{sorted(unknown_normalization)}"
            )
        calibrator_raw = quality.get("calibrator")
        calibrator = (
            "outer_train_empirical_quantiles_v1"
            if calibrator_raw is None
            or (
                isinstance(calibrator_raw, str)
                and calibrator_raw in {"", "deferred_supervised_design"}
            )
            else str(calibrator_raw)
        )

        def normalized(name: str) -> float:
            return float(
                _value_or_default(
                    normalization_raw,
                    name,
                    getattr(defaults, name),
                )
            )

        result = cls(
            q_rate_threshold=float(
                _value_or_default(quality, "rate_threshold", defaults.q_rate_threshold)
            ),
            q_morph_threshold=float(
                _value_or_default(quality, "morph_threshold", defaults.q_morph_threshold)
            ),
            minimum_coverage=float(
                _value_or_default(quality, "minimum_coverage", defaults.minimum_coverage)
            ),
            cardiac_low_hz=band[0],
            cardiac_high_hz=band[1],
            peak_density_min_bpm=density[0],
            peak_density_max_bpm=density[1],
            ppi_min_s=ppi[0],
            ppi_max_s=ppi[1],
            long_gap_max_samples=_value_or_default(
                quality, "long_gap_max_samples", defaults.long_gap_max_samples
            ),
            flatline_duration_s=float(
                _value_or_default(
                    quality, "flatline_duration_s", defaults.flatline_duration_s
                )
            ),
            calibrator=calibrator,
            calibrator_lower_quantile=quantiles[0],
            calibrator_upper_quantile=quantiles[1],
            cardiac_concentration_reference=normalized(
                "cardiac_concentration_reference"
            ),
            autocorrelation_reference=normalized("autocorrelation_reference"),
            ppi_cv_scale=normalized("ppi_cv_scale"),
            motion_rms_scale=normalized("motion_rms_scale"),
            nonflat_std_threshold=normalized("nonflat_std_threshold"),
            clipping_fraction_reference=normalized("clipping_fraction_reference"),
            saturation_fraction_reference=normalized(
                "saturation_fraction_reference"
            ),
            morph_skewness_scale=normalized("morph_skewness_scale"),
            morph_kurtosis_center=normalized("morph_kurtosis_center"),
            morph_kurtosis_scale=normalized("morph_kurtosis_scale"),
            component_pass_threshold=normalized("component_pass_threshold"),
            template_half_width_s=normalized("template_half_width_s"),
            spectral_analysis_low_hz=spectral_analysis[0],
            spectral_analysis_high_hz=spectral_analysis[1],
            welch_max_nperseg=_value_or_default(
                quality, "welch_max_nperseg", defaults.welch_max_nperseg
            ),
            template_min_peaks=_value_or_default(
                quality, "template_min_peaks", defaults.template_min_peaks
            ),
            template_min_beats=_value_or_default(
                quality, "template_min_beats", defaults.template_min_beats
            ),
            template_resample_points=_value_or_default(
                quality,
                "template_resample_points",
                defaults.template_resample_points,
            ),
            ppi_stability_min_intervals=_value_or_default(
                quality,
                "ppi_stability_min_intervals",
                defaults.ppi_stability_min_intervals,
            ),
            rate_component_weights=_weights(
                quality,
                "rate_component_weights",
                defaults.rate_component_weights,
            ),
            morph_component_weights=_weights(
                quality,
                "morph_component_weights",
                defaults.morph_component_weights,
            ),
        )
        result.validate()
        return result

    @classmethod
    def from_resolved(cls, config: Mapping[str, object]) -> "SqiConfig":
        """从 resolved YAML 解析可省略默认值 / Resolve runtime defaults."""

        quality = config.get("quality")
        if not isinstance(quality, Mapping):
            raise ValueError("resolved config['quality'] is required")
        return cls.from_quality_mapping(quality)

    def to_dict(self) -> dict[str, object]:
        """Return the fully resolved numerical policy actually used at runtime."""

        self.validate()
        return {
            "rate_threshold": float(self.q_rate_threshold),
            "morph_threshold": float(self.q_morph_threshold),
            "minimum_coverage": float(self.minimum_coverage),
            "cardiac_band_hz": [float(self.cardiac_low_hz), float(self.cardiac_high_hz)],
            "peak_density_bpm_range": [
                float(self.peak_density_min_bpm),
                float(self.peak_density_max_bpm),
            ],
            "ppi_range_s": [float(self.ppi_min_s), float(self.ppi_max_s)],
            "long_gap_max_samples": int(self.long_gap_max_samples),
            "flatline_duration_s": float(self.flatline_duration_s),
            "spectral_analysis_band_hz": [
                float(self.spectral_analysis_low_hz),
                float(self.spectral_analysis_high_hz),
            ],
            "welch_max_nperseg": int(self.welch_max_nperseg),
            "template_min_peaks": int(self.template_min_peaks),
            "template_min_beats": int(self.template_min_beats),
            "template_resample_points": int(self.template_resample_points),
            "ppi_stability_min_intervals": int(self.ppi_stability_min_intervals),
            "calibrator": self.calibrator,
            "calibrator_quantiles": [
                float(self.calibrator_lower_quantile),
                float(self.calibrator_upper_quantile),
            ],
            "rate_component_weights": dict(self.rate_component_weights),
            "morph_component_weights": dict(self.morph_component_weights),
            "component_normalization": {
                name: float(getattr(self, name))
                for name in (
                    "cardiac_concentration_reference",
                    "autocorrelation_reference",
                    "ppi_cv_scale",
                    "motion_rms_scale",
                    "nonflat_std_threshold",
                    "clipping_fraction_reference",
                    "saturation_fraction_reference",
                    "morph_skewness_scale",
                    "morph_kurtosis_center",
                    "morph_kurtosis_scale",
                    "component_pass_threshold",
                    "template_half_width_s",
                )
            },
        }


SQI_DIAGNOSTICS_SCHEMA = "ppg_frailty.sqi_raw_diagnostics.v2"


@dataclass(frozen=True)
class SqiDiagnosticConfig:
    """Physical component parameters only; no fusion weights or pass thresholds."""

    cardiac_low_hz: float = 0.5
    cardiac_high_hz: float = 3.0
    peak_density_min_bpm: float = 30.0
    peak_density_max_bpm: float = 200.0
    ppi_min_s: float = 0.3
    ppi_max_s: float = 2.0
    long_gap_max_samples: int = 100
    flatline_duration_s: float = 1.0
    template_half_width_s: float = 0.30
    spectral_analysis_low_hz: float = 0.20
    spectral_analysis_high_hz: float = 8.0
    welch_max_nperseg: int = 2048
    template_min_peaks: int = 5
    template_min_beats: int = 3
    template_resample_points: int = 101
    ppi_stability_min_intervals: int = 3

    def validate(self) -> None:
        values = (
            self.cardiac_low_hz,
            self.cardiac_high_hz,
            self.peak_density_min_bpm,
            self.peak_density_max_bpm,
            self.ppi_min_s,
            self.ppi_max_s,
            self.flatline_duration_s,
            self.template_half_width_s,
            self.spectral_analysis_low_hz,
            self.spectral_analysis_high_hz,
        )
        if not np.isfinite(values).all():
            raise ValueError("all diagnostic SQI parameters must be finite")
        if not 0.0 < self.cardiac_low_hz < self.cardiac_high_hz < CANONICAL_FS_HZ / 2.0:
            raise ValueError("invalid diagnostic cardiac band")
        if not (
            0.0 <= self.spectral_analysis_low_hz
            <= self.cardiac_low_hz
            < self.cardiac_high_hz
            <= self.spectral_analysis_high_hz
            < CANONICAL_FS_HZ / 2.0
        ):
            raise ValueError("invalid diagnostic spectral analysis band")
        if not 0.0 < self.peak_density_min_bpm < self.peak_density_max_bpm:
            raise ValueError("invalid diagnostic peak-density range")
        if not 0.0 < self.ppi_min_s < self.ppi_max_s:
            raise ValueError("invalid diagnostic PPI range")
        if (
            isinstance(self.long_gap_max_samples, bool)
            or not isinstance(self.long_gap_max_samples, (int, np.integer))
            or self.long_gap_max_samples < 0
            or self.flatline_duration_s <= 0.0
            or self.template_half_width_s <= 0.0
        ):
            raise ValueError("invalid diagnostic gap/flatline parameters")
        for name, value, minimum in (
            ("welch_max_nperseg", self.welch_max_nperseg, 2),
            ("template_min_peaks", self.template_min_peaks, 2),
            ("template_min_beats", self.template_min_beats, 2),
            ("template_resample_points", self.template_resample_points, 3),
            ("ppi_stability_min_intervals", self.ppi_stability_min_intervals, 2),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < minimum
            ):
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if self.template_min_peaks < self.template_min_beats:
            raise ValueError("template_min_peaks must be at least template_min_beats")
        if self.template_half_width_s * CANONICAL_FS_HZ < 1.0:
            raise ValueError("template_half_width_s must span at least one sample")

    @classmethod
    def from_resolved(
        cls,
        config: Mapping[str, object],
    ) -> "SqiDiagnosticConfig":
        """Parse only physical component fields from a resolved pipeline config.

        Diagnostics-only mode deliberately ignores endpoint thresholds,
        calibrator identity, routing readiness, and any learned policy fields.
        """

        quality = config.get("quality")
        if not isinstance(quality, Mapping):
            raise ValueError("resolved config['quality'] is required")
        defaults = cls()
        band = _pair(
            quality,
            "cardiac_band_hz",
            (defaults.cardiac_low_hz, defaults.cardiac_high_hz),
        )
        density = _pair(
            quality,
            "peak_density_bpm_range",
            (defaults.peak_density_min_bpm, defaults.peak_density_max_bpm),
        )
        ppi = _pair(
            quality,
            "ppi_range_s",
            (defaults.ppi_min_s, defaults.ppi_max_s),
        )
        spectral_analysis = _pair(
            quality,
            "spectral_analysis_band_hz",
            (
                defaults.spectral_analysis_low_hz,
                defaults.spectral_analysis_high_hz,
            ),
        )
        normalization_raw = quality.get("component_normalization", {})
        if normalization_raw is None:
            normalization_raw = {}
        if not isinstance(normalization_raw, Mapping):
            raise ValueError("quality.component_normalization must be a mapping")
        result = cls(
            cardiac_low_hz=float(band[0]),
            cardiac_high_hz=float(band[1]),
            peak_density_min_bpm=float(density[0]),
            peak_density_max_bpm=float(density[1]),
            ppi_min_s=float(ppi[0]),
            ppi_max_s=float(ppi[1]),
            long_gap_max_samples=_value_or_default(
                quality, "long_gap_max_samples", defaults.long_gap_max_samples
            ),
            flatline_duration_s=float(
                _value_or_default(
                    quality, "flatline_duration_s", defaults.flatline_duration_s
                )
            ),
            template_half_width_s=float(
                _value_or_default(
                    normalization_raw,
                    "template_half_width_s",
                    defaults.template_half_width_s,
                )
            ),
            spectral_analysis_low_hz=spectral_analysis[0],
            spectral_analysis_high_hz=spectral_analysis[1],
            welch_max_nperseg=_value_or_default(
                quality, "welch_max_nperseg", defaults.welch_max_nperseg
            ),
            template_min_peaks=_value_or_default(
                quality, "template_min_peaks", defaults.template_min_peaks
            ),
            template_min_beats=_value_or_default(
                quality, "template_min_beats", defaults.template_min_beats
            ),
            template_resample_points=_value_or_default(
                quality,
                "template_resample_points",
                defaults.template_resample_points,
            ),
            ppi_stability_min_intervals=_value_or_default(
                quality,
                "ppi_stability_min_intervals",
                defaults.ppi_stability_min_intervals,
            ),
        )
        result.validate()
        return result

    @classmethod
    def from_sqi_config(cls, config: SqiConfig) -> "SqiDiagnosticConfig":
        """Copy physical parameters while discarding weights/endpoint policy."""

        result = cls(
            cardiac_low_hz=config.cardiac_low_hz,
            cardiac_high_hz=config.cardiac_high_hz,
            peak_density_min_bpm=config.peak_density_min_bpm,
            peak_density_max_bpm=config.peak_density_max_bpm,
            ppi_min_s=config.ppi_min_s,
            ppi_max_s=config.ppi_max_s,
            long_gap_max_samples=config.long_gap_max_samples,
            flatline_duration_s=config.flatline_duration_s,
            template_half_width_s=config.template_half_width_s,
            spectral_analysis_low_hz=config.spectral_analysis_low_hz,
            spectral_analysis_high_hz=config.spectral_analysis_high_hz,
            welch_max_nperseg=config.welch_max_nperseg,
            template_min_peaks=config.template_min_peaks,
            template_min_beats=config.template_min_beats,
            template_resample_points=config.template_resample_points,
            ppi_stability_min_intervals=config.ppi_stability_min_intervals,
        )
        result.validate()
        return result


@dataclass(frozen=True)
class SqiDiagnosticComponent:
    """One raw observation with no normalized score or pass/fail state."""

    raw_value: float | None
    available: bool
    reason: str


@dataclass(frozen=True)
class SqiDiagnostics:
    """Raw component archive that cannot represent a routing decision."""

    components: dict[str, SqiDiagnosticComponent]
    coverage: float
    route: str
    reasons: tuple[str, ...]
    schema_version: str = SQI_DIAGNOSTICS_SCHEMA
    aggregation_performed: bool = False
    weights_applied: bool = False
    endpoint_thresholds_applied: bool = False
    affects_classification: bool = False

    def validate(self) -> None:
        if self.schema_version != SQI_DIAGNOSTICS_SCHEMA:
            raise ValueError("SQI diagnostics schema drift")
        if not 0.0 <= float(self.coverage) <= 1.0:
            raise ValueError("SQI diagnostic coverage must lie in [0,1]")
        if not self.components:
            raise ValueError("SQI diagnostics require component observations")
        if (
            self.aggregation_performed
            or self.weights_applied
            or self.endpoint_thresholds_applied
            or self.affects_classification
        ):
            raise ValueError("raw SQI diagnostics cannot encode a classifier decision")


@dataclass(frozen=True)
class SqiCalibrator:
    """outer-train empirical quantile 映射 / Train-only empirical-quantile map."""

    bounds: dict[str, tuple[float, float]]
    fitted_on_participant_ids: tuple[str, ...]
    method: str = "outer_train_empirical_quantiles_v1"

    def transform(self, name: str, value: float | None) -> float | None:
        """把 base score 映射到 train empirical [0,1] / Map a base score."""

        if value is None or not np.isfinite(value) or name not in self.bounds:
            return value
        low, high = self.bounds[name]
        if high <= low:
            return float(np.clip(value, 0.0, 1.0))
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def fit_sqi_calibrator(
    component_rows: Iterable[Mapping[str, float]],
    participant_ids: Iterable[str],
    *,
    fitted_on_participant_ids: Iterable[str],
    outer_train_participant_ids: Iterable[str],
    outer_oof_participant_ids: Iterable[str],
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> SqiCalibrator:
    """只读取声明的 outer-train rows / Fit using declared outer-train rows only."""

    if (
        not np.isfinite([lower_quantile, upper_quantile]).all()
        or not 0.0 <= float(lower_quantile) < float(upper_quantile) <= 1.0
    ):
        raise ValueError("SQI calibrator quantiles must be finite and ordered in [0,1]")
    fitted = assert_training_only(
        fitted_on_participant_ids,
        outer_train_participant_ids,
        outer_oof_participant_ids,
    )
    fitted_set = set(fitted)
    rows = tuple(component_rows)
    ids = tuple(str(value) for value in participant_ids)
    if len(rows) != len(ids):
        raise ValueError("component_rows and participant_ids must align")
    selected = [row for row, participant in zip(rows, ids) if participant in fitted_set]
    if not selected:
        raise ValueError("no outer-train SQI component rows were provided")
    names = sorted({name for row in selected for name in row})
    bounds: dict[str, tuple[float, float]] = {}
    for name in names:
        values = np.asarray(
            [row[name] for row in selected if name in row and np.isfinite(row[name])],
            dtype=np.float64,
        )
        if values.size:
            bounds[name] = (
                float(np.quantile(values, lower_quantile)),
                float(np.quantile(values, upper_quantile)),
            )
    return SqiCalibrator(bounds=bounds, fitted_on_participant_ids=fitted)


def _component(
    raw: float | None,
    normalized: float | None,
    reason: str,
    *,
    pass_threshold: float = 0.50,
) -> QualityComponent:
    """构造有限 component 或 unavailable / Build a finite or unavailable component."""

    if raw is None or normalized is None or not np.isfinite(raw) or not np.isfinite(normalized):
        return QualityComponent(None, None, QualityState.UNAVAILABLE, reason)
    score = float(np.clip(normalized, 0.0, 1.0))
    state = QualityState.PASS if score >= pass_threshold else QualityState.FAIL
    return QualityComponent(float(raw), score, state, reason)


def _welch_metrics(
    values: np.ndarray,
    fs_hz: float,
    config: SqiConfig | SqiDiagnosticConfig,
) -> tuple[float, float]:
    """返回 cardiac concentration 与归一化谱熵 / Return spectral SQI metrics."""

    concentrations: list[float] = []
    entropies: list[float] = []
    for column in range(values.shape[1]):
        frequencies, power = signal.welch(
            values[:, column],
            fs=fs_hz,
            nperseg=min(config.welch_max_nperseg, values.shape[0]),
        )
        usable = (
            (frequencies >= config.spectral_analysis_low_hz)
            & (frequencies <= config.spectral_analysis_high_hz)
        )
        cardiac = (
            (frequencies >= config.cardiac_low_hz)
            & (frequencies <= config.cardiac_high_hz)
        )
        total = float(np.sum(power[usable]))
        if total <= 0.0:
            continue
        concentrations.append(float(np.sum(power[cardiac]) / total))
        distribution = power[usable] / total
        entropy = -float(np.sum(distribution * np.log(distribution + 1e-15)))
        entropies.append(entropy / max(np.log(distribution.size), 1e-12))
    if not concentrations:
        return float("nan"), float("nan")
    return float(np.mean(concentrations)), float(np.mean(entropies))


def _autocorrelation_periodicity(
    values: np.ndarray,
    fs_hz: float,
    *,
    lag_min_s: float,
    lag_max_s: float,
) -> float:
    """Find maximum normalized ACF in the configured physiological lag range."""

    scores: list[float] = []
    low, high = int(round(lag_min_s * fs_hz)), int(round(lag_max_s * fs_hz))
    for column in range(values.shape[1]):
        x = values[:, column] - np.mean(values[:, column])
        denominator = float(np.dot(x, x))
        if denominator <= 1e-12 or x.size <= high:
            continue
        corr = signal.correlate(x, x, mode="full", method="fft")[x.size - 1 :] / denominator
        scores.append(float(np.max(corr[low : high + 1])))
    return float(np.mean(scores)) if scores else float("nan")


def _template_correlation(
    values: np.ndarray,
    pulse: PulseResult,
    fs_hz: float,
    *,
    half_width_s: float,
    minimum_peaks: int,
    minimum_beats: int,
    resample_points: int,
) -> float:
    """用 median beat template 计算逐搏相关 / Median beat-template correlation."""

    peaks = np.asarray(pulse.peaks, dtype=np.int64)
    accepted = np.asarray(pulse.accepted_peak_mask, dtype=bool)
    if peaks.size < minimum_peaks:
        return float("nan")
    channel = 0 if pulse.wavelength.upper() == "RED" else min(1, values.shape[1] - 1)
    width = int(round(half_width_s * fs_hz))
    target = np.linspace(0.0, 1.0, resample_points)
    beats: list[np.ndarray] = []
    for peak, keep in zip(peaks, accepted):
        if not keep or peak - width < 0 or peak + width >= values.shape[0]:
            continue
        segment = values[peak - width : peak + width + 1, channel]
        segment = np.interp(target, np.linspace(0.0, 1.0, segment.size), segment)
        segment -= np.mean(segment)
        scale = float(np.std(segment))
        if scale > 1e-12:
            beats.append(segment / scale)
    if len(beats) < minimum_beats:
        return float("nan")
    stack = np.vstack(beats)
    template = np.median(stack, axis=0)
    correlations = [np.corrcoef(beat, template)[0, 1] for beat in stack]
    return float(np.nanmedian(correlations))


def _endpoint(
    components: dict[str, QualityComponent],
    weights: dict[str, float],
    *,
    threshold: float,
    coverage: float,
    minimum_coverage: float,
) -> QualityEndpoint:
    """只对 available 组件重归一化权重 / Renormalize weights over available components."""

    usable = [
        (weights[name], component.normalized_value)
        for name, component in components.items()
        if name in weights
        and weights[name] > 0.0
        and component.normalized_value is not None
    ]
    if not usable:
        return QualityEndpoint(
            None, QualityState.UNAVAILABLE, threshold, components,
            ("no_available_sqi_components",), coverage,
        )
    weight_scale = max(weight for weight, _ in usable)
    scaled = tuple((weight / weight_scale, value) for weight, value in usable)
    denominator = float(sum(weight for weight, _ in scaled))
    score = float(
        sum(weight * float(value) for weight, value in scaled) / denominator
    )
    reasons: list[str] = []
    if coverage < minimum_coverage:
        reasons.append("coverage_below_threshold")
    state = QualityState.PASS if score >= threshold and coverage >= minimum_coverage else QualityState.FAIL
    if score < threshold:
        reasons.append("weighted_score_below_threshold")
    return QualityEndpoint(score, state, threshold, components, tuple(reasons), coverage)


def _calibrate_components(
    prefix: str,
    components: dict[str, QualityComponent],
    calibrator: SqiCalibrator | None,
    *,
    pass_threshold: float,
) -> dict[str, QualityComponent]:
    """应用 train-only map，保留 raw/reason / Apply a train-only component map."""

    if calibrator is None:
        return components
    calibrated: dict[str, QualityComponent] = {}
    for name, component in components.items():
        normalized = calibrator.transform(
            f"{prefix}.{name}", component.normalized_value
        )
        if normalized is None:
            calibrated[name] = component
        else:
            calibrated[name] = QualityComponent(
                raw_value=component.raw_value,
                normalized_value=normalized,
                state=(
                    QualityState.PASS
                    if normalized >= pass_threshold
                    else QualityState.FAIL
                ),
                reason=component.reason + ";outer_train_empirical_calibration",
            )
    return calibrated


def _qc_components(
    evidence: Mapping[str, object] | None,
    *,
    fs_hz: float,
    flatline_duration_s: float,
    long_gap_max_samples: int,
    clipping_fraction_reference: float = 0.02,
    saturation_fraction_reference: float = 0.02,
    component_pass_threshold: float = 0.50,
) -> dict[str, QualityComponent]:
    """显式构造 flatline/clipping/saturation/long-gap / Build QC components."""

    unavailable = {
        "flatline": _component(None, None, "qc_evidence_unavailable"),
        "clipping": _component(None, None, "qc_evidence_unavailable"),
        "saturation": _component(
            None, None, "adc_rails_unknown_saturation_not_inferred"
        ),
        "long_gap": _component(None, None, "qc_evidence_unavailable"),
    }
    if evidence is None:
        return unavailable
    channels = evidence.get("channels")
    if not isinstance(channels, Mapping) or not channels:
        return unavailable
    rows = [value for value in channels.values() if isinstance(value, Mapping)]
    if not rows:
        return unavailable
    longest_flat = max(
        float(row.get("longest_constant_run", 0.0)) for row in rows
    )
    clipping_occupancy = max(
        max(
            float(row.get("min_occupancy", 0.0)),
            float(row.get("max_occupancy", 0.0)),
        )
        for row in rows
    )
    longest_gap = max(
        float(row.get("longest_nonfinite_gap_samples", 0.0))
        for row in rows
    )
    output = {
        "flatline": _component(
            longest_flat / fs_hz,
            1.0 - min(
                longest_flat / max(flatline_duration_s * fs_hz, 1.0), 1.0
            ),
            "longest_constant_run_vs_resolved_duration",
            pass_threshold=component_pass_threshold,
        ),
        "clipping": _component(
            clipping_occupancy,
            1.0 - min(clipping_occupancy / clipping_fraction_reference, 1.0),
            "extreme_occupancy_heuristic_adc_rails_unknown",
            pass_threshold=component_pass_threshold,
        ),
        "long_gap": _component(
            longest_gap,
            1.0 if longest_gap <= float(long_gap_max_samples) else 0.0,
            "longest_nonfinite_gap_vs_resolved_samples",
            pass_threshold=component_pass_threshold,
        ),
        "saturation": unavailable["saturation"],
    }
    saturation = evidence.get("adc_saturation_fraction")
    if saturation is not None and np.isfinite(float(saturation)):
        fraction = float(saturation)
        output["saturation"] = _component(
            fraction,
            1.0 - min(fraction / saturation_fraction_reference, 1.0),
            "declared_adc_rail_saturation_fraction",
            pass_threshold=component_pass_threshold,
        )
    return output


def quality_component_scores(result: QualityResult) -> dict[str, float]:
    """导出 calibrator 输入行 / Export one row of base normalized components."""

    return {
        name: float(component.normalized_value)
        for name, component in result.components.items()
        if component.normalized_value is not None
    }


def _diagnostic_component(
    value: float | None,
    reason: str,
) -> SqiDiagnosticComponent:
    observed = None if value is None else float(value)
    available = observed is not None and bool(np.isfinite(observed))
    return SqiDiagnosticComponent(
        raw_value=observed if available else None,
        available=available,
        reason=reason if available else f"{reason}:unavailable",
    )


def evaluate_quality_diagnostics(
    values: np.ndarray | CanonicalSignalViews,
    *,
    route: SignalRoute | None = None,
    pulse: PulseResult | None = None,
    imu_processed: dict[str, np.ndarray] | None = None,
    source_valid_mask: np.ndarray | None = None,
    qc_evidence: Mapping[str, object] | None = None,
    fs_hz: float = CANONICAL_FS_HZ,
    config: SqiDiagnosticConfig | SqiConfig | None = None,
    detector_id: str | None = None,
    min_observation_sec: float = 8.0,
    min_peaks: int = 5,
    detector_parameters: Mapping[str, object] | None = None,
) -> SqiDiagnostics:
    """Compute raw SQI observations without weights, thresholds, or routing.

    This is the only evaluator used by quality.mode=diagnostics_only.
    Physical bands and physiological ranges remain explicit component
    definitions, but no endpoint score, pass/fail state, calibrator, retention
    decision, aggregation weight, or predictor input is produced.
    """

    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("SQI diagnostics require exactly 400 Hz")
    physical = (
        SqiDiagnosticConfig()
        if config is None
        else SqiDiagnosticConfig.from_sqi_config(config)
        if isinstance(config, SqiConfig)
        else config
    )
    if not isinstance(physical, SqiDiagnosticConfig):
        raise TypeError("diagnostic config must be SqiDiagnosticConfig or SqiConfig")
    physical.validate()

    coverage_override: float | None = None
    if isinstance(values, CanonicalSignalViews):
        route = values.route
        matrix = np.asarray(values.analysis_signal, dtype=np.float64)
        imu_processed = values.imu_processed
        source_valid_mask = values.source_valid_mask
        if qc_evidence is None:
            candidate = values.metadata.get("ppg_qc_metrics")
            qc_evidence = candidate if isinstance(candidate, Mapping) else None
        rate_valid = np.asarray(values.rate_valid_mask, dtype=bool)
        coverage_override = float(np.mean(rate_valid))
        if not np.all(rate_valid):
            padded = np.concatenate(([False], rate_valid, [False]))
            changes = np.diff(padded.astype(np.int8))
            runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
            if not runs:
                raise ValueError("SQI diagnostics have no artifact-valid signal run")
            start, stop = max(runs, key=lambda bounds: bounds[1] - bounds[0])
            matrix = matrix[start:stop]
            pulse = None
            if imu_processed is not None:
                imu_processed = {
                    name: (
                        np.asarray(item)[start:stop]
                        if np.asarray(item).ndim >= 1
                        and np.asarray(item).shape[0] == rate_valid.size
                        else np.asarray(item)
                    )
                    for name, item in imu_processed.items()
                }
    else:
        matrix = np.asarray(values, dtype=np.float64)
        if route is None:
            raise ValueError("explicit diagnostic array requires an explicit route")
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[1] not in (1, 2) or not np.isfinite(matrix).all():
        raise ValueError("SQI diagnostic input must be finite samples-by-one/two channels")
    assert route is not None
    if coverage_override is not None:
        coverage = coverage_override
    elif source_valid_mask is None:
        coverage = 1.0
    else:
        valid = np.asarray(source_valid_mask, dtype=bool)
        if valid.shape == (matrix.shape[0],):
            coverage = float(np.mean(valid))
        elif valid.shape == matrix.shape:
            coverage = float(np.mean(np.all(valid, axis=1)))
        else:
            raise ValueError("diagnostic source_valid_mask must align with samples")

    cardiac, entropy = _welch_metrics(matrix, fs_hz, physical)
    periodicity = _autocorrelation_periodicity(
        matrix,
        fs_hz,
        lag_min_s=physical.ppi_min_s,
        lag_max_s=physical.ppi_max_s,
    )
    red_ir = (
        abs(float(np.corrcoef(matrix[:, 0], matrix[:, 1])[0, 1]))
        if matrix.shape[1] == 2
        and np.std(matrix[:, 0]) > 0.0
        and np.std(matrix[:, 1]) > 0.0
        else float("nan")
    )
    if pulse is None:
        if detector_id is None:
            raise ValueError(
                "SQI diagnostics require a persisted detector_id "
                "when pulse is not supplied"
            )
        from ..peaks import detect_pulses

        pulse = detect_pulses(
            matrix,
            detector_id=detector_id,
            fs_hz=fs_hz,
            min_observation_sec=min_observation_sec,
            min_peaks=min_peaks,
            source_route=route,
            detector_parameters=detector_parameters,
        )
    peak_density = ppi_fraction = ppi_cv = float("nan")
    if pulse is not None:
        peak_density = float(np.asarray(pulse.peaks).size / (matrix.shape[0] / fs_hz) * 60.0)
        ppi = np.asarray(pulse.ppi_s, dtype=np.float64)
        valid_ppi = (
            np.asarray(pulse.valid_interval_mask, dtype=bool)
            & np.isfinite(ppi)
            & (ppi >= physical.ppi_min_s)
            & (ppi <= physical.ppi_max_s)
        )
        ppi_fraction = float(np.mean(valid_ppi)) if ppi.size else float("nan")
        selected = ppi[valid_ppi]
        if (
            selected.size >= physical.ppi_stability_min_intervals
            and float(np.mean(selected)) > 0.0
        ):
            ppi_cv = float(np.std(selected) / np.mean(selected))

    motion_rms = float("nan")
    if imu_processed is not None and "dynamic_magnitude" in imu_processed:
        motion = np.asarray(imu_processed["dynamic_magnitude"], dtype=np.float64)
        if motion.shape == (matrix.shape[0],):
            finite_motion = motion[np.isfinite(motion)]
            if finite_motion.size:
                motion_rms = float(np.sqrt(np.mean(np.square(finite_motion))))
    nonflat_scale = float(
        min(np.std(matrix[:, column]) for column in range(matrix.shape[1]))
    )
    components: dict[str, SqiDiagnosticComponent] = {
        "rate.cardiac_concentration": _diagnostic_component(
            cardiac, "cardiac_band_power_fraction"
        ),
        "rate.autocorrelation_periodicity": _diagnostic_component(
            periodicity, "physiological_lag_acf"
        ),
        "rate.normalized_spectral_entropy": _diagnostic_component(
            entropy, "normalized_welch_entropy"
        ),
        "rate.peak_density_bpm": _diagnostic_component(
            peak_density, "detected_peaks_per_minute"
        ),
        "rate.ppi_physiological_fraction": _diagnostic_component(
            ppi_fraction, "ppi_inside_explicit_physical_range"
        ),
        "rate.ppi_cv": _diagnostic_component(ppi_cv, "valid_ppi_coefficient_of_variation"),
        "rate.red_ir_agreement": _diagnostic_component(
            red_ir, "absolute_zero_lag_correlation"
        ),
        "rate.motion_energy_rms": _diagnostic_component(
            motion_rms, "dynamic_acceleration_rms_mps2"
        ),
        "rate.nonflat_scale": _diagnostic_component(
            nonflat_scale, "minimum_channel_standard_deviation"
        ),
        "rate.source_coverage": _diagnostic_component(
            coverage, "finite_and_route_valid_source_fraction"
        ),
    }
    qc = _qc_components(
        qc_evidence,
        fs_hz=fs_hz,
        flatline_duration_s=physical.flatline_duration_s,
        long_gap_max_samples=physical.long_gap_max_samples,
    )
    components.update(
        {
            f"rate.{name}": _diagnostic_component(item.raw_value, item.reason)
            for name, item in qc.items()
        }
    )
    reasons: list[str] = []
    if route is SignalRoute.ARTIFACT_RATE_ONLY:
        reasons.append("morphology_diagnostics_not_applicable_to_nonidentity_rate_route")
    else:
        skewness = float(np.mean(stats.skew(matrix, axis=0, bias=False)))
        kurtosis = float(
            np.mean(stats.kurtosis(matrix, axis=0, fisher=False, bias=False))
        )
        template = (
            _template_correlation(
                matrix,
                pulse,
                fs_hz,
                half_width_s=physical.template_half_width_s,
                minimum_peaks=physical.template_min_peaks,
                minimum_beats=physical.template_min_beats,
                resample_points=physical.template_resample_points,
            )
            if pulse is not None
            else float("nan")
        )
        components.update(
            {
                "morph.template_correlation": _diagnostic_component(
                    template, "median_beat_template_correlation"
                ),
                "morph.skewness": _diagnostic_component(
                    skewness, "mean_channel_skewness"
                ),
                "morph.pearson_kurtosis": _diagnostic_component(
                    kurtosis, "mean_channel_pearson_kurtosis"
                ),
            }
        )
    reasons.extend(
        f"{name}:unavailable"
        for name, item in sorted(components.items())
        if not item.available
    )
    result = SqiDiagnostics(
        components=components,
        coverage=coverage,
        route=route.value,
        reasons=tuple(reasons),
    )
    result.validate()
    return result


def evaluate_quality(
    values: np.ndarray | CanonicalSignalViews,
    *,
    route: SignalRoute | None = None,
    pulse: PulseResult | None = None,
    imu_processed: dict[str, np.ndarray] | None = None,
    source_valid_mask: np.ndarray | None = None,
    qc_evidence: Mapping[str, object] | None = None,
    fs_hz: float = CANONICAL_FS_HZ,
    config: SqiConfig,
    calibrator: SqiCalibrator | None = None,
    detector_id: str | None = None,
    min_observation_sec: float = 8.0,
    min_peaks: int = 5,
    detector_parameters: Mapping[str, object] | None = None,
) -> QualityResult:
    """公共 Q_rate/Q_morph 入口 / Public endpoint-SQI entry point."""

    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("SQI requires exactly 400 Hz")
    config.validate()
    if (
        config.calibrator == "outer_train_empirical_quantiles_v1"
        and calibrator is None
    ):
        raise ValueError("formal empirical SQI requires a fitted outer-train calibrator")
    if calibrator is not None and calibrator.method != config.calibrator:
        raise ValueError("SQI calibrator method differs from resolved config")
    canonical_views: CanonicalSignalViews | None = None
    coverage_override: float | None = None
    if isinstance(values, CanonicalSignalViews):
        views = values
        canonical_views = views
        route = views.route
        matrix = views.analysis_signal
        imu_processed = views.imu_processed
        source_valid_mask = views.source_valid_mask
        if qc_evidence is None:
            candidate = views.metadata.get("ppg_qc_metrics")
            qc_evidence = candidate if isinstance(candidate, Mapping) else None
        rate_valid = views.rate_valid_mask
        if not np.all(rate_valid):
            # English: Endpoint metrics use one contiguous artifact-valid run;
            # coverage still describes the full aligned recording.
            # 中文：端点指标只用一段连续 artifact-valid run；coverage 仍描述完整记录。
            coverage_override = float(np.mean(rate_valid))
            padded = np.concatenate(([False], rate_valid, [False]))
            changes = np.diff(padded.astype(np.int8))
            runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
            if not runs:
                raise ValueError("SQI has no artifact-valid signal run")
            start, stop = max(runs, key=lambda bounds: bounds[1] - bounds[0])
            matrix = matrix[start:stop]
            source_valid_mask = np.asarray(source_valid_mask)[start:stop]
            if imu_processed is not None:
                sliced_imu: dict[str, np.ndarray] = {}
                full_length = rate_valid.size
                for key, item in imu_processed.items():
                    array = np.asarray(item)
                    sliced_imu[key] = (
                        array[start:stop]
                        if array.ndim >= 1 and array.shape[0] == full_length
                        else array
                    )
                imu_processed = sliced_imu
    else:
        matrix = np.asarray(values, dtype=np.float64)
        if route is None:
            raise ValueError("explicit array SQI requires an explicit route")
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[1] not in (1, 2) or not np.isfinite(matrix).all():
        raise ValueError("SQI input must be finite samples-by-one/two channels")
    assert route is not None
    if coverage_override is not None:
        coverage = coverage_override
    elif source_valid_mask is None:
        coverage = 1.0
    else:
        valid = np.asarray(source_valid_mask, dtype=bool)
        if valid.shape != matrix.shape:
            raise ValueError("source_valid_mask must match signal shape")
        coverage = float(np.mean(np.all(valid, axis=1)))

    cardiac, entropy = _welch_metrics(matrix, fs_hz, config)
    periodicity = _autocorrelation_periodicity(
        matrix,
        fs_hz,
        lag_min_s=config.ppi_min_s,
        lag_max_s=config.ppi_max_s,
    )
    red_ir = (
        abs(float(np.corrcoef(matrix[:, 0], matrix[:, 1])[0, 1]))
        if matrix.shape[1] == 2 and np.std(matrix[:, 0]) > 0 and np.std(matrix[:, 1]) > 0
        else float("nan")
    )
    if pulse is None:
        if detector_id is None:
            raise ValueError(
                "SQI requires a persisted detector_id when pulse is not supplied"
            )
        from ..peaks import detect_pulses

        pulse = detect_pulses(
            canonical_views if canonical_views is not None else matrix,
            detector_id=detector_id,
            fs_hz=fs_hz,
            min_observation_sec=min_observation_sec,
            min_peaks=min_peaks,
            source_route=route,
            detector_parameters=detector_parameters,
        )
    if pulse is None:
        peak_density, ppi_plausibility, ppi_stability = (float("nan"),) * 3
    else:
        duration = matrix.shape[0] / fs_hz
        peak_density = float(np.asarray(pulse.peaks).size / duration * 60.0)
        ppi = np.asarray(pulse.ppi_s, dtype=np.float64)
        valid_ppi = np.asarray(pulse.valid_interval_mask, dtype=bool)
        physiological_ppi = (
            valid_ppi
            & np.isfinite(ppi)
            & (ppi >= config.ppi_min_s)
            & (ppi <= config.ppi_max_s)
        )
        ppi_plausibility = float(np.mean(physiological_ppi)) if ppi.size else float("nan")
        selected = ppi[physiological_ppi]
        ppi_stability = (
            float(
                np.exp(
                    -max(
                        np.std(selected) / max(np.mean(selected), 1e-12),
                        0.0,
                    )
                    / config.ppi_cv_scale
                )
            )
            if selected.size >= config.ppi_stability_min_intervals
            else float("nan")
        )
    density_score = (
        1.0 if config.peak_density_min_bpm <= peak_density <= config.peak_density_max_bpm
        else float(
            np.exp(
                -abs(
                    peak_density
                    - np.clip(
                        peak_density,
                        config.peak_density_min_bpm,
                        config.peak_density_max_bpm,
                    )
                )
                / max(config.peak_density_min_bpm, 1e-12)
            )
        )
        if np.isfinite(peak_density) else float("nan")
    )
    motion_rms = float("nan")
    if imu_processed is not None and "dynamic_magnitude" in imu_processed:
        motion = np.asarray(imu_processed["dynamic_magnitude"], dtype=np.float64)
        if motion.shape[0] == matrix.shape[0]:
            finite_motion = motion[np.isfinite(motion)]
            if finite_motion.size:
                motion_rms = float(
                    np.sqrt(np.mean(np.square(finite_motion)))
                )
    flatline_score = float(min(np.std(matrix[:, column]) for column in range(matrix.shape[1])))
    qc_components = _qc_components(
        qc_evidence,
        fs_hz=fs_hz,
        flatline_duration_s=config.flatline_duration_s,
        long_gap_max_samples=config.long_gap_max_samples,
        clipping_fraction_reference=config.clipping_fraction_reference,
        saturation_fraction_reference=config.saturation_fraction_reference,
        component_pass_threshold=config.component_pass_threshold,
    )
    rate_components = {
        "cardiac_concentration": _component(
            cardiac,
            cardiac / config.cardiac_concentration_reference,
            "cardiac_band_power_fraction",
            pass_threshold=config.component_pass_threshold,
        ),
        "autocorrelation_periodicity": _component(
            periodicity,
            periodicity / config.autocorrelation_reference,
            "physiological_lag_acf",
            pass_threshold=config.component_pass_threshold,
        ),
        "normalized_spectral_entropy": _component(
            entropy,
            1.0 - entropy,
            "lower_entropy_is_better",
            pass_threshold=config.component_pass_threshold,
        ),
        "peak_density_bpm": _component(
            peak_density,
            density_score,
            "physiological_peak_density",
            pass_threshold=config.component_pass_threshold,
        ),
        "ppi_physiological_fraction": _component(
            ppi_plausibility,
            ppi_plausibility,
            "ppi_inside_configured_physical_range",
            pass_threshold=config.component_pass_threshold,
        ),
        "ppi_stability": _component(
            ppi_stability,
            ppi_stability,
            "low_ppi_cv",
            pass_threshold=config.component_pass_threshold,
        ),
        "red_ir_agreement": _component(
            red_ir,
            red_ir,
            "absolute_zero_lag_correlation",
            pass_threshold=config.component_pass_threshold,
        ),
        "motion_energy_rms": _component(
            motion_rms,
            (
                np.exp(-motion_rms / config.motion_rms_scale)
                if np.isfinite(motion_rms)
                else None
            ),
            "lower_dynamic_acc_is_better",
            pass_threshold=config.component_pass_threshold,
        ),
        "nonflat_scale": _component(
            flatline_score,
            1.0 if flatline_score > config.nonflat_std_threshold else 0.0,
            "nonzero_channel_variance",
            pass_threshold=config.component_pass_threshold,
        ),
        "source_coverage": _component(
            coverage,
            coverage,
            "finite_source_coverage",
            pass_threshold=config.component_pass_threshold,
        ),
        **qc_components,
    }
    rate_components = _calibrate_components(
        "rate",
        rate_components,
        calibrator,
        pass_threshold=config.component_pass_threshold,
    )
    q_rate = _endpoint(
        rate_components,
        dict(config.rate_component_weights),
        threshold=config.q_rate_threshold,
        coverage=coverage, minimum_coverage=config.minimum_coverage,
    )

    all_components = {f"rate.{name}": item for name, item in rate_components.items()}
    if route is SignalRoute.ARTIFACT_RATE_ONLY:
        q_shape = QualityEndpoint(
            score=None,
            state=QualityState.NOT_APPLICABLE,
            threshold=None,
            components={},
            reasons=("non_identity_x_ar_is_rate_only",),
            coverage=coverage,
        )
        q_morph = q_shape
    else:
        skewness = float(np.mean(stats.skew(matrix, axis=0, bias=False)))
        kurtosis = float(np.mean(stats.kurtosis(matrix, axis=0, fisher=False, bias=False)))
        template = (
            _template_correlation(
                matrix,
                pulse,
                fs_hz,
                half_width_s=config.template_half_width_s,
                minimum_peaks=config.template_min_peaks,
                minimum_beats=config.template_min_beats,
                resample_points=config.template_resample_points,
            )
            if pulse is not None
            else float("nan")
        )
        morph_components = {
            "template_correlation": _component(
                template,
                max(template, 0.0) if np.isfinite(template) else None,
                "median_beat_template",
                pass_threshold=config.component_pass_threshold,
            ),
            "skewness": _component(
                skewness,
                np.exp(-abs(skewness) / config.morph_skewness_scale),
                "moderate_skewness",
                pass_threshold=config.component_pass_threshold,
            ),
            "pearson_kurtosis": _component(
                kurtosis,
                np.exp(
                    -abs(kurtosis - config.morph_kurtosis_center)
                    / config.morph_kurtosis_scale
                ),
                "moderate_kurtosis",
                pass_threshold=config.component_pass_threshold,
            ),
            "red_ir_agreement": rate_components["red_ir_agreement"],
            "cardiac_concentration": rate_components["cardiac_concentration"],
            "nonflat_scale": rate_components["nonflat_scale"],
            "source_coverage": rate_components["source_coverage"],
            "flatline": rate_components["flatline"],
            "clipping": rate_components["clipping"],
            "saturation": rate_components["saturation"],
            "long_gap": rate_components["long_gap"],
        }
        morph_components = _calibrate_components(
            "morph",
            morph_components,
            calibrator,
            pass_threshold=config.component_pass_threshold,
        )
        q_shape = _endpoint(
            morph_components,
            dict(config.morph_component_weights),
            threshold=config.q_morph_threshold,
            coverage=coverage,
            minimum_coverage=config.minimum_coverage,
        )
        all_components.update({f"morph.{name}": item for name, item in morph_components.items()})
        morph_pass = (
            q_rate.state is QualityState.PASS
            and q_shape.state is QualityState.PASS
        )
        q_morph = QualityEndpoint(
            score=q_shape.score,
            state=QualityState.PASS if morph_pass else QualityState.FAIL,
            threshold=q_shape.threshold,
            components=q_shape.components,
            reasons=tuple(
                dict.fromkeys(
                    (
                        *q_shape.reasons,
                        *(
                            ()
                            if q_rate.state is QualityState.PASS
                            else ("q_morph_requires_q_rate_pass",)
                        ),
                    )
                )
            ),
            coverage=q_shape.coverage,
        )
    state = (
        "pass"
        if q_rate.state is QualityState.PASS
        and q_morph.state in {QualityState.PASS, QualityState.NOT_APPLICABLE}
        else "fail"
    )
    result = QualityResult(
        q_rate=q_rate,
        q_morph=q_morph,
        state=state,
        components=all_components,
        reasons=tuple(dict.fromkeys(q_rate.reasons + q_morph.reasons)),
        coverage=coverage,
        fitted_on_participant_ids=(
            calibrator.fitted_on_participant_ids
            if calibrator is not None
            else ()
        ),
        q_shape=q_shape,
    )
    result.validate_for_route(route)
    return result

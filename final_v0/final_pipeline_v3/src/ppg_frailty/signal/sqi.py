"""Endpoint-aware SQI：Q_rate 与 Q_morph / Endpoint-aware signal quality.

组件覆盖 cardiac concentration、自相关、模板相关、偏度、峰度、归一化谱熵、
完整 PPI 合理性、RED/IR 一致性、motion energy、coverage/flatline。

Components cover cardiac concentration, autocorrelation, template correlation,
skewness, kurtosis, normalized spectral entropy, full PPI plausibility, RED/IR
agreement, motion energy, coverage, and flatline evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

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
from .peaks import detect_pulses
from .views import CANONICAL_FS_HZ, CanonicalSignalViews


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

    def validate(self) -> None:
        """校验所有 SQI 阈值 / Validate every explicit SQI threshold."""

        if not 0.0 <= self.q_rate_threshold <= 1.0 or not 0.0 <= self.q_morph_threshold <= 1.0:
            raise ValueError("SQI endpoint thresholds must lie in [0,1]")
        if not 0.0 < self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must lie in (0,1]")
        if not 0.0 < self.cardiac_low_hz < self.cardiac_high_hz:
            raise ValueError("invalid SQI cardiac band")
        if not 0.0 < self.peak_density_min_bpm < self.peak_density_max_bpm:
            raise ValueError("invalid peak-density range")
        if not 0.0 < self.ppi_min_s < self.ppi_max_s:
            raise ValueError("invalid PPI range")
        if self.long_gap_max_samples < 0 or self.flatline_duration_s <= 0.0:
            raise ValueError("invalid flatline/long-gap SQI thresholds")

    @classmethod
    def from_resolved(cls, config: Mapping[str, object]) -> "SqiConfig":
        """从 resolved YAML 严格解析 / Strictly parse resolved YAML."""

        quality = config.get("quality")
        if not isinstance(quality, Mapping):
            raise ValueError("resolved config['quality'] is required")
        band = quality.get("cardiac_band_hz")
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ValueError("quality.cardiac_band_hz must contain two values")
        calibrator = str(quality.get("calibrator", ""))
        if calibrator not in {
            "outer_train_empirical_quantiles_v1",
            "fixed_formula_thresholds_v1",
        }:
            raise ValueError("unsupported SQI calibrator profile")
        density = quality.get("peak_density_bpm_range")
        ppi = quality.get("ppi_range_s")
        if not isinstance(density, (list, tuple)) or len(density) != 2:
            raise ValueError("quality.peak_density_bpm_range must contain two values")
        if not isinstance(ppi, (list, tuple)) or len(ppi) != 2:
            raise ValueError("quality.ppi_range_s must contain two values")
        result = cls(
            q_rate_threshold=float(quality["rate_threshold"]),
            q_morph_threshold=float(quality["morph_threshold"]),
            cardiac_low_hz=float(band[0]),
            cardiac_high_hz=float(band[1]),
            peak_density_min_bpm=float(density[0]),
            peak_density_max_bpm=float(density[1]),
            ppi_min_s=float(ppi[0]),
            ppi_max_s=float(ppi[1]),
            long_gap_max_samples=int(quality["long_gap_max_samples"]),
            flatline_duration_s=float(quality["flatline_duration_s"]),
            calibrator=calibrator,
        )
        result.validate()
        return result


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

    def validate(self) -> None:
        if not 0.0 < self.cardiac_low_hz < self.cardiac_high_hz:
            raise ValueError("invalid diagnostic cardiac band")
        if not 0.0 < self.peak_density_min_bpm < self.peak_density_max_bpm:
            raise ValueError("invalid diagnostic peak-density range")
        if not 0.0 < self.ppi_min_s < self.ppi_max_s:
            raise ValueError("invalid diagnostic PPI range")
        if self.long_gap_max_samples < 0 or self.flatline_duration_s <= 0.0:
            raise ValueError("invalid diagnostic gap/flatline parameters")

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


def _component(raw: float | None, normalized: float | None, reason: str) -> QualityComponent:
    """构造有限 component 或 unavailable / Build a finite or unavailable component."""

    if raw is None or normalized is None or not np.isfinite(raw) or not np.isfinite(normalized):
        return QualityComponent(None, None, QualityState.UNAVAILABLE, reason)
    score = float(np.clip(normalized, 0.0, 1.0))
    state = QualityState.PASS if score >= 0.5 else QualityState.FAIL
    return QualityComponent(float(raw), score, state, reason)


def _welch_metrics(
    values: np.ndarray, fs_hz: float, config: SqiConfig
) -> tuple[float, float]:
    """返回 cardiac concentration 与归一化谱熵 / Return spectral SQI metrics."""

    concentrations: list[float] = []
    entropies: list[float] = []
    for column in range(values.shape[1]):
        frequencies, power = signal.welch(
            values[:, column], fs=fs_hz, nperseg=min(2048, values.shape[0])
        )
        usable = (frequencies >= 0.2) & (frequencies <= 8.0)
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


def _autocorrelation_periodicity(values: np.ndarray, fs_hz: float) -> float:
    """在 0.3–2.0 s lag 找最大 normalized ACF / Maximum physiological-lag ACF."""

    scores: list[float] = []
    low, high = int(round(0.30 * fs_hz)), int(round(2.00 * fs_hz))
    for column in range(values.shape[1]):
        x = values[:, column] - np.mean(values[:, column])
        denominator = float(np.dot(x, x))
        if denominator <= 1e-12 or x.size <= high:
            continue
        corr = signal.correlate(x, x, mode="full", method="fft")[x.size - 1 :] / denominator
        scores.append(float(np.max(corr[low : high + 1])))
    return float(np.mean(scores)) if scores else float("nan")


def _template_correlation(values: np.ndarray, pulse: PulseResult, fs_hz: float) -> float:
    """用 median beat template 计算逐搏相关 / Median beat-template correlation."""

    peaks = np.asarray(pulse.peaks, dtype=np.int64)
    accepted = np.asarray(pulse.accepted_peak_mask, dtype=bool)
    if peaks.size < 5:
        return float("nan")
    channel = 0 if pulse.wavelength.upper() == "RED" else min(1, values.shape[1] - 1)
    width = int(round(0.30 * fs_hz))
    target = np.linspace(0.0, 1.0, 101)
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
    if len(beats) < 3:
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
        if name in weights and component.normalized_value is not None
    ]
    if not usable:
        return QualityEndpoint(
            None, QualityState.UNAVAILABLE, threshold, components,
            ("no_available_sqi_components",), coverage,
        )
    denominator = float(sum(weight for weight, _ in usable))
    score = float(sum(weight * float(value) for weight, value in usable) / denominator)
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
                    if normalized >= 0.5
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
        ),
        "clipping": _component(
            clipping_occupancy,
            1.0 - min(clipping_occupancy / 0.02, 1.0),
            "extreme_occupancy_heuristic_adc_rails_unknown",
        ),
        "long_gap": _component(
            longest_gap,
            1.0 if longest_gap <= float(long_gap_max_samples) else 0.0,
            "longest_nonfinite_gap_vs_resolved_samples",
        ),
        "saturation": unavailable["saturation"],
    }
    saturation = evidence.get("adc_saturation_fraction")
    if saturation is not None and np.isfinite(float(saturation)):
        fraction = float(saturation)
        output["saturation"] = _component(
            fraction,
            1.0 - min(fraction / 0.02, 1.0),
            "declared_adc_rail_saturation_fraction",
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

    cardiac, entropy = _welch_metrics(matrix, fs_hz, physical)  # type: ignore[arg-type]
    periodicity = _autocorrelation_periodicity(matrix, fs_hz)
    red_ir = (
        abs(float(np.corrcoef(matrix[:, 0], matrix[:, 1])[0, 1]))
        if matrix.shape[1] == 2
        and np.std(matrix[:, 0]) > 0.0
        and np.std(matrix[:, 1]) > 0.0
        else float("nan")
    )
    if pulse is None:
        try:
            pulse = detect_pulses(matrix, fs_hz=fs_hz)
        except ValueError:
            pulse = None
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
        if selected.size >= 3 and float(np.mean(selected)) > 0.0:
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
            _template_correlation(matrix, pulse, fs_hz)
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
    periodicity = _autocorrelation_periodicity(matrix, fs_hz)
    red_ir = (
        abs(float(np.corrcoef(matrix[:, 0], matrix[:, 1])[0, 1]))
        if matrix.shape[1] == 2 and np.std(matrix[:, 0]) > 0 and np.std(matrix[:, 1]) > 0
        else float("nan")
    )
    if pulse is None:
        try:
            pulse = detect_pulses(
                canonical_views if canonical_views is not None else matrix,
                fs_hz=fs_hz,
            )
        except ValueError:
            pulse = None
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
            float(np.exp(-max(np.std(selected) / max(np.mean(selected), 1e-12), 0.0) / 0.20))
            if selected.size >= 3 else float("nan")
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
    )
    rate_components = {
        "cardiac_concentration": _component(cardiac, cardiac / 0.65, "cardiac_band_power_fraction"),
        "autocorrelation_periodicity": _component(periodicity, periodicity / 0.70, "physiological_lag_acf"),
        "normalized_spectral_entropy": _component(entropy, 1.0 - entropy, "lower_entropy_is_better"),
        "peak_density_bpm": _component(peak_density, density_score, "physiological_peak_density"),
        "ppi_physiological_fraction": _component(ppi_plausibility, ppi_plausibility, "ppi_0p3_to_2s"),
        "ppi_stability": _component(ppi_stability, ppi_stability, "low_ppi_cv"),
        "red_ir_agreement": _component(red_ir, red_ir, "absolute_zero_lag_correlation"),
        "motion_energy_rms": _component(motion_rms, np.exp(-motion_rms / 4.0) if np.isfinite(motion_rms) else None, "lower_dynamic_acc_is_better"),
        "nonflat_scale": _component(flatline_score, 1.0 if flatline_score > 1e-10 else 0.0, "nonzero_channel_variance"),
        "source_coverage": _component(coverage, coverage, "finite_source_coverage"),
        **qc_components,
    }
    rate_components = _calibrate_components(
        "rate", rate_components, calibrator
    )
    rate_weights = {
        "cardiac_concentration": 0.20, "autocorrelation_periodicity": 0.15,
        "normalized_spectral_entropy": 0.10, "peak_density_bpm": 0.08,
        "ppi_physiological_fraction": 0.15, "ppi_stability": 0.12,
        "red_ir_agreement": 0.08, "motion_energy_rms": 0.05,
        "nonflat_scale": 0.02, "source_coverage": 0.04,
        "flatline": 0.02, "clipping": 0.015,
        "saturation": 0.015, "long_gap": 0.01,
    }
    q_rate = _endpoint(
        rate_components, rate_weights, threshold=config.q_rate_threshold,
        coverage=coverage, minimum_coverage=config.minimum_coverage,
    )

    all_components = {f"rate.{name}": item for name, item in rate_components.items()}
    if route is SignalRoute.ARTIFACT_RATE_ONLY:
        q_morph = QualityEndpoint(
            score=None,
            state=QualityState.NOT_APPLICABLE,
            threshold=None,
            components={},
            reasons=("non_identity_x_ar_is_rate_only",),
            coverage=coverage,
        )
    else:
        skewness = float(np.mean(stats.skew(matrix, axis=0, bias=False)))
        kurtosis = float(np.mean(stats.kurtosis(matrix, axis=0, fisher=False, bias=False)))
        template = _template_correlation(matrix, pulse, fs_hz) if pulse is not None else float("nan")
        morph_components = {
            "template_correlation": _component(template, max(template, 0.0) if np.isfinite(template) else None, "median_beat_template"),
            "skewness": _component(skewness, np.exp(-abs(skewness) / 3.0), "moderate_skewness"),
            "pearson_kurtosis": _component(kurtosis, np.exp(-abs(kurtosis - 3.0) / 5.0), "moderate_kurtosis"),
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
            "morph", morph_components, calibrator
        )
        q_morph = _endpoint(
            morph_components,
            {"template_correlation": 0.30, "skewness": 0.08, "pearson_kurtosis": 0.08,
             "red_ir_agreement": 0.18, "cardiac_concentration": 0.16,
             "nonflat_scale": 0.04, "source_coverage": 0.12,
             "flatline": 0.03, "clipping": 0.02,
             "saturation": 0.02, "long_gap": 0.02},
            threshold=config.q_morph_threshold,
            coverage=coverage,
            minimum_coverage=config.minimum_coverage,
        )
        all_components.update({f"morph.{name}": item for name, item in morph_components.items()})
    state = "pass" if q_rate.state is QualityState.PASS and q_morph.state in {QualityState.PASS, QualityState.NOT_APPLICABLE} else "fail"
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
    )
    result.validate_for_route(route)
    return result

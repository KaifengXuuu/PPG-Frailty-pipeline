"""V2 named historical-derived reducers / V2 具名历史派生伪影削减器。

English: These reducers preserve audited algorithms and parameters found in repository
history. Every successful output is non-identity and rate-only; none is promoted to the
default reducer or claimed to preserve morphology.
中文：这些 reducer 保留仓库历史中的可审计算法和参数。所有成功输出均为非恒等、
仅速率特征支线；它们不会自动晋升为默认 reducer，也不声称保留形态学。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Mapping

import numpy as np
from scipy import interpolate, signal

from ..contracts import ArtifactReductionResult
from ..signal.views import CANONICAL_FS_HZ
from .base import ArtifactReducer, failure_result, parameters_dict, success_result, validate_ppg


HISTORICAL_EMD_SOURCE = "funcs.py:emd_sift"
HISTORICAL_CEEMD_NLMS_SOURCE = "funcs.py:ceemd_reference+nlms_anc+remove_ma_cemd_lms"
HISTORICAL_DWT_SOURCE = "pttppg_pipeline_v7.py:dwt_compress"


@dataclass(frozen=True)
class EmdSiftingConfig:
    """Frozen EMD sifting parameters / 冻结的 EMD 筛分参数。"""

    max_imfs: int = 6
    max_sift: int = 10
    sd_threshold: float = 0.2


@dataclass(frozen=True)
class CeemdLiteNlmsLegacyConfig:
    """Frozen CEEMD-lite and leaky-NLMS parameters / 冻结历史参数。"""

    pairs: int = 6
    noise_ratio: float = 0.2
    max_imfs: int = 6
    max_sift: int = 10
    sd_threshold: float = 0.2
    protect_bandwidth_hz: float = 0.25
    protect_harmonics: int = 2
    low_motion_hz: float = 0.4
    high_motion_hz: float = 6.0
    nlms_length: int = 32
    nlms_mu: float = 0.1
    nlms_leak: float = 1e-4
    random_seed: int = 2025


@dataclass(frozen=True)
class DwtA2LegacyConfig:
    """Frozen DWT A2 parameters / 冻结的 DWT A2 参数。"""

    wavelet: str = "db4"
    level: int = 2


def _validate_emd_config(config: EmdSiftingConfig) -> None:
    if config.max_imfs < 1 or config.max_sift < 1:
        raise ValueError("EMD max_imfs/max_sift must be positive")
    if not 0.0 < config.sd_threshold < 1.0:
        raise ValueError("EMD sd_threshold must be inside (0, 1)")


def _local_extrema(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match frozen derivative-sign extrema / 匹配历史导数符号极值点。"""

    derivative = np.diff(values)
    maxima = np.where((np.hstack((derivative, 0.0)) < 0.0) & (np.hstack((0.0, derivative)) > 0.0))[0]
    minima = np.where((np.hstack((derivative, 0.0)) > 0.0) & (np.hstack((0.0, derivative)) < 0.0))[0]
    maxima = maxima[(maxima > 1) & (maxima < values.size - 2)]
    minima = minima[(minima > 1) & (minima < values.size - 2)]
    return maxima, minima


def _sift_mean(values: np.ndarray, time_s: np.ndarray) -> np.ndarray | None:
    """Build natural-cubic envelopes / 构造自然三次样条包络均值。"""

    maxima, minima = _local_extrema(values)
    if maxima.size < 2 or minima.size < 2:
        return None
    max_knots = np.r_[0, maxima, values.size - 1]
    min_knots = np.r_[0, minima, values.size - 1]
    upper = interpolate.CubicSpline(time_s[max_knots], values[max_knots], bc_type="natural")(time_s)
    lower = interpolate.CubicSpline(time_s[min_knots], values[min_knots], bc_type="natural")(time_s)
    return 0.5 * (upper + lower)


def _emd_sift(
    values: np.ndarray,
    fs_hz: float,
    config: EmdSiftingConfig,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Port frozen sifting equations / 移植冻结的筛分方程。"""

    _validate_emd_config(config)
    source = np.asarray(values, dtype=np.float64).ravel()
    time_s = np.arange(source.size, dtype=np.float64) / float(fs_hz)
    imfs: list[np.ndarray] = []
    residual = source.copy()
    for _ in range(config.max_imfs):
        candidate = residual.copy()
        for _ in range(config.max_sift):
            envelope_mean = _sift_mean(candidate, time_s)
            if envelope_mean is None:
                break
            previous = candidate
            candidate = candidate - envelope_mean
            sd = float(np.sum(np.square(previous - candidate)) / (np.sum(np.square(previous)) + 1e-18))
            if sd < config.sd_threshold:
                break
        maxima, minima = _local_extrema(candidate)
        if maxima.size + minima.size < 2:
            break
        imfs.append(candidate)
        residual = residual - candidate
        residual_maxima, residual_minima = _local_extrema(residual)
        if residual_maxima.size < 1 or residual_minima.size < 1:
            break
    return imfs, residual


def _welch_peak(values: np.ndarray, fs_hz: float, low: float, high: float) -> float:
    """Return dominant Welch frequency / 返回频带主峰。"""

    segment = min(values.size, max(8, int(8.0 * fs_hz)))
    frequencies, power = signal.welch(values, fs=fs_hz, window="hann", nperseg=segment)
    mask = (frequencies >= low) & (frequencies <= high)
    if not np.any(mask):
        return float("nan")
    return float(frequencies[mask][int(np.argmax(power[mask]))])


def _ceemd_reference(
    values: np.ndarray,
    fs_hz: float,
    config: CeemdLiteNlmsLegacyConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build frozen complementary-noise reference / 构造历史运动参考。"""

    if config.pairs < 1 or config.noise_ratio <= 0.0:
        raise ValueError("CEEMD pairs/noise_ratio must be positive")
    emd_config = EmdSiftingConfig(config.max_imfs, config.max_sift, config.sd_threshold)
    source = np.asarray(values, dtype=np.float64).ravel()
    sigma = float(np.std(source) + 1e-12)
    accumulated: list[np.ndarray] | None = None
    max_levels = 0
    successful_realizations = 0
    rng = np.random.default_rng(config.random_seed)
    for _ in range(config.pairs):
        noise = rng.standard_normal(source.size) * (config.noise_ratio * sigma)
        for sign_value in (1.0, -1.0):
            imfs, _ = _emd_sift(source + sign_value * noise, fs_hz, emd_config)
            if not imfs:
                continue
            successful_realizations += 1
            if accumulated is None:
                max_levels = len(imfs)
                accumulated = [component.astype(np.float64) for component in imfs]
                continue
            if len(imfs) > max_levels:
                accumulated.extend(np.zeros_like(source) for _ in range(len(imfs) - max_levels))
                max_levels = len(imfs)
            for index, component in enumerate(imfs):
                accumulated[index] = accumulated[index] + component
    if accumulated is None or max_levels == 0:
        raise ValueError("CEEMD-lite produced no IMF; zero-reference fallback is forbidden")
    averaged = [component / (2.0 * config.pairs) for component in accumulated]
    residual = source - np.sum(np.vstack(averaged), axis=0)
    heart_rate_hz = _welch_peak(source, fs_hz, 0.6, 3.5)
    motion_indices: list[int] = []
    cardiac_indices: list[int] = []
    nyquist = 0.5 * fs_hz
    band_b, band_a = signal.butter(2, [0.6 / nyquist, 3.5 / nyquist], btype="band")
    heart_band = signal.filtfilt(band_b, band_a, source)
    for index, component in enumerate(averaged):
        dominant = _welch_peak(component, fs_hz, 0.0, 8.0)
        cardiac = np.isfinite(heart_rate_hz) and any(
            abs(dominant - harmonic * heart_rate_hz) <= config.protect_bandwidth_hz
            for harmonic in range(1, config.protect_harmonics + 1)
        )
        if cardiac:
            cardiac_indices.append(index)
        elif dominant <= config.low_motion_hz or dominant >= config.high_motion_hz:
            motion_indices.append(index)
        else:
            correlation = float(np.corrcoef(component, heart_band)[0, 1])
            (cardiac_indices if abs(correlation) >= 0.2 else motion_indices).append(index)
    reference = np.zeros_like(source)
    for index in motion_indices:
        reference += averaged[index]
    residual_peak = _welch_peak(residual, fs_hz, 0.0, 2.0)
    if np.isfinite(residual_peak) and residual_peak < 0.4:
        reference += residual
    if not np.isfinite(reference).all() or float(np.std(reference)) <= 1e-12:
        raise ValueError("CEEMD-lite produced no usable motion reference")
    return reference, {
        "successful_realizations": successful_realizations,
        "imf_count": len(averaged),
        "motion_imf_indices": motion_indices,
        "cardiac_imf_indices": cardiac_indices,
        "estimated_hr_hz": heart_rate_hz,
    }


def _nlms_clean(
    values: np.ndarray,
    reference: np.ndarray,
    config: CeemdLiteNlmsLegacyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run frozen leaky normalized LMS / 运行历史泄漏归一化 LMS。"""

    if config.nlms_length < 1 or not 0.0 < config.nlms_mu < 2.0:
        raise ValueError("NLMS length/mu is invalid")
    source = np.asarray(values, dtype=np.float64).ravel()
    motion = np.asarray(reference, dtype=np.float64).ravel()
    estimate = np.zeros(source.size, dtype=np.float64)
    clean = np.zeros(source.size, dtype=np.float64)
    weights = np.zeros(config.nlms_length, dtype=np.float64)
    buffer = np.zeros(config.nlms_length, dtype=np.float64)
    for sample in range(source.size):
        buffer[1:] = buffer[:-1]
        buffer[0] = motion[sample]
        estimate[sample] = float(np.dot(weights, buffer))
        clean[sample] = source[sample] - estimate[sample]
        denominator = 1e-6 + float(np.dot(buffer, buffer))
        weights = (
            (1.0 - config.nlms_leak) * weights
            + (config.nlms_mu * clean[sample] * buffer) / denominator
        )
    if not np.isfinite(clean).all():
        raise ValueError("NLMS produced nonfinite output")
    return clean, estimate


class EmdSiftingRateOnlyReducer(ArtifactReducer):
    """EMD reconstruction without residual / 排除 residual 的 EMD 重构。"""

    reducer_id = "emd_sifting_rate_only"
    reducer_version = "historical_derived_funcs_emd_v2"
    is_identity = False

    def __init__(self, config: EmdSiftingConfig | None = None) -> None:
        self.config = config or EmdSiftingConfig()
        _validate_emd_config(self.config)

    def reduce(self, ppg: np.ndarray, imu_processed: Mapping[str, np.ndarray] | None, *, fs_hz: float = CANONICAL_FS_HZ) -> ArtifactReductionResult:
        """Remove each residual; fail if no IMF / 无 IMF 时故障闭合。"""

        try:
            source = validate_ppg(ppg, fs_hz=fs_hz)
            output = np.empty_like(source)
            counts: list[int] = []
            residual_fraction: list[float] = []
            for channel in range(2):
                imfs, residual = _emd_sift(source[:, channel], fs_hz, self.config)
                if not imfs:
                    raise ValueError(f"EMD produced no IMF for channel {channel}")
                output[:, channel] = np.sum(np.vstack(imfs), axis=0)
                counts.append(len(imfs))
                residual_fraction.append(float(np.linalg.norm(residual) / (np.linalg.norm(source[:, channel]) + 1e-18)))
            return success_result(
                self, output, input_ppg=source, confidence=0.5,
                parameters=parameters_dict(self.config),
                diagnostics={
                    "historical_source": HISTORICAL_EMD_SOURCE,
                    "historical_derivation": "sum_imfs_excluding_residual",
                    "representation": "feature_vector",
                    "rate_only": True,
                    "morphology_preserved": False,
                    "imf_count_by_channel": counts,
                    "residual_norm_fraction_by_channel": residual_fraction,
                    "output_valid_mask": np.ones(source.shape[0], dtype=bool).tolist(),
                    "output_valid_fraction": 1.0,
                },
            )
        except (ValueError, FloatingPointError) as exc:
            return failure_result(self, str(exc), parameters=parameters_dict(self.config), diagnostics={"historical_source": HISTORICAL_EMD_SOURCE, "rate_only": True})


class CeemdLiteNlmsLegacyReducer(ArtifactReducer):
    """Frozen CEEMD-lite plus NLMS / 冻结 CEEMD-lite+NLMS。"""

    reducer_id = "ceemd_lite_nlms_legacy"
    reducer_version = "historical_funcs_ceemd_nlms_v2"
    is_identity = False

    def __init__(self, config: CeemdLiteNlmsLegacyConfig | None = None) -> None:
        self.config = config or CeemdLiteNlmsLegacyConfig()

    def reduce(self, ppg: np.ndarray, imu_processed: Mapping[str, np.ndarray] | None, *, fs_hz: float = CANONICAL_FS_HZ) -> ArtifactReductionResult:
        """Run channels independently, matching history / 双通道独立运行。"""

        try:
            source = validate_ppg(ppg, fs_hz=fs_hz)
            output = np.empty_like(source)
            channel_diagnostics: list[dict[str, object]] = []
            removed_fraction: list[float] = []
            for channel in range(2):
                reference, ceemd_diagnostics = _ceemd_reference(source[:, channel], fs_hz, self.config)
                clean, estimate = _nlms_clean(source[:, channel], reference, self.config)
                output[:, channel] = clean
                removed_fraction.append(float(np.linalg.norm(estimate) / (np.linalg.norm(source[:, channel]) + 1e-18)))
                channel_diagnostics.append(ceemd_diagnostics)
            return success_result(
                self, output, input_ppg=source, confidence=0.5,
                parameters=parameters_dict(self.config),
                diagnostics={
                    "historical_source": HISTORICAL_CEEMD_NLMS_SOURCE,
                    "representation": "feature_vector",
                    "rate_only": True,
                    "morphology_preserved": False,
                    "channel_diagnostics": channel_diagnostics,
                    "removed_norm_fraction_by_channel": removed_fraction,
                    "output_valid_mask": np.ones(source.shape[0], dtype=bool).tolist(),
                    "output_valid_fraction": 1.0,
                },
            )
        except (ValueError, FloatingPointError) as exc:
            return failure_result(self, str(exc), parameters=parameters_dict(self.config), diagnostics={"historical_source": HISTORICAL_CEEMD_NLMS_SOURCE, "rate_only": True})


class DwtA2LegacyReducer(ArtifactReducer):
    """Legacy db4 level-2 approximation interpolation / 历史 DWT A2 插值。"""

    reducer_id = "dwt_a2_legacy"
    reducer_version = "historical_pttppg_v7_dwt_a2_v2"
    is_identity = False

    def __init__(self, config: DwtA2LegacyConfig | None = None) -> None:
        self.config = config or DwtA2LegacyConfig()
        if self.config.wavelet != "db4" or self.config.level != 2:
            raise ValueError("dwt_a2_legacy is frozen to wavelet=db4, level=2")

    def reduce(self, ppg: np.ndarray, imu_processed: Mapping[str, np.ndarray] | None, *, fs_hz: float = CANONICAL_FS_HZ) -> ArtifactReductionResult:
        """Fail unavailable without PyWavelets / 缺依赖时绝不回退原信号。"""

        try:
            source = validate_ppg(ppg, fs_hz=fs_hz)
            try:
                pywt = importlib.import_module("pywt")
            except (ImportError, ModuleNotFoundError) as exc:
                return failure_result(
                    self, f"optional dependency PyWavelets is unavailable: {exc}",
                    status="unsupported", parameters=parameters_dict(self.config),
                    diagnostics={"historical_source": HISTORICAL_DWT_SOURCE, "rate_only": True, "identity_fallback_forbidden": True},
                )
            output = np.empty_like(source)
            approximation_lengths: list[int] = []
            query = np.linspace(0.0, 1.0, source.shape[0])
            for channel in range(2):
                coefficients = pywt.wavedec(source[:, channel], self.config.wavelet, level=self.config.level)
                approximation = np.asarray(coefficients[0], dtype=np.float64)
                if approximation.size < 2 or not np.isfinite(approximation).all():
                    raise ValueError(f"DWT A2 unavailable for channel {channel}")
                approximation_lengths.append(int(approximation.size))
                knots = np.linspace(0.0, 1.0, approximation.size)
                output[:, channel] = np.interp(query, knots, approximation)
            return success_result(
                self, output, input_ppg=source, confidence=0.5,
                parameters=parameters_dict(self.config),
                diagnostics={
                    "historical_source": HISTORICAL_DWT_SOURCE,
                    "historical_operation": "wavedec_A2_linear_interpolation",
                    "representation": "feature_vector",
                    "rate_only": True,
                    "morphology_preserved": False,
                    "approximation_length_by_channel": approximation_lengths,
                    "output_valid_mask": np.ones(source.shape[0], dtype=bool).tolist(),
                    "output_valid_fraction": 1.0,
                },
            )
        except (ValueError, FloatingPointError) as exc:
            return failure_result(self, str(exc), parameters=parameters_dict(self.config), diagnostics={"historical_source": HISTORICAL_DWT_SOURCE, "rate_only": True})


__all__ = [
    "CeemdLiteNlmsLegacyConfig", "CeemdLiteNlmsLegacyReducer",
    "DwtA2LegacyConfig", "DwtA2LegacyReducer",
    "EmdSiftingConfig", "EmdSiftingRateOnlyReducer",
    "HISTORICAL_CEEMD_NLMS_SOURCE", "HISTORICAL_DWT_SOURCE", "HISTORICAL_EMD_SOURCE",
]

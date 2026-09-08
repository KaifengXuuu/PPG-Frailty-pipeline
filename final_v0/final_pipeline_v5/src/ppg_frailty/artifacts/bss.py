"""双波长 PCA/FastICA/NMF 盲源分离 / Dual-wavelength blind source separation.

所有 BSS 都要求两路同步 PPG；单通道输入 fail closed。输出仅用于 rate recovery，
不宣称保留 amplitude 或 morphology。All outputs are rate-only and require two channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import warnings

import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA, NMF, PCA
from sklearn.exceptions import ConvergenceWarning

from ..contracts import ArtifactReductionResult
from ..signal.views import CANONICAL_FS_HZ
from .base import (
    ArtifactReducer,
    IMU_REFERENCE_AXES6_PROFILE_ID,
    IMU_REFERENCE_DERIVED9_AUGMENTATION_PROFILE_ID,
    failure_result,
    imu_reference_matrix,
    parameters_dict,
    success_result,
    validate_ppg,
)


@dataclass(frozen=True)
class BssConfig:
    """Legacy aggregate input retained only for source compatibility.

    New configuration should use the reducer-specific classes below.  Reducer
    constructors reject non-default values that their algorithm cannot consume.
    """

    random_state: int = 42
    max_iter: int = 1000
    tolerance: float = 1e-5
    nmf_rank: int = 2
    nperseg: int = 512
    overlap_fraction: float = 0.75
    imu_reference_profile: str = IMU_REFERENCE_AXES6_PROFILE_ID

    def validate(self) -> None:
        """验证 BSS 数值配置 / Validate numerical BSS configuration."""

        if self.max_iter <= 0 or self.tolerance <= 0.0 or self.nmf_rank <= 0:
            raise ValueError("BSS iteration/tolerance/rank must be positive")
        if self.nperseg < 32 or not 0.0 <= self.overlap_fraction < 1.0:
            raise ValueError("invalid NMF-STFT configuration")
        if self.imu_reference_profile not in {
                IMU_REFERENCE_AXES6_PROFILE_ID,
                IMU_REFERENCE_DERIVED9_AUGMENTATION_PROFILE_ID,
        }:
            raise ValueError("unknown BSS IMU reference profile")


def _validate_seed_iterations(
    random_state: int,
    max_iter: int,
    tolerance: float,
) -> None:
    if (isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer))
            or not 0 <= int(random_state) <= 0xFFFF_FFFF):
        raise ValueError("BSS random_state must be an integer in [0,2^32-1]")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
        raise ValueError("BSS max_iter must be an integer")
    if int(max_iter) <= 0 or not np.isfinite(tolerance) or float(tolerance) <= 0.0:
        raise ValueError("BSS iteration/tolerance must be positive")


def _validate_imu_profile(value: str) -> None:
    if value not in {
            IMU_REFERENCE_AXES6_PROFILE_ID,
            IMU_REFERENCE_DERIVED9_AUGMENTATION_PROFILE_ID,
    }:
        raise ValueError("unknown BSS IMU reference profile")


@dataclass(frozen=True)
class PcaBssConfig:
    """Only the motion-reference selector consumed by deterministic PCA."""

    imu_reference_profile: str = IMU_REFERENCE_AXES6_PROFILE_ID

    def validate(self) -> None:
        _validate_imu_profile(self.imu_reference_profile)


@dataclass(frozen=True)
class FastIcaBssConfig:
    """Parameters actually consumed by the FastICA reducer."""

    random_state: int = 42
    max_iter: int = 1000
    tolerance: float = 1e-5
    imu_reference_profile: str = IMU_REFERENCE_AXES6_PROFILE_ID

    def validate(self) -> None:
        _validate_seed_iterations(self.random_state, self.max_iter, self.tolerance)
        _validate_imu_profile(self.imu_reference_profile)


@dataclass(frozen=True)
class NmfBssConfig:
    """Parameters actually consumed by the spectral NMF reducer."""

    random_state: int = 42
    max_iter: int = 1000
    tolerance: float = 1e-5
    nmf_rank: int = 2
    nperseg: int = 512
    overlap_fraction: float = 0.75

    def validate(self) -> None:
        _validate_seed_iterations(self.random_state, self.max_iter, self.tolerance)
        if (isinstance(self.nmf_rank, bool) or not isinstance(self.nmf_rank, (int, np.integer))
                or int(self.nmf_rank) <= 0):
            raise ValueError("NMF rank must be a positive integer")
        if isinstance(self.nperseg, bool) or not isinstance(self.nperseg, (int, np.integer)) or int(self.nperseg) < 32:
            raise ValueError("NMF nperseg must be an integer >=32")
        if not np.isfinite(self.overlap_fraction) or not 0.0 <= float(self.overlap_fraction) < 1.0:
            raise ValueError("NMF overlap_fraction must lie in [0,1)")


def _pca_config(config: PcaBssConfig | BssConfig) -> PcaBssConfig:
    if isinstance(config, PcaBssConfig):
        return config
    defaults = BssConfig()
    inactive = (
        "random_state",
        "max_iter",
        "tolerance",
        "nmf_rank",
        "nperseg",
        "overlap_fraction",
    )
    if any(getattr(config, name) != getattr(defaults, name) for name in inactive):
        raise ValueError("PCA BSS received parameters that PCA does not consume")
    return PcaBssConfig(imu_reference_profile=config.imu_reference_profile)


def _fastica_config(config: FastIcaBssConfig | BssConfig) -> FastIcaBssConfig:
    if isinstance(config, FastIcaBssConfig):
        return config
    defaults = BssConfig()
    inactive = ("nmf_rank", "nperseg", "overlap_fraction")
    if any(getattr(config, name) != getattr(defaults, name) for name in inactive):
        raise ValueError("FastICA BSS received NMF-only parameters")
    return FastIcaBssConfig(
        random_state=config.random_state,
        max_iter=config.max_iter,
        tolerance=config.tolerance,
        imu_reference_profile=config.imu_reference_profile,
    )


def _nmf_config(config: NmfBssConfig | BssConfig) -> NmfBssConfig:
    if isinstance(config, NmfBssConfig):
        return config
    if config.imu_reference_profile != BssConfig().imu_reference_profile:
        raise ValueError("NMF BSS does not consume an IMU reference profile")
    return NmfBssConfig(
        random_state=config.random_state,
        max_iter=config.max_iter,
        tolerance=config.tolerance,
        nmf_rank=config.nmf_rank,
        nperseg=config.nperseg,
        overlap_fraction=config.overlap_fraction,
    )


def _cardiac_fraction(values: np.ndarray, fs_hz: float) -> float:
    """component 的 0.5–3.5 Hz 能量比例 / Cardiac-band component fraction."""

    frequencies, power = signal.welch(values, fs=fs_hz, nperseg=min(1024, values.size))
    usable = (frequencies >= 0.2) & (frequencies <= 8.0)
    cardiac = (frequencies >= 0.5) & (frequencies <= 3.5)
    total = float(np.sum(power[usable]))
    return float(np.sum(power[cardiac]) / total) if total > 0.0 else 0.0


def _select_source(
    sources: np.ndarray,
    motion_references: np.ndarray | None,
    fs_hz: float,
    *,
    motion_valid_mask: np.ndarray | None = None,
) -> tuple[int, list[float], list[float], list[float]]:
    """按 cardiac concentration 减 motion correlation 选源 / Select a rate source."""

    valid = (np.ones(sources.shape[0], dtype=bool) if motion_valid_mask is None else np.asarray(motion_valid_mask,
                                                                                                dtype=bool))
    if valid.shape != (sources.shape[0], ) or not np.any(valid):
        raise ValueError("motion_valid_mask must retain aligned reference samples")
    cardiac_scores: list[float] = []
    motion_correlations: list[float] = []
    combined_scores: list[float] = []
    for column in range(sources.shape[1]):
        component = sources[:, column]
        cardiac = _cardiac_fraction(component, fs_hz)
        correlation = 0.0
        component_for_correlation = component[valid]
        references_for_correlation = None if motion_references is None else motion_references[valid]
        if references_for_correlation is not None and np.std(component_for_correlation) > 1e-12:
            correlations = [
                abs(float(np.corrcoef(
                    component_for_correlation,
                    references_for_correlation[:, index],
                )[0, 1])) for index in range(references_for_correlation.shape[1])
                if np.std(references_for_correlation[:, index]) > 1e-12
            ]
            correlation = max(correlations, default=0.0)
        cardiac_scores.append(cardiac)
        motion_correlations.append(correlation)
        combined_scores.append(cardiac - 0.25 * correlation)
    return int(np.argmax(combined_scores)), cardiac_scores, motion_correlations, combined_scores


class _LinearBssReducer(ArtifactReducer):
    """PCA/FastICA 共用时域重建 / Shared time-domain reconstruction."""

    method: str

    def __init__(
        self,
        config: PcaBssConfig | FastIcaBssConfig | BssConfig | None = None,
    ) -> None:
        supplied = BssConfig() if config is None else config
        if self.method == "pca":
            config = _pca_config(supplied)  # type: ignore[arg-type]
        else:
            config = _fastica_config(supplied)  # type: ignore[arg-type]
        config.validate()
        self.config = config

    def _fit(self, source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        """返回 sources、mixing、mean 与诊断 / Return fitted linear decomposition."""

        if self.method == "pca":
            model = PCA(n_components=2, svd_solver="full")
            sources = model.fit_transform(source)
            return (
                sources,
                model.components_.T,
                model.mean_,
                {
                    "explained_variance_ratio": model.explained_variance_ratio_.tolist()
                },
            )
        model = FastICA(
            n_components=2,
            whiten="unit-variance",
            random_state=self.config.random_state,
            max_iter=self.config.max_iter,
            tol=self.config.tolerance,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            sources = model.fit_transform(source)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise ValueError("FastICA did not converge")
        return sources, np.asarray(model.mixing_), np.asarray(model.mean_), {"iterations": int(model.n_iter_)}

    def reduce(
        self,
        ppg: np.ndarray,
        imu_processed: Mapping[str, np.ndarray] | None,
        *,
        fs_hz: float = CANONICAL_FS_HZ,
    ) -> ArtifactReductionResult:
        """拟合双通道分解并只重建最佳 rate component / Reconstruct one rate source."""

        params = parameters_dict(self.config)
        try:
            preliminary = validate_ppg(ppg, fs_hz=fs_hz, allow_single_channel=True)
            if preliminary.shape[1] != 2:
                return failure_result(self, "BSS requires two synchronized PPG channels", parameters=params)
            source = preliminary
            if np.linalg.matrix_rank(source - np.mean(source, axis=0)) < 2:
                raise ValueError("dual PPG channels are rank-deficient")
            motion, motion_names, motion_valid = imu_reference_matrix(
                imu_processed,
                source.shape[0],
                profile_id=self.config.imu_reference_profile,
            )
            sources, mixing, mean, fit_diagnostics = self._fit(source)
            selected, cardiac, correlations, combined = _select_source(
                sources,
                motion,
                fs_hz,
                motion_valid_mask=motion_valid,
            )
            output = np.outer(sources[:, selected], mixing[:, selected]) + mean
            return success_result(
                self,
                output,
                input_ppg=source,
                confidence=float(np.clip(cardiac[selected], 0.0, 1.0)),
                parameters=params,
                diagnostics={
                    "method": self.method,
                    "selected_component": selected,
                    "cardiac_concentration": cardiac,
                    "motion_correlation": correlations,
                    "motion_reference_names": motion_names,
                    "imu_reference_profile": self.config.imu_reference_profile,
                    "output_valid_mask": motion_valid.tolist(),
                    "output_valid_fraction": float(np.mean(motion_valid)),
                    "selection_score": combined,
                    **fit_diagnostics,
                },
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            return failure_result(self, str(exc), parameters=params)


class PcaBssReducer(_LinearBssReducer):
    """双通道 PCA comparator / Dual-channel PCA comparator."""

    reducer_id = "pca_bss"
    reducer_version = "pca_component_select_v2"
    algorithm_kernel_description = ("对同步 RED/IR 做确定性双源 PCA，再按心率带集中度减 IMU 最大相关惩罚选择单一分量并投影回双通道；"
                                    "内核：full-SVD PCA、Welch 频谱评分和线性 mixing 重建。")
    method = "pca"


class FastIcaBssReducer(_LinearBssReducer):
    """固定 seed FastICA comparator / Fixed-seed FastICA comparator."""

    reducer_id = "fastica_bss"
    reducer_version = "fastica_component_select_v2"
    algorithm_kernel_description = ("对同步 RED/IR 做固定随机种子的双源 FastICA，再按心率带集中度减 IMU 相关惩罚选分量并回投；"
                                    "内核：unit-variance whitening、FastICA 不动点迭代与线性 mixing 重建。")
    method = "fastica"


class NmfBssReducer(ArtifactReducer):
    """双通道共享频谱 basis 的 NMF / NMF with shared dual-channel spectral bases."""

    reducer_id = "nmf_bss"
    reducer_version = "nmf_shared_spectral_basis_v1"
    algorithm_kernel_description = ("拼接 RED/IR 的非负 STFT 幅度并拟合共享谱基，选择心率带能量占比最高的基后复用原相位重建；"
                                    "内核：Hann STFT、NNDSVDA 初始化的坐标下降 NMF 与 ISTFT。")

    def __init__(self, config: NmfBssConfig | BssConfig | None = None) -> None:
        config = _nmf_config(BssConfig() if config is None else config)
        config.validate()
        self.config = config

    def reduce(
        self,
        ppg: np.ndarray,
        imu_processed: Mapping[str, np.ndarray] | None,
        *,
        fs_hz: float = CANONICAL_FS_HZ,
    ) -> ArtifactReductionResult:
        """NMF magnitude 后复用原相位重建 / Reconstruct selected NMF magnitude with phase."""

        params = parameters_dict(self.config)
        try:
            preliminary = validate_ppg(ppg, fs_hz=fs_hz, allow_single_channel=True)
            if preliminary.shape[1] != 2:
                return failure_result(self, "BSS requires two synchronized PPG channels", parameters=params)
            source = preliminary
            nperseg = min(self.config.nperseg, source.shape[0])
            if nperseg < 32:
                raise ValueError("NMF STFT input is too short")
            noverlap = min(nperseg - 1, int(round(nperseg * self.config.overlap_fraction)))
            spectra: list[np.ndarray] = []
            frequencies: np.ndarray | None = None
            for channel in range(2):
                frequencies, _, spectrum = signal.stft(
                    source[:, channel],
                    fs=fs_hz,
                    window="hann",
                    nperseg=nperseg,
                    noverlap=noverlap,
                    boundary="zeros",
                    padded=True,
                )
                spectra.append(spectrum)
            assert frequencies is not None
            frame_count = spectra[0].shape[1]
            magnitude = np.concatenate([np.abs(item) for item in spectra], axis=1)
            rank = min(self.config.nmf_rank, magnitude.shape[0], magnitude.shape[1])
            model = NMF(
                n_components=rank,
                init="nndsvda",
                solver="cd",
                random_state=self.config.random_state,
                max_iter=self.config.max_iter,
                tol=self.config.tolerance,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                basis = model.fit_transform(magnitude)
            if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                raise ValueError("NMF did not converge")
            activation = model.components_
            usable = (frequencies >= 0.2) & (frequencies <= 8.0)
            cardiac = (frequencies >= 0.5) & (frequencies <= 3.5)
            scores = [
                float(np.sum(basis[cardiac, index]) / max(np.sum(basis[usable, index]), 1e-12)) for index in range(rank)
            ]
            selected = int(np.argmax(scores))
            selected_magnitude = np.outer(basis[:, selected], activation[selected])
            outputs: list[np.ndarray] = []
            for channel in range(2):
                start, stop = channel * frame_count, (channel + 1) * frame_count
                phase = np.exp(1j * np.angle(spectra[channel]))
                selected_spectrum = selected_magnitude[:, start:stop] * phase
                _, reconstructed = signal.istft(
                    selected_spectrum,
                    fs=fs_hz,
                    window="hann",
                    nperseg=nperseg,
                    noverlap=noverlap,
                    input_onesided=True,
                    boundary=True,
                )
                if reconstructed.size < source.shape[0]:
                    raise ValueError("NMF ISTFT output lost time alignment")
                outputs.append(np.asarray(reconstructed[:source.shape[0]], dtype=np.float64))
            return success_result(
                self,
                np.column_stack(outputs),
                input_ppg=source,
                confidence=float(np.clip(scores[selected], 0.0, 1.0)),
                parameters=params,
                diagnostics={
                    "selected_component": selected,
                    "cardiac_concentration": scores,
                    "reconstruction_error": float(model.reconstruction_err_),
                    "iterations": int(model.n_iter_),
                    "nperseg_effective": int(nperseg),
                },
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            return failure_result(self, str(exc), parameters=params)

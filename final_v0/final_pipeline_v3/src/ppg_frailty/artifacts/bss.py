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
    failure_result,
    parameters_dict,
    success_result,
    validate_ppg,
)


@dataclass(frozen=True)
class BssConfig:
    """BSS 确定性参数 / Deterministic BSS parameters."""

    random_state: int = 42
    max_iter: int = 1000
    tolerance: float = 1e-5
    nmf_rank: int = 2
    nperseg: int = 512
    overlap_fraction: float = 0.75

    def validate(self) -> None:
        """验证 BSS 数值配置 / Validate numerical BSS configuration."""

        if self.max_iter <= 0 or self.tolerance <= 0.0 or self.nmf_rank <= 0:
            raise ValueError("BSS iteration/tolerance/rank must be positive")
        if self.nperseg < 32 or not 0.0 <= self.overlap_fraction < 1.0:
            raise ValueError("invalid NMF-STFT configuration")


def _motion_magnitude(
    imu_processed: Mapping[str, np.ndarray] | None, n_samples: int
) -> np.ndarray | None:
    """读取可选 motion magnitude，不伪造缺失参考 / Read optional motion magnitude."""

    if imu_processed is None:
        return None
    for key in ("dynamic_magnitude", "gyro_magnitude"):
        if key in imu_processed:
            values = np.asarray(imu_processed[key], dtype=np.float64).ravel()
            if values.size != n_samples or not np.isfinite(values).all():
                raise ValueError(f"{key} does not align with PPG")
            return values
    return None


def _cardiac_fraction(values: np.ndarray, fs_hz: float) -> float:
    """component 的 0.5–3.5 Hz 能量比例 / Cardiac-band component fraction."""

    frequencies, power = signal.welch(
        values, fs=fs_hz, nperseg=min(1024, values.size)
    )
    usable = (frequencies >= 0.2) & (frequencies <= 8.0)
    cardiac = (frequencies >= 0.5) & (frequencies <= 3.5)
    total = float(np.sum(power[usable]))
    return float(np.sum(power[cardiac]) / total) if total > 0.0 else 0.0


def _select_source(
    sources: np.ndarray,
    motion: np.ndarray | None,
    fs_hz: float,
) -> tuple[int, list[float], list[float], list[float]]:
    """按 cardiac concentration 减 motion correlation 选源 / Select a rate source."""

    cardiac_scores: list[float] = []
    motion_correlations: list[float] = []
    combined_scores: list[float] = []
    for column in range(sources.shape[1]):
        component = sources[:, column]
        cardiac = _cardiac_fraction(component, fs_hz)
        correlation = 0.0
        if motion is not None and np.std(component) > 1e-12 and np.std(motion) > 1e-12:
            correlation = abs(float(np.corrcoef(component, motion)[0, 1]))
        cardiac_scores.append(cardiac)
        motion_correlations.append(correlation)
        combined_scores.append(cardiac - 0.25 * correlation)
    return int(np.argmax(combined_scores)), cardiac_scores, motion_correlations, combined_scores


class _LinearBssReducer(ArtifactReducer):
    """PCA/FastICA 共用时域重建 / Shared time-domain reconstruction."""

    method: str

    def __init__(self, config: BssConfig = BssConfig()) -> None:
        config.validate()
        self.config = config

    def _fit(self, source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        """返回 sources、mixing、mean 与诊断 / Return fitted linear decomposition."""

        if self.method == "pca":
            model = PCA(n_components=2, svd_solver="full")
            sources = model.fit_transform(source)
            return sources, model.components_.T, model.mean_, {
                "explained_variance_ratio": model.explained_variance_ratio_.tolist()
            }
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
        return sources, np.asarray(model.mixing_), np.asarray(model.mean_), {
            "iterations": int(model.n_iter_)
        }

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
            motion = _motion_magnitude(imu_processed, source.shape[0])
            sources, mixing, mean, fit_diagnostics = self._fit(source)
            selected, cardiac, correlations, combined = _select_source(sources, motion, fs_hz)
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
                    "selection_score": combined,
                    **fit_diagnostics,
                },
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            return failure_result(self, str(exc), parameters=params)


class PcaBssReducer(_LinearBssReducer):
    """双通道 PCA comparator / Dual-channel PCA comparator."""

    reducer_id = "pca_bss"
    reducer_version = "pca_component_select_v1"
    method = "pca"


class FastIcaBssReducer(_LinearBssReducer):
    """固定 seed FastICA comparator / Fixed-seed FastICA comparator."""

    reducer_id = "fastica_bss"
    reducer_version = "fastica_component_select_v1"
    method = "fastica"


class NmfBssReducer(ArtifactReducer):
    """双通道共享频谱 basis 的 NMF / NMF with shared dual-channel spectral bases."""

    reducer_id = "nmf_bss"
    reducer_version = "nmf_shared_spectral_basis_v1"

    def __init__(self, config: BssConfig = BssConfig()) -> None:
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
                    source[:, channel], fs=fs_hz, window="hann", nperseg=nperseg,
                    noverlap=noverlap, boundary="zeros", padded=True,
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
                float(np.sum(basis[cardiac, index]) / max(np.sum(basis[usable, index]), 1e-12))
                for index in range(rank)
            ]
            selected = int(np.argmax(scores))
            selected_magnitude = np.outer(basis[:, selected], activation[selected])
            outputs: list[np.ndarray] = []
            for channel in range(2):
                start, stop = channel * frame_count, (channel + 1) * frame_count
                phase = np.exp(1j * np.angle(spectra[channel]))
                selected_spectrum = selected_magnitude[:, start:stop] * phase
                _, reconstructed = signal.istft(
                    selected_spectrum, fs=fs_hz, window="hann", nperseg=nperseg,
                    noverlap=noverlap, input_onesided=True, boundary=True,
                )
                if reconstructed.size < source.shape[0]:
                    raise ValueError("NMF ISTFT output lost time alignment")
                outputs.append(np.asarray(reconstructed[: source.shape[0]], dtype=np.float64))
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


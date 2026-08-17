"""STFT + IMU 谱掩蔽 reducer / STFT suppression using an IMU spectral mask."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import signal

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
class SpectralMaskConfig:
    """正式 motion YAML 的物理参数 / Physical formal motion-YAML parameters.

    English: Seconds are converted to samples only at the explicitly supplied
    sampling rate. ``preserve_band_hz`` is the reconstructed cardiac-rate band;
    bins outside it are zeroed, while the IMU soft mask acts inside it.

    中文：秒参数只在显式采样率下转换为样本数。``preserve_band_hz`` 是重建
    的心率频带；带外频点置零，IMU 软掩蔽仅在带内生效。
    """

    stft_window_s: float = 4.0
    stft_hop_s: float = 1.0
    imu_mask_quantile: float = 0.75
    mask_strength: float = 0.80
    preserve_band_hz: tuple[float, float] = (0.5, 3.0)
    imu_reference_profile: str = IMU_REFERENCE_AXES6_PROFILE_ID

    def validate(self) -> None:
        """验证 COLA-friendly window 参数 / Validate stable overlap parameters."""

        if self.stft_window_s <= 0.0 or not 0.0 < self.stft_hop_s <= self.stft_window_s:
            raise ValueError("STFT window/hop seconds must satisfy 0 < hop <= window")
        if not 0.0 < self.imu_mask_quantile < 1.0:
            raise ValueError("imu_mask_quantile must lie strictly inside (0,1)")
        if not 0.0 <= self.mask_strength <= 1.0:
            raise ValueError("mask_strength must lie in [0,1]")
        if len(self.preserve_band_hz) != 2:
            raise ValueError("preserve_band_hz must contain exactly [low, high]")
        low, high = (float(value) for value in self.preserve_band_hz)
        if low < 0.0 or high <= low:
            raise ValueError("preserve_band_hz must satisfy 0 <= low < high")
        if self.imu_reference_profile not in {
            IMU_REFERENCE_AXES6_PROFILE_ID,
            IMU_REFERENCE_DERIVED9_AUGMENTATION_PROFILE_ID,
        }:
            raise ValueError("unknown spectral-mask IMU reference profile")


def _stft(values: np.ndarray, fs_hz: float, nperseg: int, noverlap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """统一 Hann STFT / Shared Hann STFT."""

    return signal.stft(
        values,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )


class SpectralMaskReducer(ArtifactReducer):
    """用 IMU time-frequency magnitude 构建 soft mask / IMU-informed soft mask."""

    reducer_id = "spectral_mask"
    reducer_version = "spectral_mask_v1"

    def __init__(self, config: SpectralMaskConfig = SpectralMaskConfig()) -> None:
        config.validate()
        self.config = config

    def reduce(
        self,
        ppg: np.ndarray,
        imu_processed: Mapping[str, np.ndarray] | None,
        *,
        fs_hz: float = CANONICAL_FS_HZ,
    ) -> ArtifactReductionResult:
        """执行谱域软抑制；任何 ISTFT 对齐错误均失败 / Run aligned soft suppression."""

        params = parameters_dict(self.config)
        try:
            source = validate_ppg(ppg, fs_hz=fs_hz)
            references, names, imu_valid = imu_reference_matrix(
                imu_processed,
                source.shape[0],
                profile_id=self.config.imu_reference_profile,
            )
            requested_nperseg = int(round(self.config.stft_window_s * fs_hz))
            requested_hop = int(round(self.config.stft_hop_s * fs_hz))
            if (
                requested_nperseg < 32
                or requested_hop <= 0
                or not np.isclose(requested_nperseg, self.config.stft_window_s * fs_hz)
                or not np.isclose(requested_hop, self.config.stft_hop_s * fs_hz)
            ):
                raise ValueError("STFT seconds must map to integer samples and window >= 32")
            nperseg = min(requested_nperseg, source.shape[0])
            if nperseg < 32:
                raise ValueError("STFT input is too short")
            effective_hop = min(requested_hop, nperseg)
            noverlap = nperseg - effective_hop
            # 中文：STFT/ISTFT 会在一个窗内混合样本，因此把任何 invalid IMU row
            # 周围一个完整窗保守标为无效；值仍对齐，但下游不得使用这些位置。
            # English: Conservatively invalidate a full-window neighborhood around
            # every invalid IMU row because overlap-add mixes samples within a frame.
            invalid_neighborhood = np.convolve(
                (~imu_valid).astype(np.int64),
                np.ones(nperseg, dtype=np.int64),
                mode="same",
            ) > 0
            output_valid = ~invalid_neighborhood
            if np.count_nonzero(output_valid) < 32:
                raise ValueError("too few artifact-valid samples after IMU-mask propagation")
            motion_spectra: list[np.ndarray] = []
            frequencies: np.ndarray | None = None
            for column in range(references.shape[1]):
                frequencies, _, spectrum = _stft(
                    references[:, column], fs_hz, nperseg, noverlap
                )
                motion_spectra.append(np.abs(spectrum))
            assert frequencies is not None
            motion = np.sqrt(np.mean(np.square(motion_spectra), axis=0))
            # 中文：每个时间帧以 formal quantile 归一化 IMU 频谱，避免固定 ADC
            # 幅值阈值。English: Normalize each frame by the formal IMU quantile.
            motion_scale = np.quantile(
                motion, self.config.imu_mask_quantile, axis=0, keepdims=True
            )
            motion_unit = motion / np.maximum(motion_scale, 1e-12)
            low_hz, high_hz = self.config.preserve_band_hz
            if high_hz > fs_hz / 2.0:
                raise ValueError("preserve_band_hz exceeds Nyquist")
            preserved = (frequencies >= low_hz) & (frequencies <= high_hz)
            if np.count_nonzero(preserved) < 2:
                raise ValueError("preserve_band_hz contains fewer than two STFT bins")

            outputs: list[np.ndarray] = []
            mean_gain: list[float] = []
            for channel in range(2):
                _, _, spectrum = _stft(source[:, channel], fs_hz, nperseg, noverlap)
                ppg_magnitude = np.abs(spectrum)
                ppg_scale = np.percentile(ppg_magnitude, 95.0, axis=0, keepdims=True)
                ppg_unit = ppg_magnitude / np.maximum(ppg_scale, 1e-12)
                contamination = motion_unit / np.maximum(motion_unit + ppg_unit, 1e-12)
                # English: ``1-mask_strength`` is a derived in-band floor, not a
                # hidden parameter. Out-of-band bins are excluded from rate-only x_ar.
                # 中文：``1-mask_strength`` 是可追溯的带内下限，并非隐藏参数；
                # 带外频点不进入 rate-only x_ar。
                in_band_gain = np.clip(
                    1.0 - self.config.mask_strength * contamination,
                    1.0 - self.config.mask_strength,
                    1.0,
                )
                gain = np.zeros_like(in_band_gain)
                gain[preserved, :] = in_band_gain[preserved, :]
                _, reconstructed = signal.istft(
                    spectrum * gain,
                    fs=fs_hz,
                    window="hann",
                    nperseg=nperseg,
                    noverlap=noverlap,
                    input_onesided=True,
                    boundary=True,
                )
                if reconstructed.size < source.shape[0]:
                    raise ValueError("ISTFT output is shorter than the original grid")
                outputs.append(np.asarray(reconstructed[: source.shape[0]], dtype=np.float64))
                mean_gain.append(float(np.mean(gain[preserved, :])))
            output = np.column_stack(outputs)
            # English: Confidence is retained-signal agreement on artifact-valid
            # rows. ``1-mean_gain`` is reported only as suppression_fraction and is
            # not mislabeled as confidence (clean signals should not score poorly).
            # 中文：confidence 是有效位置上的输入/输出一致性；1-mean_gain 仅作为
            # suppression_fraction，不再把“抑制少”误报成低置信度。
            agreement: list[float] = []
            for channel in range(2):
                left = source[output_valid, channel]
                right = output[output_valid, channel]
                if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
                    agreement.append(0.0)
                else:
                    agreement.append(float(np.clip(np.corrcoef(left, right)[0, 1], 0.0, 1.0)))
            confidence = float(np.mean(agreement))
            return success_result(
                self,
                output,
                input_ppg=source,
                confidence=confidence,
                parameters=params,
                diagnostics={
                    "reference_names": names,
                    "imu_reference_profile": self.config.imu_reference_profile,
                    "nperseg_effective": int(nperseg),
                    "noverlap_effective": int(noverlap),
                    "hop_samples_effective": int(effective_hop),
                    "imu_mask_quantile": float(self.config.imu_mask_quantile),
                    "preserve_band_hz": tuple(float(value) for value in self.config.preserve_band_hz),
                    "preserve_bin_count": int(np.count_nonzero(preserved)),
                    "out_of_band_policy": "zero_rate_only_reconstruction",
                    "mean_gain_by_channel": mean_gain,
                    "suppression_fraction_by_channel": [1.0 - value for value in mean_gain],
                    "retained_signal_agreement_by_channel": agreement,
                    "imu_valid_fraction": float(np.mean(imu_valid)),
                    "output_valid_fraction": float(np.mean(output_valid)),
                    "output_valid_mask": output_valid.tolist(),
                    "invalid_reference_policy": "window_eroded_output_validity_mask",
                },
            )
        except (ValueError, FloatingPointError) as exc:
            return failure_result(self, str(exc), parameters=params)

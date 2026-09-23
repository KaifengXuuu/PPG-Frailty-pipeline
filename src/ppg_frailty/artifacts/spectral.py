"""STFT suppression using an IMU spectral mask."""

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
    """Physical parameters declared by the formal motion YAML.

    English: Seconds are converted to samples only at the explicitly supplied
    sampling rate. ``preserve_band_hz`` is the reconstructed cardiac-rate band;
    bins outside it are zeroed, while the IMU soft mask acts inside it.

    Convert seconds to samples only with an explicit sampling rate. ``preserve_band_hz``
    is the reconstruction cardiac band: zero outside it and apply the IMU soft mask only within it.
    """

    stft_window_s: float = 4.0
    stft_hop_s: float = 1.0
    imu_mask_quantile: float = 0.75
    mask_strength: float = 0.80
    preserve_band_hz: tuple[float, float] = (0.5, 3.0)
    imu_reference_profile: str = IMU_REFERENCE_AXES6_PROFILE_ID

    def validate(self) -> None:
        """Validate stable overlap parameters."""

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
    """Shared Hann STFT."""

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
    """IMU-informed soft mask."""

    reducer_id = "spectral_mask"
    reducer_version = "spectral_mask_v1"
    algorithm_kernel_description = ("Build a soft contamination mask from six-axis IMU time-frequency magnitudes; suppress RED/IR within the cardiac band and zero frequencies outside it. "
                                    "Kernel: Hann STFT/ISTFT, per-frame IMU quantile normalization and multiplicative spectral gain with a lower bound.")

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
        """Run aligned soft suppression."""

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
            if (requested_nperseg < 32 or requested_hop <= 0
                    or not np.isclose(requested_nperseg, self.config.stft_window_s * fs_hz)
                    or not np.isclose(requested_hop, self.config.stft_hop_s * fs_hz)):
                raise ValueError("STFT seconds must map to integer samples and window >= 32")
            nperseg = min(requested_nperseg, source.shape[0])
            if nperseg < 32:
                raise ValueError("STFT input is too short")
            effective_hop = min(requested_hop, nperseg)
            noverlap = nperseg - effective_hop
            # STFT/ISTFT mix samples within windows; conservatively invalidate a full
            # window around each invalid IMU row. Alignment remains, but these positions are unusable.
            # English: Conservatively invalidate a full-window neighborhood around
            # every invalid IMU row because overlap-add mixes samples within a frame.
            invalid_neighborhood = (np.convolve(
                (~imu_valid).astype(np.int64),
                np.ones(nperseg, dtype=np.int64),
                mode="same",
            ) > 0)
            output_valid = ~invalid_neighborhood
            if np.count_nonzero(output_valid) < 32:
                raise ValueError("too few artifact-valid samples after IMU-mask propagation")
            motion_spectra: list[np.ndarray] = []
            frequencies: np.ndarray | None = None
            for column in range(references.shape[1]):
                frequencies, _, spectrum = _stft(references[:, column], fs_hz, nperseg, noverlap)
                motion_spectra.append(np.abs(spectrum))
            assert frequencies is not None
            motion = np.sqrt(np.mean(np.square(motion_spectra), axis=0))
            # Normalize each IMU spectral frame by the formal quantile rather than a
            # fixed ADC-amplitude threshold.
            motion_scale = np.quantile(motion, self.config.imu_mask_quantile, axis=0, keepdims=True)
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
                # ``1-mask_strength`` is the explicit, auditable in-band gain floor;
                # out-of-band frequencies do not enter rate-only x_ar.
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
                outputs.append(np.asarray(reconstructed[:source.shape[0]], dtype=np.float64))
                mean_gain.append(float(np.mean(gain[preserved, :])))
            output = np.column_stack(outputs)
            # English: Confidence is retained-signal agreement on artifact-valid
            # rows. ``1-mean_gain`` is reported only as suppression_fraction and is
            # not mislabeled as confidence (clean signals should not score poorly).
            # Confidence measures valid-position input/output agreement; 1-mean_gain is
            # suppression_fraction, so weak suppression is not mislabeled as low confidence.
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

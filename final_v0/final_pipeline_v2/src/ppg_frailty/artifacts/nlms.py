"""带显式 delay taps 的 IMU-reference NLMS ANC / IMU-referenced tapped NLMS ANC.

该方法可能违反“参考与真实心率无关”的 ANC 前提，因此 V1 始终把输出标记为
rate-only，并将其作为非主路线比较。The output is always rate-only because motion
and true heart-rate response can be physiologically coupled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

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
class NlmsConfig:
    """NLMS 冻结参数 / Frozen NLMS parameters."""

    taps_per_delay: int = 8
    delay_taps: tuple[int, ...] = (0, 4, 8, 16)
    step_size: float = 0.15
    epsilon: float = 1e-6
    leakage: float = 1e-5
    update_gate_reference_rms: float = 0.10
    imu_reference_profile: str = IMU_REFERENCE_AXES6_PROFILE_ID

    def validate(self) -> None:
        """验证 NLMS 稳定区间 / Validate NLMS stability ranges."""

        if self.taps_per_delay <= 0 or not self.delay_taps or min(self.delay_taps) < 0:
            raise ValueError("taps_per_delay and non-negative delay_taps are required")
        if not 0.0 < self.step_size < 2.0:
            raise ValueError("NLMS step_size must lie in (0,2)")
        if self.epsilon <= 0.0 or not 0.0 <= self.leakage < 1.0:
            raise ValueError("epsilon/leakage are outside stable ranges")
        if self.imu_reference_profile not in {
            IMU_REFERENCE_AXES6_PROFILE_ID,
            IMU_REFERENCE_DERIVED9_AUGMENTATION_PROFILE_ID,
        }:
            raise ValueError("unknown NLMS IMU reference profile")


class NlmsReducer(ArtifactReducer):
    """每个 PPG 波长独立估计 IMU artifact / Independent per-wavelength ANC."""

    reducer_id = "nlms_imu_anc"
    reducer_version = "nlms_delay_taps_v1"
    algorithm_kernel_description = (
        "以六轴物理单位 IMU 为参考，对 RED/IR 分别运行带泄漏的归一化 LMS 自适应噪声抵消；"
        "内核：多延迟 tapped-reference 线性滤波，参考 RMS 达阈值时按 NLMS 规则更新权重。"
    )

    def __init__(self, config: NlmsConfig = NlmsConfig()) -> None:
        config.validate()
        self.config = config

    def reduce(
        self,
        ppg: np.ndarray,
        imu_processed: Mapping[str, np.ndarray] | None,
        *,
        fs_hz: float = CANONICAL_FS_HZ,
    ) -> ArtifactReductionResult:
        """运行 normalized LMS；失败返回 None 而非 direct 波形 / Run fail-closed NLMS."""

        params = parameters_dict(self.config)
        try:
            source = validate_ppg(ppg, fs_hz=fs_hz)
            references, names, imu_valid = imu_reference_matrix(
                imu_processed,
                source.shape[0],
                profile_id=self.config.imu_reference_profile,
            )
            offsets = sorted(
                {
                    delay + tap
                    for delay in self.config.delay_taps
                    for tap in range(self.config.taps_per_delay)
                }
            )
            dimension = references.shape[1] * len(offsets)
            weights = np.zeros((2, dimension), dtype=np.float64)
            output = source.copy()
            predictions = np.zeros_like(source)
            output_valid = np.zeros(source.shape[0], dtype=bool)
            update_count = 0
            for index in range(max(offsets), source.shape[0]):
                # 中文：一个 tapped vector 中任一 IMU row 无效时，不预测、不更新，
                # 且该 PPG row 在 artifact output mask 中保持 false。
                # English: Any invalid delayed reference excludes this output row.
                if not all(bool(imu_valid[index - offset]) for offset in offsets):
                    continue
                vector = np.concatenate([references[index - offset] for offset in offsets])
                reference_rms = float(np.sqrt(np.mean(np.square(vector))))
                prediction = weights @ vector
                error = source[index] - prediction
                predictions[index] = prediction
                output[index] = error
                output_valid[index] = True
                if reference_rms >= self.config.update_gate_reference_rms:
                    normalizer = self.config.epsilon + float(np.dot(vector, vector))
                    weights *= 1.0 - self.config.leakage
                    weights += (
                        self.config.step_size
                        * error[:, None]
                        * vector[None, :]
                        / normalizer
                    )
                    update_count += 1
            source_var = np.var(source, axis=0)
            residual_var = np.var(output, axis=0)
            explained = np.clip(1.0 - residual_var / np.maximum(source_var, 1e-12), 0.0, 1.0)
            confidence = float(np.mean(explained))
            return success_result(
                self,
                output,
                input_ppg=source,
                confidence=confidence,
                parameters=params,
                diagnostics={
                    "reference_names": names,
                    "imu_reference_profile": self.config.imu_reference_profile,
                    "tap_offsets_samples": tuple(offsets),
                    "update_count": int(update_count),
                    "imu_valid_fraction": float(np.mean(imu_valid)),
                    "output_valid_fraction": float(np.mean(output_valid)),
                    "output_valid_mask": output_valid.tolist(),
                    "invalid_reference_policy": "passthrough_value_with_output_validity_false",
                    "explained_variance_by_channel": explained.tolist(),
                    "prediction_rms_by_channel": np.sqrt(np.mean(np.square(predictions), axis=0)).tolist(),
                },
            )
        except (ValueError, FloatingPointError) as exc:
            return failure_result(self, str(exc), parameters=params)

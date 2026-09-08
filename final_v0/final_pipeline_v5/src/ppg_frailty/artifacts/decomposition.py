"""SSA 主非平稳分解 reducer / Primary singular-spectrum decomposition reducer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import signal

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
class SsaConfig:
    """SSA 嵌入和 component 选择参数 / SSA embedding and selection parameters."""

    embedding_samples: int = 160
    max_components: int = 12
    minimum_cardiac_concentration: float = 0.45
    cardiac_low_hz: float = 0.5
    cardiac_high_hz: float = 3.5

    def validate(self) -> None:
        """验证可辨识 SSA 配置 / Validate an identifiable SSA configuration."""

        if self.embedding_samples < 8 or self.max_components < 1:
            raise ValueError("SSA embedding/max_components are too small")
        if not 0.0 <= self.minimum_cardiac_concentration <= 1.0:
            raise ValueError("SSA concentration threshold must lie in [0,1]")

def _diagonal_average(elementary: np.ndarray) -> np.ndarray:
    """Hankelization/对角平均 / Diagonal averaging of one trajectory matrix."""

    rows, columns = elementary.shape
    output = np.zeros(rows + columns - 1, dtype=np.float64)
    counts = np.zeros_like(output)
    for row in range(rows):
        output[row : row + columns] += elementary[row]
        counts[row : row + columns] += 1.0
    return output / counts

def _cardiac_concentration(values: np.ndarray, fs_hz: float, config: SsaConfig) -> float:
    """计算 component 心率带能量比例 / Cardiac-band component power fraction."""

    frequencies, power = signal.welch(values, fs=fs_hz, nperseg=min(1024, values.size))
    usable = (frequencies >= 0.2) & (frequencies <= 8.0)
    cardiac = (frequencies >= config.cardiac_low_hz) & (frequencies <= config.cardiac_high_hz)
    total = float(np.sum(power[usable]))
    return float(np.sum(power[cardiac]) / total) if total > 0.0 else 0.0

def _ssa_channel(values: np.ndarray, fs_hz: float, config: SsaConfig) -> tuple[np.ndarray, dict[str, object]]:
    """分解单波长并重建 cardiac components / Reconstruct selected components."""

    n_samples = values.size
    embedding = min(config.embedding_samples, n_samples // 3)
    if embedding < 8 or n_samples - embedding + 1 < embedding:
        raise ValueError("SSA input is too short for the configured embedding")
    trajectory = np.lib.stride_tricks.sliding_window_view(values, embedding).T.copy()
    left, singular, right = np.linalg.svd(trajectory, full_matrices=False)
    count = min(config.max_components, singular.size)
    components: list[np.ndarray] = []
    scores: list[float] = []
    for index in range(count):
        elementary = singular[index] * np.outer(left[:, index], right[index])
        reconstructed = _diagonal_average(elementary)
        components.append(reconstructed)
        scores.append(_cardiac_concentration(reconstructed, fs_hz, config))
    selected = [index for index, score in enumerate(scores) if score >= config.minimum_cardiac_concentration]
    if not selected:
        # English: Selecting argmax below the registered minimum would silently
        # turn a failed decomposition into a successful rate waveform.
        # 中文：低于注册阈值时再选 argmax 会把失败分解静默伪装为成功波形。
        raise ValueError(
            "SSA has no component meeting minimum_cardiac_concentration; "
            f"best={max(scores, default=0.0):.6f}, "
            f"required={config.minimum_cardiac_concentration:.6f}"
        )
    output = np.sum([components[index] for index in selected], axis=0)
    diagnostics: dict[str, object] = {
        "embedding_samples": int(embedding),
        "singular_values": singular[:count].tolist(),
        "cardiac_concentration": scores,
        "selected_components": selected,
    }
    return np.asarray(output, dtype=np.float64), diagnostics

class SsaReducer(ArtifactReducer):
    """双波长独立 SSA，输出仅准入 rate / Independent dual-channel SSA."""

    reducer_id = "ssa_decomposition"
    reducer_version = "ssa_hankel_cardiac_select_v1"
    algorithm_kernel_description = (
        "RED/IR 分别构造 Hankel 轨迹矩阵并做 SSA，选择心率频带能量占比达阈值的分量后重建；" "内核：全矩阵 SVD、逐分量对角平均与 Welch 0.5–3.5 Hz 浓度筛选。"
    )

    def __init__(self, config: SsaConfig = SsaConfig()) -> None:
        config.validate()
        self.config = config

    def reduce(
        self,
        ppg: np.ndarray,
        imu_processed: Mapping[str, np.ndarray] | None,
        *,
        fs_hz: float = CANONICAL_FS_HZ,
    ) -> ArtifactReductionResult:
        """执行确定性 CPU SVD / Execute deterministic CPU SVD."""

        params = parameters_dict(self.config)
        try:
            source = validate_ppg(ppg, fs_hz=fs_hz)
            outputs: list[np.ndarray] = []
            channel_diagnostics: dict[str, object] = {}
            for channel, name in enumerate(("RED", "IR")):
                reconstructed, diagnostics = _ssa_channel(source[:, channel], fs_hz, self.config)
                outputs.append(reconstructed)
                channel_diagnostics[name] = diagnostics
            output = np.column_stack(outputs)
            selected_scores = [
                max(value["cardiac_concentration"]) for value in channel_diagnostics.values()  # type: ignore[index]
            ]
            return success_result(
                self,
                output,
                input_ppg=source,
                confidence=float(np.mean(selected_scores)),
                parameters=params,
                diagnostics={"channels": channel_diagnostics},
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            return failure_result(self, str(exc), parameters=params)

"""Self-contained NumPy ROCKET and explicitly separate MiniROCKET ablation.

自足 NumPy ROCKET 主实现，以及明确分离的 MiniROCKET 消融实现。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeClassifier


def _normalise_mask(x: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Return a boolean ``[N,C,T]`` validity mask / 返回布尔有效性掩码。"""

    if mask is None:
        return np.ones(x.shape, dtype=bool)
    value = np.asarray(mask, dtype=bool)
    if value.shape == (x.shape[0], x.shape[2]):
        value = np.broadcast_to(value[:, None, :], x.shape)
    if value.shape != x.shape:
        raise ValueError("mask must be [sample,time] or [sample,channel,time]")
    return value


@dataclass(frozen=True)
class RocketKernel:
    """One immutable random convolution kernel / 单个不可变随机卷积核。"""

    channel: int
    weights: np.ndarray
    dilation: int
    padding: int
    bias: float


class MaskedChannelRobustScaler:
    """Training-fitted channel median/IQR scaler that ignores padded values.

    仅在训练数据有效位置拟合的通道中位数/IQR 缩放器；填充值不参与统计。
    """

    def __init__(self) -> None:
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray, mask: np.ndarray | None = None) -> "MaskedChannelRobustScaler":
        """Fit per-channel robust statistics / 拟合逐通道稳健统计量。"""

        array = np.asarray(x, dtype=np.float64)
        if array.ndim != 3 or not np.isfinite(array).all():
            raise ValueError("x must be finite [sample,channel,time]")
        valid = _normalise_mask(array, mask)
        centers, scales = [], []
        for channel in range(array.shape[1]):
            values = array[:, channel, :][valid[:, channel, :]]
            if values.size == 0:
                raise ValueError(f"channel {channel} contains no valid training values")
            q25, median, q75 = np.percentile(values, [25.0, 50.0, 75.0])
            centers.append(float(median))
            scales.append(max(float(q75 - q25), 1e-8))
        self.center_ = np.asarray(centers, dtype=np.float64)
        self.scale_ = np.asarray(scales, dtype=np.float64)
        return self

    def transform(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Scale valid values and zero invalid positions / 缩放有效值并将无效位归零。"""

        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("MaskedChannelRobustScaler must be fitted before transform")
        array = np.asarray(x, dtype=np.float64)
        if array.ndim != 3 or array.shape[1] != self.center_.size:
            raise ValueError("input channel count differs from the fitted scaler")
        valid = _normalise_mask(array, mask)
        transformed = (array - self.center_[None, :, None]) / self.scale_[None, :, None]
        transformed[~valid] = 0.0
        if not np.isfinite(transformed).all():
            raise ValueError("scaled input contains non-finite values")
        return transformed.astype(np.float32)

    def fit_transform(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Fit then transform / 先拟合再转换。"""

        return self.fit(x, mask).transform(x, mask)


class RocketTransformer:
    """Canonical self-contained NumPy ROCKET transform.

    自足 NumPy ROCKET 转换。每个随机卷积核输出最大值与正值比例（PPV）两个特征。
    默认 10,000 个核；测试和资源受限消融可以显式降低核数。
    """

    algorithm_id = "rocket_numpy_v1"

    def __init__(self, n_kernels: int = 10_000, seed: int = 42) -> None:
        if n_kernels <= 0:
            raise ValueError("n_kernels must be positive")
        self.n_kernels = int(n_kernels)
        self.seed = int(seed)
        self.kernels_: tuple[RocketKernel, ...] | None = None
        self.input_shape_: tuple[int, int] | None = None

    def fit(self, x: np.ndarray) -> "RocketTransformer":
        """Generate deterministic kernels for the training input shape.

        针对训练输入形状生成确定性的随机卷积核。该步骤不读取标签。
        """

        array = np.asarray(x)
        if array.ndim != 3 or array.shape[1] <= 0 or array.shape[2] < 3:
            raise ValueError("x must have shape [sample,channel,time] with time >= 3")
        channels, time = int(array.shape[1]), int(array.shape[2])
        lengths = [length for length in (7, 9, 11) if length <= time]
        if not lengths:
            lengths = [time if time % 2 == 1 else time - 1]
        rng = np.random.default_rng(self.seed)
        kernels: list[RocketKernel] = []
        for _ in range(self.n_kernels):
            length = int(rng.choice(lengths))
            maximum_exponent = np.log2(max((time - 1) / max(length - 1, 1), 1.0))
            dilation = int(2 ** rng.uniform(0.0, maximum_exponent))
            receptive_field = (length - 1) * dilation + 1
            padding = receptive_field // 2 if bool(rng.integers(0, 2)) else 0
            weights = rng.normal(size=length)
            weights -= weights.mean()
            norm = np.linalg.norm(weights)
            weights = weights / max(float(norm), 1e-12)
            kernels.append(
                RocketKernel(
                    channel=int(rng.integers(0, channels)),
                    weights=weights.astype(np.float32),
                    dilation=dilation,
                    padding=padding,
                    bias=float(rng.uniform(-1.0, 1.0)),
                )
            )
        self.kernels_ = tuple(kernels)
        self.input_shape_ = (channels, time)
        return self

    @staticmethod
    def _apply_kernel(signal: np.ndarray, valid: np.ndarray, kernel: RocketKernel) -> tuple[float, float]:
        """Apply one dilated kernel and return max/PPV / 应用核并返回最大值与 PPV。"""

        dilated = np.zeros((kernel.weights.size - 1) * kernel.dilation + 1, dtype=np.float32)
        dilated[:: kernel.dilation] = kernel.weights
        if kernel.padding:
            signal = np.pad(signal, kernel.padding, mode="constant")
            valid = np.pad(valid, kernel.padding, mode="constant", constant_values=False)
        response = np.convolve(signal, dilated[::-1], mode="valid") + kernel.bias
        valid_counts = np.convolve(valid.astype(np.int16), np.ones(dilated.size, dtype=np.int16), mode="valid")
        response = response[valid_counts == dilated.size]
        if response.size == 0:
            return 0.0, 0.0
        return float(np.max(response)), float(np.mean(response > 0.0))

    def transform(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Return ``[sample,2*n_kernels]`` ROCKET features / 返回 ROCKET 特征。"""

        if self.kernels_ is None or self.input_shape_ is None:
            raise RuntimeError("RocketTransformer must be fitted before transform")
        array = np.asarray(x, dtype=np.float32)
        if array.ndim != 3 or (array.shape[1], array.shape[2]) != self.input_shape_:
            raise ValueError("input shape differs from the fitted ROCKET shape")
        valid = _normalise_mask(array, mask)
        output = np.empty((array.shape[0], self.n_kernels * 2), dtype=np.float32)
        for sample_index in range(array.shape[0]):
            for kernel_index, kernel in enumerate(self.kernels_):
                maximum, ppv = self._apply_kernel(
                    array[sample_index, kernel.channel], valid[sample_index, kernel.channel], kernel
                )
                output[sample_index, 2 * kernel_index : 2 * kernel_index + 2] = (maximum, ppv)
        if not np.isfinite(output).all():
            raise ValueError("ROCKET produced non-finite features")
        return output

    def fit_transform(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Fit shape-dependent kernels then transform / 拟合核后转换。"""

        return self.fit(x).transform(x, mask)


class RocketRidgeClassifier:
    """Fold-local robust scaling, ROCKET transform and ridge classifier.

    fold-local 稳健缩放、ROCKET 转换与岭分类器的完整管线。
    """

    model_id = "rocket_ridge_numpy"

    def __init__(self, n_kernels: int = 10_000, alpha: float = 1.0, seed: int = 42) -> None:
        self.n_kernels = int(n_kernels)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.scaler = MaskedChannelRobustScaler()
        self.transformer = RocketTransformer(n_kernels=n_kernels, seed=seed)
        self.classifier = RidgeClassifier(alpha=alpha)
        self.classes_: np.ndarray | None = None
        self.fitted_participant_ids_: tuple[str, ...] = ()
        self.fitted_object_provenance_: dict[str, dict[str, object]] = {}

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        participant_ids: list[str] | tuple[str, ...],
        sample_weight: np.ndarray | None = None,
    ) -> "RocketRidgeClassifier":
        """Fit every stateful step on one outer-training partition.

        在一个 outer-training 分区上拟合全部有状态步骤。
        """

        labels = np.asarray(y)
        if labels.shape != (len(x),) or len(participant_ids) != len(x):
            raise ValueError("labels and participant_ids must align with samples")
        scaled = self.scaler.fit_transform(x, mask)
        features = self.transformer.fit_transform(scaled, mask)
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if weights is not None and (
            weights.shape != (labels.size,) or not np.isfinite(weights).all() or np.any(weights < 0)
        ):
            raise ValueError("sample_weight must be finite, non-negative and row-aligned")
        self.classifier.fit(features, labels, sample_weight=weights)
        self.classes_ = np.asarray(self.classifier.classes_)
        self.fitted_participant_ids_ = tuple(sorted(set(str(value) for value in participant_ids)))
        self.fitted_object_provenance_ = {
            "robust_scaler": {
                "object_type": type(self.scaler).__name__,
                "fitted_participant_ids": self.fitted_participant_ids_,
            },
            "rocket_kernel_generator": {
                "object_type": type(self.transformer).__name__,
                "fitted_participant_ids": self.fitted_participant_ids_,
                "seed": self.seed,
                "n_kernels": self.n_kernels,
            },
            "ridge_classifier": {
                "object_type": type(self.classifier).__name__,
                "fitted_participant_ids": self.fitted_participant_ids_,
            },
        }
        return self

    def decision_function(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Return ridge decision scores / 返回岭分类决策分数。"""

        scaled = self.scaler.transform(x, mask)
        scores = np.asarray(self.classifier.decision_function(self.transformer.transform(scaled, mask)))
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        return scores

    def predict_proba(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Convert scores to deterministic softmax probabilities / 将分数转为 softmax 概率。"""

        scores = self.decision_function(x, mask)
        scores = scores - scores.max(axis=1, keepdims=True)
        exponent = np.exp(scores)
        return exponent / exponent.sum(axis=1, keepdims=True)


class MiniRocketAblation(RocketRidgeClassifier):
    """Explicit low-cost engineering ablation; not a reference MiniROCKET port.

    明确命名的低成本工程消融；它不是论文参考 MiniROCKET 的等价移植。该类绝不
    能静默替代主 ROCKET，实现结果必须以 ``minirocket_engineering_ablation`` 报告。
    """

    model_id = "minirocket_engineering_ablation"
    scientific_status = "ablation_not_reference_implementation"

    def __init__(self, n_kernels: int = 1_000, alpha: float = 1.0, seed: int = 42) -> None:
        super().__init__(n_kernels=n_kernels, alpha=alpha, seed=seed)

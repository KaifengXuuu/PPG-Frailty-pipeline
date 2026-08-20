"""Self-contained experimental effect-size ShapeFormer.

自足的实验性效应量 ShapeFormer。

No external PISD package or machine-specific path is imported here.  Shapelet
discovery is fitted on an explicitly supplied training partition and the fitted
shapelets can therefore be provenance-bound to that outer fold.

本模块不导入外部 PISD 包，也不依赖机器特定路径。Shapelet 发现只在显式传入的
训练分区上拟合，因此所得 shapelet 可以绑定到对应 outer fold 的 provenance。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .pisd_port import PISD_DISCOVERY_METHOD, PisdShapelets

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ExperimentalShapeFormer requires the optional 'deep' dependency: pip install .[deep]"
    ) from exc


EFFECT_SIZE_DISCOVERY_SPECS = {
    # Backward-compatible route name; values are defaults, not frozen inputs.
    "effect_size_fixed_v1": (128, 64),
    # Historical named comparison presets retain their explicit identity.
    "effect_size_fixed_400_ablation": (400, 64),
    "effect_size_fixed_800_ablation": (800, 64),
}
_PARAMETERIZED_EFFECT_SIZE_METHODS = {"effect_size_fixed_v1"}


@dataclass(frozen=True)
class EffectSizeShapelets:
    """Immutable, time-scaled outer-train shapelet bank.

    不可变、具有物理时间尺度并绑定 outer-train 的 shapelet 库。这里故意不为
    discovery 方法、采样率或 outer-fold 身份提供默认值：缺少任一字段都必须
    关闭失败，不能把 PISD 请求静默替换成效应量发现。
    """

    values: np.ndarray
    source_classes: np.ndarray
    effect_sizes: np.ndarray
    fitted_participant_ids: tuple[str, ...]
    discovery_method: str
    input_fs_hz: float
    shapelet_length_samples: int
    shapelet_length_seconds: float
    discovery_stride_samples: int
    outer_repeat_index: int
    outer_fold_index: int
    outer_train_participant_hash: str

    def __post_init__(self) -> None:
        """Validate scientific identity and physical time / 校验科学身份与物理时间。"""

        values = np.asarray(self.values, dtype=np.float32)
        classes = np.asarray(self.source_classes, dtype=np.int64)
        scores = np.asarray(self.effect_sizes, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] == 0:
            raise ValueError("shapelet values must have shape [shapelet, channel, length]")
        if classes.shape != (values.shape[0],) or scores.shape != (values.shape[0],):
            raise ValueError("shapelet metadata length must match the shapelet count")
        if not np.isfinite(values).all() or not np.isfinite(scores).all():
            raise ValueError("shapelet bank contains non-finite values")
        if self.discovery_method not in EFFECT_SIZE_DISCOVERY_SPECS:
            raise ValueError("EffectSizeShapelets requires a named fixed effect-size method")
        if not np.isfinite(self.input_fs_hz) or self.input_fs_hz <= 0.0:
            raise ValueError("input_fs_hz must be finite and positive")
        if self.shapelet_length_samples != values.shape[-1]:
            raise ValueError("shapelet_length_samples must match the stored shapelet width")
        expected_length, expected_stride = EFFECT_SIZE_DISCOVERY_SPECS[self.discovery_method]
        if (
            self.discovery_method not in _PARAMETERIZED_EFFECT_SIZE_METHODS
            and (
                self.shapelet_length_samples != expected_length
                or self.discovery_stride_samples != expected_stride
            )
        ):
            raise ValueError(
                f"{self.discovery_method} requires length/stride "
                f"{expected_length}/{expected_stride} samples"
            )
        if self.shapelet_length_samples <= 0 or self.discovery_stride_samples <= 0:
            raise ValueError("shapelet length and discovery stride must be positive")
        expected_seconds = self.shapelet_length_samples / float(self.input_fs_hz)
        if not np.isclose(self.shapelet_length_seconds, expected_seconds, rtol=0.0, atol=1e-12):
            raise ValueError("shapelet_length_seconds must equal samples / input_fs_hz")
        if self.outer_repeat_index < 0 or self.outer_fold_index < 0:
            raise ValueError("outer repeat/fold indices must be non-negative")
        participant_ids = tuple(sorted(set(str(value) for value in self.fitted_participant_ids)))
        if not participant_ids:
            raise ValueError("fitted_participant_ids cannot be empty")
        expected_hash = _participant_roster_hash(participant_ids)
        if self.outer_train_participant_hash != expected_hash:
            raise ValueError("outer_train_participant_hash does not match fitted participants")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_classes", classes)
        object.__setattr__(self, "effect_sizes", scores)
        object.__setattr__(self, "fitted_participant_ids", participant_ids)

    @property
    def algorithm_id(self) -> str:
        """Compatibility identity without fallback / 无回退的兼容只读标识。"""

        return self.discovery_method


def _participant_roster_hash(participant_ids: tuple[str, ...]) -> str:
    """Hash the exact sorted outer-train roster / 散列精确排序的 outer-train 名单。"""

    payload = "\n".join(sorted(set(str(value) for value in participant_ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _z_normalise_window(window: np.ndarray) -> np.ndarray:
    """Channel-wise stable z-normalisation / 按通道稳定 z 标准化。"""

    mean = window.mean(axis=-1, keepdims=True)
    scale = window.std(axis=-1, keepdims=True)
    return ((window - mean) / np.maximum(scale, 1e-6)).astype(np.float32, copy=False)


def _minimum_distance(series: np.ndarray, candidate: np.ndarray, stride: int) -> float:
    """Minimum normalised window distance / 最小标准化窗口距离。"""

    length = candidate.shape[-1]
    if series.shape[-1] < length:
        return float("inf")
    distances = []
    for start in range(0, series.shape[-1] - length + 1, stride):
        window = _z_normalise_window(series[:, start : start + length])
        distances.append(float(np.mean((window - candidate) ** 2)))
    return min(distances) if distances else float("inf")


def discover_effect_size_shapelets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    participant_ids: list[str] | tuple[str, ...],
    *,
    discovery_method: str,
    input_fs_hz: float,
    shapelet_length: int,
    outer_repeat_index: int,
    outer_fold_index: int,
    shapelets_per_class: int = 3,
    stride: int = 64,
    max_candidates_per_class: int = 128,
    seed: int = 42,
) -> EffectSizeShapelets:
    """Discover candidates ranked by absolute standardised class separation.

    按绝对标准化类间差异发现候选 shapelet。每个候选先计算到每条训练序列的最小
    距离，再以候选所属类别与其余类别间的 Cohen 型效应量排名。整个过程只读取
    ``x_train/y_train``；调用者必须传入 outer-train 数据及其 repeat/fold 身份。
    """

    # English: This named function implements exactly one discovery method.
    # 中文：该具名函数只实现一种发现方法；未知方法或 PISD 请求绝不隐式回退。
    if discovery_method not in EFFECT_SIZE_DISCOVERY_SPECS:
        raise ValueError(
            "this route implements only explicitly named fixed effect-size methods; "
            "PISD and unknown discovery methods never fall back"
        )
    if not np.isfinite(input_fs_hz) or input_fs_hz <= 0.0:
        raise ValueError("input_fs_hz must be finite and positive")
    if outer_repeat_index < 0 or outer_fold_index < 0:
        raise ValueError("outer repeat/fold indices must be non-negative")

    x = np.asarray(x_train, dtype=np.float32)
    y = np.asarray(y_train)
    if x.ndim != 3 or y.shape != (x.shape[0],):
        raise ValueError("x_train must be [sample,channel,time] and y_train one-dimensional")
    if len(participant_ids) != x.shape[0]:
        raise ValueError("one participant id is required per training sample")
    expected_length, expected_stride = EFFECT_SIZE_DISCOVERY_SPECS[discovery_method]
    if (
        discovery_method not in _PARAMETERIZED_EFFECT_SIZE_METHODS
        and (shapelet_length != expected_length or stride != expected_stride)
    ):
        raise ValueError(
            f"{discovery_method} requires shapelet_length={expected_length} and stride={expected_stride}"
        )
    if shapelet_length > x.shape[-1]:
        raise ValueError("shapelet_length must fit within every input series")
    if stride <= 0 or shapelets_per_class <= 0 or max_candidates_per_class <= 0:
        raise ValueError("stride and candidate counts must be positive")

    rng = np.random.default_rng(seed)
    selected_values: list[np.ndarray] = []
    selected_classes: list[int] = []
    selected_scores: list[float] = []
    for class_value in sorted(np.unique(y).tolist()):
        candidate_windows: list[np.ndarray] = []
        for sample_index in np.flatnonzero(y == class_value):
            for start in range(0, x.shape[-1] - shapelet_length + 1, stride):
                candidate_windows.append(
                    _z_normalise_window(x[sample_index, :, start : start + shapelet_length])
                )
        if len(candidate_windows) > max_candidates_per_class:
            keep = rng.choice(len(candidate_windows), max_candidates_per_class, replace=False)
            candidate_windows = [candidate_windows[index] for index in sorted(keep.tolist())]

        ranked: list[tuple[float, int, np.ndarray]] = []
        positive = y == class_value
        for candidate_index, candidate in enumerate(candidate_windows):
            distances = np.asarray(
                [_minimum_distance(series, candidate, stride) for series in x], dtype=np.float64
            )
            positive_values = distances[positive]
            negative_values = distances[~positive]
            if positive_values.size == 0 or negative_values.size == 0:
                continue
            pooled_variance = 0.5 * (
                float(np.var(positive_values, ddof=0)) + float(np.var(negative_values, ddof=0))
            )
            effect = (float(np.mean(negative_values)) - float(np.mean(positive_values))) / max(
                pooled_variance**0.5, 1e-8
            )
            ranked.append((abs(effect), candidate_index, candidate))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        for score, _, candidate in ranked[:shapelets_per_class]:
            selected_values.append(candidate)
            selected_classes.append(int(class_value))
            selected_scores.append(float(score))

    if not selected_values:
        raise ValueError("shapelet discovery produced no valid candidate")
    roster = tuple(sorted(set(str(value) for value in participant_ids)))
    return EffectSizeShapelets(
        values=np.stack(selected_values),
        source_classes=np.asarray(selected_classes),
        effect_sizes=np.asarray(selected_scores),
        fitted_participant_ids=roster,
        discovery_method=discovery_method,
        input_fs_hz=float(input_fs_hz),
        shapelet_length_samples=int(shapelet_length),
        shapelet_length_seconds=float(shapelet_length / input_fs_hz),
        discovery_stride_samples=int(stride),
        outer_repeat_index=int(outer_repeat_index),
        outer_fold_index=int(outer_fold_index),
        outer_train_participant_hash=_participant_roster_hash(roster),
    )


class ExperimentalShapeFormer(nn.Module):
    """Patch-plus-scalar-distance experimental ShapeFormer ablation.

    This architecture is intentionally not the literature ShapeFormer
    reference. When supplied a channel-specific OSD bank it is the separately
    named scalar-distance downstream ablation; the canonical route is
    implemented by LiteratureShapeFormerChannelSpecificOSD.
    """

    model_status = "experimental"
    external_pisd_supported = False
    raw_sample_token_attention = False
    attention_input_route = "non_overlapping_patch_embedding"

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        shapelets: EffectSizeShapelets | PisdShapelets,
        hidden_channels: int = 64,
        dropout: float = 0.2,
        patch_size_samples: int = 16,
        attention_heads: int = 4,
        attention_layers: int = 1,
        distance_position_chunk_size: int = 256,
        input_fs_hz: float | None = None,
        attention_feedforward_channels: int | None = None,
    ) -> None:
        """Build only from a provenance-complete fold-local bank.

        仅从 provenance 完整的 fold-local shapelet 库构建模型。
        """

        super().__init__()
        if not isinstance(shapelets, (EffectSizeShapelets, PisdShapelets)):
            raise TypeError(
                "ShapeFormer requires a provenance-complete EffectSizeShapelets or "
                "PisdShapelets bank"
            )
        if isinstance(shapelets, EffectSizeShapelets) and (
            shapelets.discovery_method not in EFFECT_SIZE_DISCOVERY_SPECS
        ):
            raise ValueError("effect-size bank identity mismatch")
        if isinstance(shapelets, PisdShapelets) and (
            shapelets.discovery_method != PISD_DISCOVERY_METHOD
        ):
            raise ValueError("PISD bank identity mismatch")
        if input_fs_hz is None or not np.isfinite(input_fs_hz) or input_fs_hz <= 0.0:
            raise ValueError("input_fs_hz is required and must be finite and positive")
        if not np.isclose(input_fs_hz, shapelets.input_fs_hz, rtol=0.0, atol=1e-12):
            raise ValueError("model input_fs_hz must match the fitted shapelet bank")
        if n_channels <= 0 or n_classes <= 1:
            raise ValueError("n_channels must be positive and n_classes must exceed one")
        if patch_size_samples < 2:
            raise ValueError(
                "patch_size_samples must be at least 2; raw sample-token attention is forbidden"
            )
        if hidden_channels <= 0 or attention_heads <= 0:
            raise ValueError("hidden_channels and attention_heads must be positive")
        if hidden_channels % attention_heads != 0:
            raise ValueError("hidden_channels must be divisible by attention_heads")
        feedforward_channels = (
            int(hidden_channels) * 2
            if attention_feedforward_channels is None
            else int(attention_feedforward_channels)
        )
        if attention_layers <= 0 or feedforward_channels <= 0:
            raise ValueError(
                "attention_layers and attention_feedforward_channels must be positive"
            )
        if distance_position_chunk_size <= 0:
            raise ValueError("distance_position_chunk_size must be positive")
        if not np.isfinite(dropout) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite in [0,1)")

        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.hidden_channels = int(hidden_channels)
        self.attention_heads = int(attention_heads)
        self.attention_layers = int(attention_layers)
        self.attention_feedforward_channels = feedforward_channels
        self.classifier_dropout = float(dropout)
        self.discovery_method = shapelets.discovery_method
        if isinstance(shapelets, PisdShapelets):
            if len(shapelets.channel_schema) != n_channels:
                raise ValueError("OSD/PISD channel_schema differs from model input channels")
            self.model_status = "ablation_channel_specific_scalar_distance"
            self.local_pisd_port_supported = False
            self.channel_specific_osd_supported = True
            self.discovery_registry_role = "ablation"
            self.parity_claim = "not_literature_shapeformer_scalar_distance_ablation"
            self.shapelet_count = shapelets.count
            self.shapelet_length = None
            self.shapelet_length_samples = None
            self.shapelet_length_seconds = None
            self.shapelet_lengths = tuple(int(value) for value in shapelets.candidate_lengths)
            self.shapelet_source_channel_names = tuple(shapelets.source_channel_names)
            self.shapelet_candidate_records = shapelets.candidate_records()
            self.num_pip_ratio = float(shapelets.num_pip_ratio)
            self.shapelets_per_class = int(shapelets.shapelets_per_class)
            self.max_discovery_windows = int(shapelets.max_discovery_windows)
            self.discovery_balance = str(shapelets.discovery_balance)
            self.information_gain_split_rule = str(
                shapelets.information_gain_split_rule
            )
            self.discovery_stride_samples = None
            self.register_buffer(
                "shapelet_source_channels",
                torch.as_tensor(shapelets.source_channels, dtype=torch.long),
                persistent=True,
            )
            self.shapelets = nn.ParameterList(
                [nn.Parameter(torch.from_numpy(value.copy())) for value in shapelets.values]
            )
        else:
            values = np.asarray(shapelets.values, dtype=np.float32)
            if values.ndim != 3 or values.shape[1] != n_channels:
                raise ValueError("fixed effect-size shapelets must be [count,n_channels,length]")
            self.model_status = "experimental"
            self.local_pisd_port_supported = False
            self.channel_specific_osd_supported = False
            self.discovery_registry_role = "ablation"
            self.parity_claim = "effect_size_ablation_not_pisd"
            self.shapelet_count = int(values.shape[0])
            self.shapelet_length = int(values.shape[-1])
            self.shapelet_length_samples = int(shapelets.shapelet_length_samples)
            self.shapelet_length_seconds = float(shapelets.shapelet_length_seconds)
            self.shapelet_lengths = tuple([self.shapelet_length] * self.shapelet_count)
            self.shapelet_source_channel_names = ()
            self.shapelet_candidate_records = ()
            class_counts = np.unique(shapelets.source_classes, return_counts=True)[1]
            if class_counts.size < 2 or not np.all(class_counts == class_counts[0]):
                raise ValueError("effect-size bank must preserve equal shapelet count per class")
            self.num_pip_ratio = None
            self.shapelets_per_class = int(class_counts[0])
            self.max_discovery_windows = None
            self.discovery_balance = None
            self.information_gain_split_rule = None
            self.discovery_stride_samples = int(shapelets.discovery_stride_samples)
            self.shapelets = nn.Parameter(torch.from_numpy(values.copy()))
        self.input_fs_hz = float(input_fs_hz)
        self.distance_position_chunk_size = int(distance_position_chunk_size)
        self.patch_size_samples = int(patch_size_samples)
        self.patch_duration_seconds = float(self.patch_size_samples / self.input_fs_hz)
        self.outer_repeat_index = int(shapelets.outer_repeat_index)
        self.outer_fold_index = int(shapelets.outer_fold_index)
        self.outer_train_participant_hash = shapelets.outer_train_participant_hash
        self.fitted_participant_ids = shapelets.fitted_participant_ids

        # English: Non-overlapping Conv1d patches create one token per patch before
        # generic attention. A patch size of one is rejected above.
        # 中文：非重叠 Conv1d patch 在通用注意力之前生成 token；上方明确拒绝
        # patch_size=1，从结构上杜绝原始采样点 token 注意力。
        self.patch_embedding = nn.Conv1d(
            n_channels,
            hidden_channels,
            kernel_size=self.patch_size_samples,
            stride=self.patch_size_samples,
            bias=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=attention_heads,
            dim_feedforward=feedforward_channels,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.patch_attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=attention_layers,
            enable_nested_tensor=False,
        )
        self.patch_norm = nn.LayerNorm(hidden_channels)
        self.feature_dim = hidden_channels + self.shapelet_count
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def provenance(self) -> dict[str, object]:
        """Return required model identity fields / 返回必需的模型身份字段。"""

        return {
            "model_status": self.model_status,
            "discovery_method": self.discovery_method,
            "external_pisd_supported": self.external_pisd_supported,
            "local_pisd_port_supported": self.local_pisd_port_supported,
            "channel_specific_osd_supported": self.channel_specific_osd_supported,
            "discovery_registry_role": self.discovery_registry_role,
            "parity_claim": self.parity_claim,
            "attention_input_route": self.attention_input_route,
            "raw_sample_token_attention": self.raw_sample_token_attention,
            "input_fs_hz": self.input_fs_hz,
            "shapelet_length_samples": self.shapelet_length_samples,
            "shapelet_length_seconds": self.shapelet_length_seconds,
            "shapelet_lengths": self.shapelet_lengths,
            "shapelet_count": self.shapelet_count,
            "shapelet_source_channel_names": self.shapelet_source_channel_names,
            "shapelet_candidate_records": self.shapelet_candidate_records,
            "information_gain_split_rule": self.information_gain_split_rule,
            "hidden_channels": self.hidden_channels,
            "attention_heads": self.attention_heads,
            "attention_layers": self.attention_layers,
            "attention_feedforward_channels": (
                self.attention_feedforward_channels
            ),
            "classifier_dropout": self.classifier_dropout,
            "distance_position_chunk_size": self.distance_position_chunk_size,
            "patch_size_samples": self.patch_size_samples,
            "patch_duration_seconds": self.patch_duration_seconds,
            "outer_repeat_index": self.outer_repeat_index,
            "outer_fold_index": self.outer_fold_index,
            "outer_train_participant_hash": self.outer_train_participant_hash,
            "fitted_participant_ids": self.fitted_participant_ids,
        }

    @staticmethod
    def _normalise_last(values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=-1, keepdim=True)
        scale = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (values - mean) / scale

    @staticmethod
    def _complexity_last(values: torch.Tensor) -> torch.Tensor:
        return (values.diff(dim=-1).square().sum(dim=-1) + 1e-3).sqrt()

    def _channel_specific_similarity(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Exact stride-one OSD/PISD distance, chunked only for memory."""

        outputs: list[torch.Tensor] = []
        for index, candidate in enumerate(self.shapelets):
            width = int(candidate.numel())
            if x.shape[-1] < width:
                raise ValueError("input is shorter than one fitted OSD/PISD candidate")
            channel = int(self.shapelet_source_channels[index].item())
            series = x[:, channel, :]
            windows = series.unfold(-1, width, 1)
            valid = None
            if mask is not None:
                valid = mask.to(dtype=torch.float32).unfold(-1, width, 1).amin(-1) > 0
                if bool(torch.any(~valid.any(dim=-1))):
                    raise ValueError("each sample needs one fully valid candidate-length window")
            normalised_candidate = self._normalise_last(candidate)
            candidate_complexity = self._complexity_last(normalised_candidate)
            best = torch.full(
                (x.shape[0],), torch.inf, dtype=x.dtype, device=x.device
            )
            for offset in range(0, windows.shape[1], self.distance_position_chunk_size):
                chunk = windows[:, offset : offset + self.distance_position_chunk_size, :]
                normalised = self._normalise_last(chunk)
                base = (normalised - normalised_candidate).square().mean(dim=-1).sqrt()
                complexity = self._complexity_last(normalised)
                correction = torch.maximum(complexity, candidate_complexity) / torch.minimum(
                    complexity, candidate_complexity
                ).clamp_min(1e-8)
                distance = base * correction
                if valid is not None:
                    distance = distance.masked_fill(
                        ~valid[:, offset : offset + self.distance_position_chunk_size],
                        torch.inf,
                    )
                best = torch.minimum(best, distance.amin(dim=-1))
            outputs.append(-best)
        return torch.stack(outputs, dim=-1)

    def _fixed_shapelet_similarity(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Fixed-width ablation distance without the old five-dimensional broadcast."""

        width = int(self.shapelet_length)
        if x.shape[-1] < width:
            raise ValueError("input is shorter than the fitted shapelets")
        windows = x.unfold(dimension=-1, size=width, step=1)
        valid = None
        if mask is not None:
            valid = mask.to(dtype=torch.float32).unfold(-1, width, 1).amin(-1) > 0
            if bool(torch.any(~valid.any(dim=-1))):
                raise ValueError("each sample needs one fully valid shapelet-length window")
        outputs: list[torch.Tensor] = []
        for candidate in self.shapelets:
            best = torch.full(
                (x.shape[0],), torch.inf, dtype=x.dtype, device=x.device
            )
            for offset in range(0, windows.shape[2], self.distance_position_chunk_size):
                chunk = windows[:, :, offset : offset + self.distance_position_chunk_size, :]
                distance = (chunk - candidate[:, None, :]).square().mean(dim=(1, 3))
                if valid is not None:
                    distance = distance.masked_fill(
                        ~valid[:, offset : offset + self.distance_position_chunk_size],
                        torch.inf,
                    )
                best = torch.minimum(best, distance.amin(dim=-1))
            outputs.append(-best)
        return torch.stack(outputs, dim=-1)

    def _shapelet_similarity(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Dispatch the reference and fixed ablation without identity fallback."""

        if self.channel_specific_osd_supported:
            return self._channel_specific_similarity(x, mask)
        return self._fixed_shapelet_similarity(x, mask)

    @staticmethod
    def _sinusoidal_position(tokens: torch.Tensor) -> torch.Tensor:
        """Return deterministic physical-order encoding / 返回确定性物理顺序编码。"""

        token_count, width = tokens.shape[1], tokens.shape[2]
        position = torch.arange(
            token_count, device=tokens.device, dtype=tokens.dtype
        )[:, None]
        even_width = (width + 1) // 2
        scale = torch.exp(
            torch.arange(even_width, device=tokens.device, dtype=tokens.dtype)
            * (-np.log(10_000.0) / max(even_width, 1))
        )
        angles = position * scale[None, :]
        encoding = torch.zeros(
            token_count, width, device=tokens.device, dtype=tokens.dtype
        )
        encoding[:, 0::2] = torch.sin(angles[:, : encoding[:, 0::2].shape[1]])
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)

    def _patch_features(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Encode fully valid patches with masked attention and pooling.

        仅编码完全有效的 patch，并在注意力与池化阶段持续使用 mask。
        """

        if x.shape[-1] < self.patch_size_samples:
            raise ValueError("input is shorter than one configured patch")
        patch_valid: torch.Tensor | None = None
        if mask is not None:
            patch_valid = (
                mask.to(dtype=torch.float32)
                .unfold(-1, self.patch_size_samples, self.patch_size_samples)
                .amin(-1)
                > 0
            )
            if bool(torch.any(~patch_valid.any(dim=-1))):
                raise ValueError("each sample needs one fully valid attention patch")
        tokens = self.patch_embedding(x).transpose(1, 2)
        tokens = tokens + self._sinusoidal_position(tokens)
        tokens = self.patch_attention(
            tokens,
            src_key_padding_mask=None if patch_valid is None else ~patch_valid,
        )
        tokens = self.patch_norm(tokens)
        if patch_valid is None:
            return tokens.mean(dim=1)
        weights = patch_valid.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward_features(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Concatenate patch and shapelet features / 拼接 patch 与 shapelet 特征。"""

        if x.ndim != 3:
            raise ValueError("ExperimentalShapeFormer expects [batch,channel,time]")
        if mask is not None:
            if mask.ndim != 2 or mask.shape != (x.shape[0], x.shape[-1]):
                raise ValueError("mask must match [batch,time]")
            # English: Placeholder values are removed before both branches.
            # 中文：在两个分支之前清除补齐占位值。
            x = x * mask.to(dtype=x.dtype).unsqueeze(1)
        patch = self._patch_features(x, mask)
        return torch.cat([patch, self._shapelet_similarity(x, mask)], dim=-1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return unnormalised logits / 返回未归一化 logits。"""

        return self.classifier(self.dropout(self.forward_features(x, mask)))

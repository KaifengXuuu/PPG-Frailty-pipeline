"""Mask-aware InceptionTime single networks and probability ensemble.

支持掩码的 InceptionTime 单网络与概率集成。
"""

from __future__ import annotations

from collections.abc import Sequence
import math

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "InceptionTime models require the optional 'deep' dependency: pip install .[deep]"
    ) from exc


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if isinstance(value, float) and (
        not math.isfinite(value) or float(normalized) != value
    ):
        raise ValueError(f"{field} must be a finite positive integer")
    if normalized <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return normalized


def _positive_odd_kernel_sizes(values: Sequence[int]) -> tuple[int, ...]:
    kernels = tuple(_positive_integer(value, field="kernel_sizes") for value in values)
    if not kernels or any(value % 2 == 0 for value in kernels):
        raise ValueError("kernel_sizes must contain positive odd integers")
    return kernels


def masked_global_average(encoded: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Average features over valid columns only / 仅对有效列求特征均值。"""

    if mask is None:
        return encoded.mean(dim=-1)
    if mask.ndim != 2 or mask.shape[0] != encoded.shape[0] or mask.shape[1] != encoded.shape[2]:
        raise ValueError("mask must have shape [batch, time] matching encoded features")
    weights = mask.to(dtype=encoded.dtype).unsqueeze(1)
    denominator = weights.sum(dim=-1)
    if bool(torch.any(denominator <= 0)):
        raise ValueError("every sample must contain at least one valid column")
    return (encoded * weights).sum(dim=-1) / denominator


class InceptionModule(nn.Module):
    """One multi-scale temporal Inception module / 单个多尺度时序 Inception 模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        kernel_sizes: Sequence[int],
        use_bottleneck: bool = True,
        dilation: int = 1,
        pool_size: int = 3,
    ) -> None:
        super().__init__()
        in_channels = _positive_integer(in_channels, field="in_channels")
        out_channels = _positive_integer(out_channels, field="out_channels")
        bottleneck_channels = _positive_integer(
            bottleneck_channels, field="bottleneck_channels"
        )
        dilation = _positive_integer(dilation, field="dilation")
        kernels = _positive_odd_kernel_sizes(kernel_sizes)
        pool_size = _positive_integer(pool_size, field="pool_size")
        if pool_size % 2 == 0:
            raise ValueError("pool_size must be a positive odd integer")
        branch_channels = bottleneck_channels if use_bottleneck and in_channels > 1 else in_channels
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if branch_channels != in_channels
            else nn.Identity()
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                branch_channels,
                out_channels,
                kernel_size=size,
                dilation=int(dilation),
                padding=int(dilation) * (size - 1) // 2,
                bias=False,
            )
            for size in kernels
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(
                kernel_size=int(pool_size),
                stride=1,
                padding=(int(pool_size) - 1) // 2,
            ),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.batch_norm = nn.BatchNorm1d(out_channels * (len(kernels) + 1))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one temporal level / 编码一个时序层级。"""

        bottleneck = self.bottleneck(x)
        branches = [convolution(bottleneck) for convolution in self.convolutions]
        branches.append(self.pool_branch(x))
        return self.activation(self.batch_norm(torch.cat(branches, dim=1)))


class InceptionBlock(nn.Module):
    """Six modules with residual links after every third module.

    六个模块组成的块，每三个模块使用一次残差连接。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        kernel_sizes: Sequence[int],
        depth: int = 6,
        dilation: int = 1,
        pool_size: int = 3,
        residual_interval: int = 3,
    ) -> None:
        super().__init__()
        out_channels = _positive_integer(out_channels, field="out_channels")
        kernel_sizes = _positive_odd_kernel_sizes(kernel_sizes)
        depth = _positive_integer(depth, field="depth")
        residual_interval = _positive_integer(
            residual_interval, field="residual_interval"
        )
        module_channels = out_channels * (len(kernel_sizes) + 1)
        self.modules_list = nn.ModuleList()
        self.residuals = nn.ModuleDict()
        current_channels = in_channels
        residual_channels = in_channels
        for index in range(depth):
            self.modules_list.append(
                InceptionModule(
                    current_channels,
                    out_channels,
                    bottleneck_channels,
                    kernel_sizes,
                    use_bottleneck=True,
                    dilation=dilation,
                    pool_size=pool_size,
                )
            )
            current_channels = module_channels
            if (index + 1) % residual_interval == 0:
                if residual_channels == module_channels:
                    # English: The reviewed shortcut applies BN only when the
                    # width already matches.  An extra 1x1 convolution here
                    # would add 16,384 unintended full-model parameters.
                    # 中文：宽度已匹配时，已审查 shortcut 仅使用 BN；若额外加入
                    # 1x1 卷积，完整模型会错误增加 16,384 个参数。
                    self.residuals[str(index)] = nn.BatchNorm1d(module_channels)
                else:
                    self.residuals[str(index)] = nn.Sequential(
                        nn.Conv1d(residual_channels, module_channels, kernel_size=1, bias=False),
                        nn.BatchNorm1d(module_channels),
                    )
                residual_channels = module_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply modules and residual paths / 应用模块与残差路径。"""

        residual = x
        output = x
        for index, module in enumerate(self.modules_list):
            output = module(output)
            key = str(index)
            if key in self.residuals:
                output = F.relu(output + self.residuals[key](residual), inplace=True)
                residual = output
        return output


class InceptionTimeSingleNetwork(nn.Module):
    """Full or small reviewed InceptionTime single network.

    完整版或小型版已审查 InceptionTime 单网络。特征矩阵的列掩码会在所有卷积后
    进入最终池化，从而避免补零列参与全局平均。
    """

    _VARIANTS = {
        "full": {
            "out_channels": 32,
            "bottleneck": 32,
            "kernels": (39, 19, 9),
            "depth": 6,
        },
        "small": {
            "out_channels": 16,
            "bottleneck": 16,
            "kernels": (39, 19, 9),
            "depth": 3,
        },
    }

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        variant: str = "full",
        dropout: float = 0.2,
        kernel_sizes: Sequence[int] | None = None,
        dilation: int = 1,
        pool_size: int = 3,
        out_channels: int | None = None,
        bottleneck_channels: int | None = None,
        depth: int | None = None,
        residual_interval: int = 3,
    ) -> None:
        super().__init__()
        if variant not in self._VARIANTS:
            raise ValueError(f"unknown InceptionTime variant: {variant}")
        settings = self._VARIANTS[variant]
        kernels = _positive_odd_kernel_sizes(
            settings["kernels"] if kernel_sizes is None else kernel_sizes
        )
        dilation = _positive_integer(dilation, field="dilation")
        pool_size = _positive_integer(pool_size, field="pool_size")
        if pool_size % 2 == 0:
            raise ValueError("pool_size must be a positive odd integer")
        n_channels = _positive_integer(n_channels, field="n_channels")
        n_classes = _positive_integer(n_classes, field="n_classes")
        if n_classes <= 1:
            raise ValueError("n_channels must be positive and n_classes must exceed one")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        resolved_out_channels = _positive_integer(
            settings["out_channels"] if out_channels is None else out_channels,
            field="out_channels",
        )
        resolved_bottleneck = _positive_integer(
            settings["bottleneck"]
            if bottleneck_channels is None
            else bottleneck_channels,
            field="bottleneck_channels",
        )
        resolved_depth = _positive_integer(
            settings["depth"] if depth is None else depth, field="depth"
        )
        residual_interval = _positive_integer(
            residual_interval, field="residual_interval"
        )
        self.variant = variant
        self.kernel_sizes = kernels
        self.dilation = int(dilation)
        self.pool_size = int(pool_size)
        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.out_channels = resolved_out_channels
        self.bottleneck_channels = resolved_bottleneck
        self.depth = resolved_depth
        self.branch_count = len(kernels) + 1
        self.residual_interval = int(residual_interval)
        self.classifier_dropout = float(dropout)
        self.feature_dim = self.out_channels * self.branch_count
        self.encoder = InceptionBlock(
            n_channels,
            self.out_channels,
            self.bottleneck_channels,
            kernels,
            depth=self.depth,
            dilation=int(dilation),
            pool_size=int(pool_size),
            residual_interval=int(residual_interval),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return mask-aware pooled features / 返回掩码感知的池化特征。"""

        if x.ndim != 3:
            raise ValueError("InceptionTime expects [batch, channel, time]")
        if mask is not None:
            if mask.ndim != 2 or mask.shape != (x.shape[0], x.shape[2]):
                raise ValueError("mask must match [batch,time]")
            # English: Zero invalid columns before convolution so stored padding
            # values cannot leak into neighbouring valid representations.
            # 中文：卷积前将无效列归零，防止补齐占位值污染相邻有效表示。
            x = x * mask.to(dtype=x.dtype).unsqueeze(1)
        encoded = self.encoder(x)
        return masked_global_average(encoded, mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return unnormalised logits / 返回未归一化 logits。"""

        return self.classifier(self.dropout(self.forward_features(x, mask)))


class FullInceptionTimeSingleNetwork(InceptionTimeSingleNetwork):
    """Full-capacity reviewed configuration / 完整容量已审查配置。"""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        dropout: float = 0.2,
        kernel_sizes: Sequence[int] | None = None,
        dilation: int = 1,
        pool_size: int = 3,
        out_channels: int | None = None,
        bottleneck_channels: int | None = None,
        depth: int | None = None,
        residual_interval: int = 3,
    ) -> None:
        super().__init__(
            n_channels,
            n_classes,
            variant="full",
            dropout=dropout,
            kernel_sizes=kernel_sizes,
            dilation=dilation,
            pool_size=pool_size,
            out_channels=out_channels,
            bottleneck_channels=bottleneck_channels,
            depth=depth,
            residual_interval=residual_interval,
        )


class SmallInceptionTimeSingleNetwork(InceptionTimeSingleNetwork):
    """Reduced-capacity reviewed configuration / 小容量已审查配置。"""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        dropout: float = 0.2,
        kernel_sizes: Sequence[int] | None = None,
        dilation: int = 1,
        pool_size: int = 3,
        out_channels: int | None = None,
        bottleneck_channels: int | None = None,
        depth: int | None = None,
        residual_interval: int = 3,
    ) -> None:
        super().__init__(
            n_channels,
            n_classes,
            variant="small",
            dropout=dropout,
            kernel_sizes=kernel_sizes,
            dilation=dilation,
            pool_size=pool_size,
            out_channels=out_channels,
            bottleneck_channels=bottleneck_channels,
            depth=depth,
            residual_interval=residual_interval,
        )


class InceptionTimeProbabilityEnsemble(nn.Module):
    """One-or-more independent members averaged in probability space."""

    def __init__(self, members: Sequence[nn.Module], member_seeds: Sequence[int]) -> None:
        super().__init__()
        if not members or len(members) != len(member_seeds):
            raise ValueError("ensemble needs at least one member and one aligned seed per member")
        normalized_seeds_list: list[int] = []
        for seed in member_seeds:
            if isinstance(seed, bool):
                raise ValueError("ensemble member_seeds must be integer values")
            try:
                normalized = int(seed)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("ensemble member_seeds must be integer values") from exc
            if isinstance(seed, float) and (
                not math.isfinite(seed) or float(normalized) != seed
            ):
                raise ValueError("ensemble member_seeds must be finite integer values")
            normalized_seeds_list.append(normalized)
        normalized_seeds = tuple(normalized_seeds_list)
        if len(normalized_seeds) != len(set(normalized_seeds)):
            raise ValueError("ensemble member_seeds must be unique")
        parameter_ids = [id(parameter) for member in members for parameter in member.parameters()]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("ensemble members must not share parameter objects")
        self.members = nn.ModuleList(members)
        self.member_seeds = normalized_seeds

    @staticmethod
    def average_member_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
        """Validate and average a ``[member,batch,class]`` tensor."""

        if probabilities.ndim != 3 or probabilities.shape[0] < 1:
            raise ValueError("member probabilities must have shape [member,batch,class]")
        if probabilities.shape[1] < 1 or probabilities.shape[2] < 2:
            raise ValueError("probability tensors need a non-empty batch and at least two classes")
        if not bool(torch.isfinite(probabilities).all()) or bool(torch.any(probabilities < 0)):
            raise ValueError("member probabilities must be finite and non-negative")
        if not torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones_like(probabilities[..., 0]),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("each member probability row must sum to one")
        return probabilities.mean(dim=0)

    def member_provenance(self) -> tuple[dict[str, int], ...]:
        """Return stable member index/seed records for OOF and bundle manifests."""

        return tuple(
            {"member_index": index, "training_seed": seed}
            for index, seed in enumerate(self.member_seeds)
        )

    def member_probabilities(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return ``[member,batch,class]`` probabilities / 返回成员概率张量。"""

        return torch.stack([F.softmax(member(x, mask), dim=-1) for member in self.members], dim=0)

    def predict_probabilities(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Average member probabilities / 对成员概率求平均。"""

        return self.average_member_probabilities(self.member_probabilities(x, mask))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return log-probabilities suitable for NLL loss / 返回可用于 NLL 的对数概率。"""

        return torch.log(self.predict_probabilities(x, mask).clamp_min(1e-12))


class InceptionTimeFiveMemberProbabilityEnsemble(InceptionTimeProbabilityEnsemble):
    """Backward-compatible name for the now N-member probability ensemble.

    Existing five-member configurations remain unchanged; callers may now pass
    any non-empty, uniquely seeded roster.
    """


DEFAULT_ENSEMBLE_MEMBER_SEEDS = (50042, 60042, 70042, 80042, 90042)
# Backward-compatible import name; no runtime equality check uses this roster.
CANONICAL_ENSEMBLE_MEMBER_SEEDS = DEFAULT_ENSEMBLE_MEMBER_SEEDS

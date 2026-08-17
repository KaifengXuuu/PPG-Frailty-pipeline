"""Mask-aware InceptionTime single networks and probability ensemble.

支持掩码的 InceptionTime 单网络与概率集成。
"""

from __future__ import annotations

from collections.abc import Sequence

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "InceptionTime models require the optional 'deep' dependency: pip install .[deep]"
    ) from exc


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
    ) -> None:
        super().__init__()
        if dilation <= 0:
            raise ValueError('dilation must be positive')
        if len(kernel_sizes) != 3 or any(size <= 0 or size % 2 == 0 for size in kernel_sizes):
            raise ValueError("kernel_sizes must contain three positive odd integers")
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
            for size in kernel_sizes
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.batch_norm = nn.BatchNorm1d(out_channels * 4)
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
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        module_channels = out_channels * 4
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
                )
            )
            current_channels = module_channels
            if (index + 1) % 3 == 0:
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
    ) -> None:
        super().__init__()
        if variant not in self._VARIANTS:
            raise ValueError(f"unknown InceptionTime variant: {variant}")
        settings = self._VARIANTS[variant]
        kernels = tuple(
            int(value) for value in (
                settings.get('kernels') if kernel_sizes is None else kernel_sizes
            )
        )
        if len(kernels) != 3 or any(value <= 0 or value % 2 == 0 for value in kernels):
            raise ValueError('kernel_sizes must contain three positive odd integers')
        if dilation <= 0:
            raise ValueError('dilation must be positive')
        self.variant = variant
        self.kernel_sizes = kernels
        self.dilation = int(dilation)
        self.feature_dim = int(settings["out_channels"]) * 4
        self.encoder = InceptionBlock(
            n_channels,
            int(settings["out_channels"]),
            int(settings["bottleneck"]),
            settings["kernels"],
            depth=int(settings["depth"]),
        )
        if kernel_sizes is not None or dilation != 1:
            # English: Rebuild only for an explicit time-scale ablation; the default
            # path remains behaviour-compatible with the reviewed parameter snapshot.
            # 中文：仅显式时间尺度消融重建编码器；默认路线保持已审查快照行为。
            self.encoder = InceptionBlock(
                n_channels,
                int(settings.get('out_channels')),
                int(settings.get('bottleneck')),
                kernels,
                depth=int(settings.get('depth')),
                dilation=int(dilation),
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

    def __init__(self, n_channels: int, n_classes: int, dropout: float = 0.2) -> None:
        super().__init__(n_channels, n_classes, variant="full", dropout=dropout)


class SmallInceptionTimeSingleNetwork(InceptionTimeSingleNetwork):
    """Reduced-capacity reviewed configuration / 小容量已审查配置。"""

    def __init__(self, n_channels: int, n_classes: int, dropout: float = 0.2) -> None:
        super().__init__(n_channels, n_classes, variant="small", dropout=dropout)


class InceptionTimeFiveMemberProbabilityEnsemble(nn.Module):
    """Exactly five independent members averaged in probability space.

    恰好五个独立成员，并在概率空间取平均。构造函数检查成员不能共享参数对象，
    防止“同一权重重复五次”伪装成集成。
    """

    def __init__(self, members: Sequence[nn.Module], member_seeds: Sequence[int]) -> None:
        super().__init__()
        if len(members) != 5 or len(member_seeds) != 5:
            raise ValueError("the canonical ensemble requires exactly five members and five seeds")
        if len(set(int(seed) for seed in member_seeds)) != 5:
            raise ValueError("ensemble member seeds must be distinct")
        parameter_ids = [id(parameter) for member in members for parameter in member.parameters()]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("ensemble members must not share parameter objects")
        self.members = nn.ModuleList(members)
        self.member_seeds = tuple(int(seed) for seed in member_seeds)

    def member_probabilities(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return ``[member,batch,class]`` probabilities / 返回成员概率张量。"""

        return torch.stack([F.softmax(member(x, mask), dim=-1) for member in self.members], dim=0)

    def predict_probabilities(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Average member probabilities / 对成员概率求平均。"""

        return self.member_probabilities(x, mask).mean(dim=0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return log-probabilities suitable for NLL loss / 返回可用于 NLL 的对数概率。"""

        return torch.log(self.predict_probabilities(x, mask).clamp_min(1e-12))

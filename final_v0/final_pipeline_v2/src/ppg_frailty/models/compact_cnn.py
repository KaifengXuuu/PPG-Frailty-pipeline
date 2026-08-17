"""Compact one-dimensional CNN reference model.

紧凑型一维卷积神经网络参考实现。

This module intentionally preserves the reviewed 79,139-parameter architecture
for the canonical eight-channel, three-class configuration.  The model accepts
``[batch, channel, time]`` tensors and exposes ``forward_features`` so that the
same encoder can be reused by the file-bag fusion route.

本模块有意保留已经审查的架构；在八通道、三分类配置下共有 79,139 个可训练参数。
模型接收 ``[批次, 通道, 时间]`` 张量，并公开 ``forward_features``，以便文件袋融合
路线复用同一个信号编码器。
"""

from __future__ import annotations

from collections.abc import Sequence

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without deep extra
    raise ImportError(
        "CompactCNN1D requires the optional 'deep' dependency: pip install .[deep]"
    ) from exc


class CompactCNN1D(nn.Module):
    """Reviewed compact CNN for raw-window frailty classification.

    已审查的原始窗口衰弱分类紧凑 CNN。全局平均池化使时间长度可变，同时避免
    将窗口位置误当成固定语义。
    """

    feature_dim: int = 128

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        dropout: float = 0.20,
        kernel_sizes: Sequence[int] = (9, 9, 7),
        dilations: Sequence[int] = (1, 1, 1),
        pool_sizes: Sequence[int] = (4, 4),
    ) -> None:
        super().__init__()
        kernels = tuple(int(value) for value in kernel_sizes)
        dilation_values = tuple(int(value) for value in dilations)
        pools = tuple(int(value) for value in pool_sizes)
        if len(kernels) != 3 or any(value <= 0 or value % 2 == 0 for value in kernels):
            raise ValueError('kernel_sizes must contain three positive odd integers')
        if len(dilation_values) != 3 or any(value <= 0 for value in dilation_values):
            raise ValueError('dilations must contain three positive integers')
        if len(pools) != 2 or any(value <= 0 for value in pools):
            raise ValueError('pool_sizes must contain two positive integers')
        self.kernel_sizes = kernels
        self.dilations = dilation_values
        self.pool_sizes = pools
        if n_channels <= 0 or n_classes <= 1:
            raise ValueError("n_channels must be positive and n_classes must exceed one")
        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.stage_channels = (32, 64, 128)
        self.stage_dropouts = (0.10, 0.15)
        self.classifier_dropout = float(dropout)

        # English: Three convolutional stages are deliberately small enough for
        # an embedded CPU deployment target.
        # 中文：三个卷积阶段特意保持紧凑，以适配嵌入式 CPU 部署目标。
        self.encoder = nn.Sequential(
            nn.Conv1d(
                n_channels, 32, kernel_size=kernels[0],
                dilation=dilation_values[0],
                padding=dilation_values[0] * (kernels[0] - 1) // 2,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pools[0]),
            nn.Dropout(0.10),
            nn.Conv1d(
                32, 64, kernel_size=kernels[1],
                dilation=dilation_values[1],
                padding=dilation_values[1] * (kernels[1] - 1) // 2,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pools[1]),
            nn.Dropout(0.15),
            nn.Conv1d(
                64, 128, kernel_size=kernels[2],
                dilation=dilation_values[2],
                padding=dilation_values[2] * (kernels[2] - 1) // 2,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return one feature vector per window / 返回每个窗口的特征向量。

        ``mask`` is accepted for a uniform encoder interface.  Raw windows are
        expected to be pre-segmented without padded columns, so a non-trivial
        mask is rejected rather than silently producing biased pooling.

        为统一编码器接口而接受 ``mask``。原始窗口应当预先切分且不含填充列；若
        传入非全真掩码则显式拒绝，避免静默地产生有偏池化。
        """

        if x.ndim != 3:
            raise ValueError("CompactCNN1D expects [batch, channel, time]")
        if mask is not None and not bool(torch.all(mask)):
            raise ValueError("CompactCNN1D raw-window route does not accept padded samples")
        return self.encoder(x).squeeze(-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return unnormalised class logits / 返回未归一化分类 logits。"""

        return self.classifier(self.dropout(self.forward_features(x, mask)))


def trainable_parameter_count(model: nn.Module) -> int:
    """Count trainable parameters / 统计可训练参数。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

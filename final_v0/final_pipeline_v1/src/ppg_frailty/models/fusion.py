"""File-bag signal/feature fusion without per-window feature duplication.

文件袋信号/特征融合，不向每个窗口复制文件级特征。
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("fusion models require the optional 'deep' dependency") from exc


class FileBagFusionClassifier(nn.Module):
    """Pool a file's window embeddings, then concatenate its feature vector once.

    先汇聚一个文件内的窗口嵌入，再且仅再拼接一次该文件的特征向量。输入
    ``file_features`` 的首维始终是文件批次，不存在窗口维，因此结构上阻止泄漏式复制。
    """

    def __init__(
        self,
        signal_encoder: nn.Module,
        signal_feature_dim: int,
        n_file_features: int,
        n_classes: int,
        *,
        feature_hidden_dim: int = 32,
        fusion_hidden_dim: int = 64,
        pooling: str = "mean",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "attention"}:
            raise ValueError("pooling must be 'mean' or 'attention'")
        if not hasattr(signal_encoder, "forward_features"):
            raise TypeError("signal_encoder must expose forward_features")
        self.signal_encoder = signal_encoder
        self.pooling = pooling
        self.attention = nn.Linear(signal_feature_dim, 1) if pooling == "attention" else None
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_file_features, feature_hidden_dim), nn.ReLU(inplace=True)
        )
        self.feature_dim = fusion_hidden_dim
        self.fusion_encoder = nn.Sequential(
            nn.Linear(signal_feature_dim + feature_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_hidden_dim, n_classes)

    def _pool_windows(self, embeddings: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Pool only valid windows / 仅汇聚有效窗口。"""

        if bool(torch.any(~valid.any(dim=1))):
            raise ValueError("every file bag must contain at least one valid window")
        if self.pooling == "mean":
            weights = valid.to(embeddings.dtype)
            return (embeddings * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True)
        assert self.attention is not None
        logits = self.attention(embeddings).squeeze(-1).masked_fill(~valid, -torch.inf)
        return (embeddings * torch.softmax(logits, dim=1).unsqueeze(-1)).sum(dim=1)

    def forward_features(
        self,
        window_bag: torch.Tensor,
        window_mask: torch.Tensor,
        file_features: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one fused embedding per file / 返回每个文件一个融合嵌入。"""

        if window_bag.ndim != 4:
            raise ValueError("window_bag must be [file,window,channel,time]")
        batch, windows, channels, time = window_bag.shape
        if window_mask.shape != (batch, windows):
            raise ValueError("window_mask must be [file,window]")
        if file_features.ndim != 2 or file_features.shape[0] != batch:
            raise ValueError("file_features must be [file,feature] and is never repeated per window")
        flattened = window_bag.reshape(batch * windows, channels, time)
        valid_flat = window_mask.to(dtype=torch.bool).reshape(batch * windows)
        flattened_mask = None
        if sample_mask is not None:
            if sample_mask.shape != (batch, windows, time):
                raise ValueError("sample_mask must be [file,window,time]")
            flattened_mask = sample_mask.reshape(batch * windows, time)
        # English: Encode only real windows.  Padded bag slots never enter the
        # signal encoder and therefore cannot trip raw-window mask contracts.
        # 中文：只编码真实窗口；补齐的袋位置不会进入信号编码器，也不会触发原始窗口
        # 的掩码契约。
        valid_encoded = self.signal_encoder.forward_features(
            flattened[valid_flat],
            None if flattened_mask is None else flattened_mask[valid_flat],
        )
        encoded = valid_encoded.new_zeros((batch * windows, valid_encoded.shape[-1]))
        encoded[valid_flat] = valid_encoded
        encoded = encoded.reshape(batch, windows, -1)
        pooled_signal = self._pool_windows(encoded, window_mask.to(dtype=torch.bool))
        # English: This call occurs once for each file, after window pooling.
        # 中文：该调用位于窗口汇聚之后，每个文件只执行一次。
        encoded_file = self.feature_encoder(file_features)
        return self.fusion_encoder(torch.cat([pooled_signal, encoded_file], dim=-1))

    def forward(
        self,
        window_bag: torch.Tensor,
        window_mask: torch.Tensor,
        file_features: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return file-level logits / 返回文件级 logits。"""

        return self.classifier(
            self.forward_features(window_bag, window_mask, file_features, sample_mask)
        )

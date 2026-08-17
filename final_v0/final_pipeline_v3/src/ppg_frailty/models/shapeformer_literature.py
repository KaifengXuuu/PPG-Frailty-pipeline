"""Faithful downstream ShapeFormer port for the V2 channel-specific OSD bank.

Discovery is implemented separately in :mod:`pisd_port`.  This module ports
the reviewed upstream ShapeBlock/token-fusion semantics while adding explicit
mask handling and query/position chunking.  Chunking changes only evaluation
order; candidate and attention positions remain stride one and exhaustive.
"""

from __future__ import annotations

import numpy as np

from .pisd_port import (
    PISD_DISCOVERY_METHOD,
    POSITION_SEARCH_NEIGHBOURHOOD_SAMPLES,
    PisdShapelets,
)

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LiteratureShapeFormer requires the optional deep dependency: pip install .[deep]"
    ) from exc


def _upstream_observed_position_width(values: torch.Tensor) -> int:
    """Match upstream position_embedding: observed max value plus one."""

    if values.ndim != 1 or values.numel() == 0 or bool(torch.any(values < 0)):
        raise ValueError("position indices must be a non-empty non-negative vector")
    return int(values.max().item()) + 1


class _ChunkedShapeFormerAttention(nn.Module):
    """Upstream Q/K/V equations with bounded query-axis temporary memory."""

    def __init__(
        self,
        embedding_channels: int,
        attention_heads: int,
        dropout: float,
        query_chunk_size: int,
    ) -> None:
        super().__init__()
        if embedding_channels % attention_heads:
            raise ValueError("embedding_channels must be divisible by attention_heads")
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        self.embedding_channels = int(embedding_channels)
        self.attention_heads = int(attention_heads)
        self.head_channels = self.embedding_channels // self.attention_heads
        self.query_chunk_size = int(query_chunk_size)
        # The upstream implementation scales by the full embedding width.
        self.scale = float(self.embedding_channels) ** -0.5
        self.key = nn.Linear(self.embedding_channels, self.embedding_channels, bias=False)
        self.value = nn.Linear(self.embedding_channels, self.embedding_channels, bias=False)
        self.query = nn.Linear(self.embedding_channels, self.embedding_channels, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.output_norm = nn.LayerNorm(self.embedding_channels)

    def forward(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("attention tokens must have shape [batch,token,channel]")
        batch, token_count, _ = tokens.shape
        if valid_mask is not None:
            if valid_mask.ndim != 2 or valid_mask.shape != (batch, token_count):
                raise ValueError("attention valid_mask must match [batch,token]")
            if bool(torch.any(~valid_mask.any(dim=1))):
                raise ValueError("each attention sample requires at least one valid token")
        keys = self.key(tokens).reshape(
            batch, token_count, self.attention_heads, self.head_channels
        ).permute(0, 2, 3, 1)
        values = self.value(tokens).reshape(
            batch, token_count, self.attention_heads, self.head_channels
        ).transpose(1, 2)
        queries = self.query(tokens).reshape(
            batch, token_count, self.attention_heads, self.head_channels
        ).transpose(1, 2)
        outputs: list[torch.Tensor] = []
        for offset in range(0, token_count, self.query_chunk_size):
            query_chunk = queries[:, :, offset : offset + self.query_chunk_size, :]
            logits = torch.matmul(query_chunk, keys) * self.scale
            if valid_mask is not None:
                logits = logits.masked_fill(
                    ~valid_mask[:, None, None, :],
                    torch.finfo(logits.dtype).min,
                )
            # Upstream constructs a dropout module but never calls it in
            # Attention.forward; applying it here would change model semantics.
            attention = torch.softmax(logits, dim=-1)
            output = torch.matmul(attention, values)
            outputs.append(output)
        merged = torch.cat(outputs, dim=2).transpose(1, 2).reshape(
            batch, token_count, self.embedding_channels
        )
        if valid_mask is not None:
            merged = merged * valid_mask.to(dtype=merged.dtype).unsqueeze(-1)
        return self.output_norm(merged)


class ChannelSpecificShapeBlock(nn.Module):
    """Reviewed ShapeBlock semantics for one variable-length source channel."""

    def __init__(
        self,
        *,
        source_channel: int,
        source_start: int,
        source_end: int,
        shapelet: np.ndarray,
        shape_embedding_channels: int,
        sequence_length: int,
        position_search_neighbourhood_samples: int,
        distance_position_chunk_size: int,
        complexity_norm: float,
        max_complexity_ratio: float,
    ) -> None:
        super().__init__()
        values = np.asarray(shapelet, dtype=np.float32)
        if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
            raise ValueError("ShapeBlock needs one finite variable-length shapelet")
        if source_channel < 0 or source_start < 0 or source_end <= source_start:
            raise ValueError("invalid ShapeBlock source metadata")
        if source_end > sequence_length:
            raise ValueError("ShapeBlock source endpoint exceeds sequence_length")
        if position_search_neighbourhood_samples != POSITION_SEARCH_NEIGHBOURHOOD_SAMPLES:
            raise ValueError("canonical ShapeBlock requires the frozen 128-sample neighbourhood")
        if distance_position_chunk_size <= 0:
            raise ValueError("distance_position_chunk_size must be positive")
        if complexity_norm <= 0.0 or max_complexity_ratio < 1.0:
            raise ValueError("invalid complexity correction constants")
        self.source_channel = int(source_channel)
        self.source_start = int(source_start)
        self.source_end = int(source_end)
        self.sequence_length = int(sequence_length)
        self.position_search_neighbourhood_samples = int(
            position_search_neighbourhood_samples
        )
        self.search_start = max(
            0, self.source_start - self.position_search_neighbourhood_samples
        )
        self.search_end = min(
            self.sequence_length,
            self.source_end + self.position_search_neighbourhood_samples,
        )
        self.distance_position_chunk_size = int(distance_position_chunk_size)
        self.complexity_norm = float(complexity_norm)
        self.max_complexity_ratio = float(max_complexity_ratio)
        self.shapelet = nn.Parameter(torch.from_numpy(values.copy()))
        self.shapelet_length = int(values.size)
        self.selected_projection = nn.Linear(
            self.shapelet_length, int(shape_embedding_channels)
        )
        self.shapelet_projection = nn.Linear(
            self.shapelet_length, int(shape_embedding_channels)
        )
        complexity = np.sqrt(np.sum(np.diff(values) ** 2)) + 1.0 / self.complexity_norm
        self.register_buffer(
            "shapelet_complexity",
            torch.tensor(float(complexity), dtype=torch.float32),
            persistent=True,
        )

    def _best_segments(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        region = values[:, self.search_start : self.search_end]
        if region.shape[-1] < self.shapelet_length:
            raise ValueError("ShapeBlock search neighbourhood is shorter than its shapelet")
        windows = region.unfold(-1, self.shapelet_length, 1)
        window_valid = None
        if valid_mask is not None:
            region_mask = valid_mask[:, self.search_start : self.search_end]
            window_valid = (
                region_mask.to(dtype=torch.float32)
                .unfold(-1, self.shapelet_length, 1)
                .amin(-1)
                > 0
            )
            if bool(torch.any(~window_valid.any(dim=-1))):
                raise ValueError("ShapeBlock has no fully valid segment in its search neighbourhood")
        best_distance = torch.full(
            (values.shape[0],),
            torch.inf,
            dtype=values.dtype,
            device=values.device,
        )
        best_position = torch.zeros(
            values.shape[0], dtype=torch.long, device=values.device
        )
        for offset in range(0, windows.shape[1], self.distance_position_chunk_size):
            chunk = windows[:, offset : offset + self.distance_position_chunk_size, :]
            base = (chunk - self.shapelet).square().sum(dim=-1)
            # Preserve the reviewed upstream ShapeBlock formula exactly:
            # window CI is the squared-difference sum plus epsilon, whereas the
            # learned shapelet CI buffer is sqrt(sum) plus epsilon.
            complexity = (
                chunk.diff(dim=-1).square().sum(dim=-1)
                + 1.0 / self.complexity_norm
            )
            shapelet_complexity = self.shapelet_complexity.to(
                dtype=complexity.dtype, device=complexity.device
            )
            correction = torch.maximum(complexity, shapelet_complexity) / torch.minimum(
                complexity, shapelet_complexity
            ).clamp_min(1e-12)
            correction = correction.clamp_max(self.max_complexity_ratio)
            distance = base * correction / float(self.shapelet_length)
            if window_valid is not None:
                distance = distance.masked_fill(
                    ~window_valid[:, offset : offset + self.distance_position_chunk_size],
                    torch.inf,
                )
            chunk_distance, chunk_position = distance.min(dim=-1)
            replace = chunk_distance < best_distance
            best_distance = torch.where(replace, chunk_distance, best_distance)
            best_position = torch.where(
                replace, chunk_position + offset, best_position
            )
        if not bool(torch.isfinite(best_distance).all()):
            raise ValueError("ShapeBlock failed to select a finite best-fit segment")
        return windows[
            torch.arange(values.shape[0], device=values.device),
            best_position,
        ]

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or self.source_channel >= x.shape[1]:
            raise ValueError("ShapeBlock input/channel mismatch")
        selected = self._best_segments(x[:, self.source_channel, :], valid_mask)
        token = self.selected_projection(selected) - self.shapelet_projection(
            self.shapelet.unsqueeze(0)
        )
        return token.unsqueeze(1)


class LiteratureShapeFormerChannelSpecificOSD(nn.Module):
    """Channel-specific variable-length ShapeFormer downstream reference."""

    model_id = "shapeformer_channel_specific_osd"
    implementation_status = "implemented_not_benchmarked_high_compute"
    literature_parity_scope = (
        "reviewed_shapeblock_token_fusion_with_v2_fold_local_channel_specific_osd"
    )

    def __init__(
        self,
        *,
        n_channels: int,
        n_classes: int,
        sequence_length: int,
        shapelets: PisdShapelets,
        local_kernel_width_samples: int,
        local_embedding_channels: int,
        shape_embedding_channels: int,
        attention_feedforward_channels: int,
        attention_heads: int,
        attention_query_chunk_size: int,
        distance_position_chunk_size: int,
        dropout: float,
        complexity_norm: float,
        max_complexity_ratio: float,
        position_search_neighbourhood_samples: int,
        input_fs_hz: float,
    ) -> None:
        super().__init__()
        if not isinstance(shapelets, PisdShapelets):
            raise TypeError("literature ShapeFormer requires a PisdShapelets bank")
        if shapelets.discovery_method != PISD_DISCOVERY_METHOD:
            raise ValueError("literature ShapeFormer requires channel_specific_osd")
        if len(shapelets.channel_schema) != n_channels:
            raise ValueError("shapelet channel schema differs from model input")
        if not np.isclose(shapelets.input_fs_hz, input_fs_hz, rtol=0.0, atol=1e-12):
            raise ValueError("shapelet bank and model input sampling rates differ")
        if position_search_neighbourhood_samples != shapelets.position_search_neighbourhood_samples:
            raise ValueError("ShapeFormer search neighbourhood differs from its bank")
        if n_channels <= 0 or n_classes != 3 or sequence_length <= 0:
            raise ValueError("canonical ShapeFormer needs positive channels/T and three classes")
        if local_kernel_width_samples <= 1:
            raise ValueError("local_kernel_width_samples must exceed one")
        if local_embedding_channels % attention_heads:
            raise ValueError("local embedding width must be divisible by attention heads")
        if shape_embedding_channels % attention_heads:
            raise ValueError("shape embedding width must be divisible by attention heads")

        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.sequence_length = int(sequence_length)
        self.input_fs_hz = float(input_fs_hz)
        self.local_kernel_width_samples = int(local_kernel_width_samples)
        self.local_embedding_channels = int(local_embedding_channels)
        self.shape_embedding_channels = int(shape_embedding_channels)
        self.attention_feedforward_channels = int(attention_feedforward_channels)
        self.attention_heads = int(attention_heads)
        self.attention_query_chunk_size = int(attention_query_chunk_size)
        self.distance_position_chunk_size = int(distance_position_chunk_size)
        self.dropout_probability = float(dropout)
        self.complexity_norm = float(complexity_norm)
        self.max_complexity_ratio = float(max_complexity_ratio)
        self.position_search_neighbourhood_samples = int(
            position_search_neighbourhood_samples
        )
        self.discovery_method = shapelets.discovery_method
        self.shapelet_count = shapelets.count
        self.shapelets_per_class = int(shapelets.shapelets_per_class)
        self.num_pip_ratio = float(shapelets.num_pip_ratio)
        self.max_discovery_windows = int(shapelets.max_discovery_windows)
        self.discovery_balance = str(shapelets.discovery_balance)
        self.pip_rounding_rule = str(shapelets.pip_rounding_rule)
        self.pip_selection_rule = str(shapelets.pip_selection_rule)
        self.candidate_generation_rule = str(shapelets.candidate_generation_rule)
        self.candidate_enumeration_rule = str(
            shapelets.candidate_enumeration_rule
        )
        self.candidate_ranking_rule = str(shapelets.candidate_ranking_rule)
        self.selected_bank_order_rule = str(shapelets.selected_bank_order_rule)
        self.discovery_position_search_boundary_rule = str(
            shapelets.discovery_position_search_boundary_rule
        )
        self.information_gain_split_rule = str(
            shapelets.information_gain_split_rule
        )
        self.shapelet_lengths = tuple(int(value) for value in shapelets.candidate_lengths)
        self.shapelet_candidate_records = shapelets.candidate_records()
        self.outer_repeat_index = int(shapelets.outer_repeat_index)
        self.outer_fold_index = int(shapelets.outer_fold_index)
        self.outer_train_participant_hash = shapelets.outer_train_participant_hash
        self.fitted_participant_ids = shapelets.fitted_participant_ids

        self.local_temporal = nn.Sequential(
            nn.Conv2d(
                1,
                self.local_embedding_channels,
                kernel_size=(1, self.local_kernel_width_samples),
                padding="same",
            ),
            nn.BatchNorm2d(self.local_embedding_channels),
            nn.GELU(),
        )
        self.local_channel_fusion = nn.Sequential(
            nn.Conv2d(
                self.local_embedding_channels,
                self.local_embedding_channels,
                kernel_size=(self.n_channels, 1),
                padding=0,
            ),
            nn.BatchNorm2d(self.local_embedding_channels),
            nn.GELU(),
        )
        self.local_position = nn.Parameter(
            torch.empty(self.sequence_length, self.local_embedding_channels)
        )
        nn.init.uniform_(self.local_position, -0.02, 0.02)
        self.local_position_dropout = nn.Dropout(self.dropout_probability)
        self.local_attention = _ChunkedShapeFormerAttention(
            self.local_embedding_channels,
            self.attention_heads,
            self.dropout_probability,
            self.attention_query_chunk_size,
        )
        self.local_norm1 = nn.LayerNorm(self.local_embedding_channels, eps=1e-5)
        self.local_norm2 = nn.LayerNorm(self.local_embedding_channels, eps=1e-5)
        self.local_feedforward = nn.Sequential(
            nn.Linear(
                self.local_embedding_channels,
                self.attention_feedforward_channels,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(
                self.attention_feedforward_channels,
                self.local_embedding_channels,
            ),
            nn.Dropout(self.dropout_probability),
        )

        self.shape_blocks = nn.ModuleList(
            [
                ChannelSpecificShapeBlock(
                    source_channel=int(shapelets.source_channels[index]),
                    source_start=int(shapelets.source_starts[index]),
                    source_end=int(shapelets.source_ends[index]),
                    shapelet=shapelets.values[index],
                    shape_embedding_channels=self.shape_embedding_channels,
                    sequence_length=self.sequence_length,
                    position_search_neighbourhood_samples=(
                        self.position_search_neighbourhood_samples
                    ),
                    distance_position_chunk_size=self.distance_position_chunk_size,
                    complexity_norm=self.complexity_norm,
                    max_complexity_ratio=self.max_complexity_ratio,
                )
                for index in range(shapelets.count)
            ]
        )
        source_channels = torch.as_tensor(shapelets.source_channels, dtype=torch.long)
        source_starts = torch.as_tensor(shapelets.source_starts, dtype=torch.long)
        source_ends = torch.as_tensor(shapelets.source_ends, dtype=torch.long)
        self.shape_position_embedding_width_policy = (
            "upstream_observed_max_plus_1_per_axis"
        )
        self.shape_channel_position_width = _upstream_observed_position_width(
            source_channels
        )
        self.shape_start_position_width = _upstream_observed_position_width(
            source_starts
        )
        self.shape_end_position_width = _upstream_observed_position_width(
            source_ends
        )
        self.attention_probability_dropout_applied = False
        self.register_buffer(
            "shape_channel_position",
            torch.nn.functional.one_hot(
                source_channels, num_classes=self.shape_channel_position_width
            ).to(torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "shape_start_position",
            torch.nn.functional.one_hot(
                source_starts, num_classes=self.shape_start_position_width
            ).to(torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "shape_end_position",
            torch.nn.functional.one_hot(
                source_ends, num_classes=self.shape_end_position_width
            ).to(torch.float32),
            persistent=True,
        )
        self.channel_position_projection = nn.Linear(
            self.shape_channel_position_width, self.shape_embedding_channels
        )
        self.start_position_projection = nn.Linear(
            self.shape_start_position_width, self.shape_embedding_channels
        )
        self.end_position_projection = nn.Linear(
            self.shape_end_position_width, self.shape_embedding_channels
        )
        self.shapelet_information_gain_weights = nn.Parameter(
            torch.as_tensor(shapelets.information_gains, dtype=torch.float32),
            requires_grad=True,
        )
        self.shape_attention = _ChunkedShapeFormerAttention(
            self.shape_embedding_channels,
            self.attention_heads,
            self.dropout_probability,
            max(self.shapelet_count, 1),
        )
        self.shape_norm1 = nn.LayerNorm(self.shape_embedding_channels, eps=1e-5)
        self.shape_norm2 = nn.LayerNorm(self.shape_embedding_channels, eps=1e-5)
        self.shape_feedforward = nn.Sequential(
            nn.Linear(
                self.shape_embedding_channels,
                self.attention_feedforward_channels,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(
                self.attention_feedforward_channels,
                self.shape_embedding_channels,
            ),
            nn.Dropout(self.dropout_probability),
        )
        self.feature_dim = (
            self.local_embedding_channels + self.shape_embedding_channels
        )
        self.classifier = nn.Linear(self.feature_dim, self.n_classes)

    def _generic_features(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        local = self.local_temporal(x.unsqueeze(1))
        local = self.local_channel_fusion(local).squeeze(2).transpose(1, 2)
        positioned = self.local_position_dropout(
            local + self.local_position[: local.shape[1]].unsqueeze(0)
        )
        attended = self.local_norm1(
            local + self.local_attention(positioned, valid_mask)
        )
        output = self.local_norm2(attended + self.local_feedforward(attended))
        if valid_mask is None:
            return output.mean(dim=1)
        weights = valid_mask.to(dtype=output.dtype).unsqueeze(-1)
        return (output * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _shape_features(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shape_tokens = torch.cat(
            [block(x, valid_mask) for block in self.shape_blocks],
            dim=1,
        )
        positions = (
            self.channel_position_projection(self.shape_channel_position)
            + self.start_position_projection(self.shape_start_position)
            + self.end_position_projection(self.shape_end_position)
        )
        shape_tokens = shape_tokens + positions.unsqueeze(0)
        attended = shape_tokens + self.shape_attention(shape_tokens)
        weights = self.shapelet_information_gain_weights[None, :, None]
        attended = self.shape_norm1(attended * weights)
        output = self.shape_norm2(
            attended + self.shape_feedforward(attended)
        ) * weights
        # The reviewed upstream architecture uses the first contextualised
        # shape token as its global shape representation.
        return output[:, 0, :]

    def forward_features(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.n_channels, self.sequence_length):
            raise ValueError(
                "literature ShapeFormer expects [batch,n_channels,sequence_length]"
            )
        valid_mask = None
        if mask is not None:
            if mask.ndim != 2 or mask.shape != (x.shape[0], self.sequence_length):
                raise ValueError("mask must match [batch,sequence_length]")
            valid_mask = mask.to(dtype=torch.bool)
            if bool(torch.any(~valid_mask.any(dim=1))):
                raise ValueError("each sample requires at least one valid timestep")
            x = x * valid_mask.to(dtype=x.dtype).unsqueeze(1)
        generic = self._generic_features(x, valid_mask)
        shape = self._shape_features(x, valid_mask)
        return torch.cat((shape, generic), dim=1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.classifier(self.forward_features(x, mask))

    def provenance(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "implementation_status": self.implementation_status,
            "literature_parity_scope": self.literature_parity_scope,
            "discovery_method": self.discovery_method,
            "position_search_neighbourhood_samples": (
                self.position_search_neighbourhood_samples
            ),
            "num_pip_ratio": self.num_pip_ratio,
            "pip_rounding_rule": self.pip_rounding_rule,
            "pip_selection_rule": self.pip_selection_rule,
            "candidate_generation_rule": self.candidate_generation_rule,
            "candidate_enumeration_rule": self.candidate_enumeration_rule,
            "candidate_ranking_rule": self.candidate_ranking_rule,
            "selected_bank_order_rule": self.selected_bank_order_rule,
            "discovery_position_search_boundary_rule": (
                self.discovery_position_search_boundary_rule
            ),
            "information_gain_split_rule": self.information_gain_split_rule,
            "max_discovery_windows": self.max_discovery_windows,
            "discovery_balance": self.discovery_balance,
            "shapelet_length_samples": None,
            "shapelet_lengths": self.shapelet_lengths,
            "shapelet_candidate_records": self.shapelet_candidate_records,
            "shapelet_token_formula": (
                "selected_projection(raw_best_segment)-shapelet_projection(shapelet)"
            ),
            "shapelet_position_embeddings": ("source_channel", "source_start", "source_end"),
            "shape_position_embedding_width_policy": (
                self.shape_position_embedding_width_policy
            ),
            "shape_channel_position_width": self.shape_channel_position_width,
            "shape_start_position_width": self.shape_start_position_width,
            "shape_end_position_width": self.shape_end_position_width,
            "attention_probability_dropout_applied": (
                self.attention_probability_dropout_applied
            ),
            "shapelet_weighting": "learnable_initialised_from_information_gain",
            "global_shape_pooling": "first_contextualised_shape_token",
            "generic_branch_input": "full_multivariate_input",
            "generic_branch_channel_count": self.n_channels,
            "distance_position_chunk_size": self.distance_position_chunk_size,
            "attention_query_chunk_size": self.attention_query_chunk_size,
            "outer_repeat_index": self.outer_repeat_index,
            "outer_fold_index": self.outer_fold_index,
            "outer_train_participant_hash": self.outer_train_participant_hash,
            "fitted_participant_ids": self.fitted_participant_ids,
        }


__all__ = [
    "ChannelSpecificShapeBlock",
    "LiteratureShapeFormerChannelSpecificOSD",
]

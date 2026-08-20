"""Isolated legacy effect-size ShapeFormer port.

This module preserves the executable algorithm used by the historical
``frailty_3class_classifier.py`` effect-size route without replacing either
the faithful channel-specific OSD model or the newer joint-channel
effect-size ablation.  Discovery is explicitly bound to one outer-training
scope by the model factory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as functional
except ImportError as exc:  # pragma: no cover - optional deep dependency
    raise ImportError(
        "LegacyEffectSizeShapeFormer requires the optional deep dependency"
    ) from exc


LEGACY_EFFECT_SIZE_DISCOVERY_METHOD = "legacy_effect_size_channelwise_v1"
LEGACY_DISCOVERY_BALANCE = "legacy_class_window_balanced"


def _participant_roster_hash(participant_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(set(map(str, participant_ids)))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selection_hash(
    participant_ids: Sequence[str],
    file_ids: Sequence[str],
    window_ids: Sequence[str],
) -> str:
    payload = "\n".join(
        f"{participant}\t{file_id}\t{window_id}"
        for participant, file_id, window_id in zip(
            participant_ids, file_ids, window_ids
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LegacyEffectSizeShapelets:
    """Provenance-complete channel-wise fixed-length legacy shapelet bank."""

    values: tuple[np.ndarray, ...]
    source_sample_indices: np.ndarray
    source_starts: np.ndarray
    source_ends: np.ndarray
    source_scores: np.ndarray
    source_weights: np.ndarray
    source_classes: np.ndarray
    source_channels: np.ndarray
    source_participant_ids: tuple[str, ...]
    source_file_ids: tuple[str, ...]
    source_window_ids: tuple[str, ...]
    discovery_indices: np.ndarray
    discovery_participant_ids: tuple[str, ...]
    discovery_file_ids: tuple[str, ...]
    discovery_window_ids: tuple[str, ...]
    discovery_selection_hash: str
    fitted_participant_ids: tuple[str, ...]
    discovery_method: str
    discovery_balance: str
    input_fs_hz: float
    sequence_length_samples: int
    shapelet_length_samples: int
    discovery_stride_samples: int
    shapelets_per_class: int
    max_discovery_windows: int
    candidates_per_class_channel: int
    enumerated_candidate_count: int
    retained_candidate_count: int
    outer_repeat_index: int
    outer_fold_index: int
    outer_train_participant_hash: str

    def __post_init__(self) -> None:
        values = tuple(np.asarray(value, dtype=np.float32) for value in self.values)
        count = len(values)
        if count == 0 or any(
            value.ndim != 1
            or value.shape[0] != int(self.shapelet_length_samples)
            or not np.isfinite(value).all()
            for value in values
        ):
            raise ValueError("legacy shapelets must be finite fixed-length vectors")
        arrays = {
            "source_sample_indices": np.asarray(
                self.source_sample_indices, dtype=np.int64
            ),
            "source_starts": np.asarray(self.source_starts, dtype=np.int64),
            "source_ends": np.asarray(self.source_ends, dtype=np.int64),
            "source_scores": np.asarray(self.source_scores, dtype=np.float64),
            "source_weights": np.asarray(self.source_weights, dtype=np.float64),
            "source_classes": np.asarray(self.source_classes, dtype=np.int64),
            "source_channels": np.asarray(self.source_channels, dtype=np.int64),
        }
        if any(value.shape != (count,) for value in arrays.values()):
            raise ValueError("legacy shapelet metadata must align with the bank")
        if not np.isfinite(arrays["source_scores"]).all() or not np.isfinite(
            arrays["source_weights"]
        ).all():
            raise ValueError("legacy shapelet scores/weights must be finite")
        if np.any(arrays["source_weights"] <= 0.0) or not np.isclose(
            arrays["source_weights"].sum(), count, rtol=0.0, atol=1e-6
        ):
            raise ValueError("legacy shapelet weights must be positive and sum to count")
        if np.any(arrays["source_ends"] - arrays["source_starts"] != int(
            self.shapelet_length_samples
        )):
            raise ValueError("legacy source endpoints disagree with shapelet length")
        if np.any(arrays["source_starts"] < 0) or np.any(
            arrays["source_ends"] > int(self.sequence_length_samples)
        ):
            raise ValueError("legacy source endpoints exceed the input sequence")
        discovery_indices = np.asarray(self.discovery_indices, dtype=np.int64)
        if discovery_indices.ndim != 1 or discovery_indices.size == 0:
            raise ValueError("legacy discovery selection cannot be empty")
        if discovery_indices.size > int(self.max_discovery_windows):
            raise ValueError("legacy discovery selection exceeds its configured cap")
        if not set(arrays["source_sample_indices"].tolist()) <= set(
            discovery_indices.tolist()
        ):
            raise ValueError("legacy shapelet source lies outside discovery selection")
        identities = (
            self.source_participant_ids,
            self.source_file_ids,
            self.source_window_ids,
        )
        if any(len(items) != count for items in identities):
            raise ValueError("one source identity is required per legacy shapelet")
        discovery_identities = (
            self.discovery_participant_ids,
            self.discovery_file_ids,
            self.discovery_window_ids,
        )
        if any(len(items) != discovery_indices.size for items in discovery_identities):
            raise ValueError("legacy discovery identities must align with indices")
        if self.discovery_selection_hash != _selection_hash(*discovery_identities):
            raise ValueError("legacy discovery selection hash drifted")
        if self.discovery_method != LEGACY_EFFECT_SIZE_DISCOVERY_METHOD:
            raise ValueError("legacy discovery method identity mismatch")
        if self.discovery_balance != LEGACY_DISCOVERY_BALANCE:
            raise ValueError("legacy discovery balance identity mismatch")
        positive_controls = {
            "sequence_length_samples": self.sequence_length_samples,
            "shapelet_length_samples": self.shapelet_length_samples,
            "discovery_stride_samples": self.discovery_stride_samples,
            "shapelets_per_class": self.shapelets_per_class,
            "max_discovery_windows": self.max_discovery_windows,
            "candidates_per_class_channel": self.candidates_per_class_channel,
            "enumerated_candidate_count": self.enumerated_candidate_count,
            "retained_candidate_count": self.retained_candidate_count,
        }
        if any(int(value) <= 0 for value in positive_controls.values()):
            raise ValueError("legacy discovery numeric controls must be positive")
        if not np.isfinite(self.input_fs_hz) or float(self.input_fs_hz) <= 0.0:
            raise ValueError("legacy input_fs_hz must be finite and positive")
        classes, class_counts = np.unique(
            arrays["source_classes"], return_counts=True
        )
        if classes.size < 2 or np.any(class_counts != int(self.shapelets_per_class)):
            raise ValueError("legacy bank must preserve shapelets_per_class")
        fitted = tuple(sorted(set(map(str, self.fitted_participant_ids))))
        if not fitted or self.outer_train_participant_hash != _participant_roster_hash(
            fitted
        ):
            raise ValueError("legacy outer-train participant hash drifted")
        if self.outer_repeat_index < 0 or self.outer_fold_index < 0:
            raise ValueError("legacy outer repeat/fold indices must be non-negative")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "discovery_indices", discovery_indices)
        object.__setattr__(self, "fitted_participant_ids", fitted)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def info(self) -> np.ndarray:
        """Return the exact six-column metadata consumed by the legacy model."""

        return np.column_stack(
            (
                self.source_sample_indices,
                self.source_starts,
                self.source_ends,
                self.source_weights,
                self.source_classes,
                self.source_channels,
            )
        ).astype(np.float32)

    def candidate_records(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "source_sample_index": int(self.source_sample_indices[index]),
                "source_start": int(self.source_starts[index]),
                "source_end": int(self.source_ends[index]),
                "source_score": float(self.source_scores[index]),
                "initial_weight": float(self.source_weights[index]),
                "source_class": int(self.source_classes[index]),
                "source_channel": int(self.source_channels[index]),
                "source_participant_id": self.source_participant_ids[index],
                "source_file_id": self.source_file_ids[index],
                "source_window_id": self.source_window_ids[index],
            }
            for index in range(self.count)
        )


def _balanced_indices(y: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """Historical class-window-balanced capped selection."""

    if len(y) <= maximum:
        return np.arange(len(y), dtype=np.int64)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, maximum // len(classes))
    chosen: list[int] = []
    for class_value in classes:
        indices = np.flatnonzero(y == class_value)
        take = min(per_class, indices.size)
        chosen.extend(
            rng.choice(indices, size=take, replace=False).astype(int).tolist()
        )
    if len(chosen) < maximum:
        remaining = np.setdiff1d(
            np.arange(len(y)), np.asarray(chosen, dtype=np.int64)
        )
        if remaining.size:
            fill = min(maximum - len(chosen), remaining.size)
            chosen.extend(
                rng.choice(remaining, size=fill, replace=False).astype(int).tolist()
            )
    return np.asarray(sorted(set(chosen)), dtype=np.int64)


def _overlaps_existing(
    start: int,
    end: int,
    occupied: Sequence[tuple[int, int]],
    *,
    maximum_overlap: float = 0.50,
) -> bool:
    length = max(1, end - start)
    return any(
        max(0, min(end, old_end) - max(start, old_start)) / length
        > maximum_overlap
        for old_start, old_end in occupied
    )


def discover_legacy_effect_size_shapelets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    participant_ids: Sequence[str],
    file_ids: Sequence[str],
    window_ids: Sequence[str],
    *,
    discovery_method: str,
    input_fs_hz: float,
    sequence_length_samples: int,
    shapelet_length_samples: int,
    discovery_stride_samples: int,
    shapelets_per_class: int,
    max_discovery_windows: int,
    candidates_per_class_channel: int,
    outer_repeat_index: int,
    outer_fold_index: int,
    seed: int,
) -> LegacyEffectSizeShapelets:
    """Run the historical channel-wise effect-map discovery on outer-train only."""

    if discovery_method != LEGACY_EFFECT_SIZE_DISCOVERY_METHOD:
        raise ValueError("legacy effect-size discovery never falls back")
    x = np.asarray(x_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.int64)
    participants = tuple(map(str, participant_ids))
    files = tuple(map(str, file_ids))
    windows = tuple(map(str, window_ids))
    if x.ndim != 3 or y.shape != (x.shape[0],):
        raise ValueError("legacy discovery expects x=[window,channel,time] and y=[window]")
    if x.shape[-1] != int(sequence_length_samples):
        raise ValueError("legacy discovery sequence length differs from its config")
    if not np.isfinite(x).all():
        raise ValueError("legacy discovery requires finite input")
    if any(len(items) != x.shape[0] for items in (participants, files, windows)):
        raise ValueError("legacy discovery requires one identity per window")
    controls = {
        "shapelet_length_samples": shapelet_length_samples,
        "discovery_stride_samples": discovery_stride_samples,
        "shapelets_per_class": shapelets_per_class,
        "max_discovery_windows": max_discovery_windows,
        "candidates_per_class_channel": candidates_per_class_channel,
    }
    if any(
        isinstance(value, (bool, np.bool_)) or int(value) <= 0
        for value in controls.values()
    ):
        raise ValueError("legacy discovery controls must be positive integers")
    if int(shapelet_length_samples) > x.shape[-1]:
        raise ValueError("legacy shapelet length exceeds the input sequence")
    if int(shapelet_length_samples) < 4:
        raise ValueError(
            "legacy shapelet length must be at least four effective samples"
        )
    if not np.isfinite(input_fs_hz) or float(input_fs_hz) <= 0.0:
        raise ValueError("legacy input_fs_hz must be finite and positive")
    classes = np.unique(y)
    if classes.size < 2 or int(max_discovery_windows) < classes.size:
        raise ValueError("legacy discovery needs every class in its capped selection")

    discovery_indices = _balanced_indices(y, int(max_discovery_windows), int(seed))
    discovery_x = x[discovery_indices]
    discovery_y = y[discovery_indices]
    rng = np.random.default_rng(seed)
    selected: list[tuple[int, int, int, float, int, int, np.ndarray]] = []
    enumerated_candidate_count = 0
    retained_candidate_count = 0
    length = int(shapelet_length_samples)
    stride = int(discovery_stride_samples)

    for class_value in sorted(np.unique(discovery_y).tolist()):
        class_indices = np.flatnonzero(discovery_y == class_value)
        other_indices = np.flatnonzero(discovery_y != class_value)
        if class_indices.size == 0 or other_indices.size == 0:
            raise ValueError("legacy discovery cap removed a required class")
        class_mean = np.mean(discovery_x[class_indices], axis=0)
        other_mean = np.mean(discovery_x[other_indices], axis=0)
        pooled_std = np.std(discovery_x, axis=0) + 1e-4
        effect = np.abs(class_mean - other_mean) / pooled_std
        candidates: list[tuple[float, int, int, int]] = []
        for channel in range(discovery_x.shape[1]):
            channel_candidates: list[tuple[float, int, int, int]] = []
            for start in range(0, discovery_x.shape[-1] - length + 1, stride):
                end = start + length
                channel_candidates.append(
                    (float(np.mean(effect[channel, start:end])), channel, start, end)
                )
                enumerated_candidate_count += 1
            channel_candidates.sort(key=lambda item: item[0], reverse=True)
            retained = channel_candidates[: int(candidates_per_class_channel)]
            retained_candidate_count += len(retained)
            candidates.extend(retained)
        candidates.sort(key=lambda item: item[0], reverse=True)
        occupied: dict[int, list[tuple[int, int]]] = {
            channel: [] for channel in range(discovery_x.shape[1])
        }
        selected_for_class = 0
        for score, channel, start, end in candidates:
            if selected_for_class >= int(shapelets_per_class):
                break
            if _overlaps_existing(start, end, occupied[channel]):
                continue
            target = class_mean[channel, start:end]
            segments = discovery_x[class_indices, channel, start:end]
            distances = np.mean(np.square(segments - target[None, :]), axis=1)
            local_source = int(class_indices[int(np.argmin(distances))])
            source_index = int(discovery_indices[local_source])
            selected.append(
                (
                    source_index,
                    start,
                    end,
                    score,
                    int(class_value),
                    channel,
                    x[source_index, channel, start:end].copy(),
                )
            )
            occupied[channel].append((start, end))
            selected_for_class += 1
        while selected_for_class < int(shapelets_per_class):
            local_source = int(rng.choice(class_indices))
            channel = int(rng.integers(0, discovery_x.shape[1]))
            start = int(
                rng.integers(0, discovery_x.shape[-1] - length + 1)
            )
            end = start + length
            source_index = int(discovery_indices[local_source])
            selected.append(
                (
                    source_index,
                    start,
                    end,
                    1e-3,
                    int(class_value),
                    channel,
                    x[source_index, channel, start:end].copy(),
                )
            )
            selected_for_class += 1

    if not selected:
        raise ValueError("legacy discovery produced no shapelets")
    scores = np.asarray([row[3] for row in selected], dtype=np.float64)
    shifted = scores - scores.max()
    weights = np.exp(shifted)
    weights = weights / weights.sum() * len(selected)
    source_indices = np.asarray([row[0] for row in selected], dtype=np.int64)
    discovery_participants = tuple(participants[index] for index in discovery_indices)
    discovery_files = tuple(files[index] for index in discovery_indices)
    discovery_windows = tuple(windows[index] for index in discovery_indices)
    roster = tuple(sorted(set(participants)))
    return LegacyEffectSizeShapelets(
        values=tuple(row[6] for row in selected),
        source_sample_indices=source_indices,
        source_starts=np.asarray([row[1] for row in selected]),
        source_ends=np.asarray([row[2] for row in selected]),
        source_scores=scores,
        source_weights=weights,
        source_classes=np.asarray([row[4] for row in selected]),
        source_channels=np.asarray([row[5] for row in selected]),
        source_participant_ids=tuple(participants[index] for index in source_indices),
        source_file_ids=tuple(files[index] for index in source_indices),
        source_window_ids=tuple(windows[index] for index in source_indices),
        discovery_indices=discovery_indices,
        discovery_participant_ids=discovery_participants,
        discovery_file_ids=discovery_files,
        discovery_window_ids=discovery_windows,
        discovery_selection_hash=_selection_hash(
            discovery_participants, discovery_files, discovery_windows
        ),
        fitted_participant_ids=roster,
        discovery_method=discovery_method,
        discovery_balance=LEGACY_DISCOVERY_BALANCE,
        input_fs_hz=float(input_fs_hz),
        sequence_length_samples=int(sequence_length_samples),
        shapelet_length_samples=length,
        discovery_stride_samples=stride,
        shapelets_per_class=int(shapelets_per_class),
        max_discovery_windows=int(max_discovery_windows),
        candidates_per_class_channel=int(candidates_per_class_channel),
        enumerated_candidate_count=enumerated_candidate_count,
        retained_candidate_count=retained_candidate_count,
        outer_repeat_index=int(outer_repeat_index),
        outer_fold_index=int(outer_fold_index),
        outer_train_participant_hash=_participant_roster_hash(roster),
    )


class LegacyShapeFormerAttention(nn.Module):
    """Attention equation used by the historical local and shape branches."""

    def __init__(self, embedding_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if embedding_dim <= 0 or heads <= 0 or embedding_dim % heads:
            raise ValueError("legacy attention embedding must be divisible by heads")
        self.heads = int(heads)
        self.scale = float(embedding_dim) ** -0.5
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = values.shape
        keys = self.key(values).reshape(batch, tokens, self.heads, -1).permute(
            0, 2, 3, 1
        )
        projected_values = self.value(values).reshape(
            batch, tokens, self.heads, -1
        ).transpose(1, 2)
        queries = self.query(values).reshape(
            batch, tokens, self.heads, -1
        ).transpose(1, 2)
        attention = functional.softmax(
            torch.matmul(queries, keys) * self.scale, dim=-1
        )
        output = torch.matmul(self.dropout(attention), projected_values)
        output = output.transpose(1, 2).reshape(batch, tokens, -1)
        return self.output_norm(output)


class LegacyLearnablePosition(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float, maximum: int) -> None:
        super().__init__()
        self.values = nn.Parameter(torch.empty(maximum, embedding_dim))
        nn.init.uniform_(self.values, -0.02, 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.values[: values.shape[1]].unsqueeze(0))


def _one_hot_positions(values: np.ndarray) -> torch.Tensor:
    positions = torch.as_tensor(values, dtype=torch.long)
    width = int(positions.max().item()) + 1 if positions.numel() else 1
    return torch.eye(width, dtype=torch.float32)[positions]


class LegacyShapeBlock(nn.Module):
    """Historical complexity-adjusted source-channel best-segment token."""

    def __init__(
        self,
        *,
        source_channel: int,
        source_start: int,
        source_end: int,
        shapelet: np.ndarray,
        shape_embedding_channels: int,
        sequence_length_samples: int,
        search_window_samples: int,
        complexity_norm: float,
        max_complexity_ratio: float,
    ) -> None:
        super().__init__()
        array = np.asarray(shapelet, dtype=np.float32).reshape(-1)
        if array.size < 2:
            raise ValueError("legacy ShapeBlock requires at least two samples")
        self.source_channel = int(source_channel)
        self.kernel_size = int(array.size)
        self.search_start = max(0, int(source_start) - int(search_window_samples))
        self.search_end = min(
            int(sequence_length_samples),
            int(source_end) + int(search_window_samples),
        )
        self.complexity_norm = float(complexity_norm)
        self.max_complexity_ratio = float(max_complexity_ratio)
        self.shapelet = nn.Parameter(torch.from_numpy(array.copy()))
        self.selected_projection = nn.Linear(
            self.kernel_size, shape_embedding_channels
        )
        self.shapelet_projection = nn.Linear(
            self.kernel_size, shape_embedding_channels
        )
        complexity = np.sqrt(np.sum(np.square(np.diff(array)))) + (
            1.0 / self.complexity_norm
        )
        self.register_buffer(
            "shapelet_complexity", torch.tensor(complexity, dtype=torch.float32)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        series = values[
            :, self.source_channel, self.search_start : self.search_end
        ]
        if series.shape[1] < self.kernel_size:
            raise ValueError("legacy ShapeBlock search interval is too short")
        windows = series.unfold(1, self.kernel_size, 1).contiguous()
        flat = windows.reshape(-1, self.kernel_size)
        differences = torch.square(series[:, 1:] - series[:, :-1])
        complexities = differences.unfold(
            1, self.kernel_size - 1, 1
        ).contiguous().reshape(-1, self.kernel_size - 1).sum(dim=1)
        complexities = complexities + (1.0 / self.complexity_norm)
        shapelet_complexity = self.shapelet_complexity.to(values.device)
        ratios = torch.maximum(complexities, shapelet_complexity) / torch.minimum(
            complexities, shapelet_complexity
        )
        ratios = torch.where(torch.isfinite(ratios), ratios, torch.ones_like(ratios))
        ratios = torch.clamp(ratios, max=self.max_complexity_ratio)
        distances = torch.square(flat - self.shapelet).sum(dim=1)
        distances = (distances * ratios) / self.kernel_size
        distances = torch.where(
            torch.isfinite(distances), distances, torch.ones_like(distances)
        ).reshape(values.shape[0], -1)
        best = torch.argmin(distances, dim=1)
        selected = windows[
            torch.arange(values.shape[0], device=values.device), best
        ]
        return (
            self.selected_projection(selected)
            - self.shapelet_projection(self.shapelet.unsqueeze(0))
        ).unsqueeze(1)


class LegacyEffectSizeShapeFormer(nn.Module):
    """Functional local+shape-token downstream from the historical classifier."""

    model_id = "shapeformer_legacy_effect_size_port"
    registry_role = "ablation"

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        shapelets: LegacyEffectSizeShapelets,
        *,
        sequence_length_samples: int,
        local_kernel_width_samples: int = 8,
        local_embedding_channels: int = 48,
        shape_embedding_channels: int = 128,
        attention_feedforward_channels: int = 256,
        attention_heads: int = 4,
        dropout: float = 0.30,
        shapelet_search_window_samples: int = 64,
        complexity_norm: float = 1000.0,
        max_complexity_ratio: float = 3.0,
        input_fs_hz: float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(shapelets, LegacyEffectSizeShapelets):
            raise TypeError("legacy ShapeFormer requires its legacy discovery bank")
        if input_fs_hz is None or not np.isclose(
            input_fs_hz, shapelets.input_fs_hz, rtol=0.0, atol=1e-12
        ):
            raise ValueError("legacy model and shapelet sampling rates differ")
        if int(sequence_length_samples) != int(shapelets.sequence_length_samples):
            raise ValueError("legacy model and discovery sequence lengths differ")
        positive = {
            "n_channels": n_channels,
            "local_kernel_width_samples": local_kernel_width_samples,
            "local_embedding_channels": local_embedding_channels,
            "shape_embedding_channels": shape_embedding_channels,
            "attention_feedforward_channels": attention_feedforward_channels,
            "attention_heads": attention_heads,
            "shapelet_search_window_samples": shapelet_search_window_samples,
        }
        if n_classes <= 1 or any(int(value) <= 0 for value in positive.values()):
            raise ValueError("legacy model dimensions must be positive")
        if int(local_kernel_width_samples) > int(sequence_length_samples):
            raise ValueError("legacy local kernel exceeds the sequence")
        if int(local_embedding_channels) % int(attention_heads) or int(
            shape_embedding_channels
        ) % int(attention_heads):
            raise ValueError("legacy embedding widths must be divisible by heads")
        if not np.isfinite(dropout) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("legacy dropout must be finite in [0,1)")
        if (
            not np.isfinite(complexity_norm)
            or float(complexity_norm) <= 0.0
            or not np.isfinite(max_complexity_ratio)
            or float(max_complexity_ratio) < 1.0
        ):
            raise ValueError("legacy complexity controls are invalid")
        if np.any(shapelets.source_channels >= int(n_channels)):
            raise ValueError("legacy shapelet source channel exceeds model input")
        if set(shapelets.source_classes.tolist()) != set(range(int(n_classes))):
            raise ValueError(
                "legacy shapelet classes must match the classifier label space"
            )

        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.sequence_length_samples = int(sequence_length_samples)
        self.input_fs_hz = float(input_fs_hz)
        self.local_kernel_width_samples = int(local_kernel_width_samples)
        self.local_embedding_channels = int(local_embedding_channels)
        self.shape_embedding_channels = int(shape_embedding_channels)
        self.attention_feedforward_channels = int(
            attention_feedforward_channels
        )
        self.attention_heads = int(attention_heads)
        self.dropout_probability = float(dropout)
        self.shapelet_search_window_samples = int(
            shapelet_search_window_samples
        )
        self.complexity_norm = float(complexity_norm)
        self.max_complexity_ratio = float(max_complexity_ratio)
        self.discovery_method = shapelets.discovery_method
        self.discovery_balance = shapelets.discovery_balance
        self.shapelet_length_samples = int(shapelets.shapelet_length_samples)
        self.shapelet_count = int(shapelets.count)
        self.discovery_stride_samples = int(shapelets.discovery_stride_samples)
        self.shapelets_per_class = int(shapelets.shapelets_per_class)
        self.max_discovery_windows = int(shapelets.max_discovery_windows)
        self.candidates_per_class_channel = int(
            shapelets.candidates_per_class_channel
        )
        self.discovery_window_count = int(shapelets.discovery_indices.size)
        self.enumerated_candidate_count = int(
            shapelets.enumerated_candidate_count
        )
        self.retained_candidate_count = int(shapelets.retained_candidate_count)
        self.outer_repeat_index = int(shapelets.outer_repeat_index)
        self.outer_fold_index = int(shapelets.outer_fold_index)
        self.outer_train_participant_hash = shapelets.outer_train_participant_hash
        self.fitted_participant_ids = shapelets.fitted_participant_ids
        self.shapelet_candidate_records = shapelets.candidate_records()

        self.local_temporal = nn.Sequential(
            nn.Conv2d(
                1,
                self.local_embedding_channels,
                kernel_size=(1, self.local_kernel_width_samples),
                padding="same",
            ),
            nn.BatchNorm2d(self.local_embedding_channels),
            nn.GELU(),
            nn.Conv2d(
                self.local_embedding_channels,
                self.local_embedding_channels,
                kernel_size=(self.n_channels, 1),
                padding="valid",
            ),
            nn.BatchNorm2d(self.local_embedding_channels),
            nn.GELU(),
        )
        self.local_position = LegacyLearnablePosition(
            self.local_embedding_channels,
            self.dropout_probability,
            self.sequence_length_samples,
        )
        self.local_attention = LegacyShapeFormerAttention(
            self.local_embedding_channels,
            self.attention_heads,
            self.dropout_probability,
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
                LegacyShapeBlock(
                    source_channel=int(shapelets.source_channels[index]),
                    source_start=int(shapelets.source_starts[index]),
                    source_end=int(shapelets.source_ends[index]),
                    shapelet=shapelets.values[index],
                    shape_embedding_channels=self.shape_embedding_channels,
                    sequence_length_samples=self.sequence_length_samples,
                    search_window_samples=self.shapelet_search_window_samples,
                    complexity_norm=self.complexity_norm,
                    max_complexity_ratio=self.max_complexity_ratio,
                )
                for index in range(shapelets.count)
            ]
        )
        self.shapelet_weights = nn.Parameter(
            torch.as_tensor(shapelets.source_weights, dtype=torch.float32)
        )
        channel_positions = _one_hot_positions(shapelets.source_channels)
        start_positions = _one_hot_positions(shapelets.source_starts)
        end_positions = _one_hot_positions(shapelets.source_ends)
        self.register_buffer("channel_positions", channel_positions)
        self.register_buffer("start_positions", start_positions)
        self.register_buffer("end_positions", end_positions)
        self.channel_projection = nn.Linear(
            channel_positions.shape[1], self.shape_embedding_channels
        )
        self.start_projection = nn.Linear(
            start_positions.shape[1], self.shape_embedding_channels
        )
        self.end_projection = nn.Linear(
            end_positions.shape[1], self.shape_embedding_channels
        )
        self.shape_attention = LegacyShapeFormerAttention(
            self.shape_embedding_channels,
            self.attention_heads,
            self.dropout_probability,
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
            self.shape_embedding_channels + self.local_embedding_channels
        )
        self.classifier = nn.Linear(self.feature_dim, self.n_classes)

    def forward_features(
        self,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (
            self.n_channels,
            self.sequence_length_samples,
        ):
            raise ValueError("legacy ShapeFormer input shape mismatch")
        if mask is not None:
            if mask.shape != (values.shape[0], self.sequence_length_samples):
                raise ValueError("legacy ShapeFormer mask shape mismatch")
            if not bool(mask.to(dtype=torch.bool).all()):
                raise ValueError(
                    "legacy ShapeFormer requires complete, unpadded windows"
                )
        local = self.local_temporal(values.unsqueeze(1)).squeeze(2).permute(0, 2, 1)
        positioned = self.local_position(local)
        local = self.local_norm1(local + self.local_attention(positioned))
        local = self.local_norm2(local + self.local_feedforward(local))
        local = local.mean(dim=1)

        shape = torch.cat(
            [block(values) for block in self.shape_blocks], dim=1
        )
        positions = (
            self.channel_projection(self.channel_positions)
            + self.start_projection(self.start_positions)
            + self.end_projection(self.end_positions)
        )
        shape = shape + positions.unsqueeze(0)
        shape = shape + self.shape_attention(shape)
        weights = self.shapelet_weights[None, :, None]
        shape = self.shape_norm1(shape * weights)
        shape = self.shape_norm2(shape + self.shape_feedforward(shape)) * weights
        shape = shape[:, 0, :]
        return torch.cat((shape, local), dim=1)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.classifier(self.forward_features(values, mask))

    def provenance(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "registry_role": self.registry_role,
            "discovery_method": self.discovery_method,
            "discovery_balance": self.discovery_balance,
            "input_fs_hz": self.input_fs_hz,
            "sequence_length_samples": self.sequence_length_samples,
            "shapelet_length_samples": self.shapelet_length_samples,
            "discovery_stride_samples": self.discovery_stride_samples,
            "shapelets_per_class": self.shapelets_per_class,
            "shapelet_count": self.shapelet_count,
            "max_discovery_windows": self.max_discovery_windows,
            "candidates_per_class_channel": self.candidates_per_class_channel,
            "discovery_window_count": self.discovery_window_count,
            "enumerated_candidate_count": self.enumerated_candidate_count,
            "retained_candidate_count": self.retained_candidate_count,
            "local_kernel_width_samples": self.local_kernel_width_samples,
            "local_embedding_channels": self.local_embedding_channels,
            "shape_embedding_channels": self.shape_embedding_channels,
            "attention_feedforward_channels": self.attention_feedforward_channels,
            "attention_heads": self.attention_heads,
            "dropout": self.dropout_probability,
            "shapelet_search_window_samples": self.shapelet_search_window_samples,
            "complexity_norm": self.complexity_norm,
            "max_complexity_ratio": self.max_complexity_ratio,
            "outer_repeat_index": self.outer_repeat_index,
            "outer_fold_index": self.outer_fold_index,
            "outer_train_participant_hash": self.outer_train_participant_hash,
            "fitted_participant_ids": self.fitted_participant_ids,
            "shapelet_candidate_records": self.shapelet_candidate_records,
        }


__all__ = [
    "LEGACY_DISCOVERY_BALANCE",
    "LEGACY_EFFECT_SIZE_DISCOVERY_METHOD",
    "LegacyEffectSizeShapeFormer",
    "LegacyEffectSizeShapelets",
    "discover_legacy_effect_size_shapelets",
]

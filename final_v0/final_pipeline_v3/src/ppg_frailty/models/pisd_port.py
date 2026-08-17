"""Fold-local, channel-specific OSD/PISD discovery for ShapeFormer V2.

The literature-reference route deliberately has no fixed shapelet length and no
candidate stride. Every candidate is the interval bounded by three consecutive
perceptually-important points on one source channel. Failure is explicit; this
module never calls the fixed-length effect-size implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Callable
from typing import Iterable

import numpy as np


PISD_DISCOVERY_METHOD = "channel_specific_osd"
NUM_PIP_RATIO = 0.20
SHAPELETS_PER_CLASS = 3
MAX_DISCOVERY_WINDOWS = 180
DISCOVERY_BALANCE = "participant_file_balanced"
POSITION_SEARCH_NEIGHBOURHOOD_SAMPLES = 128
PIP_ROUNDING_RULE = "floor_ratio_minimum_5_capped_at_actual_T"
PIP_SELECTION_RULE = "upstream_zscored_time_index_perpendicular_distance_first_max"
CANDIDATE_GENERATION_RULE = "insertion_stage_three_consecutive_pips_half_open"
CANDIDATE_ENUMERATION_RULE = "upstream_class_channel_source_sample_insertion_order"
CANDIDATE_RANKING_RULE = "upstream_numpy_default_argsort_then_reverse"
SELECTED_BANK_ORDER_RULE = "upstream_per_class_start_sample_default_argsort"
DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE = (
    "upstream_pcs_start_minus_w_plus_1_end_plus_w_half_open"
)
INFORMATION_GAIN_SPLIT_RULE = "upstream_positive_recall_grid_0p2"


def _participant_roster_hash(participant_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(str(value) for value in participant_ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PisdShapelets:
    """Variable-length, channel-specific candidates bound to one outer fold.

    ``source_ends`` use the Python half-open convention. Consequently,
    ``candidate_lengths == source_ends - source_starts``.
    """

    values: tuple[np.ndarray, ...]
    source_classes: np.ndarray
    information_gains: np.ndarray
    source_sample_indices: np.ndarray
    source_channels: np.ndarray
    source_starts: np.ndarray
    source_ends: np.ndarray
    source_start_seconds: np.ndarray
    source_end_seconds: np.ndarray
    candidate_lengths: np.ndarray
    source_channel_names: tuple[str, ...]
    source_participant_ids: tuple[str, ...]
    source_file_ids: tuple[str, ...]
    source_window_ids: tuple[str, ...]
    discovery_sequence_lengths: np.ndarray
    pip_counts: np.ndarray
    fitted_participant_ids: tuple[str, ...]
    discovery_participant_ids: tuple[str, ...]
    discovery_file_ids: tuple[str, ...]
    discovery_window_ids: tuple[str, ...]
    discovery_selection_hash: str
    channel_schema: tuple[str, ...]
    discovery_method: str
    discovery_balance: str
    input_fs_hz: float
    num_pip_ratio: float
    shapelets_per_class: int
    max_discovery_windows: int
    discovery_window_count: int
    position_search_neighbourhood_samples: int
    pip_rounding_rule: str
    pip_selection_rule: str
    candidate_generation_rule: str
    candidate_enumeration_rule: str
    candidate_ranking_rule: str
    selected_bank_order_rule: str
    discovery_position_search_boundary_rule: str
    information_gain_split_rule: str
    outer_repeat_index: int
    outer_fold_index: int
    outer_train_participant_hash: str

    def __post_init__(self) -> None:
        values = tuple(np.asarray(value, dtype=np.float32) for value in self.values)
        count = len(values)
        if count == 0 or any(value.ndim != 1 or value.size < 2 for value in values):
            raise ValueError("OSD/PISD values must be non-empty one-dimensional candidates")
        if any(not np.isfinite(value).all() for value in values):
            raise ValueError("OSD/PISD candidates contain non-finite values")
        integer_arrays = {
            "source_classes": np.asarray(self.source_classes, dtype=np.int64),
            "source_sample_indices": np.asarray(self.source_sample_indices, dtype=np.int64),
            "source_channels": np.asarray(self.source_channels, dtype=np.int64),
            "source_starts": np.asarray(self.source_starts, dtype=np.int64),
            "source_ends": np.asarray(self.source_ends, dtype=np.int64),
            "candidate_lengths": np.asarray(self.candidate_lengths, dtype=np.int64),
            "discovery_sequence_lengths": np.asarray(self.discovery_sequence_lengths, dtype=np.int64),
            "pip_counts": np.asarray(self.pip_counts, dtype=np.int64),
        }
        scores = np.asarray(self.information_gains, dtype=np.float64)
        start_seconds = np.asarray(self.source_start_seconds, dtype=np.float64)
        end_seconds = np.asarray(self.source_end_seconds, dtype=np.float64)
        for name, array in (
            *integer_arrays.items(),
            ("information_gains", scores),
            ("source_start_seconds", start_seconds),
            ("source_end_seconds", end_seconds),
        ):
            if array.shape != (count,):
                raise ValueError(f"{name} must contain one value per shapelet")
        source_text = (
            self.source_channel_names,
            self.source_participant_ids,
            self.source_file_ids,
            self.source_window_ids,
        )
        if any(len(values) != count or any(not str(value).strip() for value in values) for values in source_text):
            raise ValueError("source channel/participant/file/window provenance must match shapelet count")
        if (
            not np.isfinite(scores).all()
            or np.any(scores < -1.0)
            or np.any(scores > 1.0)
        ):
            raise ValueError(
                "information gains must be finite in [-1,1]; -1 is the exact "
                "upstream no-grid-point sentinel"
            )
        if not np.isfinite(start_seconds).all() or not np.isfinite(end_seconds).all():
            raise ValueError("shapelet time endpoints must be finite")
        if any(np.any(array < 0) for array in integer_arrays.values()):
            raise ValueError("shapelet index metadata must be non-negative")
        lengths = integer_arrays["candidate_lengths"]
        starts = integer_arrays["source_starts"]
        ends = integer_arrays["source_ends"]
        sequence_lengths = integer_arrays["discovery_sequence_lengths"]
        if not np.array_equal(lengths, ends - starts):
            raise ValueError("candidate length must equal end_sample_exclusive - start_sample")
        if not np.array_equal(lengths, np.asarray([value.size for value in values])):
            raise ValueError("candidate length metadata differs from stored values")
        if np.any(ends > sequence_lengths):
            raise ValueError("shapelet endpoint exceeds its discovery sequence")
        if not np.allclose(start_seconds, starts / float(self.input_fs_hz), rtol=0.0, atol=1e-12):
            raise ValueError("start seconds must equal start samples / input_fs_hz")
        if not np.allclose(end_seconds, ends / float(self.input_fs_hz), rtol=0.0, atol=1e-12):
            raise ValueError("end seconds must equal end samples / input_fs_hz")
        if self.discovery_method != PISD_DISCOVERY_METHOD:
            raise ValueError(f"PisdShapelets requires discovery_method={PISD_DISCOVERY_METHOD}")
        if self.discovery_balance != DISCOVERY_BALANCE:
            raise ValueError(f"PISD discovery_balance must be {DISCOVERY_BALANCE}")
        if not np.isfinite(self.input_fs_hz) or self.input_fs_hz <= 0.0:
            raise ValueError("input_fs_hz must be finite and positive")
        if not np.isclose(self.num_pip_ratio, NUM_PIP_RATIO, rtol=0.0, atol=1e-12):
            raise ValueError(f"num_pip_ratio is frozen at {NUM_PIP_RATIO}")
        if self.shapelets_per_class != SHAPELETS_PER_CLASS:
            raise ValueError(f"shapelets_per_class is frozen at {SHAPELETS_PER_CLASS}")
        if self.max_discovery_windows != MAX_DISCOVERY_WINDOWS:
            raise ValueError(f"max_discovery_windows is frozen at {MAX_DISCOVERY_WINDOWS}")
        if self.position_search_neighbourhood_samples != POSITION_SEARCH_NEIGHBOURHOOD_SAMPLES:
            raise ValueError(
                "position_search_neighbourhood_samples is frozen at "
                f"{POSITION_SEARCH_NEIGHBOURHOOD_SAMPLES}; it is never a shapelet length"
            )
        if self.pip_rounding_rule != PIP_ROUNDING_RULE:
            raise ValueError(f"pip_rounding_rule must be {PIP_ROUNDING_RULE}")
        if self.pip_selection_rule != PIP_SELECTION_RULE:
            raise ValueError(f"pip_selection_rule must be {PIP_SELECTION_RULE}")
        if self.candidate_generation_rule != CANDIDATE_GENERATION_RULE:
            raise ValueError(
                f"candidate_generation_rule must be {CANDIDATE_GENERATION_RULE}"
            )
        if self.candidate_enumeration_rule != CANDIDATE_ENUMERATION_RULE:
            raise ValueError(
                f"candidate_enumeration_rule must be {CANDIDATE_ENUMERATION_RULE}"
            )
        if self.candidate_ranking_rule != CANDIDATE_RANKING_RULE:
            raise ValueError(
                f"candidate_ranking_rule must be {CANDIDATE_RANKING_RULE}"
            )
        if self.selected_bank_order_rule != SELECTED_BANK_ORDER_RULE:
            raise ValueError(
                f"selected_bank_order_rule must be {SELECTED_BANK_ORDER_RULE}"
            )
        if (
            self.discovery_position_search_boundary_rule
            != DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE
        ):
            raise ValueError(
                "discovery_position_search_boundary_rule must be "
                f"{DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE}"
            )
        if self.information_gain_split_rule != INFORMATION_GAIN_SPLIT_RULE:
            raise ValueError(
                "information_gain_split_rule must be "
                f"{INFORMATION_GAIN_SPLIT_RULE}"
            )
        if not 0 < self.discovery_window_count <= self.max_discovery_windows:
            raise ValueError("discovery_window_count is outside the frozen capacity")
        source_class_values, class_counts = np.unique(
            integer_arrays["source_classes"], return_counts=True
        )
        if (
            tuple(source_class_values.tolist()) != (0, 1, 2)
            or np.any(class_counts != self.shapelets_per_class)
            or count != 9
        ):
            raise ValueError(
                "Frailty3 OSD/PISD bank requires source_classes {0,1,2} "
                "and exactly three shapelets per class"
            )
        if self.outer_repeat_index < 0 or self.outer_fold_index < 0:
            raise ValueError("outer repeat/fold indices must be non-negative")
        roster = tuple(sorted(set(str(value) for value in self.fitted_participant_ids)))
        if not roster or self.outer_train_participant_hash != _participant_roster_hash(roster):
            raise ValueError("PISD outer-train roster/hash is empty or inconsistent")
        if not self.discovery_participant_ids or not self.discovery_file_ids or not self.discovery_window_ids:
            raise ValueError("PISD discovery roster must be persisted")
        channel_schema = tuple(map(str, self.channel_schema))
        if not channel_schema or len(channel_schema) != len(set(channel_schema)):
            raise ValueError("channel_schema must be non-empty and unique")
        source_channels = integer_arrays["source_channels"]
        if np.any(source_channels >= len(channel_schema)):
            raise ValueError("source channel index is outside channel_schema")
        if any(
            name != channel_schema[int(index)]
            for name, index in zip(self.source_channel_names, source_channels)
        ):
            raise ValueError("source channel names/indices disagree with channel_schema")
        selection_payload = "\n".join(
            f"{participant}\t{file_id}\t{window_id}"
            for participant, file_id, window_id in zip(
                self.discovery_participant_ids,
                self.discovery_file_ids,
                self.discovery_window_ids,
            )
        ).encode("utf-8")
        if self.discovery_selection_hash != hashlib.sha256(selection_payload).hexdigest():
            raise ValueError("discovery_selection_hash does not match the selected window roster")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_classes", integer_arrays["source_classes"])
        object.__setattr__(self, "information_gains", scores)
        object.__setattr__(self, "source_sample_indices", integer_arrays["source_sample_indices"])
        object.__setattr__(self, "source_channels", integer_arrays["source_channels"])
        object.__setattr__(self, "source_starts", starts)
        object.__setattr__(self, "source_ends", ends)
        object.__setattr__(self, "source_start_seconds", start_seconds)
        object.__setattr__(self, "source_end_seconds", end_seconds)
        object.__setattr__(self, "candidate_lengths", lengths)
        object.__setattr__(self, "discovery_sequence_lengths", sequence_lengths)
        object.__setattr__(self, "pip_counts", integer_arrays["pip_counts"])
        object.__setattr__(self, "source_channel_names", tuple(map(str, self.source_channel_names)))
        object.__setattr__(self, "source_participant_ids", tuple(map(str, self.source_participant_ids)))
        object.__setattr__(self, "source_file_ids", tuple(map(str, self.source_file_ids)))
        object.__setattr__(self, "source_window_ids", tuple(map(str, self.source_window_ids)))
        object.__setattr__(self, "fitted_participant_ids", roster)
        object.__setattr__(self, "discovery_participant_ids", tuple(map(str, self.discovery_participant_ids)))
        object.__setattr__(self, "discovery_file_ids", tuple(map(str, self.discovery_file_ids)))
        object.__setattr__(self, "discovery_window_ids", tuple(map(str, self.discovery_window_ids)))
        object.__setattr__(self, "channel_schema", channel_schema)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def shapelet_length_samples(self) -> None:
        """The reference deliberately has no fixed shapelet length."""

        return None

    @property
    def shapelet_length_seconds(self) -> None:
        """The reference deliberately has no fixed physical duration."""

        return None

    def candidate_records(self) -> tuple[dict[str, object], ...]:
        """Return archive-ready provenance without duplicating waveform values."""

        return tuple(
            {
                "shapelet_index": index,
                "source_class": int(self.source_classes[index]),
                "source_participant_id": self.source_participant_ids[index],
                "source_file_id": self.source_file_ids[index],
                "source_sample_index": int(self.source_sample_indices[index]),
                "source_channel": int(self.source_channels[index]),
                "source_channel_name": self.source_channel_names[index],
                "source_window_id": self.source_window_ids[index],
                "start_sample": int(self.source_starts[index]),
                "end_sample_exclusive": int(self.source_ends[index]),
                "start_seconds": float(self.source_start_seconds[index]),
                "end_seconds": float(self.source_end_seconds[index]),
                "candidate_length_samples": int(self.candidate_lengths[index]),
                "discovery_sequence_length": int(self.discovery_sequence_lengths[index]),
                "num_pips": int(self.pip_counts[index]),
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
                "position_search_neighbourhood_samples": int(
                    self.position_search_neighbourhood_samples
                ),
                "information_gain": float(self.information_gains[index]),
            }
            for index in range(self.count)
        )


def _zscored_time_axis(length: int) -> np.ndarray:
    """Match ``scipy.stats.zscore(range(T))`` used by upstream auto_piss."""

    axis = np.arange(int(length), dtype=np.float64)
    return (axis - axis.mean()) / axis.std(ddof=0)


def _perpendicular_distances(
    values: np.ndarray,
    positions: np.ndarray,
    left: int,
    right: int,
    time_axis: np.ndarray,
) -> np.ndarray:
    """Match upstream ``pd_distance`` on its z-scored time coordinate."""

    right_x = float(time_axis[right])
    left_x = float(time_axis[left])
    slope = (float(values[right]) - float(values[left])) / (right_x - left_x)
    intercept = float(values[right]) - right_x * slope
    return np.abs(
        -values[positions] + slope * time_axis[positions] + intercept
    ) / np.sqrt(slope**2 + 1.0)


def _segment_candidate(
    values: np.ndarray,
    left: int,
    right: int,
    time_axis: np.ndarray | None = None,
) -> tuple[float, int] | None:
    if right - left <= 1:
        return None
    positions = np.arange(left + 1, right, dtype=np.int64)
    axis = _zscored_time_axis(values.size) if time_axis is None else time_axis
    distances = _perpendicular_distances(values, positions, left, right, axis)
    local = int(np.argmax(distances))
    return float(distances[local]), int(positions[local])


def _pip_indices(signal: np.ndarray, count: int) -> np.ndarray:
    """Select PIPs with the exact upstream global first-maximum rule."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
        raise ValueError("PIP extraction requires a finite one-dimensional signal")
    target = max(3, min(int(count), values.size))
    pips = [0, values.size - 1]
    remaining = list(range(1, values.size - 1))
    axis = _zscored_time_axis(values.size)
    for _ in range(target - 2):
        best_distance = -1.0
        best_position = -1
        for position in remaining:
            right_offset = int(np.searchsorted(pips, position, side="right"))
            left = pips[right_offset - 1]
            right = pips[right_offset]
            distance = float(
                _perpendicular_distances(
                    values,
                    np.asarray((position,), dtype=np.int64),
                    left,
                    right,
                    axis,
                )[0]
            )
            if distance > best_distance:
                best_distance = distance
                best_position = position
        if best_position < 0:
            raise RuntimeError("PIP extraction failed before reaching the derived count")
        pips.append(best_position)
        pips.sort()
        remaining.remove(best_position)
    return np.asarray(pips, dtype=np.int64)


def _pip_count(sequence_length: int, ratio: float) -> int:
    # Upstream OSD/PISD applies floor(0.20*T), with a minimum of five PIPs.
    # The count is capped by actual T. Thus 64 is derived only for T=320 and
    # is never a fixed project constant.
    return min(int(sequence_length), max(5, int(float(ratio) * int(sequence_length))))


def _insertion_stage_three_pip_intervals(
    signal: np.ndarray,
    count: int,
) -> tuple[tuple[int, int], ...]:
    """Port upstream auto_piss_extractor candidate endpoints.

    Each inserted PIP contributes the interval spanning its adjacent PIPs and
    the neighbouring three-PIP intervals when present. Insertion-stage
    intervals (including repeated endpoints) are retained; keeping only the
    final PIP triples changes the literature candidate set. The right endpoint
    is half-open, matching upstream series[start:end] extraction.
    """

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
        raise ValueError("PIP extraction requires a finite one-dimensional signal")
    target = max(3, min(int(count), values.size))
    pips = [0, values.size - 1]
    remaining = list(range(1, values.size - 1))
    time_axis = _zscored_time_axis(values.size)
    intervals: list[tuple[int, int]] = []
    for _ in range(target - 2):
        best_distance = -1.0
        best_position = -1
        for position in remaining:
            right_offset = int(np.searchsorted(pips, position, side="right"))
            left = pips[right_offset - 1]
            right = pips[right_offset]
            distance = float(
                _perpendicular_distances(
                    values,
                    np.asarray((position,), dtype=np.int64),
                    left,
                    right,
                    time_axis,
                )[0]
            )
            if distance > best_distance:
                best_distance = distance
                best_position = position
        if best_position < 0:
            raise RuntimeError("PISD PIP extraction failed before reaching the derived count")
        pips.append(best_position)
        pips.sort()
        remaining.remove(best_position)
        position_index = pips.index(best_position)
        intervals.append((pips[position_index - 1], pips[position_index + 1]))
        if position_index > 1:
            intervals.append((pips[position_index - 2], pips[position_index]))
        if position_index < len(pips) - 2:
            intervals.append((pips[position_index], pips[position_index + 2]))
    return tuple(intervals)


def _balanced_discovery_indices(
    y: np.ndarray,
    participant_ids: tuple[str, ...],
    file_ids: tuple[str, ...],
    *,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Select class-, participant-, then within-participant file-balanced windows."""

    classes = tuple(sorted(np.unique(y).tolist()))
    if len(classes) < 2:
        raise ValueError("OSD/PISD discovery needs at least two classes")
    base, remainder = divmod(maximum, len(classes))
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for class_offset, class_value in enumerate(classes):
        quota = base + int(class_offset < remainder)
        groups: dict[str, dict[str, list[int]]] = {}
        for index in np.flatnonzero(y == class_value).tolist():
            groups.setdefault(participant_ids[index], {}).setdefault(
                file_ids[index], []
            ).append(index)
        participant_queues: dict[str, list[int]] = {}
        for participant, by_file in groups.items():
            for indices in by_file.values():
                rng.shuffle(indices)
            queue: list[int] = []
            active_files = sorted(by_file)
            while active_files:
                next_files: list[str] = []
                for file_id in active_files:
                    indices = by_file[file_id]
                    if indices:
                        queue.append(indices.pop())
                    if indices:
                        next_files.append(file_id)
                active_files = next_files
            participant_queues[participant] = queue
        participants = sorted(participant_queues)
        cursor = 0
        while participants and cursor < quota:
            next_participants: list[str] = []
            for participant in participants:
                values = participant_queues[participant]
                if values and cursor < quota:
                    selected.append(values.pop(0))
                    cursor += 1
                if values:
                    next_participants.append(participant)
            participants = next_participants
    return np.asarray(sorted(selected), dtype=np.int64)


def _z_normalise(window: np.ndarray) -> np.ndarray:
    mean = window.mean()
    scale = window.std()
    return (window - mean) / max(float(scale), 1e-6)


def _complexity(window: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.diff(window, axis=-1) ** 2, axis=-1) + 1e-3)


def _pisd_min_distance(
    series: np.ndarray,
    candidate: np.ndarray,
    *,
    source_start: int,
    source_end: int,
    position_search_neighbourhood_samples: int,
    position_chunk_size: int,
) -> float:
    """Exact stride-one raw PISD distance within the source neighbourhood."""

    values = np.asarray(series, dtype=np.float64)
    shapelet = np.asarray(candidate, dtype=np.float64)
    width = shapelet.size
    if values.size < width:
        return float("inf")
    # Match upstream pcs_extractor exactly. Its discovery-time left boundary is
    # start - w + 1 (distinct from ShapeBlock's inference-time start - w).
    search_start = max(
        0,
        int(source_start) - int(position_search_neighbourhood_samples) + 1,
    )
    search_end = min(values.size, int(source_end) + int(position_search_neighbourhood_samples))
    region = values[search_start:search_end]
    if region.size < width:
        return float("inf")
    windows = np.lib.stride_tricks.sliding_window_view(region, width)
    candidate_complexity = float(_complexity(shapelet))
    best = float("inf")
    for offset in range(0, windows.shape[0], position_chunk_size):
        chunk = np.asarray(windows[offset : offset + position_chunk_size])
        base = np.sqrt(np.sum((chunk - shapelet[None, :]) ** 2, axis=-1))
        complexity = _complexity(chunk)
        correction = np.maximum(complexity, candidate_complexity) / np.maximum(
            np.minimum(complexity, candidate_complexity), 1e-8
        )
        best = min(best, float(np.min(base * correction)))
    return best


def _binary_entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    proportion = float(np.mean(values))
    if proportion <= 0.0 or proportion >= 1.0:
        return 0.0
    return float(-proportion * np.log2(proportion) - (1.0 - proportion) * np.log2(1.0 - proportion))


def _information_gain(distances: np.ndarray, positive: np.ndarray) -> float:
    """Match upstream find_best_split_point_and_info_gain numerically.

    Upstream evaluates only when rounded target-class recall on the left side
    equals a floating marker that starts at 0.2 and increments by 0.2. It does
    not scan every distinct distance threshold. The historical -1 sentinel
    is retained when no marker is reached because changing it changes ranking.
    """

    distance = np.asarray(distances, dtype=np.float64)
    target = np.asarray(positive, dtype=bool)
    if (
        distance.ndim != 1
        or target.shape != distance.shape
        or distance.size < 2
        or not np.isfinite(distance).all()
    ):
        raise ValueError("information gain requires aligned finite vectors")
    labels = target[np.argsort(distance)]
    right_positive = int(labels.sum())
    if right_positive <= 0:
        raise ValueError("information gain target class has no positive samples")
    total_positive = right_positive
    right_negative = int(labels.size - right_positive)
    left_positive = 0
    left_negative = 0
    best = -1.0
    recall_marker = 0.2

    def entropy_counts(positive_count: int, negative_count: int) -> float:
        total = positive_count + negative_count
        result = 0.0
        positive_rate = positive_count / total
        negative_rate = negative_count / total
        if positive_rate > 0.0:
            result -= positive_rate * np.log2(positive_rate)
        if negative_rate > 0.0:
            result -= negative_rate * np.log2(negative_rate)
        return float(result)

    parent = entropy_counts(total_positive, right_negative)
    for label in labels[:-1]:
        if bool(label):
            left_positive += 1
            right_positive -= 1
        else:
            left_negative += 1
            right_negative -= 1
        if round(left_positive / total_positive, 1) == recall_marker:
            left_count = left_positive + left_negative
            right_count = right_positive + right_negative
            gain = parent - (
                (left_count / labels.size)
                * entropy_counts(left_positive, left_negative)
                + (right_count / labels.size)
                * entropy_counts(right_positive, right_negative)
            )
            best = max(best, float(gain))
            if recall_marker < 1.0:
                recall_marker += 0.2
            else:
                break
    return float(best)


def _select_nonoverlapping_candidate_rows(
    ranked: list[tuple[float, int, int, int, int, int]],
    count: int,
) -> list[tuple[float, int, int, int, int, int]]:
    """Match upstream class selection: avoid same-channel overlap, then fill."""

    selected: list[tuple[float, int, int, int, int, int]] = []
    selected_positions: set[int] = set()
    for position, row in enumerate(ranked):
        _, _, channel, start, end, _ = row
        overlaps = any(
            other_channel == channel
            and max(start, other_start) < min(end, other_end)
            for _, _, other_channel, other_start, other_end, _ in selected
        )
        if not overlaps:
            selected.append(row)
            selected_positions.add(position)
            if len(selected) == count:
                return selected
    for position, row in enumerate(ranked):
        if position not in selected_positions:
            selected.append(row)
            if len(selected) == count:
                break
    return selected


def _rank_candidate_rows_upstream(
    candidate_rows: list[tuple[float, int, int, int, int, int]],
) -> list[tuple[float, int, int, int, int, int]]:
    """Reproduce upstream default NumPy argsort followed by reverse traversal."""

    if not candidate_rows:
        return []
    scores = np.asarray([row[0] for row in candidate_rows], dtype=np.float64)
    order = scores.argsort()
    return [candidate_rows[int(index)] for index in order[::-1]]


def _order_selected_candidate_rows_upstream(
    selected_rows: list[tuple[float, int, int, int, int, int]],
) -> list[tuple[float, int, int, int, int, int]]:
    """Reproduce upstream per-class final default argsort on start sample."""

    if not selected_rows:
        return []
    starts = np.asarray([row[3] for row in selected_rows], dtype=np.int64)
    order = starts.argsort()
    return [selected_rows[int(index)] for index in order]


def discover_pisd_shapelets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    participant_ids: list[str] | tuple[str, ...],
    file_ids: list[str] | tuple[str, ...],
    window_ids: list[str] | tuple[str, ...],
    channel_schema: list[str] | tuple[str, ...],
    *,
    discovery_method: str,
    input_fs_hz: float,
    outer_repeat_index: int,
    outer_fold_index: int,
    num_pip_ratio: float,
    shapelets_per_class: int,
    max_discovery_windows: int,
    discovery_balance: str,
    position_search_neighbourhood_samples: int,
    sequence_lengths: np.ndarray | None = None,
    distance_position_chunk_size: int = 256,
    seed: int = 42,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> PisdShapelets:
    """Discover the canonical variable-length channel-specific OSD/PISD bank."""

    if discovery_method != PISD_DISCOVERY_METHOD:
        raise ValueError(
            f"the reference requires discovery_method={PISD_DISCOVERY_METHOD}; no fallback is allowed"
        )
    if not np.isclose(num_pip_ratio, NUM_PIP_RATIO, rtol=0.0, atol=1e-12):
        raise ValueError(f"num_pip_ratio is frozen at {NUM_PIP_RATIO}")
    if int(shapelets_per_class) != SHAPELETS_PER_CLASS:
        raise ValueError(f"shapelets_per_class is frozen at {SHAPELETS_PER_CLASS}")
    if int(max_discovery_windows) != MAX_DISCOVERY_WINDOWS:
        raise ValueError(f"max_discovery_windows is frozen at {MAX_DISCOVERY_WINDOWS}")
    if discovery_balance != DISCOVERY_BALANCE:
        raise ValueError(f"discovery_balance must be {DISCOVERY_BALANCE}")
    if int(position_search_neighbourhood_samples) != POSITION_SEARCH_NEIGHBOURHOOD_SAMPLES:
        raise ValueError(
            "position_search_neighbourhood_samples must be exactly 128; "
            "it is a position-search neighbourhood, never a shapelet length"
        )
    if not np.isfinite(input_fs_hz) or input_fs_hz <= 0.0:
        raise ValueError("input_fs_hz must be finite and positive")
    if outer_repeat_index < 0 or outer_fold_index < 0:
        raise ValueError("outer repeat/fold indices must be non-negative")
    if distance_position_chunk_size <= 0:
        raise ValueError("distance_position_chunk_size must be positive")

    x = np.asarray(x_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.int64)
    participants = tuple(map(str, participant_ids))
    files = tuple(map(str, file_ids))
    windows = tuple(map(str, window_ids))
    channels = tuple(map(str, channel_schema))
    if x.ndim != 3 or y.shape != (x.shape[0],):
        raise ValueError("x/y must align as [sample,channel,time]")
    if len(participants) != x.shape[0] or len(files) != x.shape[0] or len(windows) != x.shape[0]:
        raise ValueError("one participant/file/window identity is required per discovery window")
    if len(channels) != x.shape[1] or len(channels) != len(set(channels)):
        raise ValueError("channel_schema must uniquely name every input channel")
    if not np.isfinite(x).all() or np.unique(y).size < 2:
        raise ValueError("PISD discovery requires finite inputs and at least two classes")
    lengths = (
        np.full(x.shape[0], x.shape[-1], dtype=np.int64)
        if sequence_lengths is None
        else np.asarray(sequence_lengths, dtype=np.int64)
    )
    if lengths.shape != (x.shape[0],) or np.any(lengths < 3) or np.any(lengths > x.shape[-1]):
        raise ValueError("sequence_lengths must give a valid actual T for every window")

    discovery_indices = _balanced_discovery_indices(
        y, participants, files, maximum=max_discovery_windows, seed=seed
    )
    if discovery_indices.size == 0:
        raise ValueError("participant/file-balanced discovery selected no windows")
    discovery_x = x[discovery_indices]
    discovery_y = y[discovery_indices]
    discovery_lengths = lengths[discovery_indices]
    selected_values: list[np.ndarray] = []
    selected_classes: list[int] = []
    selected_scores: list[float] = []
    selected_samples: list[int] = []
    selected_channels: list[int] = []
    selected_starts: list[int] = []
    selected_ends: list[int] = []
    selected_lengths: list[int] = []
    selected_channel_names: list[str] = []
    selected_participants: list[str] = []
    selected_files: list[str] = []
    selected_windows: list[str] = []
    selected_sequence_lengths: list[int] = []
    selected_pip_counts: list[int] = []
    evaluated_candidates = 0

    for class_value in sorted(np.unique(discovery_y).tolist()):
        # Candidate waveforms are never retained in this list. Only compact
        # metadata and scores are sorted; selected waveforms are reconstructed
        # from discovery_x. This keeps exact exhaustive enumeration memory-safe
        # without adding a hidden candidate cap.
        candidate_rows: list[tuple[float, int, int, int, int, int]] = []
        positive = discovery_y == class_value
        positive_indices = np.flatnonzero(positive).tolist()
        # Local upstream enumerates class -> channel -> source sample ->
        # insertion-stage candidate. This ordinal matters for equal IG scores
        # because its default NumPy argsort is then traversed in reverse.
        for channel in range(x.shape[1]):
            for local_index in positive_indices:
                if should_cancel is not None and should_cancel():
                    raise InterruptedError(
                        f"PISD discovery cancelled after {evaluated_candidates} candidates"
                    )
                global_index = int(discovery_indices[local_index])
                actual_t = int(discovery_lengths[local_index])
                pip_count = _pip_count(actual_t, num_pip_ratio)
                intervals = _insertion_stage_three_pip_intervals(
                    discovery_x[local_index, channel, :actual_t],
                    pip_count,
                )
                for start, end in intervals:
                    if end - start < 2:
                        continue
                    if should_cancel is not None and should_cancel():
                        raise InterruptedError(
                            f"PISD discovery cancelled after {evaluated_candidates} candidates"
                        )
                    candidate = discovery_x[local_index, channel, start:end]
                    distances = np.asarray(
                        [
                            _pisd_min_distance(
                                discovery_x[row, channel, : int(discovery_lengths[row])],
                                candidate,
                                source_start=start,
                                source_end=end,
                                position_search_neighbourhood_samples=(
                                    position_search_neighbourhood_samples
                                ),
                                position_chunk_size=distance_position_chunk_size,
                            )
                            for row in range(discovery_x.shape[0])
                        ],
                        dtype=np.float64,
                    )
                    score = _information_gain(distances, positive)
                    candidate_rows.append(
                        (score, global_index, channel, start, end, pip_count)
                    )
                    evaluated_candidates += 1
                    if progress_callback is not None and evaluated_candidates % 100 == 0:
                        progress_callback(
                            {
                                "event": "candidate_batch_complete",
                                "class_value": int(class_value),
                                "evaluated_candidate_count": evaluated_candidates,
                                "source_window_id": windows[global_index],
                                "source_channel": channel,
                            }
                        )
        ranked = _rank_candidate_rows_upstream(candidate_rows)
        if len(ranked) < shapelets_per_class:
            raise RuntimeError(
                f"PISD failure: class {class_value} produced {len(ranked)} candidates; "
                f"{shapelets_per_class} are required and effect-size fallback is forbidden"
            )
        chosen = _order_selected_candidate_rows_upstream(
            _select_nonoverlapping_candidate_rows(ranked, shapelets_per_class)
        )
        for score, sample_index, channel, start, end, pip_count in chosen:
            candidate = x[sample_index, channel, start:end].copy()
            selected_values.append(candidate)
            selected_classes.append(int(class_value))
            selected_scores.append(float(score))
            selected_samples.append(sample_index)
            selected_channels.append(channel)
            selected_starts.append(start)
            selected_ends.append(end)
            selected_lengths.append(end - start)
            selected_channel_names.append(channels[channel])
            selected_participants.append(participants[sample_index])
            selected_files.append(files[sample_index])
            selected_windows.append(windows[sample_index])
            selected_sequence_lengths.append(int(lengths[sample_index]))
            selected_pip_counts.append(pip_count)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "class_complete",
                    "class_value": int(class_value),
                    "evaluated_candidate_count": evaluated_candidates,
                    "ranked_candidate_count": len(ranked),
                }
            )

    roster = tuple(sorted(set(participants)))
    starts = np.asarray(selected_starts, dtype=np.int64)
    ends = np.asarray(selected_ends, dtype=np.int64)
    discovery_participants = tuple(participants[index] for index in discovery_indices)
    discovery_files = tuple(files[index] for index in discovery_indices)
    discovery_windows = tuple(windows[index] for index in discovery_indices)
    selection_payload = "\n".join(
        f"{participant}\t{file_id}\t{window_id}"
        for participant, file_id, window_id in zip(
            discovery_participants, discovery_files, discovery_windows
        )
    ).encode("utf-8")
    return PisdShapelets(
        values=tuple(selected_values),
        source_classes=np.asarray(selected_classes),
        information_gains=np.asarray(selected_scores),
        source_sample_indices=np.asarray(selected_samples),
        source_channels=np.asarray(selected_channels),
        source_starts=starts,
        source_ends=ends,
        source_start_seconds=starts / float(input_fs_hz),
        source_end_seconds=ends / float(input_fs_hz),
        candidate_lengths=np.asarray(selected_lengths),
        source_channel_names=tuple(selected_channel_names),
        source_participant_ids=tuple(selected_participants),
        source_file_ids=tuple(selected_files),
        source_window_ids=tuple(selected_windows),
        discovery_sequence_lengths=np.asarray(selected_sequence_lengths),
        pip_counts=np.asarray(selected_pip_counts),
        fitted_participant_ids=roster,
        discovery_participant_ids=discovery_participants,
        discovery_file_ids=discovery_files,
        discovery_window_ids=discovery_windows,
        discovery_selection_hash=hashlib.sha256(selection_payload).hexdigest(),
        channel_schema=channels,
        discovery_method=discovery_method,
        discovery_balance=discovery_balance,
        input_fs_hz=float(input_fs_hz),
        num_pip_ratio=float(num_pip_ratio),
        shapelets_per_class=int(shapelets_per_class),
        max_discovery_windows=int(max_discovery_windows),
        discovery_window_count=int(discovery_indices.size),
        position_search_neighbourhood_samples=int(
            position_search_neighbourhood_samples
        ),
        pip_rounding_rule=PIP_ROUNDING_RULE,
        pip_selection_rule=PIP_SELECTION_RULE,
        candidate_generation_rule=CANDIDATE_GENERATION_RULE,
        candidate_enumeration_rule=CANDIDATE_ENUMERATION_RULE,
        candidate_ranking_rule=CANDIDATE_RANKING_RULE,
        selected_bank_order_rule=SELECTED_BANK_ORDER_RULE,
        discovery_position_search_boundary_rule=(
            DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE
        ),
        information_gain_split_rule=INFORMATION_GAIN_SPLIT_RULE,
        outer_repeat_index=int(outer_repeat_index),
        outer_fold_index=int(outer_fold_index),
        outer_train_participant_hash=_participant_roster_hash(roster),
    )


__all__ = [
    "DISCOVERY_BALANCE",
    "MAX_DISCOVERY_WINDOWS",
    "NUM_PIP_RATIO",
    "POSITION_SEARCH_NEIGHBOURHOOD_SAMPLES",
    "PIP_ROUNDING_RULE",
    "PIP_SELECTION_RULE",
    "CANDIDATE_GENERATION_RULE",
    "CANDIDATE_ENUMERATION_RULE",
    "CANDIDATE_RANKING_RULE",
    "SELECTED_BANK_ORDER_RULE",
    "DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE",
    "INFORMATION_GAIN_SPLIT_RULE",
    "PISD_DISCOVERY_METHOD",
    "PisdShapelets",
    "SHAPELETS_PER_CLASS",
    "_insertion_stage_three_pip_intervals",
    "_order_selected_candidate_rows_upstream",
    "_pisd_min_distance",
    "_rank_candidate_rows_upstream",
    "discover_pisd_shapelets",
]

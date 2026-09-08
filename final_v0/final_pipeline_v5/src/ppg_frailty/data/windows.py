"""工程与 DL 共用 WindowPlan / Unified engineering and DL window planning.

中文：所有切窗行为均是构造参数：物理时长、hop、端点对齐、短记录动作、
尾段 padding 和窗口上限。WindowPlan 只规划索引；实际信号值由 extract_window
读取，因此不会意外修改 amplitude-preserving signal view。

English: Every window behavior is explicit: physical duration, hop, alignment,
short-record action, tail padding, and cap. WindowPlan emits indices only;
extract_window reads values without mutating the amplitude-preserving source view.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class ShortRecordError(ValueError):
    """短记录按显式 reject policy 被拒绝 / Explicit short-record rejection."""


@dataclass(frozen=True)
class WindowSlice:
    """一个 time-aligned window / One explicitly masked time window."""

    source_record_id: str
    fs: float
    start_sample: int
    end_sample: int
    valid_length: int
    window_length: int
    padding_mask: tuple[bool, ...]


@dataclass(frozen=True)
class WindowPlan:
    """没有隐藏默认的窗口规划 / Window planning with no hidden defaults."""

    source_record_id: str
    window_seconds: float
    hop_seconds: float
    end_alignment: str
    short_record_action: str
    include_padded_tail: bool
    max_windows: int | None
    cap_policy: str
    # Zero preserves the historical direct ``WindowPlan`` API (all explicitly
    # requested padded rows are kept). Config-resolved plans always materialize
    # an explicit value, with the pipeline default remaining 1.0/complete-only.
    min_valid_fraction: float = 0.0
    # Fractional caps migrate the legacy ``cnn_max_windows_fraction`` control
    # without inventing a second window planner.  It is mutually exclusive
    # with ``max_windows`` and is resolved against this recording's candidate
    # count immediately before uniform-progress selection.
    max_window_fraction: float | None = None

    @staticmethod
    def _sample_count(seconds: float, fs: float, *, field_name: str) -> int:
        """将物理时间严格转换为整数 samples / Convert seconds exactly."""

        raw = float(seconds) * float(fs)
        rounded = int(round(raw))
        if rounded <= 0 or not np.isclose(raw, rounded, rtol=0.0, atol=1e-9):
            raise ValueError(f"{field_name} must map to a positive integer sample count")
        return rounded

    def _validate(self) -> None:
        """校验所有显式 policy / Validate every explicit policy."""

        if not self.source_record_id:
            raise ValueError("source_record_id must be non-empty")
        if (not np.isfinite(float(self.window_seconds)) or not np.isfinite(float(self.hop_seconds))
                or self.window_seconds <= 0.0 or self.hop_seconds <= 0.0):
            raise ValueError("window/hop seconds must be finite and positive")
        if self.end_alignment not in {
                "start",
                "end",
                "include_right_aligned_if_distinct",
        }:
            raise ValueError("end_alignment must be start, end, or " "include_right_aligned_if_distinct")
        if self.short_record_action not in {"reject", "pad_right"}:
            raise ValueError("short_record_action must be reject or pad_right")
        if self.include_padded_tail and self.end_alignment != "start":
            raise ValueError("padded tail is only defined for start alignment")
        if self.max_windows is not None and self.max_windows <= 0:
            raise ValueError("max_windows must be positive or null")
        if self.max_window_fraction is not None and (isinstance(self.max_window_fraction, bool)
                                                     or not np.isfinite(float(self.max_window_fraction))
                                                     or not 0.0 < float(self.max_window_fraction) <= 1.0):
            raise ValueError("max_window_fraction must be null or finite in (0,1]")
        if self.max_windows is not None and self.max_window_fraction is not None:
            raise ValueError("max_windows and max_window_fraction are mutually exclusive")
        if self.cap_policy not in {"uniform_progress", "not_applicable"}:
            raise ValueError("unsupported cap_policy")
        has_cap = self.max_windows is not None or self.max_window_fraction is not None
        if has_cap and self.cap_policy != "uniform_progress":
            raise ValueError("a finite cap requires uniform_progress")
        if not has_cap and self.cap_policy != "not_applicable":
            raise ValueError("uncapped plans require cap_policy=not_applicable")
        if (isinstance(self.min_valid_fraction, bool) or not np.isfinite(float(self.min_valid_fraction))
                or not 0.0 <= float(self.min_valid_fraction) <= 1.0):
            raise ValueError("min_valid_fraction must be finite in [0,1]")

    def plan(self, n_samples: int, fs: float) -> tuple[WindowSlice, ...]:
        """生成确定性索引和 mask / Generate deterministic indices and masks."""

        self._validate()
        if n_samples < 0 or fs <= 0.0:
            raise ValueError("n_samples must be non-negative and fs positive")
        window = self._sample_count(self.window_seconds, fs, field_name="window_seconds")
        hop = self._sample_count(self.hop_seconds, fs, field_name="hop_seconds")
        if n_samples < window:
            if self.short_record_action == "reject":
                raise ShortRecordError(f"{self.source_record_id}: {n_samples} < {window} samples")
            candidate = self._slice(
                start=0,
                valid_length=n_samples,
                window_length=window,
                fs=fs,
            )
            return ((candidate, )
                    if candidate.valid_length / candidate.window_length >= float(self.min_valid_fraction) else ())

        if self.end_alignment in {
                "start",
                "include_right_aligned_if_distinct",
        }:
            starts = list(range(0, n_samples - window + 1, hop))
            if self.end_alignment == "include_right_aligned_if_distinct":
                # Match the historical classifier exactly: retain the regular
                # complete-window grid anchored at sample zero, then append the
                # last complete right-aligned window only when it is distinct.
                right_aligned_start = n_samples - window
                if starts[-1] != right_aligned_start:
                    starts.append(right_aligned_start)
            elif self.include_padded_tail:
                next_start = starts[-1] + hop
                if next_start < n_samples:
                    starts.append(next_start)
        else:
            # 中文：从最后一个完整窗口向前对齐，再恢复时间升序。
            # English: Align backward from the last complete window, then sort.
            starts = sorted(range(n_samples - window, -1, -hop))

        slices = tuple(
            self._slice(
                start=start,
                valid_length=min(window, n_samples - start),
                window_length=window,
                fs=fs,
            ) for start in starts if min(window, n_samples - start) / window >= float(self.min_valid_fraction))
        effective_cap = self.max_windows
        if self.max_window_fraction is not None:
            effective_cap = max(
                1,
                int(math.ceil(len(slices) * float(self.max_window_fraction))),
            )
        if effective_cap is not None and len(slices) > effective_cap:
            selected = _uniform_indices(len(slices), effective_cap)
            slices = tuple(slices[index] for index in selected)
        return slices

    def _slice(
        self,
        *,
        start: int,
        valid_length: int,
        window_length: int,
        fs: float,
    ) -> WindowSlice:
        """构造右侧 padding mask / Build a right-padding mask."""

        valid = max(0, min(int(valid_length), int(window_length)))
        mask = (False, ) * valid + (True, ) * (window_length - valid)
        return WindowSlice(
            source_record_id=self.source_record_id,
            fs=float(fs),
            start_sample=int(start),
            end_sample=int(start + valid),
            valid_length=valid,
            window_length=int(window_length),
            padding_mask=mask,
        )


def _uniform_indices(length: int, count: int) -> tuple[int, ...]:
    """按 recording progress 均匀选位置 / Uniformly select ordered positions."""

    if count <= 0 or length < count:
        raise ValueError("uniform selection requires 0 < count <= length")
    if count == 1:
        return (0, )
    indices = tuple(int(round(index * (length - 1) / (count - 1))) for index in range(count))
    if len(set(indices)) != count:
        raise RuntimeError("uniform index selection produced duplicates")
    return indices


def extract_window(
    values: np.ndarray,
    window: WindowSlice,
    *,
    pad_value: float,
) -> np.ndarray:
    """复制并按 mask 右填充窗口 / Copy and right-pad one planned window."""

    source = np.asarray(values)
    if source.ndim < 1:
        raise ValueError("window source must have a sample dimension")
    if window.end_sample > source.shape[0]:
        raise ValueError("window exceeds source samples")
    segment = np.array(source[window.start_sample:window.end_sample], copy=True)
    if segment.shape[0] != window.valid_length:
        raise ValueError("window valid_length/source mismatch")
    pad_rows = window.window_length - window.valid_length
    if pad_rows > 0:
        widths = [(0, pad_rows)] + [(0, 0)] * (segment.ndim - 1)
        segment = np.pad(segment, widths, mode="constant", constant_values=pad_value)
    return segment


__all__ = [
    "ShortRecordError",
    "WindowPlan",
    "WindowSlice",
    "extract_window",
]

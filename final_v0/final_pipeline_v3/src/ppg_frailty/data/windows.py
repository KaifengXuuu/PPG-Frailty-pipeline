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
        if self.window_seconds <= 0.0 or self.hop_seconds <= 0.0:
            raise ValueError("window/hop seconds must be positive")
        if self.end_alignment not in {"start", "end"}:
            raise ValueError("end_alignment must be start or end")
        if self.short_record_action not in {"reject", "pad_right"}:
            raise ValueError("short_record_action must be reject or pad_right")
        if self.include_padded_tail and self.end_alignment != "start":
            raise ValueError("padded tail is only defined for start alignment")
        if self.max_windows is not None and self.max_windows <= 0:
            raise ValueError("max_windows must be positive or null")
        if self.cap_policy not in {"uniform_progress", "not_applicable"}:
            raise ValueError("unsupported cap_policy")
        if self.max_windows is not None and self.cap_policy != "uniform_progress":
            raise ValueError("a finite cap requires uniform_progress")
        if self.max_windows is None and self.cap_policy != "not_applicable":
            raise ValueError("uncapped plans require cap_policy=not_applicable")

    def plan(self, n_samples: int, fs: float) -> tuple[WindowSlice, ...]:
        """生成确定性索引和 mask / Generate deterministic indices and masks."""

        self._validate()
        if n_samples < 0 or fs <= 0.0:
            raise ValueError("n_samples must be non-negative and fs positive")
        window = self._sample_count(
            self.window_seconds, fs, field_name="window_seconds"
        )
        hop = self._sample_count(self.hop_seconds, fs, field_name="hop_seconds")
        if n_samples < window:
            if self.short_record_action == "reject":
                raise ShortRecordError(
                    f"{self.source_record_id}: {n_samples} < {window} samples"
                )
            return (
                self._slice(
                    start=0,
                    valid_length=n_samples,
                    window_length=window,
                    fs=fs,
                ),
            )

        if self.end_alignment == "start":
            starts = list(range(0, n_samples - window + 1, hop))
            if self.include_padded_tail:
                next_start = starts[-1] + hop
                if next_start < n_samples:
                    starts.append(next_start)
        else:
            # 中文：从最后一个完整窗口向前对齐，再恢复时间升序。
            # English: Align backward from the last complete window, then sort.
            starts = sorted(range(n_samples - window, -1, -hop))

        if self.max_windows is not None and len(starts) > self.max_windows:
            starts = [starts[index] for index in _uniform_indices(len(starts), self.max_windows)]
        return tuple(
            self._slice(
                start=start,
                valid_length=min(window, n_samples - start),
                window_length=window,
                fs=fs,
            )
            for start in starts
        )

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
        mask = (False,) * valid + (True,) * (window_length - valid)
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
        return (0,)
    indices = tuple(
        int(round(index * (length - 1) / (count - 1))) for index in range(count)
    )
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
    segment = np.array(
        source[window.start_sample : window.end_sample], copy=True
    )
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

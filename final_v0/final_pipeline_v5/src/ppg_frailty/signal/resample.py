"""Anti-aliased DL-only views; feature and timing paths remain at 400 Hz."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

import numpy as np
from scipy import signal

# Historical named comparison targets.  This tuple is retained as catalog
# metadata; it is not a runtime allow-list.
V2_DL_RESAMPLING_TARGETS_HZ = (100.0, 160.0, 200.0)

def _audited_ratio(source_fs_hz: float, target_fs_hz: float, message: str) -> Fraction:
    ratio = Fraction(str(float(target_fs_hz) / float(source_fs_hz))).limit_denominator(10_000)
    realized = float(source_fs_hz) * ratio.numerator / ratio.denominator
    if not np.isclose(realized, float(target_fs_hz), atol=1e-12, rtol=0.0):
        raise ValueError(message)
    return ratio

def validate_dl_resampling_config(config: object) -> dict[str, object]:
    """Validate a DL-only sampling switch without changing canonical views."""
    if not isinstance(config, Mapping):
        raise ValueError("resolved signal.dl_resampling must be a mapping")
    required_keys = {
        "enabled",
        "target_fs_hz",
        "method",
        "preserve_feature_grid_hz",
    }
    if frozenset(config) not in {frozenset(required_keys), frozenset(required_keys | {"case_id"})}:
        raise ValueError("resolved signal.dl_resampling key mismatch")
    enabled = config["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError("signal.dl_resampling.enabled must be boolean")
    target_raw = config["target_fs_hz"]
    source_raw = config["preserve_feature_grid_hz"]
    if isinstance(target_raw, bool) or isinstance(source_raw, bool):
        raise ValueError("DL sampling rates must be numeric rather than boolean")
    try:
        target = float(target_raw)
        source_grid = float(source_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("DL sampling rates must be numeric") from exc
    if config["method"] != "polyphase_anti_alias":
        raise ValueError("V2 DL resampling requires polyphase anti-alias filtering")
    if source_grid != 400.0:
        raise ValueError("V2 DL resampling must preserve the 400-Hz feature grid")
    if enabled and (not np.isfinite(target) or target <= 0.0 or target > source_grid):
        raise ValueError(
            "enabled V2 DL target must be finite, positive, and no higher " "than the preserved source grid"
        )
    if not enabled and target != source_grid:
        raise ValueError("disabled V2 DL resampling must retain the 400-Hz target")
    _audited_ratio(source_grid, target, "target_fs_hz cannot be represented by the audited ratio")
    case_id = config.get("case_id")
    if case_id is not None:
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("DL resampling case_id must be a non-empty string")
        from ..models.time_scale import fixed_kernel_case

        case = fixed_kernel_case(str(case_id))
        if bool(enabled) != (float(case.dl_fs_hz) != 400.0) or target != float(case.dl_fs_hz):
            raise ValueError("DL resampling case_id disagrees with enabled/target")
    resolved = {
        "enabled": enabled,
        "target_fs_hz": target,
        "method": "polyphase_anti_alias",
        "preserve_feature_grid_hz": 400.0,
    }
    if case_id is not None:
        resolved["case_id"] = str(case_id)
    return resolved

def prepare_configured_dl_input(
    values: np.ndarray,
    sample_mask: np.ndarray,
    *,
    target_fs_hz: float,
    source_fs_hz: float = 400.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Resample valid prefixes of masked ``[window, channel, sample]`` DL inputs."""
    array = np.asarray(values, dtype=np.float32)
    mask = np.asarray(sample_mask, dtype=bool)
    source = float(source_fs_hz)
    target = float(target_fs_hz)
    if (
        array.ndim != 3
        or array.shape[0] == 0
        or array.shape[2] == 0
        or mask.shape != (array.shape[0], array.shape[2])
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            "configured DL input requires finite [sample,channel,T] windows " "and a matching [sample,T] mask"
        )
    if not np.isfinite(source) or not np.isfinite(target) or source <= 0.0 or target <= 0.0 or target > source:
        raise ValueError("configured DL source/target rates must be finite and satisfy " "0 < target <= source")
    ratio = _audited_ratio(source, target, "target_fs_hz cannot be represented by the audited ratio")

    valid_lengths = mask.sum(axis=1).astype(np.int64)
    expected_mask = np.arange(mask.shape[1], dtype=np.int64)[None, :] < valid_lengths[:, None]
    if not np.array_equal(mask, expected_mask):
        raise ValueError("configured DL resampling requires valid-prefix masks")
    if np.any(valid_lengths < 2):
        raise ValueError("each configured DL input needs at least two valid samples")

    target_length = int(round(array.shape[2] * target / source))
    if target_length < 2:
        raise ValueError("configured DL rate/window combination yields fewer than 2 samples")
    output = np.zeros((array.shape[0], array.shape[1], target_length), dtype=np.float32)
    output_mask = np.zeros((array.shape[0], target_length), dtype=bool)
    for sample_index, valid_length in enumerate(valid_lengths.tolist()):
        valid = array[sample_index, :, :valid_length]
        if ratio.numerator == ratio.denominator:
            transformed = valid
        else:
            transformed = signal.resample_poly(
                valid,
                up=ratio.numerator,
                down=ratio.denominator,
                axis=-1,
                window=("kaiser", 5.0),
                padtype="constant",
            ).astype(np.float32, copy=False)
        expected_valid = min(
            target_length,
            int(round(valid_length * target / source)),
        )
        copied = min(expected_valid, transformed.shape[-1], target_length)
        if copied < 1 or abs(transformed.shape[-1] - expected_valid) > 1:
            raise RuntimeError("polyphase resampler produced an unexpected valid length")
        output[sample_index, :, :copied] = transformed[:, :copied]
        output_mask[sample_index, :copied] = True

    return (
        output,
        output_mask,
        {
            "profile_kind": "configured_dl_resampling",
            "source_fs_hz": source,
            "target_fs_hz": target,
            "source_sequence_length_samples": int(array.shape[2]),
            "output_sequence_length_samples": target_length,
            "resample_up": int(ratio.numerator),
            "resample_down": int(ratio.denominator),
            "method": "scipy_signal_resample_poly_kaiser_beta5_constant_pad",
            "mask_transform": "contiguous_valid_prefix_scaled_with_dl_sampling_rate",
        },
    )

@dataclass(frozen=True)
class DlResampleResult:
    """独立 DL view 与比例合同 / Separate DL view and ratio contract."""
    values: np.ndarray
    source_fs_hz: float
    target_fs_hz: float
    up: int
    down: int
    method: str = "scipy_signal_resample_poly_anti_alias"

def resample_dl_view(
    values: np.ndarray,
    *,
    target_fs_hz: float,
    source_fs_hz: float = 400.0,
    axis: int = -1,
) -> DlResampleResult:
    """Build a separate polyphase view from finite 400-Hz data."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim not in (1, 2) or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("DL resampling requires a non-empty finite 1-D or 2-D array")
    if not np.isclose(float(source_fs_hz), 400.0, atol=0.0, rtol=0.0):
        raise ValueError("V2 DL resampling source must remain the canonical 400 Hz grid")
    if not np.isfinite(target_fs_hz) or float(target_fs_hz) <= 0.0:
        raise ValueError("target_fs_hz must be finite and positive")
    ratio = _audited_ratio(
        source_fs_hz, target_fs_hz, "target_fs_hz cannot be represented by the audited rational ratio"
    )
    output = signal.resample_poly(
        array,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=axis,
        padtype="line",
    )
    return DlResampleResult(
        values=np.asarray(output, dtype=np.float64),
        source_fs_hz=float(source_fs_hz),
        target_fs_hz=float(target_fs_hz),
        up=int(ratio.numerator),
        down=int(ratio.denominator),
    )

@dataclass(frozen=True)
class SynchronizedResampleResult:
    """One jointly resampled, row-synchronous multichannel time grid."""
    values: np.ndarray
    channel_schema: tuple[str, ...]
    timestamps_s: np.ndarray
    source_fs_hz: float
    target_fs_hz: float
    source_sample_count: int
    target_sample_count: int
    up: int
    down: int
    method: str = "scipy_signal_resample_poly_anti_alias_line_pad_v2"
    timing_origin_s: float = 0.0

    def validate(self) -> None:
        matrix = np.asarray(self.values)
        times = np.asarray(self.timestamps_s)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.channel_schema):
            raise ValueError("synchronized resample values/schema are misaligned")
        if matrix.shape[0] != self.target_sample_count or times.shape != (matrix.shape[0],):
            raise ValueError("synchronized resample target timeline is misaligned")
        if not np.isfinite(matrix).all() or not np.isfinite(times).all():
            raise ValueError("synchronized resample output must be finite")
        if len(set(self.channel_schema)) != len(self.channel_schema):
            raise ValueError("synchronized channel names must be unique and ordered")
        expected = np.arange(matrix.shape[0], dtype=np.float64) / self.target_fs_hz
        if not np.array_equal(times, expected):
            raise ValueError("synchronized resample timestamps drifted from the target grid")

def resample_synchronized_channels(
    values: np.ndarray,
    *,
    channel_schema: tuple[str, ...] | list[str],
    source_fs_hz: float,
    target_fs_hz: float = 400.0,
) -> SynchronizedResampleResult:
    """Jointly resample channels along axis zero while preserving synchrony."""
    matrix = np.asarray(values, dtype=np.float64)
    schema = tuple(str(name) for name in channel_schema)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(schema):
        raise ValueError("synchronized input must be non-empty samples-by-schema")
    if not schema or len(set(schema)) != len(schema) or any(not name for name in schema):
        raise ValueError("synchronized channel schema must be non-empty, ordered and unique")
    if not np.isfinite(matrix).all():
        raise ValueError("synchronized input must be finite")
    if (
        not np.isfinite(source_fs_hz)
        or not np.isfinite(target_fs_hz)
        or float(source_fs_hz) <= 0.0
        or float(target_fs_hz) <= 0.0
    ):
        raise ValueError("source and target sampling rates must be finite and positive")
    ratio = _audited_ratio(source_fs_hz, target_fs_hz, "target/source rate ratio is not exactly representable")
    output = np.asarray(
        signal.resample_poly(
            matrix,
            up=ratio.numerator,
            down=ratio.denominator,
            axis=0,
            padtype="line",
        ),
        dtype=np.float64,
    )
    expected_count = int(np.ceil(matrix.shape[0] * ratio.numerator / ratio.denominator))
    if output.shape != (expected_count, matrix.shape[1]):
        raise RuntimeError("polyphase synchronized output length drift")
    result = SynchronizedResampleResult(
        values=output,
        channel_schema=schema,
        timestamps_s=np.arange(output.shape[0], dtype=np.float64) / float(target_fs_hz),
        source_fs_hz=float(source_fs_hz),
        target_fs_hz=float(target_fs_hz),
        source_sample_count=int(matrix.shape[0]),
        target_sample_count=int(output.shape[0]),
        up=int(ratio.numerator),
        down=int(ratio.denominator),
    )
    result.validate()
    return result

__all__ = [
    "DlResampleResult", "SynchronizedResampleResult", "V2_DL_RESAMPLING_TARGETS_HZ", "prepare_configured_dl_input",
    "resample_dl_view", "resample_synchronized_channels", "validate_dl_resampling_config",
]

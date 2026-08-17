"""DL-only 抗混叠重采样 / Anti-aliased DL-only resampling.

中文：采集、峰时序、形态学和 feature audit 始终保留 400 Hz；本模块只创建
具备显式 provenance 的独立 DL view，绝不覆写原数组或伪装采样率。
English: Acquisition, peak timing, morphology and feature audit remain at 400 Hz.
This module creates a separate provenance-carrying DL view and never mutates input.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

import numpy as np
from scipy import signal


V2_DL_RESAMPLING_TARGETS_HZ = (100.0, 160.0, 200.0)


def validate_dl_resampling_config(config: object) -> dict[str, object]:
    """Validate the V2 DL-only sampling switch without touching canonical views.

    The reference line is disabled on the 400-Hz grid. Explicit enabled cases
    may target 100, 160, or 200 Hz only. In every case feature, peak-timing,
    morphology, and audit views remain on the independent 400-Hz grid.
    """

    if not isinstance(config, Mapping):
        raise ValueError("resolved signal.dl_resampling must be a mapping")
    required_keys = {
        "enabled",
        "target_fs_hz",
        "method",
        "preserve_feature_grid_hz",
    }
    if frozenset(config) not in {
        frozenset(required_keys), frozenset(required_keys | {"case_id"})
    }:
        raise ValueError("resolved signal.dl_resampling key mismatch")
    enabled = config["enabled"]
    target = float(config["target_fs_hz"])
    if not isinstance(enabled, bool):
        raise ValueError("signal.dl_resampling.enabled must be boolean")
    if config["method"] != "polyphase_anti_alias":
        raise ValueError("V2 DL resampling requires polyphase anti-alias filtering")
    if float(config["preserve_feature_grid_hz"]) != 400.0:
        raise ValueError("V2 DL resampling must preserve the 400-Hz feature grid")
    if enabled and target not in V2_DL_RESAMPLING_TARGETS_HZ:
        raise ValueError("enabled V2 DL target must be one of 100, 160, or 200 Hz")
    if not enabled and target != 400.0:
        raise ValueError("disabled V2 DL resampling must retain the 400-Hz target")
    case_id = config.get("case_id")
    if case_id is not None:
        from ..models.time_scale import fixed_kernel_case

        case = fixed_kernel_case(str(case_id))
        if bool(enabled) != (float(case.dl_fs_hz) != 400.0) or target != float(
            case.dl_fs_hz
        ):
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
    """生成独立 polyphase view / Build a separate polyphase-resampled view.

    中文：仅接受有限 1D/2D 数据和 400 Hz source；目标率必须能被有限有理数精确
    表达。SciPy polyphase 内部提供抗混叠低通，不对 feature branch 做任何操作。
    English: Finite 1-D/2-D input and the canonical 400-Hz source are required.
    SciPy polyphase filtering supplies anti-aliasing and leaves the feature branch alone.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim not in (1, 2) or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("DL resampling requires a non-empty finite 1-D or 2-D array")
    if not np.isclose(float(source_fs_hz), 400.0, atol=0.0, rtol=0.0):
        raise ValueError("V2 DL resampling source must remain the canonical 400 Hz grid")
    if not np.isfinite(target_fs_hz) or float(target_fs_hz) <= 0.0:
        raise ValueError("target_fs_hz must be finite and positive")
    ratio = Fraction(str(float(target_fs_hz) / float(source_fs_hz))).limit_denominator(10_000)
    realized = float(source_fs_hz) * ratio.numerator / ratio.denominator
    if not np.isclose(realized, float(target_fs_hz), atol=1e-12, rtol=0.0):
        raise ValueError("target_fs_hz cannot be represented by the audited rational ratio")
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
    """Jointly resample aligned channels; intended for PTT 500→400 adaptation.

    Every channel is filtered and resampled in the same call along axis zero.
    This prevents independent rounding or padding from breaking PPG/ECG/IMU
    synchrony. The source array is never overwritten.
    """

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
    ratio = Fraction(str(float(target_fs_hz) / float(source_fs_hz))).limit_denominator(10_000)
    realized = float(source_fs_hz) * ratio.numerator / ratio.denominator
    if not np.isclose(realized, float(target_fs_hz), atol=1e-12, rtol=0.0):
        raise ValueError("target/source rate ratio is not exactly representable")
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
    "DlResampleResult",
    "SynchronizedResampleResult",
    "V2_DL_RESAMPLING_TARGETS_HZ",
    "resample_dl_view",
    "resample_synchronized_channels",
    "validate_dl_resampling_config",
]

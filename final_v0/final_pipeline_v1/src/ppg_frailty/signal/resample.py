"""DL-only 抗混叠重采样 / Anti-aliased DL-only resampling.

中文：采集、峰时序、形态学和 feature audit 始终保留 400 Hz；本模块只创建
具备显式 provenance 的独立 DL view，绝不覆写原数组或伪装采样率。
English: Acquisition, peak timing, morphology and feature audit remain at 400 Hz.
This module creates a separate provenance-carrying DL view and never mutates input.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal


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
        raise ValueError("V1 DL resampling source must remain the canonical 400 Hz grid")
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


__all__ = ["DlResampleResult", "resample_dl_view"]

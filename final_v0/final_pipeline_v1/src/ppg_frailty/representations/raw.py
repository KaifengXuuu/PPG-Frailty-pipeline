"""Raw 多通道窗口构建 / Raw multichannel window construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..signal.views import CANONICAL_FS_HZ, CanonicalSignalViews, WindowPlan


@dataclass(frozen=True)
class RawWindows:
    """固定 [N,8,T] 数据与有效 mask / Fixed windows and validity masks."""

    values: np.ndarray
    valid_mask: np.ndarray
    start_samples: np.ndarray
    candidate_count: int
    dropped_invalid_count: int


def _robust_scale(values: np.ndarray) -> np.ndarray:
    """按通道 median/IQR，无信息通道回退 1 / Channel median/IQR scaling."""

    center = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    scale = q75 - q25
    mad = np.median(np.abs(values - center), axis=0) * 1.4826
    scale = np.where(scale > 1e-8, scale, np.where(mad > 1e-8, mad, 1.0))
    return (values - center) / scale


def build_raw_windows(views: CanonicalSignalViews, plan: WindowPlan) -> RawWindows:
    """按唯一计划生成 8 通道窗口 / Build 8-channel windows from the sole plan.

    中文：PPG 从合法 analysis route 取值；IMU 取 dynamic acceleration 与 gyro。
    所有参考窗口必须完整，避免 padding 被解释为生理零值。
    English: PPG comes from the legal analysis route; IMU uses dynamic acceleration
    and gyro. Reference windows must be complete.
    """

    views.validate()
    if plan.source_record_id != str(views.metadata.get("record_id", "")):
        raise ValueError("WindowPlan source_record_id must match signal metadata")
    windows = plan.plan(views.x_filter.shape[0], CANONICAL_FS_HZ)
    dynamic = np.asarray(views.imu_processed["dynamic_acc_mps2"], dtype=np.float64)
    gyro = np.asarray(views.imu_processed["gyro_rads"], dtype=np.float64)
    matrix = np.column_stack((views.analysis_signal, dynamic, gyro))
    if matrix.shape != (views.x_filter.shape[0], 8):
        raise ValueError("raw route requires aligned [RED,IR,AX,AY,AZ,GX,GY,GZ]")
    imu_valid = np.asarray(
        views.imu_processed.get("imu_valid_mask", np.ones(matrix.shape[0], dtype=bool)),
        dtype=bool,
    )
    if imu_valid.shape != (matrix.shape[0],):
        raise ValueError("imu_valid_mask must align with raw samples")
    rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    starts: list[int] = []
    dropped_invalid = 0
    for item in windows:
        if item.valid_length != item.window_length or bool(np.any(item.padding_mask)):
            raise ValueError("reference raw route does not accept padded windows")
        segment = matrix[item.start_sample:item.end_sample]
        valid_rows = imu_valid[item.start_sample:item.end_sample]
        # 中文：EKF 在线初始化的无估计样本只能导致该窗口 drop，不能填零或让整条
        # recording 因窗口外的启动样本失败。English: no-estimate EKF startup rows
        # drop the affected window; they are never zero-imputed and do not invalidate
        # otherwise complete later windows.
        if not np.all(valid_rows) or not np.isfinite(segment).all():
            dropped_invalid += 1
            continue
        rows.append(_robust_scale(segment).T.astype(np.float32))
        masks.append(~np.asarray(item.padding_mask, dtype=bool))
        starts.append(item.start_sample)
    if not rows:
        raise ValueError("window plan produced no complete raw windows")
    return RawWindows(
        np.stack(rows),
        np.stack(masks),
        np.asarray(starts, dtype=np.int64),
        len(windows),
        dropped_invalid,
    )


__all__ = ["RawWindows", "build_raw_windows"]

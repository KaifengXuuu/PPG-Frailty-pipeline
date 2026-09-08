"""Raw 多通道窗口构建 / Raw multichannel window construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Any

import numpy as np

from ..normalization import (
    FALLBACK_MAD,
    FALLBACK_ONE,
    IMU_NONE,
    PPG_NONE,
    PPG_ROBUST,
    PPG_STANDARD_ZSCORE,
    RawNormalizationConfig,
)
from ..signal.views import CANONICAL_FS_HZ, CanonicalSignalViews, WindowPlan


@dataclass(frozen=True)
class RawWindows:
    """固定 [N,8,T] 数据与有效 mask / Fixed windows and validity masks."""

    values: np.ndarray
    valid_mask: np.ndarray
    start_samples: np.ndarray
    candidate_count: int
    dropped_invalid_count: int
    provenance: dict[str, object] = field(default_factory=dict)
    window_quality_scores: np.ndarray | None = None
    window_aggregation_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate optional row-aligned score and aggregation-selection vectors.

        The vectors are deliberately separate from ``provenance``: downstream
        prediction identities need their actual values, while persisted
        provenance stores only content hashes and counts.
        """

        if self.window_quality_scores is not None:
            scores = np.asarray(self.window_quality_scores, dtype=np.float32)
            if scores.shape != (int(np.asarray(self.values).shape[0]), ):
                raise ValueError("window_quality_scores must contain one value per raw window")
            if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
                raise ValueError("window_quality_scores must be finite in [0,1]")
            object.__setattr__(self, "window_quality_scores", scores)
        if self.window_aggregation_mask is None:
            return
        aggregation_mask = np.asarray(self.window_aggregation_mask, dtype=bool)
        if aggregation_mask.shape != (int(np.asarray(self.values).shape[0]), ):
            raise ValueError("window_aggregation_mask must contain one flag per raw window")
        object.__setattr__(self, "window_aggregation_mask", aggregation_mask)


def _fallback_channel_scale(
    values: np.ndarray,
    *,
    config: RawNormalizationConfig,
) -> np.ndarray:
    """Return the configured fallback scale for every DL input channel."""

    if config.iqr_fallback == FALLBACK_ONE:
        return np.ones(values.shape[1], dtype=np.float64)
    if config.iqr_fallback == FALLBACK_MAD:
        median = np.median(values, axis=0)
        return np.median(np.abs(values - median), axis=0) / float(config.mad_consistency_divisor)
    if values.shape[0] <= config.standard_ddof:
        return np.full(values.shape[1], np.nan, dtype=np.float64)
    return np.std(values, axis=0, ddof=config.standard_ddof)


def _normalize_dl_window(
    values: np.ndarray,
    *,
    config: RawNormalizationConfig,
) -> np.ndarray:
    """Normalize every channel of one DL window without mutating source views."""

    source = np.asarray(values, dtype=np.float64)
    if config.raw_ppg == PPG_NONE:
        return source.copy()
    if config.raw_ppg == PPG_STANDARD_ZSCORE:
        center = np.mean(source, axis=0)
        scale = (np.std(source, axis=0, ddof=config.standard_ddof)
                 if source.shape[0] > config.standard_ddof else np.full(source.shape[1], np.nan, dtype=np.float64))
    elif config.raw_ppg == PPG_ROBUST:
        center = np.median(source, axis=0)
        q25, q75 = np.percentile(source, [25.0, 75.0], axis=0)
        scale = (q75 - q25) / float(config.robust_iqr_divisor)
        fallback = _fallback_channel_scale(source, config=config)
        scale = np.where(
            np.isfinite(scale) & (scale > config.scale_epsilon),
            scale,
            fallback,
        )
    else:  # defensive: RawNormalizationConfig owns strategy registration.
        raise ValueError(f"unsupported raw PPG normalization: {config.raw_ppg}")

    scale = np.where(
        np.isfinite(scale) & (scale > config.scale_epsilon),
        scale,
        1.0,
    )
    normalized = (source - center) / scale
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    if config.clip_after_scale is not None:
        normalized = np.clip(normalized, *config.clip_after_scale)
    return normalized


def build_raw_windows(
    views: CanonicalSignalViews,
    plan: WindowPlan,
    *,
    normalization: Mapping[str, Any] | None = None,
) -> RawWindows:
    """按唯一计划生成 8 通道窗口 / Build 8-channel windows from the sole plan.

    中文：PPG 始终从 amplitude-preserving x_filter 取值；非恒等 x_ar 仅可进入
    rate/PPI/PRV 路径。IMU 从 physical-unit processed view 复制 dynamic
    acceleration 与 gyro。八通道仅在 DL 窗口副本内逐通道归一化，上游视图
    不会被覆盖。完整窗口是默认；显式启用 padding 时只在有效前缀上归一化，
    padded 后缀保持中性零值并由 valid_mask 从模型池化和 fold 变换中排除。
    English: PPG comes from the amplitude-preserving direct analysis view; IMU
    comes from the physical dynamic-acceleration and gyroscope view. Complete
    windows are the default; explicitly padded samples are neutral zeros
    accompanied by a false validity mask.
    """

    views.validate()
    normalization_config = RawNormalizationConfig.from_mapping(normalization)
    if plan.source_record_id != str(views.metadata.get("record_id", "")):
        raise ValueError("WindowPlan source_record_id must match signal metadata")
    windows = plan.plan(views.x_filter.shape[0], CANONICAL_FS_HZ)
    dynamic = np.asarray(views.imu_processed["dynamic_acc_mps2"], dtype=np.float64)
    gyro = np.asarray(views.imu_processed["gyro_rads"], dtype=np.float64)
    matrix = np.column_stack((
        views.x_analysis,
        dynamic,
        gyro,
    ))
    if matrix.shape != (views.x_filter.shape[0], 8):
        raise ValueError("raw route requires aligned " "[RED,IR,A_dyn_x,A_dyn_y,A_dyn_z,GX,GY,GZ]")
    imu_valid = np.asarray(
        views.imu_processed.get("imu_valid_mask", np.ones(matrix.shape[0], dtype=bool)),
        dtype=bool,
    )
    if imu_valid.shape != (matrix.shape[0], ):
        raise ValueError("imu_valid_mask must align with raw samples")
    rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    starts: list[int] = []
    dropped_invalid = 0
    for item in windows:
        segment = matrix[item.start_sample:item.end_sample]
        valid_rows = imu_valid[item.start_sample:item.end_sample]
        # 中文：EKF 在线初始化的无估计样本只能导致该窗口 drop，不能填零或让整条
        # recording 因窗口外的启动样本失败。English: no-estimate EKF startup rows
        # drop the affected window; they are never zero-imputed and do not invalidate
        # otherwise complete later windows.
        if not np.all(valid_rows) or not np.isfinite(segment).all():
            dropped_invalid += 1
            continue
        # This temporary matrix is the sole DL branch. CanonicalSignalViews
        # retains amplitude-preserving PPG and physical-unit processed IMU.
        normalized = np.zeros((item.window_length, matrix.shape[1]), dtype=np.float64)
        normalized[:item.valid_length] = _normalize_dl_window(segment, config=normalization_config)
        rows.append(normalized.T.astype(np.float32))
        masks.append(~np.asarray(item.padding_mask, dtype=bool))
        starts.append(item.start_sample)
    if not rows:
        raise ValueError("window plan produced no valid raw windows")
    return RawWindows(
        values=np.stack(rows),
        valid_mask=np.stack(masks),
        start_samples=np.asarray(starts, dtype=np.int64),
        candidate_count=len(windows),
        dropped_invalid_count=dropped_invalid,
        provenance={
            "ppg_source_view":
            "x_filter",
            "analysis_source_view":
            "x_analysis_amplitude_preserving",
            "processed_imu_source_view":
            "processed_imu_physical",
            "dl_tensor_view":
            "x_dl_all8_window_norm",
            "non_identity_x_ar_eligible_for_raw_predictor":
            False,
            "normalization_config":
            normalization_config.to_mapping(),
            "dl_all8_normalization":
            normalization_config.raw_ppg,
            "ppg_normalization":
            normalization_config.raw_ppg,
            "dl_normalized_channels": ([] if normalization_config.raw_ppg == PPG_NONE else [
                "RED",
                "IR",
                "A_dyn_x",
                "A_dyn_y",
                "A_dyn_z",
                "GX",
                "GY",
                "GZ",
            ]),
            "ppg_clip_after_scale":
            (None if normalization_config.raw_ppg == PPG_NONE or normalization_config.clip_after_scale is None else
             list(normalization_config.clip_after_scale)),
            "dl_clip_after_scale":
            (None if normalization_config.raw_ppg == PPG_NONE or normalization_config.clip_after_scale is None else
             list(normalization_config.clip_after_scale)),
            "imu_normalization": ("all8_per_window_then_no_post_transform" if normalization_config.raw_imu == IMU_NONE
                                  else "all8_per_window_then_explicit_legacy_outer_train_transform"),
            "imu_normalization_requested":
            normalization_config.raw_imu,
            "imu_channel_schema": [
                "A_dyn_x",
                "A_dyn_y",
                "A_dyn_z",
                "GX",
                "GY",
                "GZ",
            ],
            "imu_units": [
                "m/s^2",
                "m/s^2",
                "m/s^2",
                "rad/s",
                "rad/s",
                "rad/s",
            ],
            "derived_motion_channels_in_frailty_tensor":
            False,
            "padding_policy":
            ("valid_prefix_neutral_zero_with_explicit_mask" if any(not bool(mask.all())
                                                                   for mask in masks) else "complete_windows"),
            "minimum_valid_fraction":
            float(plan.min_valid_fraction),
        },
    )


__all__ = ["RawWindows", "build_raw_windows"]

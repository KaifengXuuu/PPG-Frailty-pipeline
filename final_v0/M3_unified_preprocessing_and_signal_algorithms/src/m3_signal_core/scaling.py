"""仅训练折拟合的缩放与 raw8 混合视图 / Training-fold-only scaling and raw8 view.

中文：PPG 形状使用可逆逐窗 robust normalization，同时另存原始幅值；IMU 的
物理强度通过训练折拟合 scaler 保留。任何非 training 角色调用 fit 都直接失败。

English: PPG shape uses reversible per-window robust normalization while raw amplitude
is retained separately. IMU physical intensity is preserved through a scaler fitted
only on the training fold. Fit calls from any non-training role fail immediately.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _stable_id_hash(ids: Sequence[str]) -> str:
    """哈希稳定排序的训练 ID / Hash stable-sorted training identifiers."""

    payload = "\n".join(sorted(str(value) for value in ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class FoldScaler:
    """训练折 median imputer + robust/standard scaler / Fold-only feature scaler."""

    method: str = "robust"
    clip: float | None = None
    impute_values: np.ndarray | None = None
    center: np.ndarray | None = None
    scale: np.ndarray | None = None
    zero_scale_mask: np.ndarray | None = None
    training_ids_sha256: str = ""
    fit_role: str = ""

    def fit(
        self,
        values: np.ndarray,
        *,
        fit_role: str,
        training_ids: Sequence[str],
    ) -> "FoldScaler":
        """仅接受 training role / Fit only when the caller declares training role."""

        if fit_role != "training":
            raise ValueError("FoldScaler.fit is restricted to fit_role='training'")
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("values must be a non-empty samples×features matrix")
        finite = np.where(np.isfinite(x), x, np.nan)
        impute = np.nanmedian(finite, axis=0)
        if not np.isfinite(impute).all():
            raise ValueError("at least one feature is entirely non-finite in training")
        filled = np.where(np.isfinite(x), x, impute)
        if self.method == "robust":
            q25, q75 = np.percentile(filled, [25.0, 75.0], axis=0)
            center = np.median(filled, axis=0)
            scale = (q75 - q25) / 1.349
        elif self.method == "standard":
            center = np.mean(filled, axis=0)
            scale = np.std(filled, axis=0, ddof=0)
        else:
            raise ValueError("method must be 'robust' or 'standard'")
        zero_scale_mask = ~(np.isfinite(scale) & (np.abs(scale) > 1e-12))
        scale = np.where(~zero_scale_mask, scale, 1.0)
        self.impute_values = np.asarray(impute, dtype=np.float64)
        self.center = np.asarray(center, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.zero_scale_mask = np.asarray(zero_scale_mask, dtype=bool)
        self.training_ids_sha256 = _stable_id_hash(training_ids)
        self.fit_role = fit_role
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """使用冻结统计量变换 / Transform with frozen fold statistics."""

        if self.center is None or self.scale is None or self.impute_values is None:
            raise RuntimeError("FoldScaler must be fitted before transform")
        x = np.asarray(values, dtype=np.float64)
        if x.shape[-1] != self.center.size:
            raise ValueError("feature count does not match fitted scaler")
        filled = np.where(np.isfinite(x), x, self.impute_values)
        output = (filled - self.center) / self.scale
        if self.clip is not None:
            output = np.clip(output, -float(self.clip), float(self.clip))
        return np.asarray(output, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """序列化冻结统计量 / Serialize frozen statistics."""

        if self.center is None or self.scale is None or self.impute_values is None:
            raise RuntimeError("cannot serialize an unfitted scaler")
        return {
            "method": self.method,
            "clip": self.clip,
            "impute_values": self.impute_values.tolist(),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "zero_scale_mask": self.zero_scale_mask.tolist(),
            "training_ids_sha256": self.training_ids_sha256,
            "fit_role": self.fit_role,
        }


@dataclass
class FoldAmplitudeRiskModel:
    """训练折拟合的 PPG 振幅 SQI 风险门 / Fold-fitted amplitude-risk gate."""

    threshold_abs_robust_z: float = 6.0
    feature_order: tuple[str, ...] = ("log_ac", "log_abs_dc", "log_acdc")
    center: np.ndarray | None = None
    scale: np.ndarray | None = None
    training_ids_sha256: str = ""

    def fit(
        self,
        raw_feature_rows: np.ndarray,
        *,
        fit_role: str,
        training_ids: Sequence[str],
    ) -> "FoldAmplitudeRiskModel":
        """仅用 training fold 拟合 median/MAD-IQR scale / Fit on training only."""

        if fit_role != "training":
            raise ValueError("amplitude risk fit is restricted to training")
        values = np.asarray(raw_feature_rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_order):
            raise ValueError("raw_feature_rows must match feature_order")
        if not np.isfinite(values).all():
            raise ValueError("amplitude-risk training features must be finite")
        center = np.median(values, axis=0)
        mad_scale = 1.4826 * np.median(np.abs(values - center), axis=0)
        q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
        iqr_scale = (q75 - q25) / 1.349
        scale = np.maximum.reduce(
            [mad_scale, iqr_scale, np.full(center.shape, 1e-12)]
        )
        self.center = center.astype(np.float64)
        self.scale = scale.astype(np.float64)
        self.training_ids_sha256 = _stable_id_hash(training_ids)
        return self

    def evaluate(self, raw_feature_row: np.ndarray) -> dict[str, Any]:
        """输出 z 与风险，不把 heuristic 当作饱和真值 / Return a risk-only flag."""

        if self.center is None or self.scale is None:
            raise RuntimeError("FoldAmplitudeRiskModel must be fitted before evaluate")
        values = np.asarray(raw_feature_row, dtype=np.float64)
        if values.shape != self.center.shape or not np.isfinite(values).all():
            return {
                "status": "unavailable",
                "reason": "amplitude_features_invalid",
                "robust_z": None,
                "sqi_risk": True,
            }
        robust_z = (values - self.center) / self.scale
        return {
            "status": "evaluated",
            "reason": "amplitude_outlier_training_fold"
            if np.any(np.abs(robust_z) > self.threshold_abs_robust_z)
            else "within_training_fold_range",
            "robust_z": robust_z.tolist(),
            "sqi_risk": bool(np.any(np.abs(robust_z) > self.threshold_abs_robust_z)),
        }

    def to_dict(self) -> dict[str, Any]:
        """序列化可追溯 artifact / Serialize a traceable fold artifact."""

        if self.center is None or self.scale is None:
            raise RuntimeError("cannot serialize an unfitted amplitude model")
        return {
            "threshold_abs_robust_z": self.threshold_abs_robust_z,
            "feature_order": list(self.feature_order),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "training_ids_sha256": self.training_ids_sha256,
        }


def robust_window_scale(
    values: np.ndarray,
    *,
    clip: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐通道可逆 robust normalization / Reversible per-channel window scaling."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("values must be a non-empty samples×channels matrix")
    if not np.isfinite(x).all():
        raise ValueError("quality repair must run before window scaling")
    center = np.median(x, axis=0)
    q25, q75 = np.percentile(x, [25.0, 75.0], axis=0)
    scale = (q75 - q25) / 1.349
    if np.any(scale <= 1e-12):
        raise ValueError("zero_iqr_channel_requires_no_estimate")
    normalized = (x - center) / scale
    if clip is not None:
        normalized = np.clip(normalized, -float(clip), float(clip))
    return normalized.astype(np.float64), center.astype(np.float64), scale.astype(np.float64)


def build_raw8_model_view(
    ppg_red_ir: np.ndarray,
    imu_si: np.ndarray,
    imu_fold_scaler: FoldScaler,
) -> tuple[np.ndarray, dict[str, list[float] | list[str]]]:
    """构建 dynamic-acc raw8 模型视图 / Build the dynamic-acceleration raw8 view."""

    ppg = np.asarray(ppg_red_ir, dtype=np.float64)
    imu = np.asarray(imu_si, dtype=np.float64)
    if (
        imu_fold_scaler.method != "robust"
        or imu_fold_scaler.clip is not None
        or imu_fold_scaler.fit_role != "training"
    ):
        raise ValueError("raw8_dynamic_view_requires_frozen_training_robust_scaler")
    if ppg.ndim != 2 or ppg.shape[1] != 2:
        raise ValueError("ppg_red_ir must have exactly two channels")
    if imu.ndim != 2 or imu.shape[1] != 6 or imu.shape[0] != ppg.shape[0]:
        raise ValueError("imu_si must align with PPG and have six channels")
    ppg_view, center, scale = robust_window_scale(ppg)
    imu_view = imu_fold_scaler.transform(imu)
    combined = np.column_stack([ppg_view, imu_view]).astype(np.float64)
    metadata: dict[str, list[float] | list[str]] = {
        "feature_schema_version": ["m3_raw8_dynamic_sequence.v1"],
        "ppg_window_center": center.tolist(),
        "ppg_window_scale": scale.tolist(),
        "channel_order": ["RED", "IR", "AX_dyn", "AY_dyn", "AZ_dyn", "GX", "GY", "GZ"],
    }
    return combined, metadata

"""V2 outer-train-only vector transform / V2 外层训练折专用向量变换.

English: One fitted artifact serves ordered matrices and finite fusion tensors.
中文: 同一拟合产物同时服务有序矩阵与有限值融合张量, 避免数据泄漏或隐式丢失
validity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..contracts import FeatureVectorV1
from ..provenance import assert_training_only
from ..representations.feature_vector import validate_feature_vector
from .registry import default_registry, registry_for_feature_names


TRANSFORM_SCHEMA_VERSION = "feature_vector_outer_train_median_iqr_v2"
def fusion_tensor_schema_version(feature_names: Sequence[str]) -> str:
    """Return a width- and registry-bound fusion tensor identity."""

    registry = registry_for_feature_names(feature_names)
    width = 2 * len(registry.names)
    return (
        f"feature_vector_values_plus_validity_{width}_"
        f"registry-{registry.sha256[:12]}_v4"
    )


FUSION_TENSOR_SCHEMA_VERSION = fusion_tensor_schema_version(default_registry().names)


def _artifact_hash(
    center: np.ndarray,
    scale: np.ndarray,
    valid_count: np.ndarray,
    feature_names: Sequence[str],
    fitted_ids: Sequence[str],
    registry_sha256: str,
) -> str:
    payload = {
        "schema_version": TRANSFORM_SCHEMA_VERSION,
        "center": np.asarray(center, dtype=np.float64).tolist(),
        "scale": np.asarray(scale, dtype=np.float64).tolist(),
        "valid_count": np.asarray(valid_count, dtype=np.int64).tolist(),
        "feature_names": list(feature_names),
        "fitted_on_participant_ids": list(fitted_ids),
        "registry_sha256": registry_sha256,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FoldFeatureVectorTransform:
    """Median/IQR statistics fitted only on declared outer-train participants."""

    center: np.ndarray
    scale: np.ndarray
    valid_count: np.ndarray
    feature_names: tuple[str, ...]
    fitted_on_participant_ids: tuple[str, ...]
    registry_sha256: str
    artifact_sha256: str
    schema_version: str = TRANSFORM_SCHEMA_VERSION

    def validate(self) -> None:
        width = len(self.feature_names)
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        count = np.asarray(self.valid_count, dtype=np.int64)
        if center.shape != (width,) or scale.shape != (width,) or count.shape != (width,):
            raise ValueError("feature-vector transform arrays lost registry alignment")
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("feature-vector transform statistics must be finite")
        if np.any(scale <= 0.0) or np.any(count < 0):
            raise ValueError("feature-vector transform scale/count is invalid")
        if not self.fitted_on_participant_ids:
            raise ValueError("feature-vector transform requires an outer-train roster")
        registry = registry_for_feature_names(self.feature_names)
        if self.feature_names != registry.names or self.registry_sha256 != registry.sha256:
            raise ValueError("feature-vector transform registry identity drift")
        expected = _artifact_hash(
            center,
            scale,
            count,
            self.feature_names,
            self.fitted_on_participant_ids,
            self.registry_sha256,
        )
        if self.schema_version != TRANSFORM_SCHEMA_VERSION or self.artifact_sha256 != expected:
            raise ValueError("feature-vector transform artifact identity drift")


@dataclass(frozen=True)
class FoldTransformedFeatureBatch:
    """Matrix contexts plus one finite value/validity tensor for fusion."""

    contexts: tuple[FeatureVectorV1, ...]
    fusion_tensor: np.ndarray
    tensor_schema: tuple[str, ...]
    provenance: dict[str, object]
    schema_version: str = ""

    def validate(self) -> None:
        tensor = np.asarray(self.fusion_tensor, dtype=np.float64)
        if tensor.ndim != 2 or tensor.shape[0] != len(self.contexts):
            raise ValueError("fusion feature tensor must be rows-by-channels")
        if tensor.shape[1] != len(self.tensor_schema) or not np.isfinite(tensor).all():
            raise ValueError("fusion feature tensor width/finite contract failed")
        if not self.contexts:
            raise ValueError("transformed feature batch cannot be empty")
        for context in self.contexts:
            validate_feature_vector(context)
            if context.provenance.get("fold_standardized") is not True:
                raise ValueError("batch context is not fold standardized")
        registry = registry_for_feature_names(self.contexts[0].feature_names)
        expected_schema = registry.names + tuple(
            f"{name}.validity" for name in registry.names
        )
        if (
            self.schema_version != fusion_tensor_schema_version(registry.names)
            or tuple(self.tensor_schema) != expected_schema
            or self.provenance.get("registry_sha256") != registry.sha256
            or self.provenance.get("fold_standardized") is not True
            or self.provenance.get("feature_vector_transform_schema")
            != TRANSFORM_SCHEMA_VERSION
        ):
            raise ValueError("stale or inconsistent fusion feature tensor schema")


def fit_fold_feature_vector_transform(
    vectors: Iterable[FeatureVectorV1],
    participant_ids: Iterable[str],
    *,
    fitted_on_participant_ids: Iterable[str],
    outer_train_participant_ids: Iterable[str],
    outer_oof_participant_ids: Iterable[str],
) -> FoldFeatureVectorTransform:
    """Fit per-feature median/IQR with MAD/one fallback on outer-train rows only."""

    fitted = assert_training_only(
        fitted_on_participant_ids,
        outer_train_participant_ids,
        outer_oof_participant_ids,
    )
    items = tuple(vectors)
    ids = tuple(str(value) for value in participant_ids)
    if len(items) != len(ids) or not items:
        raise ValueError("vectors and participant_ids must be non-empty and aligned")
    registry = registry_for_feature_names(items[0].feature_names)
    for vector in items:
        validate_feature_vector(vector)
        if tuple(vector.feature_names) != registry.names:
            raise ValueError("feature-vector registry order differs across rows")
        if vector.provenance.get("registry_sha256") != registry.sha256:
            raise ValueError("feature-vector registry hash drift")
    selected = [vector for vector, participant in zip(items, ids) if participant in set(fitted)]
    selected_ids = {participant for participant in ids if participant in set(fitted)}
    if selected_ids != set(fitted) or not selected:
        raise ValueError("not every declared fitted participant has a feature vector")
    width = len(registry.names)
    center = np.zeros(width, dtype=np.float64)
    scale = np.ones(width, dtype=np.float64)
    valid_count = np.zeros(width, dtype=np.int64)
    for column in range(width):
        values = np.asarray(
            [
                vector.values[column]
                for vector in selected
                if vector.validity[column] and np.isfinite(vector.values[column])
            ],
            dtype=np.float64,
        )
        valid_count[column] = values.size
        if not values.size:
            continue
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25.0, 75.0])
        robust_scale = float(q75 - q25)
        if robust_scale <= 1e-12:
            robust_scale = float(1.4826 * np.median(np.abs(values - median)))
        center[column] = median
        scale[column] = robust_scale if robust_scale > 1e-12 else 1.0
    artifact_sha256 = _artifact_hash(
        center,
        scale,
        valid_count,
        registry.names,
        fitted,
        registry.sha256,
    )
    artifact = FoldFeatureVectorTransform(
        center=center,
        scale=scale,
        valid_count=valid_count,
        feature_names=registry.names,
        fitted_on_participant_ids=fitted,
        registry_sha256=registry.sha256,
        artifact_sha256=artifact_sha256,
    )
    artifact.validate()
    return artifact


def transform_feature_vector(
    vector: FeatureVectorV1,
    transform: FoldFeatureVectorTransform,
) -> FeatureVectorV1:
    """Standardize valid values while preserving NaN/false unavailable slots."""

    validate_feature_vector(vector)
    transform.validate()
    if tuple(vector.feature_names) != transform.feature_names:
        raise ValueError("feature vector and transform schemas differ")
    if vector.provenance.get("registry_sha256") != transform.registry_sha256:
        raise ValueError("feature vector and transform registry hashes differ")
    valid = np.asarray(vector.validity, dtype=bool) & np.isfinite(vector.values)
    values = np.full(len(transform.feature_names), np.nan, dtype=np.float64)
    values[valid] = (
        np.asarray(vector.values, dtype=np.float64)[valid] - transform.center[valid]
    ) / transform.scale[valid]
    provenance = dict(vector.provenance)
    provenance.update(
        {
            "fold_standardized": True,
            "feature_vector_transform_schema": transform.schema_version,
            "feature_vector_transform_sha256": transform.artifact_sha256,
            "feature_vector_transform_fitted_on_participant_ids": list(
                transform.fitted_on_participant_ids
            ),
            "standardization": "outer_train_median_iqr_mad_then_one",
            "unavailable_value_policy": "NaN_with_validity_false",
        }
    )
    result = FeatureVectorV1(
        values=values,
        validity=valid.copy(),
        feature_names=transform.feature_names,
        schema_version=vector.schema_version + "+fold_vector_robust_v2",
        provenance=provenance,
    )
    return validate_feature_vector(result)


def transform_feature_vector_batch(
    vectors: Iterable[FeatureVectorV1],
    transform: FoldFeatureVectorTransform,
) -> FoldTransformedFeatureBatch:
    """Create transformed matrix contexts and a finite paired tensor for fusion."""

    contexts = tuple(transform_feature_vector(vector, transform) for vector in vectors)
    if not contexts:
        raise ValueError("at least one feature vector is required")
    values = np.vstack(
        [
            np.where(context.validity, context.values, 0.0)
            for context in contexts
        ]
    )
    validity = np.vstack([context.validity.astype(np.float64) for context in contexts])
    tensor = np.column_stack((values, validity))
    schema = transform.feature_names + tuple(
        f"{name}.validity" for name in transform.feature_names
    )
    batch = FoldTransformedFeatureBatch(
        contexts=contexts,
        fusion_tensor=tensor,
        tensor_schema=schema,
        provenance={
            "fold_standardized": True,
            "feature_vector_transform_sha256": transform.artifact_sha256,
            "feature_vector_transform_schema": transform.schema_version,
            "fitted_on_participant_ids": list(transform.fitted_on_participant_ids),
            "validity_encoding": "paired_explicit_0_1_channels_v2",
            "invalid_value_encoding": "neutral_zero",
            "registry_sha256": transform.registry_sha256,
        },
        schema_version=fusion_tensor_schema_version(transform.feature_names),
    )
    batch.validate()
    return batch


__all__ = [
    "FUSION_TENSOR_SCHEMA_VERSION",
    "fusion_tensor_schema_version",
    "TRANSFORM_SCHEMA_VERSION",
    "FoldFeatureVectorTransform",
    "FoldTransformedFeatureBatch",
    "fit_fold_feature_vector_transform",
    "transform_feature_vector",
    "transform_feature_vector_batch",
]

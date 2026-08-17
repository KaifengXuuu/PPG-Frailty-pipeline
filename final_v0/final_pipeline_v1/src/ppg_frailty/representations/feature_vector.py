"""FeatureVectorV1 合同检查 / FeatureVectorV1 contract validation."""

import numpy as np

from ..contracts import FeatureVectorV1


def validate_feature_vector(vector: FeatureVectorV1) -> FeatureVectorV1:
    """拒绝宽度漂移与伪造有效零 / Reject width drift and invalid encodings."""

    values = np.asarray(vector.values, dtype=np.float64)
    validity = np.asarray(vector.validity, dtype=bool)
    if values.ndim != 1 or validity.shape != values.shape or len(vector.feature_names) != values.size:
        raise ValueError("FeatureVectorV1 values/validity/names must align")
    if len(vector.feature_names) != len(set(vector.feature_names)):
        raise ValueError("FeatureVectorV1 names must be unique")
    if np.any(validity & ~np.isfinite(values)):
        raise ValueError("valid physiological features must be finite")
    if np.any((~validity) & np.isfinite(values)):
        raise ValueError("unavailable physiological features must be NaN")
    return vector


__all__ = ["validate_feature_vector"]

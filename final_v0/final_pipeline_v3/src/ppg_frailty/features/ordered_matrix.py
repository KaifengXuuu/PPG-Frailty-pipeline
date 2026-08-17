"""有序特征矩阵的规范路径 / Canonical ordered-matrix path.

中文：绑定冻结 registry 与 typed matrix。English: Bind the frozen registry to the typed ordered matrix.
"""

from ..contracts import EngineeringFeatureSequence, OrderedFeatureMatrixV1
from .registry import FeatureRegistry, build_ordered_matrix, default_registry

__all__ = [
    "EngineeringFeatureSequence",
    "FeatureRegistry",
    "OrderedFeatureMatrixV1",
    "build_ordered_matrix",
    "default_registry",
]

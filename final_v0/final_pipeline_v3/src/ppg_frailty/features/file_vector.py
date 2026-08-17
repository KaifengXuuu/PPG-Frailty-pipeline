"""文件级向量的规范路径 / Canonical file-vector path.

中文：绑定冻结注册表与 typed vector。English: Bind the frozen registry to the typed file vector.
"""

from ..contracts import FeatureVectorV1
from .registry import FeatureRegistry, build_feature_vector, default_registry, summarize_engineering

__all__ = [
    "FeatureRegistry",
    "FeatureVectorV1",
    "build_feature_vector",
    "default_registry",
    "summarize_engineering",
]

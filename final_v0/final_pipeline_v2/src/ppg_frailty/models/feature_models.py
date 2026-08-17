"""文件向量基线模型规范门面 / Canonical feature-model facade.

中文：复用 fold-local imputer/scaler 基线。English: Re-export the fold-local feature baseline.
"""

from .feature_baselines import FeatureVectorBaseline

__all__ = ["FeatureVectorBaseline"]

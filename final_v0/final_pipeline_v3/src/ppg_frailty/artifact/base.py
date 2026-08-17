"""ArtifactReducer 规范门面 / Canonical ArtifactReducer facade.

中文：复用唯一 reducer 合同与结果校验。English: Reuse the sole reducer contract and validator.
"""

from ..artifacts.base import ArtifactReducer, validate_result

__all__ = ["ArtifactReducer", "validate_result"]

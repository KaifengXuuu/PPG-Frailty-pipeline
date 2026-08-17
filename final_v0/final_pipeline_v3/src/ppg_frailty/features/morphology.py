"""形态学特征规范门面 / Canonical morphology-feature facade.

中文：复用 direct-route 形态学守卫。English: Re-export morphology with its direct-route guard.
"""

from ..signal.morphology import MorphologyResult, extract_morphology, require_direct_route

__all__ = ["MorphologyResult", "extract_morphology", "require_direct_route"]

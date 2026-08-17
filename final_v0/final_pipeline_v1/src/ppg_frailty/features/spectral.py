"""谱特征的规范组合门面 / Canonical spectral-feature composition facade.

中文：谱计算仍由已测试的 PRV 与 engineering 实现负责，本门面不复制 Welch
细节。English: Tested PRV/engineering implementations retain spectral authority.
"""

from ..signal.prv import PrvResult, compute_prv
from .engineering import EngineeringExtraction, extract_engineering_features

__all__ = [
    "EngineeringExtraction",
    "PrvResult",
    "compute_prv",
    "extract_engineering_features",
]

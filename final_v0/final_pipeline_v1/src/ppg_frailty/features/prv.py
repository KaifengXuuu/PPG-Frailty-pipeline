"""PRV 特征规范门面 / Canonical PRV-feature facade.

中文：复用时域与频域 PRV 算法。English: Re-export the tested time- and frequency-domain PRV algorithm.
"""

from ..signal.prv import PrvResult, compute_prv

__all__ = ["PrvResult", "compute_prv"]

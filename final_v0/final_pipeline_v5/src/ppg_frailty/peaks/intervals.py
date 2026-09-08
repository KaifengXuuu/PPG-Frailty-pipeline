"""PPI/PRV 规范门面 / Canonical PPI/PRV facade.

中文：复用相邻峰间隔算法。English: Re-export the tested adjacent-peak interval algorithm.
"""

from ..signal.prv import PrvResult, compute_prv

__all__ = ["PrvResult", "compute_prv"]

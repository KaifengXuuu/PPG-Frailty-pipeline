"""非平稳 SSA 门面 / Non-stationary SSA facade.

中文：重导出已测试 SSA reducer。English: Re-export the tested SSA reducer without copying it.
"""

from ..artifacts.decomposition import SsaConfig, SsaReducer

__all__ = ["SsaConfig", "SsaReducer"]

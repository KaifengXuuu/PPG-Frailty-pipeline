"""完整性校验 bundle 加载 / Integrity-checked bundle loading.

中文：直接复用唯一加载器。English: Re-export the sole hash-verifying bundle loader.
"""

from ..training.bundle import LoadedBundle, assert_golden_parity, load_bundle

__all__ = ["LoadedBundle", "assert_golden_parity", "load_bundle"]

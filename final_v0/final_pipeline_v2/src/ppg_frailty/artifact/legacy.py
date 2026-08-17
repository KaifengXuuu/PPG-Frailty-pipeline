"""V2 historical reducer facade / V2 历史 reducer 规范门面。

中文：实现保留在 plural artifacts 包；正式注册路径固定从本模块导入。
English: Implementations remain in artifacts while this is the stable public path.
"""

from ..artifacts.legacy import (
    CeemdLiteNlmsLegacyConfig,
    CeemdLiteNlmsLegacyReducer,
    DwtA2LegacyConfig,
    DwtA2LegacyReducer,
    EmdSiftingConfig,
    EmdSiftingRateOnlyReducer,
)

__all__ = [
    "CeemdLiteNlmsLegacyConfig",
    "CeemdLiteNlmsLegacyReducer",
    "DwtA2LegacyConfig",
    "DwtA2LegacyReducer",
    "EmdSiftingConfig",
    "EmdSiftingRateOnlyReducer",
]

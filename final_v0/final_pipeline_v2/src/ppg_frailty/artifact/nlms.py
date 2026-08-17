"""IMU-NLMS rate-only 门面 / IMU-NLMS rate-only facade.

中文：复用 ANC 比较实现及其配置。English: Re-export the ANC comparator and its strict config.
"""

from ..artifacts.nlms import NlmsConfig, NlmsReducer

__all__ = ["NlmsConfig", "NlmsReducer"]

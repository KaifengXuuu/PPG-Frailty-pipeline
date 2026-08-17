"""InceptionTime 单网络规范门面 / Canonical single-network InceptionTime facade.

中文：复用 full/small 单网络 port。English: Re-export the tested full and small single-network port.
"""

from .inception import (
    FullInceptionTimeSingleNetwork,
    InceptionBlock,
    InceptionModule,
    InceptionTimeSingleNetwork,
    SmallInceptionTimeSingleNetwork,
    masked_global_average,
)

__all__ = [
    "FullInceptionTimeSingleNetwork",
    "InceptionBlock",
    "InceptionModule",
    "InceptionTimeSingleNetwork",
    "SmallInceptionTimeSingleNetwork",
    "masked_global_average",
]

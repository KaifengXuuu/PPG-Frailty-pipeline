"""STFT-IMU 谱抑制门面 / STFT-IMU spectral suppression facade.

中文：复用正式谱掩蔽实现。English: Re-export the formal spectral-mask implementation.
"""

from ..artifacts.spectral import SpectralMaskConfig, SpectralMaskReducer

__all__ = ["SpectralMaskConfig", "SpectralMaskReducer"]

"""ROCKET + Ridge 规范门面 / Canonical ROCKET-plus-Ridge facade.

中文：复用 mask-aware scaler、kernel 与 ridge。English: Re-export the mask-aware scaler, kernels and ridge.
"""

from .rocket import (
    MaskedChannelRobustScaler,
    MiniRocketAblation,
    RocketKernel,
    RocketRidgeClassifier,
    RocketTransformer,
)

__all__ = [
    "MaskedChannelRobustScaler",
    "MiniRocketAblation",
    "RocketKernel",
    "RocketRidgeClassifier",
    "RocketTransformer",
]

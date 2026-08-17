"""Frozen formal and explicitly non-formal motion detector constructors.

The formal V2 tensor is frozen to the ordered 11-channel schema exported by
representations.motion: RED, IR, three dynamic-acceleration axes, three
gyroscope axes, dynamic-acceleration magnitude, angular-rate magnitude, and
jerk magnitude. Formal callers use build_formal_motion_cnn, whose zero-argument
interface prevents channel-count or ordering drift.

The parameterized constructor remains available only for explicitly named
development experiments. A model built through it is not a formal V2 model,
even when the supplied channel names happen to equal the formal schema.

The historical ten-channel builder is different: it reproduces the archived
``B_light_cnn`` architecture for provenance and backup use.  It is not the
removed V2 IR-only ablation and its archived scores are not V2 results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..representations.motion import (
    MOTION_NETWORK_CHANNEL_SCHEMA,
    MOTION_NETWORK_SCHEMA_SHA256,
    MOTION_WINDOW_SAMPLES,
)

from ..motion_ids import (
    FORMAL_MOTION_MODEL_ID,
    HISTORICAL_LIGHT_CNN_MODEL_ID,
    PARAMETERIZED_LIGHT_CNN_ID,
)

HISTORICAL_LIGHT_CNN_CHANNELS = (
    "ppg_single_historical_loader_pleth_2_first",
    "acc_dynamic_x",
    "acc_dynamic_y",
    "acc_dynamic_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_magnitude",
    "gyro_magnitude",
    "jerk_magnitude",
)


@dataclass(frozen=True)
class LightCnnArchitecture:
    """Complete constructor identity for one lightweight 1-D CNN.

    ``channel_names`` is part of the identity.  An integer channel count alone
    is insufficient because it could silently swap physical inputs.
    """

    channel_names: tuple[str, ...]
    base_channels: int = 12
    kernel_sizes: tuple[int, int, int] = (9, 7, 5)
    normalization: str = "group_norm_one_group"
    activation: str = "gelu"
    pooling: str = "avgpool2_avgpool2_adaptiveavgpool1"
    output: str = "single_motion_logit"

    def validate(self) -> None:
        if not self.channel_names:
            raise ValueError("motion CNN requires at least one input channel")
        if any(not str(name).strip() for name in self.channel_names):
            raise ValueError("motion CNN channel names must be non-empty")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("motion CNN channel names must be unique and ordered")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if self.kernel_sizes != (9, 7, 5):
            raise ValueError("this V2 port only implements the archived 9/7/5 kernels")

    @property
    def in_channels(self) -> int:
        return len(self.channel_names)

    @property
    def parameter_count(self) -> int:
        """Exact trainable-parameter count, including affine GroupNorm terms."""

        self.validate()
        c = self.in_channels
        b = self.base_channels
        first = c * b * 9 + b + 2 * b
        second = b * (2 * b) * 7 + (2 * b) + 2 * (2 * b)
        third = (2 * b) * (2 * b) * 5 + (2 * b) + 2 * (2 * b)
        head = (2 * b) + 1
        return int(first + second + third + head)


try:  # PyTorch remains an explicit optional dependency profile.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in dependency-profile CI.
    torch = None
    nn = None


if nn is not None:

    class LightCnnMotionDetector(nn.Module):
        """Exact 9/7/5 lightweight binary-logit architecture.

        This class defines architecture only.  It performs no training,
        threshold selection, split construction, or external evaluation.
        """

        def __init__(self, architecture: LightCnnArchitecture):
            super().__init__()
            architecture.validate()
            b = int(architecture.base_channels)
            self.architecture = architecture
            self.net = nn.Sequential(
                nn.Conv1d(architecture.in_channels, b, kernel_size=9, padding=4),
                nn.GroupNorm(1, b),
                nn.GELU(),
                nn.AvgPool1d(2),
                nn.Conv1d(b, 2 * b, kernel_size=7, padding=3),
                nn.GroupNorm(1, 2 * b),
                nn.GELU(),
                nn.AvgPool1d(2),
                nn.Conv1d(2 * b, 2 * b, kernel_size=5, padding=2),
                nn.GroupNorm(1, 2 * b),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(2 * b, 1),
            )

        def forward(self, values: Any) -> Any:
            if values.ndim != 3:
                raise ValueError("motion CNN input must have shape [batch, channel, sample]")
            if int(values.shape[1]) != self.architecture.in_channels:
                raise ValueError(
                    "motion CNN channel count does not match its frozen ordered schema"
                )
            return self.net(values).squeeze(1)

else:

    class LightCnnMotionDetector:  # type: ignore[no-redef]
        """Dependency guard retaining importability without optional PyTorch."""

        def __init__(self, architecture: LightCnnArchitecture):
            architecture.validate()
            raise ImportError(
                "LightCnnMotionDetector requires the optional 'deep' PyTorch profile"
            )


def build_parameterized_light_cnn(
    channel_names: Sequence[str],
    *,
    base_channels: int = 12,
) -> LightCnnMotionDetector:
    """Build a development-only, non-formal parameterized motion CNN.

    This constructor never conveys formal status. Passing eight raw source
    names does not imply a final network tensor, and passing the formal schema
    does not substitute for build_formal_motion_cnn.
    """

    architecture = LightCnnArchitecture(
        channel_names=tuple(str(name) for name in channel_names),
        base_channels=int(base_channels),
    )
    architecture.validate()
    return LightCnnMotionDetector(architecture)


def build_formal_motion_cnn() -> LightCnnMotionDetector:
    """Construct only the frozen 11-channel, 3200-sample V2 motion model.

    The generic builder remains available for explicitly named development
    experiments. Formal code must use this zero-argument constructor so caller
    input cannot silently reorder, omit, or derive channels.
    """

    model = build_parameterized_light_cnn(MOTION_NETWORK_CHANNEL_SCHEMA)
    model.motion_input_contract = {
        "channel_schema": MOTION_NETWORK_CHANNEL_SCHEMA,
        "schema_sha256": MOTION_NETWORK_SCHEMA_SHA256,
        "window_samples": MOTION_WINDOW_SAMPLES,
    }
    return model


def build_historical_light_cnn_backup() -> LightCnnMotionDetector:
    """Construct the archived ten-channel Light CNN architecture only.

    The source ``DetectorConfig.base_channels`` was 24, while the historical
    factory passed ``max(8, base_channels // 2)`` into ``B_light_cnn``.  The
    resolved network width is therefore 12 and the exact parameter count is
    6,181.  Weights are not loaded by this constructor.
    """

    return build_parameterized_light_cnn(
        HISTORICAL_LIGHT_CNN_CHANNELS,
        base_channels=12,
    )


def count_trainable_parameters(model: Any) -> int:
    """Count trainable parameters without initiating inference or training."""

    if not hasattr(model, "parameters"):
        raise TypeError("model does not expose PyTorch parameters")
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


__all__ = [
    "FORMAL_MOTION_MODEL_ID",
    "HISTORICAL_LIGHT_CNN_CHANNELS",
    "HISTORICAL_LIGHT_CNN_MODEL_ID",
    "LightCnnArchitecture",
    "LightCnnMotionDetector",
    "PARAMETERIZED_LIGHT_CNN_ID",
    "build_historical_light_cnn_backup",
    "build_formal_motion_cnn",
    "build_parameterized_light_cnn",
    "count_trainable_parameters",
]

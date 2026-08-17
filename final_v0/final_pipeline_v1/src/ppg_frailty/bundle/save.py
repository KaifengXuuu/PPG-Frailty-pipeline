"""§5.14 完整 metadata 的 bundle 保存 / Bundle saving with complete metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..training.bundle import (
    REQUIRED_METADATA as TRAINING_REQUIRED_METADATA,
    save_bundle,
    validate_bundle_metadata as validate_training_bundle_metadata,
)


REQUIRED_V1_METADATA = frozenset(TRAINING_REQUIRED_METADATA)


def validate_bundle_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """委托唯一 §5.14 schema / Delegate to the sole §5.14 validator.

    中文：canonical facade 不再维护第二套易漂移字段表；训练与部署共用同一
    fail-closed validator。English: The facade owns no second drifting schema.
    """

    return validate_training_bundle_metadata(metadata)


def save_bundle_strict(
    model: Any,
    directory: str | Path,
    *,
    model_config: Mapping[str, Any],
    input_spec: Mapping[str, Any] | Any,
    metadata: Mapping[str, Any],
    golden_inputs: Mapping[str, np.ndarray],
    transforms: Any = None,
    pipeline_adapter: Any = None,
    strict_metadata: bool,
    parity_atol: float = 1e-6,
) -> Path:
    """验证完整合同后调用已测试写入器 / Validate, then call the tested writer.

    中文：正式导出调用方必须显式传 strict_metadata=True，并可序列化
    raw-record→model-input adapter；任何兼容宽松模式都被拒绝。
    English: Formal exporters must explicitly opt into the strict contract and may
    bundle the raw-record-to-model-input adapter; compatibility mode is forbidden.
    """

    if strict_metadata is not True:
        raise ValueError("formal bundle export requires strict_metadata=True")

    return save_bundle(
        model,
        directory,
        model_config=model_config,
        input_spec=input_spec,
        metadata=validate_bundle_metadata(metadata),
        golden_inputs=golden_inputs,
        transforms=transforms,
        pipeline_adapter=pipeline_adapter,
        parity_atol=parity_atol,
    )


__all__ = ["REQUIRED_V1_METADATA", "save_bundle_strict", "validate_bundle_metadata"]

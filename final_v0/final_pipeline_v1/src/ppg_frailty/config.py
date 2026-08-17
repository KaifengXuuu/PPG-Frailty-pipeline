"""严格、无隐藏默认的配置合同 / Strict configuration with no hidden defaults."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


TOP_LEVEL_KEYS = {
    "schema_version",
    "config_id",
    "manifest",
    "splits",
    "output",
    "representation_mode",
    "roles",
    "signal",
    "windows",
    "quality",
    "artifact",
    "features",
    "model",
    "training",
    "aggregation",
    "evaluation",
}


def _strict_mapping(value: Any, name: str) -> dict[str, Any]:
    """验证对象类型 / Require a string-keyed mapping."""

    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _require_exact_keys(
    mapping: Mapping[str, Any],
    required: set[str],
    *,
    context: str,
) -> None:
    """拒绝缺字段和未知字段 / Reject missing and unknown fields."""

    observed = set(mapping)
    missing = sorted(required - observed)
    unknown = sorted(observed - required)
    if missing or unknown:
        raise ValueError(
            f"{context} key mismatch: missing={missing}, unknown={unknown}"
        )


def canonical_json_bytes(value: Any) -> bytes:
    """稳定严格 JSON / Render canonical strict JSON bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class PipelineConfig:
    """规范实验配置 / Canonical experiment configuration.

    中文：各 section 保留为不可变 mapping，以允许新增明确字段而不在代码中产生
    隐藏默认。reference YAML 必须逐项写出所有行为参数。

    English: Sections remain immutable-by-convention mappings so explicit fields can
    evolve without code defaults. Reference YAML files must spell out every behavior.
    """

    payload: dict[str, Any]
    source_path: str
    sha256: str

    @property
    def config_id(self) -> str:
        """返回配置 ID / Return the configuration identity."""

        return str(self.payload["config_id"])

    @property
    def representation_mode(self) -> str:
        """返回表征模式 / Return the representation mode."""

        return str(self.payload["representation_mode"])

    def section(self, name: str) -> dict[str, Any]:
        """读取一个显式 section / Return one explicit section."""

        if name not in TOP_LEVEL_KEYS:
            raise KeyError(name)
        value = self.payload[name]
        return _strict_mapping(value, name)

    def to_dict(self) -> dict[str, Any]:
        """复制可序列化配置 / Copy the serializable payload."""

        return json.loads(json.dumps(self.payload, allow_nan=False))


def validate_config_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """执行跨 section 的 fail-closed 配置验证 / Validate cross-section invariants."""

    data = _strict_mapping(payload, "config")
    _require_exact_keys(data, TOP_LEVEL_KEYS, context="config")
    if data["schema_version"] != "ppg_frailty.pipeline_config.v1":
        raise ValueError("unsupported schema_version")
    if data["representation_mode"] not in {
        "raw",
        "feature_vector",
        "feature_matrix",
        "fusion",
    }:
        raise ValueError("unsupported representation_mode")
    roles = data["roles"]
    if not isinstance(roles, list) or not roles or not all(
        role in {"B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2"}
        for role in roles
    ):
        raise ValueError("roles must be a non-empty registered role list")
    for section in TOP_LEVEL_KEYS - {
        "schema_version",
        "config_id",
        "representation_mode",
        "roles",
    }:
        _strict_mapping(data[section], section)
    training = _strict_mapping(data["training"], "training")
    if training.get("epoch_rule") not in {"fixed_epoch", "inner_grouped_selection"}:
        raise ValueError("training.epoch_rule must be explicit")
    if training.get("outer_labels_visible_to_trainer") is not False:
        raise ValueError("outer labels must be unavailable to the trainer")
    artifact = _strict_mapping(data["artifact"], "artifact")
    if artifact.get("selection_scope") != "run_before_evaluation":
        raise ValueError("artifact route must be selected before evaluation")
    aggregation = _strict_mapping(data["aggregation"], "aggregation")
    if aggregation.get("hierarchy") != [
        "window",
        "file",
        "role",
        "participant",
    ]:
        raise ValueError("reference aggregation hierarchy is frozen")
    return data


def load_config(path: str | Path) -> PipelineConfig:
    """加载 YAML/JSON 并计算规范 hash / Load and hash a strict config."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    data = validate_config_payload(_strict_mapping(payload, "config"))
    digest = hashlib.sha256(canonical_json_bytes(data)).hexdigest()
    return PipelineConfig(data, source.as_posix(), digest)


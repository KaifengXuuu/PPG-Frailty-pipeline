"""Integrity-checked model bundles with golden prediction parity.

带完整性校验与 golden prediction 一致性的模型 bundle。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from ..models import ModelInputSpec, create_model, normalize_model_config
from ..provenance import sha256_file


BUNDLE_FORMAT_VERSION = "ppg_frailty_bundle_v1"
REQUIRED_METADATA = {
    "model_identity",
    "representation_mode",
    "signal_route",
    "class_order",
    "channel_schema",
    "preprocessing",
    "preprocessing_hash",
    "resampling",
    "window_plan",
    "feature_registry",
    "feature_hash",
    "feature_vector_schema",
    "ordered_matrix_schema",
    "mask_semantics",
    "validity_policy",
    "fitted_objects",
    "representation_state",
    "pooling_rule",
    "aggregation_rule",
    "manifest_hash",
    "fold_hash",
    "manifest_version",
    "fold_registry_version",
    "code_version",
    "environment",
    "dependency_status",
    "golden_case",
}

_STRUCTURED_METADATA = {
    "model_identity",
    "preprocessing",
    "resampling",
    "window_plan",
    "feature_registry",
    "feature_vector_schema",
    "ordered_matrix_schema",
    "mask_semantics",
    "validity_policy",
    "representation_state",
    "environment",
    "golden_case",
}


@dataclass(frozen=True)
class LoadedBundle:
    """Validated bundle held in memory / 内存中的已校验 bundle。"""

    model: Any
    transforms: Any
    manifest: dict[str, Any]
    directory: Path
    pipeline_adapter: Any = None


def _jsonable(value: Any) -> Any:
    """Convert dataclasses/arrays/scalars to strict JSON / 转换为严格 JSON。"""

    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise TypeError("bundle JSON payload cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"bundle metadata is not JSON serialisable: {type(value).__name__}")


def validate_bundle_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete §5.14 deployment contract.

    English: Raw, vector, matrix and fusion routes use the same keys. A field
    that does not apply must be an explicit structured status object rather than
    disappearing from the manifest.

    中文：raw、vector、matrix 与 fusion 路线使用同一组键。不适用字段必须写成
    显式的结构化状态对象，不得从 manifest 中消失。
    """

    missing = sorted(REQUIRED_METADATA - set(metadata))
    if missing:
        raise ValueError(f"bundle metadata is missing required fields: {missing}")
    normalized = _jsonable(metadata)
    for name in REQUIRED_METADATA:
        value = normalized[name]
        if value is None or isinstance(value, str) and not value.strip():
            raise ValueError(f"bundle metadata field {name!r} must be explicit and non-empty")
        if isinstance(value, (list, dict)) and not value:
            raise ValueError(f"bundle metadata field {name!r} must not be empty")
    for name in _STRUCTURED_METADATA:
        if not isinstance(normalized[name], dict):
            raise TypeError(f"bundle metadata field {name!r} must be a mapping")
    model_identity = normalized["model_identity"]
    if not {"name", "machine_id", "version"} <= set(model_identity):
        raise ValueError("model_identity requires name, machine_id and version")
    class_order = tuple(normalized["class_order"])
    channels = tuple(normalized["channel_schema"])
    if len(class_order) < 2 or len(class_order) != len(set(class_order)):
        raise ValueError("bundle class_order must contain unique declared classes")
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("bundle channel_schema must contain unique declared channels")
    if not isinstance(normalized["fitted_objects"], list):
        raise TypeError("bundle fitted_objects must be an exhaustive list")
    return normalized


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical JSON atomically / 原子写入规范 JSON。"""

    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _is_torch_model(model: Any) -> bool:
    """Check torch type lazily / 延迟检查 torch 类型。"""

    try:
        import torch
    except ImportError:
        return False
    return isinstance(model, torch.nn.Module)


def _predict_model(model: Any, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Representation-aware probability prediction / representation 感知概率预测。"""

    if _is_torch_model(model):
        import torch

        model.eval()
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        with torch.no_grad():
            tensors = {
                key: torch.as_tensor(value, device=device)
                for key, value in inputs.items()
            }
            if "window_bag" in tensors:
                logits = model(
                    tensors["window_bag"].float(),
                    tensors["window_mask"].bool(),
                    tensors["file_features"].float(),
                    tensors.get("sample_mask", None).bool()
                    if tensors.get("sample_mask") is not None
                    else None,
                )
                probability = torch.softmax(logits, dim=-1)
            elif hasattr(model, "predict_probabilities"):
                probability = model.predict_probabilities(
                    tensors["x"].float(),
                    tensors.get("mask", None).bool() if tensors.get("mask") is not None else None,
                )
            else:
                logits = model(
                    tensors["x"].float(),
                    tensors.get("mask", None).bool() if tensors.get("mask") is not None else None,
                )
                probability = torch.softmax(logits, dim=-1)
        return probability.detach().cpu().numpy().astype(np.float64)

    x = np.asarray(inputs["x"])
    mask = inputs.get("mask")
    try:
        probability = model.predict_proba(x, mask=None if mask is None else np.asarray(mask))
    except TypeError:
        probability = model.predict_proba(x)
    return np.asarray(probability, dtype=np.float64)


def _apply_transforms(transforms: Any, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply a bundled fitted transform without hiding its interface.

    应用 bundle 内的已拟合转换，同时不隐藏其接口。优先要求 transform_inputs；
    单数组转换仅作用于 x，并保留 mask 等结构字段。
    """

    copied = {str(key): np.asarray(value) for key, value in inputs.items()}
    if transforms is None:
        return copied
    if hasattr(transforms, "transform_inputs"):
        result = transforms.transform_inputs(copied)
        if not isinstance(result, Mapping):
            raise TypeError("transform_inputs must return a mapping")
        return {str(key): np.asarray(value) for key, value in result.items()}
    if not hasattr(transforms, "transform") or "x" not in copied:
        raise TypeError("bundled transforms must expose transform_inputs or transform(x)")
    mask = copied.get("mask")
    if mask is not None:
        try:
            copied["x"] = np.asarray(transforms.transform(copied["x"], mask=mask))
        except TypeError:
            copied["x"] = np.asarray(transforms.transform(copied["x"]))
    else:
        copied["x"] = np.asarray(transforms.transform(copied["x"]))
    return copied


def save_bundle(
    model: Any,
    directory: str | Path,
    *,
    model_config: Mapping[str, Any],
    input_spec: ModelInputSpec | Mapping[str, Any],
    metadata: Mapping[str, Any],
    golden_inputs: Mapping[str, np.ndarray],
    transforms: Any = None,
    pipeline_adapter: Any = None,
    parity_atol: float = 1e-6,
) -> Path:
    """Save an immutable bundle, reload it and enforce golden parity.

    保存不可变 bundle，随后立即重载并强制 golden parity。缺少关键 provenance 字段、
    文件哈希不匹配或往返预测偏差都会关闭失败。
    """

    normalized_metadata = validate_bundle_metadata(metadata)
    if not golden_inputs or "x" not in golden_inputs and "window_bag" not in golden_inputs:
        raise ValueError("golden_inputs must contain x or window_bag")
    if not np.isfinite(parity_atol) or parity_atol < 0:
        raise ValueError("parity_atol must be finite and non-negative")
    target = Path(directory)
    if target.exists():
        raise FileExistsError(f"bundle target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=str(target.parent))
    )
    try:
        kind = "torch" if _is_torch_model(model) else "estimator"
        state_name = "state.pt" if kind == "torch" else "estimator.joblib"
        state_path = staging / state_name
        if kind == "torch":
            import torch

            torch.save(model.state_dict(), state_path)
        else:
            joblib.dump(model, state_path, compress=3)

        file_hashes = {state_name: sha256_file(state_path)}
        if transforms is not None:
            transforms_path = staging / "transforms.joblib"
            joblib.dump(transforms, transforms_path, compress=3)
            file_hashes[transforms_path.name] = sha256_file(transforms_path)
        if pipeline_adapter is not None:
            adapter_path = staging / "pipeline_adapter.joblib"
            joblib.dump(pipeline_adapter, adapter_path, compress=3)
            file_hashes[adapter_path.name] = sha256_file(adapter_path)

        expected = _predict_model(model, _apply_transforms(transforms, golden_inputs))
        if expected.ndim != 2 or not np.isfinite(expected).all():
            raise ValueError("golden prediction must be finite [sample,class]")
        golden_path = staging / "golden.npz"
        arrays = {f"input__{key}": np.asarray(value) for key, value in golden_inputs.items()}
        arrays["expected_probabilities"] = expected
        np.savez_compressed(golden_path, **arrays)
        file_hashes[golden_path.name] = sha256_file(golden_path)

        spec = ModelInputSpec.from_value(input_spec)
        normalized_model_config = normalize_model_config(model_config)
        if normalized_metadata["representation_mode"] != str(spec.mode.value):
            raise ValueError("metadata representation_mode disagrees with input_spec")
        if len(normalized_metadata["class_order"]) != spec.n_classes:
            raise ValueError("metadata class_order disagrees with input_spec n_classes")
        if expected.shape[1] != spec.n_classes:
            raise ValueError("golden prediction class count disagrees with input_spec")
        if spec.channel_schema and tuple(normalized_metadata["channel_schema"]) != tuple(
            spec.channel_schema
        ):
            raise ValueError("metadata channel_schema disagrees with input_spec")
        declared_machine_id = normalized_metadata["model_identity"]["machine_id"]
        if declared_machine_id != normalized_model_config["model_id"]:
            raise ValueError("metadata model_identity disagrees with model_config")
        if (
            normalized_metadata["model_identity"]["name"]
            != normalized_model_config["canonical_model_name"]
        ):
            raise ValueError("metadata canonical model name disagrees with model_config")
        runtime_model_id = getattr(model, "model_id", declared_machine_id)
        if runtime_model_id != declared_machine_id:
            raise ValueError("runtime model identity disagrees with bundle metadata")
        manifest = {
            "bundle_format": BUNDLE_FORMAT_VERSION,
            "kind": kind,
            "state_file": state_name,
            "model_config": _jsonable(normalized_model_config),
            "canonical_model_name": normalized_model_config["canonical_model_name"],
            "machine_model_id": normalized_model_config["model_id"],
            "input_spec": _jsonable(spec),
            "metadata": normalized_metadata,
            "required_metadata_fields": sorted(REQUIRED_METADATA),
            "file_hashes": file_hashes,
            "golden_parity_atol": float(parity_atol),
            "golden_case_hash": file_hashes[golden_path.name],
            "pipeline_adapter_boundary": (
                "serialized_raw_record_to_model_input_mapping"
                if pipeline_adapter is not None
                else "not_bundled"
            ),
            "transactional_save": "same_filesystem_staging_then_atomic_rename",
            "joblib_trust_boundary": "load_only_integrity_verified_local_bundles",
        }
        _atomic_json(staging / "manifest.json", manifest)
        loaded = load_bundle(staging)
        assert_golden_parity(loaded, atol=parity_atol)
        if target.exists():
            raise FileExistsError(f"bundle target appeared during staging: {target}")
        # English: Same-filesystem directory rename is the single commit point.
        # 中文：同一文件系统内的目录重命名是唯一提交点。
        os.rename(staging, target)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_bundle(
    directory: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> LoadedBundle:
    """Verify hashes and schemas before loading executable state.

    加载可执行状态前校验全部文件哈希与 metadata/schema 期望。
    """

    target = Path(directory)
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("bundle manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_format") != BUNDLE_FORMAT_VERSION:
        raise ValueError("unsupported bundle format")
    metadata = validate_bundle_metadata(manifest.get("metadata", {}))
    if set(manifest.get("required_metadata_fields", ())) != REQUIRED_METADATA:
        raise ValueError("bundle metadata schema declaration is stale or incomplete")
    if expected_metadata is not None:
        expected = _jsonable(expected_metadata)
        for name, value in expected.items():
            if name not in metadata or metadata[name] != value:
                raise ValueError(f"bundle metadata mismatch for expected field: {name}")
    file_hashes = manifest.get("file_hashes", {})
    if not isinstance(file_hashes, dict):
        raise ValueError("bundle file_hashes must be an object")
    required_files = {str(manifest.get("state_file", "")), "golden.npz"}
    if not required_files <= set(file_hashes):
        raise ValueError("bundle does not hash every required payload file")
    expected_names = set(file_hashes) | {"manifest.json"}
    actual_names = {path.name for path in target.iterdir()}
    if actual_names != expected_names:
        raise ValueError("bundle contains missing or unexpected unverified files")
    for name, expected_hash in file_hashes.items():
        path = target / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"bundle file integrity check failed: {name}")

    state_path = target / manifest["state_file"]
    if manifest["kind"] == "torch":
        import torch

        model = create_model(manifest["model_config"], manifest["input_spec"])
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older supported torch fallback
            state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
    elif manifest["kind"] == "estimator":
        # English: Integrity is checked first; joblib remains a trusted-local format.
        # 中文：先校验完整性；joblib 仍被视为仅限可信本地的格式。
        model = joblib.load(state_path)
    else:
        raise ValueError("unknown bundle kind")
    transforms_path = target / "transforms.joblib"
    transforms = (
        joblib.load(transforms_path)
        if transforms_path.name in file_hashes
        else None
    )
    adapter_path = target / "pipeline_adapter.joblib"
    pipeline_adapter = (
        joblib.load(adapter_path)
        if adapter_path.name in file_hashes
        else None
    )
    return LoadedBundle(
        model=model,
        transforms=transforms,
        manifest=manifest,
        directory=target,
        pipeline_adapter=pipeline_adapter,
    )


def predict_bundle(
    bundle: LoadedBundle | str | Path, inputs: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Predict through a validated loaded bundle / 通过已校验 bundle 预测。"""

    loaded = load_bundle(bundle) if isinstance(bundle, (str, Path)) else bundle
    transformed = _apply_transforms(loaded.transforms, inputs)
    probability = _predict_model(loaded.model, transformed)
    if not np.isfinite(probability).all() or not np.allclose(
        probability.sum(axis=1), 1.0, atol=1e-6
    ):
        raise RuntimeError("bundle prediction is not a finite probability matrix")
    return probability


def predict_bundle_raw(
    bundle: LoadedBundle | str | Path,
    raw_record: Any,
) -> np.ndarray:
    """Run a serialised raw-record adapter before normal bundle inference.

    先运行已序列化的 raw-record adapter，再执行常规 bundle 推理。adapter 必须
    显式实现 transform_record(raw_record) 或可调用接口，并返回模型输入 mapping。
    """

    loaded = load_bundle(bundle) if isinstance(bundle, (str, Path)) else bundle
    adapter = loaded.pipeline_adapter
    if adapter is None:
        raise RuntimeError("bundle does not contain a raw-record pipeline adapter")
    if hasattr(adapter, "transform_record"):
        inputs = adapter.transform_record(raw_record)
    elif callable(adapter):
        inputs = adapter(raw_record)
    else:
        raise TypeError("pipeline adapter must be callable or expose transform_record")
    if not isinstance(inputs, Mapping):
        raise TypeError("pipeline adapter must return a model-input mapping")
    return predict_bundle(loaded, {str(key): np.asarray(value) for key, value in inputs.items()})


def assert_golden_parity(bundle: LoadedBundle | str | Path, *, atol: float | None = None) -> None:
    """Assert saved and reloaded probabilities agree / 断言保存前后概率一致。"""

    loaded = load_bundle(bundle) if isinstance(bundle, (str, Path)) else bundle
    golden_path = loaded.directory / "golden.npz"
    with np.load(golden_path, allow_pickle=False) as archive:
        expected = np.asarray(archive["expected_probabilities"], dtype=np.float64)
        inputs = {
            key.removeprefix("input__"): np.asarray(archive[key])
            for key in archive.files
            if key.startswith("input__")
        }
    tolerance = (
        float(loaded.manifest["golden_parity_atol"]) if atol is None else float(atol)
    )
    actual = predict_bundle(loaded, inputs)
    if expected.shape != actual.shape or not np.allclose(expected, actual, atol=tolerance, rtol=0.0):
        maximum = float(np.max(np.abs(expected - actual))) if expected.shape == actual.shape else float("inf")
        raise RuntimeError(f"golden prediction parity failed; maximum absolute error={maximum}")


def assert_repeated_bundle_parity(
    bundle: str | Path,
    *,
    iterations: int = 10_000,
    reload_each_iteration: bool = True,
) -> None:
    """Stress repeated load/predict without repeatedly saving to disk.

    重复执行 load/predict 压力验证，但绝不重复磁盘 save。默认 10,000 轮；为减少
    CI 时间可显式关闭每轮 reload，但正式序列化门禁必须保持默认值。
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    path = Path(bundle)
    initial = load_bundle(path)
    with np.load(initial.directory / "golden.npz", allow_pickle=False) as archive:
        expected = np.asarray(archive["expected_probabilities"], dtype=np.float64)
        inputs = {
            key.removeprefix("input__"): np.asarray(archive[key])
            for key in archive.files
            if key.startswith("input__")
        }
    tolerance = float(initial.manifest["golden_parity_atol"])
    loaded = initial
    for index in range(iterations):
        if reload_each_iteration and index:
            loaded = load_bundle(path)
        actual = predict_bundle(loaded, inputs)
        if expected.shape != actual.shape or not np.allclose(
            expected, actual, atol=tolerance, rtol=0.0
        ):
            raise RuntimeError(f"bundle repeated parity failed at iteration {index + 1}")

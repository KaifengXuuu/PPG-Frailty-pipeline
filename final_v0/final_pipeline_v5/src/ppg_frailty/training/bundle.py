"""Integrity-checked model bundles with golden prediction parity."""
from __future__ import annotations
import json
import hashlib
import importlib.metadata
import os
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping
import joblib
import numpy as np
from ..models import (
    ModelInputSpec,
    create_model,
    normalize_model_config,
    validate_frozen_model_run_provenance,
    validate_resolved_architecture,
)
from ..provenance import sha256_file, stable_payload_sha256

BUNDLE_FORMAT_VERSION = "ppg_frailty_bundle_parity_v3"
FINAL_BUNDLE_PARITY_ATOL = 1e-06
GENERIC_BUNDLE_KIND = "generic_research"
TRUSTED_FINAL_REFIT_BUNDLE_KIND = "trusted_final_refit_v2"
_ENSEMBLE_MODEL_KINDS = frozenset({"ensemble", "five_member_ensemble"})

def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)

def _require_hashes(value: object, names: tuple[str, ...]) -> None:
    for name in names:
        if not _is_sha256(getattr(value, name)):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")

def _is_ensemble_model_kind(value: object) -> bool:
    """Recognize the generic kind and its historical five-member alias."""
    return str(value) in _ENSEMBLE_MODEL_KINDS

def _validated_refit_seed_roster(values: object) -> tuple[int, ...]:
    """Return a non-empty unique roster representable in persisted int64 fields."""
    if not isinstance(values, (list, tuple)):
        raise ValueError("final-refit training_seeds must be an ordered list or tuple")
    raw_values = tuple(values)
    seeds: list[int] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("final-refit training_seeds must contain integers")
        try:
            seed = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("final-refit training_seeds must contain integers") from exc
        if isinstance(value, (float, np.floating)) and (not np.isfinite(value) or float(value) != float(seed)):
            raise ValueError("final-refit training_seeds must contain finite integers")
        if seed < 0 or seed > 4294967295:
            raise ValueError("final-refit training_seeds must be in the executable uint32 range")
        seeds.append(seed)
    roster = tuple(seeds)
    if not roster or len(roster) != len(set(roster)):
        raise ValueError("final-refit training_seeds must be non-empty and unique")
    return roster


REQUIRED_RUNTIME_ENVIRONMENT_KEYS = tuple("python python_implementation numpy scipy scikit_learn joblib torch".split())
REQUIRED_METADATA = set(
    """model_identity representation_mode signal_route class_order channel_schema preprocessing preprocessing_hash
    resampling window_plan feature_registry feature_hash feature_vector_schema ordered_matrix_schema mask_semantics
    validity_policy fitted_objects representation_state pooling_rule aggregation_rule manifest_hash fold_hash
    manifest_version fold_registry_version pipeline_generation config_hash balance_hash run_hash source_snapshot_hash
    code_version environment dependency_status serialization_trust golden_case""".split()
)
_STRUCTURED_METADATA = set(
    """model_identity preprocessing resampling window_plan feature_registry feature_vector_schema ordered_matrix_schema
    mask_semantics validity_policy representation_state environment serialization_trust golden_case""".split()
)

def current_runtime_environment() -> dict[str, str]:
    """Return descriptive runtime versions for troubleshooting."""

    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "not_installed"

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": str(np.__version__),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "joblib": str(joblib.__version__),
        "torch": version("torch"),
    }

@dataclass(frozen=True)
class LoadedBundle:
    """Validated bundle held in memory / 内存中的已校验 bundle。"""

    model: Any
    transforms: Any
    manifest: dict[str, Any]
    directory: Path
    pipeline_adapter: Any = None

@dataclass(frozen=True)
class FinalRefitPlan:
    """Human-selected full-29-participant refit identity, never performance evidence."""

    purpose: str
    config_hash: str
    model_id: str
    participant_ids: tuple[str, ...]
    training_seeds: tuple[int, ...]
    fixed_epochs: int | None
    epoch_rule: str
    model_family: str
    oof_evidence_hash: str
    model_kind: str
    registry_hash: str
    source_snapshot_hash: str
    manual_selection_hash: str
    resolved_model_config_hash: str
    architecture_parameters_hash: str
    input_schema_hash: str
    training_config_hash: str
    frozen_run_provenance_hash: str
    representation_mode: str
    performance_evidence: str = "outer_oof_only_no_refit_self_evaluation"

    def __post_init__(self) -> None:
        participants = tuple(sorted(set(map(str, self.participant_ids))))
        seeds = _validated_refit_seed_roster(self.training_seeds)
        if not self.purpose.strip() or not self.model_id.strip() or len(participants) != 29:
            raise ValueError("final refit requires purpose, model_id and exactly 29 participants")
        if self.model_kind not in {"single_model", *_ENSEMBLE_MODEL_KINDS}:
            raise ValueError("invalid final-refit model_kind")
        if self.model_kind == "single_model" and len(seeds) != 1:
            raise ValueError("single-model final refit requires exactly one training seed")
        if self.model_family == "deep":
            if (
                self.epoch_rule != "fixed_epoch"
                or isinstance(self.fixed_epochs, (bool, np.bool_))
                or not isinstance(self.fixed_epochs, (int, np.integer))
                or int(self.fixed_epochs) <= 0
            ):
                raise ValueError("deep final refit fixed_epochs must be a positive integer")
            object.__setattr__(self, "fixed_epochs", int(self.fixed_epochs))
        elif self.model_family == "classical_or_rocket":
            if (
                self.epoch_rule != "not_applicable"
                or self.fixed_epochs is not None
                or self.model_kind != "single_model"
            ):
                raise ValueError("estimator final refit requires a single model and no epoch")
        else:
            raise ValueError("model_family must be deep or classical_or_rocket")
        if self.representation_mode not in {"raw", "feature_vector", "feature_matrix", "fusion"}:
            raise ValueError("final refit representation_mode is invalid")
        _require_hashes(
            self,
            (
                "config_hash",
                "oof_evidence_hash",
                "registry_hash",
                "source_snapshot_hash",
                "manual_selection_hash",
                "resolved_model_config_hash",
                "architecture_parameters_hash",
                "input_schema_hash",
                "training_config_hash",
                "frozen_run_provenance_hash",
            ),
        )
        object.__setattr__(self, "participant_ids", participants)
        object.__setattr__(self, "training_seeds", seeds)

@dataclass(frozen=True)
class FinalRefitBinding:
    """Canonical selected model/input/training/run identity used for refit."""

    resolved_model_config: Mapping[str, Any]
    input_spec: Mapping[str, Any]
    training_config: Mapping[str, Any]
    frozen_run_provenance: Mapping[str, Any]
    resolved_model_config_hash: str
    architecture_parameters_hash: str
    input_schema_hash: str
    training_config_hash: str
    frozen_run_provenance_hash: str

def canonical_input_spec_payload(input_spec: ModelInputSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Return the one strict-JSON identity used by refits and bundle adapters."""
    spec = ModelInputSpec.from_value(input_spec)
    return {
        "representation_mode": spec.mode.value,
        "n_channels": int(spec.n_channels),
        "n_classes": int(spec.n_classes),
        "n_file_features": int(spec.n_file_features),
        "feature_names": list(spec.feature_names),
        "channel_schema": list(spec.channel_schema),
    }

def input_spec_sha256(input_spec: ModelInputSpec | Mapping[str, Any]) -> str:
    """Hash the canonical input boundary shared by refit and inference."""
    return stable_payload_sha256(canonical_input_spec_payload(input_spec))

def materialize_final_refit_binding(
    *,
    resolved_model_config: Mapping[str, Any],
    input_spec: ModelInputSpec | Mapping[str, Any],
    training_config: Any,
    frozen_run_provenance: Mapping[str, Any],
    config_hash: str,
    registry_hash: str,
    source_snapshot_hash: str,
    manual_selection_hash: str,
    oof_evidence_hash: str,
) -> FinalRefitBinding:
    """Create the canonical model/input/training identity used by refit."""
    normalized_model = normalize_model_config(resolved_model_config)
    architecture = normalized_model.get("architecture_parameters")
    if not isinstance(architecture, Mapping) or not architecture:
        raise ValueError("final refit model config requires architecture_parameters")
    spec = ModelInputSpec.from_value(input_spec)
    spec_payload = canonical_input_spec_payload(spec)
    if is_dataclass(training_config):
        training_payload = asdict(training_config)
    elif isinstance(training_config, Mapping):
        training_payload = dict(training_config)
    else:
        raise TypeError("training_config must be a dataclass or mapping")
    validated_run = validate_frozen_model_run_provenance(frozen_run_provenance)
    normalized_model_json = _jsonable(normalized_model)
    architecture_json = _jsonable(dict(architecture))
    spec_json = _jsonable(spec_payload)
    training_json = _jsonable(training_payload)
    hashes = {
        "resolved_model_config_hash": stable_payload_sha256(normalized_model_json),
        "architecture_parameters_hash": stable_payload_sha256(architecture_json),
        "input_schema_hash": input_spec_sha256(spec),
        "training_config_hash": stable_payload_sha256(training_json),
    }
    enriched_run = {
        **validated_run,
        "config_hash": str(config_hash),
        "registry_hash": str(registry_hash),
        "source_snapshot_hash": str(source_snapshot_hash),
        "manual_selection_hash": str(manual_selection_hash),
        "oof_evidence_hash": str(oof_evidence_hash),
        **hashes,
        "representation_mode": spec.mode.value,
    }
    enriched_run_json = _jsonable(enriched_run)
    return FinalRefitBinding(
        resolved_model_config=normalized_model_json,
        input_spec=spec_json,
        training_config=training_json,
        frozen_run_provenance=enriched_run_json,
        **hashes,
        frozen_run_provenance_hash=stable_payload_sha256(enriched_run_json),
    )

@dataclass(frozen=True)
class FinalRefitExecution:
    """Completed full-cohort fit; never a replacement for archived OOF evidence."""

    plan: FinalRefitPlan
    result: Any
    scope: Any
    dataset_hash: str
    binding: FinalRefitBinding
    execution_hash: str

@dataclass(frozen=True)
class FrozenRepresentationTransformArchive:
    """All-cohort fitted transforms at the already-transformed model boundary."""

    representation_mode: str
    input_schema_hash: str
    fitted_on_participant_ids: tuple[str, ...]
    fitted_artifacts: Mapping[str, Any]
    provenance: Mapping[str, Any]
    source_records_hash: str
    dataset_hash: str
    boundary: str = "already_preprocessed_and_fitted_transforms_applied_model_input"

    @staticmethod
    def _raw_imu_is_explicit_noop(provenance: Mapping[str, Any]) -> bool:
        """Recognize the V2 no-fitted-object raw-IMU sentinel."""
        raw_imu = provenance.get("raw_imu")
        return isinstance(raw_imu, Mapping) and (
            raw_imu.get("schema_version") == "not_applicable_all8_window_normalized_v1"
            and raw_imu.get("artifact_sha256") is None
            and tuple(raw_imu.get("fitted_on_participant_ids", ())) == ()
            and raw_imu.get("strategy") == "none_after_all8_per_window_robust"
            and raw_imu.get("parameters") is None
        )

    def __post_init__(self) -> None:
        mode = str(self.representation_mode)
        if mode not in {"raw", "feature_vector", "feature_matrix", "fusion"}:
            raise ValueError("unsupported final transform representation_mode")
        participants = tuple(sorted(set(map(str, self.fitted_on_participant_ids))))
        if len(participants) != 29:
            raise ValueError("final transform archive must be fitted on exactly 29 participants")
        _require_hashes(self, ("input_schema_hash", "source_records_hash", "dataset_hash"))
        if not isinstance(self.provenance, Mapping) or not self.provenance:
            raise ValueError("final transform archive requires explicit provenance")
        provenance = dict(self.provenance)
        artifacts = dict(self.fitted_artifacts)
        expected = {
            "raw": {"raw_imu"},
            "feature_vector": set(),
            "feature_matrix": {"feature_vector", "engineering"},
            "fusion": {"raw_imu", "feature_vector"},
        }[mode]
        if mode in {"raw", "fusion"} and self._raw_imu_is_explicit_noop(provenance):
            expected.remove("raw_imu")
        if set(artifacts) != expected:
            raise ValueError("final fitted transform set differs from the representation contract")
        for name, artifact in artifacts.items():
            if tuple(sorted(set(map(str, getattr(artifact, "fitted_on_participant_ids", ()))))) != participants:
                raise ValueError(f"fitted transform {name!r} is not bound to the exact all-29 roster")
            validate = getattr(artifact, "validate", None)
            if callable(validate):
                validate()
        if self.boundary != "already_preprocessed_and_fitted_transforms_applied_model_input":
            raise ValueError("final transform archive boundary drift")
        object.__setattr__(self, "representation_mode", mode)
        object.__setattr__(self, "fitted_on_participant_ids", participants)
        object.__setattr__(self, "fitted_artifacts", artifacts)
        object.__setattr__(self, "provenance", provenance)

    def transform_inputs(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Validate and preserve already-transformed model inputs."""
        if not isinstance(inputs, Mapping) or not inputs:
            raise TypeError("final bundle model inputs must be a non-empty mapping")
        copied = {str(name): np.asarray(value) for name, value in inputs.items()}
        for name, value in copied.items():
            if np.isinf(value).any() or (
                self.representation_mode != "feature_vector" and (not np.isfinite(value).all())
            ):
                raise ValueError(f"final bundle model input {name!r} violates representation finiteness")
        return copied

def _golden_inputs_hash(inputs: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(b"ppg_frailty_final_golden_inputs_v2\x00")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("golden_inputs must be a non-empty mapping")
    for name in sorted(inputs):
        value = np.ascontiguousarray(np.asarray(inputs[name]))
        if np.isinf(value).any():
            raise ValueError("golden_inputs cannot contain infinity")
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()

@dataclass(frozen=True)
class TrustedFinalBundleMaterialization:
    """Resolved publication payload for one completed refit."""

    metadata: Mapping[str, Any]
    golden_inputs: Mapping[str, np.ndarray]
    transforms: FrozenRepresentationTransformArchive
    pipeline_adapter: Any
    execution_hash: str
    dataset_hash: str
    source_records_hash: str
    golden_inputs_hash: str
    materialization_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", validate_bundle_metadata(self.metadata))
        object.__setattr__(
            self,
            "golden_inputs",
            {str(name): np.asarray(value) for name, value in self.golden_inputs.items()},
        )

def _bind_trusted_final_bundle_materialization(
    execution: FinalRefitExecution,
    *,
    metadata: Mapping[str, Any],
    golden_inputs: Mapping[str, np.ndarray],
    transforms: FrozenRepresentationTransformArchive,
    pipeline_adapter: Any,
    source_records_hash: str,
) -> TrustedFinalBundleMaterialization:
    """Bind resolved publication components to one refit execution."""
    normalized_metadata = validate_bundle_metadata(metadata)
    golden_hash = _golden_inputs_hash(golden_inputs)
    payload = {
        "metadata": normalized_metadata,
        "execution_hash": execution.execution_hash,
        "dataset_hash": execution.dataset_hash,
        "source_records_hash": str(source_records_hash),
        "golden_inputs_hash": golden_hash,
        "transform_input_schema_hash": transforms.input_schema_hash,
        "transform_boundary": transforms.boundary,
    }
    return TrustedFinalBundleMaterialization(
        metadata=normalized_metadata,
        golden_inputs=golden_inputs,
        transforms=transforms,
        pipeline_adapter=pipeline_adapter,
        execution_hash=execution.execution_hash,
        dataset_hash=execution.dataset_hash,
        source_records_hash=str(source_records_hash),
        golden_inputs_hash=golden_hash,
        materialization_hash=stable_payload_sha256(payload),
    )

def _save_trusted_final_refit_bundle(
    execution: FinalRefitExecution, directory: str | Path, *, materialization: TrustedFinalBundleMaterialization
) -> LoadedBundle:
    """Publish one completed full-cohort refit bundle."""
    plan = execution.plan
    final_identity = {
        "purpose": plan.purpose,
        "performance_evidence": plan.performance_evidence,
        "manual_selection_hash": plan.manual_selection_hash,
        "selection_record_file_sha256": materialization.metadata["selection_record_file_sha256"],
        "oof_evidence_hash": plan.oof_evidence_hash,
        "config_hash": plan.config_hash,
        "registry_hash": plan.registry_hash,
        "dataset_hash": execution.dataset_hash,
        "source_snapshot_hash": plan.source_snapshot_hash,
        "resolved_model_config_hash": plan.resolved_model_config_hash,
        "architecture_parameters_hash": plan.architecture_parameters_hash,
        "input_schema_hash": plan.input_schema_hash,
        "training_config_hash": plan.training_config_hash,
        "frozen_run_provenance_hash": plan.frozen_run_provenance_hash,
        "scope_membership_hash": execution.scope.membership_hash,
        "execution_hash": execution.execution_hash,
        "bundle_materialization_hash": materialization.materialization_hash,
        "source_records_hash": materialization.source_records_hash,
        "golden_inputs_hash": materialization.golden_inputs_hash,
        "participant_count": len(plan.participant_ids),
        "participant_ids": plan.participant_ids,
        "training_seeds": plan.training_seeds,
        "epoch_rule": plan.epoch_rule,
        "fixed_epochs": plan.fixed_epochs,
        "model_id": plan.model_id,
        "model_kind": plan.model_kind,
        "model_family": plan.model_family,
        "representation_mode": plan.representation_mode,
    }
    resolved_metadata = {
        **dict(materialization.metadata),
        "pipeline_generation": "final_pipeline_v2",
        "representation_mode": plan.representation_mode,
        "config_hash": plan.config_hash,
        "fold_hash": execution.scope.fold_hash,
        "run_hash": execution.execution_hash,
        "source_snapshot_hash": plan.source_snapshot_hash,
        "final_refit_identity": final_identity,
    }
    target = _save_bundle_impl(
        execution.result.model,
        directory,
        model_config=execution.binding.resolved_model_config,
        input_spec=execution.binding.input_spec,
        metadata=resolved_metadata,
        golden_inputs=materialization.golden_inputs,
        transforms=materialization.transforms,
        pipeline_adapter=materialization.pipeline_adapter,
        bundle_kind=TRUSTED_FINAL_REFIT_BUNDLE_KIND,
    )
    loaded = load_bundle(target)
    assert_golden_parity(loaded, atol=FINAL_BUNDLE_PARITY_ATOL)
    archived = loaded.manifest["metadata"].get("final_refit_identity")
    if archived != _jsonable(final_identity):
        raise RuntimeError("final-refit bundle identity changed during save/reload")
    return loaded

def save_final_refit_bundle(
    execution: FinalRefitExecution,
    directory: str | Path,
    *,
    materialization: TrustedFinalBundleMaterialization,
) -> LoadedBundle:
    """Save a completed refit using its already-materialized inference boundary."""
    return _save_trusted_final_refit_bundle(execution, directory, materialization=materialization)

def _execute_prepared_full_cohort_refit(
    plan: FinalRefitPlan,
    trainer: Any,
    full_dataset: Any,
    *,
    registry_hash: str,
    binding: FinalRefitBinding,
    model_factory: Any = None,
    estimator: Any = None,
) -> FinalRefitExecution:
    """Fit the resolved model on all participants using the unchanged trainer path."""
    from .trainer import FullCohortRefitScope, dataset_binding_hash

    spec = ModelInputSpec.from_value(binding.input_spec)
    normalized_model = dict(binding.resolved_model_config)
    scope = FullCohortRefitScope(
        participant_ids=plan.participant_ids,
        registry_hash=str(registry_hash),
        config_hash=plan.config_hash,
        oof_evidence_hash=plan.oof_evidence_hash,
    ).bind_training_dataset(full_dataset)
    declared_architecture = normalized_model["architecture_parameters"]
    if plan.model_family == "deep":
        if model_factory is None or estimator is not None:
            raise ValueError("deep final refit requires exactly model_factory")
        preview = model_factory()
        validate_resolved_architecture(preview, declared_architecture, spec)
        result = trainer.fit(model_factory, full_dataset, scope)
    else:
        if estimator is None or model_factory is not None:
            raise ValueError("estimator final refit requires exactly estimator")
        validate_resolved_architecture(estimator, declared_architecture, spec)
        result = trainer.fit_estimator(estimator, full_dataset, scope)
    validate_resolved_architecture(result.model, declared_architecture, spec)
    dataset_hash = dataset_binding_hash(full_dataset)
    execution_hash = stable_payload_sha256(
        {
            "plan": asdict(plan),
            "dataset_hash": dataset_hash,
            "scope_membership_hash": scope.membership_hash,
            "model_state_hash": result.provenance.state_hash,
            "binding_hash": binding.frozen_run_provenance_hash,
        }
    )
    return FinalRefitExecution(
        plan=plan, result=result, scope=scope, dataset_hash=dataset_hash, binding=binding, execution_hash=execution_hash
    )

def execute_full_cohort_refit(
    plan: FinalRefitPlan,
    trainer: Any,
    full_dataset: Any,
    *,
    registry_hash: str,
    binding: FinalRefitBinding,
    model_factory: Any = None,
    estimator: Any = None,
) -> FinalRefitExecution:
    """Fit one resolved case on the full cohort without an authority wrapper."""
    return _execute_prepared_full_cohort_refit(
        plan,
        trainer,
        full_dataset,
        registry_hash=registry_hash,
        binding=binding,
        model_factory=model_factory,
        estimator=estimator,
    )

def _jsonable(value: Any) -> Any:
    """Convert dataclasses/arrays/scalars to strict JSON / 转换为严格 JSON。"""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (not np.isfinite(value)):
        raise TypeError("bundle JSON payload cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"bundle metadata is not JSON serialisable: {type(value).__name__}")

def validate_bundle_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the shared metadata contract used by every representation."""
    missing = sorted(REQUIRED_METADATA - set(metadata))
    if missing:
        raise ValueError(f"bundle metadata is missing required fields: {missing}")
    normalized = _jsonable(metadata)
    for name in _STRUCTURED_METADATA:
        if not isinstance(normalized[name], dict):
            raise TypeError(f"bundle metadata field {name!r} must be a mapping")
    return normalized

def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical JSON atomically / 原子写入规范 JSON。"""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
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

def _prediction_device(model: Any) -> str:
    """Return the single device on which golden probabilities are computed."""
    if not _is_torch_model(model):
        return "cpu"
    devices = {str(value.device) for value in (*tuple(model.parameters()), *tuple(model.buffers()))}
    if len(devices) > 1:
        raise ValueError("bundle golden parity does not support a multi-device model")
    return next(iter(devices), "cpu")

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
            tensors = {key: torch.as_tensor(value, device=device) for key, value in inputs.items()}
            if "window_bag" in tensors:
                logits = model(
                    tensors["window_bag"].float(),
                    tensors["window_mask"].bool(),
                    tensors["file_features"].float(),
                    tensors.get("sample_mask", None).bool() if tensors.get("sample_mask") is not None else None,
                )
                probability = torch.softmax(logits, dim=-1)
            elif hasattr(model, "predict_probabilities"):
                probability = model.predict_probabilities(
                    tensors["x"].float(), tensors.get("mask", None).bool() if tensors.get("mask") is not None else None
                )
            else:
                logits = model(
                    tensors["x"].float(), tensors.get("mask", None).bool() if tensors.get("mask") is not None else None
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

def _validate_trusted_final_refit_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility reader for the retained final-refit identity payload."""
    identity = manifest.get("metadata", {}).get("final_refit_identity", {})
    return _jsonable(dict(identity))

def _save_bundle_impl(
    model: Any,
    directory: str | Path,
    *,
    model_config: Mapping[str, Any],
    input_spec: ModelInputSpec | Mapping[str, Any],
    metadata: Mapping[str, Any],
    golden_inputs: Mapping[str, np.ndarray],
    transforms: Any = None,
    pipeline_adapter: Any = None,
    bundle_kind: str,
) -> Path:
    """Save an immutable bundle, reload it and enforce golden parity.

    保存不可变 bundle，随后立即重载并强制 golden parity。缺少关键 provenance 字段、
    文件哈希不匹配或往返预测偏差都会关闭失败。
    """
    normalized_metadata = validate_bundle_metadata(metadata)
    if not golden_inputs or ("x" not in golden_inputs and "window_bag" not in golden_inputs):
        raise ValueError("golden_inputs must contain x or window_bag")
    target = Path(directory)
    if target.exists():
        raise FileExistsError(f"bundle target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=str(target.parent)))
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
        golden_prediction_device = _prediction_device(model)
        expected = _predict_model(model, _apply_transforms(transforms, golden_inputs))
        if expected.ndim != 2 or not np.isfinite(expected).all():
            raise ValueError("golden prediction must be finite [sample,class]")
        golden_path = staging / "golden.npz"
        arrays = {f"input__{key}": np.asarray(value) for key, value in golden_inputs.items()}
        arrays["expected_probabilities"] = expected
        np.savez_compressed(golden_path, **arrays)
        file_hashes[golden_path.name] = sha256_file(golden_path)
        spec = ModelInputSpec.from_value(input_spec)
        input_hash = input_spec_sha256(spec)
        adapter_contract = _pipeline_adapter_contract(pipeline_adapter, input_spec=spec, input_hash=input_hash)
        normalized_model_config = normalize_model_config(model_config)
        manifest = {
            "bundle_format": BUNDLE_FORMAT_VERSION,
            "bundle_kind": bundle_kind,
            "kind": kind,
            "state_file": state_name,
            "model_config": _jsonable(normalized_model_config),
            "canonical_model_name": normalized_model_config["canonical_model_name"],
            "machine_model_id": normalized_model_config["model_id"],
            "pipeline_generation": normalized_metadata["pipeline_generation"],
            "config_hash": normalized_metadata["config_hash"],
            "balance_hash": normalized_metadata["balance_hash"],
            "run_hash": normalized_metadata["run_hash"],
            "source_snapshot_hash": normalized_metadata["source_snapshot_hash"],
            "input_spec": _jsonable(spec),
            "input_spec_hash": input_hash,
            "metadata": normalized_metadata,
            "required_metadata_fields": sorted(REQUIRED_METADATA),
            "file_hashes": file_hashes,
            "golden_parity_atol": FINAL_BUNDLE_PARITY_ATOL,
            "golden_prediction_device": golden_prediction_device,
            "golden_device_policy": "same_device_before_and_after_serialization",
            "golden_case_hash": file_hashes[golden_path.name],
            "pipeline_adapter_boundary": "serialized_raw_record_to_model_input_mapping"
            if pipeline_adapter is not None
            else "not_bundled",
            "pipeline_adapter_contract": adapter_contract,
            "transactional_save": "same_filesystem_staging_then_atomic_rename",
            "joblib_trust_boundary": "hash_integrity_is_not_authentication; load_only_user-verified_local_source",
        }
        _atomic_json(staging / "manifest.json", manifest)
        loaded = load_bundle(staging)
        if kind == "torch":
            loaded.model.to(golden_prediction_device)
        assert_golden_parity(loaded, atol=FINAL_BUNDLE_PARITY_ATOL)
        if target.exists():
            raise FileExistsError(f"bundle target appeared during staging: {target}")
        os.rename(staging, target)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)

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
) -> Path:
    """Publish a generic research bundle that cannot impersonate a final refit."""
    return _save_bundle_impl(
        model,
        directory,
        model_config=model_config,
        input_spec=input_spec,
        metadata=metadata,
        golden_inputs=golden_inputs,
        transforms=transforms,
        pipeline_adapter=pipeline_adapter,
        bundle_kind=GENERIC_BUNDLE_KIND,
    )

def verify_bundle(
    directory: str | Path, load_model: bool = False, *, expected_metadata: Mapping[str, Any] | None = None
) -> dict[str, Any] | LoadedBundle:
    """Verify one bundle and optionally load its executable state."""
    target = Path(directory)
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("bundle manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_format") != BUNDLE_FORMAT_VERSION:
        raise ValueError("unsupported bundle format")
    metadata = validate_bundle_metadata(manifest.get("metadata", {}))
    spec = ModelInputSpec.from_value(manifest.get("input_spec", {}))
    input_hash = input_spec_sha256(spec)
    adapter_contract = manifest.get("pipeline_adapter_contract")
    if expected_metadata is not None:
        expected = _jsonable(expected_metadata)
        for name, value in expected.items():
            if name not in metadata or metadata[name] != value:
                raise ValueError(f"bundle metadata mismatch for expected field: {name}")
    file_hashes = manifest.get("file_hashes", {})
    state_name = manifest.get("state_file")
    if not isinstance(state_name, str) or Path(state_name).name != state_name:
        raise ValueError("bundle state_file must be a plain filename")
    required_files = {state_name, "golden.npz"}
    if not required_files <= set(file_hashes):
        raise ValueError("bundle does not hash every required payload file")
    for name, expected_hash in file_hashes.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("bundle file_hashes contains an unsafe filename")
        path = target / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"bundle file integrity check failed: {name}")
    if not load_model:
        return manifest
    state_path = target / state_name
    if manifest["kind"] == "torch":
        import torch

        model = create_model(manifest["model_config"], manifest["input_spec"])
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
    elif manifest["kind"] == "estimator":
        model = joblib.load(state_path)
    else:
        raise ValueError("unknown bundle kind")
    transforms_path = target / "transforms.joblib"
    transforms = joblib.load(transforms_path) if transforms_path.name in file_hashes else None
    adapter_path = target / "pipeline_adapter.joblib"
    pipeline_adapter = joblib.load(adapter_path) if adapter_path.name in file_hashes else None
    actual_adapter_contract = _pipeline_adapter_contract(pipeline_adapter, input_spec=spec, input_hash=input_hash)
    if actual_adapter_contract != adapter_contract:
        raise ValueError("serialized pipeline adapter disagrees with bundle contract")
    return LoadedBundle(
        model=model, transforms=transforms, manifest=manifest, directory=target, pipeline_adapter=pipeline_adapter
    )

def load_bundle(directory: str | Path, *, expected_metadata: Mapping[str, Any] | None = None) -> LoadedBundle:
    """Load executable state after :func:`verify_bundle` succeeds."""
    loaded = verify_bundle(directory, load_model=True, expected_metadata=expected_metadata)
    assert isinstance(loaded, LoadedBundle)
    return loaded

def predict_bundle(bundle: LoadedBundle | str | Path, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Predict through a validated loaded bundle / 通过已校验 bundle 预测。"""
    loaded = load_bundle(bundle) if isinstance(bundle, (str, Path)) else bundle
    transformed = _apply_transforms(loaded.transforms, inputs)
    probability = _predict_model(loaded.model, transformed)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-06):
        raise RuntimeError("bundle prediction is not a finite probability matrix")
    return probability

def _pipeline_adapter_contract(
    adapter: Any, *, input_spec: ModelInputSpec | Mapping[str, Any], input_hash: str
) -> dict[str, Any]:
    """Validate the serialized adapter against its exact model-input boundary."""
    if adapter is None:
        return {"status": "not_bundled"}
    spec = ModelInputSpec.from_value(input_spec)
    representation_mode = str(getattr(adapter, "representation_mode", ""))
    adapter_hash = str(getattr(adapter, "input_schema_hash", ""))
    raw_roles = getattr(adapter, "allowed_role_families", None)
    if representation_mode != spec.mode.value:
        raise ValueError("pipeline adapter representation_mode disagrees with input_spec")
    if adapter_hash != input_hash:
        raise ValueError("pipeline adapter input_schema_hash disagrees with input_spec")
    if raw_roles is None:
        raise ValueError("pipeline adapter must declare allowed_role_families")
    from .aggregation import canonical_role_family

    allowed_roles = tuple((canonical_role_family(value) for value in raw_roles))
    if not allowed_roles or len(allowed_roles) != len(set(allowed_roles)):
        raise ValueError("bundle adapter allowed_role_families must be non-empty and unique")
    if not hasattr(adapter, "transform_record") and (not callable(adapter)):
        raise TypeError("pipeline adapter must be callable or expose transform_record")
    return {
        "status": "bundled",
        "representation_mode": representation_mode,
        "input_schema_hash": adapter_hash,
        "allowed_role_families": list(allowed_roles),
        "boundary": str(getattr(adapter, "boundary", "")),
    }

def predict_bundle_raw(bundle: LoadedBundle | str | Path, raw_record: Any) -> np.ndarray:
    """Run a serialized raw-record adapter before normal bundle inference."""
    loaded = load_bundle(bundle) if isinstance(bundle, (str, Path)) else bundle
    adapter = loaded.pipeline_adapter
    if adapter is None:
        raise RuntimeError("bundle does not contain a raw-record pipeline adapter")
    spec = ModelInputSpec.from_value(loaded.manifest.get("input_spec", {}))
    input_hash = input_spec_sha256(spec)
    contract = loaded.manifest.get("pipeline_adapter_contract")
    try:
        actual_contract = _pipeline_adapter_contract(adapter, input_spec=spec, input_hash=input_hash)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("bundle pipeline adapter is not bound to its input schema") from exc
    if not isinstance(contract, dict) or actual_contract != contract:
        raise RuntimeError("bundle pipeline adapter is not bound to its input schema")
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
            key.removeprefix("input__"): np.asarray(archive[key]) for key in archive.files if key.startswith("input__")
        }
    tolerance = float(loaded.manifest["golden_parity_atol"]) if atol is None else float(atol)
    if loaded.manifest.get("kind") == "torch":
        import torch

        device = str(loaded.manifest.get("golden_prediction_device", "cpu"))
        if device.startswith("cuda") and (not torch.cuda.is_available()):
            raise RuntimeError(
                "golden parity requires the original CUDA prediction device; "
                "CPU portability is a separate inference contract"
            )
        loaded.model.to(device)
    actual = predict_bundle(loaded, inputs)
    if expected.shape != actual.shape or not np.allclose(expected, actual, atol=tolerance, rtol=0.0):
        maximum = float(np.max(np.abs(expected - actual))) if expected.shape == actual.shape else float("inf")
        raise RuntimeError(f"golden prediction parity failed; maximum absolute error={maximum}")

def assert_repeated_bundle_parity(
    bundle: str | Path, *, iterations: int = 10000, reload_each_iteration: bool = True
) -> None:
    """Stress repeated load/predict without repeatedly saving to disk."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    path = Path(bundle)
    initial = load_bundle(path)
    with np.load(initial.directory / "golden.npz", allow_pickle=False) as archive:
        expected = np.asarray(archive["expected_probabilities"], dtype=np.float64)
        inputs = {
            key.removeprefix("input__"): np.asarray(archive[key]) for key in archive.files if key.startswith("input__")
        }
    tolerance = float(initial.manifest["golden_parity_atol"])
    loaded = initial
    for index in range(iterations):
        if reload_each_iteration and index:
            loaded = load_bundle(path)
        actual = predict_bundle(loaded, inputs)
        if expected.shape != actual.shape or not np.allclose(expected, actual, atol=tolerance, rtol=0.0):
            raise RuntimeError(f"bundle repeated parity failed at iteration {index + 1}")

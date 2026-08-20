"""Integrity-checked model bundles with golden prediction parity.

带完整性校验与 golden prediction 一致性的模型 bundle。
"""

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
    normalize_model_id,
    resolve_seed_policy,
    validate_frozen_model_run_provenance,
    validate_resolved_architecture,
)
from ..module_registry import model_factory_contract
from ..provenance import sha256_file, stable_payload_sha256


BUNDLE_FORMAT_VERSION = "ppg_frailty_bundle_parity_v3"
FINAL_BUNDLE_PARITY_ATOL = 1e-6
GENERIC_BUNDLE_KIND = "generic_research"
TRUSTED_FINAL_REFIT_BUNDLE_KIND = "trusted_final_refit_v2"
_TRUSTED_FINAL_IDENTITY_FIELDS = frozenset(
    {
        "purpose", "performance_evidence", "manual_selection_hash",
        "selection_record_file_sha256", "oof_evidence_hash", "config_hash",
        "registry_hash", "dataset_hash", "source_snapshot_hash",
        "resolved_model_config_hash",
        "architecture_parameters_hash", "input_schema_hash",
        "training_config_hash", "frozen_run_provenance_hash",
        "scope_membership_hash", "execution_hash",
        "bundle_materialization_hash", "source_records_hash",
        "golden_inputs_hash", "participant_count", "participant_ids",
        "training_seeds", "epoch_rule", "fixed_epochs", "model_id",
        "model_kind", "model_family", "representation_mode",
    }
)
_ESTIMATOR_NOT_APPLICABLE = "not_applicable_estimator_native"
_ENSEMBLE_MODEL_KINDS = frozenset({"ensemble", "five_member_ensemble"})


def _is_ensemble_model_kind(value: object) -> bool:
    """Recognize the generic kind and its historical five-member alias."""

    return str(value) in _ENSEMBLE_MODEL_KINDS


def _model_uses_estimator(model_id: str) -> bool:
    """Resolve estimator behavior from the registry, not a duplicated ID set."""

    return model_factory_contract(str(model_id))["execution_backend"] == "estimator"


def _model_is_ensemble(model_id: str) -> bool:
    """Resolve ensemble behavior from the model factory-field capability."""

    return "member_seeds" in set(
        model_factory_contract(str(model_id))["factory_fields"]
    )


def _validated_refit_seed_roster(values: object) -> tuple[int, ...]:
    """Return a non-empty unique roster representable in persisted int64 fields."""

    if not isinstance(values, (list, tuple)):
        raise ValueError(
            "final-refit training_seeds must be an ordered list or tuple"
        )
    raw_values = tuple(values)
    seeds: list[int] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("final-refit training_seeds must contain integers")
        try:
            seed = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("final-refit training_seeds must contain integers") from exc
        if isinstance(value, (float, np.floating)) and (
            not np.isfinite(value) or float(value) != float(seed)
        ):
            raise ValueError("final-refit training_seeds must contain finite integers")
        if seed < 0 or seed > 0xFFFF_FFFF:
            raise ValueError(
                "final-refit training_seeds must be in the executable uint32 range"
            )
        seeds.append(seed)
    roster = tuple(seeds)
    if not roster or len(roster) != len(set(roster)):
        raise ValueError("final-refit training_seeds must be non-empty and unique")
    return roster
REQUIRED_RUNTIME_ENVIRONMENT_KEYS = (
    "python",
    "python_implementation",
    "numpy",
    "scipy",
    "scikit_learn",
    "joblib",
    "torch",
)
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
    "pipeline_generation",
    "config_hash",
    "balance_hash",
    "run_hash",
    "source_snapshot_hash",
    "code_version",
    "environment",
    "dependency_status",
    "serialization_trust",
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
    "serialization_trust",
    "golden_case",
}


def current_runtime_environment() -> dict[str, str]:
    """Return descriptive runtime versions for troubleshooting.

    Bundle loading records these values but deliberately does not require exact
    version equality.
    """

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
        if not str(self.purpose).strip() or not str(self.model_id).strip():
            raise ValueError("final refit requires purpose and model_id")
        if len(participants) != 29:
            raise ValueError("V2 final refit must use exactly all 29 internal participants")
        if self.model_kind == "single_model":
            if len(seeds) != 1:
                raise ValueError("single-model final refit requires exactly one training seed")
        elif _is_ensemble_model_kind(self.model_kind):
            pass
        else:
            raise ValueError(
                "model_kind must be single_model, ensemble, or the legacy "
                "five_member_ensemble alias"
            )
        if self.model_family == "deep":
            if self.epoch_rule != "fixed_epoch":
                raise ValueError("deep final refit requires epoch_rule=fixed_epoch")
            if (
                isinstance(self.fixed_epochs, (bool, np.bool_))
                or not isinstance(self.fixed_epochs, (int, np.integer))
                or int(self.fixed_epochs) <= 0
            ):
                raise ValueError(
                    "deep final refit fixed_epochs must be a positive integer"
                )
            object.__setattr__(self, "fixed_epochs", int(self.fixed_epochs))
        elif self.model_family == "classical_or_rocket":
            if self.epoch_rule != "not_applicable" or self.fixed_epochs is not None:
                raise ValueError("classical/ROCKET final refit must not carry a fake epoch")
            if self.model_kind != "single_model":
                raise ValueError("classical/ROCKET final refit is not the Inception ensemble")
        else:
            raise ValueError("model_family must be deep or classical_or_rocket")
        if self.representation_mode not in {
            "raw", "feature_vector", "feature_matrix", "fusion"
        }:
            raise ValueError("final refit representation_mode is invalid")
        for name in (
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
        ):
            digest = str(getattr(self, name))
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
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


def canonical_input_spec_payload(
    input_spec: ModelInputSpec | Mapping[str, Any],
) -> dict[str, Any]:
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
    """Create canonical hashes that a FinalRefitPlan must copy exactly."""

    for name, digest in (
        ("config_hash", config_hash),
        ("registry_hash", registry_hash),
        ("source_snapshot_hash", source_snapshot_hash),
        ("manual_selection_hash", manual_selection_hash),
        ("oof_evidence_hash", oof_evidence_hash),
    ):
        if len(str(digest)) != 64 or any(
            character not in "0123456789abcdef" for character in str(digest)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
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
    if dict(validated_run["architecture_parameters"]) != dict(architecture):
        raise ValueError("frozen run provenance architecture differs from model config")
    expected_channels = (
        tuple(spec.feature_names)
        if spec.mode.value == "feature_vector"
        else tuple(spec.channel_schema)
    )
    if tuple(validated_run["input_channels_order"]) != expected_channels:
        raise ValueError("frozen run input channel/order differs from input_spec")
    _canonical_name, machine_id = normalize_model_id(
        str(normalized_model["model_id"])
    )
    estimator_model = _model_uses_estimator(machine_id)
    training_pairs = {
        "class_weighting": "class_weighting",
        "sampler": "sampler",
    }
    if not estimator_model:
        training_pairs.update(
            {
                "loss": "loss",
                "optimizer": "optimizer",
                "learning_rate": "learning_rate",
                "weight_decay": "weight_decay",
                "label_smoothing": "label_smoothing",
            }
        )
    for provenance_name, training_name in training_pairs.items():
        if validated_run[provenance_name] != training_payload[training_name]:
            raise ValueError(
                f"frozen run {provenance_name} differs from training_config"
            )
    if estimator_model:
        expected_not_applicable = {
            "loss": _ESTIMATOR_NOT_APPLICABLE,
            "optimizer": _ESTIMATOR_NOT_APPLICABLE,
            "learning_rate": _ESTIMATOR_NOT_APPLICABLE,
            "weight_decay": _ESTIMATOR_NOT_APPLICABLE,
            "dropout": _ESTIMATOR_NOT_APPLICABLE,
            "label_smoothing": _ESTIMATOR_NOT_APPLICABLE,
        }
        for name, expected in expected_not_applicable.items():
            if validated_run[name] != expected:
                raise ValueError(
                    f"frozen estimator run {name} must be explicitly not applicable"
                )
        expected_gradient = {
            "enabled": False,
            "max_norm": None,
            "status": _ESTIMATOR_NOT_APPLICABLE,
        }
    else:
        expected_gradient = {
            "enabled": training_payload["gradient_clip_norm"] is not None,
            "max_norm": training_payload["gradient_clip_norm"],
        }
    if dict(validated_run["gradient_clipping"]) != expected_gradient:
        raise ValueError("frozen run gradient clipping differs from training_config")
    # ShapeFormer configs contain dataclass discovery banks with NumPy arrays.
    # Canonicalise them using the same strict JSON conversion used by bundle
    # manifests; NaN/Inf remain forbidden.
    normalized_model_json = _jsonable(normalized_model)
    architecture_json = _jsonable(dict(architecture))
    spec_json = _jsonable(spec_payload)
    training_json = _jsonable(training_payload)
    resolved_hash = stable_payload_sha256(normalized_model_json)
    architecture_hash = stable_payload_sha256(architecture_json)
    input_hash = input_spec_sha256(spec)
    training_hash = stable_payload_sha256(training_json)
    enriched_run = {
        **validated_run,
        "config_hash": str(config_hash),
        "registry_hash": str(registry_hash),
        "source_snapshot_hash": str(source_snapshot_hash),
        "manual_selection_hash": str(manual_selection_hash),
        "oof_evidence_hash": str(oof_evidence_hash),
        "resolved_model_config_hash": resolved_hash,
        "architecture_parameters_hash": architecture_hash,
        "input_schema_hash": input_hash,
        "training_config_hash": training_hash,
        "representation_mode": spec.mode.value,
    }
    enriched_run_json = _jsonable(enriched_run)
    run_hash = stable_payload_sha256(enriched_run_json)
    return FinalRefitBinding(
        resolved_model_config=normalized_model_json,
        input_spec=spec_json,
        training_config=training_json,
        frozen_run_provenance=enriched_run_json,
        resolved_model_config_hash=resolved_hash,
        architecture_parameters_hash=architecture_hash,
        input_schema_hash=input_hash,
        training_config_hash=training_hash,
        frozen_run_provenance_hash=run_hash,
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
    """All-29 fitted representation artifacts at an already-transformed boundary.

    Deployment-device preprocessing remains deferred under V2-026.  Consequently
    this archive preserves the exact fitted objects and their provenance, while
    ``transform_inputs`` is deliberately an identity operation on model-ready
    inputs.  This prevents a saved scaler from being applied twice during golden
    parity or deployment inference.
    """

    representation_mode: str
    input_schema_hash: str
    fitted_on_participant_ids: tuple[str, ...]
    fitted_artifacts: Mapping[str, Any]
    provenance: Mapping[str, Any]
    source_records_hash: str
    dataset_hash: str
    boundary: str = "already_preprocessed_and_fitted_transforms_applied_model_input"

    def __post_init__(self) -> None:
        mode = str(self.representation_mode)
        if mode not in {"raw", "feature_vector", "feature_matrix", "fusion"}:
            raise ValueError("unsupported final transform representation_mode")
        participants = tuple(sorted(set(map(str, self.fitted_on_participant_ids))))
        if len(participants) != 29:
            raise ValueError("final transform archive must be fitted on exactly 29 participants")
        for name in ("input_schema_hash", "source_records_hash", "dataset_hash"):
            digest = str(getattr(self, name))
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        artifacts = dict(self.fitted_artifacts)
        expected_names = {
            "raw": {"raw_imu"},
            "feature_vector": set(),
            "feature_matrix": {"feature_vector", "engineering"},
            "fusion": {"raw_imu", "feature_vector"},
        }[mode]
        if set(artifacts) != expected_names:
            raise ValueError(
                "final fitted transform set differs from the representation contract"
            )
        for name, artifact in artifacts.items():
            fitted = tuple(
                sorted(set(map(str, getattr(artifact, "fitted_on_participant_ids", ()))))
            )
            if fitted != participants:
                raise ValueError(
                    f"fitted transform {name!r} is not bound to the exact all-29 roster"
                )
            validate = getattr(artifact, "validate", None)
            if callable(validate):
                validate()
        if not isinstance(self.provenance, Mapping) or not self.provenance:
            raise ValueError("final transform archive requires explicit provenance")
        if self.boundary != (
            "already_preprocessed_and_fitted_transforms_applied_model_input"
        ):
            raise ValueError("final transform archive boundary drift")
        object.__setattr__(self, "representation_mode", mode)
        object.__setattr__(self, "fitted_on_participant_ids", participants)
        object.__setattr__(self, "fitted_artifacts", artifacts)
        object.__setattr__(self, "provenance", dict(self.provenance))

    def transform_inputs(
        self, inputs: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Validate and preserve already-transformed model inputs."""

        if not isinstance(inputs, Mapping) or not inputs:
            raise TypeError("final bundle model inputs must be a non-empty mapping")
        copied = {str(name): np.asarray(value) for name, value in inputs.items()}
        for name, value in copied.items():
            if np.isinf(value).any() or (
                self.representation_mode != "feature_vector"
                and not np.isfinite(value).all()
            ):
                raise ValueError(
                    f"final bundle model input {name!r} violates representation finiteness"
                )
        return copied


def _golden_inputs_hash(inputs: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(b"ppg_frailty_final_golden_inputs_v2\0")
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
    """Internally derived publication payload bound to one completed refit."""

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
        metadata = validate_bundle_metadata(self.metadata)
        golden = {str(name): np.asarray(value) for name, value in self.golden_inputs.items()}
        observed_golden_hash = _golden_inputs_hash(golden)
        for name in (
            "execution_hash", "dataset_hash", "source_records_hash",
            "golden_inputs_hash", "materialization_hash",
        ):
            digest = str(getattr(self, name))
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if observed_golden_hash != self.golden_inputs_hash:
            raise ValueError("trusted final golden-input hash mismatch")
        payload = {
            "metadata": metadata,
            "execution_hash": self.execution_hash,
            "dataset_hash": self.dataset_hash,
            "source_records_hash": self.source_records_hash,
            "golden_inputs_hash": self.golden_inputs_hash,
            "transform_input_schema_hash": self.transforms.input_schema_hash,
            "transform_boundary": self.transforms.boundary,
        }
        if stable_payload_sha256(payload) != self.materialization_hash:
            raise ValueError("trusted final bundle materialization hash mismatch")
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "golden_inputs", golden)

    def assert_matches_execution(self, execution: FinalRefitExecution) -> None:
        plan = execution.plan
        metadata = dict(self.metadata)
        if self.execution_hash != execution.execution_hash:
            raise ValueError("final bundle payload execution hash mismatch")
        if self.dataset_hash != execution.dataset_hash:
            raise ValueError("final bundle payload dataset hash mismatch")
        if self.transforms.dataset_hash != execution.dataset_hash:
            raise ValueError("final transform archive dataset hash mismatch")
        if self.transforms.source_records_hash != self.source_records_hash:
            raise ValueError("final transform/source record hash mismatch")
        if self.transforms.input_schema_hash != plan.input_schema_hash:
            raise ValueError("final transform archive input schema mismatch")
        if self.transforms.fitted_on_participant_ids != plan.participant_ids:
            raise ValueError("final transform archive participant roster mismatch")
        expected = {
            "pipeline_generation": "final_pipeline_v2",
            "representation_mode": plan.representation_mode,
            "config_hash": plan.config_hash,
            "run_hash": execution.execution_hash,
            "source_snapshot_hash": plan.source_snapshot_hash,
        }
        for name, value in expected.items():
            if metadata.get(name) != value:
                raise ValueError(f"final bundle metadata differs from execution: {name}")
        adapter_mode = str(getattr(self.pipeline_adapter, "representation_mode", ""))
        adapter_hash = str(getattr(self.pipeline_adapter, "input_schema_hash", ""))
        if adapter_mode != plan.representation_mode or adapter_hash != plan.input_schema_hash:
            raise ValueError("final bundle adapter differs from execution input boundary")


def _bind_trusted_final_bundle_materialization(
    execution: FinalRefitExecution,
    *,
    metadata: Mapping[str, Any],
    golden_inputs: Mapping[str, np.ndarray],
    transforms: FrozenRepresentationTransformArchive,
    pipeline_adapter: Any,
    source_records_hash: str,
) -> TrustedFinalBundleMaterialization:
    """Bind internally derived publication components to one refit execution."""

    if not isinstance(execution, FinalRefitExecution):
        raise TypeError("final bundle materialization requires FinalRefitExecution")
    if not isinstance(transforms, FrozenRepresentationTransformArchive):
        raise TypeError("final bundle materialization requires frozen transform archive")
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
    result = TrustedFinalBundleMaterialization(
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
    result.assert_matches_execution(execution)
    return result


def _save_trusted_final_refit_bundle(
    execution: FinalRefitExecution,
    directory: str | Path,
    *,
    materialization: TrustedFinalBundleMaterialization,
) -> LoadedBundle:
    """Publish only one internally materialised, execution-bound final bundle."""

    if not isinstance(materialization, TrustedFinalBundleMaterialization):
        raise TypeError("save_final_refit_bundle requires trusted internal materialization")
    materialization.assert_matches_execution(execution)

    plan = execution.plan
    final_identity = {
        "purpose": plan.purpose,
        "performance_evidence": plan.performance_evidence,
        "manual_selection_hash": plan.manual_selection_hash,
        "selection_record_file_sha256":
            materialization.metadata["selection_record_file_sha256"],
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


def save_final_refit_bundle(*args: Any, **kwargs: Any) -> LoadedBundle:
    """Reject the obsolete caller-supplied final publication boundary."""

    del args, kwargs
    raise RuntimeError(
        "caller_supplied_final_bundle_materialization_disabled; use "
        "experiment.execute_final_refit_from_verified_artifacts"
    )


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
    """Internal implementation primitive for the verified artifact executor.

    This function intentionally remains outside the public training facade.  Its
    caller must be ``experiment.execute_final_refit_from_verified_artifacts``,
    which owns source verification and internal materialisation.
    """

    from .trainer import (
        FullCohortRefitScope,
        UnifiedTrainer,
        dataset_binding_hash,
    )

    if not isinstance(trainer, UnifiedTrainer):
        raise TypeError("trainer must be UnifiedTrainer")
    _canonical_plan_name, plan_machine_id = normalize_model_id(plan.model_id)
    plan_is_ensemble = _is_ensemble_model_kind(plan.model_kind)
    if plan_is_ensemble != _model_is_ensemble(plan_machine_id):
        raise ValueError(
            "final model_kind and registry ensemble capability disagree"
        )
    binding_pairs = {
        "resolved_model_config_hash": binding.resolved_model_config_hash,
        "architecture_parameters_hash": binding.architecture_parameters_hash,
        "input_schema_hash": binding.input_schema_hash,
        "training_config_hash": binding.training_config_hash,
        "frozen_run_provenance_hash": binding.frozen_run_provenance_hash,
    }
    for name, observed in binding_pairs.items():
        if getattr(plan, name) != observed:
            raise ValueError(f"FinalRefitPlan {name} differs from the supplied binding")
    if stable_payload_sha256(asdict(trainer.config)) != binding.training_config_hash:
        raise ValueError("executor trainer config differs from the final-refit binding")
    if (
        not plan_is_ensemble
        and int(trainer.config.seed) != plan.training_seeds[0]
    ):
        raise ValueError(
            "single-model trainer orchestration seed differs from FinalRefitPlan"
        )
    if plan.registry_hash != str(registry_hash):
        raise ValueError("FinalRefitPlan registry_hash differs from executor input")
    run = dict(binding.frozen_run_provenance)
    for name in (
        "config_hash",
        "registry_hash",
        "source_snapshot_hash",
        "manual_selection_hash",
        "oof_evidence_hash",
        "representation_mode",
    ):
        if run[name] != getattr(plan, name):
            raise ValueError(f"final refit run provenance {name} differs from plan")
    if tuple(run["random_seeds"]) != plan.training_seeds:
        raise ValueError("final refit run provenance seed roster differs from plan")
    try:
        resolved_run_seeds = (
            resolve_seed_policy(
                str(run["seed_policy"]), member_seeds=plan.training_seeds
            )
            if plan_is_ensemble
            else resolve_seed_policy(
                str(run["seed_policy"]), seed=plan.training_seeds[0]
            )
        )
    except ValueError as exc:
        raise ValueError(
            "final refit run provenance seed_policy differs from plan"
        ) from exc
    if resolved_run_seeds != plan.training_seeds:
        raise ValueError("final refit run provenance seed_policy differs from plan")
    epoch_identity = dict(run["epoch_rule"])
    if plan.model_family == "deep":
        if (
            epoch_identity.get("rule") != "fixed_epoch"
            or epoch_identity.get("fixed_epochs") != plan.fixed_epochs
        ):
            raise ValueError("final deep run provenance epoch identity differs from plan")
    elif (
        epoch_identity.get("rule") != "not_applicable"
        or epoch_identity.get("fixed_epochs") is not None
    ):
        raise ValueError("final classical/ROCKET run provenance must use no epoch")
    # Training balance and prediction aggregation are independent, hash-bound
    # modules.  The former controls row mass and train/inner diagnostics; the
    # latter controls the persisted participant view.  Binding both values is
    # required, but equality between them is not.
    aggregation = dict(run["aggregation"])
    if aggregation.get("balance_line") not in {
        "line_a_equal_files",
        "line_b_equal_role_families",
    }:
        raise ValueError("final run aggregation balance line is not registered")
    spec = ModelInputSpec.from_value(binding.input_spec)
    normalized_model = dict(binding.resolved_model_config)
    if normalized_model["model_id"] != normalize_model_id(plan.model_id)[1]:
        raise ValueError("final refit model config identity differs from plan")
    if plan_is_ensemble:
        if tuple(normalized_model.get("member_seeds", ())) != plan.training_seeds:
            raise ValueError("final ensemble model config seed roster differs from plan")
    elif int(normalized_model.get("seed", -1)) != plan.training_seeds[0]:
        raise ValueError("final single-model config seed differs from plan")
    scope = FullCohortRefitScope(
        participant_ids=plan.participant_ids,
        registry_hash=str(registry_hash),
        config_hash=plan.config_hash,
        oof_evidence_hash=plan.oof_evidence_hash,
    ).bind_training_dataset(full_dataset)
    if run["fold_hash"] != scope.fold_hash:
        raise ValueError("final refit run provenance fold_hash differs from all-29 scope")
    declared_architecture = normalized_model["architecture_parameters"]
    if plan.model_family == "deep":
        if model_factory is None or estimator is not None:
            raise ValueError("deep final refit requires exactly model_factory")
        if (
            trainer.config.epoch_rule != "fixed_epoch"
            or trainer.config.fixed_epochs != plan.fixed_epochs
        ):
            raise ValueError("deep trainer epoch identity differs from FinalRefitPlan")
        preview = model_factory()
        validate_resolved_architecture(preview, declared_architecture, spec)
        result = trainer.fit(model_factory, full_dataset, scope)
    else:
        if estimator is None or model_factory is not None:
            raise ValueError("classical/ROCKET final refit requires exactly estimator")
        validate_resolved_architecture(estimator, declared_architecture, spec)
        result = trainer.fit_estimator(estimator, full_dataset, scope)

    if result.provenance is None:
        raise RuntimeError("final refit did not produce fitted-object provenance")
    if result.provenance.fitted_participant_ids != plan.participant_ids:
        raise RuntimeError("final refit provenance does not contain the exact all-29 roster")
    _, expected_machine_id = normalize_model_id(plan.model_id)
    observed_model_id = str(getattr(result.model, "model_id", ""))
    if observed_model_id != expected_machine_id:
        raise RuntimeError(
            f"final refit model identity mismatch: {observed_model_id} != {expected_machine_id}"
        )
    if plan_is_ensemble:
        if result.provenance.member_training_seeds != plan.training_seeds:
            raise RuntimeError("final ensemble refit member seed roster drifted")
    elif result.provenance.training_seed != plan.training_seeds[0]:
        raise RuntimeError("final single-model refit seed drifted")
    if result.provenance.training_seed != int(trainer.config.seed):
        raise RuntimeError("final refit orchestration seed drifted from trainer config")
    if result.selected_epoch != plan.fixed_epochs:
        raise RuntimeError("final refit selected_epoch differs from the frozen plan")
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
        plan=plan,
        result=result,
        scope=scope,
        dataset_hash=dataset_hash,
        binding=binding,
        execution_hash=execution_hash,
    )


def execute_full_cohort_refit(*args: Any, **kwargs: Any) -> FinalRefitExecution:
    """Reject the obsolete caller-prepared dataset/model refit boundary."""

    del args, kwargs
    raise RuntimeError(
        "caller_prepared_full_cohort_refit_disabled; use "
        "experiment.execute_final_refit_from_verified_artifacts"
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
    if normalized["pipeline_generation"] != "final_pipeline_v2":
        raise ValueError("bundle pipeline_generation must be exactly final_pipeline_v2")
    for name in ("config_hash", "balance_hash", "run_hash", "source_snapshot_hash"):
        value = str(normalized[name])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"bundle {name} must be a lowercase SHA-256 digest")
    trust = normalized["serialization_trust"]
    if trust.get("trusted_local_only") is not True or trust.get("authenticated_signature") is not False:
        raise ValueError(
            "joblib bundle trust must explicitly state trusted_local_only=true and "
            "authenticated_signature=false"
        )
    return normalized


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical JSON atomically / 原子写入规范 JSON。"""

    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                _jsonable(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
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


def _validate_trusted_final_refit_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the typed all-29 identity that generic bundles cannot author."""

    metadata = manifest.get("metadata")
    identity = (
        metadata.get("final_refit_identity")
        if isinstance(metadata, Mapping) else None
    )
    if not isinstance(identity, Mapping) or set(identity) != _TRUSTED_FINAL_IDENTITY_FIELDS:
        raise ValueError("trusted final-refit identity field schema drift")
    normalized = _jsonable(dict(identity))
    participants = tuple(str(value) for value in normalized["participant_ids"])
    raw_seeds = normalized["training_seeds"]
    if (
        normalized["performance_evidence"]
        != "outer_oof_only_no_refit_self_evaluation"
        or len(participants) != 29
        or len(set(participants)) != 29
        or normalized["participant_count"] != 29
        or not str(normalized["purpose"]).strip()
        or normalized["representation_mode"]
        not in {"raw", "feature_vector", "feature_matrix", "fusion"}
        or normalized["model_kind"]
        not in {"single_model", "ensemble", "five_member_ensemble"}
        or normalized["model_family"]
        not in {"deep", "classical_or_rocket"}
    ):
        raise ValueError("trusted final-refit scope/model identity invalid")
    try:
        seeds = _validated_refit_seed_roster(raw_seeds)
    except ValueError as exc:
        raise ValueError("trusted final-refit seed roster invalid") from exc
    if (
        normalized["model_kind"] == "single_model" and len(seeds) != 1
    ):
        raise ValueError("trusted final-refit seed roster invalid")
    _, machine_model_id = normalize_model_id(str(normalized["model_id"]))
    kind_is_ensemble = _is_ensemble_model_kind(normalized["model_kind"])
    if kind_is_ensemble != _model_is_ensemble(machine_model_id):
        raise ValueError("trusted final-refit model kind/identity mismatch")
    family_is_classical = normalized["model_family"] == "classical_or_rocket"
    if family_is_classical != _model_uses_estimator(machine_model_id):
        raise ValueError("trusted final-refit model family/identity mismatch")
    if normalized["model_family"] == "deep":
        fixed_epochs = normalized["fixed_epochs"]
        if (
            normalized["epoch_rule"] != "fixed_epoch"
            or isinstance(fixed_epochs, (bool, np.bool_))
            or not isinstance(fixed_epochs, (int, np.integer))
            or int(fixed_epochs) <= 0
        ):
            raise ValueError("trusted deep final-refit epoch identity invalid")
    elif (
        normalized["epoch_rule"] != "not_applicable"
        or normalized["fixed_epochs"] is not None
        or kind_is_ensemble
    ):
        raise ValueError("trusted classical final-refit epoch/model identity invalid")
    for name in (
        "manual_selection_hash", "selection_record_file_sha256",
        "oof_evidence_hash", "config_hash", "registry_hash", "dataset_hash",
        "source_snapshot_hash",
        "resolved_model_config_hash", "architecture_parameters_hash",
        "input_schema_hash", "training_config_hash",
        "frozen_run_provenance_hash", "scope_membership_hash",
        "execution_hash", "bundle_materialization_hash",
        "source_records_hash", "golden_inputs_hash",
    ):
        value = str(normalized[name])
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"trusted final-refit digest invalid: {name}")
    if (
        manifest.get("config_hash") != normalized["config_hash"]
        or manifest.get("run_hash") != normalized["execution_hash"]
        or manifest.get("source_snapshot_hash")
        != normalized["source_snapshot_hash"]
        or manifest.get("machine_model_id")
        != machine_model_id
    ):
        raise ValueError("trusted final-refit manifest identity mismatch")
    return normalized


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
    if bundle_kind not in {
        GENERIC_BUNDLE_KIND,
        TRUSTED_FINAL_REFIT_BUNDLE_KIND,
    }:
        raise ValueError("unknown bundle kind")
    if (
        bundle_kind == GENERIC_BUNDLE_KIND
        and "final_refit_identity" in normalized_metadata
    ):
        raise ValueError(
            "generic bundle cannot claim trusted final-refit identity"
        )
    if (
        bundle_kind == TRUSTED_FINAL_REFIT_BUNDLE_KIND
        and "final_refit_identity" not in normalized_metadata
    ):
        raise ValueError("trusted final-refit bundle identity missing")
    if not golden_inputs or "x" not in golden_inputs and "window_bag" not in golden_inputs:
        raise ValueError("golden_inputs must contain x or window_bag")
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
        input_hash = input_spec_sha256(spec)
        adapter_contract = _pipeline_adapter_contract(
            pipeline_adapter,
            input_spec=spec,
            input_hash=input_hash,
        )
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
            "golden_case_hash": file_hashes[golden_path.name],
            "pipeline_adapter_boundary": (
                "serialized_raw_record_to_model_input_mapping"
                if pipeline_adapter is not None
                else "not_bundled"
            ),
            "pipeline_adapter_contract": adapter_contract,
            "transactional_save": "same_filesystem_staging_then_atomic_rename",
            "joblib_trust_boundary": (
                "hash_integrity_is_not_authentication; load_only_user-verified_local_source"
            ),
        }
        _atomic_json(staging / "manifest.json", manifest)
        if bundle_kind == TRUSTED_FINAL_REFIT_BUNDLE_KIND:
            _validate_trusted_final_refit_manifest(manifest)
        loaded = load_bundle(staging)
        assert_golden_parity(loaded, atol=FINAL_BUNDLE_PARITY_ATOL)
        if target.exists():
            raise FileExistsError(f"bundle target appeared during staging: {target}")
        # English: Same-filesystem directory rename is the single commit point.
        # 中文：同一文件系统内的目录重命名是唯一提交点。
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
    bundle_kind = manifest.get("bundle_kind")
    if bundle_kind not in {
        GENERIC_BUNDLE_KIND,
        TRUSTED_FINAL_REFIT_BUNDLE_KIND,
    }:
        raise ValueError("bundle kind is missing or unsupported")
    if bundle_kind == GENERIC_BUNDLE_KIND:
        metadata_payload = manifest.get("metadata")
        if (
            isinstance(metadata_payload, Mapping)
            and "final_refit_identity" in metadata_payload
        ):
            raise ValueError(
                "generic bundle cannot contain final-refit identity"
            )
    else:
        _validate_trusted_final_refit_manifest(manifest)
    if manifest.get("golden_parity_atol") != FINAL_BUNDLE_PARITY_ATOL:
        raise ValueError("bundle golden parity tolerance differs from frozen V2 policy")
    metadata = validate_bundle_metadata(manifest.get("metadata", {}))
    for name in (
        "pipeline_generation",
        "config_hash",
        "balance_hash",
        "run_hash",
        "source_snapshot_hash",
    ):
        if manifest.get(name) != metadata[name]:
            raise ValueError(f"bundle top-level {name} disagrees with validated metadata")
    if set(manifest.get("required_metadata_fields", ())) != REQUIRED_METADATA:
        raise ValueError("bundle metadata schema declaration is stale or incomplete")
    spec = ModelInputSpec.from_value(manifest.get("input_spec", {}))
    input_hash = input_spec_sha256(spec)
    if manifest.get("input_spec_hash") != input_hash:
        raise ValueError("bundle input_spec_hash disagrees with canonical input_spec")
    adapter_contract = manifest.get("pipeline_adapter_contract")
    if not isinstance(adapter_contract, dict):
        raise ValueError("bundle pipeline_adapter_contract must be explicit")
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
    actual_adapter_contract = _pipeline_adapter_contract(
        pipeline_adapter,
        input_spec=spec,
        input_hash=input_hash,
    )
    if actual_adapter_contract != adapter_contract:
        raise ValueError("serialized pipeline adapter disagrees with bundle contract")
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


def _pipeline_adapter_contract(
    adapter: Any,
    *,
    input_spec: ModelInputSpec | Mapping[str, Any],
    input_hash: str,
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

    allowed_roles = tuple(canonical_role_family(value) for value in raw_roles)
    if not allowed_roles or len(allowed_roles) != len(set(allowed_roles)):
        raise ValueError(
            "bundle adapter allowed_role_families must be non-empty and unique"
        )
    if not hasattr(adapter, "transform_record") and not callable(adapter):
        raise TypeError("pipeline adapter must be callable or expose transform_record")
    return {
        "status": "bundled",
        "representation_mode": representation_mode,
        "input_schema_hash": adapter_hash,
        "allowed_role_families": list(allowed_roles),
        "boundary": str(getattr(adapter, "boundary", "")),
    }


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
    spec = ModelInputSpec.from_value(loaded.manifest.get("input_spec", {}))
    input_hash = input_spec_sha256(spec)
    contract = loaded.manifest.get("pipeline_adapter_contract")
    try:
        actual_contract = _pipeline_adapter_contract(
            adapter,
            input_spec=spec,
            input_hash=input_hash,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "bundle pipeline adapter is not bound to its input schema"
        ) from exc
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

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from ppg_frailty.bundle import build_model_input_adapter
from ppg_frailty.config import load_config
from ppg_frailty.models import ModelInputSpec, create_model, normalize_model_config
from ppg_frailty.provenance import stable_payload_sha256
from ppg_frailty.training.bundle import (
    FrozenRepresentationTransformArchive,
    current_runtime_environment,
    input_spec_sha256,
    save_bundle,
)
from ppg_frailty.v5.environment import DEFAULT_LOCK, load_environment_lock
from ppg_frailty.v5.model_config_export import export_model_config
from ppg_frailty.v5.request_runner import (
    REQUEST_BINDING_ENV,
    RequestRecordingStudyRunner,
    read_anchored_request,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = "pipeline_output/example_run"


def _exact_environment_check() -> dict[str, object]:
    lock = load_environment_lock()
    return {
        "schema_version": "ppg_frailty.environment_check.v1",
        "status": "passed",
        "lock_id": lock["lock_id"],
        "observed": {
            "python": str(lock["runtime"]["python"]),
            "packages": {
                name: str(version)
                for name, version in lock["runtime"]["packages"].items()
            },
            "accelerator": {
                "cuda_available": True,
                "device_count": 1,
                "selected_device_index": 0,
                "selected_device_available": True,
                **{
                    name: lock["accelerator"][name]
                    for name in (
                        "gpu_name",
                        "compute_capability",
                        "driver_version",
                        "torch_cuda",
                        "cudnn",
                    )
                },
            },
            "determinism": dict(lock["determinism"]),
        },
        "mismatches": [],
    }


def _publish_request(
    study: Path, relative: str, payload: dict[str, object]
) -> tuple[str, str]:
    RequestRecordingStudyRunner(
        pipeline_root=study,
        pre_run_artifacts={relative: payload},
    )._publish_pre_run_artifacts(study)
    _, digest, _ = read_anchored_request(study, relative)
    return relative, digest


def _write_study(
    root: Path, *, resolved_config: dict[str, object] | None = None
) -> tuple[Path, dict[str, object]]:
    study = root / STUDY_PATH
    case = study / "cases/reference"
    case.mkdir(parents=True)
    default_config: dict[str, object] = {
        "schema_version": "example",
        "config_id": "example_config",
        "representation_mode": "raw",
        "model": {"model_id": "InceptionTimeSmall", "dropout": 0.2},
        "training": {
            "optimizer": "adamw",
            "class_weighting": "inverse_frequency",
            "batch_size": 16,
        },
        "signal": {
            "imu": {"gravity_method": "sensor_filter_only_no_gravity_removal"},
            "ppg_filter": {"family": "butterworth_sos"},
            "dl_resampling": {"enabled": False},
        },
        "aggregation": {"balance_line": "line_b_equal_role_families"},
        "artifact": {
            "motion_detector_enabled": False,
            "denoiser_enabled": False,
            "reducer": "identity",
        },
    }
    config = default_config if resolved_config is None else resolved_config
    (case / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (study / "study_manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "resumed_case_count": 0,
                "cases": [
                    {
                        "case_id": "reference",
                        "case_directory": "cases/reference",
                        "resolved_config_path": "cases/reference/resolved_config.yaml",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (study / "v5_data_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ppg_frailty.v5_data_products.v1",
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )
    tables = study / "tables"
    tables.mkdir()
    with (tables / "v5_fold_models.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["case_id", "repeat", "fold", "model_id", "state_hash"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "reference",
                "repeat": "0",
                "fold": "0",
                "model_id": "InceptionTimeSmall",
                "state_hash": "a" * 64,
            }
        )
    _publish_request(
        study,
        "v5_run_request.json",
        {
            "schema_version": "ppg_frailty.v5_run_request.v1",
            "command": "run",
            "environment_policy": "exact",
            "environment_lock_sha256": sha256_file(DEFAULT_LOCK),
            "environment_check": _exact_environment_check(),
            "resumed": False,
            "refit_requested": False,
            "execution_binding": {"binding_sha256": "8" * 64},
            "configuration_resolution": {
                "module_selections": [
                    {
                        "family": "imu_gravity",
                        "module_id": "sensor_filter_only_no_gravity_removal",
                    }
                ]
            },
        },
    )
    return study, config


def _write_real_raw_bundle(
    study: Path,
    config: dict[str, object],
    *,
    training_request: str = "v5_run_request.json",
) -> Path:
    config_path = study / "cases/reference/resolved_config.yaml"
    config_hash = load_config(config_path).sha256
    model_section = dict(config["model"])  # type: ignore[arg-type]
    channel_schema = tuple(model_section["input_channel_order"])
    spec = ModelInputSpec(
        "raw",
        n_channels=len(channel_schema),
        n_classes=3,
        channel_schema=channel_schema,
    )
    resolved_model = {
        "model_id": "inception_small",
        "seed": 42,
        "seed_policy": model_section["seed_policy"],
        "dropout": model_section["dropout"],
        "kernel_sizes": model_section["kernel_sizes"],
        "dilation": model_section["dilation"],
        "architecture_parameters": model_section["architecture_parameters"],
    }
    normalized_model = normalize_model_config(resolved_model)
    model = create_model(resolved_model, spec).eval()
    metadata = {
        "model_identity": {
            "name": normalized_model["canonical_model_name"],
            "machine_id": normalized_model["model_id"],
            "version": "test",
        },
        "representation_mode": "raw",
        "signal_route": {"status": "frozen_v2_raw"},
        "class_order": [0, 1, 2],
        "channel_schema": list(channel_schema),
        "preprocessing": {"boundary": "frozen_v2_raw"},
        "preprocessing_hash": "preprocessing",
        "resampling": {"status": "configured"},
        "window_plan": {"status": "configured"},
        "feature_registry": {"status": "not_model_input"},
        "feature_hash": "not_applicable",
        "feature_vector_schema": {"status": "not_model_input"},
        "ordered_matrix_schema": {"status": "not_model_input"},
        "mask_semantics": {"sample_mask": "true_is_valid"},
        "validity_policy": {"status": "frozen"},
        "fitted_objects": [{"name": "trained_classifier"}],
        "representation_state": {"status": "model_ready"},
        "pooling_rule": {"status": "model_defined"},
        "aggregation_rule": "line_b_equal_role_families",
        "manifest_hash": "manifest",
        "fold_hash": "fold",
        "manifest_version": "internal_records_v2",
        "fold_registry_version": "frailty3_future_corrected_sgkf5_v2",
        "pipeline_generation": "final_pipeline_v2",
        "config_hash": config_hash,
        "balance_hash": "b" * 64,
        "run_hash": "c" * 64,
        "source_snapshot_hash": "d" * 64,
        "code_version": "test",
        "environment": current_runtime_environment(),
        "dependency_status": {"status": "test"},
        "serialization_trust": {
            "trusted_local_only": True,
            "authenticated_signature": False,
        },
        "golden_case": {"status": "test"},
    }
    transform = FrozenRepresentationTransformArchive(
        representation_mode="raw",
        input_schema_hash=input_spec_sha256(spec),
        fitted_on_participant_ids=tuple(f"P{index:02d}" for index in range(29)),
        fitted_artifacts={},
        provenance={
            "raw_imu": {
                "schema_version": "not_applicable_all8_window_normalized_v1",
                "artifact_sha256": None,
                "fitted_on_participant_ids": [],
                "strategy": "none_after_all8_per_window_robust",
                "parameters": None,
            }
        },
        source_records_hash="e" * 64,
        dataset_hash="f" * 64,
    )
    adapter = build_model_input_adapter(
        "raw",
        input_schema_hash=input_spec_sha256(spec),
        allowed_role_families=tuple(
            config["training"]["classifier_role_families"]  # type: ignore[index]
        ),
    )
    golden = {
        "x": np.random.default_rng(42).normal(size=(1, 8, 64)).astype(np.float32),
        "mask": np.ones((1, 64), dtype=bool),
    }
    bundle = study / "cases/reference/checkpoint"
    request_payload, request_hash, _ = read_anchored_request(
        study, training_request
    )
    binding = {
        "schema_version": "ppg_frailty.v5_checkpoint_request_binding.v1",
        "request_path": training_request,
        "request_sha256": request_hash,
        "environment_lock_sha256": request_payload["environment_lock_sha256"],
    }
    previous = os.environ.get(REQUEST_BINDING_ENV)
    os.environ[REQUEST_BINDING_ENV] = json.dumps(binding, sort_keys=True)
    try:
        save_bundle(
            model,
            bundle,
            model_config=resolved_model,
            input_spec=spec,
            metadata=metadata,
            golden_inputs=golden,
            transforms=transform,
            pipeline_adapter=adapter,
        )
    finally:
        if previous is None:
            os.environ.pop(REQUEST_BINDING_ENV, None)
        else:
            os.environ[REQUEST_BINDING_ENV] = previous
    manifest_hash = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    data_manifest = json.loads(
        (study / "v5_data_manifest.json").read_text(encoding="utf-8")
    )
    data_manifest["published_models"] = [
        {
            "case_id": "reference",
            "model_role": "research_outer_fold_median_for_dashboard_trial",
            "deployment_status": "research_only",
            "bundle_manifest": "cases/reference/checkpoint/manifest.json",
            "bundle_manifest_sha256": manifest_hash,
            "repeat": 0,
            "fold": 0,
        }
    ]
    (study / "v5_data_manifest.json").write_text(
        json.dumps(data_manifest), encoding="utf-8"
    )
    return bundle


def _promote_test_bundle_to_trusted_refit(bundle: Path) -> None:
    """Materialize a structurally valid refit bundle for export-boundary tests."""

    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = "9" * 64
    manifest["bundle_kind"] = "trusted_final_refit_v2"
    manifest["metadata"]["final_refit_identity"] = {
        "purpose": "test_all29_refit_export_boundary",
        "performance_evidence": "outer_oof_only_no_refit_self_evaluation",
        "manual_selection_hash": digest,
        "selection_record_file_sha256": digest,
        "oof_evidence_hash": digest,
        "config_hash": manifest["config_hash"],
        "registry_hash": digest,
        "dataset_hash": digest,
        "source_snapshot_hash": manifest["source_snapshot_hash"],
        "resolved_model_config_hash": digest,
        "architecture_parameters_hash": digest,
        "input_schema_hash": manifest["input_spec_hash"],
        "training_config_hash": digest,
        "frozen_run_provenance_hash": digest,
        "scope_membership_hash": digest,
        "execution_hash": manifest["run_hash"],
        "bundle_materialization_hash": digest,
        "source_records_hash": digest,
        "golden_inputs_hash": digest,
        "participant_count": 29,
        "participant_ids": [f"P{index:02d}" for index in range(29)],
        "training_seeds": [42],
        "epoch_rule": "fixed_epoch",
        "fixed_epochs": 1,
        "model_id": manifest["machine_model_id"],
        "model_kind": "single_model",
        "model_family": "deep",
        "representation_mode": "raw",
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_resume_refit_request(study: Path) -> tuple[str, str]:
    original = json.loads((study / "v5_run_request.json").read_text(encoding="utf-8"))
    original.update(
        {
            "schema_version": "ppg_frailty.v5_resume_request.v1",
            "environment_policy": "exact",
            "environment_check": _exact_environment_check(),
            "resumed": True,
            "refit_requested": True,
        }
    )
    immutable = study / "request_history/refit-reference.json"
    relative, digest = _publish_request(
        study, immutable.relative_to(study).as_posix(), original
    )
    # A mutable latest pointer may subsequently change; the refit export must
    # never consult it after the refit manifest binds the immutable artifact.
    latest = dict(original)
    latest["refit_requested"] = False
    (study / "v5_resume_request.json").write_text(
        json.dumps(latest), encoding="utf-8"
    )
    return relative, digest


def _write_resume_training_request(study: Path) -> tuple[str, str]:
    original = json.loads((study / "v5_run_request.json").read_text(encoding="utf-8"))
    original.update(
        {
            "schema_version": "ppg_frailty.v5_resume_request.v1",
            "resumed": True,
            "refit_requested": False,
        }
    )
    return _publish_request(
        study, "request_history/resume-folds.json", original
    )


def test_export_model_config_is_read_only_and_preserves_complete_defaults(
    tmp_path: Path,
) -> None:
    study, config = _write_study(tmp_path)
    source_before = {
        path.relative_to(study): path.read_bytes()
        for path in study.rglob("*")
        if path.is_file()
    }

    result = export_model_config(STUDY_PATH, pipeline_root=tmp_path)

    exported = tmp_path / "model_config/example_run"
    assert result["output_directory"] == "model_config/example_run"
    assert result["capabilities"] == {
        "configuration_reuse_only": True,
        "learned_weights_available": False,
        "new_participant_inference_available": False,
        "new_participant_inference_status": "unavailable_without_compatible_verified_bundle",
    }
    assert (exported / "available_modules.json").is_file()
    assert (exported / "README.md").is_file()
    assert yaml.safe_load(
        (exported / "cases/reference/resolved_pipeline_config.yaml").read_text(
            encoding="utf-8"
        )
    ) == config
    module_defaults = yaml.safe_load(
        (exported / "cases/reference/pipeline_module_defaults.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert module_defaults["parameter_defaults"] == config
    assert module_defaults["requested_module_selections"] == [
        {
            "family": "imu_gravity",
            "module_id": "sensor_filter_only_no_gravity_removal",
        }
    ]
    assert {
        "family": "model",
        "module_id": "InceptionTimeSmall",
        "source_path": "model.model_id",
        "derivation": "exact_resolved_config_field",
    } in module_defaults["derived_module_defaults"]
    model_parameters = yaml.safe_load(
        (exported / "cases/reference/model_reuse_parameters.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert model_parameters["deployment_capability"]["learned_weights_available"] is False
    assert model_parameters["fold_provenance"]["fold_model_row_count"] == 1
    with (exported / "cases/reference/fold_model_parameters.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        assert list(csv.DictReader(stream))[0]["state_hash"] == "a" * 64
    source_after = {
        path.relative_to(study): path.read_bytes()
        for path in study.rglob("*")
        if path.is_file()
    }
    assert source_after == source_before


def test_export_model_config_refuses_to_overwrite_existing_export(tmp_path: Path) -> None:
    _write_study(tmp_path)
    export_model_config(STUDY_PATH, pipeline_root=tmp_path)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_model_config(STUDY_PATH, pipeline_root=tmp_path)


def test_export_model_config_rejects_non_pipeline_and_symlink_escape(
    tmp_path: Path,
) -> None:
    (tmp_path / "pipeline_output").mkdir()
    outside = tmp_path / "artifacts/example_run"
    outside.mkdir(parents=True)

    with pytest.raises(ValueError, match="must remain inside"):
        export_model_config(outside, pipeline_root=tmp_path)
    with pytest.raises((IsADirectoryError, FileNotFoundError)):
        export_model_config(tmp_path / "pipeline_output", pipeline_root=tmp_path)

    (tmp_path / "pipeline_output/escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="must remain inside"):
        export_model_config(
            tmp_path / "pipeline_output/escape", pipeline_root=tmp_path
        )


def test_export_model_config_replace_refreshes_same_source_atomically(
    tmp_path: Path,
) -> None:
    study, _ = _write_study(tmp_path)
    export_model_config(STUDY_PATH, pipeline_root=tmp_path)
    with (study / "tables/v5_fold_models.csv").open(
        "a", encoding="utf-8", newline=""
    ) as stream:
        stream.write("reference,0,1,InceptionTimeSmall," + "b" * 64 + "\n")

    result = export_model_config(
        STUDY_PATH,
        pipeline_root=tmp_path,
        replace_existing=True,
    )

    assert result["output_directory"] == "model_config/example_run"
    exported = tmp_path / "model_config/example_run"
    with (exported / "cases/reference/fold_model_parameters.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        assert len(list(csv.DictReader(stream))) == 2
    assert not list((tmp_path / "model_config").glob(".example_run.*"))


def test_export_rejects_an_incomplete_hash_only_bundle(tmp_path: Path) -> None:
    study, _ = _write_study(tmp_path)
    bundle = study / "cases/reference/checkpoint"
    bundle.mkdir(parents=True)
    (bundle / "state.pt").write_bytes(b"learned weights")
    (bundle / "golden.npz").write_bytes(b"golden case")
    hashes = {
        name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        for name in ("state.pt", "golden.npz")
    }
    bundle_manifest = {
        "bundle_kind": "generic_research",
        "pipeline_adapter_boundary": "not_bundled",
        "file_hashes": hashes,
    }
    (bundle / "manifest.json").write_text(
        json.dumps(bundle_manifest), encoding="utf-8"
    )
    manifest_hash = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    data_manifest = json.loads(
        (study / "v5_data_manifest.json").read_text(encoding="utf-8")
    )
    data_manifest["published_models"] = [
        {
            "case_id": "reference",
            "model_role": "research_outer_fold_median_for_dashboard_trial",
            "deployment_status": "research_only",
            "bundle_manifest": "cases/reference/checkpoint/manifest.json",
            "bundle_manifest_sha256": manifest_hash,
            "selection_manifest": "models/reference/selection.json",
            "repeat": 2,
            "fold": 3,
            "balanced_accuracy": 0.6,
        }
    ]
    (study / "v5_data_manifest.json").write_text(
        json.dumps(data_manifest), encoding="utf-8"
    )

    result = export_model_config(STUDY_PATH, pipeline_root=tmp_path)

    exported = tmp_path / "model_config/example_run/cases/reference/learned_model"
    assert result["capabilities"]["learned_weights_available"] is False
    assert result["capabilities"]["configuration_reuse_only"] is True
    assert not exported.exists()


def test_export_does_not_mark_hash_only_fake_bundle_inferable(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/presets/finalcase.yaml").read_text(encoding="utf-8")
    )
    study, _ = _write_study(tmp_path, resolved_config=config)
    bundle = study / "cases/reference/checkpoint"
    bundle.mkdir(parents=True)
    (bundle / "state.pt").write_bytes(b"learned weights")
    hashes = {"state.pt": hashlib.sha256(b"learned weights").hexdigest()}
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "bundle_kind": "generic_research",
                "pipeline_adapter_boundary": "not_bundled",
                "file_hashes": hashes,
            }
        ),
        encoding="utf-8",
    )
    manifest_hash = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    data_manifest = json.loads(
        (study / "v5_data_manifest.json").read_text(encoding="utf-8")
    )
    data_manifest["published_models"] = [
        {
            "case_id": "reference",
            "bundle_manifest": "cases/reference/checkpoint/manifest.json",
            "bundle_manifest_sha256": manifest_hash,
        }
    ]
    (study / "v5_data_manifest.json").write_text(
        json.dumps(data_manifest), encoding="utf-8"
    )

    result = export_model_config(STUDY_PATH, pipeline_root=tmp_path)

    assert result["capabilities"]["new_participant_inference_available"] is False
    assert (
        result["capabilities"]["new_participant_inference_status"]
        == "unavailable_without_compatible_verified_bundle"
    )
    assert result["cases"][0]["bundle_path"] is None
    assert result["cases"][0]["new_participant_inference"] is False
    assert result["cases"][0]["inference_validation"]["status"] == "unavailable"
    assert "bundle" in result["cases"][0]["inference_validation"]["reason"]


def test_export_verifies_real_bundle_once_and_marks_raw_inference_ready(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/presets/finalcase.yaml").read_text(encoding="utf-8")
    )
    study, config = _write_study(tmp_path, resolved_config=config)
    _write_real_raw_bundle(study, config)

    result = export_model_config(STUDY_PATH, pipeline_root=tmp_path)

    assert result["capabilities"]["new_participant_inference_available"] is True
    case = result["cases"][0]
    assert case["new_participant_inference"] is True
    assert case["inference_validation"]["status"] == "ready"
    assert case["bundle_path"] == "cases/reference/learned_model"

    config_path = study / "cases/reference/resolved_config.yaml"
    changed_config = dict(config)
    changed_config["config_id"] = "tampered_after_training"
    config_path.write_text(
        yaml.safe_dump(changed_config, sort_keys=False), encoding="utf-8"
    )
    mismatched = export_model_config(
        STUDY_PATH,
        pipeline_root=tmp_path,
        replace_existing=True,
    )
    assert mismatched["cases"][0]["new_participant_inference"] is False
    assert "config hashes differ" in mismatched["cases"][0]["inference_validation"][
        "reason"
    ]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    # Training-request metadata is descriptive and is not revalidated here.
    (study / "v5_run_request.json").write_text("{}\n", encoding="utf-8")
    replay = export_model_config(STUDY_PATH, pipeline_root=tmp_path, replace_existing=True)
    assert replay["cases"][0]["new_participant_inference"] is True


def test_refit_bundle_is_preferred_without_revalidating_request_history(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/presets/finalcase.yaml").read_text(encoding="utf-8")
    )
    study, config = _write_study(tmp_path, resolved_config=config)
    training_request, training_request_hash = _write_resume_refit_request(study)
    bundle = _write_real_raw_bundle(
        study, config, training_request=training_request
    )
    _promote_test_bundle_to_trusted_refit(bundle)
    bundle_hash = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    bundle_payload = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    refit_intent = "cases/reference/refit_intent.json"
    (study / refit_intent).write_text(
        json.dumps(
            {
                "schema_version": "ppg_frailty.v5_refit_intent.v1",
                "status": "approved_before_model_fit",
                "training_request": training_request,
                "training_request_sha256": training_request_hash,
            }
        ),
        encoding="utf-8",
    )
    refit_intent_hash = hashlib.sha256((study / refit_intent).read_bytes()).hexdigest()
    (study / "v5_refit_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ppg_frailty.v5_refit_manifest.v1",
                "default_refit": False,
                "cases": [
                    {
                        "case_id": "reference",
                        "status": "trusted_all29_refit_published",
                        "performance_evidence": (
                            "outer_oof_only_no_refit_self_evaluation"
                        ),
                        "bundle_manifest": "cases/reference/checkpoint/manifest.json",
                        "bundle_manifest_sha256": bundle_hash,
                        "selection_record": "selection.json",
                        "selection_record_sha256": "7" * 64,
                        "confirmed_config_sha256": bundle_payload["config_hash"],
                        "training_request": training_request,
                        "training_request_sha256": training_request_hash,
                        "environment_lock_sha256": sha256_file(DEFAULT_LOCK),
                        "environment_check_sha256": (
                            stable_payload_sha256(_exact_environment_check())
                        ),
                        "driver_version": load_environment_lock()["accelerator"][
                            "driver_version"
                        ],
                        "dataset_hash": bundle_payload["metadata"][
                            "final_refit_identity"
                        ]["dataset_hash"],
                        "source_records_hash": bundle_payload["metadata"][
                            "final_refit_identity"
                        ]["source_records_hash"],
                        "golden_inputs_hash": bundle_payload["metadata"][
                            "final_refit_identity"
                        ]["golden_inputs_hash"],
                        "preprocessing_hash": bundle_payload["metadata"][
                            "preprocessing_hash"
                        ],
                        "feature_hash": bundle_payload["metadata"]["feature_hash"],
                        "refit_intent": refit_intent,
                        "refit_intent_sha256": refit_intent_hash,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    study_manifest_path = study / "study_manifest.json"
    study_manifest = json.loads(study_manifest_path.read_text(encoding="utf-8"))
    study_manifest["resumed_case_count"] = 1
    study_manifest_path.write_text(json.dumps(study_manifest), encoding="utf-8")

    result = export_model_config(STUDY_PATH, pipeline_root=tmp_path)

    case = result["cases"][0]
    assert case["new_participant_inference"] is True
    assert case["model_role"] == "all29_full_cohort_refit"
    assert case["inference_validation"]["status"] == "ready"
    (study / training_request).write_text("{}\n", encoding="utf-8")
    replay = export_model_config(STUDY_PATH, pipeline_root=tmp_path, replace_existing=True)
    assert replay["cases"][0]["model_role"] == "all29_full_cohort_refit"


def test_median_fold_bundle_does_not_depend_on_request_history(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/presets/finalcase.yaml").read_text(encoding="utf-8")
    )
    study, config = _write_study(tmp_path, resolved_config=config)
    training_request, training_request_hash = _write_resume_training_request(study)
    _write_real_raw_bundle(
        study, config, training_request=training_request
    )

    result = export_model_config(STUDY_PATH, pipeline_root=tmp_path)

    assert result["cases"][0]["new_participant_inference"] is True
    assert result["cases"][0]["model_role"] == "research_outer_fold_median_for_dashboard_trial"


def test_incomplete_refit_record_is_not_exported_as_a_model(
    tmp_path: Path,
) -> None:
    study, _ = _write_study(tmp_path)
    (study / "v5_refit_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ppg_frailty.v5_refit_manifest.v1",
                "default_refit": False,
                "cases": [
                    {
                        "case_id": "reference",
                        "status": "trusted_all29_refit_published",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = export_model_config(STUDY_PATH, pipeline_root=tmp_path)
    assert result["cases"][0]["bundle_path"] is None
    assert result["cases"][0]["model_role"] is None

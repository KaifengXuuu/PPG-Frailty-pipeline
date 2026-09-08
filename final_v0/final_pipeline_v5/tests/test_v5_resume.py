from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ppg_frailty import experiment
from ppg_frailty.study import StudyRunner, load_study_plan
import ppg_frailty.study.recovery as recovery
from ppg_frailty.training import bundle as training_bundle


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _attach_checkpoint(cell: Path, summary: dict) -> None:
    checkpoint = cell / "model_checkpoint"
    checkpoint.mkdir()
    state = checkpoint / "model.joblib"
    golden = checkpoint / "golden.npz"
    state.write_bytes(b"minimal-model-state")
    golden.write_bytes(b"minimal-golden-case")
    input_spec = {
        "representation_mode": "raw",
        "n_channels": 8,
        "n_classes": 3,
        "n_file_features": 0,
        "feature_names": [],
        "channel_schema": [
            "RED",
            "IR",
            "A_dyn_x",
            "A_dyn_y",
            "A_dyn_z",
            "GX",
            "GY",
            "GZ",
        ],
    }
    run_hash = _digest("minimal-run")
    metadata = {
        name: {"status": "not_applicable"}
        for name in training_bundle.REQUIRED_METADATA
    }
    metadata.update(
        {
            "model_identity": {
                "name": "MinimalModel",
                "machine_id": summary["model_machine_id"],
                "version": "registered_v2",
            },
            "representation_mode": summary["representation_mode"],
            "class_order": summary["class_order"],
            "channel_schema": input_spec["channel_schema"],
            "fitted_objects": [
                {
                    "name": "trained_classifier",
                    "kind": "outer_train_only_fitted_model",
                    "provenance": summary["fitted_provenance"],
                }
            ],
            "preprocessing_hash": summary["preprocessing_hash"],
            "feature_hash": summary["feature_hash"],
            "fold_hash": summary["fitted_provenance"]["fold_hash"],
            "pipeline_generation": "final_pipeline_v2",
            "config_hash": summary["config_hash"],
            "balance_hash": _digest("minimal-balance"),
            "run_hash": run_hash,
            "source_snapshot_hash": summary["source_version"],
            "code_version": summary["code_commit"],
            "serialization_trust": {
                "trusted_local_only": True,
                "authenticated_signature": False,
                "integrity": "hashes_plus_reload_golden_prediction_parity",
            },
            "golden_case": {
                "boundary": "already_preprocessed_fold_model_input",
                "source_identity": {
                    "repeat": summary["repeat_index"],
                    "fold": summary["fold_index"],
                    "scope": "first_outer_train_model_input",
                },
                "expected_output": "three_class_probability",
            },
        }
    )
    file_hashes = {
        state.name: training_bundle.sha256_file(state),
        golden.name: training_bundle.sha256_file(golden),
    }
    manifest = {
        "bundle_format": training_bundle.BUNDLE_FORMAT_VERSION,
        "bundle_kind": training_bundle.GENERIC_BUNDLE_KIND,
        "kind": "estimator",
        "state_file": state.name,
        "model_config": {
            "model_id": summary["model_machine_id"],
            "canonical_model_name": "MinimalModel",
        },
        "canonical_model_name": "MinimalModel",
        "machine_model_id": summary["model_machine_id"],
        "pipeline_generation": "final_pipeline_v2",
        "config_hash": summary["config_hash"],
        "balance_hash": metadata["balance_hash"],
        "run_hash": run_hash,
        "source_snapshot_hash": summary["source_version"],
        "input_spec": input_spec,
        "input_spec_hash": training_bundle.stable_payload_sha256(input_spec),
        "metadata": metadata,
        "required_metadata_fields": sorted(training_bundle.REQUIRED_METADATA),
        "file_hashes": file_hashes,
        "golden_parity_atol": training_bundle.FINAL_BUNDLE_PARITY_ATOL,
        "golden_prediction_device": "cpu",
        "golden_device_policy": "same_device_before_and_after_serialization",
        "golden_case_hash": file_hashes[golden.name],
        "pipeline_adapter_boundary": "not_bundled",
        "pipeline_adapter_contract": {"status": "not_bundled"},
        "transactional_save": "same_filesystem_staging_then_atomic_rename",
        "joblib_trust_boundary": (
            "hash_integrity_is_not_authentication; "
            "load_only_user-verified_local_source"
        ),
    }
    manifest_path = checkpoint / "manifest.json"
    _write_json(manifest_path, manifest)
    summary["learned_model_checkpoint"] = {
        "schema_version": "ppg_frailty.v5_fold_checkpoint.v1",
        "purpose": "research_outer_fold_model_for_replay_and_dashboard_trial",
        "deployment_status": "not_final_refit_outer_train_subset_only",
        "selection_metric_use": (
            "outer_oof_balanced_accuracy_used_only_after_all_folds_finish"
        ),
        "model_input_boundary": "already_preprocessed_fold_model_input",
        "manifest_path": "model_checkpoint/manifest.json",
        "manifest_sha256": training_bundle.sha256_file(manifest_path),
        "state_file": f"model_checkpoint/{state.name}",
        "state_sha256": file_hashes[state.name],
        "golden_parity_atol": training_bundle.FINAL_BUNDLE_PARITY_ATOL,
    }


def _externalized_cell_fixture(tmp_path: Path) -> tuple[Path, str, str]:
    """Create the smallest current-V5 cell needed by recovery validation."""

    cell = tmp_path / "repeat_00" / "fold_00"
    cell.mkdir(parents=True)
    config_id = "minimal_v5_cell"
    config_hash = "a" * 64
    full_window_evidence = {
        "window_000": {
            "direct": {
                "state": "pass",
                "reasons": [],
                "q_rate": {
                    "state": "pass",
                    "score": 0.9,
                    "coverage": 1.0,
                    "threshold": 0.5,
                    "reasons": [],
                    "components": {"ibi_count": 9},
                },
            },
            "post_reduction": None,
        }
    }
    evidence_count, evidence_sha256 = experiment._write_route_window_sqi_evidence(
        cell,
        {
            "repeat_index": 0,
            "fold_index": 0,
            "config_hash": config_hash,
            "route_window_sqi_evidence": [
                {
                    "record_id": "record-1",
                    "participant_id": "participant-1",
                    "role": "B",
                    "windows": full_window_evidence,
                }
            ],
        },
    )
    route_row = {
        "record_id": "record-1",
        "participant_id": "participant-1",
        "role": "B",
        "route_artifact": {
            "cells": [{"config_sha256": config_hash}],
            "native_window_sqi_evidence": experiment._compact_window_sqi_evidence(
                full_window_evidence
            ),
        },
    }
    quality_row = {
        "record_id": "record-1",
        "participant_id": "participant-1",
        "role": "B",
        "diagnostic_reason": "minimal_fixture",
    }
    summary = {
        "status": "passed",
        "repeat_index": 0,
        "fold_index": 0,
        "config_id": config_id,
        "config_hash": config_hash,
        "canonical_config_hash": config_hash,
        "class_order": [0, 1, 2],
        "representation_mode": "raw",
        "preprocessing_hash": _digest("minimal-preprocessing"),
        "feature_hash": _digest("minimal-feature"),
        "model_hash": _digest("minimal-model-state"),
        "model_machine_id": "minimal_model",
        "source_version": _digest("minimal-source"),
        "code_commit": "not_git_bound",
        "fitted_provenance": {
            "fold_hash": _digest("minimal-fold"),
            "state_hash": _digest("minimal-model-state"),
            "training_seed": 42,
        },
        "frozen_model_run_provenance": {
            "fold_hash": _digest("minimal-fold"),
            "random_seeds": [42],
        },
        "route_artifacts_row_count": 1,
        "quality_diagnostic_row_count": 1,
        "route_window_sqi_evidence_artifact": (
            "route_window_sqi_evidence.jsonl.gz"
        ),
        "route_window_sqi_evidence_row_count": evidence_count,
        "route_window_sqi_evidence_compression": "gzip_mtime0_jsonl",
        "route_window_sqi_evidence_report_consumed": False,
        "route_window_sqi_evidence_sha256": evidence_sha256,
    }
    _attach_checkpoint(cell, summary)
    _write_json(
        cell / "route_artifacts.json",
        {
            "schema_version": "ppg_frailty.route_artifacts.v2",
            "repeat_index": 0,
            "fold_index": 0,
            "rows": [route_row],
        },
    )
    _write_json(
        cell / "quality_diagnostics.json",
        {
            "schema_version": "ppg_frailty.quality_diagnostics.v2",
            "rows": [quality_row],
        },
    )
    _write_json(
        cell / "metrics_per_fold_seed.json",
        {
            "schema_version": "ppg_frailty.metrics_per_fold_seed.v2",
            "pipeline_generation": "final_pipeline_v2",
            "cells": [summary],
        },
    )
    _write_json(
        cell / "run_manifest.json",
        {
            "schema_version": "ppg_frailty.run_manifest.v2",
            "pipeline_generation": "final_pipeline_v2",
            "status": "passed",
            "scientific_scope": "frozen_5x5_scientific_benchmark",
            "cell": summary,
            "mandatory_artifacts": [
                "run_manifest.json",
                "metrics_per_fold_seed.json",
                "route_artifacts.json",
                "quality_diagnostics.json",
                "route_window_sqi_evidence.jsonl.gz",
                "model_checkpoint/manifest.json",
            ],
        },
    )
    return cell, config_id, config_hash


def _resume_fixture(tmp_path: Path):
    plan = load_study_plan(ROOT / "configs/studies/finalcase.yaml")
    runner = StudyRunner(pipeline_root=ROOT, output_layout="v5")
    case = runner.expand(plan).cases[0]
    output = tmp_path / "run"
    state = output / ".runner_state" / case.case_id
    attempt = state / "attempts/attempt_001"
    attempt.mkdir(parents=True)
    config = output / "configs" / f"{case.case_id}.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("placeholder: true\n", encoding="utf-8")
    return runner, plan, case, output, state, attempt, config


def _complete_result(case_id: str, config_hash: str, output: Path) -> dict:
    return {
        "status": "passed",
        "scientific_scope": "frozen_5x5_scientific_benchmark",
        "config_id": case_id,
        "config_hash": config_hash,
        "repeat_indices": list(range(5)),
        "fold_indices": list(range(5)),
        "output_dir": str(output),
        "cell_results": [
            {
                "status": "passed",
                "repeat_index": repeat,
                "fold_index": fold,
            }
            for repeat in range(5)
            for fold in range(5)
        ],
        "metrics": {"passed_cell_count": 25},
        "failure_reasons": [],
    }


def test_v5_resume_indexes_an_already_published_complete_case(
    tmp_path: Path, monkeypatch,
) -> None:
    runner, plan, case, output, state, attempt, config = _resume_fixture(tmp_path)
    published = output / case.case_id
    published.mkdir()
    monkeypatch.setattr(
        recovery,
        "validate_published_complete_experiment",
        lambda *_args, **_kwargs: _complete_result(
            str(case.config["config_id"]), case.config_sha256, published
        ),
    )

    record = runner._recover_complete_interrupted_pass(
        case, config, state, plan
    )

    assert record is not None and record["status"] == "passed"
    assert json.loads((published / "case_result.json").read_text())["attempt"] == 1
    assert (attempt / "attempt_result.json").is_file()


def test_v5_resume_archives_partial_staging_without_reusing_fits(
    tmp_path: Path, monkeypatch,
) -> None:
    runner, plan, case, output, state, attempt, config = _resume_fixture(tmp_path)
    staging = output / f".{case.case_id}.staging.123"
    staging.mkdir()
    (staging / "partial.txt").write_text("preserved", encoding="utf-8")
    monkeypatch.setattr(
        recovery,
        "recover_completed_full_experiment_staging",
        lambda *_args, **_kwargs: None,
    )

    record = runner._recover_complete_interrupted_pass(
        case, config, state, plan
    )

    assert record is None
    archived = attempt / "interrupted_partial_staging"
    assert (archived / "partial.txt").read_text(encoding="utf-8") == "preserved"
    audit = json.loads((attempt / "attempt_result.json").read_text())
    assert audit["status"] == "interrupted_incomplete"
    assert audit["model_fits_reused"] is False


def test_v5_recovery_reuses_externalized_cell_and_nested_checkpoint(
    tmp_path: Path, monkeypatch,
) -> None:
    cell, config_id, config_hash = _externalized_cell_fixture(tmp_path)
    immutable_names = (
        "route_artifacts.json",
        "quality_diagnostics.json",
        "metrics_per_fold_seed.json",
        "run_manifest.json",
        "route_window_sqi_evidence.jsonl.gz",
        "model_checkpoint/manifest.json",
    )
    before = {name: (cell / name).read_bytes() for name in immutable_names}

    manifest = recovery._safe_mandatory_artifacts(cell)
    assert "model_checkpoint/manifest.json" in manifest["mandatory_artifacts"]
    monkeypatch.setattr(
        experiment,
        "_write_route_window_sqi_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "an externalized V5 evidence archive must never be regenerated"
        ),
    )

    summary, quality, mode = recovery._externalize_legacy_sqi_payload(
        cell,
        config_id=config_id,
        config_hash=config_hash,
    )

    assert mode == "validate_and_reuse_externalized_sqi_evidence_v1"
    assert summary["route_window_sqi_evidence_row_count"] == 1
    assert len(quality["rows"]) == 1
    assert {name: (cell / name).read_bytes() for name in immutable_names} == before


def test_v5_recovery_rejects_externalized_evidence_hash_conflict(
    tmp_path: Path,
) -> None:
    cell, config_id, config_hash = _externalized_cell_fixture(tmp_path)
    metrics_path = cell / "metrics_per_fold_seed.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["cells"][0]["route_window_sqi_evidence_sha256"] = "0" * 64
    _write_json(metrics_path, metrics)

    with pytest.raises(ValueError, match="SHA-256 drift"):
        recovery._externalize_legacy_sqi_payload(
            cell,
            config_id=config_id,
            config_hash=config_hash,
        )


def test_v5_recovery_rejects_missing_checkpoint_state(tmp_path: Path) -> None:
    cell, _, _ = _externalized_cell_fixture(tmp_path)
    (cell / "model_checkpoint" / "model.joblib").unlink()

    with pytest.raises(ValueError, match="payload integrity drift"):
        recovery._safe_mandatory_artifacts(cell)


def test_v5_recovery_rejects_tampered_checkpoint_state(tmp_path: Path) -> None:
    cell, _, _ = _externalized_cell_fixture(tmp_path)
    (cell / "model_checkpoint" / "model.joblib").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="payload integrity drift"):
        recovery._safe_mandatory_artifacts(cell)


def test_v5_recovery_rejects_extra_checkpoint_payload(tmp_path: Path) -> None:
    cell, _, _ = _externalized_cell_fixture(tmp_path)
    (cell / "model_checkpoint" / "unverified.bin").write_bytes(b"extra")

    with pytest.raises(ValueError, match="missing or extra payloads"):
        recovery._safe_mandatory_artifacts(cell)


def test_v5_recovery_rejects_checkpoint_payload_symlink(tmp_path: Path) -> None:
    cell, _, _ = _externalized_cell_fixture(tmp_path)
    state = cell / "model_checkpoint" / "model.joblib"
    state.unlink()
    outside = tmp_path / "outside-model.joblib"
    outside.write_bytes(b"minimal-model-state")
    state.symlink_to(outside)

    with pytest.raises(ValueError, match="payload integrity drift"):
        recovery._safe_mandatory_artifacts(cell)


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../outside.json",
        "./artifact.json",
        "nested//artifact.json",
        "/absolute/artifact.json",
    ),
)
def test_v5_recovery_rejects_noncanonical_mandatory_paths(
    tmp_path: Path, unsafe_name: str,
) -> None:
    cell = tmp_path / "cell"
    cell.mkdir()
    _write_json(
        cell / "run_manifest.json",
        {
            "schema_version": "ppg_frailty.run_manifest.v2",
            "status": "passed",
            "mandatory_artifacts": ["run_manifest.json", unsafe_name],
        },
    )

    with pytest.raises(ValueError, match="unsafe"):
        recovery._safe_mandatory_artifacts(cell)


def test_v5_recovery_rejects_mandatory_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_json(outside / "artifact.json", {"outside": True})
    cell = tmp_path / "cell"
    cell.mkdir()
    (cell / "linked").symlink_to(outside, target_is_directory=True)
    _write_json(
        cell / "run_manifest.json",
        {
            "schema_version": "ppg_frailty.run_manifest.v2",
            "status": "passed",
            "mandatory_artifacts": [
                "run_manifest.json",
                "linked/artifact.json",
            ],
        },
    )

    with pytest.raises(ValueError, match="symlink"):
        recovery._safe_mandatory_artifacts(cell)

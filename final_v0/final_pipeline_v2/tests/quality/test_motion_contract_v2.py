"""Small contract/import tests only: no scientific training, CV, ablation, or PTT run."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from ppg_frailty.models.motion import (
    HISTORICAL_LIGHT_CNN_CHANNELS,
    LightCnnArchitecture,
    build_historical_light_cnn_backup,
    build_formal_motion_cnn,
    build_motion_derived_augmentation_cnn,
    build_parameterized_light_cnn,
    count_trainable_parameters,
)
from ppg_frailty.quality.motion import (
    HISTORICAL_LIGHT_CNN_EVIDENCE,
    MOTION_INTERNAL_EVIDENCE_SCHEMA,
    MOTION_MAJOR_METRIC_FIELDS,
    MOTION_OOF_PARTICIPANT_REPEAT_ROWS,
    MOTION_SPLIT_CSV_SHA256,
    MOTION_SPLIT_REGISTRY_ID,
    MOTION_THRESHOLD_FIT_SCOPE,
    MOTION_THRESHOLD_RULE_ID,
    MOTION_THRESHOLD_SCORE_ORIGIN,
    MotionFoldJob,
    MotionOptionId,
    fit_train_only_midpoint_threshold,
    load_motion_fold_jobs,
    motion_activity_label,
    motion_contract_payload,
    resolve_motion_option,
    validate_motion_major_metrics,
)
from ppg_frailty.quality.motion_runner import (
    FormalMotionEntryRequiredError,
    MOTION_WINDOW_OOF_SCHEMA,
    MotionFitContext,
    MotionFittedArtifact,
    MotionWindowExample,
    _write_parquet,
    audit_ptt_external_readiness,
    run_internal_motion_oof,
    run_ptt_external_evaluation,
)
from ppg_frailty.quality.motion_reference import (
    PttImuUnitEvidenceRequired,
    load_ptt_imu_unit_evidence,
    run_formal_internal_motion_reference,
    run_formal_ptt_motion_reference,
)
from ppg_frailty.data.external_manifest import (
    M2_EXTERNAL_RELATIVE_PATH,
    PTT_ADOPTED_ACCELERATION_CONVERSION,
    PTT_ADOPTED_ACCELERATION_UNIT,
    PTT_DATASET_ID,
    PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
    PTT_IMU_UNIT_EVIDENCE_SHA256,
    load_m2_external_manifest,
)
from ppg_frailty.quality.motion_adapters import write_formal_motion_input_schema
from ppg_frailty.representations.motion import MOTION_NETWORK_SCHEMA_SHA256


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frozen_threshold(participant_ids: tuple[str, ...]) -> dict[str, object]:
    scores: list[float] = []
    labels: list[int] = []
    row_participants: list[str] = []
    for participant_id in participant_ids:
        scores.extend((0.2, 0.8))
        labels.extend((0, 1))
        row_participants.extend((participant_id, participant_id))
    return fit_train_only_midpoint_threshold(
        scores,
        labels,
        row_participants,
        training_participant_ids=participant_ids,
    ).as_dict()


def _complete_internal_evidence(tmp_path: Path) -> dict[str, object]:
    split_path = PIPELINE_ROOT / "splits" / "sgkf5_seed42_v2.csv"
    jobs = load_motion_fold_jobs(split_path)
    all_participants = tuple(
        sorted(set(jobs[0].train_participant_ids) | set(jobs[0].oof_participant_ids))
    )
    schema_path = tmp_path / "model_input_schema.json"
    _, schema_file_sha = write_formal_motion_input_schema(schema_path)
    oof_path = tmp_path / "motion_window_oof.parquet"

    cell_evidence: list[dict[str, object]] = []
    cell_metrics: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    for job in jobs:
        cell_dir = tmp_path / f"repeat_{job.repeat_index}" / f"fold_{job.fold_index}"
        cell_dir.mkdir(parents=True)
        model_path = cell_dir / "model.mock"
        model_path.write_bytes(
            f"unit-test-model-{job.repeat_index}-{job.fold_index}".encode("ascii")
        )
        threshold = _frozen_threshold(job.train_participant_ids)
        model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        for participant_id in job.oof_participant_ids:
            for role, label, probability in (("B", 0, 0.2), ("S1", 1, 0.8)):
                oof_rows.append(
                    {
                        "schema_version": MOTION_WINDOW_OOF_SCHEMA,
                        "repeat_index": job.repeat_index,
                        "fold_index": job.fold_index,
                        "split_seed": job.split_seed,
                        "training_seed": job.training_seed,
                        "window_id": f"{participant_id}_{role}",
                        "participant_id": participant_id,
                        "file_id": f"{participant_id}_{role}",
                        "role_family": role,
                        "activity_label": label,
                        "p_active": probability,
                        "threshold": 0.5,
                        "predicted_activity": label,
                        "score_origin": "strict_outer_oof_model_prediction",
                        "threshold_score_origin": MOTION_THRESHOLD_SCORE_ORIGIN,
                        "model_artifact_sha256": model_sha,
                    }
                )
        cell_evidence.append(
            {
                "repeat_index": job.repeat_index,
                "fold_index": job.fold_index,
                "training_participant_count": len(job.train_participant_ids),
                "oof_participant_count": len(job.oof_participant_ids),
                "threshold_fit_scope": MOTION_THRESHOLD_FIT_SCOPE,
                "model_artifact_path": str(model_path),
                "model_artifact_sha256": model_sha,
                "threshold": threshold,
                "threshold_artifact_sha256": _payload_sha256(threshold),
                "oof_window_count": 2 * len(job.oof_participant_ids),
            }
        )
        cell_metrics.append(
            {
                "repeat_index": job.repeat_index,
                "fold_index": job.fold_index,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "ece": 0.2,
            }
        )
    oof_sha = _write_parquet(oof_path, oof_rows)

    final_dir = tmp_path / "final_all_internal"
    final_dir.mkdir()
    final_model_path = final_dir / "model.mock"
    final_model_path.write_bytes(b"unit-test-final-model")
    final_threshold = _frozen_threshold(all_participants)
    return {
        "schema_version": MOTION_INTERNAL_EVIDENCE_SCHEMA,
        "execution_status": "completed_formal_not_smoke",
        "scientific_scope": "frailty29_single_sgkf5_oof",
        "model_id": "formal_local_supervised_motion_detector_v2",
        "split_registry_id": MOTION_SPLIT_REGISTRY_ID,
        "split_registry_csv_sha256": MOTION_SPLIT_CSV_SHA256,
        "split_registry_csv_path": str(split_path),
        "participant_count": 29,
        "oof_participant_repeat_rows": MOTION_OOF_PARTICIPANT_REPEAT_ROWS,
        "model_input_schema_status": "frozen_before_training",
        "model_input_schema_path": str(schema_path),
        "model_input_schema_file_sha256": schema_file_sha,
        "model_input_schema_sha256": MOTION_NETWORK_SCHEMA_SHA256,
        "threshold_rule_id": MOTION_THRESHOLD_RULE_ID,
        "threshold_score_origin": MOTION_THRESHOLD_SCORE_ORIGIN,
        "major_metric_names": list(MOTION_MAJOR_METRIC_FIELDS),
        "model_and_threshold_frozen_before_ptt": True,
        "cell_evidence": cell_evidence,
        "cell_metrics": cell_metrics,
        "window_oof_row_count": len(oof_rows),
        "window_oof_parquet_path": str(oof_path),
        "window_oof_parquet_sha256": oof_sha,
        "major_metrics": {
            "participant_macro_balanced_accuracy": 1.0,
            "worst_fold_balanced_accuracy": 1.0,
            "participant_macro_f1": 1.0,
            "ece": 0.2,
            "parameter_count": 1,
            "inference_cost": {
                "device": "mock_cpu",
                "batch_size": 1,
                "window_samples": 3200,
                "warmup_iterations": 1,
                "timed_iterations": 1,
                "latency_ms_per_window_p50": 1.0,
                "latency_ms_per_window_p95": 1.0,
                "throughput_windows_per_second": 1.0,
            },
        },
        "final_model": {
            "artifact_path": str(final_model_path),
            "artifact_sha256": hashlib.sha256(
                final_model_path.read_bytes()
            ).hexdigest(),
            "training_participant_ids": list(all_participants),
            "parameter_count": 1,
            "inference_cost": {
                "device": "mock_cpu",
                "batch_size": 1,
                "window_samples": 3200,
                "warmup_iterations": 1,
                "timed_iterations": 1,
                "latency_ms_per_window_p50": 1.0,
                "latency_ms_per_window_p95": 1.0,
                "throughput_windows_per_second": 1.0,
            },
        },
        "final_threshold": final_threshold,
        "final_threshold_artifact_sha256": _payload_sha256(final_threshold),
    }


def test_three_options_and_role_labels_are_exact() -> None:
    assert resolve_motion_option(None).option_id is MotionOptionId.SQI_ONLY
    assert resolve_motion_option("sqi_plus_motion_override").formal_default is False
    assert resolve_motion_option("historical_light_cnn_backup").execution_status == (
        "historical_frozen_backup_not_v2_run"
    )
    assert [motion_activity_label(role) for role in ("B", "R1", "R4", "S1", "W2")] == [
        0,
        0,
        0,
        1,
        1,
    ]


def test_frailty29_registry_resolves_single_seed42_sgkf5() -> None:
    jobs = load_motion_fold_jobs(
        PIPELINE_ROOT / "splits" / "sgkf5_seed42_v2.csv"
    )
    assert len(jobs) == 5
    assert {job.split_seed for job in jobs} == {42}
    assert {job.training_seed for job in jobs} == {42}
    assert {(job.repeat_index, job.fold_index) for job in jobs} == {
        (0, fold_index)
        for fold_index in range(5)
    }
    for job in jobs:
        assert not set(job.train_participant_ids) & set(job.oof_participant_ids)
        assert len(job.train_participant_ids) + len(job.oof_participant_ids) == 29


def test_midpoint_is_participant_balanced_train_only_and_rejects_leakage() -> None:
    artifact = fit_train_only_midpoint_threshold(
        scores=[0.1, 0.2, 0.8, 0.9, 0.15, 0.85],
        labels=[0, 0, 1, 1, 0, 1],
        participant_ids=["p1", "p1", "p1", "p1", "p2", "p2"],
        training_participant_ids=["p1", "p2"],
        forbidden_oof_participant_ids=["held_out"],
        forbidden_ptt_participant_ids=["ptt:s1"],
    )
    assert artifact.static_center == pytest.approx(0.15)
    assert artifact.motion_center == pytest.approx(0.85)
    assert artifact.threshold == pytest.approx(0.5)
    artifact.validate()

    with pytest.raises(ValueError, match="OOF/PTT"):
        fit_train_only_midpoint_threshold(
            scores=[0.1, 0.9],
            labels=[0, 1],
            participant_ids=["held_out", "held_out"],
            training_participant_ids=["held_out"],
            forbidden_oof_participant_ids=["held_out"],
        )
    with pytest.raises(ValueError, match="may not use OOF"):
        fit_train_only_midpoint_threshold(
            scores=[0.1, 0.9],
            labels=[0, 1],
            participant_ids=["p1", "p1"],
            training_participant_ids=["p1"],
            score_origin="outer_oof_predictions",
        )


def test_mock_bytes_and_arbitrary_evidence_are_not_report_ready(
    tmp_path: Path,
) -> None:
    evidence = _complete_internal_evidence(tmp_path)
    archive = tmp_path / "evidence.json"
    archive.write_text(json.dumps(evidence, allow_nan=False), encoding="utf-8")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    audit = audit_ptt_external_readiness(archive, expected_sha256=archive_sha)
    assert audit.ready is False
    assert "canonical_internal_motion_entry_missing" in audit.reasons
    assert "canonical_internal_source_evidence_missing" in audit.reasons
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_ptt_external_readiness(archive, expected_sha256="0" * 64)

def test_major_metric_schema_includes_only_confirmed_comparison_summary() -> None:
    validate_motion_major_metrics(
        {
            "participant_macro_balanced_accuracy": 0.71,
            "worst_fold_balanced_accuracy": 0.62,
            "participant_macro_f1": 0.69,
            "ece": 0.08,
            "parameter_count": 6181,
            "inference_cost": {
                "device": "cpu_model_string",
                "batch_size": 1,
                "window_samples": 2048,
                "warmup_iterations": 10,
                "timed_iterations": 100,
                "latency_ms_per_window_p50": 1.0,
                "latency_ms_per_window_p95": 1.5,
                "throughput_windows_per_second": 900.0,
            },
        }
    )
    with pytest.raises(ValueError, match="missing fields"):
        validate_motion_major_metrics({})


def test_formal_builder_needs_explicit_schema_and_historical_model_is_exact() -> None:
    with pytest.raises(ValueError, match="at least one"):
        LightCnnArchitecture(channel_names=()).validate()
    with pytest.raises(ValueError, match="unique"):
        LightCnnArchitecture(channel_names=("RED", "RED")).validate()

    torch = pytest.importorskip("torch")
    explicit = build_parameterized_light_cnn(("RED", "IR", "AX"))
    assert explicit.architecture.in_channels == 3
    assert explicit(torch.zeros(2, 3, 64)).shape == (2,)
    with pytest.raises(ValueError, match="channel count"):
        explicit(torch.zeros(2, 2, 64))
    formal = build_formal_motion_cnn()
    assert formal.architecture.in_channels == 8
    assert formal.motion_input_contract["window_samples"] == 3200
    assert formal.motion_input_contract["derived_signal_augmentation"] is False
    augmented = build_motion_derived_augmentation_cnn()
    assert augmented.architecture.in_channels == 11
    assert augmented.motion_input_contract["derived_signal_augmentation"] is True
    assert augmented.motion_input_contract["frailty_predictor_eligible"] is False

    historical = build_historical_light_cnn_backup()
    assert historical.architecture.channel_names == HISTORICAL_LIGHT_CNN_CHANNELS
    assert historical.architecture.parameter_count == 6181
    assert count_trainable_parameters(historical) == 6181
    assert HISTORICAL_LIGHT_CNN_EVIDENCE.external_balanced_accuracy == pytest.approx(
        0.7802173951757236
    )
    assert HISTORICAL_LIGHT_CNN_EVIDENCE.status.endswith("not_v2_execution")


def test_yaml_and_python_contract_agree_without_running_science() -> None:
    yaml_payload = yaml.safe_load(
        (PIPELINE_ROOT / "configs" / "motion_detector_contract_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    python_payload = motion_contract_payload()
    assert yaml_payload["schema_version"] == python_payload["schema_version"]
    assert yaml_payload["selection"]["default_option"] == python_payload["default_option"]
    assert yaml_payload["internal_training_and_oof"]["split_csv_sha256"] == (
        python_payload["internal_protocol"]["registry_csv_sha256"]
    )
    assert yaml_payload["threshold"]["rule_id"] == python_payload["threshold"]["rule_id"]
    assert yaml_payload["ptt_external_readiness_audit"]["split_csv_sha256"] == (
        python_payload["external_ptt_readiness_audit"]["registry_csv_sha256"]
    )
    assert yaml_payload["ptt_external_readiness_audit"]["unit_evidence_sha256"] == (
        python_payload["external_ptt_readiness_audit"]["unit_evidence_sha256"]
    )
    assert yaml_payload["ptt_external_readiness_audit"]["acceleration_source_unit"] == (
        PTT_ADOPTED_ACCELERATION_UNIT
    )
    assert yaml_payload["ptt_external_readiness_audit"]["acceleration_conversion"] == (
        PTT_ADOPTED_ACCELERATION_CONVERSION
    )
    assert yaml_payload["major_comparison_metrics"] == python_payload["major_metrics"]
    assert yaml_payload["formal_model"]["network_tensor_schema"]["implied_channel_count"] == 8
    assert yaml_payload["formal_model"]["network_tensor_schema"]["ordered_channels"] == (
        python_payload["formal_input"]["network_channel_schema"]
    )
    assert (
        yaml_payload["derived_signal_augmentation_ablation"]["network_tensor_schema"][
            "implied_channel_count"
        ]
        == 11
    )
    assert (
        yaml_payload["derived_signal_augmentation_ablation"]["network_tensor_schema"][
            "ordered_channels"
        ]
        == python_payload["derived_signal_augmentation_ablation"][
            "network_channel_schema"
        ]
    )
    assert (
        yaml_payload["derived_signal_augmentation_ablation"]["frailty_predictor_eligible"]
        is False
    )
    assert yaml_payload["internal_training_and_oof"]["n_repeats"] == 1
    assert yaml_payload["internal_training_and_oof"]["split_seeds"] == [42]
    assert yaml_payload["internal_training_and_oof"]["training_seed"] == 42


def test_injected_runner_cannot_enter_formal_even_with_valid_sgkf5(
    tmp_path: Path,
) -> None:
    split_path = PIPELINE_ROOT / "splits" / "sgkf5_seed42_v2.csv"
    jobs = list(load_motion_fold_jobs(split_path))
    participants = tuple(
        sorted(set(jobs[0].train_participant_ids) | set(jobs[0].oof_participant_ids))
    )
    first, second = jobs[0], jobs[1]
    first_id, second_id = first.oof_participant_ids[0], second.oof_participant_ids[0]
    first_oof = tuple(
        second_id if value == first_id else value for value in first.oof_participant_ids
    )
    second_oof = tuple(
        first_id if value == second_id else value for value in second.oof_participant_ids
    )
    jobs[0] = replace(
        first,
        train_participant_ids=tuple(sorted(set(participants) - set(first_oof))),
        oof_participant_ids=first_oof,
    )
    jobs[1] = replace(
        second,
        train_participant_ids=tuple(sorted(set(participants) - set(second_oof))),
        oof_participant_ids=second_oof,
    )
    examples = tuple(
        MotionWindowExample(
            window_id=f"{participant}_{role}",
            participant_id=participant,
            file_id=f"{participant}_{role}",
            role_or_activity=role,
            activity_label=label,
            values=np.asarray([0.1], dtype=np.float64),
            dataset_id="synthetic_split_guard",
        )
        for participant in participants
        for role, label in (("B", 0), ("S1", 1))
    )

    def must_not_fit(*_args, **_kwargs):
        raise AssertionError("fit callback reached before split binding rejection")

    with pytest.raises(FormalMotionEntryRequiredError, match="injected examples"):
        run_internal_motion_oof(
            examples,
            jobs,
            fit_model=must_not_fit,
            predict_probability=lambda *_args: (),
            model_input_schema_path=tmp_path / "not_reached.json",
            expected_model_input_schema_sha256="0" * 64,
            output_dir=tmp_path / "out",
            motion_split_csv_path=split_path,
            execution_mode="formal",
        )


def test_canonical_signatures_expose_no_examples_callbacks_or_arbitrary_model() -> None:
    forbidden = {
        "examples",
        "fold_jobs",
        "fit_model",
        "predict_probability",
        "load_frozen_model",
        "model",
        "tensor",
    }
    internal = set(inspect.signature(run_formal_internal_motion_reference).parameters)
    external = set(inspect.signature(run_formal_ptt_motion_reference).parameters)
    assert not forbidden & internal
    assert not forbidden & external


def test_injected_ptt_runner_always_rejects_before_callbacks(tmp_path: Path) -> None:
    touched = False

    def must_not_run(*_args, **_kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("injected PTT callback was reached")

    arbitrary = MotionWindowExample(
        window_id="arbitrary",
        participant_id="arbitrary",
        file_id="arbitrary",
        role_or_activity="sit",
        activity_label=0,
        values=np.zeros((11, 3200), dtype=np.float32),
        dataset_id="arbitrary",
    )
    with pytest.raises(FormalMotionEntryRequiredError, match="injected PTT"):
        run_ptt_external_evaluation(
            [arbitrary],
            internal_evidence_path=tmp_path / "arbitrary.json",
            expected_internal_evidence_sha256="0" * 64,
            ptt_split_csv=tmp_path / "arbitrary.csv",
            load_frozen_model=must_not_run,
            predict_probability=must_not_run,
            output_dir=tmp_path / "out",
        )
    assert touched is False


def test_ptt_unit_conflict_is_structured_and_official_metadata_alone_cannot_bypass(
    tmp_path: Path,
) -> None:
    with pytest.raises(PttImuUnitEvidenceRequired) as captured:
        run_formal_ptt_motion_reference(
            REPOSITORY_ROOT,
            internal_evidence_path=tmp_path / "not_reached.json",
            expected_internal_evidence_sha256="0" * 64,
            output_dir=tmp_path / "not_reached",
        )
    assert captured.value.payload["ready"] is False
    assert captured.value.payload["unit_guessing_allowed"] is False
    assert (
        captured.value.payload["concrete_conflict_evidence"]["status"]
        == "project_resolved_v2_036_source_manifest_conflict_retained"
    )

    records = [
        row
        for row in load_m2_external_manifest(
            REPOSITORY_ROOT / M2_EXTERNAL_RELATIVE_PATH
        )
        if row.dataset_id == PTT_DATASET_ID
    ]
    canonical = load_ptt_imu_unit_evidence(
        REPOSITORY_ROOT / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
        expected_sha256=PTT_IMU_UNIT_EVIDENCE_SHA256,
        expected_records=records,
    )
    official_only = replace(
        canonical,
        acceleration_unit="g",
        acceleration_conversion="g_to_m_per_s2",
        resolution_basis="official_metadata_only",
    )
    with pytest.raises(ValueError, match="identity/status"):
        official_only.validate(records)


def test_injected_one_job_smoke_exercises_runner_without_scientific_cv(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"channels":["explicit_test_value"]}\n', encoding="utf-8")
    schema_sha = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    examples = [
        MotionWindowExample(
            window_id=f"{participant}_{role}_{index}",
            participant_id=participant,
            file_id=f"{participant}_{role}",
            role_or_activity=role,
            activity_label=0 if role == "B" else 1,
            values=np.asarray([score], dtype=np.float64),
            dataset_id="synthetic_import_smoke",
        )
        for participant, offset in (("p1", 0.0), ("p2", 0.02), ("p3", 0.04), ("p4", 0.06))
        for role, base in (("B", 0.15), ("S1", 0.85))
        for index, score in enumerate((base + offset, base + offset + 0.01))
    ]
    job = MotionFoldJob(
        repeat_index=0,
        fold_index=0,
        split_seed=42,
        training_seed=42,
        train_participant_ids=("p1", "p2"),
        oof_participant_ids=("p3", "p4"),
    )

    def fit_model(rows, context: MotionFitContext) -> MotionFittedArtifact:
        artifact = context.artifact_directory / "model.mock"
        artifact.write_bytes(b"mock-only-no-training")
        return MotionFittedArtifact(
            runtime_model=object(),
            model_id="formal_local_supervised_motion_detector_v2",
            artifact_path=str(artifact),
            artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            model_input_schema_sha256=context.model_input_schema_sha256,
            training_participant_ids=context.training_participant_ids,
            parameter_count=1,
            inference_cost={
                "device": "mock_cpu",
                "batch_size": 1,
                "window_samples": 1,
                "warmup_iterations": 1,
                "timed_iterations": 1,
                "latency_ms_per_window_p50": 1.0,
                "latency_ms_per_window_p95": 1.0,
                "throughput_windows_per_second": 1.0,
            },
        )

    def predict_probability(_model, rows):
        assert all(tuple(vars(row)) == ("values",) for row in rows)
        return [float(row.values[0]) for row in rows]

    result = run_internal_motion_oof(
        examples,
        [job],
        fit_model=fit_model,
        predict_probability=predict_probability,
        model_input_schema_path=schema_path,
        expected_model_input_schema_sha256=schema_sha,
        output_dir=tmp_path / "smoke",
        execution_mode="smoke",
        write_artifacts=True,
    )
    assert result.evidence["execution_status"] == "completed_smoke_not_formal"
    assert len(result.window_oof_rows) == 8
    assert result.evidence_path is not None
    assert result.evidence_sha256 is not None
    smoke_audit = audit_ptt_external_readiness(
        result.evidence_path,
        expected_sha256=result.evidence_sha256,
    )
    assert smoke_audit.ready is False
    assert "internal_sgkf5_not_completed_formally" in smoke_audit.reasons

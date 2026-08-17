"""Tiny synthetic smoke for the explicit run-directory comparison orchestrator."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest

from ppg_frailty.cli import build_parser
from ppg_frailty.contracts import to_strict_json_value
from ppg_frailty.experiment import (
    _read_trusted_comparison_run,
    _strict_json,
    _validate_onnx_certificate_parity_metrics,
    _write_artifact_index,
    _write_empty_oof,
    build_comparison_archive_from_run_directories,
    execute_final_refit_from_verified_artifacts,
    final_refit_preflight_from_verified_artifacts,
    verify_manual_selection_record,
    winner_release_preflight,
    write_manual_selection_record,
)
from ppg_frailty.onnx_winner import (
    _verify_artifact_index as verify_onnx_artifact_index,
    _write_artifact_index as write_onnx_artifact_index,
    produce_onnx_winner_certificate,
    validate_certificate_export_policy,
)
from ppg_frailty.provenance import stable_payload_sha256
from ppg_frailty.training import (
    OofPredictionRow,
    OofWriter,
    ParticipantPrediction,
    build_config_metrics_from_predictions_and_fold_summaries,
    read_oof_parquet,
    verify_comparison_archive,
)


_SPLIT_SEEDS = (42, 10042, 20042, 30042, 40042)
ROOT = Path(__file__).resolve().parents[2]


class ComparisonArchiveOrchestratorTests(unittest.TestCase):
    def test_onnx_winner_gate_rejects_large_error_and_untrusted_producer(self) -> None:
        with self.assertRaisesRegex(ValueError, "parity_metrics_invalid"):
            _validate_onnx_certificate_parity_metrics(
                {
                    "absolute_tolerance": 1e-5,
                    "relative_tolerance": 1e-5,
                    "maximum_absolute_error": 0.9,
                    "maximum_relative_error": 0.9,
                    "mean_absolute_error": 0.2,
                    "argmax_class_order_match": True,
                    "parity_passed": True,
                }
            )
        valid_policy = {
            "opset_version": 17,
            "absolute_tolerance": 1e-5,
            "relative_tolerance": 1e-5,
            "maximum_absolute_error": 0.0,
            "maximum_relative_error": 0.0,
            "mean_absolute_error": 0.0,
            "argmax_class_order_match": True,
            "parity_passed": True,
        }
        validate_certificate_export_policy(valid_policy)
        with self.assertRaisesRegex(ValueError, "opset_policy_invalid"):
            validate_certificate_export_policy(
                {**valid_policy, "opset_version": 18}
            )
        with self.assertRaisesRegex(ValueError, "parity_metrics_invalid"):
            validate_certificate_export_policy(
                {
                    **valid_policy,
                    "absolute_tolerance": 1.0,
                    "relative_tolerance": 1.0,
                    "maximum_absolute_error": 0.9,
                    "maximum_relative_error": 0.9,
                    "mean_absolute_error": 0.2,
                }
            )
        producer_parameters = inspect.signature(
            produce_onnx_winner_certificate
        ).parameters
        for forbidden in (
            "opset_version", "absolute_tolerance", "relative_tolerance",
        ):
            self.assertNotIn(forbidden, producer_parameters)
        for required in (
            "expected_final_bundle_manifest_sha256",
            "final_refit_attestation_path",
            "expected_final_refit_attestation_sha256",
        ):
            self.assertIn(required, producer_parameters)
        with self.assertRaisesRegex(PermissionError, "confirm"):
            produce_onnx_winner_certificate(
                bundle_directory="not_reached",
                selection_record="caller_authored_selection.json",
                expected_selection_file_sha256="2" * 64,
                expected_final_bundle_manifest_sha256="3" * 64,
                final_refit_attestation_path="not_reached_attestation.json",
                expected_final_refit_attestation_sha256="4" * 64,
                config_path="not_reached.yaml",
                output_directory="not_reached",
                confirm_onnx_execution=False,
            )
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            producer = Path(directory)
            artifact = producer / "winner.onnx"
            artifact.write_bytes(b"reviewed-tiny-artifact")
            write_onnx_artifact_index(producer)
            verify_onnx_artifact_index(producer)
            artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "artifact hash drift"):
                verify_onnx_artifact_index(producer)

    def _run_fixture(
        self,
        root: Path,
        *,
        config_id: str,
        machine_id: str,
        probability_shift: float,
        source_snapshot_hash: str = "6" * 64,
    ) -> Path:
        run_root = root / config_id
        run_root.mkdir()
        config_hash = hashlib.sha256(config_id.encode("utf-8")).hexdigest()
        source_state = {
            "schema_version": "ppg_frailty.scientific_source_gate.v1",
            "v2_path": "final_v0/final_pipeline_v2",
            "git_head_commit": "a" * 40,
            "v2_pyproject_tracked": True,
            "status_entry_count": 0,
            "status_sha256": hashlib.sha256(b"").hexdigest(),
            "status_preview": [],
            "tracked_and_clean": True,
        }
        dependency_rows = [
            {
                "profile_id": profile_id,
                "requirements_file": f"requirements/{profile_id}.txt",
                "lock_status": "validated_exact_lock",
                "python_version": "3.11.14",
                "requirements_sha256": "b" * 64,
                "resolved_packages": ["fixture==1"],
                "live_runtime": {"live_exact_match": True},
                "exact_lock_ready": True,
            }
            for profile_id in ("core", "formal_benchmark")
        ]
        dependency_gate = {
            "schema_version": "ppg_frailty.dependency_gate.v2",
            "pipeline_generation": "final_pipeline_v2",
            "operation": "formal_benchmark",
            "config_id": config_id,
            "required_profile_ids": ["core", "formal_benchmark"],
            "require_exact_lock": True,
            "all_required_exact_locks_ready": True,
            "profiles": dependency_rows,
        }
        baseline = {
            "schema_version": "ppg_frailty.formal_execution_gate_capture.v2",
            "pipeline_generation": "final_pipeline_v2",
            "operation": "formal_benchmark",
            "config_id": config_id,
            "config_hash": config_hash,
            "config_source_relative_path":
                "configs/synthetic_tracked_fixture.yaml",
            "config_source_file_sha256": "c" * 64,
            "source_tree_state": source_state,
            "source_snapshot_sha256": source_snapshot_hash,
            "dependency_gate": dependency_gate,
            "dependency_gate_sha256":
                stable_payload_sha256(dependency_gate),
        }
        baseline_sha256 = stable_payload_sha256(baseline)
        phases = ["entry_preflight"]
        for phase_repeat in range(5):
            for phase_fold in range(5):
                prefix = (
                    f"repeat_{phase_repeat:02d}_fold_{phase_fold:02d}"
                )
                phases.extend((f"{prefix}_pre", f"{prefix}_post"))
        phases.extend(("root_artifacts_pre", "root_commit_pre"))
        checkpoints = []
        for phase in phases:
            checkpoint_payload = {
                "phase": phase,
                "capture_sha256": baseline_sha256,
                "source_snapshot_sha256": source_snapshot_hash,
                "dependency_gate_sha256":
                    baseline["dependency_gate_sha256"],
            }
            checkpoints.append({
                **checkpoint_payload,
                "checkpoint_sha256":
                    stable_payload_sha256(checkpoint_payload),
            })
        checkpoint_by_phase = {
            row["phase"]: row for row in checkpoints
        }
        oof_rows: list[OofPredictionRow] = []
        member_rows: list[OofPredictionRow] = []
        ensemble_base = {
            "inception_full_five_member_ensemble": "inception_full",
            "inception_matrix_five_member_ensemble": "inception_matrix",
        }.get(machine_id)
        member_seeds = (42, 10042, 20042, 30042, 40042)
        predictions: list[ParticipantPrediction] = []
        summaries: list[dict[str, object]] = []
        fold_ba: dict[str, float] = {}
        fold_confusions: dict[str, list[list[int]]] = {}
        fold_rosters: dict[str, list[str]] = {}
        participants = [f"P{index:02d}" for index in range(29)]
        fold_by_index = [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6 + [4] * 5
        frozen_provenance = {
            "architecture_parameters": {"fixture": "same_comparison_base"},
            "input_channels_order": ["fixture_feature"],
            "sampling_rate_hz": 400.0,
            "window_plan": {"window_s": 5.0},
            "hop_plan": {"hop_s": 2.5},
            "normalization": {"method": "outer_train_only"},
            "padding_mask": {"padding": "none"},
            "feature_schema_hash": "4" * 64,
            "sqi_routing": {"mode": "off"},
            "loss": "not_applicable_estimator_native",
            "class_weighting": "outer_train_participant_balanced",
            "sampler": "none",
            "epoch_rule": {"rule": "not_applicable", "fixed_epochs": None},
            "optimizer": "not_applicable_estimator_native",
            "learning_rate": "not_applicable_estimator_native",
            "weight_decay": "not_applicable_estimator_native",
            "dropout": "not_applicable_estimator_native",
            "label_smoothing": "not_applicable_estimator_native",
            "gradient_clipping": {
                "enabled": False,
                "max_norm": None,
                "reason": "not_applicable_estimator_native",
            },
            "seed_policy": (
                "cv_fixed_five_member_seed_roster"
                if ensemble_base
                else "outer_cv_repeat_seed_equals_split_seed"
            ),
            "random_seeds": list(member_seeds if ensemble_base else (42,)),
            "fold_hash": "2" * 64,
            "aggregation": {"line": "line_a_equal_files"},
            "calibration": {"fit_scope": "outer_training_only"},
        }
        for repeat, split_seed in enumerate(_SPLIT_SEEDS):
            for fold in range(5):
                key = f"r{repeat}f{fold}"
                matrix = [[0, 0, 0] for _ in range(3)]
                roster = [
                    participant_id
                    for index, participant_id in enumerate(participants)
                    if fold_by_index[index] == fold
                ]
                for index, participant_id in enumerate(participants):
                    if fold_by_index[index] == fold:
                        label = index % 3
                        matrix[label][label] += 1
                fold_confusions[key] = matrix
                fold_rosters[key] = roster
                cell_frozen_provenance = dict(frozen_provenance)
                if not ensemble_base:
                    cell_frozen_provenance["random_seeds"] = [split_seed]
                summary = {
                    "status": "passed",
                    "repeat_index": repeat,
                    "fold_index": fold,
                    "split_seed": split_seed,
                    "training_seed": None if ensemble_base else split_seed,
                    "member_training_seeds": (
                        list(member_seeds) if ensemble_base else []
                    ),
                    "seed_policy": (
                        "cv_fixed_five_member_seed_roster"
                        if ensemble_base
                        else "outer_cv_repeat_seed_equals_split_seed"
                    ),
                    "model_machine_id": machine_id,
                    "class_order": [0, 1, 2],
                    "scientific_scope": "frozen_5x5_scientific_benchmark",
                    "metrics": {
                        "balanced_accuracy": 1.0,
                        "confusion_matrix": matrix,
                    },
                    "operational_metrics": {
                        "status": "measured_explicit_cpu_batch1_request",
                        "parameter_count": 100,
                        "model_latency_p50_ms": 0.25,
                        "model_latency_p95_ms": 0.50,
                    },
                    "frozen_model_run_provenance": cell_frozen_provenance,
                    "quality_mode": "off",
                    "balance_line": "line_a_equal_files",
                    "representation_transform_provenance": {
                        "fixture": "same_comparison_base"
                    },
                    "sqi_calibrator_provenance": {"mode": "off"},
                    "physical_recording_qc": [],
                }
                prefix = f"repeat_{repeat:02d}_fold_{fold:02d}"
                summary["formal_execution_gate_checkpoints"] = {
                    "pre": checkpoint_by_phase[f"{prefix}_pre"],
                    "post": checkpoint_by_phase[f"{prefix}_post"],
                }
                summaries.append(summary)
                fold_ba[key] = 1.0
                cell = run_root / f"repeat_{repeat:02d}_fold_{fold:02d}"
                cell.mkdir()
                _strict_json(
                    cell / "run_manifest.json",
                    {
                        "schema_version": "ppg_frailty.run_manifest.v2",
                        "pipeline_generation": "final_pipeline_v2",
                        "status": "passed",
                        "scientific_scope": "frozen_5x5_scientific_benchmark",
                        "cell": summary,
                    },
                )
                for name in (
                    "metrics_per_fold_seed.json",
                    "confusion_matrices.json",
                    "quality_diagnostics.json",
                ):
                    _strict_json(cell / name, {"status": "synthetic_contract_fixture"})
            for index, participant_id in enumerate(participants):
                label = index % 3
                fold = fold_by_index[index]
                wrong = (label + 1) % 3
                probability = [0.1, 0.1, 0.1]
                probability[label] = 0.8 - probability_shift
                probability[wrong] += probability_shift
                values = tuple(probability)
                subject_row = (
                    OofPredictionRow(
                        participant_id=participant_id,
                        file_id=f"participant::{participant_id}",
                        role="participant",
                        label=label,
                        probabilities=values,
                        repeat=repeat,
                        fold=fold,
                        split_seed=split_seed,
                        training_seed=None if ensemble_base else split_seed,
                        config_hash=config_hash,
                        manifest_hash="1" * 64,
                        fold_hash="2" * 64,
                        preprocessing_hash="3" * 64,
                        feature_hash="4" * 64,
                        model_hash="5" * 64,
                        representation_mode="feature_vector",
                        signal_route="direct",
                        quality_score=1.0,
                        retained=True,
                        level="participant",
                        class_order=(0, 1, 2),
                        source_snapshot_hash=source_snapshot_hash,
                        code_commit="synthetic_tracked_commit",
                        data_schema_id="synthetic_data_v2",
                        feature_schema_id="synthetic_features_v2",
                        model_version="synthetic_model_v2",
                        aggregation_rule="line_a_equal_files",
                        environment_hash="7" * 64,
                        manifest_version="synthetic_manifest_v2",
                        fold_registry_version="synthetic_folds_v2",
                        artifact_reducer_name="identity",
                        artifact_reducer_version="v2",
                        route_status="direct_retained",
                        prediction_kind=(
                            "ensemble_average" if ensemble_base else "single_model"
                        ),
                        member_training_seeds=member_seeds if ensemble_base else (),
                        ensemble_base_model_id=ensemble_base or "",
                    )
                )
                oof_rows.append(subject_row)
                if ensemble_base:
                    for member_index, member_seed in enumerate(member_seeds):
                        member_rows.append(
                            replace(
                                subject_row,
                                training_seed=member_seed,
                                member_index=member_index,
                                prediction_kind="ensemble_member",
                                member_training_seeds=(),
                                model_hash=hashlib.sha256(
                                    f"{config_id}:{member_index}".encode("utf-8")
                                ).hexdigest(),
                            )
                        )
                predictions.append(
                    ParticipantPrediction(participant_id, label, repeat, values)
                )
        for repeat in range(5):
            for fold in range(5):
                cell = run_root / f"repeat_{repeat:02d}_fold_{fold:02d}"
                selected_subjects = tuple(
                    row
                    for row in oof_rows
                    if row.repeat == repeat and row.fold == fold
                )
                OofWriter().write(
                    selected_subjects,
                    cell / "oof_subject_predictions.parquet",
                )
                if ensemble_base:
                    OofWriter().write(
                        tuple(
                            row
                            for row in member_rows
                            if row.repeat == repeat and row.fold == fold
                        ),
                        cell / "oof_member_predictions.parquet",
                    )
                else:
                    _write_empty_oof(
                        cell / "oof_member_predictions.parquet",
                        "single_model_runner_ensemble_comparison_not_executed",
                    )
                _write_empty_oof(
                    cell / "oof_window_predictions.parquet",
                    "synthetic_contract_fixture_no_window_rows",
                )
                _write_empty_oof(
                    cell / "oof_file_predictions.parquet",
                    "synthetic_contract_fixture_no_file_rows",
                )
                _write_artifact_index(cell)
        OofWriter().write(oof_rows, run_root / "oof_subject_predictions.parquet")
        if member_rows:
            OofWriter().write(
                member_rows,
                run_root / "oof_member_predictions.parquet",
            )
        else:
            _write_empty_oof(
                run_root / "oof_member_predictions.parquet",
                "single_model_runner_ensemble_comparison_not_executed",
            )
        _strict_json(
            run_root / "metrics_per_fold_seed.json",
            {
                "schema_version": "ppg_frailty.metrics_per_fold_seed.v2",
                "pipeline_generation": "final_pipeline_v2",
                "status": "passed",
                "cells": summaries,
            },
        )
        metrics, bootstrap = build_config_metrics_from_predictions_and_fold_summaries(
            config_id=config_id,
            registry_role="comparison" if ensemble_base else "reference",
            predictions=predictions,
            fold_balanced_accuracies=fold_ba,
            fold_confusion_matrices=fold_confusions,
            fold_participant_rosters=fold_rosters,
            inference_cost={
                "cpu_batch1_model_only_p50_ms_mean_across_25_outer_cells": 0.25,
                "cpu_batch1_model_only_p95_ms_mean_across_25_outer_cells": 0.50,
            },
            parameter_count=100,
            n_bootstrap_resamples=12,
            bootstrap_seed=42,
            eligible=True,
            exclusion_reason="",
        )
        _strict_json(
            run_root / "config_metrics_v2.json",
            {
                "schema_version": "ppg_frailty.config_metrics.v2",
                "pipeline_generation": "final_pipeline_v2",
                "status": "passed_trusted_metrics_rebuilt_from_typed_oof",
                "config_id": config_id,
                "config_hash": config_hash,
                "independent_test": False,
                "fold_protocol": "frozen_repeated_grouped_5x5",
                "seeds": list(_SPLIT_SEEDS),
                "operational_measurement_status": (
                    "measured_all_25_cells_explicit_request"
                ),
                "config_metrics": to_strict_json_value(asdict(metrics)),
                "bootstrap_results": to_strict_json_value(
                    [asdict(value) for value in bootstrap]
                ),
            },
        )
        _strict_json(
            run_root / "run_manifest.json",
            {
                "schema_version": "ppg_frailty.run_manifest.v2",
                "pipeline_generation": "final_pipeline_v2",
                "status": "passed",
                "scientific_scope": "frozen_5x5_scientific_benchmark",
                "config_id": config_id,
                "config_hash": config_hash,
                "repeat_indices": list(range(5)),
                "fold_indices": list(range(5)),
                "provenance": {
                    "manifest_hash": "1" * 64,
                    "fold_hash": "2" * 64,
                    "source_snapshot_sha256": source_snapshot_hash,
                    "source_tree_state": source_state,
                    "dependency_gate": dependency_gate,
                    "formal_execution_gate_baseline_sha256":
                        baseline_sha256,
                    "root_artifacts_pre_checkpoint":
                        checkpoint_by_phase["root_artifacts_pre"],
                },
            },
        )
        gate_payload = {
            "schema_version":
                "ppg_frailty.formal_execution_gate_evidence.v2",
            "pipeline_generation": "final_pipeline_v2",
            "baseline": baseline,
            "baseline_sha256": baseline_sha256,
            "checkpoints": checkpoints,
        }
        _strict_json(
            run_root / "formal_execution_gate_evidence.json",
            {
                **gate_payload,
                "formal_execution_gate_evidence_sha256":
                    stable_payload_sha256(gate_payload),
            },
        )
        _write_artifact_index(run_root)
        return run_root

    @staticmethod
    def _reindex(directory: Path) -> None:
        (directory / "artifact_index.json").unlink()
        _write_artifact_index(directory)

    def test_tiny_archive_rebuilds_metrics_and_never_selects(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as raw:
            root = Path(raw)
            reference = self._run_fixture(
                root,
                config_id="reference_fixture",
                machine_id="logistic_regression",
                probability_shift=0.0,
            )
            candidate = self._run_fixture(
                root,
                config_id="candidate_fixture",
                machine_id="rbf_svm",
                probability_shift=0.05,
            )
            result = build_comparison_archive_from_run_directories(
                {"reference_fixture": reference, "candidate_fixture": candidate},
                reference_config_id="reference_fixture",
                comparison_family="synthetic_contract_fixture",
                comparison_id="tiny_fixture",
                run_id="run_001",
                output_root=root / "archives",
                n_bootstrap_resamples=12,
                n_permutation_resamples=16,
                statistics_seed=42,
            )
            self.assertEqual(result["status"], "passed")
            archive = Path(result["output_directory"])
            manifest = (archive / "run_manifest.json").read_text(encoding="utf-8")
            self.assertIn('"automatic_selection": false', manifest)
            self.assertIn('"independent_test": false', manifest)
            metrics = (archive / "metrics_all_configs.json").read_text(encoding="utf-8")
            self.assertIn('"macro_f1_lcb95"', metrics)
            index = verify_comparison_archive(archive)
            self.assertEqual(index["schema_version"], "comparison_artifact_index_v2")
            ranking = (archive / "ranking_top10.csv").read_text(encoding="utf-8")
            self.assertIn("balanced_accuracy_lcb95", ranking)
            self.assertIn("macro_f1_lcb95", ranking)
            self.assertEqual(len(ranking.strip().splitlines()), 3)
            self.assertEqual(
                (archive / "selection_record.json").read_text(encoding="utf-8").strip(),
                "[]",
            )
            selection_path = root / "purpose_specific_selection.json"
            selection = write_manual_selection_record(
                archive,
                config_id="candidate_fixture",
                purpose="deployment_candidate",
                human_rationale="explicit human fixture decision",
                output_path=selection_path,
            )
            self.assertEqual(selection["registry_role"], "reference")
            self.assertEqual(selection["config_id"], "candidate_fixture")
            self.assertEqual(
                (archive / "selection_record.json").read_text(encoding="utf-8").strip(),
                "[]",
            )
            self.assertEqual(
                verify_manual_selection_record(selection_path)["purpose"],
                "deployment_candidate",
            )
            with self.assertRaisesRegex(PermissionError, "confirm"):
                final_refit_preflight_from_verified_artifacts(
                    candidate,
                    selection_path,
                    expected_selection_file_sha256="0" * 64,
                    comparison_archive=archive,
                    config_path=root / "not_reached.yaml",
                    confirm_scientific_execution=False,
                )
            with self.assertRaisesRegex(PermissionError, "confirm"):
                execute_final_refit_from_verified_artifacts(
                    candidate,
                    selection_path,
                    expected_selection_file_sha256="0" * 64,
                    comparison_archive=archive,
                    config_path=root / "not_reached.yaml",
                    bundle_directory=root / "not_reached_bundle",
                    confirm_scientific_execution=False,
                )
            refit_parameters = inspect.signature(
                execute_final_refit_from_verified_artifacts
            ).parameters
            for forbidden in (
                "trainer", "full_dataset", "binding", "model_factory",
                "estimator", "metadata", "parity_atol",
            ):
                self.assertNotIn(forbidden, refit_parameters)

    def test_cli_requires_explicit_run_directories(self) -> None:
        """Parser keeps comparison statistics behind an explicit command."""
        parsed = build_parser().parse_args(
            [
                "comparison-archive",
                "--run", "reference_fixture=artifacts/reference_fixture",
                "--run", "candidate_fixture=artifacts/candidate_fixture",
                "--reference-config-id", "reference_fixture",
                "--comparison-family", "models",
                "--comparison-id", "model_compare",
                "--run-id", "run_001",
                "--output-root", "artifacts/comparisons",
                "--allowed-authority-difference", "frozen.architecture_parameters",
            ]
        )
        self.assertEqual(parsed.command, "comparison-archive")
        self.assertEqual(parsed.bootstrap_resamples, 10_000)
        self.assertEqual(parsed.permutation_resamples, 100_000)
        self.assertEqual(
            parsed.allowed_authority_difference,
            ["frozen.architecture_parameters"],
        )

    def test_trusted_reader_revalidates_five_members_plus_average(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as raw:
            run = self._run_fixture(
                Path(raw),
                config_id="ensemble_fixture",
                machine_id="inception_full_five_member_ensemble",
                probability_shift=0.0,
            )
            trusted = _read_trusted_comparison_run(
                "ensemble_fixture",
                run,
                n_bootstrap_resamples=12,
                bootstrap_seed=42,
            )
            self.assertEqual(
                trusted["machine_id"],
                "inception_full_five_member_ensemble",
            )
            averages = read_oof_parquet(run / "oof_subject_predictions.parquet")
            members = read_oof_parquet(run / "oof_member_predictions.parquet")
            self.assertEqual({row.training_seed for row in averages}, {None})
            self.assertEqual(
                {row.training_seed for row in members},
                set(_SPLIT_SEEDS),
            )
        selection = build_parser().parse_args(
            [
                "record-selection",
                "--comparison-archive", "artifacts/comparisons/model/run",
                "--config-id", "candidate",
                "--purpose", "deployment",
                "--rationale", "human analysis",
                "--output", "artifacts/selections/deployment.json",
            ]
        )
        self.assertEqual(selection.command, "record-selection")
        refit = build_parser().parse_args(
            [
                "final-refit",
                "--run-directory", "artifacts/runs/candidate",
                "--selection-record", "artifacts/selections/deployment.json",
                "--selection-record-sha256", "0" * 64,
                "--comparison-archive", "artifacts/comparisons/model/run",
                "--config", "reference_static_feature_vector_v2",
            ]
        )
        self.assertEqual(refit.command, "final-refit")
        self.assertFalse(refit.confirm_scientific_execution)
        release = build_parser().parse_args(
            [
                "winner-release-preflight",
                "--bundle-directory", "artifacts/final/bundle",
                "--bundle-manifest-sha256", "3" * 64,
                "--final-refit-attestation",
                "artifacts/final/bundle/final_refit_attestation.json",
                "--final-refit-attestation-sha256", "4" * 64,
                "--onnx-model", "artifacts/final/model.onnx",
                "--certificate", "artifacts/final/onnx_certificate.json",
                "--certificate-sha256", "1" * 64,
                "--selection-record", "artifacts/selections/deployment.json",
                "--selection-record-sha256", "2" * 64,
                "--config", "reference_static_feature_vector_v2",
            ]
        )
        self.assertEqual(release.command, "winner-release-preflight")
        producer = build_parser().parse_args(
            [
                "produce-onnx-winner-certificate",
                "--bundle-directory", "artifacts/final/bundle",
                "--bundle-manifest-sha256", "3" * 64,
                "--final-refit-attestation",
                "artifacts/final/bundle/final_refit_attestation.json",
                "--final-refit-attestation-sha256", "4" * 64,
                "--selection-record", "artifacts/selections/deployment.json",
                "--selection-record-sha256", "2" * 64,
                "--config", "reference_static_feature_vector_v2",
                "--output-dir", "artifacts/final/onnx_gate",
            ]
        )
        self.assertEqual(producer.command, "produce-onnx-winner-certificate")
        self.assertFalse(producer.confirm_onnx_execution)
        for removed in (
            "opset_version", "absolute_tolerance", "relative_tolerance",
        ):
            self.assertFalse(hasattr(producer, removed))
        materialize = build_parser().parse_args(
            [
                "materialize-ablation-config",
                "--base-config", "reference_static_line_a_v2",
                "--family", "fixed_kernel_samples",
                "--profile-id", "compactcnn1d__fs_100",
                "--output", "configs/generated/compact_fs100_v2.yaml",
            ]
        )
        self.assertEqual(materialize.command, "materialize-ablation-config")
        self.assertEqual(materialize.family, "fixed_kernel_samples")

    def test_comparison_rejects_same_roster_with_different_source_authority(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as raw:
            root = Path(raw)
            reference = self._run_fixture(
                root,
                config_id="source_reference",
                machine_id="logistic_regression",
                probability_shift=0.0,
            )
            candidate = self._run_fixture(
                root,
                config_id="source_candidate",
                machine_id="rbf_svm",
                probability_shift=0.0,
                source_snapshot_hash="8" * 64,
            )
            with self.assertRaisesRegex(
                ValueError,
                "comparison_undeclared_authority_difference.*source_snapshot",
            ):
                build_comparison_archive_from_run_directories(
                    {"source_reference": reference, "source_candidate": candidate},
                    reference_config_id="source_reference",
                    comparison_family="invalid_cross_source_fixture",
                    comparison_id="must_fail",
                    run_id="run_001",
                    output_root=root / "archives",
                    n_bootstrap_resamples=4,
                    n_permutation_resamples=4,
                )

    def test_trusted_reader_rebuilds_operational_metrics_from_cells(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as raw:
            run = self._run_fixture(
                Path(raw),
                config_id="operational_tamper",
                machine_id="logistic_regression",
                probability_shift=0.0,
            )
            path = run / "config_metrics_v2.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["config_metrics"]["parameter_count"] = 999
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            self._reindex(run)
            with self.assertRaisesRegex(
                ValueError,
                "comparison_run_operational_metric_drift.*parameter_count",
            ):
                _read_trusted_comparison_run(
                    "operational_tamper",
                    run,
                    n_bootstrap_resamples=4,
                    bootstrap_seed=42,
                )

    def test_trusted_reader_rejects_cell_root_oof_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as raw:
            run = self._run_fixture(
                Path(raw),
                config_id="cell_root_tamper",
                machine_id="logistic_regression",
                probability_shift=0.0,
            )
            cell = run / "repeat_00_fold_00"
            path = cell / "oof_subject_predictions.parquet"
            rows = list(read_oof_parquet(path))
            rows[0] = replace(rows[0], model_hash="9" * 64)
            OofWriter().write(rows, path)
            self._reindex(cell)
            self._reindex(run)
            with self.assertRaisesRegex(
                ValueError,
                "comparison_run_cell_root_subject_oof_drift",
            ):
                _read_trusted_comparison_run(
                    "cell_root_tamper",
                    run,
                    n_bootstrap_resamples=4,
                    bootstrap_seed=42,
                )


if __name__ == "__main__":
    unittest.main()

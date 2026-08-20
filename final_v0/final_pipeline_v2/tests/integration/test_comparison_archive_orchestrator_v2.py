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
    _write_empty_oof,
    build_comparison_archive_from_run_directories,
    execute_final_refit_from_verified_artifacts,
    final_refit_preflight_from_verified_artifacts,
    verify_manual_selection_record,
    write_manual_selection_record,
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
_MEMBER_SEEDS = (50042, 60042, 70042, 80042, 90042)
ROOT = Path(__file__).resolve().parents[2]


def _write_artifact_index(directory: Path) -> None:
    """Compatibility no-op: ordinary run directories no longer use an index gate."""

    del directory


class ComparisonArchiveOrchestratorTests(unittest.TestCase):
    def _run_fixture(
        self,
        root: Path,
        *,
        config_id: str,
        machine_id: str,
        probability_shift: float,
        source_snapshot_hash: str = "6" * 64,
        seed_policy: str | None = None,
        member_seeds: tuple[int, ...] = _MEMBER_SEEDS,
        fixed_training_seed: int = 123,
        evaluation_statistics: dict[str, int] | None = None,
    ) -> Path:
        run_root = root / config_id
        run_root.mkdir()
        config_hash = hashlib.sha256(config_id.encode("utf-8")).hexdigest()
        oof_rows: list[OofPredictionRow] = []
        member_rows: list[OofPredictionRow] = []
        ensemble_base = {
            "inception_full_five_member_ensemble": "inception_full",
            "inception_matrix_five_member_ensemble": "inception_matrix",
        }.get(machine_id)
        single_seed_policy = seed_policy or "outer_cv_repeat_seed_equals_split_seed"
        ensemble_seed_policy = seed_policy or "cv_fixed_five_member_seed_roster"
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
                ensemble_seed_policy
                if ensemble_base
                else single_seed_policy
            ),
            "random_seeds": list(
                member_seeds
                if ensemble_base
                else (
                    (50042,)
                    if single_seed_policy
                    == "cv_fixed_member0_seed_50042_comparator"
                    else (42,)
                )
            ),
            "fold_hash": "2" * 64,
            "aggregation": {"line": "line_a_equal_files"},
            "calibration": {"fit_scope": "outer_training_only"},
        }
        for repeat, split_seed in enumerate(_SPLIT_SEEDS):
            single_training_seed = (
                50042
                if single_seed_policy == "cv_fixed_member0_seed_50042_comparator"
                else (
                    fixed_training_seed
                    if single_seed_policy
                    in {"fixed", "fixed_explicit", "final_refit_single_seed_42"}
                    else split_seed
                )
            )
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
                    cell_frozen_provenance["random_seeds"] = [
                        single_training_seed
                    ]
                summary = {
                    "status": "passed",
                    "repeat_index": repeat,
                    "fold_index": fold,
                    "split_seed": split_seed,
                    "training_seed": None if ensemble_base else single_training_seed,
                    "member_training_seeds": (
                        list(member_seeds) if ensemble_base else []
                    ),
                    "seed_policy": (
                        ensemble_seed_policy
                        if ensemble_base
                        else single_seed_policy
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
                if evaluation_statistics is not None:
                    summary["evaluation_policy"] = {
                        "statistics": dict(evaluation_statistics)
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
                        training_seed=(
                            None if ensemble_base else single_training_seed
                        ),
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
            n_bootstrap_resamples=(
                12
                if evaluation_statistics is None
                else int(evaluation_statistics["bootstrap_replicates"])
            ),
            bootstrap_seed=(
                42
                if evaluation_statistics is None
                else int(evaluation_statistics["seed"])
            ),
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
                "training_seeds": (
                    list(member_seeds)
                    if ensemble_base
                    else (
                        [50042]
                        if single_seed_policy
                        == "cv_fixed_member0_seed_50042_comparator"
                        else (
                            [fixed_training_seed]
                            if single_seed_policy
                            in {"fixed", "fixed_explicit", "final_refit_single_seed_42"}
                            else list(_SPLIT_SEEDS)
                        )
                    )
                ),
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
                },
            },
        )
        _write_artifact_index(run_root)
        return run_root

    @staticmethod
    def _reindex(directory: Path) -> None:
        index = directory / "artifact_index.json"
        if index.exists():
            index.unlink()
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
            preflight_parameters = set(
                inspect.signature(
                    final_refit_preflight_from_verified_artifacts
                ).parameters
            )
            self.assertEqual(
                preflight_parameters,
                {
                    "run_directory", "selection_record", "comparison_archive",
                    "config_path",
                },
            )
            refit_parameters = inspect.signature(
                execute_final_refit_from_verified_artifacts
            ).parameters
            self.assertEqual(
                set(refit_parameters),
                {
                    "run_directory", "selection_record", "comparison_archive",
                    "config_path", "bundle_directory",
                },
            )
            for forbidden in (
                "trainer", "full_dataset", "binding", "model_factory",
                "estimator", "metadata", "parity_atol",
            ):
                self.assertNotIn(forbidden, refit_parameters)

    def test_archive_inherits_hash_bound_statistics_policy(self) -> None:
        statistics = {
            "bootstrap_replicates": 7,
            "paired_permutation_replicates": 9,
            "seed": 13,
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as raw:
            root = Path(raw)
            reference = self._run_fixture(
                root,
                config_id="policy_reference",
                machine_id="logistic_regression",
                probability_shift=0.0,
                evaluation_statistics=statistics,
            )
            candidate = self._run_fixture(
                root,
                config_id="policy_candidate",
                machine_id="rbf_svm",
                probability_shift=0.05,
                evaluation_statistics=statistics,
            )
            result = build_comparison_archive_from_run_directories(
                {"policy_reference": reference, "policy_candidate": candidate},
                reference_config_id="policy_reference",
                comparison_family="configured_statistics",
                comparison_id="configured_statistics",
                run_id="run_001",
                output_root=root / "archives",
            )
            manifest = json.loads(
                (Path(result["output_directory"]) / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["bootstrap_policy"]["resamples"], 7)
            self.assertEqual(manifest["paired_permutation_policy"]["resamples"], 9)
            self.assertEqual(manifest["bootstrap_policy"]["seed"], 13)

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
        self.assertIsNone(parsed.bootstrap_resamples)
        self.assertIsNone(parsed.permutation_resamples)
        self.assertIsNone(parsed.statistics_seed)
        self.assertEqual(
            parsed.allowed_authority_difference,
            ["frozen.architecture_parameters"],
        )

    def test_trusted_reader_accepts_fixed_ensemble_outer_cv_seed_matrix(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as raw:
            run = self._run_fixture(
                Path(raw),
                config_id="ensemble_fixture",
                machine_id="inception_full_five_member_ensemble",
                probability_shift=0.0,
            )
            _read_trusted_comparison_run(
                "ensemble_fixture",
                run,
                n_bootstrap_resamples=12,
                bootstrap_seed=42,
            )
            averages = read_oof_parquet(run / "oof_subject_predictions.parquet")
            members = read_oof_parquet(run / "oof_member_predictions.parquet")
            self.assertEqual({row.training_seed for row in averages}, {None})
            self.assertEqual(
                {row.training_seed for row in members},
                set(_MEMBER_SEEDS),
            )

    def test_trusted_reader_accepts_configurable_fixed_seed_and_n_member_ensemble(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as raw:
            root = Path(raw)
            fixed = self._run_fixture(
                root,
                config_id="fixed_seed_fixture",
                machine_id="logistic_regression",
                probability_shift=0.0,
                seed_policy="fixed",
                fixed_training_seed=77,
            )
            fixed_loaded = _read_trusted_comparison_run(
                "fixed_seed_fixture",
                fixed,
                n_bootstrap_resamples=12,
                bootstrap_seed=42,
            )
            self.assertEqual(fixed_loaded["seed_policy"], "fixed")

            ensemble = self._run_fixture(
                root,
                config_id="two_member_ensemble_fixture",
                machine_id="inception_full_five_member_ensemble",
                probability_shift=0.0,
                seed_policy="member_roster",
                member_seeds=(17, 29),
            )
            ensemble_loaded = _read_trusted_comparison_run(
                "two_member_ensemble_fixture",
                ensemble,
                n_bootstrap_resamples=12,
                bootstrap_seed=42,
            )
            self.assertEqual(ensemble_loaded["seed_policy"], "member_roster")
            self.assertEqual(
                len(read_oof_parquet(ensemble / "oof_member_predictions.parquet")),
                290,
            )

    def test_trusted_reader_scopes_member0_policy_without_changing_other_singles(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as raw:
            root = Path(raw)
            ordinary = self._run_fixture(
                root,
                config_id="ordinary_inception_fixture",
                machine_id="inception_full",
                probability_shift=0.0,
            )
            comparator = self._run_fixture(
                root,
                config_id="member0_inception_fixture",
                machine_id="inception_full",
                probability_shift=0.0,
                seed_policy="cv_fixed_member0_seed_50042_comparator",
            )
            ordinary_loaded = _read_trusted_comparison_run(
                "ordinary_inception_fixture",
                ordinary,
                n_bootstrap_resamples=12,
                bootstrap_seed=42,
            )
            comparator_loaded = _read_trusted_comparison_run(
                "member0_inception_fixture",
                comparator,
                n_bootstrap_resamples=12,
                bootstrap_seed=42,
            )
            self.assertEqual(
                ordinary_loaded["seed_policy"],
                "outer_cv_repeat_seed_equals_split_seed",
            )
            self.assertEqual(
                comparator_loaded["seed_policy"],
                "cv_fixed_member0_seed_50042_comparator",
            )
            ordinary_rows = read_oof_parquet(
                ordinary / "oof_subject_predictions.parquet"
            )
            comparator_rows = read_oof_parquet(
                comparator / "oof_subject_predictions.parquet"
            )
            self.assertEqual(
                {row.training_seed for row in ordinary_rows},
                set(_SPLIT_SEEDS),
            )
            self.assertEqual(
                {row.training_seed for row in comparator_rows},
                {50042},
            )

    def test_ensemble_factor_archive_requires_matching_member0_reference(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as raw:
            root = Path(raw)
            ensemble_config_id = (
                "comparison_inception_full_five_member_ensemble_line_b_v2"
            )
            comparator_config_id = (
                "comparison_inception_full_member0_comparator_line_b_v2"
            )
            unrelated = self._run_fixture(
                root,
                config_id="unrelated_reference",
                machine_id="logistic_regression",
                probability_shift=0.0,
            )
            ensemble = self._run_fixture(
                root,
                config_id=ensemble_config_id,
                machine_id="inception_full_five_member_ensemble",
                probability_shift=0.0,
            )
            with self.assertRaisesRegex(
                ValueError,
                "ensemble_factor_comparison_requires_matching_member0_reference",
            ):
                build_comparison_archive_from_run_directories(
                    {
                        "unrelated_reference": unrelated,
                        ensemble_config_id: ensemble,
                    },
                    reference_config_id="unrelated_reference",
                    comparison_family="ensemble_factor",
                    comparison_id="requires_matched_reference",
                    run_id="run_001",
                    output_root=root / "archives",
                    n_bootstrap_resamples=12,
                    n_permutation_resamples=16,
                    statistics_seed=42,
                )
            ordinary_inception_config_id = "formal_inception_full_line_b_v2"
            ordinary_inception = self._run_fixture(
                root,
                config_id=ordinary_inception_config_id,
                machine_id="inception_full",
                probability_shift=0.0,
            )
            with self.assertRaisesRegex(
                ValueError,
                "ensemble_factor_comparison_requires_matching_member0_reference",
            ):
                build_comparison_archive_from_run_directories(
                    {
                        ordinary_inception_config_id: ordinary_inception,
                        ensemble_config_id: ensemble,
                    },
                    reference_config_id=ordinary_inception_config_id,
                    comparison_family="ensemble_factor",
                    comparison_id="reject_ordinary_inception_as_member0",
                    run_id="run_001",
                    output_root=root / "ordinary_archives",
                    n_bootstrap_resamples=12,
                    n_permutation_resamples=16,
                    statistics_seed=42,
                )
            comparator = self._run_fixture(
                root,
                config_id=comparator_config_id,
                machine_id="inception_full",
                probability_shift=0.0,
                seed_policy="cv_fixed_member0_seed_50042_comparator",
            )
            matrix_comparator_config_id = (
                "comparison_inception_matrix_member0_comparator_line_b_v2"
            )
            matrix_comparator = self._run_fixture(
                root,
                config_id=matrix_comparator_config_id,
                machine_id="inception_matrix",
                probability_shift=0.0,
                seed_policy="cv_fixed_member0_seed_50042_comparator",
            )
            with self.assertRaisesRegex(
                ValueError,
                "ensemble_factor_comparison_requires_matching_member0_reference",
            ):
                build_comparison_archive_from_run_directories(
                    {
                        matrix_comparator_config_id: matrix_comparator,
                        ensemble_config_id: ensemble,
                    },
                    reference_config_id=matrix_comparator_config_id,
                    comparison_family="ensemble_factor",
                    comparison_id="reject_matrix_reference_for_raw_ensemble",
                    run_id="run_001",
                    output_root=root / "mismatched_archives",
                    n_bootstrap_resamples=12,
                    n_permutation_resamples=16,
                    statistics_seed=42,
                )
            matched = build_comparison_archive_from_run_directories(
                {
                    comparator_config_id: comparator,
                    ensemble_config_id: ensemble,
                },
                reference_config_id=comparator_config_id,
                comparison_family="ensemble_factor",
                comparison_id="matched_member0_reference",
                run_id="run_001",
                output_root=root / "matched_archives",
                n_bootstrap_resamples=12,
                n_permutation_resamples=16,
                statistics_seed=42,
                allowed_authority_differences=(
                    "oof.training_seed",
                    "oof.prediction_kind",
                    "oof.member_training_seeds",
                    "oof.ensemble_base_model_id",
                    "frozen.seed_policy",
                    "frozen.random_seeds",
                ),
            )
            self.assertEqual(matched["status"], "passed")
            self.assertEqual(
                matched["reference_config_id"],
                comparator_config_id,
            )

    def test_matrix_ensemble_factor_accepts_only_explicit_matrix_member0(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as raw:
            root = Path(raw)
            comparator_config_id = (
                "comparison_inception_matrix_member0_comparator_line_b_v2"
            )
            ensemble_config_id = (
                "comparison_inception_matrix_five_member_ensemble_line_b_v2"
            )
            comparator = self._run_fixture(
                root,
                config_id=comparator_config_id,
                machine_id="inception_matrix",
                probability_shift=0.0,
                seed_policy="cv_fixed_member0_seed_50042_comparator",
            )
            ensemble = self._run_fixture(
                root,
                config_id=ensemble_config_id,
                machine_id="inception_matrix_five_member_ensemble",
                probability_shift=0.0,
            )
            result = build_comparison_archive_from_run_directories(
                {
                    comparator_config_id: comparator,
                    ensemble_config_id: ensemble,
                },
                reference_config_id=comparator_config_id,
                comparison_family="ensemble_factor",
                comparison_id="matched_matrix_member0_reference",
                run_id="run_001",
                output_root=root / "archives",
                n_bootstrap_resamples=12,
                n_permutation_resamples=16,
                statistics_seed=42,
                allowed_authority_differences=(
                    "oof.training_seed",
                    "oof.prediction_kind",
                    "oof.member_training_seeds",
                    "oof.ensemble_base_model_id",
                    "frozen.seed_policy",
                    "frozen.random_seeds",
                ),
            )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["reference_config_id"], comparator_config_id)

    def test_selection_refit_and_materialization_cli_contracts(self) -> None:
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
                "--comparison-archive", "artifacts/comparisons/model/run",
                "--config", "reference_static_feature_vector_v2",
                "--bundle-directory", "artifacts/bundles/deployment",
            ]
        )
        self.assertEqual(refit.command, "final-refit")
        self.assertEqual(refit.bundle_directory, "artifacts/bundles/deployment")
        materialize = build_parser().parse_args(
            [
                "materialize-ablation-config",
                "--base-config", "reference_static_role_aware_v2",
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

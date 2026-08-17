"""Tiny representation-dispatch contract tests; no CV, ablation, or real training."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import ppg_frailty.experiment as experiment
import ppg_frailty.pipeline as pipeline
from ppg_frailty.contracts import EngineeringFeatureSequence, SignalRoute
from ppg_frailty.features import (
    EngineeringExtraction,
    build_feature_vector,
    default_registry,
)
from ppg_frailty.representations import RawWindows
from ppg_frailty.training import RawWindowDataset, SampleIdentity


TRAIN_IDS = ("train_a", "train_b")
OOF_IDS = ("heldout",)


def _state(participant: str, value: float, *, role: str = "R3") -> experiment._RuntimeRecord:
    row = SimpleNamespace(
        participant_id=participant,
        record_id=f"{participant}_{role}",
        role=role,
        class_id={"train_a": 0, "train_b": 1, "heldout": 2}.get(participant, 0),
    )
    state = experiment._RuntimeRecord(
        row=row,
        retained=True,
        route=SignalRoute.DIRECT,
        intended_route=SignalRoute.DIRECT,
        route_status="retained_direct_quality_off",
        artifact_name="identity",
        artifact_version="identity_v1",
    )
    state.vector = build_feature_vector(
        {"prv.hr_mean_bpm": value},
        feature_validity={"prv.hr_mean_bpm": True},
        provenance={"route": SignalRoute.DIRECT.value, "record_id": row.record_id},
    )
    sequence = EngineeringFeatureSequence(
        values=np.asarray(((value, value + 1.0), (value + 0.5, value + 1.5))),
        start_samples=np.asarray((0, 2000), dtype=np.int64),
        valid_row_mask=np.ones(2, dtype=bool),
        channel_schema=("fixture_a", "fixture_b"),
        schema_version="engineering_fixture_v1",
    )
    state.engineering = EngineeringExtraction(
        sequence=sequence,
        value_validity=np.ones((2, 2), dtype=bool),
        route=SignalRoute.DIRECT,
        reasons=(),
    )
    raw = np.zeros((2, 8, 16), dtype=np.float32)
    raw[:, 0, :] = np.arange(16, dtype=np.float32)
    raw[:, 1, :] = value
    for channel in range(6):
        raw[:, channel + 2, :] = value + channel + np.linspace(0.0, 1.0, 16)
    state.raw_windows = RawWindows(
        values=raw,
        valid_mask=np.ones((2, 16), dtype=bool),
        start_samples=np.asarray((0, 1000), dtype=np.int64),
        candidate_count=2,
        dropped_invalid_count=0,
    )
    return state


def _states() -> list[experiment._RuntimeRecord]:
    return [_state("train_a", 1.0), _state("train_b", 3.0), _state("heldout", 50.0)]


class RepresentationDispatchTest(unittest.TestCase):
    def test_progress_events_expose_refresh_consumer_fields(self) -> None:
        events: list[dict[str, object]] = []

        experiment._notify_progress(
            events.append,
            "cell_start",
            current_cell=3,
            total_cells=25,
        )

        self.assertEqual(
            events,
            [
                {
                    "stage": "cell_start",
                    "event": "cell_start",
                    "current_cell": 3,
                    "total_cells": 25,
                    "current": 3,
                    "total": 25,
                }
            ],
        )

    def test_preprocess_fits_one_full_role_b_imu_calibration_per_participant(self) -> None:
        rows = [
            SimpleNamespace(
                participant_id="P01",
                record_id="P01_B",
                role="B",
                qc_status="pass",
                duration_s=300.0,
                fs=400.0,
                n_samples=120_000,
            ),
            SimpleNamespace(
                participant_id="P01",
                record_id="P01_R1",
                role="R1",
                qc_status="pass",
                duration_s=300.0,
                fs=400.0,
                n_samples=120_000,
            ),
        ]
        # The reduced classifier cap may retain only R1. Calibration must still
        # resolve B from the full manifest rather than the capped state list.
        states = [experiment._RuntimeRecord(row=rows[1])]
        loader_calls: list[tuple[str, int | None]] = []

        def loader(row: object, maximum: int | None) -> dict[str, object]:
            loader_calls.append((str(row.record_id), maximum))
            return {
                "record_id": str(row.record_id),
                "fs_hz": 400.0,
                "ppg": np.ones((400, 2), dtype=np.float64),
                "acc": np.ones((400, 3), dtype=np.float64),
                "gyro": np.zeros((400, 3), dtype=np.float64),
                "acc_unit": "g",
                "gyro_unit": "deg/s",
            }

        calibration = SimpleNamespace(
            schema_version="ppg_frailty.motion_imu_calibration.v2",
            participant_id="P01",
            file_id="P01_B",
            source_role="B",
            artifact_sha256="a" * 64,
        )
        seen_payloads: list[dict[str, object]] = []

        def build_signal_views(
            payload: dict[str, object],
            _config: dict[str, object],
        ) -> object:
            seen_payloads.append(payload)
            self.assertIs(payload["imu_calibration"], calibration)
            self.assertEqual(payload["participant_id"], "P01")
            return SimpleNamespace()

        runtime = {
            "np": np,
            "build_signal_views": build_signal_views,
            "roll_pitch_ekf_config_from_resolved": lambda value: value,
            "fit_motion_imu_calibration": lambda *args, **kwargs: calibration,
        }
        config = SimpleNamespace()
        config.section = lambda name: {
            "imu": {"gravity_method": "calibrated_roll_pitch_ekf"}
        }
        config.to_dict = lambda: {"signal": config.section("signal")}

        with patch.object(experiment, "_runtime_imports", return_value=runtime):
            experiment._preprocess_records(
                states,
                config,
                maximum_seconds=1.0,
                loader=loader,
                calibration_rows=rows,
            )

        self.assertEqual(loader_calls.count(("P01_B", None)), 1)
        self.assertEqual(len(seen_payloads), 1)
        self.assertTrue(all(state.views is not None for state in states))
        self.assertTrue(
            all(
                state.diagnostic_components["imu_calibration"][
                    "fallback_used"
                ]
                is False
                for state in states
            )
        )

    def test_one_cell_root_writer_is_descriptive_not_25_cell_inference(self) -> None:
        predicted = _state("heldout", 2.0, role="R3")
        dataset = experiment._materialize_representation_dataset(
            [predicted],
            ("heldout",),
            "raw",
        )
        common = {
            "repeat": 0,
            "fold": 0,
            "split_seed": 42,
            "training_seed": 42,
            "config_hash": "c" * 64,
            "manifest_hash": "m" * 64,
            "fold_hash": "f" * 64,
            "preprocessing_hash": "p" * 64,
            "feature_hash": "e" * 64,
            "model_hash": "d" * 64,
            "representation_mode": "raw",
            "class_order": (0, 1, 2),
            "code_commit": "not_git_bound",
            "data_schema_id": "data_v2",
            "feature_schema_id": "raw8",
            "model_version": "model_v2",
            "aggregation_rule": "line_b_equal_role_families",
            "environment_hash": "n" * 64,
            "manifest_version": "manifest_v2",
            "fold_registry_version": "folds_v2",
            "source_snapshot_hash": "not_source_hash_bound",
        }
        probabilities = np.asarray(((0.2, 0.3, 0.5), (0.1, 0.2, 0.7)))
        window_rows, file_rows, role_rows, subject_rows = experiment._make_oof(
            [predicted],
            ("heldout",),
            dataset.identities,
            probabilities,
            common,
            balance_line="line_b_equal_role_families",
        )
        summary = {
            "status": "passed",
            "repeat_index": 0,
            "fold_index": 0,
            "class_order": [0, 1, 2],
            "quality_mode": "off",
            "quality_diagnostics": [],
            "scientific_scope": "selected_outer_cells_descriptive",
            "training_history": [{"epoch": 1, "train_loss": 0.75}],
            "learning_curve_contract": {
                "status": "outer_train_loss_only_fixed_epoch",
                "outer_heldout_used_for_epoch_selection_or_curve": False,
            },
            "metrics": {
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            },
        }
        cell = experiment._CellResult(
            summary=summary,
            file_rows=file_rows,
            subject_rows=subject_rows,
            window_rows=window_rows,
            role_rows=role_rows,
        )
        result = experiment.ExperimentResult(
            status="passed",
            scientific_scope="selected_outer_cells_descriptive",
            config_id="fixture_v2",
            config_hash="c" * 64,
            repeat_indices=(0,),
            fold_indices=(0,),
            output_dir=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment._write_full_root_artifacts(root, (cell,), result)
            metrics = json.loads(
                (root / "config_metrics_v2.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["status"], "partial_descriptive_only")
            self.assertEqual(metrics["successful_cell_count"], 1)
            self.assertFalse(metrics["formal_comparison_eligible"])
            self.assertTrue((root / "oof_role_predictions.parquet").is_file())
            cell_directory = root / "cell"
            experiment._write_cell_artifacts(cell_directory, cell)
            history = json.loads(
                (cell_directory / "training_history.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                history["rows"],
                [{"repeat": 0, "fold": 0, "epoch": 1, "train_loss": 0.75}],
            )
            self.assertFalse(
                history["learning_curve_contract"][
                    "outer_heldout_used_for_epoch_selection_or_curve"
                ]
            )

    def test_absolute_output_directory_is_allowed_but_never_overwritten(self) -> None:
        paths = pipeline.PipelinePaths.discover()
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "external_archive"
            resolved = experiment._resolve_output_directory(
                paths,
                requested.resolve(),
                "unused",
            )
            self.assertEqual(resolved, requested.resolve())
            requested.mkdir()
            with self.assertRaises(FileExistsError):
                experiment._resolve_output_directory(
                    paths,
                    requested.resolve(),
                    "unused",
                )

    def test_extract_vector_keeps_coverage_out_of_predictors(self) -> None:
        state = experiment._RuntimeRecord(
            row=SimpleNamespace(record_id="f1", role="B"),
            views=SimpleNamespace(
                x_filter=np.zeros((400, 2), dtype=np.float64),
                x_native=np.zeros((400, 2), dtype=np.float64),
            ),
            retained=True,
            route=SignalRoute.DIRECT,
        )
        registry = default_registry()
        api = {
            "SignalRoute": SignalRoute,
            "QualityState": SimpleNamespace(PASS="pass"),
            "detect_pulses": lambda *_a, **_k: object(),
            "compute_prv": lambda *_a, **_k: SimpleNamespace(
                values={"coverage": 0.75, "hr_mean_bpm": 60.0},
                validity={"coverage": True, "hr_mean_bpm": True},
            ),
            "canonicalize_role_family": lambda value: value,
            "WindowPlan": lambda **_k: object(),
            "extract_engineering_features": lambda *_a, **_k: object(),
            "summarize_engineering": lambda *_a, **_k: ({}, {}),
            "extract_morphology": lambda *_a, **_k: SimpleNamespace(
                aggregate_values={},
                aggregate_validity={},
            ),
            "extract_dual_optical": lambda *_a, **_k: SimpleNamespace(
                aggregate_values={},
                aggregate_validity={},
            ),
            "default_registry": lambda: registry,
            "build_feature_vector": build_feature_vector,
        }
        report = SimpleNamespace(
            window_profiles={"engineering": {}},
        )
        with patch(
            "ppg_frailty.experiment._runtime_imports",
            return_value=api,
        ):
            experiment._extract_vector(state, report)
        self.assertTrue(state.retained)
        self.assertNotIn("prv.coverage", state.vector.feature_names)
        self.assertEqual(
            state.diagnostic_components["non_predictor_features"][
                "prv.coverage"
            ],
            {"value": 0.75, "valid": True},
        )
        availability = state.diagnostic_components["predictor_availability"]
        self.assertEqual(availability["predictor_count"], len(registry.names))
        self.assertEqual(availability["available_predictor_count"], 1)

    def test_all_frailty_raw_models_share_one_canonical_8ch_binding(self) -> None:
        identity = SampleIdentity(
            participant_id="p1",
            file_id="f1",
            role="B",
            label=0,
            signal_route=SignalRoute.DIRECT.value,
        )
        values = np.stack(
            [
                np.full((4,), float(channel), dtype=np.float32)
                for channel in range(8)
            ]
        )[None, :, :]
        dataset = RawWindowDataset(
            values,
            (identity,),
            np.ones((1, 4), dtype=bool),
        )
        unchanged, compact_binding = (
            experiment._bind_raw_dataset_for_model(
                dataset,
                "compact_cnn",
                declared_channel_order=experiment._CANONICAL_RAW_CHANNEL_SCHEMA,
            )
        )
        self.assertIs(unchanged, dataset)
        self.assertEqual(unchanged.values.shape, (1, 8, 4))
        self.assertEqual(
            experiment._model_input_spec(unchanged, "raw").channel_schema,
            experiment._CANONICAL_RAW_CHANNEL_SCHEMA,
        )
        self.assertEqual(
            compact_binding["status"],
            "canonical_frailty_raw_8_identity",
        )
        self.assertFalse(compact_binding["derived_motion_channels_present"])

        shapeformer, shapeformer_binding = experiment._bind_raw_dataset_for_model(
            dataset,
            "shapeformer_channel_specific_osd",
            declared_channel_order=experiment._CANONICAL_RAW_CHANNEL_SCHEMA,
        )
        self.assertIs(shapeformer, dataset)
        np.testing.assert_array_equal(shapeformer.values, values)
        self.assertEqual(
            experiment._model_input_spec(shapeformer, "raw").channel_schema,
            experiment._CANONICAL_RAW_CHANNEL_SCHEMA,
        )
        self.assertFalse(shapeformer_binding["silent_channel_slicing"])

        with self.assertRaisesRegex(
            RuntimeError,
            "frailty_model_input_channel_order_must_equal_canonical_8ch_schema",
        ):
            experiment._bind_raw_dataset_for_model(
                dataset,
                "shapeformer_channel_specific_osd",
                declared_channel_order=("RED", "IR"),
            )

        invalid = RawWindowDataset(
            np.zeros((1, 11, 4), dtype=np.float32),
            (identity,),
            np.ones((1, 4), dtype=bool),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "canonical_frailty_raw_tensor_must_be_8_channels",
        ):
            experiment._bind_raw_dataset_for_model(
                invalid,
                "compact_cnn",
                declared_channel_order=experiment._CANONICAL_RAW_CHANNEL_SCHEMA,
            )

    def test_fold_artifacts_and_all_typed_datasets_materialize(self) -> None:
        raw_states = _states()
        raw_provenance = experiment._fit_representation_artifacts(
            raw_states, "raw", TRAIN_IDS, OOF_IDS
        )
        self.assertEqual(
            tuple(raw_provenance["raw_imu"]["fitted_on_participant_ids"]), TRAIN_IDS
        )
        raw_dataset = experiment._materialize_representation_dataset(
            raw_states, OOF_IDS, "raw"
        )
        self.assertEqual(raw_dataset.representation_mode, "raw")
        self.assertEqual(len(raw_dataset), 2)
        self.assertEqual({item.role for item in raw_dataset.identities}, {"R"})

        matrix_states = _states()
        matrix_provenance = experiment._fit_representation_artifacts(
            matrix_states, "feature_matrix", TRAIN_IDS, OOF_IDS
        )
        self.assertEqual(
            tuple(matrix_provenance["feature_vector"]["fitted_on_participant_ids"]),
            TRAIN_IDS,
        )
        self.assertEqual(
            tuple(matrix_provenance["engineering"]["fitted_on_participant_ids"]),
            TRAIN_IDS,
        )
        matrix_dataset = experiment._materialize_representation_dataset(
            matrix_states, OOF_IDS, "feature_matrix"
        )
        self.assertEqual(matrix_dataset.representation_mode, "feature_matrix")
        self.assertEqual(matrix_dataset.values.shape[0], 1)
        self.assertEqual(matrix_dataset.values.shape[2], 32)

        fusion_states = _states()
        fusion_provenance = experiment._fit_representation_artifacts(
            fusion_states, "fusion", TRAIN_IDS, OOF_IDS
        )
        self.assertEqual(
            tuple(fusion_provenance["raw_imu"]["fitted_on_participant_ids"]), TRAIN_IDS
        )
        self.assertEqual(
            tuple(fusion_provenance["feature_vector"]["fitted_on_participant_ids"]),
            TRAIN_IDS,
        )
        fusion_dataset = experiment._materialize_representation_dataset(
            fusion_states, OOF_IDS, "fusion"
        )
        self.assertEqual(fusion_dataset.representation_mode, "fusion")
        self.assertTrue(np.isfinite(fusion_dataset.file_features).all())

        vector_dataset = experiment._materialize_representation_dataset(
            _states(), OOF_IDS, "feature_vector"
        )
        self.assertEqual(vector_dataset.representation_mode, "feature_vector")

    def test_raw_window_oof_aggregates_and_keeps_no_window_drop(self) -> None:
        predicted = _state("heldout", 5.0, role="R2")
        dropped = _state("dropped", 7.0, role="B")
        dropped.row.class_id = 1
        dropped.retained = False
        dropped.raw_windows = None
        dropped.reason = "raw_windows_failed:fixture"
        dropped.route_status = "dropped_raw_window_failure"
        dataset = experiment._materialize_representation_dataset(
            [predicted], ("heldout",), "raw"
        )
        common = {
            "repeat": 0,
            "fold": 0,
            "split_seed": 42,
            "training_seed": 42,
            "config_hash": "config",
            "manifest_hash": "manifest",
            "fold_hash": "fold",
            "preprocessing_hash": "preprocessing",
            "feature_hash": "feature",
            "model_hash": "model",
            "representation_mode": "raw",
            "class_order": (0, 1, 2),
            "code_commit": "commit",
            "data_schema_id": "data",
            "feature_schema_id": "raw8",
            "model_version": "model_v1",
            "aggregation_rule": "line_a_equal_files",
            "environment_hash": "environment",
            "manifest_version": "manifest_v1",
            "fold_registry_version": "folds_v1",
            "source_snapshot_hash": "source_snapshot",
        }
        probabilities = np.asarray(((0.8, 0.1, 0.1), (0.2, 0.3, 0.5)))
        window_rows, file_rows, role_rows, subject_rows = experiment._make_oof(
            [predicted, dropped],
            ("heldout", "dropped"),
            dataset.identities,
            probabilities,
            common,
            balance_line="line_a_equal_files",
        )
        self.assertEqual(role_rows, ())
        self.assertEqual(len(window_rows), 2)
        retained_file = next(row for row in file_rows if row.participant_id == "heldout")
        np.testing.assert_allclose(retained_file.probabilities, (0.5, 0.2, 0.3))
        dropped_file = next(row for row in file_rows if row.participant_id == "dropped")
        self.assertFalse(dropped_file.retained)
        dropped_subject = next(row for row in subject_rows if row.participant_id == "dropped")
        self.assertFalse(dropped_subject.retained)

        role_common = dict(
            common,
            aggregation_rule="line_b_equal_role_families",
        )
        _, _, canonical_role_rows, _ = experiment._make_oof(
            [predicted, dropped],
            ("heldout", "dropped"),
            dataset.identities,
            probabilities,
            role_common,
            balance_line="line_b_equal_role_families",
        )
        self.assertTrue(canonical_role_rows)
        self.assertTrue(all(row.level == "role" for row in canonical_role_rows))
        self.assertTrue(
            all(
                row.file_id == f"role::{row.participant_id}::{row.role}"
                for row in canonical_role_rows
            )
        )

    def test_zero_retained_oof_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            experiment._ExperimentProtocolError,
            "outer_oof_zero_retained_predictions",
        ):
            experiment._require_retained_oof(
                (SimpleNamespace(retained=False), SimpleNamespace(retained=False))
            )

    def test_effect_size_fixed_v1_has_no_pip_parameter(self) -> None:
        section = {
            "model_id": "ShapeFormerEffectSizeFixedV1",
            "discovery_method": "effect_size_fixed_v1",
            "input_fs_hz": 400.0,
            "shapelet_length_samples": 128,
            "shapelets_per_class": 3,
            "discovery_stride_samples": 64,
            "max_candidates_per_class": 128,
            "hidden_channels": 64,
            "dropout": 0.2,
            "patch_size_samples": 16,
            "attention_heads": 4,
            "attention_layers": 2,
            "distance_position_chunk_size": 256,
            "seed_policy": "outer_cv_repeat_seed_equals_split_seed",
            "architecture_parameters": {
                "model_id": "shapeformer_effect_size_fixed_v1",
            },
        }
        config = SimpleNamespace(
            section=lambda name: {
                "model": section,
                "signal": {"internal_fs_hz": 400.0},
            }[name]
        )
        resolved, machine_id = experiment._resolved_model_config(
            config,
            training_seed=42,
        )
        self.assertEqual(machine_id, "shapeformer_effect_size_fixed_v1")
        self.assertNotIn("num_pip_ratio", resolved)
        self.assertEqual(resolved["shapelet_length_samples"], 128)
        self.assertEqual(resolved["discovery_stride_samples"], 64)
        resolved_repeat_3, _ = experiment._resolved_model_config(
            config,
            training_seed=30042,
        )
        self.assertEqual(resolved_repeat_3["seed"], 30042)

    def test_experiment_result_has_explicit_v2_identity(self) -> None:
        payload = experiment.ExperimentResult(
            status="failed_closed",
            scientific_scope="unit_contract",
            config_id="fixture",
            config_hash="hash",
            repeat_indices=(0,),
            fold_indices=(0,),
            output_dir=None,
        ).to_dict()
        self.assertEqual(payload["schema_version"], "ppg_frailty.experiment_result.v2")
        self.assertEqual(payload["pipeline_generation"], "final_pipeline_v2")

    def test_public_comparison_defaults_are_thirteen_nonensemble_models(self) -> None:
        """Inspect identities without running a comparison / 只检查身份，不执行比较。"""

        default_models = inspect.signature(
            pipeline.run_model_comparison
        ).parameters["models"].default
        self.assertEqual(len(default_models), 13)
        self.assertIn("ShapeFormerChannelSpecificOSD", default_models)
        self.assertIn("ShapeFormerEffectSizeFixedV1", default_models)
        self.assertNotIn("InceptionTimeFullFiveMemberEnsemble", default_models)
        self.assertNotIn("InceptionTimeMatrixFiveMemberEnsemble", default_models)
        source = inspect.getsource(pipeline.run_model_comparison)
        self.assertIn('"comparison_only": True', source)
        self.assertIn('"channel_specific_osd"', source)
        self.assertIn('"effect_size_fixed_v1"', source)
        self.assertNotIn('"ShapeFormerPISDPort"', source)

    def test_fixed_kernel_dispatch_replaces_v1_physical_time_name(self) -> None:
        """Inspect the V2 dispatcher without executing ablation / 不执行消融。"""

        source = inspect.getsource(pipeline.run_ablation)
        self.assertIn('factor == "fixed_kernel_samples"', source)
        self.assertNotIn('factor == "physical_time"', source)
        self.assertNotIn("build_physical_time_cases", source)
        self.assertNotIn("create_time_scaled_model", source)
        formal_source = inspect.getsource(experiment._execute_cell_unchecked)
        self.assertIn("prepare_fixed_kernel_dl_input", formal_source)
        self.assertIn("dl_case_id", formal_source)
        self.assertIn("canonical_features_and_peaks_unchanged", formal_source)

    def test_dispatch_is_not_vector_only_and_ensemble_is_explicit_comparison(self) -> None:
        source = inspect.getsource(experiment._execute_cell_unchecked)
        for mode in ("raw", "feature_vector", "feature_matrix", "fusion"):
            self.assertIn(repr(mode), source)
        self.assertNotIn("unsupported_representation_for_current_runner", source)
        config = SimpleNamespace(
            section=lambda name: {
                "model_id": "InceptionTimeFullFiveMemberEnsemble",
                "comparison_only": True,
                "member_seeds": [50042, 60042, 70042, 80042, 90042],
                "seed_policy": "cv_fixed_five_member_seed_roster",
                "member_seed_roster_id": "cv_fixed_five_member_seed_roster",
                "dropout": 0.2,
                "kernel_sizes": [39, 19, 9],
                "dilation": 1,
                "architecture_parameters": {
                    "model_id": "inception_full_five_member_ensemble",
                },
            }
        )
        resolved, machine_id = experiment._resolved_model_config(
            config,
            training_seed=50042,
        )
        self.assertEqual(machine_id, "inception_full_five_member_ensemble")
        self.assertEqual(
            resolved["member_seeds"],
            (50042, 60042, 70042, 80042, 90042),
        )

    def test_member0_comparator_seed_is_outer_cv_only(self) -> None:
        config = SimpleNamespace(
            section=lambda name: {
                "model_id": "InceptionTimeFull",
                "seed_policy": "cv_fixed_member0_seed_50042_comparator",
                "dropout": 0.2,
                "kernel_sizes": [39, 19, 9],
                "dilation": 1,
                "architecture_parameters": {"model_id": "inception_full"},
            }
        )
        outer, machine_id = experiment._resolved_model_config(
            config,
            training_seed=50042,
            seed_scope="outer_cv",
        )
        final, final_machine_id = experiment._resolved_model_config(
            config,
            training_seed=42,
            seed_scope="final_refit",
        )
        self.assertEqual(machine_id, "inception_full")
        self.assertEqual(final_machine_id, "inception_full")
        self.assertEqual(outer["seed"], 50042)
        self.assertEqual(final["seed"], 42)
        with self.assertRaisesRegex(
            experiment._ExperimentProtocolError,
            "single_model_final_refit_seed_must_be_42",
        ):
            experiment._resolved_model_config(
                config,
                training_seed=50042,
                seed_scope="final_refit",
            )


if __name__ == "__main__":
    unittest.main()

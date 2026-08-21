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
    ENGINEERING_SCHEMA_VERSION,
    EngineeringExtraction,
    build_feature_vector,
    default_registry,
    engineering_feature_names,
    registry_for_groups,
)
from ppg_frailty.peaks import BeatPairAudit, BeatPairingResult
from ppg_frailty.representations import RawWindows
from ppg_frailty.signal.motion_imu import MOTION_IMU_CALIBRATION_SCHEMA
from ppg_frailty.signal.optical import OpticalBeatAudit
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
    engineering_names = engineering_feature_names()
    engineering_values = np.tile(
        np.linspace(value, value + 1.0, len(engineering_names), dtype=np.float64),
        (2, 1),
    )
    sequence = EngineeringFeatureSequence(
        values=engineering_values,
        start_samples=np.asarray((0, 2000), dtype=np.int64),
        valid_row_mask=np.ones(2, dtype=bool),
        channel_schema=engineering_names,
        schema_version=ENGINEERING_SCHEMA_VERSION,
    )
    state.engineering = EngineeringExtraction(
        sequence=sequence,
        value_validity=np.ones(engineering_values.shape, dtype=bool),
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
    def test_composable_file_bag_fusion_preserves_optional_registry_role(self) -> None:
        self.assertEqual(
            experiment._registry_role_for_machine_id("file_bag_fusion"),
            "optional",
        )

    def test_source_snapshot_is_deterministic_content_addressed_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "z.py").write_text("VALUE = 2\n", encoding="utf-8")
            package = root / "nested"
            package.mkdir()
            source = package / "a.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")

            first = experiment._source_tree_sha256(root)
            second = experiment._source_tree_sha256(root)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 64)
            self.assertTrue(all(value in "0123456789abcdef" for value in first))

            source.write_text("VALUE = 3\n", encoding="utf-8")
            self.assertNotEqual(first, experiment._source_tree_sha256(root))

    def test_final_refit_source_snapshot_must_match_current_code(self) -> None:
        current = experiment._source_version()
        rows = (SimpleNamespace(source_snapshot_hash=current),)
        self.assertEqual(experiment._validated_oof_source_snapshot(rows), current)

        with self.assertRaisesRegex(ValueError, "current_source_differs"):
            experiment._validated_oof_source_snapshot(
                (SimpleNamespace(source_snapshot_hash="0" * 64),)
            )

        with self.assertRaisesRegex(ValueError, "identity_drift"):
            experiment._validated_oof_source_snapshot(
                (SimpleNamespace(source_snapshot_hash="not-a-sha256"),)
            )

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
            schema_version=MOTION_IMU_CALIBRATION_SCHEMA,
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
            "training_history": [
                {"epoch": 1, "training_loss": 0.75},
                {
                    "epoch": 1,
                    "training_participant_balanced_accuracy": 0.5,
                    "training_balanced_accuracy_unit": "participant",
                    "training_balanced_accuracy_aggregation_rule": (
                        "line_b_equal_role_families"
                    ),
                    "training_data_scope": "full_outer_train_only",
                    "outer_heldout_used": False,
                    "metric_used_for_selection_or_checkpoint": False,
                },
            ],
            "learning_curve_contract": {
                "status": "outer_train_loss_and_participant_ba_fixed_epoch",
                "training_data_scope": "full_outer_train_only",
                "outer_heldout_used_for_epoch_selection_or_curve": False,
                "training_metric": "training_participant_balanced_accuracy",
                "training_metric_unit": "participant",
                "training_metric_aggregation_rule": "line_b_equal_role_families",
                "training_metric_used_for_epoch_selection_or_checkpoint": False,
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
                [
                    {"repeat": 0, "fold": 0, "epoch": 1, "training_loss": 0.75},
                    {
                        "repeat": 0,
                        "fold": 0,
                        "epoch": 1,
                        "training_participant_balanced_accuracy": 0.5,
                        "training_balanced_accuracy_unit": "participant",
                        "training_balanced_accuracy_aggregation_rule": (
                            "line_b_equal_role_families"
                        ),
                        "training_data_scope": "full_outer_train_only",
                        "outer_heldout_used": False,
                        "metric_used_for_selection_or_checkpoint": False,
                    },
                ],
            )
            self.assertFalse(
                history["learning_curve_contract"][
                    "outer_heldout_used_for_epoch_selection_or_curve"
                ]
            )
            self.assertFalse(
                history["learning_curve_contract"][
                    "training_metric_used_for_epoch_selection_or_checkpoint"
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
        pulse = object()
        observed_prv_kwargs: dict[str, object] = {}
        observed_peak_kwargs: dict[str, object] = {}

        def compute_prv_fixture(*_args: object, **kwargs: object) -> SimpleNamespace:
            observed_prv_kwargs.update(kwargs)
            return SimpleNamespace(
                values={"coverage": 0.75, "hr_mean_bpm": 60.0},
                validity={"coverage": True, "hr_mean_bpm": True},
            )

        def detect_fixture(*_args: object, **kwargs: object) -> dict[str, object]:
            observed_peak_kwargs.update(kwargs)
            return {"RED": pulse, "IR": pulse}
        paired_row = BeatPairAudit(
            reference_wavelength="RED",
            secondary_wavelength="IR",
            reference_peak_ordinal=0,
            reference_peak_sample=100,
            red_peak_ordinal=0,
            red_peak_sample=100,
            ir_peak_ordinal=0,
            ir_peak_sample=120,
            lag_samples_ir_minus_red=20,
            lag_s_ir_minus_red=0.05,
            pair_valid=True,
            reason_codes=("paired",),
        )
        pairing = BeatPairingResult(
            detector_id="aboy_project_v1",
            reference_wavelength="RED",
            secondary_wavelength="IR",
            reference_score=1.0,
            reference_coverage=1.0,
            secondary_score=1.0,
            secondary_coverage=1.0,
            red_detection_run_id="red-run",
            ir_detection_run_id="ir-run",
            red_detector_version="aboy_project_v1:red",
            ir_detector_version="aboy_project_v1:ir",
            red_selected_polarity=1,
            ir_selected_polarity=-1,
            red_block_hri_provenance_hash="0" * 64,
            ir_block_hri_provenance_hash="1" * 64,
            rows=(paired_row,),
        )
        optical_audit = OpticalBeatAudit(
            pairing=paired_row,
            red_left_valley_sample=80,
            red_right_valley_sample=180,
            ir_left_valley_sample=90,
            ir_right_valley_sample=190,
            optical_valid=True,
            reason_codes=("paired",),
        )
        api = {
            "SignalRoute": SignalRoute,
            "QualityState": SimpleNamespace(PASS="pass"),
            "detect_pulses_per_wavelength": detect_fixture,
            "select_reference_wavelength": lambda _pulses: "RED",
            "compute_prv": compute_prv_fixture,
            "canonicalize_role_family": lambda value: value,
            "WindowPlan": lambda **_k: object(),
            "extract_engineering_features": lambda *_a, **_k: object(),
            "summarize_engineering": lambda *_a, **_k: ({}, {}),
            "extract_morphology": lambda *_a, **_k: SimpleNamespace(
                aggregate_values={},
                aggregate_validity={},
            ),
            "extract_dual_optical": lambda *_a, **_k: SimpleNamespace(
                aggregate_values={"red_ac_median": 1.25},
                aggregate_validity={"red_ac_median": True},
                schema_version="dual_optical_fixture_v2",
                pairing=pairing,
                beat_audit=(optical_audit,),
                diagnostics={"affects_prediction": False},
                reasons=(),
            ),
            "default_registry": lambda: registry,
            "registry_for_groups": registry_for_groups,
            "build_feature_vector": build_feature_vector,
        }
        report = SimpleNamespace(
            window_profiles={"engineering": {}},
            peak_detector={
                "detector_id": "aboy_project_v1",
                "min_observation_sec": 6.5,
                "min_peaks": 3,
            },
        )
        with patch(
            "ppg_frailty.experiment._runtime_imports",
            return_value=api,
        ):
            experiment._extract_vector(
                state,
                report,
                {
                    "time_prv_min_duration_s": 45.0,
                    "matrix_k": 9,
                },
            )
        self.assertTrue(state.retained)
        self.assertEqual(
            observed_peak_kwargs,
            {
                "detector_id": "aboy_project_v1",
                "min_observation_sec": 6.5,
                "min_peaks": 3,
            },
        )
        self.assertIs(state.route, SignalRoute.DIRECT)
        self.assertNotIn("prv.coverage", state.vector.feature_names)
        optical_index = state.vector.feature_names.index("optical.red_ac_median")
        self.assertEqual(state.vector.values[optical_index], 1.25)
        self.assertTrue(state.vector.validity[optical_index])
        self.assertEqual(
            observed_prv_kwargs["config"].time_prv_min_duration_s,
            45.0,
        )
        self.assertEqual(
            state.vector.provenance["prv_config"]["time_prv_min_duration_s"],
            45.0,
        )
        self.assertEqual(
            state.diagnostic_components["non_predictor_features"][
                "prv.coverage"
            ],
            {"value": 0.75, "valid": True},
        )
        availability = state.diagnostic_components["predictor_availability"]
        self.assertEqual(availability["predictor_count"], len(registry.names))
        self.assertEqual(availability["available_predictor_count"], 2)
        compact = state.diagnostic_components["dual_optical_pairing"]
        self.assertNotIn("pairing", compact)
        self.assertNotIn("beat_audit", compact)
        self.assertNotIn("rows", compact["pairing_summary"])
        self.assertEqual(compact["pairing_summary"]["row_count"], 1)
        self.assertEqual(compact["pairing_summary"]["valid_pair_count"], 1)
        self.assertEqual(compact["beat_audit_summary"]["row_count"], 1)
        self.assertEqual(
            compact["beat_audit_summary"]["optical_valid_count"],
            1,
        )
        self.assertEqual(
            compact["pairing_summary"]["detector_id"],
            "aboy_project_v1",
        )
        encoded = json.dumps(compact, sort_keys=True)
        self.assertNotIn("red_peak_sample", encoded)
        self.assertLess(len(encoded), 4_000)

        fallback = experiment._compact_dual_optical_diagnostics(
            SimpleNamespace(
                schema_version="dual_optical_fixture_v2",
                pairing=None,
            )
        )
        self.assertEqual(fallback["status"], "summary_unavailable_noncausal")
        self.assertNotIn("pairing", fallback)
        self.assertNotIn("beat_audit", fallback)
        self.assertTrue(state.retained)
        self.assertIs(state.route, SignalRoute.DIRECT)
        self.assertEqual(state.vector.values[optical_index], 1.25)

        def forbidden_shape_feature(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("rate_only_direct must not extract shape features")

        rate_only_state = experiment._RuntimeRecord(
            row=SimpleNamespace(record_id="f2", role="B"),
            views=SimpleNamespace(
                x_filter=np.zeros((400, 2), dtype=np.float64),
                x_native=np.zeros((400, 2), dtype=np.float64),
            ),
            retained=True,
            route=SignalRoute.DIRECT,
            route_status="rate_only_direct",
            quality_tier="acceptable",
            shape_features_eligible=False,
        )
        rate_only_api = {
            **api,
            "extract_engineering_features": forbidden_shape_feature,
            "extract_morphology": forbidden_shape_feature,
            "extract_dual_optical": forbidden_shape_feature,
        }
        with patch(
            "ppg_frailty.experiment._runtime_imports",
            return_value=rate_only_api,
        ):
            experiment._extract_vector(rate_only_state, report, {})
        self.assertTrue(rate_only_state.retained)
        self.assertIsNone(rate_only_state.engineering)
        self.assertFalse(
            rate_only_state.vector.validity[
                rate_only_state.vector.feature_names.index(
                    "morphology.amplitude_median"
                )
            ]
        )
        self.assertFalse(
            rate_only_state.vector.validity[
                rate_only_state.vector.feature_names.index("optical.red_ac_median")
            ]
        )
        self.assertNotIn(
            "dual_optical_pairing", rate_only_state.diagnostic_components
        )

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
        self.assertEqual(raw_provenance["raw_imu"]["fitted_on_participant_ids"], ())
        self.assertEqual(
            raw_provenance["raw_imu"]["strategy"],
            "none_after_all8_per_window_robust",
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
        self.assertNotIn("feature_vector", matrix_provenance)
        self.assertEqual(
            tuple(matrix_provenance["engineering"]["fitted_on_participant_ids"]),
            TRAIN_IDS,
        )
        matrix_dataset = experiment._materialize_representation_dataset(
            matrix_states, OOF_IDS, "feature_matrix"
        )
        self.assertEqual(matrix_dataset.representation_mode, "feature_matrix")
        self.assertEqual(matrix_dataset.values.shape[0], 1)
        self.assertEqual(matrix_dataset.values.shape[1:], (115, 150))

        fusion_states = _states()
        fusion_provenance = experiment._fit_representation_artifacts(
            fusion_states, "fusion", TRAIN_IDS, OOF_IDS
        )
        self.assertEqual(fusion_provenance["raw_imu"]["fitted_on_participant_ids"], ())
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

    def test_zero_retained_oof_is_reported_as_complete_abstention(self) -> None:
        metrics = experiment._evaluate_subjects(
            tuple(
                SimpleNamespace(retained=False, label=label)
                for label in (0, 1, 2)
            ),
            total=3,
        )
        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertEqual(metrics["abstention_aware_balanced_accuracy"], 0.0)
        self.assertEqual(metrics["abstention_aware_macro_f1"], 0.0)
        self.assertEqual(metrics["coverage_rate"], 0.0)
        self.assertEqual(metrics["abstention_count"], 3)

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

    def test_channel_specific_scalar_ablation_reaches_experiment_factory(self) -> None:
        section = {
            "model_id": "ShapeFormerChannelSpecificScalarDistanceAblation",
            "seed_policy": "outer_cv_repeat_seed_equals_split_seed",
            "num_pip_ratio": 0.35,
            "shapelets_per_class": 2,
            "max_discovery_windows": 24,
            "position_search_neighbourhood_samples": 17,
            "hidden_channels": 12,
            "dropout": 0.15,
            "patch_size_samples": 5,
            "attention_heads": 3,
            "attention_layers": 2,
            "distance_position_chunk_size": 19,
            "architecture_parameters": {
                "model_id": (
                    "shapeformer_channel_specific_scalar_distance_ablation"
                ),
            },
        }
        config = SimpleNamespace(section=lambda _name: section)
        resolved, machine_id = experiment._resolved_model_config(
            config,
            training_seed=20042,
        )
        self.assertEqual(
            machine_id,
            "shapeformer_channel_specific_scalar_distance_ablation",
        )
        self.assertEqual(resolved["num_pip_ratio"], 0.35)
        self.assertEqual(resolved["position_search_neighbourhood_samples"], 17)
        self.assertEqual(resolved["hidden_channels"], 12)
        self.assertEqual(resolved["attention_heads"], 3)
        self.assertEqual(resolved["seed"], 20042)
        self.assertEqual(
            experiment._model_capability_contract(machine_id)["execution_backend"],
            "torch",
        )
        self.assertEqual(
            experiment._registry_role_for_machine_id(machine_id),
            "ablation",
        )

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

    def test_public_comparison_defaults_exclude_retired_rocket_models(self) -> None:
        """Inspect identities without running a comparison / 只检查身份，不执行比较。"""

        default_models = inspect.signature(
            pipeline.run_model_comparison
        ).parameters["models"].default
        self.assertEqual(len(default_models), 10)
        self.assertIn("ShapeFormerChannelSpecificOSD", default_models)
        self.assertIn("ShapeFormerEffectSizeFixedV1", default_models)
        self.assertNotIn("InceptionTimeFullFiveMemberEnsemble", default_models)
        self.assertNotIn("InceptionTimeMatrixFiveMemberEnsemble", default_models)
        self.assertNotIn("ROCKET", default_models)
        self.assertNotIn("MiniROCKET", default_models)
        self.assertNotIn("InceptionTimeMatrix", default_models)
        self.assertEqual(
            inspect.signature(pipeline.run_model_comparison)
            .parameters["ensemble_size"]
            .default,
            5,
        )
        source = inspect.getsource(pipeline.run_model_comparison)
        self.assertNotIn('"comparison_only": True', source)
        self.assertIn("model_factory_contract", source)
        self.assertIn("resolved_architecture_parameters", source)
        self.assertIn("PISD_DISCOVERY_METHOD", source)
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
        sampling_source = inspect.getsource(experiment._prepare_dl_input_dataset)
        self.assertIn("prepare_fixed_kernel_dl_input", sampling_source)
        self.assertIn("dl_case_id", formal_source)
        self.assertIn("canonical_features_and_peaks_unchanged", formal_source)

    def test_dispatch_is_not_vector_only_and_legacy_ensemble_preset_resolves(self) -> None:
        source = inspect.getsource(experiment._execute_cell_unchecked)
        for mode in ("raw", "feature_vector", "feature_matrix", "fusion"):
            self.assertIn(repr(mode), source)
        self.assertNotIn("unsupported_representation_for_current_runner", source)
        config = SimpleNamespace(
            section=lambda name: {
                "model_id": "InceptionTimeFullFiveMemberEnsemble",
                "comparison_only": True,
                "member_seeds": [50042, 60042, 70042, 80042, 90042],
                "ensemble_size": 5,
                "seed_policy": "cv_fixed_five_member_seed_roster",
                "member_seed_roster_id": "cv_fixed_five_member_seed_roster",
                "dropout": 0.2,
                "kernel_sizes": [39, 19, 9],
                "dilation": 1,
                "architecture_parameters": {
                    "model_id": "inception_full_five_member_ensemble",
                    "member_count": 5,
                    "member_seeds": [50042, 60042, 70042, 80042, 90042],
                    "member_variant": "full",
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
        selected_seed, _ = experiment._resolved_model_config(
            config,
            training_seed=50042,
            seed_scope="final_refit",
        )
        self.assertEqual(selected_seed["seed"], 50042)
        self.assertEqual(selected_seed["seed_policy"], "fixed_explicit")

    def test_arbitrary_ensemble_roster_reaches_runtime_and_final_policy(self) -> None:
        model = {
            "model_id": "InceptionTimeFullFiveMemberEnsemble",
            "ensemble_size": 2,
            "member_seeds": [17, 29],
            "seed_policy": "member_roster",
            "dropout": 0.2,
            "kernel_sizes": [39, 19, 9],
            "dilation": 1,
            "architecture_parameters": {
                "model_id": "inception_full_five_member_ensemble",
                "member_count": 2,
                "member_seeds": [17, 29],
                "member_variant": "full",
            },
        }
        sections = {"model": model, "training": {"seed": 123}}
        config = SimpleNamespace(
            config_id="custom_two_member",
            sha256="a" * 64,
            section=lambda name: sections[name],
        )
        resolved, machine_id = experiment._resolved_model_config(
            config, training_seed=123
        )
        self.assertEqual(machine_id, "inception_full_five_member_ensemble")
        self.assertEqual(resolved["member_seeds"], (17, 29))
        self.assertEqual(resolved["seed_policy"], "member_roster")
        self.assertEqual(
            experiment._outer_cv_model_training_seed(
                config, {"training_seed": 40042}
            ),
            123,
        )
        policy = experiment.final_refit_policy(config)
        self.assertEqual(policy["refit"]["kind"], "probability_ensemble")
        self.assertEqual(policy["refit"]["member_seeds"], [17, 29])
        self.assertEqual(policy["refit"]["orchestration_seed"], 123)

        one_member_model = dict(model)
        one_member_model.update(
            {
                "ensemble_size": 1,
                "member_seeds": [17],
                "architecture_parameters": {
                    **model["architecture_parameters"],
                    "member_count": 1,
                    "member_seeds": [17],
                },
            }
        )
        one_member_sections = {
            "model": one_member_model,
            "training": {"seed": 123},
        }
        one_member_config = SimpleNamespace(
            config_id="custom_one_member_ensemble",
            sha256="b" * 64,
            section=lambda name: one_member_sections[name],
        )
        resolved_one, _ = experiment._resolved_model_config(
            one_member_config, training_seed=123
        )
        self.assertEqual(resolved_one["member_seeds"], (17,))
        self.assertEqual(
            experiment.final_refit_policy(one_member_config)["refit"]["member_seeds"],
            [17],
        )


if __name__ == "__main__":
    unittest.main()

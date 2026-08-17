"""Tiny representation-dispatch contract tests; no CV, ablation, or real training."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest

import numpy as np

import ppg_frailty.experiment as experiment
import ppg_frailty.pipeline as pipeline
from ppg_frailty.contracts import EngineeringFeatureSequence, SignalRoute
from ppg_frailty.features import EngineeringExtraction, build_feature_vector
from ppg_frailty.representations import RawWindows


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
        {"sqi.q_rate": value},
        feature_validity={"sqi.q_rate": True},
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
        window_rows, file_rows, subject_rows = experiment._make_oof(
            [predicted, dropped],
            ("heldout", "dropped"),
            dataset.identities,
            probabilities,
            common,
            balance_line="line_a_equal_files",
        )
        self.assertEqual(len(window_rows), 2)
        retained_file = next(row for row in file_rows if row.participant_id == "heldout")
        np.testing.assert_allclose(retained_file.probabilities, (0.5, 0.2, 0.3))
        dropped_file = next(row for row in file_rows if row.participant_id == "dropped")
        self.assertFalse(dropped_file.retained)
        dropped_subject = next(row for row in subject_rows if row.participant_id == "dropped")
        self.assertFalse(dropped_subject.retained)

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
                "member_seeds": [42, 10042, 20042, 30042, 40042],
                "seed_policy": "pending_cv_repeat_member_seed_matrix_decision",
                "member_seed_roster_id": "cv_fixed_five_member_seed_roster",
                "dropout": 0.2,
                "kernel_sizes": [39, 19, 9],
                "dilation": 1,
                "architecture_parameters": {
                    "model_id": "inception_full_five_member_ensemble",
                },
            }
        )
        with self.assertRaisesRegex(
            experiment._ExperimentProtocolError,
            "ensemble_cv_repeat_member_seed_matrix_decision_pending",
        ):
            experiment._resolved_model_config(config, training_seed=42)


if __name__ == "__main__":
    unittest.main()

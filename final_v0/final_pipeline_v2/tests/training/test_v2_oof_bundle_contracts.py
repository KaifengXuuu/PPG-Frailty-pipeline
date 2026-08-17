"""Safe V2 OOF, bundle, ensemble-roster and final-refit contract tests."""

from __future__ import annotations

import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ppg_frailty.bundle import build_model_input_adapter
from ppg_frailty.models import (
    ModelInputSpec,
    create_model,
    materialize_architecture_parameters,
)
from ppg_frailty.training import (
    FeatureVectorDataset,
    FinalRefitPlan,
    FittedObjectProvenance,
    FullCohortRefitScope,
    OofPredictionRow,
    OofWriter,
    RawWindowDataset,
    SampleIdentity,
    TrainingConfig,
    TrainingResult,
    UnifiedTrainer,
    assert_golden_parity,
    current_runtime_environment,
    load_bundle,
    materialize_final_refit_binding,
    read_oof_parquet,
    read_oof_parquet_metadata,
    save_bundle,
    save_final_refit_bundle,
    validate_expected_oof_roster,
)
from ppg_frailty.training.bundle import (
    FrozenRepresentationTransformArchive,
    _execute_prepared_full_cohort_refit,
    _save_trusted_final_refit_bundle,
)


def _trace_row(
    *,
    retained: bool = True,
    member_index: int | None = None,
    prediction_kind: str = "single_model",
) -> OofPredictionRow:
    member_seeds = (50042, 60042, 70042, 80042, 90042)
    training_seed = (
        member_seeds[member_index]
        if prediction_kind == "ensemble_member" and member_index is not None
        else 42 if prediction_kind == "single_model" else None
    )
    probabilities = () if not retained else (0.7, 0.2, 0.1)
    return OofPredictionRow(
        participant_id="P00",
        file_id="participant::P00",
        role="participant",
        label=0,
        probabilities=probabilities,
        repeat=0,
        fold=0,
        split_seed=42,
        training_seed=training_seed,
        config_hash="config",
        manifest_hash="manifest",
        fold_hash="fold",
        preprocessing_hash="preprocess",
        feature_hash="feature",
        model_hash="model",
        representation_mode="raw",
        signal_route="direct",
        quality_score=1.0 if retained else 0.0,
        retained=retained,
        level="participant",
        member_index=member_index,
        prediction_kind=prediction_kind,
        member_training_seeds=(
            member_seeds if prediction_kind == "ensemble_average" else ()
        ),
        ensemble_base_model_id=(
            "inception_full" if prediction_kind.startswith("ensemble_") else ""
        ),
        class_order=(0, 1, 2) if retained else (),
        code_commit="commit",
        data_schema_id="data_v1",
        feature_schema_id="feature_v1",
        model_version="model_v1",
        aggregation_rule="line_a_equal_files",
        environment_hash="environment",
        manifest_version="manifest_v1",
        fold_registry_version="fold_v1",
        artifact_reducer_name="identity",
        artifact_reducer_version="identity_v1",
        route_status="retained" if retained else "dropped",
        source_snapshot_hash="snapshot",
        rejection_reason=None if retained else "hard_qc_rejection",
    )


def _bundle_metadata() -> dict[str, object]:
    return {
        "model_identity": {
            "name": "CompactCNN1D",
            "machine_id": "compact_cnn",
            "version": "contract",
        },
        "representation_mode": "raw",
        "signal_route": "direct",
        "class_order": [0, 1, 2],
        "channel_schema": [
            "RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"
        ],
        "preprocessing": {"name": "frozen_model_input", "version": "v2"},
        "preprocessing_hash": "preprocessing",
        "resampling": {"status": "not_applied", "method": "none"},
        "window_plan": {"name": "contract", "length_samples": 64},
        "feature_registry": {"status": "not_applicable", "version": "v2"},
        "feature_hash": "not_applicable",
        "feature_vector_schema": {"status": "not_applicable"},
        "ordered_matrix_schema": {"status": "not_applicable"},
        "mask_semantics": {"sample_mask": "true_is_valid"},
        "validity_policy": {"unavailable": "nan_and_false"},
        "fitted_objects": ["model"],
        "representation_state": {"kind": "raw_weights"},
        "pooling_rule": "window_then_file",
        "aggregation_rule": "line_a_equal_files",
        "manifest_hash": "manifest",
        "fold_hash": "fold",
        "manifest_version": "manifest_v2",
        "fold_registry_version": "fold_v2",
        "pipeline_generation": "final_pipeline_v2",
        "config_hash": "a" * 64,
        "balance_hash": "b" * 64,
        "run_hash": "c" * 64,
        "source_snapshot_hash": "d" * 64,
        "code_version": "contract",
        "environment": current_runtime_environment(),
        "dependency_status": "contract_runtime",
        "serialization_trust": {
            "trusted_local_only": True,
            "authenticated_signature": False,
        },
        "golden_case": {"id": "contract", "n_samples": 1},
    }


class TypedOofContracts(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow unavailable")
    def test_populated_and_empty_oof_use_the_exact_v2_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            populated = OofWriter().write(
                (_trace_row(),),
                Path(temporary) / "populated.parquet",
            )
            empty = OofWriter().write_empty(
                Path(temporary) / "empty.parquet",
                "level_deliberately_absent",
            )
            self.assertEqual(read_oof_parquet(populated), (_trace_row(),))
            self.assertEqual(read_oof_parquet(empty), ())
            self.assertEqual(
                read_oof_parquet_metadata(empty)["empty_reason"],
                "level_deliberately_absent",
            )

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow unavailable")
    def test_list_children_are_nonnullable_and_null_element_tamper_is_rejected(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from ppg_frailty.training.oof import _arrow_schema

        schema = _arrow_schema(pa)
        for name in ("probabilities", "member_training_seeds", "class_order"):
            self.assertFalse(schema.field(name).type.value_field.nullable)
        with tempfile.TemporaryDirectory() as temporary:
            valid_path = OofWriter().write(
                (_trace_row(),),
                Path(temporary) / "valid.parquet",
            )
            table = pq.read_table(valid_path)
            probability_index = table.schema.get_field_index("probabilities")
            tampered = table.set_column(
                probability_index,
                "probabilities",
                pa.array([[0.7, None, 0.3]], type=pa.list_(pa.float64())),
            )
            tampered_path = Path(temporary) / "tampered.parquet"
            pq.write_table(tampered, tampered_path)
            with self.assertRaisesRegex(ValueError, "schema"):
                read_oof_parquet(tampered_path)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow unavailable")
    def test_all_six_dropped_ensemble_rows_validate_and_roundtrip(self) -> None:
        rows = tuple(
            _trace_row(
                retained=False,
                member_index=index,
                prediction_kind="ensemble_member",
            )
            for index in range(5)
        ) + (_trace_row(retained=False, prediction_kind="ensemble_average"),)
        validate_expected_oof_roster(
            rows,
            {(0, 0, 42): ("P00",)},
            expected_config_hashes=("config",),
            expected_member_count=5,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = OofWriter().write(rows, Path(temporary) / "ensemble.parquet")
            self.assertEqual(read_oof_parquet(path), rows)


class BundleContracts(unittest.TestCase):
    def test_environment_versions_are_descriptive_and_golden_is_stable(self) -> None:
        spec = ModelInputSpec(
            "raw",
            n_channels=8,
            n_classes=3,
            channel_schema=(
                "RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"
            ),
        )
        config = {
            "model_id": "compact_cnn",
            "seed": 42,
            "dropout": 0.2,
            "kernel_sizes": [9, 9, 7],
            "dilations": [1, 1, 1],
            "pool_sizes": [4, 4],
        }
        config["architecture_parameters"] = materialize_architecture_parameters(config, spec)
        model = create_model(config, spec).eval()
        golden = {
            "x": np.random.default_rng(42).normal(size=(1, 8, 64)).astype(np.float32)
        }
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "bundle"
            save_bundle(
                model,
                target,
                model_config=config,
                input_spec=spec,
                metadata=_bundle_metadata(),
                golden_inputs=golden,
            )
            loaded = load_bundle(target)
            assert_golden_parity(loaded)
            forged_metadata = _bundle_metadata()
            forged_metadata["final_refit_identity"] = {"caller": "forged"}
            with self.assertRaisesRegex(
                ValueError,
                "generic bundle cannot claim trusted final-refit identity",
            ):
                save_bundle(
                    model,
                    Path(temporary) / "forged-final",
                    model_config=config,
                    input_spec=spec,
                    metadata=forged_metadata,
                    golden_inputs=golden,
                )
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["golden_parity_atol"] = 1.0
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "tolerance"):
                load_bundle(target)
            manifest["golden_parity_atol"] = 1e-6
            manifest["metadata"]["environment"]["python"] = "0.0.0"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            loaded_with_recorded_mismatch = load_bundle(target)
            self.assertEqual(
                loaded_with_recorded_mismatch.manifest["metadata"]["environment"][
                    "python"
                ],
                "0.0.0",
            )

    def test_pipeline_generation_aliases_are_rejected(self) -> None:
        for alias in ("V1", "V2"):
            metadata = _bundle_metadata()
            metadata["pipeline_generation"] = alias
            from ppg_frailty.training import validate_bundle_metadata

            with self.assertRaisesRegex(ValueError, "exactly final_pipeline_v2"):
                validate_bundle_metadata(metadata)


class FinalRefitContracts(unittest.TestCase):
    def test_experiment_entry_points_do_not_require_a_git_checkout(self) -> None:
        import ppg_frailty.experiment as experiment_module

        self.assertEqual(experiment_module._code_version(), "not_git_bound")

    def test_final_ensemble_plan_requires_exact_five_member_seed_roster(self) -> None:
        participants = tuple(f"P{i:02d}" for i in range(29))
        common = {
            "purpose": "ensemble_contract",
            "config_hash": "a" * 64,
            "model_id": "InceptionTimeFullFiveMemberEnsemble",
            "participant_ids": participants,
            "fixed_epochs": 10,
            "epoch_rule": "fixed_epoch",
            "model_family": "deep",
            "oof_evidence_hash": "b" * 64,
            "model_kind": "five_member_ensemble",
            "registry_hash": "c" * 64,
            "source_snapshot_hash": "d" * 64,
            "manual_selection_hash": "e" * 64,
            "resolved_model_config_hash": "f" * 64,
            "architecture_parameters_hash": "1" * 64,
            "input_schema_hash": "2" * 64,
            "training_config_hash": "3" * 64,
            "frozen_run_provenance_hash": "4" * 64,
            "representation_mode": "raw",
        }
        plan = FinalRefitPlan(
            training_seeds=(50042, 60042, 70042, 80042, 90042),
            **common,
        )
        self.assertEqual(plan.training_seeds, (50042, 60042, 70042, 80042, 90042))
        with self.assertRaisesRegex(ValueError, "exact five member seeds"):
            FinalRefitPlan(training_seeds=(42,), **common)

    def test_ensemble_executor_accepts_seed50042_without_running_training(self) -> None:
        participants = tuple(f"P{index:02d}" for index in range(29))
        identities = tuple(
            SampleIdentity(
                participant_id=participant,
                file_id=f"{participant}_B",
                role="B",
                label=index % 3,
                signal_route="direct",
            )
            for index, participant in enumerate(participants)
        )
        dataset = RawWindowDataset(
            np.zeros((29, 8, 16), dtype=np.float32),
            identities,
        )
        channels = (
            "RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ",
        )
        input_spec = ModelInputSpec(
            "raw",
            n_channels=8,
            n_classes=3,
            channel_schema=channels,
        )
        member_seeds = (50042, 60042, 70042, 80042, 90042)
        model_config = {
            "model_id": "InceptionTimeFullFiveMemberEnsemble",
            "comparison_only": True,
            "member_seeds": member_seeds,
            "dropout": 0.2,
            "kernel_sizes": (39, 19, 9),
            "dilation": 1,
        }
        model_config["architecture_parameters"] = materialize_architecture_parameters(
            model_config,
            input_spec,
        )
        training_config = TrainingConfig(seed=50042)
        scope = FullCohortRefitScope(
            participants,
            registry_hash="c" * 64,
            config_hash="a" * 64,
            oof_evidence_hash="b" * 64,
        ).bind_training_dataset(dataset)
        run_provenance = {
            "architecture_parameters": model_config["architecture_parameters"],
            "input_channels_order": channels,
            "sampling_rate_hz": 100.0,
            "window_plan": {"representation_mode": "raw"},
            "hop_plan": {"hop_s": 2.5},
            "normalization": {"scope": "all29_final_refit"},
            "padding_mask": {"padding": "none"},
            "feature_schema_hash": "f" * 64,
            "sqi_routing": {"mode": "off"},
            "loss": training_config.loss,
            "class_weighting": training_config.class_weighting,
            "sampler": training_config.sampler,
            "epoch_rule": {"rule": "fixed_epoch", "fixed_epochs": 10},
            "optimizer": training_config.optimizer,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "dropout": 0.2,
            "label_smoothing": training_config.label_smoothing,
            "gradient_clipping": {"enabled": False, "max_norm": None},
            "seed_policy": "final_refit_five_member_seeds",
            "random_seeds": member_seeds,
            "fold_hash": scope.fold_hash,
            "aggregation": {
                "balance_line": training_config.expected_aggregation_rule,
            },
            "calibration": {"fit_scope": "all29_final_refit"},
        }
        binding = materialize_final_refit_binding(
            resolved_model_config=model_config,
            input_spec=input_spec,
            training_config=training_config,
            frozen_run_provenance=run_provenance,
            config_hash="a" * 64,
            registry_hash="c" * 64,
            source_snapshot_hash="d" * 64,
            manual_selection_hash="e" * 64,
            oof_evidence_hash="b" * 64,
        )
        plan = FinalRefitPlan(
            purpose="ensemble_contract",
            config_hash="a" * 64,
            model_id="InceptionTimeFullFiveMemberEnsemble",
            participant_ids=participants,
            training_seeds=member_seeds,
            fixed_epochs=10,
            epoch_rule="fixed_epoch",
            model_family="deep",
            oof_evidence_hash="b" * 64,
            model_kind="five_member_ensemble",
            registry_hash="c" * 64,
            source_snapshot_hash="d" * 64,
            manual_selection_hash="e" * 64,
            resolved_model_config_hash=binding.resolved_model_config_hash,
            architecture_parameters_hash=binding.architecture_parameters_hash,
            input_schema_hash=binding.input_schema_hash,
            training_config_hash=binding.training_config_hash,
            frozen_run_provenance_hash=binding.frozen_run_provenance_hash,
            representation_mode="raw",
        )
        fake_model = SimpleNamespace(
            model_id="inception_full_five_member_ensemble"
        )
        result = TrainingResult(
            model=fake_model,
            selected_epoch=10,
            provenance=FittedObjectProvenance(
                object_type="InceptionTimeFullFiveMemberEnsemble",
                fitted_participant_ids=participants,
                outer_membership_hash=scope.membership_hash,
                registry_hash="c" * 64,
                fold_hash=scope.fold_hash,
                epoch_rule="fixed_epoch",
                selected_epoch=10,
                state_hash="9" * 64,
                dataset_binding_hash=scope.train_dataset_hash or "",
                training_balance=training_config.training_balance,
                expected_aggregation_rule=training_config.expected_aggregation_rule,
                epoch_profile=training_config.epoch_profile,
                execution_mode="formal",
                training_seed=50042,
                member_training_seeds=member_seeds,
                member_state_hashes=tuple(str(index) * 64 for index in range(1, 6)),
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "ensemble final refit orchestration seed must be 50042",
        ):
            _execute_prepared_full_cohort_refit(
                plan,
                UnifiedTrainer(TrainingConfig(seed=42)),
                dataset,
                registry_hash="c" * 64,
                binding=binding,
                model_factory=lambda: fake_model,
            )
        with (
            patch(
                "ppg_frailty.training.bundle.validate_resolved_architecture"
            ),
            patch.object(UnifiedTrainer, "fit", return_value=result) as fit,
        ):
            execution = _execute_prepared_full_cohort_refit(
                plan,
                UnifiedTrainer(training_config),
                dataset,
                registry_hash="c" * 64,
                binding=binding,
                model_factory=lambda: fake_model,
            )
        fit.assert_called_once()
        self.assertEqual(execution.plan.training_seeds, member_seeds)
        self.assertEqual(execution.result.provenance.training_seed, 50042)
        self.assertEqual(
            execution.result.provenance.member_training_seeds,
            member_seeds,
        )

    def test_verified_executor_exposes_no_caller_injection_boundary(self) -> None:
        from ppg_frailty.experiment import (
            execute_final_refit_from_verified_artifacts,
        )

        parameters = set(
            inspect.signature(execute_final_refit_from_verified_artifacts).parameters
        )
        forbidden = {
            "trainer", "full_dataset", "binding", "model_factory", "estimator",
            "metadata", "transforms", "pipeline_adapter", "golden_inputs",
            "parity_atol",
        }
        self.assertFalse(parameters & forbidden)
        self.assertIn("bundle_directory", parameters)
        with self.assertRaises(FileNotFoundError):
            execute_final_refit_from_verified_artifacts(
                "missing-run",
                "missing-selection",
                comparison_archive="missing-comparison",
                config_path="missing-config",
                bundle_directory="artifacts/final/missing",
            )
        from ppg_frailty.training import execute_full_cohort_refit

        with self.assertRaisesRegex(RuntimeError, "caller_prepared"):
            execute_full_cohort_refit(object(), object(), object())
        with self.assertRaisesRegex(RuntimeError, "caller_supplied"):
            save_final_refit_bundle(object(), object(), metadata={})

    def test_transform_archive_requires_exact_all29_roster(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 29"):
            FrozenRepresentationTransformArchive(
                representation_mode="feature_vector",
                input_schema_hash="1" * 64,
                fitted_on_participant_ids=tuple(f"P{i:02d}" for i in range(28)),
                fitted_artifacts={},
                provenance={"status": "test"},
                source_records_hash="2" * 64,
                dataset_hash="3" * 64,
            )

    def test_feature_vector_adapter_preserves_nan_missingness_but_rejects_inf(self) -> None:
        adapter = build_model_input_adapter(
            "feature_vector",
            input_schema_hash="1" * 64,
        )
        transformed = adapter.transform_record(
            {"x": np.asarray((1.0, np.nan), dtype=np.float64)}
        )
        self.assertTrue(np.isnan(transformed["x"][0, 1]))
        with self.assertRaisesRegex(ValueError, "not infinity"):
            adapter.transform_record(
                {"x": np.asarray((1.0, np.inf), dtype=np.float64)}
            )

    def test_classical_executor_fits_exact_all_29_with_seed42_and_no_epoch(self) -> None:
        from ppg_frailty.models.feature_baselines import FeatureVectorBaseline

        participants = tuple(f"P{index:02d}" for index in range(29))
        identities = tuple(
            SampleIdentity(
                participant_id=participant,
                file_id=f"{participant}_B",
                role="B",
                label=index % 3,
                signal_route="direct",
            )
            for index, participant in enumerate(participants)
        )
        values = np.column_stack(
            (
                np.arange(29, dtype=np.float32),
                np.asarray([index % 3 for index in range(29)], dtype=np.float32),
            )
        )
        dataset = FeatureVectorDataset(values, ("a", "b"), identities)
        estimator = FeatureVectorBaseline(
            "logistic_regression",
            ("a", "b"),
            seed=42,
            class_weight=None,
            logistic_c=1.0,
            logistic_max_iter=5000,
            logistic_solver="lbfgs",
        )
        training_config = TrainingConfig()
        input_spec = ModelInputSpec(
            "feature_vector",
            n_classes=3,
            feature_names=("a", "b"),
        )
        model_config = {
            "model_id": "logistic_regression",
            "seed": 42,
            "class_weight": None,
            "logistic_c": 1.0,
            "logistic_max_iter": 5000,
            "logistic_solver": "lbfgs",
        }
        model_config["architecture_parameters"] = materialize_architecture_parameters(
            model_config,
            input_spec,
        )
        scope = FullCohortRefitScope(
            participants,
            registry_hash="c" * 64,
            config_hash="a" * 64,
            oof_evidence_hash="b" * 64,
        )
        run_provenance = {
            "architecture_parameters": model_config["architecture_parameters"],
            "input_channels_order": ("a", "b"),
            "sampling_rate_hz": 400.0,
            "window_plan": {"representation_mode": "feature_vector"},
            "hop_plan": {"hop_s": 1.0},
            "normalization": {"scope": "all29_final_refit"},
            "padding_mask": {"padding": "none"},
            "feature_schema_hash": "f" * 64,
            "sqi_routing": {"mode": "off"},
            "loss": "not_applicable_estimator_native",
            "class_weighting": training_config.class_weighting,
            "sampler": training_config.sampler,
            "epoch_rule": {"rule": "not_applicable", "fixed_epochs": None},
            "optimizer": "not_applicable_estimator_native",
            "learning_rate": "not_applicable_estimator_native",
            "weight_decay": "not_applicable_estimator_native",
            "dropout": "not_applicable_estimator_native",
            "label_smoothing": "not_applicable_estimator_native",
            "gradient_clipping": {
                "enabled": False,
                "max_norm": None,
                "status": "not_applicable_estimator_native",
            },
            "seed_policy": "final_refit_single_seed_42",
            "random_seeds": (42,),
            "fold_hash": scope.fold_hash,
            "aggregation": {"balance_line": "line_a_equal_files"},
            "calibration": {"fit_scope": "all29_final_refit"},
        }
        binding = materialize_final_refit_binding(
            resolved_model_config=model_config,
            input_spec=input_spec,
            training_config=training_config,
            frozen_run_provenance=run_provenance,
            config_hash="a" * 64,
            registry_hash="c" * 64,
            source_snapshot_hash="d" * 64,
            manual_selection_hash="e" * 64,
            oof_evidence_hash="b" * 64,
        )
        plan = FinalRefitPlan(
            purpose="contract",
            config_hash="a" * 64,
            model_id="LogisticRegressionL2",
            participant_ids=participants,
            training_seeds=(42,),
            fixed_epochs=None,
            epoch_rule="not_applicable",
            model_family="classical_or_rocket",
            oof_evidence_hash="b" * 64,
            model_kind="single_model",
            registry_hash="c" * 64,
            source_snapshot_hash="d" * 64,
            manual_selection_hash="e" * 64,
            resolved_model_config_hash=binding.resolved_model_config_hash,
            architecture_parameters_hash=binding.architecture_parameters_hash,
            input_schema_hash=binding.input_schema_hash,
            training_config_hash=binding.training_config_hash,
            frozen_run_provenance_hash=binding.frozen_run_provenance_hash,
            representation_mode="feature_vector",
        )
        execution = _execute_prepared_full_cohort_refit(
            plan,
            UnifiedTrainer(training_config),
            dataset,
            registry_hash="c" * 64,
            binding=binding,
            estimator=estimator,
        )
        self.assertEqual(execution.result.selected_epoch, None)
        self.assertEqual(
            execution.result.provenance.fitted_participant_ids,
            participants,
        )
        self.assertEqual(execution.result.provenance.training_seed, 42)
        self.assertEqual(execution.scope.scope_kind, "final_refit_all_29")
        self.assertEqual(execution.binding, binding)
        self.assertEqual(len(execution.execution_hash), 64)

        import ppg_frailty.experiment as experiment_module
        from ppg_frailty.provenance import stable_payload_sha256

        source_records = tuple(
            {
                "participant_id": participant,
                "file_id": f"F{index:02d}",
                "route_status": "retained_direct",
            }
            for index, participant in enumerate(participants)
        )
        source_records_hash = stable_payload_sha256(source_records)
        representation_provenance = {
            "feature_vector_estimator_pipeline": {
                "status": "fitted_inside_final_all29_estimator",
                "fitted_on_participant_ids": participants,
            }
        }
        materialized = experiment_module._TrustedFull29Materialization(
            dataset=dataset,
            input_spec=input_spec,
            fitted_objects={},
            representation_provenance=representation_provenance,
            quality_provenance={"mode": "off", "classification_effect": "none"},
            preprocessing_hash="6" * 64,
            feature_hash="f" * 64,
            feature_schema_id="feature_registry_contract_v2",
            feature_contract={"feature_names": ("a", "b")},
            source_records=source_records,
            source_records_hash=source_records_hash,
            dataset_hash=execution.dataset_hash,
            golden_inputs={"x": values[:1]},
        )
        sections = {
            "manifest": {
                "class_id_order": [0, 1, 2],
                "manifest_version": "internal_records_v2",
            },
            "signal": {
                "internal_fs_hz": 400.0,
                "dl_resampling": {"enabled": False, "target_fs_hz": 400.0},
            },
            "windows": {
                "engineering": {"length_s": 10.0, "hop_s": 5.0, "padding": "none"},
                "raw_dl": {"length_s": 5.0, "hop_s": 2.5, "padding": "none"},
                "shared_planner_version": "window_plan_v1",
            },
            "quality": {"mode": "off", "supervised_route_ready": False},
            "artifact": {"reducer": "identity", "reducer_version": "identity_v1"},
            "features": {"registry_id": "feature_vector_thesis_115_v2"},
            "model": {"variant": "final_refit_contract"},
            "training": {"training_balance": "equal_files"},
            "aggregation": {
                "balance_line": "line_a_equal_files",
                "window_to_file": "ordinary_mean",
            },
            "splits": {"registry_id": "fold_contract_v2"},
        }
        config = SimpleNamespace(
            representation_mode="feature_vector",
            sha256="a" * 64,
        )
        config.section = lambda name: sections[name]
        publication = experiment_module._canonical_final_bundle_materialization(
            execution,
            materialized,
            config,
            SimpleNamespace(manifest_hash="manifest"),
            {
                "source_snapshot_sha256": "d" * 64,
                "live_dependency_gate": {"status": "verified"},
                "selection_record_file_sha256": "7" * 64,
            },
            SimpleNamespace(
                repository_root=Path(__file__).resolve().parents[4]
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            loaded = _save_trusted_final_refit_bundle(
                execution,
                Path(temporary) / "final_bundle",
                materialization=publication,
            )
            assert_golden_parity(loaded)
            self.assertEqual(
                loaded.manifest["metadata"]["final_refit_identity"]["execution_hash"],
                execution.execution_hash,
            )
            self.assertEqual(
                loaded.pipeline_adapter.input_schema_hash,
                plan.input_schema_hash,
            )
            (loaded.directory / "unexpected.txt").write_text(
                "tamper", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "unexpected unverified"):
                load_bundle(loaded.directory)

        wrong_estimator = FeatureVectorBaseline(
            "logistic_regression",
            ("a", "b"),
            seed=42,
            class_weight=None,
            logistic_c=1.0,
            logistic_max_iter=1000,
            logistic_solver="lbfgs",
        )
        with self.assertRaisesRegex(ValueError, "architecture_parameters differ"):
            _execute_prepared_full_cohort_refit(
                plan,
                UnifiedTrainer(training_config),
                dataset,
                registry_hash="c" * 64,
                binding=binding,
                estimator=wrong_estimator,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

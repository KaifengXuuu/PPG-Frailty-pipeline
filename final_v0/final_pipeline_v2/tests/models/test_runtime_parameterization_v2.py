"""Runtime-selectable model parameters and seed strategies."""

from __future__ import annotations

import unittest
import copy
import hashlib
from pathlib import Path

import torch

from ppg_frailty.config import (
    canonical_json_bytes,
    load_config,
    load_formal_experiment_catalog,
    validate_config_payload,
)
from ppg_frailty.models.factory import (
    FRAILTY_RAW_CHANNEL_SCHEMA,
    ModelInputSpec,
    create_model,
    materialize_architecture_parameters,
    resolve_seed_policy,
    validate_frozen_model_run_provenance,
)
from ppg_frailty.models.inception import (
    InceptionTimeFiveMemberProbabilityEnsemble,
    InceptionTimeProbabilityEnsemble,
    InceptionTimeSingleNetwork,
)
from ppg_frailty.module_registry import (
    derived_model_ensemble_size,
    list_modules,
    materialize_model_architecture,
    model_factory_contract,
    validate_model_config,
)


def _explicit(config: dict[str, object], spec: ModelInputSpec) -> dict[str, object]:
    payload = dict(config)
    payload["architecture_parameters"] = materialize_architecture_parameters(
        payload, spec
    )
    return payload


class RuntimeModelParameterizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_spec = ModelInputSpec(
            "raw",
            n_channels=8,
            n_classes=3,
            channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA,
        )

    def test_legacy_ensemble_name_accepts_an_arbitrary_unique_roster(self) -> None:
        members = [
            InceptionTimeSingleNetwork(
                2,
                3,
                variant="small",
                out_channels=4,
                bottleneck_channels=3,
                depth=1,
            )
            for _ in range(2)
        ]
        ensemble = InceptionTimeFiveMemberProbabilityEnsemble(
            members, (17, 29)
        ).eval()
        self.assertIsInstance(ensemble, InceptionTimeProbabilityEnsemble)
        with torch.no_grad():
            probabilities = ensemble.member_probabilities(torch.randn(3, 2, 24))
        self.assertEqual(tuple(probabilities.shape), (2, 3, 3))
        torch.testing.assert_close(
            ensemble.average_member_probabilities(probabilities),
            probabilities.mean(dim=0),
        )
        self.assertEqual(
            ensemble.member_provenance(),
            (
                {"member_index": 0, "training_seed": 17},
                {"member_index": 1, "training_seed": 29},
            ),
        )

    def test_runtime_normalization_strategies_are_registered_modules(self) -> None:
        self.assertEqual(
            {row["module_id"] for row in list_modules("normalization")},
            {
                "ppg_per_window_robust",
                "ppg_per_window_standard_zscore",
                "ppg_none",
                "imu_outer_train_robust",
                "imu_outer_train_mean_std",
                "imu_none",
            },
        )

    def test_executable_strategy_families_are_registered(self) -> None:
        expected = {
            "optimizer": {"adam", "adamw", "sgd", "rmsprop"},
            "sampler": {
                "balance_line_weighted_v2",
                "uniform_replacement",
                "exhaustive_shuffle_without_replacement",
                "subject_balanced",
                "class_subject_balanced",
            },
            "loss": {"cross_entropy", "balanced_softmax", "focal_loss"},
            "class_weighting": {
                "inverse_frequency",
                "effective_number",
                "none",
            },
            "class_count_basis": {"participant", "row"},
            "training_balance": {"equal_files", "equal_role_families"},
            "epoch_selection": {"fixed_epoch", "inner_grouped_selection"},
            "quality_mode": {"off", "diagnostics_only", "route"},
            "window_quality_selection": {
                "none",
                "legacy_per_file_top_fraction",
            },
            "shapeformer_discovery_balance": {
                "participant_file_balanced",
                "class_window_balanced",
            },
            "aggregation": {
                "line_a_equal_files",
                "line_b_equal_role_families",
            },
        }
        for family, module_ids in expected.items():
            with self.subTest(family=family):
                rows = list_modules(family)
                self.assertEqual({row["module_id"] for row in rows}, module_ids)
                self.assertTrue(all(row["implementation"] for row in rows))
        loss_ids = {row["module_id"] for row in list_modules("loss")}
        self.assertNotIn("weighted_ce", loss_ids)
        uniform = next(
            row
            for row in list_modules("sampler")
            if row["module_id"] == "uniform_replacement"
        )
        self.assertEqual(tuple(uniform["runtime_dependencies"]), ("torch",))

    def test_experiment_factory_fields_are_owned_by_the_module_registry(self) -> None:
        compact = model_factory_contract("CompactCNN1D")
        self.assertEqual(compact["machine_model_id"], "compact_cnn")
        self.assertIn("stage_channels", compact["factory_fields"])
        self.assertIn("stage_channels", compact["optional_factory_fields"])
        scalar = model_factory_contract(
            "shapeformer_channel_specific_scalar_distance_ablation"
        )
        self.assertIn("num_pip_ratio", scalar["factory_fields"])
        self.assertIn("hidden_channels", scalar["factory_fields"])
        self.assertIn("hidden_channels", scalar["optional_factory_fields"])
        self.assertEqual(compact["execution_backend"], "torch")
        self.assertEqual(
            model_factory_contract("LogisticRegressionL2")["execution_backend"],
            "estimator",
        )
        for model_id in (
            "InceptionTimeFull",
            "InceptionTimeSmall",
            "InceptionTimeMatrix",
        ):
            fields = set(model_factory_contract(model_id)["factory_fields"])
            self.assertTrue(
                {
                    "pool_size",
                    "out_channels",
                    "bottleneck_channels",
                    "depth",
                    "residual_interval",
                }
                <= fields
            )
        fusion_fields = set(
            model_factory_contract("FileBagFusionInception")["factory_fields"]
        )
        self.assertIn("signal_pool_size", fusion_fields)
        self.assertIn("signal_out_channels", fusion_fields)
        self.assertIn("signal_depth", fusion_fields)
        composer = model_factory_contract("FileBagFusion")
        self.assertEqual(composer["machine_model_id"], "file_bag_fusion")
        self.assertIn("signal_encoder", composer["factory_fields"])
        self.assertIn("signal_encoder", composer["optional_factory_fields"])

    def test_generic_file_bag_fusion_encoder_is_validated_and_hashed(self) -> None:
        section = {
            "model_id": "FileBagFusion",
            "input_channels": 8,
            "input_channels_resolution": "canonical_frailty_raw_8",
            "input_channel_order": list(FRAILTY_RAW_CHANNEL_SCHEMA),
            "n_classes": 3,
            "seed_policy": "outer_repeat",
            "signal_encoder": {
                "model_id": "ShapeFormerEffectSizeFixedV1",
                "input_fs_hz": 100.0,
                "shapelet_length_samples": 16,
                "discovery_stride_samples": 8,
                "hidden_channels": 12,
                "attention_heads": 3,
            },
            "feature_hidden_dim": 7,
            "fusion_hidden_dim": 11,
            "pooling": "attention",
            "dropout": 0.15,
        }
        identity = validate_model_config(section, "fusion")
        self.assertEqual(identity["machine_model_id"], "file_bag_fusion")
        architecture = materialize_model_architecture(section, "fusion")
        self.assertEqual(
            architecture["signal_encoder"]["model_id"],
            "shapeformer_effect_size_fixed_v1",
        )
        self.assertEqual(architecture["signal_feature_dim"], 21)
        self.assertEqual(
            architecture["signal_encoder"]["attention_feedforward_channels"],
            24,
        )
        changed = copy.deepcopy(section)
        changed["signal_encoder"]["hidden_channels"] = 15
        changed_architecture = materialize_model_architecture(changed, "fusion")
        self.assertNotEqual(
            canonical_json_bytes(architecture),
            canonical_json_bytes(changed_architecture),
        )
        changed_feedforward = copy.deepcopy(section)
        changed_feedforward["signal_encoder"][
            "attention_feedforward_channels"
        ] = 29
        changed_feedforward_architecture = materialize_model_architecture(
            changed_feedforward, "fusion"
        )
        self.assertEqual(
            changed_feedforward_architecture["signal_encoder"][
                "attention_feedforward_channels"
            ],
            29,
        )
        self.assertNotEqual(
            canonical_json_bytes(architecture),
            canonical_json_bytes(changed_feedforward_architecture),
        )
        nested_seed = copy.deepcopy(section)
        nested_seed["signal_encoder"]["seed"] = 9
        with self.assertRaisesRegex(ValueError, "fold-owned fields"):
            validate_model_config(nested_seed, "fusion")
        nested_policy = copy.deepcopy(section)
        nested_policy["signal_encoder"]["seed_policy"] = "outer_repeat"
        with self.assertRaisesRegex(ValueError, "fold-owned fields"):
            validate_model_config(nested_policy, "fusion")
        invalid_discovery = copy.deepcopy(section)
        invalid_discovery["signal_encoder"] = {
            "model_id": "ShapeFormerChannelSpecificOSD",
            "discovery_balance": "invented_balance",
        }
        with self.assertRaisesRegex(ValueError, "discovery_balance must be"):
            validate_model_config(invalid_discovery, "fusion")
        estimator_encoder = copy.deepcopy(section)
        estimator_encoder["signal_encoder"] = {
            "model_id": "LogisticRegressionL2"
        }
        with self.assertRaisesRegex(ValueError, "registered raw feature encoders"):
            validate_model_config(estimator_encoder, "fusion")
        defaulted = {
            key: value
            for key, value in section.items()
            if key
            not in {
                "signal_encoder",
                "feature_hidden_dim",
                "fusion_hidden_dim",
                "pooling",
                "dropout",
            }
        }
        validate_model_config(defaulted, "fusion")
        default_architecture = materialize_model_architecture(defaulted, "fusion")
        self.assertEqual(
            default_architecture["signal_encoder"]["model_id"], "compact_cnn"
        )
        self.assertEqual(default_architecture["feature_hidden_dim"], 32)
        self.assertEqual(default_architecture["fusion_hidden_dim"], 64)

    def test_ensemble_rejects_empty_misaligned_duplicate_and_shared_members(self) -> None:
        member = InceptionTimeSingleNetwork(2, 3, variant="small")
        with self.assertRaisesRegex(ValueError, "at least one"):
            InceptionTimeFiveMemberProbabilityEnsemble((), ())
        with self.assertRaisesRegex(ValueError, "aligned"):
            InceptionTimeFiveMemberProbabilityEnsemble((member,), (1, 2))
        with self.assertRaisesRegex(ValueError, "unique"):
            InceptionTimeFiveMemberProbabilityEnsemble(
                (member, InceptionTimeSingleNetwork(2, 3, variant="small")),
                (1, 1),
            )
        with self.assertRaisesRegex(ValueError, "share parameter"):
            InceptionTimeFiveMemberProbabilityEnsemble((member, member), (1, 2))

    def test_factory_builds_two_member_noncanonical_ensemble_without_comparison_flag(self) -> None:
        config = {
            "model_id": "inception_full_five_member_ensemble",
            "seed_policy": "member_roster",
            "member_seeds": (7, 13),
            "variant": "small",
            "dropout": 0.15,
            "kernel_sizes": (11, 5),
            "dilation": 2,
            "pool_size": 5,
            "out_channels": 4,
            "bottleneck_channels": 3,
            "depth": 2,
            "residual_interval": 1,
        }
        model = create_model(_explicit(config, self.raw_spec), self.raw_spec).eval()
        self.assertEqual(len(model.members), 2)
        self.assertEqual(model.member_seeds, (7, 13))
        self.assertEqual(model.training_seeds, (7, 13))
        self.assertEqual(model.members[0].kernel_sizes, (11, 5))
        self.assertEqual(model.members[0].branch_count, 3)
        self.assertEqual(model.members[0].pool_size, 5)
        with torch.no_grad():
            output = model(torch.randn(2, 8, 48))
        self.assertEqual(tuple(output.shape), (2, 3))

        one_member_config = dict(config, member_seeds=(23,))
        one_member = create_model(
            _explicit(one_member_config, self.raw_spec), self.raw_spec
        )
        self.assertEqual(len(one_member.members), 1)
        self.assertEqual(one_member.member_seeds, (23,))

    def test_pipeline_model_contract_binds_size_roster_and_architecture(self) -> None:
        factory_config = {
            "model_id": "inception_full_five_member_ensemble",
            "seed_policy": "member_roster",
            "member_seeds": (7, 13),
            "dropout": 0.2,
            "kernel_sizes": (39, 19, 9),
            "dilation": 1,
        }
        section = {
            "model_id": "InceptionTimeFullFiveMemberEnsemble",
            "variant": "full_probability_ensemble",
            "input_channels": 8,
            "input_channels_resolution": "canonical_frailty_raw_8",
            "input_channel_order": list(FRAILTY_RAW_CHANNEL_SCHEMA),
            "n_classes": 3,
            "ensemble_size": 2,
            "member_seeds": [7, 13],
            "seed_policy": "member_roster",
            "mask_aware_pooling": True,
            "dropout": 0.2,
            "kernel_sizes": [39, 19, 9],
            "dilation": 1,
            "architecture_parameters": materialize_architecture_parameters(
                factory_config, self.raw_spec
            ),
        }
        identity = validate_model_config(section, "raw")
        self.assertEqual(
            identity["machine_model_id"],
            "inception_full_five_member_ensemble",
        )
        one_member_section = dict(section)
        one_member_section.update(
            {
                "ensemble_size": 1,
                "member_seeds": [23],
                "architecture_parameters": materialize_architecture_parameters(
                    dict(factory_config, member_seeds=(23,)), self.raw_spec
                ),
            }
        )
        validate_model_config(one_member_section, "raw")
        mismatched_size = dict(section, ensemble_size=99)
        with self.assertRaisesRegex(ValueError, "derived field mismatch"):
            validate_model_config(mismatched_size, "raw")
        derived_size = dict(section)
        derived_size.pop("ensemble_size")
        identity = validate_model_config(derived_size, "raw")
        self.assertEqual(
            identity["machine_model_id"],
            "inception_full_five_member_ensemble",
        )
        self.assertEqual(derived_model_ensemble_size(derived_size), 2)
        derived_architecture = materialize_model_architecture(derived_size, "raw")
        self.assertEqual(derived_architecture["member_count"], 2)
        self.assertEqual(derived_architecture["member_seeds"], [7, 13])
        for field, value in (("member_seeds", [7, 7]),):
            bad = dict(section)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_model_config(bad, "raw")
        legacy_named = dict(section)
        legacy_named["seed_policy"] = "cv_fixed_five_member_seed_roster"
        with self.assertRaisesRegex(ValueError, "denotes five members"):
            validate_model_config(legacy_named, "raw")

    def test_architecture_is_derived_from_top_level_inputs(self) -> None:
        section = {
            "model_id": "CompactCNN1D",
            "input_channels": 8,
            "input_channels_resolution": "canonical_frailty_raw_8",
            "input_channel_order": list(FRAILTY_RAW_CHANNEL_SCHEMA),
            "n_classes": 3,
            "seed_policy": "fixed_explicit",
            "dropout": 0.37,
            "kernel_sizes": [5, 7, 3],
            "dilations": [2, 1, 3],
            "pool_sizes": [2, 3],
            "stage_channels": [12, 20, 28],
            "stage_dropouts": [0.05, 0.25],
            # Legacy derived values may be stale; they are not runtime inputs.
            "architecture_parameters": {
                "model_id": "compact_cnn",
                "representation_mode": "raw",
                "n_classes": 3,
                "classifier_dropout": 0.2,
            },
            "ensemble_size": 9,
        }
        with self.assertRaisesRegex(ValueError, "derived field mismatch"):
            validate_model_config(section, "raw")
        input_section = dict(section)
        input_section.pop("architecture_parameters")
        input_section.pop("ensemble_size")
        validate_model_config(input_section, "raw")
        architecture = materialize_model_architecture(input_section, "raw")
        self.assertEqual(architecture["classifier_dropout"], 0.37)
        self.assertEqual(architecture["stage_channels"], [12, 20, 28])
        self.assertEqual(derived_model_ensemble_size(input_section), 1)
        factory_section = {
            field: section[field]
            for field in (
                "model_id",
                "seed_policy",
                "dropout",
                "kernel_sizes",
                "dilations",
                "pool_sizes",
                "stage_channels",
                "stage_dropouts",
            )
        }
        model = create_model(factory_section, self.raw_spec)
        self.assertEqual(model.stage_channels, (12, 20, 28))
        self.assertEqual(model.classifier_dropout, 0.37)

    def test_effective_config_hash_payload_refreshes_derived_model_metadata(self) -> None:
        root = Path(__file__).resolve().parents[2]
        base = load_config(
            root / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()
        stale = copy.deepcopy(base)
        stale["model"]["dropout"] = 0.37
        with self.assertRaisesRegex(ValueError, "derived field mismatch"):
            validate_config_payload(stale)

        omitted = copy.deepcopy(base)
        omitted["model"]["dropout"] = 0.37
        for field in ("architecture_parameters", "ensemble_size", "variant"):
            omitted["model"].pop(field)
        effective = validate_config_payload(omitted)
        explicit = copy.deepcopy(effective)
        explicit_effective = validate_config_payload(explicit)
        self.assertEqual(
            hashlib.sha256(canonical_json_bytes(effective)).hexdigest(),
            hashlib.sha256(canonical_json_bytes(explicit_effective)).hexdigest(),
        )
        self.assertEqual(effective["model"]["ensemble_size"], 1)
        self.assertEqual(
            effective["model"]["variant"], "reference_not_wang_fcn"
        )
        self.assertEqual(
            effective["model"]["architecture_parameters"]["classifier_dropout"],
            0.37,
        )
        bad_variant = copy.deepcopy(base)
        bad_variant["model"]["variant"] = "stale_free_text"
        with self.assertRaisesRegex(ValueError, "variant derived field mismatch"):
            validate_config_payload(bad_variant)

    def test_legacy_ensemble_annotations_do_not_create_runtime_hash_axes(self) -> None:
        root = Path(__file__).resolve().parents[2]
        base = load_config(
            root / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()
        catalog = load_formal_experiment_catalog(
            root / "configs/formal_experiment_catalog_v2.yaml"
        )
        ensemble = next(
            copy.deepcopy(entry["model"])
            for entry in catalog["entries"]
            if entry["entry_id"] == "inception_full_five_member_ensemble"
        )
        base["model"] = ensemble
        base["output"]["write_member_oof"] = True
        annotated = validate_config_payload(copy.deepcopy(base))
        self.assertNotIn("comparison_only", annotated["model"])
        self.assertNotIn("member_seed_roster_id", annotated["model"])

        omitted = copy.deepcopy(base)
        omitted["model"].pop("comparison_only")
        omitted["model"].pop("member_seed_roster_id")
        effective_omitted = validate_config_payload(omitted)
        self.assertEqual(
            hashlib.sha256(canonical_json_bytes(annotated)).hexdigest(),
            hashlib.sha256(canonical_json_bytes(effective_omitted)).hexdigest(),
        )
        for field, value in (
            ("comparison_only", False),
            ("member_seed_roster_id", "arbitrary_hash_only_label"),
        ):
            invalid = copy.deepcopy(base)
            invalid["model"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, field
            ):
                validate_config_payload(invalid)
        invalid_single = load_config(
            root / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()
        invalid_single["model"]["comparison_only"] = True
        with self.assertRaisesRegex(ValueError, "no legacy ensemble metadata"):
            validate_config_payload(invalid_single)

    def test_temporal_model_and_window_contract_fails_before_runtime(self) -> None:
        root = Path(__file__).resolve().parents[2]
        base = load_config(
            root / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()

        padded = copy.deepcopy(base)
        padded["windows"]["raw_dl"].update(
            {
                "padding": "right_zero_pad_short_records",
                "min_valid_fraction": 0.5,
            }
        )
        with self.assertRaisesRegex(ValueError, "does not implement mask-aware"):
            validate_config_payload(padded)

        too_short = copy.deepcopy(base)
        too_short["signal"]["dl_resampling"].update(
            {"enabled": True, "target_fs_hz": 2.0}
        )
        with self.assertRaisesRegex(ValueError, "pool_sizes chain"):
            validate_config_payload(too_short)

        shapeformer_model = {
            "model_id": "ShapeFormerChannelSpecificOSD",
            "input_channels": 8,
            "input_channels_resolution": "canonical_frailty_raw_8",
            "input_channel_order": list(FRAILTY_RAW_CHANNEL_SCHEMA),
            "n_classes": 3,
            "seed_policy": "fixed_explicit",
            "mask_aware_pooling": True,
            "input_fs_hz": 400.0,
            "sequence_length_samples": 2000,
        }
        valid_shapeformer = copy.deepcopy(base)
        valid_shapeformer["model"] = shapeformer_model
        validate_config_payload(valid_shapeformer)

        mismatched_rate = copy.deepcopy(valid_shapeformer)
        mismatched_rate["model"].pop("architecture_parameters", None)
        mismatched_rate["model"]["input_fs_hz"] = 200.0
        with self.assertRaisesRegex(ValueError, "input_fs_hz must match"):
            validate_config_payload(mismatched_rate)

        mismatched_length = copy.deepcopy(valid_shapeformer)
        mismatched_length["model"].pop("architecture_parameters", None)
        mismatched_length["model"]["sequence_length_samples"] = 1999
        with self.assertRaisesRegex(ValueError, "sequence_length_samples"):
            validate_config_payload(mismatched_length)

        aligned_resampling = copy.deepcopy(valid_shapeformer)
        aligned_resampling["model"].pop("architecture_parameters", None)
        aligned_resampling["model"]["input_fs_hz"] = 128.0
        aligned_resampling["model"]["sequence_length_samples"] = 640
        aligned_resampling["signal"]["dl_resampling"].update(
            {"enabled": True, "target_fs_hz": 128.0}
        )
        validate_config_payload(aligned_resampling)

        for model_id, field in (
            (
                "ShapeFormerChannelSpecificScalarDistanceAblation",
                "patch_size_samples",
            ),
            ("ShapeFormerEffectSizeFixedV1", "shapelet_length_samples"),
        ):
            invalid = copy.deepcopy(base)
            invalid["model"] = {
                "model_id": model_id,
                "input_channels": 8,
                "input_channels_resolution": "canonical_frailty_raw_8",
                "input_channel_order": list(FRAILTY_RAW_CHANNEL_SCHEMA),
                "n_classes": 3,
                "seed_policy": "fixed_explicit",
                "mask_aware_pooling": True,
                "input_fs_hz": 400.0,
                field: 2001,
            }
            with self.subTest(model_id=model_id), self.assertRaisesRegex(
                ValueError, field
            ):
                validate_config_payload(invalid)

    def test_compact_defaults_and_nondefault_time_scale_are_both_executable(self) -> None:
        default_config = {"model_id": "compact_cnn", "seed": 41}
        default = create_model(
            _explicit(default_config, self.raw_spec), self.raw_spec
        ).eval()
        self.assertEqual(default.kernel_sizes, (9, 9, 7))
        custom_config = {
            "model_id": "compact_cnn",
            "seed_policy": "fixed_explicit",
            "seed": 43,
            "dropout": 0.35,
            "kernel_sizes": (5, 7, 3),
            "dilations": (2, 1, 3),
            "pool_sizes": (2, 3),
            "stage_channels": (12, 20, 28),
            "stage_dropouts": (0.05, 0.25),
        }
        custom = create_model(_explicit(custom_config, self.raw_spec), self.raw_spec).eval()
        self.assertEqual(custom.kernel_sizes, (5, 7, 3))
        self.assertEqual(custom.dilations, (2, 1, 3))
        self.assertEqual(custom.pool_sizes, (2, 3))
        self.assertEqual(custom.stage_channels, (12, 20, 28))
        self.assertEqual(custom.stage_dropouts, (0.05, 0.25))
        self.assertEqual(custom.feature_dim, 28)
        self.assertEqual(custom.classifier.in_features, 28)
        with torch.no_grad():
            self.assertEqual(tuple(default(torch.randn(1, 8, 96)).shape), (1, 3))
            self.assertEqual(tuple(custom(torch.randn(1, 8, 96)).shape), (1, 3))

    def test_feature_baseline_ranges_are_runtime_parameters_with_defaults(self) -> None:
        feature_spec = ModelInputSpec(
            "feature_vector",
            n_channels=0,
            n_classes=3,
            feature_names=("rate", "prv"),
        )
        logistic_config = {
            "model_id": "logistic_regression",
            "seed": 7,
            "logistic_c": 0.37,
            "logistic_max_iter": 321,
            "logistic_solver": "saga",
        }
        logistic = create_model(
            _explicit(logistic_config, feature_spec), feature_spec
        )
        estimator = logistic.pipeline.named_steps["model"]
        self.assertEqual(estimator.C, 0.37)
        self.assertEqual(estimator.max_iter, 321)
        self.assertEqual(estimator.solver, "saga")

        tree_config = {
            "model_id": "extra_trees",
            "seed": 11,
            "extra_trees_n_estimators": 137,
            "extra_trees_n_jobs": -1,
            "extra_trees_max_features": "log2",
            "extra_trees_min_samples_leaf": 0.125,
        }
        trees = create_model(_explicit(tree_config, feature_spec), feature_spec)
        tree_estimator = trees.pipeline.named_steps["model"]
        self.assertEqual(tree_estimator.n_estimators, 137)
        self.assertEqual(tree_estimator.n_jobs, -1)
        self.assertEqual(tree_estimator.max_features, "log2")
        self.assertEqual(tree_estimator.min_samples_leaf, 0.125)

        logistic_default = create_model(
            _explicit({"model_id": "logistic_regression"}, feature_spec),
            feature_spec,
        )
        self.assertEqual(logistic_default.logistic_c, 1.0)
        self.assertEqual(logistic_default.logistic_max_iter, 5000)
        self.assertEqual(logistic_default.logistic_solver, "lbfgs")

    def test_classical_class_weighting_has_one_training_owned_entry(self) -> None:
        feature_spec = ModelInputSpec(
            "feature_vector",
            n_channels=0,
            n_classes=3,
            feature_names=("rate", "prv"),
        )
        with self.assertRaisesRegex(ValueError, "training.class_weighting"):
            materialize_architecture_parameters(
                {
                    "model_id": "logistic_regression",
                    "class_weight": "balanced",
                },
                feature_spec,
            )
        section = {
            "model_id": "ExtraTrees",
            "input_channels": 0,
            "input_channels_resolution": "not_applicable_feature_vector",
            "n_classes": 3,
            "ensemble_size": 1,
            "seed_policy": "fixed_explicit",
            "class_weight": {0: 1.0, 1: 2.0, 2: 3.0},
        }
        with self.assertRaisesRegex(ValueError, "training.class_weighting"):
            validate_model_config(section, "feature_vector")

    def test_registry_accepts_nondefault_classical_and_shapeformer_ranges(self) -> None:
        base = {
            "input_channels": 0,
            "input_channels_resolution": "not_applicable_feature_vector",
            "n_classes": 3,
            "ensemble_size": 1,
            "seed_policy": "fixed_explicit",
        }
        validate_model_config(
            {
                **base,
                "model_id": "LogisticRegressionL2",
                "logistic_c": 0.37,
                "logistic_max_iter": 321,
                "logistic_solver": "saga",
            },
            "feature_vector",
        )
        validate_model_config(
            {
                **base,
                "model_id": "ExtraTrees",
                "extra_trees_n_estimators": 137,
                "extra_trees_n_jobs": -1,
                "extra_trees_max_features": "log2",
                "extra_trees_min_samples_leaf": 0.125,
            },
            "feature_vector",
        )
        raw_base = {
            **base,
            "input_channels": 8,
            "input_channels_resolution": "canonical_frailty_raw_8",
            "input_channel_order": list(FRAILTY_RAW_CHANNEL_SCHEMA),
            "mask_aware_pooling": True,
        }
        validate_model_config(
            {
                **raw_base,
                "model_id": "ShapeFormerChannelSpecificOSD",
                "num_pip_ratio": 0.35,
                "shapelets_per_class": 2,
                "max_discovery_windows": 24,
                "position_search_neighbourhood_samples": 17,
                "sequence_length_samples": 640,
                "local_kernel_width_samples": 5,
                "local_embedding_channels": 12,
                "shape_embedding_channels": 20,
                "attention_feedforward_channels": 31,
                "attention_heads": 4,
                "attention_query_chunk_size": 19,
                "distance_position_chunk_size": 23,
                "dropout": 0.17,
                "complexity_norm": 700.0,
                "max_complexity_ratio": 2.5,
            },
            "raw",
        )
        scalar = {
            **raw_base,
            "model_id": "ShapeFormerChannelSpecificScalarDistanceAblation",
            "hidden_channels": 12,
            "attention_heads": 3,
            "patch_size_samples": 5,
        }
        identity = validate_model_config(scalar, "raw")
        self.assertEqual(
            identity["machine_model_id"],
            "shapeformer_channel_specific_scalar_distance_ablation",
        )

        invalid_logistic = {
            **base,
            "model_id": "LogisticRegressionL2",
            "logistic_c": -0.1,
        }
        with self.assertRaisesRegex(ValueError, "logistic_c"):
            validate_model_config(invalid_logistic, "feature_vector")

    def test_inception_structure_and_outer_repeat_seed_are_runtime_values(self) -> None:
        config = {
            "model_id": "inception_full",
            "seed_policy": "outer_repeat",
            "outer_repeat_seed": 314,
            "seed": 314,
            "dropout": 0.05,
            "kernel_sizes": (13, 7),
            "dilation": 3,
            "pool_size": 7,
            "out_channels": 5,
            "bottleneck_channels": 2,
            "depth": 4,
            "residual_interval": 2,
        }
        model = create_model(_explicit(config, self.raw_spec), self.raw_spec).eval()
        self.assertEqual(model.out_channels, 5)
        self.assertEqual(model.bottleneck_channels, 2)
        self.assertEqual(model.depth, 4)
        self.assertEqual(model.residual_interval, 2)
        self.assertEqual(model.training_seeds, (314,))
        self.assertEqual(model.feature_dim, 15)
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.randn(2, 8, 64)).shape), (2, 3))

        bad = dict(config)
        bad["seed"] = 315
        with self.assertRaisesRegex(ValueError, "differs"):
            create_model(_explicit(bad, self.raw_spec), self.raw_spec)

    def test_fusion_inception_structure_is_runtime_selectable(self) -> None:
        spec = ModelInputSpec(
            "fusion",
            n_channels=8,
            n_classes=3,
            n_file_features=6,
            channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA,
        )
        config = {
            "model_id": "fusion_inception",
            "seed": 17,
            "signal_variant": "small",
            "signal_dropout": 0.1,
            "signal_kernel_sizes": (13, 7),
            "signal_dilation": 2,
            "signal_pool_size": 5,
            "signal_out_channels": 7,
            "signal_bottleneck_channels": 3,
            "signal_depth": 4,
            "signal_residual_interval": 2,
            "feature_hidden_dim": 11,
            "fusion_hidden_dim": 13,
            "pooling": "mean",
            "dropout": 0.2,
        }
        model = create_model(config, spec)
        self.assertEqual(model.signal_encoder.pool_size, 5)
        self.assertEqual(model.signal_encoder.out_channels, 7)
        self.assertEqual(model.signal_encoder.bottleneck_channels, 3)
        self.assertEqual(model.signal_encoder.depth, 4)
        self.assertEqual(model.signal_encoder.residual_interval, 2)
        self.assertEqual(model.signal_feature_dim, 21)

    def test_invalid_constructor_ranges_fail_closed(self) -> None:
        for kwargs, pattern in (
            ({"dropout": float("nan")}, "dropout"),
            ({"kernel_sizes": (8, 3)}, "positive odd"),
            ({"pool_size": 4}, "positive odd"),
            ({"depth": 0}, "positive integer"),
            ({"residual_interval": 0}, "positive integer"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, pattern):
                    InceptionTimeSingleNetwork(2, 3, variant="small", **kwargs)
        with self.assertRaisesRegex(ValueError, "finite integer"):
            resolve_seed_policy("fixed_explicit", seed=1.5)
        for seed in (-1, 2**32):
            with self.subTest(seed=seed), self.assertRaisesRegex(
                ValueError, "uint32"
            ):
                resolve_seed_policy("fixed_explicit", seed=seed)
        with self.assertRaisesRegex(ValueError, "ordered list or tuple"):
            resolve_seed_policy("member_roster", member_seeds={1, 2})

    def test_seed_policy_provenance_checks_declared_execution_shape(self) -> None:
        base = {
            name: "value"
            for name in (
                "architecture_parameters",
                "sampling_rate_hz",
                "window_plan",
                "hop_plan",
                "normalization",
                "padding_mask",
                "feature_schema_hash",
                "sqi_routing",
                "loss",
                "class_weighting",
                "sampler",
                "epoch_rule",
                "optimizer",
                "learning_rate",
                "weight_decay",
                "dropout",
                "label_smoothing",
                "gradient_clipping",
                "fold_hash",
                "aggregation",
                "calibration",
            )
        }
        base["input_channels_order"] = ("a", "b")
        fixed = validate_frozen_model_run_provenance(
            {**base, "seed_policy": "fixed_explicit", "random_seeds": (71,)}
        )
        self.assertEqual(fixed["random_seeds"], (71,))
        roster = validate_frozen_model_run_provenance(
            {
                **base,
                "seed_policy": "member_roster",
                "random_seeds": (7, 13, 29),
            }
        )
        self.assertEqual(roster["random_seeds"], (7, 13, 29))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_frozen_model_run_provenance(
                {
                    **base,
                    "seed_policy": "fixed_explicit",
                    "random_seeds": (7, 13),
                }
            )
        with self.assertRaisesRegex(ValueError, "declares seed 50042"):
            validate_frozen_model_run_provenance(
                {
                    **base,
                    "seed_policy": "cv_fixed_member0_seed_50042_comparator",
                    "random_seeds": (7,),
                }
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

"""Composable feature-group registry and schema propagation tests."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import ppg_frailty.experiment as experiment
from ppg_frailty.config import load_config, validate_config_payload
from ppg_frailty.contracts import EngineeringFeatureSequence, SignalRoute
from ppg_frailty.features import (
    ENGINEERING_SCHEMA_VERSION,
    FEATURE_GROUP_ORDER,
    ORDERED_MATRIX_SCHEMA_VERSION,
    EngineeringExtraction,
    build_feature_vector,
    build_ordered_matrix,
    canonicalize_feature_groups,
    default_registry,
    engineering_feature_names,
    fit_fold_feature_transform,
    fit_fold_feature_vector_transform,
    legacy_feature_groups,
    ordered_matrix_schema_version,
    registry_for_groups,
    transform_engineering,
    transform_feature_vector_batch,
)
from ppg_frailty.module_registry import list_modules
from ppg_frailty.representations import (
    validate_feature_matrix,
    validate_feature_vector,
)
from ppg_frailty.study.expand import expand_study, parse_study_plan


PIPELINE_ROOT = Path(__file__).resolve().parents[2]


def _engineering() -> EngineeringExtraction:
    names = engineering_feature_names()
    values = np.vstack(
        (
            np.linspace(0.0, 1.0, len(names)),
            np.linspace(1.0, 2.0, len(names)),
        )
    )
    return EngineeringExtraction(
        sequence=EngineeringFeatureSequence(
            values=values,
            start_samples=np.asarray((0, 2_000), dtype=np.int64),
            valid_row_mask=np.ones(2, dtype=bool),
            channel_schema=names,
            schema_version=ENGINEERING_SCHEMA_VERSION,
        ),
        value_validity=np.ones(values.shape, dtype=bool),
        route=SignalRoute.DIRECT,
        reasons=(),
    )


class FeatureGroupProfileTests(unittest.TestCase):
    def test_full_dimensions_groups_and_legacy_migrations_are_explicit(self) -> None:
        registry = default_registry()
        self.assertEqual(len(registry.names), 282)
        self.assertEqual(registry.schema_version, "feature_vector_282_v3")
        self.assertEqual(
            ORDERED_MATRIX_SCHEMA_VERSION,
            ordered_matrix_schema_version(32, registry),
        )
        self.assertEqual(
            canonicalize_feature_groups(
                ["morphology", "PPI", "morphology", "hrv_time"]
            ),
            ("ppi_basic_rate", "hrv_time_domain", "morphology"),
        )
        self.assertEqual(legacy_feature_groups("PPI"), ("ppi_basic_rate",))
        self.assertEqual(
            legacy_feature_groups("HRV"),
            (
                "ppi_basic_rate",
                "hrv_time_domain",
                "hrv_spectral",
                "hrv_nonlinear",
            ),
        )
        self.assertEqual(
            legacy_feature_groups("morphology"),
            ("morphology", "dual_optical"),
        )
        self.assertEqual(
            legacy_feature_groups("morphology_ppi_hrv"),
            (
                "ppi_basic_rate",
                "hrv_time_domain",
                "hrv_spectral",
                "hrv_nonlinear",
                "morphology",
                "dual_optical",
            ),
        )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            registry_for_groups([])
        with self.assertRaisesRegex(ValueError, "unknown feature group"):
            registry_for_groups(["invented"])

    def test_config_derives_registry_and_schema_from_enabled_groups(self) -> None:
        source = load_config(
            PIPELINE_ROOT / "configs" / "reference_static_feature_vector_v2.yaml"
        )
        payload = source.to_dict()
        payload["features"]["enabled_groups"] = [
            "morphology",
            "ppi",
            "morphology",
        ]
        payload["features"]["registry_id"] = "forged"
        payload["features"]["file_vector_schema"] = "forged"
        payload["features"]["matrix_schema"] = "forged"
        resolved = validate_config_payload(copy.deepcopy(payload))
        registry = registry_for_groups(("ppi_basic_rate", "morphology"))
        self.assertEqual(
            resolved["features"]["enabled_groups"],
            ["ppi_basic_rate", "morphology"],
        )
        self.assertEqual(resolved["features"]["registry_id"], registry.schema_version)
        self.assertEqual(
            resolved["features"]["file_vector_schema"], registry.schema_version
        )
        self.assertEqual(
            resolved["features"]["matrix_schema"],
            ordered_matrix_schema_version(32, registry),
        )
        self.assertNotEqual(resolved["features"]["registry_id"], "forged")

        raw = load_config(
            PIPELINE_ROOT / "configs" / "reference_static_role_aware_v2.yaml"
        ).to_dict()
        raw["features"]["enabled_groups"] = ["ppi_basic_rate"]
        with self.assertRaisesRegex(ValueError, "not consumed by raw"):
            validate_config_payload(raw)

        fusion = load_config(
            PIPELINE_ROOT / "configs" / "reference_static_fusion_v2.yaml"
        ).to_dict()
        fusion["features"]["matrix_k"] = 7
        with self.assertRaisesRegex(ValueError, "only for feature_matrix"):
            validate_config_payload(fusion)

    def test_prv_controls_require_a_predictor_consumer(self) -> None:
        source = load_config(
            PIPELINE_ROOT / "configs" / "reference_static_feature_vector_v2.yaml"
        ).to_dict()
        cases = (
            ("rate_prv_min_duration_s", 7.0, "ppi_basic_rate"),
            ("time_prv_min_duration_s", 45.0, "hrv_time_domain"),
            ("spectral_prv_min_duration_s", 240.0, "hrv_spectral"),
            ("sample_entropy", {"m": 3}, "hrv_nonlinear"),
        )
        for field, value, active_group in cases:
            with self.subTest(field=field, status="inactive"):
                payload = copy.deepcopy(source)
                payload["features"]["enabled_groups"] = ["morphology"]
                payload["features"][field] = value
                with self.assertRaisesRegex(ValueError, "not consumed"):
                    validate_config_payload(payload)
            with self.subTest(field=field, status="active"):
                payload = copy.deepcopy(source)
                payload["features"]["enabled_groups"] = [active_group]
                payload["features"][field] = value
                resolved = validate_config_payload(payload)
                if field == "sample_entropy":
                    self.assertEqual(resolved["features"][field]["m"], 3)
                else:
                    self.assertEqual(resolved["features"][field], value)

        nonlinear_time = copy.deepcopy(source)
        nonlinear_time["features"]["enabled_groups"] = ["hrv_nonlinear"]
        nonlinear_time["features"]["time_prv_min_intervals"] = 24
        resolved = validate_config_payload(nonlinear_time)
        self.assertEqual(resolved["features"]["time_prv_min_intervals"], 24)

    def test_raw_rejects_prv_controls_and_comparison_metadata_cannot_forge_hash(self) -> None:
        raw = load_config(
            PIPELINE_ROOT / "configs" / "reference_static_role_aware_v2.yaml"
        ).to_dict()
        raw["features"]["rate_prv_min_duration_s"] = 7.0
        with self.assertRaisesRegex(ValueError, "raw representation"):
            validate_config_payload(raw)

        vector = load_config(
            PIPELINE_ROOT / "configs" / "reference_static_feature_vector_v2.yaml"
        ).to_dict()
        vector["features"]["prv_library_comparison_scope"] = "invented_hash_only_scope"
        with self.assertRaisesRegex(ValueError, "fixed comparison provenance"):
            validate_config_payload(vector)

    def test_one_registry_drives_vector_transform_fusion_matrix_and_dataset(self) -> None:
        registry = registry_for_groups(("ppi_basic_rate", "morphology"))
        self.assertEqual(len(registry.names), 25)
        vectors = tuple(
            build_feature_vector(
                {
                    "prv.ppi_mean_s": value,
                    "morphology.amplitude_median": value * 2.0,
                },
                feature_validity={
                    "prv.ppi_mean_s": True,
                    "morphology.amplitude_median": True,
                },
                provenance={"route": SignalRoute.DIRECT.value},
                registry=registry,
            )
            for value in (1.0, 3.0)
        )
        for vector in vectors:
            self.assertIs(validate_feature_vector(vector), vector)
            self.assertEqual(vector.feature_names, registry.names)

        vector_transform = fit_fold_feature_vector_transform(
            vectors,
            ("p1", "p2"),
            fitted_on_participant_ids=("p1", "p2"),
            outer_train_participant_ids=("p1", "p2"),
            outer_oof_participant_ids=("p3",),
        )
        batch = transform_feature_vector_batch(vectors, vector_transform)
        self.assertEqual(batch.fusion_tensor.shape, (2, 50))
        self.assertIn("values_plus_validity_50", batch.schema_version)

        engineering = _engineering()
        engineering_transform = fit_fold_feature_transform(
            (engineering,),
            fitted_on_participant_ids=("p1",),
            outer_train_participant_ids=("p1",),
            outer_oof_participant_ids=("p3",),
        )
        matrix = build_ordered_matrix(
            transform_engineering(engineering, engineering_transform),
            context=batch.contexts[0],
            provenance={"route": SignalRoute.DIRECT.value},
            k=7,
        )
        self.assertIs(validate_feature_matrix(matrix), matrix)
        self.assertEqual(matrix.values.shape, (2 * (115 + 25), 7))
        self.assertEqual(
            matrix.schema_version,
            ordered_matrix_schema_version(7, registry),
        )

        states = []
        for participant, vector, label in zip(
            ("p1", "p2"), vectors, (0, 1)
        ):
            state = experiment._RuntimeRecord(
                row=SimpleNamespace(
                    participant_id=participant,
                    record_id=f"{participant}_B",
                    role="B",
                    class_id=label,
                ),
                retained=True,
                route=SignalRoute.DIRECT,
            )
            state.vector = vector
            states.append(state)
        dataset = experiment._materialize_representation_dataset(
            states, ("p1", "p2"), "feature_vector"
        )
        self.assertEqual(dataset.values.shape, (2, 25))
        self.assertEqual(dataset.feature_names, registry.names)

    def test_module_catalog_lists_each_composable_feature_group(self) -> None:
        descriptors = list_modules("feature_group")
        self.assertEqual(
            {row["module_id"] for row in descriptors}, set(FEATURE_GROUP_ORDER)
        )
        self.assertTrue(
            all(row["scientific_status"] == "runtime_selectable_composable_group"
                for row in descriptors)
        )

    def test_study_axis_rederives_schema_without_hidden_changed_fields(self) -> None:
        plan = parse_study_plan(
            {
                "schema_version": "ppg_frailty.study_plan.v2",
                "study": {
                    "study_id": "feature_groups_smoke_v2",
                    "kind": "ablation",
                    "purpose": "Verify executable group selection only.",
                    "flow_position": "unit test",
                    "decision_role": "ablation",
                    "reference_case_id": None,
                    "thesis_sections": [],
                },
                "base_config": "configs/reference_static_feature_vector_v2.yaml",
                "axes": [
                    {
                        "path": "features.enabled_groups",
                        "values": [
                            ["morphology", "ppi"],
                            list(FEATURE_GROUP_ORDER),
                        ],
                        "reference": list(FEATURE_GROUP_ORDER),
                    }
                ],
                "execution": {
                    "repeats": [0],
                    "folds": [0],
                    "jobs": 1,
                    "parallel_level": "cases",
                    "continue_on_error": True,
                    "allow_parallel_deep": False,
                    "measure_operational_costs": False,
                },
                "output": {"root": "artifacts/studies"},
                "report": {
                    "top_k": 1,
                    "write_html": False,
                    "write_static_figures": False,
                    "calibration_bins": 5,
                },
            }
        )
        expansion = expand_study(plan, pipeline_root=PIPELINE_ROOT)
        self.assertEqual(len(expansion.cases), 2)
        selected = next(
            case
            for case in expansion.cases
            if case.config["features"]["enabled_groups"]
            == ["ppi_basic_rate", "morphology"]
        )
        features = selected.config["features"]
        self.assertEqual(
            features["enabled_groups"], ["ppi_basic_rate", "morphology"]
        )
        registry = registry_for_groups(features["enabled_groups"])
        self.assertEqual(features["registry_id"], registry.schema_version)
        self.assertEqual(
            features["matrix_schema"],
            ordered_matrix_schema_version(features["matrix_k"], registry),
        )


if __name__ == "__main__":
    unittest.main()

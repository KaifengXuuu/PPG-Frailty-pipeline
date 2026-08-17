"""Schema-only contracts for deterministic sparse formal-catalogue sweeps."""

from __future__ import annotations

import unittest

from ppg_frailty.study import (
    AxisSpec,
    CatalogCaseSpec,
    CatalogSpec,
    FormalProfileSpec,
    SparseSearchSpec,
    StudyInfo,
    StudyPlan,
    catalog_case_spec_from_mapping,
    catalog_cases_from_mapping,
    catalog_spec_from_mapping,
    formal_profile_spec_from_mapping,
    sparse_search_spec_from_mapping,
)


CATALOG_ENTRIES = (
    "compact_cnn",
    "inception_full",
    "inception_small",
    "inception_matrix",
    "rocket_numpy",
    "minirocket_ablation",
    "logistic_regression",
    "rbf_svm",
    "extra_trees",
    "shapeformer_channel_specific_osd",
    "shapeformer_effect_size_fixed_v1",
    "fusion_compact",
    "fusion_inception",
)


def _study(kind: str) -> StudyInfo:
    return StudyInfo(
        study_id=f"{kind}_schema_test",
        kind=kind,
        purpose="Schema-only deterministic orchestration test.",
        flow_position="Before expansion or execution.",
        decision_role=(
            "candidate_comparison" if kind == "catalog_sweep" else "single_run"
        ),
    )


def _catalog_cases() -> tuple[CatalogCaseSpec, ...]:
    groups = {
        "inception_matrix": "feature_matrix",
        "rocket_numpy": "feature_matrix",
        "minirocket_ablation": "feature_matrix",
        "logistic_regression": "feature_vector",
        "rbf_svm": "feature_vector",
        "extra_trees": "feature_vector",
        "fusion_compact": "fusion",
        "fusion_inception": "fusion",
    }
    cases: list[CatalogCaseSpec] = []
    for index, entry in enumerate(CATALOG_ENTRIES):
        if entry == "compact_cnn":
            overrides = {}
            formal_profile = FormalProfileSpec(
                family="fixed_kernel_samples",
                profile_id="compactcnn1d__fs_100",
            )
        else:
            overrides = {"training.learning_rate": 0.0001 * (index + 1)}
            formal_profile = None
        cases.append(
            CatalogCaseSpec(
                case_id=f"{entry}__screen_{index:02d}",
                catalog_entry=entry,
                screen_profile_id=f"screen_{index:02d}",
                output_group=groups.get(entry, "raw"),
                overrides=overrides,
                rationale=f"Exercise deterministic profile {index}.",
                formal_profile=formal_profile,
            )
        )
    return tuple(cases)


class CatalogSweepSchemaTests(unittest.TestCase):
    def test_catalog_sweep_serializes_explicit_sparse_cases(self) -> None:
        plan = StudyPlan(
            schema_version="ppg_frailty.study_plan.v2",
            study=_study("catalog_sweep"),
            catalog=CatalogSpec(path="configs/formal_experiment_catalog_v2.yaml"),
            search=SparseSearchSpec(
                method="deterministic_sparse_profiles",
                selection_seed=42,
                interpretation="Fixed profiles selected before execution.",
                controlled_factors=("Line B", "EKF", "quality off"),
                notes=("Classical estimators do not consume epochs.",),
            ),
            cases=_catalog_cases(),
        )

        payload = plan.to_dict()

        self.assertEqual(
            tuple(payload),
            (
                "schema_version",
                "study",
                "catalog",
                "search",
                "cases",
                "execution",
                "output",
                "report",
            ),
        )
        self.assertNotIn("base_config", payload)
        self.assertNotIn("axes", payload)
        self.assertEqual(payload["catalog"]["balance_line"], "line_b")
        self.assertEqual(payload["catalog"]["scope"], "ordinary_13")
        self.assertFalse(payload["search"]["runtime_sampling"])
        self.assertEqual(len(payload["cases"]), 13)
        fixed = payload["cases"][0]
        self.assertEqual(fixed["overrides"], {})
        self.assertEqual(
            fixed["formal_profile"],
            {
                "family": "fixed_kernel_samples",
                "profile_id": "compactcnn1d__fs_100",
            },
        )
        self.assertEqual(payload["cases"][1]["formal_profile"], None)

    def test_legacy_study_shape_remains_unchanged(self) -> None:
        plan = StudyPlan(
            schema_version="ppg_frailty.study_plan.v2",
            study=_study("single"),
            base_config="configs/reference_static_role_aware_v2.yaml",
        )

        payload = plan.to_dict()

        self.assertEqual(
            tuple(payload),
            (
                "schema_version",
                "study",
                "base_config",
                "axes",
                "execution",
                "output",
                "report",
            ),
        )
        self.assertEqual(payload["base_config"], plan.base_config)
        self.assertEqual(payload["axes"], [])
        self.assertNotIn("catalog", payload)
        self.assertNotIn("search", payload)
        self.assertNotIn("cases", payload)

    def test_catalog_sweep_rejects_mixed_or_incomplete_modes(self) -> None:
        arguments = {
            "schema_version": "ppg_frailty.study_plan.v2",
            "study": _study("catalog_sweep"),
            "catalog": CatalogSpec(
                path="configs/formal_experiment_catalog_v2.yaml"
            ),
            "search": SparseSearchSpec(
                method="deterministic_sparse_profiles",
                selection_seed=42,
                runtime_sampling=False,
                interpretation="Fixed profiles.",
            ),
            "cases": _catalog_cases(),
        }
        with self.assertRaisesRegex(ValueError, "cannot define base_config"):
            StudyPlan(**arguments, base_config="base.yaml")
        with self.assertRaisesRegex(ValueError, "cannot define Cartesian axes"):
            StudyPlan(
                **arguments,
                axes=(
                    AxisSpec(
                        path="training.learning_rate",
                        values=(0.0003, 0.001),
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "13 distinct"):
            StudyPlan(**{**arguments, "cases": _catalog_cases()[:-1]})
        duplicate = list(_catalog_cases())
        duplicate[1] = CatalogCaseSpec(
            case_id=duplicate[0].case_id,
            catalog_entry=duplicate[1].catalog_entry,
            screen_profile_id=duplicate[1].screen_profile_id,
            output_group=duplicate[1].output_group,
            overrides=duplicate[1].overrides,
            rationale=duplicate[1].rationale,
        )
        with self.assertRaisesRegex(ValueError, "case_id values must be unique"):
            StudyPlan(**{**arguments, "cases": tuple(duplicate)})
        with self.assertRaisesRegex(ValueError, "cannot define catalog"):
            StudyPlan(
                schema_version="ppg_frailty.study_plan.v2",
                study=_study("single"),
                base_config="base.yaml",
                catalog=arguments["catalog"],
            )

    def test_strict_mapping_helpers_reject_typos_and_unsafe_cases(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown=.*typo"):
            catalog_spec_from_mapping(
                {
                    "path": "catalog.yaml",
                    "balance_line": "line_b",
                    "scope": "ordinary_13",
                    "typo": True,
                }
            )
        with self.assertRaisesRegex(ValueError, "runtime_sampling must be false"):
            sparse_search_spec_from_mapping(
                {
                    "method": "deterministic_sparse_profiles",
                    "selection_seed": 42,
                    "runtime_sampling": True,
                    "interpretation": "Invalid runtime draw.",
                }
            )
        with self.assertRaisesRegex(ValueError, "integer"):
            SparseSearchSpec(
                method="deterministic_sparse_profiles",
                selection_seed=42.0,  # type: ignore[arg-type]
                runtime_sampling=False,
                interpretation="Invalid float seed.",
            )
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            sparse_search_spec_from_mapping(
                {
                    "method": "deterministic_sparse_profiles",
                    "selection_seed": 42,
                    "interpretation": "Bad controlled factor.",
                    "controlled_factors": [""],
                }
            )
        with self.assertRaisesRegex(ValueError, "filesystem-safe"):
            catalog_case_spec_from_mapping(
                {
                    "case_id": "../unsafe",
                    "catalog_entry": "compact_cnn",
                    "screen_profile_id": "screen_00",
                    "output_group": "raw",
                    "overrides": {"training.learning_rate": 0.001},
                    "rationale": "Unsafe ID.",
                    "formal_profile": None,
                }
            )
        with self.assertRaisesRegex(ValueError, "output_group"):
            catalog_case_spec_from_mapping(
                {
                    "case_id": "safe_case",
                    "catalog_entry": "compact_cnn",
                    "screen_profile_id": "screen_00",
                    "output_group": "feature",
                    "overrides": {"training.learning_rate": 0.001},
                    "rationale": "Unknown output group.",
                    "formal_profile": None,
                }
            )
        with self.assertRaisesRegex(ValueError, "dotted paths"):
            catalog_case_spec_from_mapping(
                {
                    "case_id": "safe_case",
                    "catalog_entry": "compact_cnn",
                    "screen_profile_id": "screen_00",
                    "output_group": "raw",
                    "overrides": {"training": 10},
                    "rationale": "Invalid override path.",
                    "formal_profile": None,
                }
            )
        with self.assertRaisesRegex(ValueError, "parent/child"):
            catalog_case_spec_from_mapping(
                {
                    "case_id": "safe_case",
                    "catalog_entry": "compact_cnn",
                    "screen_profile_id": "screen_00",
                    "output_group": "raw",
                    "overrides": {
                        "model.architecture_parameters": {},
                        "model.architecture_parameters.dropout": 0.2,
                    },
                    "rationale": "Ambiguous overrides.",
                    "formal_profile": None,
                }
            )
        with self.assertRaisesRegex(ValueError, "overrides or a formal_profile"):
            catalog_case_spec_from_mapping(
                {
                    "case_id": "safe_case",
                    "catalog_entry": "compact_cnn",
                    "screen_profile_id": "screen_00",
                    "output_group": "raw",
                    "overrides": {},
                    "rationale": "Missing profile definition.",
                    "formal_profile": None,
                }
            )
        with self.assertRaisesRegex(ValueError, "key mismatch"):
            formal_profile_spec_from_mapping(
                {
                    "family": "fixed_kernel_samples",
                    "profile_id": "compactcnn1d__fs_100",
                    "unexpected": True,
                }
            )
        with self.assertRaisesRegex(TypeError, "must be a list"):
            catalog_cases_from_mapping({"case_id": "not_a_list"})


if __name__ == "__main__":
    unittest.main()

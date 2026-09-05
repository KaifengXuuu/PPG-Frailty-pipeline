from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from ppg_frailty.catalog import resolved_catalog_payloads
from ppg_frailty.reporting.components import (
    TEST_COMPONENT_COLUMNS,
    TEST_COMPONENT_VIEW_SCHEMAS,
    TOP_MODEL_CONFIGURATION_COLUMNS,
    build_motion_peak_test_component_rows,
    build_pipeline_test_component_rows,
    build_top_model_configuration_rows,
    markdown_test_component_table,
    write_test_component_markdown,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_ROOT = ROOT / "configs/studies/static_line_b_staged_v2"


class ReportingComponentContractTests(unittest.TestCase):
    def test_top_five_configuration_table_is_complete_ranked_and_hash_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = []
            leaderboard = []
            for rank in range(1, 7):
                case_id = f"case-{rank}"
                relative = Path("cases") / case_id / "resolved_config.yaml"
                target = root / relative
                target.parent.mkdir(parents=True)
                target.write_text(
                    yaml.safe_dump(
                        {
                            "manifest": {
                                "path": "manifests/frailty29.csv",
                                "source_manifest_sha256": f"hidden-{rank}",
                            },
                            "roles": ["B", "R1", "S1", "W1"],
                            "representation_mode": "raw",
                            "signal": {
                                "channel_order": ["RED", "IR", "AX", "AY"],
                                "imu": {"gravity_method": f"profile-{rank}"},
                            },
                            "model": {
                                "model_id": "InceptionTimeSmall",
                                "architecture_parameters": {
                                    "module_depth": 3,
                                    "filters_per_branch": 16,
                                },
                            },
                            "training": {"batch_size": 16, "learning_rate": 0.0003},
                            "artifact": {},
                        },
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                cases.append(
                    {"case_id": case_id, "resolved_config_path": relative.as_posix()}
                )
                leaderboard.append({"case_id": case_id, "predictive_rank": rank})

            rows = build_top_model_configuration_rows(
                root,
                {"cases": cases},
                tuple(reversed(leaderboard)),
                top_k=5,
            )

        self.assertEqual({row["predictive_rank"] for row in rows}, set(range(1, 6)))
        self.assertNotIn("case-6", {row["case_id"] for row in rows})
        self.assertEqual(
            {field for field, _label in TOP_MODEL_CONFIGURATION_COLUMNS},
            set(rows[0]),
        )
        by_case_and_path = {
            (row["case_id"], row["parameter_path"]): row["resolved_value"]
            for row in rows
        }
        self.assertEqual(
            by_case_and_path[("case-1", "model.architecture_parameters.module_depth")],
            "3",
        )
        self.assertEqual(
            by_case_and_path[("case-1", "roles")],
            '["B","R1","S1","W1"]',
        )
        serialized = json.dumps(rows, ensure_ascii=False).lower()
        self.assertNotIn("sha256", serialized)
        self.assertNotIn("hidden-", serialized)

    def test_human_component_views_are_complete_and_narrow(self) -> None:
        lossless_fields = {field for field, _label in TEST_COMPONENT_COLUMNS}
        view_fields = {
            field
            for _title, schema in TEST_COMPONENT_VIEW_SCHEMAS
            for field, _label in schema
        }
        self.assertEqual(view_fields, lossless_fields)
        self.assertTrue(
            all(len(schema) <= 8 for _title, schema in TEST_COMPONENT_VIEW_SCHEMAS)
        )

        row = {
            field: f"value-{field}"
            for field, _label in TEST_COMPONENT_COLUMNS
        }
        row["component_role"] = "classifier"
        row["module_id"] = "CompactCNN1D"
        row["execution_state"] = "executed"
        row["reporter_profile_id"] = "multiclass_participant_oof_v1"
        rendered = markdown_test_component_table([row])
        for title, _schema in TEST_COMPONENT_VIEW_SCHEMAS:
            self.assertIn(f"### {title}", rendered)
        markdown_headers = [
            line
            for line in rendered.splitlines()
            if line.startswith("| ") and not line.startswith("| Column")
        ]
        self.assertTrue(markdown_headers)
        self.assertTrue(all(line.count("|") - 1 <= 8 for line in markdown_headers))

    def test_logistic_trainer_reports_classical_fit_not_dl_epochs(self) -> None:
        config = next(
            value
            for value in resolved_catalog_payloads(
                pipeline_root=ROOT,
                line="line_b",
            )
            if value["model"]["model_id"] == "LogisticRegressionL2"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = root / "resolved_configs/logistic.yaml"
            resolved.parent.mkdir()
            resolved.write_text(yaml.safe_dump(config), encoding="utf-8")
            rows = build_pipeline_test_component_rows(
                root,
                {
                    "cases": [
                        {
                            "case_id": "logistic",
                            "resolved_config_path": "resolved_configs/logistic.yaml",
                        }
                    ]
                },
            )

        trainer = next(row for row in rows if row["component_role"] == "trainer")
        self.assertEqual(
            trainer["module_id"],
            "sklearn.linear_model.LogisticRegression",
        )
        self.assertIn(
            '"epoch_rule":"not_applicable_classical_estimator"',
            trainer["fixed_parameters"],
        )
        self.assertIn('"logistic_max_iter":5000', trainer["fixed_parameters"])
        self.assertNotIn('"fixed_epochs"', trainer["fixed_parameters"])

    def test_legacy_bridge_uses_effective_profile_not_catalog_carrier(self) -> None:
        config = {
            "manifest": {
                "path": "manifests/internal.csv",
                "source_dataset_id": "frailty29_named_data",
                "expected_participant_count": 29,
                "expected_record_count": 261,
                "channel_order": ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"],
                "class_name_order": ["pre-frail", "robust", "young"],
            },
            "splits": {"path": "splits/grouped.csv", "registry_id": "grouped-v1"},
            "roles": ["B", "R1"],
            # Deliberately conflicting carrier values. The bridge controls below
            # are the hash-bound values that were actually executed.
            "signal": {"internal_fs_hz": 400.0},
            "windows": {"raw_dl": {"length_s": 5.0, "hop_s": 2.5}},
            "model": {"model_id": "CompactCNN1D"},
            "training": {"optimizer": "adam", "batch_size": 64, "device": "cuda"},
            "evaluation": {"primary_metric": "balanced_accuracy"},
        }
        controls = {
            "ppg_preprocessing": "legacy_detrend_bandpass_0p2_8",
            "imu_preprocessing": "legacy_filtered_axes",
            "target_fs_hz": 64,
            "window_seconds": 15.0,
            "hop_seconds": 3.0,
            "historical_retained_fraction": 0.9,
            "max_windows_per_file": None,
            "allow_short_record_padding": True,
            "normalization": "per_window_all_eight",
            "sampler": "exhaustive_shuffle_without_replacement",
            "class_weighting": "outer_train_window_inverse_frequency",
            "optimizer": "adamw",
            "batch_size": 32,
            "fixed_epochs": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "training_metric_aggregation_rule": "line_a_equal_files",
            "primary_report_aggregation_view": "window_balanced_to_participant",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = root / "resolved_configs/bridge-b0.yaml"
            resolved.parent.mkdir()
            resolved.write_text(yaml.safe_dump(config), encoding="utf-8")
            (root / "study_plan.yaml").write_text(
                yaml.safe_dump(
                    {
                        "legacy_bridge": {
                            "profiles": [
                                {
                                    "catalog_case_id": "bridge-b0",
                                    "profile_id": "B0",
                                    "factor_id": "baseline",
                                    "changed_control_paths": [],
                                    "controls": controls,
                                    "interpretation": "complete legacy baseline",
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            rows = build_pipeline_test_component_rows(
                root,
                {
                    "cases": [
                        {
                            "case_id": "bridge-b0",
                            "resolved_config_path": "resolved_configs/bridge-b0.yaml",
                        }
                    ]
                },
            )

        profile = next(
            row for row in rows
            if row["component_role"] == "legacy_bridge_effective_profile"
        )
        trainer = next(row for row in rows if row["component_role"] == "trainer")
        window = next(row for row in rows if row["component_role"] == "window_planner")
        aggregation = next(row for row in rows if row["component_role"] == "aggregation")
        self.assertIn('"sampling_rate_hz":64', profile["input_data"])
        self.assertIn('"normalization":"per_window_all_eight"', profile["input_data"])
        self.assertIn('"optimizer":"adamw"', trainer["fixed_parameters"])
        self.assertIn('"batch_size":32', trainer["fixed_parameters"])
        self.assertIn('"length_s":15.0', window["fixed_parameters"])
        self.assertIn('"hop_s":3.0', window["fixed_parameters"])
        self.assertEqual(aggregation["module_id"], "window_balanced_to_participant")

    def test_pipeline_rows_name_input_data_and_remove_hashes(self) -> None:
        config = {
            "manifest": {
                "path": "manifests/internal.csv",
                "source_dataset_id": "frailty29_named_data",
                "source_manifest_sha256": "do-not-report",
                "expected_participant_count": 29,
                "expected_record_count": 261,
                "channel_order": ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"],
                "class_name_order": ["pre-frail", "robust", "young"],
            },
            "splits": {"path": "splits/grouped.csv", "registry_id": "grouped-v1"},
            "roles": ["B", "R1"],
            "representation_mode": "raw",
            "signal": {
                "internal_fs_hz": 400.0,
                "ppg_native_unit": "raw_counts",
                "accelerometer_input_unit": "g",
                "gyroscope_input_unit": "deg/s",
                "peak_detector": {"detector_id": "aboy_project_v2"},
                "ppg_filter": {"family": "butterworth_sos", "low_hz": 0.2, "high_hz": 8.0},
                "gap_repair": {"method": "linear_inside_only"},
                "analysis_view": {"direct_source": "x_filter"},
                "imu": {"gravity_method": "calibrated_roll_pitch_ekf"},
                "normalization": {"dl": "all8_per_window_robust"},
                "dl_resampling": {"enabled": False, "target_fs_hz": 400.0},
            },
            "windows": {"shared_planner_version": "v1", "raw_dl": {"length_s": 5.0, "hop_s": 2.5}, "engineering": {"length_s": 10.0, "hop_s": 2.0}},
            "quality": {"mode": "off"},
            "artifact": {"motion_detector_enabled": False, "denoiser_enabled": False, "reducer": "identity", "parameters": {}},
            "features": {"registry_id": "features-v1", "file_vector_schema": "vector-v1"},
            "model": {"model_id": "CompactCNN1D", "input_channel_order": ["RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"]},
            "training": {"optimizer": "adam", "learning_rate": 0.001},
            "aggregation": {"balance_line": "line_b_equal_role_families"},
            "evaluation": {"primary_metric": "balanced_accuracy"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = root / "resolved_configs/case-a.yaml"
            resolved.parent.mkdir()
            resolved.write_text(yaml.safe_dump(config), encoding="utf-8")
            rows = build_pipeline_test_component_rows(
                root,
                {"cases": [{"case_id": "case-a", "resolved_config_path": "resolved_configs/case-a.yaml"}]},
            )
            classifier = next(row for row in rows if row["component_role"] == "classifier")
            identity = next(row for row in rows if row["component_role"] == "denoiser")
            self.assertEqual(classifier["module_id"], "CompactCNN1D")
            self.assertIn("frailty29_named_data", classifier["input_data"])
            self.assertIn("manifests/internal.csv", classifier["input_data"])
            serialized = json.dumps(rows, ensure_ascii=False).lower()
            self.assertNotIn("sha256", serialized)
            self.assertNotIn("do-not-report", serialized)
            self.assertIn('"runtime_reducer_version":"identity_exact_v1"', identity["fixed_parameters"])
            standalone = write_test_component_markdown(root, rows).read_text(encoding="utf-8")
            self.assertIn(markdown_test_component_table(rows), standalone)
            self.assertTrue(standalone.endswith("\n"))
            self.assertFalse(standalone.endswith("\n\n"))

    def test_stage5_lists_every_denoiser_with_resolved_kernel_metadata(self) -> None:
        plan = yaml.safe_load((PLAN_ROOT / "stage5_pre.yaml").read_text(encoding="utf-8"))
        rows = build_motion_peak_test_component_rows(
            plan,
            {
                "study_type": "stage5_pre_motion_ptt",
                "stages": {"ptt_denoiser_benchmark": {"status": "passed"}},
            },
        )
        denoisers = [row for row in rows if row["component_role"] == "denoiser"]
        self.assertEqual(
            {row["module_id"] for row in denoisers},
            set(plan["denoiser_benchmark"]["reducers"]),
        )
        self.assertTrue(
            all(1 <= len(row["algorithm_kernel_description"]) <= 300 for row in denoisers)
        )
        self.assertTrue(all(row["execution_state"] == "executed" for row in denoisers))
        fastica = next(row for row in denoisers if row["module_id"] == "fastica_bss")
        self.assertIn('"max_iter":1000', fastica["fixed_parameters"])
        self.assertIn("FastICA", fastica["algorithm_kernel_description"])


if __name__ == "__main__":
    unittest.main()

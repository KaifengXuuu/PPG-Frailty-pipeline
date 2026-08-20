"""No-training contracts for the advisory-audit Stage 3 Legacy Bridge."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ppg_frailty.study import (
    NullProgressSink,
    StudyRunner,
    default_experiment_executor,
    load_study_plan,
    parse_study_plan,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT
    / "configs"
    / "studies"
    / "static_line_b_staged_v2"
    / "stage3_alter.yaml"
)


def _passing_executor(case, _config, _directory, plan, _sink):
    return {
        "status": "passed",
        "case_id": case.case_id,
        "cell_results": [
            {
                "status": "passed",
                "repeat_index": repeat,
                "fold_index": fold,
            }
            for repeat in plan.execution.repeats
            for fold in plan.execution.folds
        ],
    }


class Stage3AlterBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = load_study_plan(PLAN_PATH)
        bridge = self.plan.legacy_bridge
        if bridge is None:  # pragma: no cover - clearer failure than AttributeError
            self.fail("stage3_alter.yaml must declare legacy_bridge")
        self.bridge = bridge

    def test_source_specification_hash_is_current(self) -> None:
        source = ROOT.parent.parent / self.bridge.source_specification
        self.assertTrue(source.is_file())
        self.assertEqual(
            hashlib.sha256(source.read_bytes()).hexdigest(),
            self.bridge.source_specification_sha256,
        )

    def test_phase0_is_enabled_but_strictly_advisory(self) -> None:
        phase0 = self.bridge.phase0
        self.assertTrue(phase0["enabled"])
        self.assertTrue(phase0["advisory_only"])
        self.assertFalse(phase0["mandatory"])
        self.assertFalse(phase0["affects_training_execution"])
        self.assertNotIn("training_gate_decisions", phase0)
        self.assertNotIn("stop_conditions", phase0)
        self.assertEqual(phase0["manifest_expected_rows"], 261)
        self.assertEqual(phase0["static_expected_record_count"], 145)
        self.assertEqual(
            phase0["required_channel_order"],
            ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"],
        )
        self.assertEqual(
            phase0["training_input_source"],
            "fresh_current_raw_csv_bytes_independent_of_phase0",
        )
        self.assertEqual(phase0["historical_cache_mismatch_training_effect"], "none")
        self.assertIn(
            "optional_phase0_manifest_source_channel_imu_cache_and_split_advisory_auditor",
            self.bridge.required_runtime_capabilities,
        )
        self.assertIn(
            "phase0_status_never_changes_or_blocks_training",
            self.bridge.required_runtime_capabilities,
        )

    def test_budget_and_two_report_orders_are_frozen(self) -> None:
        self.assertEqual(
            self.bridge.budget,
            {
                "case_count": 9,
                "repeat_indices": [0],
                "fold_indices": [0, 1, 2, 3, 4],
                "training_seed": 42,
                "fixed_epochs": 10,
                "early_stopping": False,
                "outer_label_checkpoint_selection": False,
                "fit_count": 45,
                "model_epoch_count": 450,
                "phase0_fit_count": 0,
                "phase0_model_epoch_count": 0,
            },
        )
        self.assertEqual(len(self.plan.cases), 9)
        self.assertEqual(self.bridge.budget["fit_count"], 45)
        self.assertEqual(self.bridge.budget["model_epoch_count"], 450)
        self.assertEqual(
            self.bridge.execution_order[1:],
            (
                "compact_cnn__L7_v2_training_bundle_fixed10",
                "compact_cnn__L5_uniform_replacement_fixed10",
                "compact_cnn__L6_v2_line_b_balance_fixed10",
                "compact_cnn__L4_v2_imu_fold_scaled_fixed10",
                "compact_cnn__L3_v2_imu_window_scaled_fixed10",
                "compact_cnn__L2_legacy400_w5_fixed10",
                "compact_cnn__L1_legacy64_w5_fixed10",
                "compact_cnn__L0_legacy64_w15_fixed10",
            ),
        )
        self.assertEqual(
            self.bridge.adjacent_comparisons,
            tuple(
                f"{self.bridge.numeric_profile_order[level]}->"
                f"{self.bridge.numeric_profile_order[level + 1]}"
                for level in range(1, 8)
            ),
        )

    def test_balance_bundle_and_aggregation_views_are_auditable(self) -> None:
        by_profile = {
            (profile["model_id"], profile["profile_id"]): profile
            for profile in self.bridge.profiles
        }
        l6_contract = set(by_profile[("CompactCNN1D", "L6")]["contract"])
        self.assertIn(
            "balance_line_weighted_v2_equal_B_R_role_family_mass",
            l6_contract,
        )
        self.assertIn(
            "v2_unique_outer_training_participant_count_class_weights",
            l6_contract,
        )
        self.assertIn(
            "legacy_window_balanced_direct_participant_mean",
            self.bridge.aggregation_views,
        )
        self.assertIn(
            "v2_role_balanced_window_file_role_family_participant_mean",
            self.bridge.aggregation_views,
        )
        self.assertNotIn("existing_c0_policy", self.bridge.to_dict())

    def test_schema_round_trip_defaults_audit_off_and_rejects_gate_semantics(self) -> None:
        round_trip = parse_study_plan(self.plan.to_dict())
        self.assertEqual(round_trip.legacy_bridge.to_dict(), self.bridge.to_dict())

        disabled = copy.deepcopy(self.plan.to_dict())
        disabled["legacy_bridge"]["phase0"]["enabled"] = False
        self.assertFalse(parse_study_plan(disabled).legacy_bridge.phase0["enabled"])

        defaulted = copy.deepcopy(self.plan.to_dict())
        del defaulted["legacy_bridge"]["phase0"]["enabled"]
        self.assertFalse(parse_study_plan(defaulted).legacy_bridge.phase0["enabled"])

        invalid_switch = copy.deepcopy(self.plan.to_dict())
        invalid_switch["legacy_bridge"]["phase0"]["enabled"] = "yes"
        with self.assertRaisesRegex(TypeError, "enabled must be boolean"):
            parse_study_plan(invalid_switch)

        gating = copy.deepcopy(self.plan.to_dict())
        gating["legacy_bridge"]["phase0"]["mandatory"] = True
        with self.assertRaisesRegex(ValueError, "must not be mandatory"):
            parse_study_plan(gating)

        affects_training = copy.deepcopy(self.plan.to_dict())
        affects_training["legacy_bridge"]["phase0"][
            "affects_training_execution"
        ] = True
        with self.assertRaisesRegex(ValueError, "must not affect training"):
            parse_study_plan(affects_training)

        budget_drift = copy.deepcopy(self.plan.to_dict())
        budget_drift["legacy_bridge"]["budget"]["fit_count"] = 44
        with self.assertRaisesRegex(ValueError, "fit_count must be 45"):
            parse_study_plan(budget_drift)

    def test_expand_binds_all_nine_profiles_without_canonical_overrides(self) -> None:
        expansion = StudyRunner(pipeline_root=ROOT).expand(self.plan)
        self.assertEqual(len(expansion.cases), 9)
        self.assertEqual(self.plan.execution.device, "cuda")
        self.assertTrue(
            all(case.config["training"]["device"] == "cuda" for case in expansion.cases)
        )
        self.assertNotIn(
            "training.device",
            {row["parameter_path"] for row in expansion.varied_parameters},
        )
        self.assertIn(
            "training.device",
            {row["parameter_path"] for row in expansion.controlled_parameters},
        )
        expected = {
            str(profile["catalog_case_id"]): str(profile["profile_id"])
            for profile in self.bridge.profiles
        }
        self.assertEqual(
            {
                case.case_id: case.changed_values[
                    "study.legacy_bridge_profile"
                ]
                for case in expansion.cases
            },
            expected,
        )
        self.assertTrue(
            all(not case.config.get("legacy_bridge") for case in expansion.cases)
        )
        self.assertTrue(
            all(
                case.config["quality"]["flatline_duration_s"] == 1.0
                for case in expansion.cases
            )
        )

    def test_disabled_phase0_skips_auditor_without_changing_45_cells(self) -> None:
        disabled = copy.deepcopy(self.plan.to_dict())
        disabled["legacy_bridge"]["phase0"]["enabled"] = False
        plan = parse_study_plan(disabled)

        def forbidden_auditor(**_kwargs):
            raise AssertionError("disabled advisory audit must not run")

        with tempfile.TemporaryDirectory() as temporary:
            result = StudyRunner(
                pipeline_root=ROOT,
                executor=_passing_executor,
                phase0_runner=forbidden_auditor,
            ).run(plan, output_root=temporary)
            self.assertEqual(result.status, "passed")
            self.assertEqual(result.planned_case_count, 9)
            self.assertEqual(result.reported_cell_count, 45)
            self.assertFalse((result.output_directory / "phase0_audit.json").exists())

    def test_stop_decision_is_recorded_but_does_not_block_training(self) -> None:
        audit_calls = 0

        def stopped_auditor(**_kwargs):
            nonlocal audit_calls
            audit_calls += 1
            return {
                "schema_version": "ppg_frailty.legacy_v2_phase0_result.v1",
                "decision": "STOP",
                "advisory_checks_passed": False,
                "stop_reasons": ["synthetic_advisory_finding"],
                "limitations": [],
                "outputs": {},
            }

        with tempfile.TemporaryDirectory() as temporary:
            result = StudyRunner(
                pipeline_root=ROOT,
                executor=_passing_executor,
                phase0_runner=stopped_auditor,
            ).run(self.plan, output_root=temporary)
            self.assertEqual(result.status, "passed")
            self.assertEqual(result.reported_cell_count, 45)
            self.assertTrue((result.output_directory / "phase0_audit.json").is_file())
        self.assertEqual(audit_calls, 1)

    def test_audit_exception_and_source_hash_drift_do_not_block_training(self) -> None:
        def broken_auditor(**_kwargs):
            raise RuntimeError("synthetic audit failure")

        altered = replace(
            self.plan,
            legacy_bridge=replace(
                self.bridge,
                source_specification_sha256="0" * 64,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = StudyRunner(
                pipeline_root=ROOT,
                executor=_passing_executor,
                phase0_runner=broken_auditor,
            ).run(altered, output_root=temporary)
            self.assertEqual(result.status, "passed")
            self.assertEqual(result.reported_cell_count, 45)
            self.assertTrue((result.output_directory / "phase0_audit.json").is_file())

    def test_default_executor_dispatches_bridge_without_gate_path(self) -> None:
        case = StudyRunner(pipeline_root=ROOT).expand(self.plan).cases[1]
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "attempt_001"
            attempt.mkdir(parents=True)
            result = {
                "status": "passed",
                "config_id": case.config["config_id"],
                "cell_results": [],
            }
            with patch(
                "ppg_frailty.experiment.run_legacy_bridge_outer_cell",
                return_value=result,
            ) as bridge_cell, patch(
                "ppg_frailty.experiment.run_outer_cell",
            ) as canonical_cell, patch(
                "ppg_frailty.experiment.run_full_experiment",
            ) as canonical_full:
                observed = default_experiment_executor(
                    case,
                    attempt / "resolved_config.yaml",
                    attempt,
                    self.plan,
                    NullProgressSink(),
                )
            self.assertEqual(observed["status"], "passed")
            self.assertEqual(bridge_cell.call_count, 5)
            self.assertFalse(canonical_cell.called)
            self.assertFalse(canonical_full.called)
            for call in bridge_cell.call_args_list:
                self.assertEqual(call.kwargs["profile_id"], "L7")
                self.assertNotIn("phase0_gate_path", call.kwargs)


if __name__ == "__main__":
    unittest.main()

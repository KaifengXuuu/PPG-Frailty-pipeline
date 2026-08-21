"""No-training contracts for the repeated CompactCNN Stage 3 v3 follow-up."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ppg_frailty.legacy_bridge import bridge_profile_from_case
from ppg_frailty import experiment
from ppg_frailty.study import (
    NullProgressSink,
    StudyRunner,
    default_experiment_executor,
    load_study_plan,
    parse_study_plan,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "configs/studies/static_line_b_staged_v2/stage3_v3.yaml"


class Stage3V3FollowupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = load_study_plan(PLAN_PATH)
        self.bridge = self.plan.legacy_bridge
        if self.bridge is None:  # pragma: no cover
            self.fail("stage3_v3.yaml must declare legacy_bridge")

    def test_budget_roster_and_cuda_contract_are_exact(self) -> None:
        self.assertEqual(self.bridge.design, "field_driven_followup_v1")
        self.assertEqual(len(self.plan.cases), 2)
        self.assertEqual(self.plan.execution.repeats, (0, 1, 2, 3, 4))
        self.assertEqual(self.plan.execution.folds, (0, 1, 2, 3, 4))
        self.assertEqual(self.bridge.budget["fit_count"], 50)
        self.assertEqual(self.bridge.budget["model_epoch_count"], 500)
        self.assertEqual(self.plan.execution.device, "cuda")
        self.assertEqual(self.plan.execution.jobs, 1)
        self.assertEqual(
            tuple((row["model_id"], row["profile_id"]) for row in self.bridge.profiles),
            (
                ("CompactCNN1D", "B0_B2"),
                ("CompactCNN1D", "B0_B1_B2"),
            ),
        )

    def test_profiles_are_complete_B0_combinations_and_pair_on_B1(self) -> None:
        baseline = dict(self.bridge.baseline_controls or {})
        b2, b12 = self.bridge.profiles
        self.assertEqual(
            set(b2["changed_control_paths"]),
            {"controls.window_seconds", "controls.hop_seconds"},
        )
        self.assertEqual(
            set(b12["changed_control_paths"]),
            {
                "controls.target_fs_hz",
                "controls.window_seconds",
                "controls.hop_seconds",
            },
        )
        self.assertEqual(
            (b2["controls"]["target_fs_hz"], b12["controls"]["target_fs_hz"]),
            (64, 400),
        )
        for row in (b2, b12):
            self.assertEqual(
                (
                    row["controls"]["window_seconds"],
                    row["controls"]["hop_seconds"],
                ),
                (5.0, 2.5),
            )
            self.assertEqual(row["controls"]["historical_retained_fraction"], 0.9)
            self.assertIsNone(row["controls"]["max_windows_per_file"])
            self.assertEqual(row["controls"]["optimizer"], baseline["optimizer"])
            self.assertEqual(row["controls"]["batch_size"], baseline["batch_size"])
        self.assertIsNone(b2["reference_case_id"])
        self.assertEqual(b12["reference_case_id"], b2["case_id"])

    def test_round_trip_and_runtime_resolution_are_field_driven(self) -> None:
        reparsed = parse_study_plan(self.plan.to_dict())
        self.assertEqual(reparsed.legacy_bridge.to_dict(), self.bridge.to_dict())
        b2 = bridge_profile_from_case(
            self.plan.cases[0].case_id,
            self.bridge.profiles,
            protocol_design=self.bridge.design,
        )
        b12 = bridge_profile_from_case(
            self.plan.cases[1].case_id,
            self.bridge.profiles,
            protocol_design=self.bridge.design,
        )
        self.assertEqual((b2.target_fs_hz, b2.window_seconds, b2.hop_seconds), (64.0, 5.0, 2.5))
        self.assertEqual(
            (b12.target_fs_hz, b12.window_seconds, b12.hop_seconds),
            (400.0, 5.0, 2.5),
        )
        self.assertEqual(b2.protocol_design, "field_driven_followup_v1")
        self.assertNotEqual(b2.training_identity_sha256, b12.training_identity_sha256)

    def test_schema_rejects_noop_budget_and_cpu_drift(self) -> None:
        no_op = copy.deepcopy(self.plan.to_dict())
        no_op["legacy_bridge"]["factor_overrides"]["B0_B2"]["overrides"]["window_seconds"] = 15.0
        with self.assertRaisesRegex(ValueError, "no-op or undeclared"):
            parse_study_plan(no_op)
        budget = copy.deepcopy(self.plan.to_dict())
        budget["legacy_bridge"]["budget"]["fit_count"] = 49
        with self.assertRaisesRegex(ValueError, "budget is inconsistent"):
            parse_study_plan(budget)
        cpu = copy.deepcopy(self.plan.to_dict())
        cpu["execution"]["device"] = "cpu"
        with self.assertRaisesRegex(ValueError, "serial CUDA"):
            parse_study_plan(cpu)

    def test_expansion_and_executor_cover_fifty_cells_without_phase0(self) -> None:
        expansion = StudyRunner(pipeline_root=ROOT).expand(self.plan)
        self.assertEqual(len(expansion.cases), 2)
        self.assertTrue(
            all(
                case.config["training"]["device"] == "cuda"
                for case in expansion.cases
            )
        )
        result = {"status": "passed", "cell_results": []}
        with tempfile.TemporaryDirectory() as temporary, patch(
            "ppg_frailty.experiment.run_legacy_bridge_outer_cell",
            return_value=result,
        ) as bridge_cell:
            observed = default_experiment_executor(
                expansion.cases[1],
                Path(temporary) / "resolved_config.yaml",
                Path(temporary),
                self.plan,
                NullProgressSink(),
            )
        self.assertEqual(observed["status"], "passed")
        self.assertEqual(bridge_cell.call_count, 25)
        for call in bridge_cell.call_args_list:
            self.assertEqual(call.kwargs["profile_id"], "B0_B1_B2")
            self.assertEqual(call.kwargs["protocol_design"], "field_driven_followup_v1")
            self.assertEqual(
                call.kwargs["profile_definition_sha256"],
                call.kwargs["profile_definition"]["controls_sha256"],
            )
            self.assertNotIn("source_specification", call.kwargs)

        def fake_executor(case, _config, _directory, plan, _sink):
            return {
                "status": "passed",
                "case_id": case.case_id,
                "cell_results": [
                    {"status": "passed", "repeat_index": repeat, "fold_index": fold}
                    for repeat in plan.execution.repeats
                    for fold in plan.execution.folds
                ],
            }

        def forbidden_phase0(**_kwargs):
            raise AssertionError("field-driven follow-up must not run Phase 0")

        with tempfile.TemporaryDirectory() as temporary:
            run = StudyRunner(
                pipeline_root=ROOT,
                executor=fake_executor,
                phase0_runner=forbidden_phase0,
            ).run(self.plan, output_root=temporary)
        self.assertEqual(run.status, "passed")
        self.assertEqual((run.planned_case_count, run.passed_case_count), (2, 2))
        self.assertEqual(
            (
                run.planned_cell_count,
                run.reported_cell_count,
                run.passed_cell_count,
            ),
            (50, 50, 50),
        )

    def test_field_driven_entrypoint_allows_repeats_and_cumulative_stays_r0(self) -> None:
        row = self.bridge.profiles[0]
        with patch.object(experiment, "_run_one_outer_cell", return_value="ok") as run:
            observed = experiment.run_legacy_bridge_outer_cell(
                "resolved.yaml",
                4,
                3,
                "output",
                profile_id="B0_B2",
                protocol_design="field_driven_followup_v1",
                profile_definition=row,
                profile_definition_sha256=row["controls_sha256"],
            )
        self.assertEqual(observed, "ok")
        self.assertEqual(run.call_args.kwargs["repeat_index"], 4)
        with self.assertRaisesRegex(ValueError, "frozen to repeat_index=0"):
            experiment.run_legacy_bridge_outer_cell(
                "resolved.yaml",
                1,
                0,
                "output",
                profile_id="L0",
                protocol_design="cumulative_chain_v1",
            )


if __name__ == "__main__":
    unittest.main()

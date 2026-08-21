"""No-training contracts for the two-model Stage 3 centered star."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ppg_frailty.experiment as experiment
from ppg_frailty.study import (
    NullProgressSink,
    StudyRunner,
    default_experiment_executor,
    load_study_plan,
    parse_study_plan,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT / "configs/studies/static_line_b_staged_v2/stage3_star.yaml"
)


class Stage3CenteredStarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = load_study_plan(PLAN_PATH)
        self.bridge = self.plan.legacy_bridge
        if self.bridge is None:  # pragma: no cover
            self.fail("stage3_star.yaml must declare legacy_bridge")

    def test_budget_roster_order_and_gpu_are_exact(self) -> None:
        self.assertEqual(self.bridge.design, "centered_star_v1")
        self.assertEqual(len(self.plan.cases), 16)
        self.assertEqual(len(self.bridge.profiles), 16)
        self.assertEqual(self.bridge.budget["fit_count"], 400)
        self.assertEqual(self.bridge.budget["model_epoch_count"], 4000)
        self.assertEqual(self.plan.execution.repeats, (0, 1, 2, 3, 4))
        self.assertEqual(self.plan.execution.folds, (0, 1, 2, 3, 4))
        self.assertEqual(self.plan.execution.device, "cuda")
        self.assertEqual(self.plan.execution.jobs, 1)
        self.assertFalse(self.plan.execution.allow_parallel_deep)
        self.assertEqual(
            tuple(
                (row["model_id"], row["profile_id"])
                for row in self.bridge.profiles
            ),
            tuple(
                (model, f"B{index}")
                for index in range(8)
                for model in ("CompactCNN1D", "InceptionTimeFull")
            ),
        )

    def test_profiles_are_model_neutral_single_factor_stars(self) -> None:
        baseline = dict(self.bridge.baseline_controls or {})
        by_key = {
            (row["model_id"], row["profile_id"]): row
            for row in self.bridge.profiles
        }
        for index in range(8):
            profile_id = f"B{index}"
            compact = by_key[("CompactCNN1D", profile_id)]
            inception = by_key[("InceptionTimeFull", profile_id)]
            self.assertEqual(compact["controls"], inception["controls"])
            self.assertEqual(compact["controls_sha256"], inception["controls_sha256"])
            observed = {
                f"controls.{key}"
                for key, value in compact["controls"].items()
                if value != baseline[key]
            }
            self.assertEqual(observed, set(compact["changed_control_paths"]))
        b2 = by_key[("CompactCNN1D", "B2")]["controls"]
        self.assertEqual(b2["historical_retained_fraction"], 0.9)
        self.assertIsNone(b2["max_windows_per_file"])
        b7 = by_key[("CompactCNN1D", "B7")]["controls"]
        self.assertEqual(
            {key for key in baseline if baseline[key] != b7[key]},
            {"primary_report_aggregation_view"},
        )

    def test_inline_hash_round_trip_and_fourteen_legal_contrasts(self) -> None:
        controls = {
            f"B{index}": next(
                row["controls"]
                for row in self.bridge.profiles
                if row["model_id"] == "CompactCNN1D"
                and row["profile_id"] == f"B{index}"
            )
            for index in range(8)
        }
        digest = hashlib.sha256(
            json.dumps(
                controls,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(digest, self.bridge.source_specification_sha256)
        self.assertEqual(len(self.bridge.centered_comparisons), 14)
        self.assertTrue(
            all(row["profile_id"] != "B0" for row in self.bridge.centered_comparisons)
        )
        self.assertEqual(
            parse_study_plan(self.plan.to_dict()).legacy_bridge.to_dict(),
            self.bridge.to_dict(),
        )

    def test_schema_rejects_multifactor_and_fit_budget_drift(self) -> None:
        multifactor = copy.deepcopy(self.plan.to_dict())
        multifactor["legacy_bridge"]["factor_overrides"]["B1"]["overrides"][
            "normalization"
        ] = "ppg_window_imu_outer_train_fold"
        with self.assertRaisesRegex(ValueError, "single-factor"):
            parse_study_plan(multifactor)
        wrong_budget = copy.deepcopy(self.plan.to_dict())
        wrong_budget["legacy_bridge"]["budget"]["fit_count"] = 90
        with self.assertRaisesRegex(ValueError, "16 cases/400 fits/4000 epochs"):
            parse_study_plan(wrong_budget)
        cpu = copy.deepcopy(self.plan.to_dict())
        cpu["execution"]["device"] = "cpu"
        with self.assertRaisesRegex(ValueError, "serial CUDA"):
            parse_study_plan(cpu)

    def test_expansion_and_runner_pass_full_controls_without_legacy_source(
        self,
    ) -> None:
        expansion = StudyRunner(pipeline_root=ROOT).expand(self.plan)
        self.assertEqual(len(expansion.cases), 16)
        self.assertTrue(
            all(case.config["training"]["device"] == "cuda" for case in expansion.cases)
        )
        case = expansion.cases[7]  # Inception B3
        result = {"status": "passed", "cell_results": []}
        with tempfile.TemporaryDirectory() as temporary, patch(
            "ppg_frailty.experiment.run_legacy_bridge_outer_cell",
            return_value=result,
        ) as bridge_cell:
            observed = default_experiment_executor(
                case,
                Path(temporary) / "resolved_config.yaml",
                Path(temporary),
                self.plan,
                NullProgressSink(),
            )
        self.assertEqual(observed["status"], "passed")
        self.assertEqual(bridge_cell.call_count, 25)
        self.assertEqual(
            {
                (call.kwargs["repeat_index"], call.kwargs["fold_index"])
                for call in bridge_cell.call_args_list
            },
            {(repeat, fold) for repeat in range(5) for fold in range(5)},
        )
        for call in bridge_cell.call_args_list:
            self.assertEqual(call.kwargs["profile_id"], "B3")
            self.assertEqual(call.kwargs["protocol_design"], "centered_star_v1")
            self.assertEqual(
                call.kwargs["profile_definition_sha256"],
                call.kwargs["profile_definition"]["controls_sha256"],
            )
            self.assertNotIn("source_specification", call.kwargs)
            self.assertNotIn("source_specification_sha256", call.kwargs)

    def test_centered_star_entrypoint_accepts_every_frozen_repeat(self) -> None:
        row = self.bridge.profiles[0]
        with patch.object(
            experiment, "_run_one_outer_cell", return_value="ok"
        ) as run:
            for repeat in range(5):
                observed = experiment.run_legacy_bridge_outer_cell(
                    "resolved.yaml",
                    repeat,
                    3,
                    "output",
                    profile_id="B0",
                    protocol_design="centered_star_v1",
                    profile_definition=row,
                    profile_definition_sha256=row["controls_sha256"],
                )
                self.assertEqual(observed, "ok")
        self.assertEqual(
            [call.kwargs["repeat_index"] for call in run.call_args_list],
            list(range(5)),
        )

    def test_no_training_runner_counts_four_hundred_cells_and_skips_phase0(
        self,
    ) -> None:
        def fake_executor(case, _config, _directory, plan, _sink):
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

        def forbidden_phase0(**_kwargs):
            raise AssertionError("centered star must not run legacy Phase 0")

        with tempfile.TemporaryDirectory() as temporary:
            result = StudyRunner(
                pipeline_root=ROOT,
                executor=fake_executor,
                phase0_runner=forbidden_phase0,
            ).run(self.plan, output_root=temporary)
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.planned_case_count, 16)
        self.assertEqual(result.passed_case_count, 16)
        self.assertEqual(result.planned_cell_count, 400)
        self.assertEqual(result.reported_cell_count, 400)
        self.assertEqual(result.passed_cell_count, 400)


if __name__ == "__main__":
    unittest.main()

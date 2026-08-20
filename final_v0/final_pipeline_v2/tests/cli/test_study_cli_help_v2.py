"""CLI discovery tests; help paths must never import data or start training."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class StudyCliHelpTests(unittest.TestCase):
    def invoke(self, script: str, *arguments: str) -> str:
        completed = subprocess.run(
            [sys.executable, str(ROOT / script), *arguments],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed.stdout

    def test_single_pipeline_help(self) -> None:
        output = self.invoke("frailty_3class_pipeline_v2.py", "--help")
        self.assertIn("--config", output)
        self.assertIn("--resume", output)
        self.assertIn("--jobs", output)
        self.assertIn("--measure-operational-costs", output)

    def test_artifact_comparison_accepts_peak_thresholds(self) -> None:
        from ppg_frailty.cli import build_parser

        arguments = build_parser().parse_args(
            [
                "compare",
                "artifacts",
                "--min-observation-sec",
                "6.5",
                "--min-peaks",
                "3",
            ]
        )
        self.assertEqual(arguments.min_observation_sec, 6.5)
        self.assertEqual(arguments.min_peaks, 3)

    def test_sweep_and_grid_help(self) -> None:
        output = self.invoke("frailty_3class_sweep_v2.py", "--help")
        self.assertIn("ablation", output)
        self.assertIn("grid", output)
        self.assertIn("report", output)
        grid = self.invoke("frailty_3class_sweep_v2.py", "grid", "--help")
        self.assertIn("--vary", grid)
        self.assertIn("--output-root", grid)
        self.assertIn("--device", grid)
        self.assertIn("--measure-operational-costs", grid)
        self.assertIn("--no-measure-operational-costs", grid)

    def test_top_level_operational_cost_switches_reach_study_plans(self) -> None:
        from frailty_3class_pipeline_v2 import (
            build_parser as build_pipeline_parser,
            plan_from_args,
        )
        from frailty_3class_sweep_v2 import (
            _run_plan,
            build_parser as build_sweep_parser,
        )

        pipeline_args = build_pipeline_parser().parse_args(
            [
                "--config",
                "configs/reference_static_role_aware_v2.yaml",
                "--measure-operational-costs",
            ]
        )
        self.assertTrue(
            plan_from_args(pipeline_args).execution.measure_operational_costs
        )

        plan_path = ROOT / "configs" / "studies" / "single_config_v2.yaml"
        loaded = _run_plan(
            build_sweep_parser().parse_args(["run", "--plan", str(plan_path)])
        )
        self.assertTrue(loaded.execution.measure_operational_costs)
        disabled = _run_plan(
            build_sweep_parser().parse_args(
                [
                    "run",
                    "--plan",
                    str(plan_path),
                    "--no-measure-operational-costs",
                ]
            )
        )
        self.assertFalse(disabled.execution.measure_operational_costs)
        cuda = _run_plan(
            build_sweep_parser().parse_args(
                ["run", "--plan", str(plan_path), "--device", "cuda"]
            )
        )
        self.assertEqual(cuda.execution.device, "cuda")

    def test_checked_in_study_plans_declare_operational_cost_policy(self) -> None:
        from ppg_frailty.study import load_study_plan

        study_root = ROOT / "configs" / "studies"
        expected = {
            "ablation_fixed_epochs_v2.yaml": True,
            "grid_optimizer_v2.yaml": True,
            "single_config_v2.yaml": True,
            "static_line_b_all_models_v2.yaml": True,
            "static_line_b_staged_v2/01_representation_baselines_v2.yaml": False,
            "static_line_b_staged_v2/02_competitive_routes_models_v2.yaml": False,
            "static_line_b_staged_v2/stage_last_shapeformer_stability_v2.yaml": False,
            "static_line_b_staged_v2/04_selected_inception_ensemble_v2.yaml": True,
            "static_line_b_staged_v2/05_sqi_motion_finalists_v2.yaml": True,
            "static_line_b_staged_v2/06_sequential_single_factor_ablation_v2.yaml": True,
        }
        for relative_path, enabled in expected.items():
            with self.subTest(plan=relative_path):
                plan = load_study_plan(study_root / relative_path)
                self.assertIs(
                    plan.execution.measure_operational_costs,
                    enabled,
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ppg_frailty.evaluate.decision_bias_oracle import (
    enumerate_simplex_biases,
    load_decision_bias_oracle_plan,
    load_participant_oracle_dataset,
    run_decision_bias_oracle,
    search_decision_bias_oracle,
)


class DecisionBiasOracleTest(unittest.TestCase):
    def test_step_point_zero_one_has_5151_simplex_points(self) -> None:
        biases = enumerate_simplex_biases(0.01)
        self.assertEqual(biases.shape, (5151, 3))
        np.testing.assert_allclose(biases.sum(axis=1), 1.0)
        self.assertTrue(bool((biases >= 0.0).all()))

    def test_known_bias_recovers_three_class_decisions(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        probabilities = np.asarray(
            [
                [0.50, 0.30, 0.20],
                [0.51, 0.29, 0.20],
                [0.45, 0.40, 0.15],
                [0.44, 0.39, 0.17],
                [0.44, 0.20, 0.36],
                [0.43, 0.21, 0.36],
            ],
            dtype=np.float64,
        )
        result = search_decision_bias_oracle(
            labels, probabilities, class_order=(0, 1, 2), step=0.01
        )
        self.assertAlmostEqual(result.baseline_balanced_accuracy, 1.0 / 3.0)
        self.assertAlmostEqual(result.oracle_balanced_accuracy, 1.0)
        self.assertAlmostEqual(sum(result.best_bias), 1.0)

    def _write_fixture(self, root: Path, *, omit_last: bool = False) -> Path:
        study = root / "source_study"
        prediction = (
            study
            / "raw/final_model/attempts/attempt_001/experiment/oof_subject_predictions.parquet"
        )
        prediction.parent.mkdir(parents=True)
        rows = []
        probabilities = {
            "p0": [0.60, 0.25, 0.15],
            "p1": [0.30, 0.55, 0.15],
            "p2": [0.30, 0.20, 0.50],
        }
        for repeat in (0, 1):
            for label, participant_id in enumerate(("p0", "p1", "p2")):
                rows.append(
                    {
                        "participant_id": participant_id,
                        "label": label,
                        "probabilities": np.asarray(probabilities[participant_id]),
                        "repeat": repeat,
                        "level": "participant",
                        "retained": True,
                        "prediction_kind": "single_model",
                        "class_order": np.asarray([0, 1, 2]),
                        "config_hash": "config-a",
                        "aggregation_rule": "line_b_equal_role_families",
                    }
                )
        if omit_last:
            rows.pop()
        pd.DataFrame(rows).to_parquet(prediction, index=False)
        plan = {
            "schema_version": "ppg_frailty.stage0_decision_bias_oracle.v1",
            "study": {"study_id": "stage0-test"},
            "source": {
                "study_dir": str(study),
                "case_id": "final_model",
                "prediction_file": None,
                "prediction_level": "participant",
                "prediction_kind": "single_model",
                "expected_participants": 3,
                "expected_repeats": [0, 1],
                "expected_class_order": [0, 1, 2],
                "repeat_aggregation": "arithmetic_mean_probability_per_participant",
            },
            "oracle": {
                "bias_parameterization": "nonnegative_sum_one_simplex",
                "step": 0.01,
                "objective": "balanced_accuracy",
                "prediction_rule": "argmax_probability_plus_bias",
                "prediction_tie_break": "first_class_in_declared_class_order",
                "optimum_tie_break": "closest_to_equal_bias_then_lexicographic",
            },
            "interpretation": {
                "substantial_upper_bound": 0.70,
                "limited_upper_bound": 0.65,
            },
            "output": {
                "slug": "stage0-test",
                "write_static_figures": False,
                "write_excel_workbook": False,
            },
        }
        plan_path = root / "plan.yaml"
        plan_path.write_text(yaml.safe_dump(plan), encoding="utf-8")
        return plan_path

    def test_complete_roster_is_averaged_to_one_row_per_participant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = load_decision_bias_oracle_plan(
                self._write_fixture(root), pipeline_root=root
            )
            dataset = load_participant_oracle_dataset(plan)
            self.assertEqual(dataset.source_rows, 6)
            self.assertEqual(dataset.probabilities.shape, (3, 3))
            self.assertEqual(dataset.participant_ids, ("p0", "p1", "p2"))

    def test_incomplete_repeat_roster_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = load_decision_bias_oracle_plan(
                self._write_fixture(root, omit_last=True), pipeline_root=root
            )
            with self.assertRaisesRegex(ValueError, "row count mismatch"):
                load_participant_oracle_dataset(plan)

    def test_end_to_end_report_carries_leakage_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._write_fixture(root)
            output = run_decision_bias_oracle(
                plan_path,
                pipeline_root=root,
                output_root=root / "outputs",
            )
            result = (output / "stage0_result.json").read_text(encoding="utf-8")
            report = (output / "STUDY_SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn('"eligible_as_predictive_performance": false', result)
            self.assertIn("LEAKED UPPER BOUND", report)
            self.assertTrue((output / "tables/bias_grid.csv").is_file())
            self.assertTrue((output / "tables/TABLE_COLUMN_DEFINITIONS.md").is_file())


if __name__ == "__main__":
    unittest.main()

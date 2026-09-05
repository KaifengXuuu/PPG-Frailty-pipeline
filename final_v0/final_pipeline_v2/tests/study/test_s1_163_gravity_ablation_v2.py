"""No-training contracts for all-role gravity-removal supplements."""

from __future__ import annotations

from pathlib import Path
import unittest

from ppg_frailty.study import StudyRunner, load_study_plan, validate_canonical_expansion


ROOT = Path(__file__).resolve().parents[2]
PLAN = (
    ROOT
    / "configs"
    / "studies"
    / "static_line_b_staged_v2"
    / "stage_ablation_s1_163_gravity_removal_v1.yaml"
)
ARCHITECTURE_PLAN = (
    ROOT
    / "configs"
    / "studies"
    / "static_line_b_staged_v2"
    / "final_case_all_roles_inception_architecture_comparison_v1.yaml"
)
SMALL_SUPPLEMENT_PLAN = (
    ROOT
    / "configs"
    / "studies"
    / "static_line_b_staged_v2"
    / "stage0_inception_small_no_gravity_supplement_v1.yaml"
)


class S1163GravityAblationTests(unittest.TestCase):
    def test_plan_is_matched_all_role_five_by_five_ablation(self) -> None:
        plan = load_study_plan(PLAN)
        expansion = validate_canonical_expansion(
            StudyRunner(pipeline_root=ROOT).expand(plan)
        )
        self.assertEqual(plan.study.decision_role, "ablation")
        self.assertEqual(plan.execution.repeats, (0, 1, 2, 3, 4))
        self.assertEqual(plan.execution.folds, (0, 1, 2, 3, 4))
        self.assertEqual(plan.execution.device, "cuda")
        self.assertEqual(plan.execution.jobs, 1)
        self.assertFalse(plan.execution.continue_on_error)
        self.assertEqual(len(expansion.cases), 2)

        by_id = {case.case_id: case.config for case in expansion.cases}
        reference = by_id["s1_163_all_roles__profile_a_gravity_removal"]
        candidate = by_id["s1_163_all_roles__no_gravity_removal"]

        for section in (
            "manifest",
            "roles",
            "windows",
            "quality",
            "artifact",
            "features",
            "model",
            "training",
            "aggregation",
            "evaluation",
        ):
            self.assertEqual(reference[section], candidate[section])
        reference_signal = dict(reference["signal"])
        candidate_signal = dict(candidate["signal"])
        reference_imu = reference_signal.pop("imu")
        candidate_imu = candidate_signal.pop("imu")
        self.assertEqual(reference_signal, candidate_signal)

        self.assertEqual(
            reference_imu["gravity_method"], "profile_a_lowpass_0p3hz"
        )
        self.assertEqual(
            candidate_imu["gravity_method"],
            "sensor_filter_only_no_gravity_removal",
        )
        self.assertEqual(
            set(reference_imu) - set(candidate_imu),
            {"gravity_lowpass_hz", "gravity_filter_order"},
        )
        self.assertFalse(set(candidate_imu) - set(reference_imu))
        for key in set(reference_imu) & set(candidate_imu):
            if key not in {"gravity_method", "comparison_method"}:
                self.assertEqual(reference_imu[key], candidate_imu[key])
        for key in (
            "initialization",
            "sensor_lowpass_acc_hz",
            "sensor_lowpass_gyro_hz",
            "sensor_filter_order",
            "calibration_start_s",
            "calibration_stop_s",
            "gravity_mps2",
            "output_units",
            "required_axes",
            "failure_action",
        ):
            self.assertEqual(reference_imu[key], candidate_imu[key])

        self.assertEqual(reference["model"]["model_id"], "InceptionTimeFull")
        self.assertEqual(reference["roles"], ["B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2"])
        self.assertEqual(reference["training"]["fixed_epochs"], 15)
        self.assertEqual(reference["training"]["batch_size"], 32)
        self.assertEqual(reference["training"]["optimizer"], "adamw")
        self.assertEqual(reference["training"]["learning_rate"], 0.001)
        self.assertEqual(reference["training"]["weight_decay"], 0.0005)
        self.assertEqual(reference["training"]["label_smoothing"], 0.10)
        self.assertEqual(reference["training"]["class_count_basis"], "participant")
        self.assertEqual(reference["windows"]["raw_dl"]["length_s"], 5.0)
        self.assertEqual(reference["windows"]["raw_dl"]["hop_s"], 2.5)
        self.assertEqual(reference["windows"]["raw_dl"]["cap_fraction_per_file"], 0.90)

    def test_small_supplement_matches_completed_small_except_gravity_profile(self) -> None:
        runner = StudyRunner(pipeline_root=ROOT)
        architecture_plan = load_study_plan(ARCHITECTURE_PLAN)
        architecture = validate_canonical_expansion(
            runner.expand(architecture_plan)
        )
        supplement_plan = load_study_plan(SMALL_SUPPLEMENT_PLAN)
        supplement = validate_canonical_expansion(
            runner.expand(supplement_plan)
        )
        self.assertEqual(supplement_plan.execution.repeats, (0, 1, 2, 3, 4))
        self.assertEqual(supplement_plan.execution.folds, (0, 1, 2, 3, 4))
        self.assertEqual(supplement_plan.execution.device, "cuda")
        self.assertEqual(supplement_plan.execution.jobs, 1)
        self.assertFalse(supplement_plan.execution.continue_on_error)
        self.assertEqual(supplement_plan.report.detailed_configuration_top_k, 5)
        self.assertEqual(len(supplement.cases), 1)

        reference = next(
            case.config
            for case in architecture.cases
            if case.case_id == "tuned_all_roles__inception_small"
        )
        candidate = supplement.cases[0].config
        for section in (
            "manifest",
            "roles",
            "windows",
            "quality",
            "artifact",
            "features",
            "model",
            "training",
            "aggregation",
            "evaluation",
        ):
            self.assertEqual(reference[section], candidate[section])

        reference_signal = dict(reference["signal"])
        candidate_signal = dict(candidate["signal"])
        reference_imu = reference_signal.pop("imu")
        candidate_imu = candidate_signal.pop("imu")
        self.assertEqual(reference_signal, candidate_signal)
        self.assertEqual(reference_imu["gravity_method"], "profile_a_lowpass_0p3hz")
        self.assertEqual(
            candidate_imu["gravity_method"],
            "sensor_filter_only_no_gravity_removal",
        )
        self.assertEqual(
            set(reference_imu) - set(candidate_imu),
            {"gravity_lowpass_hz", "gravity_filter_order"},
        )
        self.assertFalse(set(candidate_imu) - set(reference_imu))
        for key in set(reference_imu) & set(candidate_imu):
            if key not in {"gravity_method", "comparison_method"}:
                self.assertEqual(reference_imu[key], candidate_imu[key])
        for key in (
            "initialization",
            "sensor_lowpass_acc_hz",
            "sensor_lowpass_gyro_hz",
            "sensor_filter_order",
            "calibration_start_s",
            "calibration_stop_s",
            "gravity_mps2",
            "output_units",
            "required_axes",
            "failure_action",
        ):
            self.assertEqual(reference_imu[key], candidate_imu[key])


if __name__ == "__main__":
    unittest.main()

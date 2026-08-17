"""M2 fold artifact 与 PTT ECG 评价测试 / Fold and ECG-reference tests."""

from __future__ import annotations

import unittest

import numpy as np

from m3_signal_core import (
    evaluate_ppg_against_ecg,
    fit_fold_scaler,
    fit_transit_delay,
    resolve_m2_fold,
)


class FoldBindingTests(unittest.TestCase):
    """验证训练 roster 与 OOF 边界 / Validate training and OOF boundaries."""

    def test_exact_m2_training_roster_builds_artifact(self) -> None:
        """精确 train roster 可拟合 / Exact training roster can fit."""

        fold = resolve_m2_fold(0, 0)
        subjects = fold["train_subject_ids"]
        values = np.column_stack(
            [
                np.arange(len(subjects), dtype=np.float64),
                np.arange(len(subjects), dtype=np.float64) ** 2,
            ]
        )
        scaler, artifact = fit_fold_scaler(
            values,
            subjects,
            ["feature_a", "feature_b"],
            repeat_index=0,
            fold_index=0,
            preprocessing_profile_ids=["frailty3_raw8_classifier_400_v1"],
        )
        self.assertEqual(
            artifact["fold_registry_id"], "frailty3_future_corrected_sgkf5_v2"
        )
        self.assertEqual(
            artifact["fold_registry_payload_sha256"],
            "0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46",
        )
        self.assertEqual(
            artifact["feature_schema_version"], "m3_raw8_dynamic_sequence.v1"
        )
        self.assertEqual(artifact["status"], "locked")
        self.assertEqual(
            artifact["preprocessing_registry_id"],
            "m3_preprocessing_profiles_corrected_v1",
        )
        self.assertFalse(
            set(artifact["subject_partition"]["train_subject_ids"])
            & set(artifact["subject_partition"]["oof_validation_subject_ids"])
        )
        self.assertEqual(artifact["transformers"][0]["method"], "robust_median_iqr")
        self.assertEqual(len(artifact["parameters_sha256"]), 64)
        self.assertEqual(scaler.transform(values).shape, values.shape)

    def test_future_fold_scaler_rejects_standard_or_clipping(self) -> None:
        """future hybrid scaling 只允许 robust/no-clip / Enforce frozen D4 route."""

        fold = resolve_m2_fold(0, 0)
        subjects = fold["train_subject_ids"]
        values = np.arange(len(subjects) * 2, dtype=np.float64).reshape(-1, 2)
        with self.assertRaisesRegex(ValueError, "requires_robust_no_clip"):
            fit_fold_scaler(
                values,
                subjects,
                ["feature_a", "feature_b"],
                repeat_index=0,
                fold_index=0,
                preprocessing_profile_ids=["frailty3_raw8_classifier_400_v1"],
                method="standard",
            )
        with self.assertRaisesRegex(ValueError, "requires_robust_no_clip"):
            fit_fold_scaler(
                values,
                subjects,
                ["feature_a", "feature_b"],
                repeat_index=0,
                fold_index=0,
                preprocessing_profile_ids=["frailty3_raw8_classifier_400_v1"],
                clip=6.0,
            )

    def test_oof_subject_is_rejected_before_fit(self) -> None:
        """故意混入 OOF 必须失败 / Deliberate OOF contamination must fail."""

        fold = resolve_m2_fold(0, 0)
        subjects = list(fold["train_subject_ids"])
        subjects[0] = fold["oof_validation_subject_ids"][0]
        with self.assertRaisesRegex(ValueError, "oof_subject_present_in_fit"):
            fit_fold_scaler(
                np.ones((len(subjects), 2)),
                subjects,
                ["feature_a", "feature_b"],
                repeat_index=0,
                fold_index=0,
                preprocessing_profile_ids=["frailty3_raw8_classifier_400_v1"],
            )


class PttReferenceTests(unittest.TestCase):
    """验证 training-only transit delay / Validate train-only transit delay."""

    def test_delay_correction_and_disjoint_evaluation(self) -> None:
        """训练延迟校正独立 subject / Fit delay and score a disjoint subject."""

        ecg = np.arange(400, 4400, 400, dtype=np.int64)
        pairs = {
            "train_1": (ecg, ecg + 80),
            "train_2": (ecg, ecg + 82),
        }
        artifact = fit_transit_delay(
            pairs,
            training_subject_ids=["train_1", "train_2"],
            fs_hz=400.0,
            fit_role="training",
            training_split_id="fixture_train_v1",
        )
        score = evaluate_ppg_against_ecg(
            ecg,
            ecg + artifact.delay_samples,
            evaluation_subject_id="eval_1",
            fs_hz=400.0,
            delay_artifact=artifact,
        )
        self.assertLess(score["raw"]["f1"], 0.1)
        self.assertEqual(score["delay_corrected"]["f1"], 1.0)
        self.assertIsNone(score["raw"]["ppi_error_ms_mae"])
        self.assertEqual(score["raw"]["hr_error_bpm"], 0.0)
        self.assertEqual(score["delay_corrected"]["timing_error_ms_mae"], 0.0)
        self.assertEqual(score["delay_corrected"]["ppi_error_ms_mae"], 0.0)
        self.assertEqual(score["ppi_error_ms_mae"], 0.0)
        self.assertEqual(score["hr_error_bpm"], 0.0)
        self.assertEqual(score["training_split_id"], "fixture_train_v1")

    def test_delay_fit_and_eval_leakage_are_rejected(self) -> None:
        """fit role 与 subject overlap 均 fail closed / Reject leakage."""

        ecg = np.arange(400, 2400, 400, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "requires_training_role"):
            fit_transit_delay(
                {"s1": (ecg, ecg + 80)},
                training_subject_ids=["s1"],
                fs_hz=400.0,
                fit_role="oof_validation",
                training_split_id="fixture_train_v1",
            )
        artifact = fit_transit_delay(
            {"s1": (ecg, ecg + 80)},
            training_subject_ids=["s1"],
            fs_hz=400.0,
            fit_role="training",
            training_split_id="fixture_train_v1",
        )
        with self.assertRaisesRegex(ValueError, "evaluation_subject_present"):
            evaluate_ppg_against_ecg(
                ecg,
                ecg + 80,
                evaluation_subject_id="s1",
                fs_hz=400.0,
                delay_artifact=artifact,
            )


if __name__ == "__main__":
    # 中文：标准库 runner 与 pytest discovery 共用。
    # English: Share tests between unittest and pytest discovery.
    unittest.main()

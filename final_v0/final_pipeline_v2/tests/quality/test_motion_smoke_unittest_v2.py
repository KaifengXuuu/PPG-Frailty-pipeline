"""Standard-library motion contract smoke tests; no training, CV, or PTT run.

标准库 motion 合同冒烟测试；不执行训练、交叉验证、消融或 PTT 外测。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from ppg_frailty.models.motion import LightCnnArchitecture
from ppg_frailty.quality.motion import (
    MotionOptionId,
    fit_train_only_midpoint_threshold,
    load_motion_fold_jobs,
    motion_activity_label,
    motion_contract_payload,
    resolve_motion_option,
)


ROOT = Path(__file__).resolve().parents[2]


class MotionContractSmokeTests(unittest.TestCase):
    """Exercise only deterministic contract construction / 仅检查确定性合同。"""

    def test_default_options_and_role_labels(self) -> None:
        self.assertIs(resolve_motion_option(None).option_id, MotionOptionId.SQI_ONLY)
        motion_evidence = resolve_motion_option("sqi_plus_motion_override")
        self.assertFalse(motion_evidence.formal_default)
        self.assertEqual(
            motion_evidence.execution_status,
            "external_ptt_evidence_protocol_not_classifier_runtime",
        )
        self.assertEqual(
            motion_evidence.classifier_effect,
            "none_not_dispatched_by_core_classifier_pipeline",
        )
        self.assertEqual(
            [motion_activity_label(role) for role in ("B", "R1", "R4", "S1", "W2")],
            [0, 0, 0, 1, 1],
        )

    def test_registry_resolves_single_seed42_sgkf5(self) -> None:
        jobs = load_motion_fold_jobs(ROOT / "splits" / "sgkf5_seed42_v2.csv")
        self.assertEqual(len(jobs), 5)
        self.assertEqual({job.split_seed for job in jobs}, {42})
        self.assertEqual({job.training_seed for job in jobs}, {42})
        for job in jobs:
            self.assertFalse(set(job.train_participant_ids) & set(job.oof_participant_ids))
            self.assertEqual(len(job.train_participant_ids) + len(job.oof_participant_ids), 29)
        self.assertEqual(
            sorted(len(job.oof_participant_ids) for job in jobs),
            [5, 6, 6, 6, 6],
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 drift"):
            load_motion_fold_jobs(
                ROOT / "splits" / "sgkf5_repeated_grouped_5x5_v2.csv"
            )

    def test_midpoint_is_exact_and_train_only(self) -> None:
        artifact = fit_train_only_midpoint_threshold(
            scores=(0.1, 0.9, 0.2, 0.8),
            labels=(0, 1, 0, 1),
            participant_ids=("p1", "p1", "p2", "p2"),
            training_participant_ids=("p1", "p2"),
        )
        self.assertAlmostEqual(artifact.static_center, 0.15)
        self.assertAlmostEqual(artifact.motion_center, 0.85)
        self.assertAlmostEqual(artifact.threshold, 0.5)
        with self.assertRaisesRegex(ValueError, "OOF/PTT"):
            fit_train_only_midpoint_threshold(
                scores=(0.1, 0.9),
                labels=(0, 1),
                participant_ids=("held", "held"),
                training_participant_ids=("held",),
                forbidden_oof_participant_ids=("held",),
            )

    def test_schema_must_be_explicit_and_payload_does_not_claim_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one input channel"):
            LightCnnArchitecture(channel_names=()).validate()
        architecture = LightCnnArchitecture(channel_names=("RED", "IR", "acc_x"))
        architecture.validate()
        self.assertGreater(architecture.parameter_count, 0)
        payload = motion_contract_payload()
        self.assertEqual(payload["default_option"], "sqi_only")
        self.assertEqual(payload["execution_status"], "implemented_contract_registered_not_run")
        self.assertEqual(
            payload["external_ptt_readiness_audit"]["status"],
            "read_only_complete_internal_formal_evidence_and_exact_v2_036_unit_artifact",
        )
        self.assertEqual(
            payload["external_ptt_readiness_audit"]["execution_authority"],
            "none_audit_does_not_control_evaluation",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

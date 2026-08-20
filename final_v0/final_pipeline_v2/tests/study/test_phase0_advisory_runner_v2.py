"""No-training checks that Phase 0 is advisory rather than a study gate."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ppg_frailty.study import StudyRunner, load_study_plan


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    PIPELINE_ROOT
    / "configs/studies/static_line_b_staged_v2/stage3_alter.yaml"
)


def _passing_executor(case, _config, _directory, plan, _sink):
    return {
        "status": "passed",
        "config_id": case.config["config_id"],
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


class Phase0AdvisoryRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = load_study_plan(PLAN_PATH)

    def test_stop_decision_is_persisted_but_does_not_block_cases(self) -> None:
        case_calls: list[str] = []

        def executor(case, *args):
            case_calls.append(case.case_id)
            return _passing_executor(case, *args)

        def stopped_audit(**_kwargs):
            return {
                "schema_version": "ppg_frailty.legacy_v2_phase0_result.v1",
                "decision": "STOP",
                "advisory_checks_passed": False,
                "stop_reasons": ["synthetic_advisory_finding"],
            }

        with tempfile.TemporaryDirectory() as temporary:
            result = StudyRunner(
                pipeline_root=PIPELINE_ROOT,
                executor=executor,
                phase0_runner=stopped_audit,
            ).run(self.plan, output_root=temporary)
            audit = json.loads(
                (result.output_directory / "phase0_audit.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result.status, "passed")
        self.assertEqual(case_calls, [case.case_id for case in self.plan.cases])
        self.assertTrue(audit["advisory_only"])
        self.assertFalse(audit["affects_training_execution"])
        self.assertEqual(audit["audit_status"], "completed")
        self.assertEqual(audit["audit_decision"], "STOP")
        self.assertEqual(audit["audit_result"]["stop_reasons"], [
            "synthetic_advisory_finding"
        ])

    def test_audit_exception_is_persisted_but_does_not_block_cases(self) -> None:
        case_calls = 0

        def executor(case, *args):
            nonlocal case_calls
            case_calls += 1
            return _passing_executor(case, *args)

        def broken_audit(**_kwargs):
            raise RuntimeError("synthetic audit failure")

        with tempfile.TemporaryDirectory() as temporary:
            result = StudyRunner(
                pipeline_root=PIPELINE_ROOT,
                executor=executor,
                phase0_runner=broken_audit,
            ).run(self.plan, output_root=temporary)
            audit = json.loads(
                (result.output_directory / "phase0_audit.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result.status, "passed")
        self.assertEqual(case_calls, len(self.plan.cases))
        self.assertEqual(audit["audit_status"], "error")
        self.assertEqual(audit["error_type"], "RuntimeError")
        self.assertIn("synthetic audit failure", audit["error"])

    def test_resume_refreshes_advisory_audit_without_invalidating_cases(self) -> None:
        audit_calls = 0
        case_calls = 0

        def executor(case, *args):
            nonlocal case_calls
            case_calls += 1
            return _passing_executor(case, *args)

        def changing_audit(**_kwargs):
            nonlocal audit_calls
            audit_calls += 1
            return {
                "schema_version": "ppg_frailty.legacy_v2_phase0_result.v1",
                "decision": "STOP" if audit_calls == 1 else "PASS",
                "advisory_checks_passed": audit_calls != 1,
                "stop_reasons": (
                    ["first_advisory_finding"] if audit_calls == 1 else []
                ),
            }

        with tempfile.TemporaryDirectory() as temporary:
            runner = StudyRunner(
                pipeline_root=PIPELINE_ROOT,
                executor=executor,
                phase0_runner=changing_audit,
            )
            first = runner.run(self.plan, output_root=temporary)
            resumed = runner.run(
                self.plan,
                resume_directory=first.output_directory,
            )
            audit = json.loads(
                (resumed.output_directory / "phase0_audit.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(audit_calls, 2)
        self.assertEqual(case_calls, len(self.plan.cases))
        self.assertEqual(resumed.resumed_case_count, len(self.plan.cases))
        self.assertEqual(audit["audit_decision"], "PASS")


if __name__ == "__main__":
    unittest.main()

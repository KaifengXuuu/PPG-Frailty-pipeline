"""V2 CLI 身份与无训练 smoke / V2 CLI identity and non-training smoke."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from ppg_frailty.cli import build_parser, main


class V2CliTests(unittest.TestCase):
    """验证公开命令默认指向 V2 / Check public commands bind to V2."""

    def _call(self, arguments: list[str]) -> tuple[int, dict[str, object]]:
        with io.StringIO() as stream, redirect_stdout(stream):
            code = main(arguments)
            payload = json.loads(stream.getvalue())
        return code, payload

    def test_profiles_validation_reports_pending_locks_without_failure(self) -> None:
        """未安装 profile 可见但不伪装 exact lock / Pending is explicit."""

        code, payload = self._call(["validate", "--profiles-only"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["schema_version"], "ppg_frailty.profile_validation.v2")
        self.assertEqual(payload["dependency_profile_count"], 6)
        self.assertEqual(payload["supplemental_optional_input_count"], 1)
        self.assertIs(payload["all_exact_locks_ready"], False)

    def test_default_smoke_is_v2_preflight_without_training_or_ablation(self) -> None:
        """smoke 只做合同预检 / Smoke performs preflight only."""

        code, payload = self._call(["smoke", "--config", "default"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["pipeline_generation"], "final_pipeline_v2")
        self.assertEqual(payload["config"]["config_id"], "reference_static_feature_vector_lr_line_a_v2")
        self.assertIs(payload["training_executed"], False)
        self.assertIs(payload["ablations_executed"], False)
        self.assertIs(payload["scientific_metrics_emitted"], False)

    def test_matrix_and_fusion_aliases_are_non_training_preflights(self) -> None:
        """新增入口只做preflight / New representation aliases do not train."""

        expected = {
            "matrix-line-a": "reference_static_feature_matrix_inception_full_line_a_v2",
            "fusion-line-a": "reference_static_fusion_compact_line_a_v2",
        }
        for alias, config_id in expected.items():
            code, payload = self._call(["smoke", "--config", alias])
            self.assertEqual(code, 0)
            self.assertEqual(payload["config"]["config_id"], config_id)
            self.assertIs(payload["training_executed"], False)
            self.assertIs(payload["ablations_executed"], False)

    def test_motion_validate_is_contract_only(self) -> None:
        """Motion入口只验证合同和fold / Never trains or evaluates PTT."""

        code, payload = self._call(["motion-validate"])
        self.assertEqual(code, 0)
        self.assertEqual(
            payload["schema_version"],
            "ppg_frailty.motion_contract_validation.v2",
        )
        self.assertEqual(payload["internal_fold_job_count"], 5)
        self.assertEqual(payload["ptt_assignment_row_count"], 110)
        self.assertIs(payload["training_executed"], False)
        self.assertIs(payload["ptt_evaluation_executed"], False)

    def test_safe_suite_and_fixed_sample_factor_are_explicit(self) -> None:
        """默认测试不触发comparison，时间尺度名称不伪装物理匹配。"""

        safe = build_parser().parse_args(["test"])
        self.assertEqual(safe.suite, "safe")
        opted_in = build_parser().parse_args(["test", "--suite", "all"])
        self.assertEqual(opted_in.suite, "all")
        fixed = build_parser().parse_args(
            ["ablate", "--factor", "fixed_kernel_samples"]
        )
        self.assertEqual(fixed.factor, "fixed_kernel_samples")
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["ablate", "--factor", "physical_time"])

    def test_catalog_inspection_includes_profiles_but_executes_nothing(self) -> None:
        code, payload = self._call(["catalog", "--line", "line_a"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["candidate_count"], 13)
        self.assertEqual(payload["ensemble_comparison_count"], 2)
        self.assertEqual(payload["fixed_kernel_case_count"], 12)
        self.assertIs(payload["ablation_profile_auto_run"], False)
        self.assertIs(payload["training_executed"], False)

    def test_legacy_inspection_is_never_formal(self) -> None:
        """历史入口只返回 provenance / Legacy inspection is provenance-only."""

        code, payload = self._call(["validate", "--legacy-config", "reference_static_v1"])
        self.assertEqual(code, 0)
        self.assertIn("historical/v1_transition/configs", payload["source_path"])
        self.assertIs(payload["formal_v2_eligible"], False)
        self.assertEqual(payload["scientific_scope"], "copied_historical_provenance_only")

    def test_prv_comparison_is_a_named_compare_command(self) -> None:
        """PRV 库对照不伪装 pipeline backend / PRV comparison is explicit."""

        arguments = build_parser().parse_args(
            ["compare", "prv-backends", "--backends", "local", "--fixtures", "steady_75bpm"]
        )
        self.assertEqual(arguments.comparison, "prv-backends")
        self.assertEqual(arguments.backends, ["local"])
        self.assertEqual(arguments.fixtures, ["steady_75bpm"])

    def test_prv_comparison_executes_only_fixed_vector_functions(self) -> None:
        """CLI 运行本地函数对照但不清洗或分类 / No cleaner or classifier."""

        code, payload = self._call(
            ["compare", "prv-backends", "--backends", "local", "--fixtures", "steady_75bpm"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["schema_version"], "ppg_frailty.prv_backend_comparison.v2")
        self.assertEqual(
            payload["status"],
            "diagnostic_success_not_exact_profile_evidence",
        )
        self.assertEqual(payload["comparison_scope"], "fixed_ppi_function_outputs_only")
        self.assertIs(payload["cleaner_applied"], False)
        self.assertIs(payload["classifier_integrated"], False)
        self.assertEqual(
            payload["execution_authority"]["status"],
            "local_diagnostic_no_optional_dependency_gate",
        )
        self.assertEqual(len(payload["fixtures"]), 1)
        fixture = payload["fixtures"][0]
        self.assertEqual(fixture["fixture_id"], "steady_75bpm")
        row = fixture["backends"][0]
        self.assertEqual(row["backend"], "local")
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["input_sha256"], fixture["input_sha256"])
        self.assertIs(row["cleaner_applied"], False)
        self.assertIs(row["classifier_integrated"], False)

    def test_aura_formal_wrapper_requires_and_archives_exact_gates(self) -> None:
        """Only the CLI gate may upgrade a diagnostic Aura result to passed."""

        diagnostic = {
            "schema_version": "ppg_frailty.prv_backend_comparison.v2",
            "status": "diagnostic_success_not_exact_profile_evidence",
            "execution_authority": {
                "status": "diagnostic_only_unverified_runtime",
                "formal_optional_profile_evidence": False,
            },
            "comparison_scope": "fixed_ppi_function_outputs_only",
            "requested_backends": ["aura_hrv_analysis"],
            "cleaner_applied": False,
            "classifier_integrated": False,
            "fixtures": [],
        }
        with (
            patch(
                "ppg_frailty.config.dependency_gate_report",
                return_value={"all_required_exact_locks_ready": True},
            ),
            patch(
                "ppg_frailty.experiment._require_scientific_source_gate",
                return_value={"tracked_and_clean": True},
            ),
            patch(
                "ppg_frailty.experiment._source_snapshot_sha256",
                return_value="a" * 64,
            ),
            patch(
                "ppg_frailty.features.prv_backend_compare."
                "run_prv_backend_comparison",
                return_value=diagnostic,
            ),
        ):
            code, payload = self._call(
                ["compare", "prv-backends", "--backends", "aura_hrv_analysis"]
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            payload["schema_version"],
            "ppg_frailty.prv_backend_comparison_formal.v2",
        )
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(
            payload["diagnostic_result_status"],
            "diagnostic_success_not_exact_profile_evidence",
        )
        self.assertIs(
            payload["execution_authority"]["formal_optional_profile_evidence"],
            True,
        )
        self.assertIn("aura_hrv_analysis", payload["execution_authority"]["gates"])

    def test_prv_registry_has_one_formal_and_two_function_only_rows(self) -> None:
        """三backend科学身份不能混淆 / PRV backend statuses remain distinct."""

        code, payload = self._call(["list-modules", "--family", "prv_backend"])
        self.assertEqual(code, 0)
        statuses = {row["module_id"]: row["scientific_status"] for row in payload["modules"]}
        self.assertEqual(statuses["local"], "formal_primary")
        self.assertEqual(statuses["aura_hrv_analysis"], "function_comparison_only")
        self.assertEqual(statuses["rhenan_hrv"], "legacy_function_comparison_only")


if __name__ == "__main__":
    unittest.main()

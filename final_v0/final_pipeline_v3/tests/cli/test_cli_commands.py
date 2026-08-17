"""非交互 CLI 的 strict-JSON 黑盒验收 / Strict-JSON CLI black-box acceptance."""

from __future__ import annotations

import json
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def run_cli(*arguments: str) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    """在活动 V2 source 上运行命令 / Run a command against the active V2 source."""

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "ppg_frailty.cli", *arguments],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    stream = completed.stdout if completed.returncode == 0 else completed.stderr
    payload = json.loads(stream.strip())
    return completed, payload


class CliCommandTests(unittest.TestCase):
    """验证可直接运行的目录、预检与量化命令 / Exercise public commands."""

    def test_list_modules_is_machine_readable(self) -> None:
        """模块目录含四表示且实现路径规范 / Registry is strict JSON."""

        completed, payload = run_cli("list-modules")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertGreaterEqual(int(payload["count"]), 24)
        implementations = [item["implementation"] for item in payload["modules"]]
        self.assertFalse(any("ppg_frailty.artifacts." in value for value in implementations))

    def test_validate_all_formal_v2_configs_pass(self) -> None:
        """五份正式pipeline配置通过；motion合同不混入 / Five formal configs."""

        completed, payload = run_cli("validate", "--all-configs")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(
            {row["config"] for row in payload["results"]},
            {
                "reference_static_feature_vector_v2.yaml",
                "reference_static_feature_vector_line_b_v2.yaml",
                "reference_static_feature_matrix_v2.yaml",
                "reference_static_fusion_v2.yaml",
                "reference_static_line_a_v2.yaml",
            },
        )
        self.assertTrue(all(row["status"] == "passed" for row in payload["results"]))

    def test_dl_fs_ablation_is_directly_runnable(self) -> None:
        """DL fs catalog 量化命令覆盖 4 rates / Ablation exposes four rates."""

        completed, payload = run_cli("ablate", "--factor", "dl_fs", "--seed", "42")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual([row["dl_fs_hz"] for row in payload["results"]], [100, 160, 200, 400])

    def test_reduced_artifact_comparison_runs_real_reducers(self) -> None:
        """synthetic 对照实际调用 reducer，不声明科学 benchmark / Run reducers."""

        completed, payload = run_cli(
            "compare",
            "artifacts",
            "--reducers",
            "identity",
            "spectral_mask",
            "spectral",
            "--duration-s",
            "8",
            "--seed",
            "42",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(len(payload["results"]), 5)
        self.assertEqual(
            {row["canonical_module_id"] for row in payload["results"][:2]},
            {"raw_no_denoise", "quality_only"},
        )
        reducer_rows = [row for row in payload["results"] if "requested_module_id" in row]
        self.assertEqual(reducer_rows[1]["canonical_module_id"], "spectral_mask")
        self.assertFalse(reducer_rows[1]["legacy_alias_used"])
        self.assertEqual(reducer_rows[2]["canonical_module_id"], "spectral_mask")
        self.assertTrue(reducer_rows[2]["legacy_alias_used"])
        self.assertIn("not_external_ptt_benchmark", payload["scientific_scope"])

    def test_imu_gravity_comparison_quantifies_both_frozen_routes(self) -> None:
        """EKF/LPF 对照给出真值误差与覆盖率 / Quantify both gravity routes."""

        completed, payload = run_cli(
            "compare", "imu-gravity", "--duration-s", "8", "--seed", "42"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(
            {row["canonical_module_id"] for row in payload["results"]},
            {"ekf_no_precalibration", "lowpass_gravity_0p3hz"},
        )
        for row in payload["results"]:
            self.assertGreater(row["coverage"], 0.0)
            self.assertGreaterEqual(row["gravity_rmse_mps2"], 0.0)
            self.assertGreaterEqual(row["dynamic_mae_mps2"], 0.0)
        self.assertIn("synthetic_known_truth", payload["scientific_scope"])

    def test_reduced_model_comparison_fits_and_forwards(self) -> None:
        """同一命令实际 forward CNN 并拟合 L2 / Forward and fit real models."""

        completed, payload = run_cli(
            "compare",
            "models",
            "--models",
            "CompactCNN1D",
            "LogisticRegressionL2",
            "ROCKET",
            "FileBagFusionCompact",
            "--seed",
            "42",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["status"], "passed")
        kinds = {row["model_id"]: row["quantitation_kind"] for row in payload["results"]}
        self.assertEqual(kinds["LogisticRegressionL2"], "reduced_synthetic_fit")
        self.assertEqual(kinds["CompactCNN1D"], "forward_contract_probability_sum_error")
        self.assertEqual(
            {row["representation_mode"] for row in payload["results"]},
            {"raw", "feature_vector", "feature_matrix", "fusion"},
        )
        indexed = {row["canonical_model_id"]: row for row in payload["results"]}
        self.assertEqual(indexed["ROCKET"]["machine_model_id"], "rocket_numpy")
        self.assertEqual(indexed["ROCKET"]["n_kernels"], 64)
        self.assertEqual(indexed["CompactCNN1D"]["variant"], "reviewed_compact")

    def test_shapeformer_machine_id_builds_strict_discovery_provenance(self) -> None:
        """machine ID 直达仍须构建完整 outer-fold provenance / Strict bank."""

        completed, payload = run_cli(
            "compare", "models", "--models", "shapeformer_effect_size_fixed_v1", "--seed", "42"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(len(payload["results"]), 1)
        row = payload["results"][0]
        self.assertEqual(row["canonical_model_id"], "ShapeFormerEffectSizeFixedV1")
        self.assertEqual(row["machine_model_id"], "shapeformer_effect_size_fixed_v1")
        self.assertEqual(row["discovery_method"], "effect_size_fixed_v1")
        self.assertEqual(row["model_status"], "experimental")
        self.assertTrue(row["finite_probabilities"])

    def test_run_experiment_dispatch_preserves_public_budgets_and_cleans_temp(self) -> None:
        """mock dispatch 验证预算隔离；仅清理精确 V2 temp / Dispatch budgets."""

        from ppg_frailty.cli import main
        from ppg_frailty.experiment import ExperimentResult

        temporary_root = (ROOT / "artifacts" / "tmp").resolve()
        temporary_root.mkdir(parents=True, exist_ok=True)
        created_path: Path | None = None
        with tempfile.TemporaryDirectory(
            prefix="ppg_frailty_v2_experiment_cli_", dir=temporary_root
        ) as directory:
            created_path = Path(directory).resolve()
            created_path.relative_to(temporary_root)
            reduced_target = created_path / "reduced_result"
            full_target = created_path / "full_result"

            def outcome(scope: str, target: Path) -> ExperimentResult:
                """构造 strict mock 结论 / Build a strict mocked outcome."""

                return ExperimentResult(
                    status="passed",
                    scientific_scope=scope,
                    config_id="reference_static_feature_vector_lr_line_a_v2",
                    config_hash="config_hash",
                    repeat_indices=(2,),
                    fold_indices=(3,),
                    output_dir=str(target),
                )

            with patch(
                "ppg_frailty.experiment.run_reduced_fold_experiment",
                return_value=outcome("smoke_not_scientific_benchmark", reduced_target),
            ) as reduced, io.StringIO() as stream, redirect_stdout(stream):
                code = main(
                    [
                        "run-experiment", "--config", "default",
                        "--budget", "reduced-smoke", "--repeat", "2", "--fold", "3",
                        "--output-dir", reduced_target.relative_to(ROOT).as_posix(),
                    ]
                )
                payload = json.loads(stream.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["scientific_scope"], "smoke_not_scientific_benchmark")
            reduced.assert_called_once_with(
                "configs/reference_static_feature_vector_v2.yaml",
                repeat_index=2,
                fold_index=3,
                output_dir=reduced_target.relative_to(ROOT).as_posix(),
            )

            with patch(
                "ppg_frailty.experiment.run_full_experiment",
                return_value=outcome("frozen_5x5_scientific_benchmark", full_target),
            ) as full, io.StringIO() as stream, redirect_stdout(stream):
                code = main(
                    [
                        "run-experiment", "--config", "default",
                        "--budget", "full",
                        "--output-dir", full_target.relative_to(ROOT).as_posix(),
                    ]
                )
                self.assertEqual(code, 2)
                full.assert_not_called()
            with patch(
                "ppg_frailty.experiment.run_full_experiment",
                return_value=outcome("frozen_5x5_scientific_benchmark", full_target),
            ) as full, io.StringIO() as stream, redirect_stdout(stream):
                code = main(
                    [
                        "run-experiment", "--config", "default",
                        "--budget", "full",
                        "--output-dir", full_target.relative_to(ROOT).as_posix(),
                        "--confirm-scientific-execution",
                    ]
                )
                payload = json.loads(stream.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["scientific_scope"], "frozen_5x5_scientific_benchmark")
            # English: Absence of repeats/folds delegates the public 5x5 defaults;
            # no shortened-record or epoch option can enter this call.
            # 中文：不传 repeats/folds 即沿用公共 5×5 默认，且调用中不存在记录
            # 截短或 epoch override 参数。
            full.assert_called_once_with(
                "configs/reference_static_feature_vector_v2.yaml",
                output_dir=full_target.relative_to(ROOT).as_posix(),
                measure_operational_costs=False,
                confirm_scientific_execution=True,
            )
        self.assertIsNotNone(created_path)
        self.assertFalse(created_path.exists())

    def test_smoke_output_can_use_and_clean_exact_v2_temporary_directory(self) -> None:
        """仅清理由本测试创建的 V2 temp / Clean only the exact V2 temp created here."""

        temporary_root = (ROOT / "artifacts" / "tmp").resolve()
        temporary_root.mkdir(parents=True, exist_ok=True)
        created_path: Path | None = None
        with tempfile.TemporaryDirectory(
            prefix="ppg_frailty_v2_cli_", dir=temporary_root
        ) as directory:
            created_path = Path(directory).resolve()
            created_path.relative_to(temporary_root)
            output_path = created_path / "smoke.json"
            completed, payload = run_cli(
                "run",
                "--config",
                "default",
                "--mode",
                "smoke",
                "--output",
                output_path.relative_to(ROOT).as_posix(),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output_path.is_file())
            self.assertEqual(payload["status"], "smoke_passed")
            persisted = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIs(persisted["scientific_metrics_emitted"], False)
        self.assertIsNotNone(created_path)
        self.assertFalse(created_path.exists())


if __name__ == "__main__":
    unittest.main()

"""V2 活动身份验收 / Active V2 identity acceptance."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class V2IdentityAcceptanceTests(unittest.TestCase):
    """防止 V1 copied snapshot 变成活动身份 / Prevent copied V1 activation."""

    def test_packaging_and_formal_configs_have_v2_identity(self) -> None:
        """活动 metadata/config 全是 V2 / Active metadata and configs are V2."""

        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(project["name"], "ppg-frailty-final-pipeline-v2")
        self.assertTrue(str(project["version"]).startswith("2."))
        formal_names = {
            "reference_static_feature_vector_v2.yaml",
            "reference_static_feature_vector_line_b_v2.yaml",
            "reference_static_feature_matrix_v2.yaml",
            "reference_static_fusion_v2.yaml",
            "reference_static_line_a_v2.yaml",
        }
        formal = [ROOT / "configs" / name for name in sorted(formal_names)]
        self.assertTrue(all(path.is_file() for path in formal))
        for path in formal:
            text = path.read_text(encoding="utf-8")
            self.assertIn("schema_version: ppg_frailty.pipeline_config.v2", text)
        motion = ROOT / "configs/motion_detector_contract_v2.yaml"
        self.assertIn(
            "schema_version: ppg_frailty.motion_detector_contract.v2",
            motion.read_text(encoding="utf-8"),
        )

    def test_validator_exports_only_v2_identity(self) -> None:
        """validator 输出 V2 schema / Validator emits a V2 schema."""

        completed = subprocess.run(
            [sys.executable, "-B", "tools/validate_v2.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "ppg_frailty.v2_validation.v2")
        self.assertEqual(payload["pipeline_generation"], "final_pipeline_v2")
        self.assertEqual(payload["status"], "passed")


if __name__ == "__main__":
    unittest.main()

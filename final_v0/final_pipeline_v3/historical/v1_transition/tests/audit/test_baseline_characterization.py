"""Executable guards for the frozen dev0 baseline and historical evidence.

English: The audit JSON files are evidence only when their file fingerprints and
scientific eligibility flags are reproducible.  These tests re-read the immutable
root sources as bytes and prevent historical, non-strict scores from entering V1.

中文：只有当源文件指纹和科学资格标记可复算时，审计 JSON 才是有效证据。
这些测试按字节重新读取只读根目录源文件，并阻止非严格历史分数进入 V1 排名。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


V1_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = V1_ROOT.parents[1]
AUDIT_ROOT = V1_ROOT / "artifacts" / "audit"


def _sha256(path: Path) -> str:
    """Hash one file as a byte stream / 以字节流复算单个文件哈希。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BaselineCharacterizationTests(unittest.TestCase):
    """Keep baseline evidence immutable and scientifically non-promotional.

    中文：同时检查物理文件身份和历史结果资格，避免“文件还在”被误解为
    “结果可进入 corrected leaderboard”。
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Load strict JSON once / 只加载一次严格 JSON。"""

        cls.inventory = json.loads(
            (AUDIT_ROOT / "baseline_inventory.json").read_text(encoding="utf-8")
        )
        cls.characterization = json.loads(
            (AUDIT_ROOT / "legacy_characterization.json").read_text(encoding="utf-8")
        )

    def test_every_frozen_source_fingerprint_matches_live_bytes(self) -> None:
        """Root history remains byte-identical / 根历史源文件必须保持逐字节一致。"""

        fingerprints = self.inventory["source_fingerprints"]
        self.assertGreaterEqual(len(fingerprints), 9)
        for relative_path, expected in fingerprints.items():
            source = REPOSITORY_ROOT / relative_path
            with self.subTest(relative_path=relative_path):
                self.assertTrue(source.is_file())
                self.assertEqual(source.stat().st_size, int(expected["bytes"]))
                self.assertEqual(_sha256(source), expected["sha256"])

    def test_manifest_fold_and_cohort_identity_are_frozen(self) -> None:
        """Audit binds the accepted M2 roster / 审计绑定已接受的 M2 roster。"""

        internal = self.inventory["internal_dataset"]
        self.assertEqual(internal["recordings"], 261)
        self.assertEqual(internal["participants"], 29)
        self.assertEqual(internal["class_participants"], {
            "Pre-Frail": 9,
            "Robust/Non-Frail": 12,
            "Young": 8,
        })
        self.assertEqual(set(internal["roles"]), {
            "B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2"
        })
        self.assertTrue(all(value == 29 for value in internal["roles"].values()))
        self.assertEqual(
            internal["fold_registry_payload_sha256"],
            "0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46",
        )

    def test_historical_scores_are_never_v1_eligible(self) -> None:
        """Known old metrics are characterization only / 旧指标只能用于历史刻画。"""

        metrics = self.characterization["known_metrics"]
        self.assertGreaterEqual(len(metrics), 4)
        self.assertTrue(all(item["eligible"] is False for item in metrics))
        self.assertTrue(all(item["reason"] for item in metrics))
        self.assertIn("historical_non_strict", self.characterization["status"])

    def test_reviewed_architecture_counts_are_recorded(self) -> None:
        """Snapshot names/counts cannot drift silently / 模型名称与参数数不得静默漂移。"""

        snapshots = self.characterization["network_snapshots"]
        self.assertEqual(snapshots["CompactCNN1D"]["parameters"], 79_139)
        self.assertEqual(
            snapshots["InceptionTimeFull_single_network"]["parameters"], 456_579
        )
        self.assertEqual(
            snapshots["InceptionTimeSmall_single_network"]["parameters"], 57_027
        )
        self.assertEqual(
            snapshots["CompactCNN1D"]["origin_name"], "Cnn1DClassifier"
        )


if __name__ == "__main__":
    unittest.main()

"""冻结 internal fold 合同测试 / Frozen internal fold contract tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from ppg_frailty.data import (  # noqa: E402
    FrozenFoldRegistry,
    load_frozen_memberships,
    load_m2_internal_manifest,
    materialize_assignments,
)
from ppg_frailty.data.folds import (  # noqa: E402
    M2_SPLIT_REGISTRY_ID,
    M2_SPLIT_RELATIVE_PATH,
    validate_frozen_memberships,
)


class FrozenFoldTests(unittest.TestCase):
    """确保 outer folds 复制而非重算 / Ensure outer folds are copied."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_m2_internal_manifest(
            REPOSITORY_ROOT,
            verify_sources=False,
        )
        cls.registry = load_frozen_memberships(
            REPOSITORY_ROOT / M2_SPLIT_RELATIVE_PATH
        )
        cls.assignments = materialize_assignments(cls.registry, cls.manifest)

    def test_frozen_registry_invariants(self) -> None:
        audit = validate_frozen_memberships(self.registry, self.manifest)
        self.assertEqual(audit.registry_id, M2_SPLIT_REGISTRY_ID)
        self.assertEqual(audit.participant_count, 29)
        self.assertEqual(audit.repeat_count, 5)
        self.assertEqual(audit.assignment_count, 145)
        self.assertTrue(audit.all_classes_present)
        self.assertTrue(audit.class_balance_spread_at_most_one)
        self.assertTrue(audit.train_oof_disjoint)
        self.assertTrue(audit.oof_partition_exact)

    def test_registry_resolves_exact_partition(self) -> None:
        registry = FrozenFoldRegistry(tuple(self.assignments))
        split = registry.get_split(repeat_index=0, fold_index=0)
        train = set(split["train_participant_ids"])
        oof = set(split["oof_participant_ids"])
        self.assertFalse(train & oof)
        self.assertEqual(train | oof, set(registry.participant_ids))
        self.assertEqual(len(train) + len(oof), 29)

    def test_modified_registry_fails_file_identity(self) -> None:
        """任何 JSON 变更都必须失败 / Any JSON mutation must fail identity."""

        source = REPOSITORY_ROOT / M2_SPLIT_RELATIVE_PATH
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["runtime_split_recomputation_allowed"] = True
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.json"
            changed.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_frozen_memberships(changed)


if __name__ == "__main__":
    unittest.main()

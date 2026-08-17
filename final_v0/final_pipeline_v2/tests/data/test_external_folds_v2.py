"""Focused tests for the frozen formal PTT repeated grouped registry."""

from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from ppg_frailty.data.external_folds import (
    PTT_FORMAL_ALGORITHM,
    PTT_FORMAL_FOLD_SIZES,
    PTT_FORMAL_REPEAT_SEEDS,
    build_formal_ptt_fold_rows,
    load_formal_ptt_repeated_folds,
    materialize_formal_ptt_repeated_folds,
    resolve_formal_ptt_split,
)
from ppg_frailty.data.external_manifest import (
    M2_EXTERNAL_RELATIVE_PATH,
    load_m2_external_manifest,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]


class FormalPttFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_m2_external_manifest(
            REPOSITORY_ROOT / M2_EXTERNAL_RELATIVE_PATH
        )
        cls.rows = build_formal_ptt_fold_rows(cls.records)

    def test_repeat_seed_sizes_and_activity_balance(self) -> None:
        self.assertEqual(len(self.rows), 110)
        for repeat_index, seed in enumerate(PTT_FORMAL_REPEAT_SEEDS):
            repeat = [
                row for row in self.rows if int(row["repeat_index"]) == repeat_index
            ]
            self.assertEqual({int(row["split_seed"]) for row in repeat}, {seed})
            counts = Counter(int(row["fold_index"]) for row in repeat)
            self.assertEqual(
                tuple(counts[index] for index in range(5)),
                PTT_FORMAL_FOLD_SIZES,
            )
            self.assertEqual(len({row["subject_id"] for row in repeat}), 22)
            self.assertTrue(
                all(
                    row["activity_raw"] == '["run","sit","walk"]'
                    for row in repeat
                )
            )
            self.assertTrue(
                all(row["assignment_algorithm"] == PTT_FORMAL_ALGORITHM for row in repeat)
            )
            self.assertNotIn("stratified", PTT_FORMAL_ALGORITHM)
            self.assertNotIn("class", set(repeat[0]))

    def test_every_resolved_split_is_disjoint_and_exhaustive(self) -> None:
        all_subjects = {row["subject_id"] for row in self.rows}
        for repeat_index in range(5):
            for fold_index, expected_size in enumerate(PTT_FORMAL_FOLD_SIZES):
                split = resolve_formal_ptt_split(
                    self.rows,
                    repeat_index=repeat_index,
                    fold_index=fold_index,
                )
                train = set(split["train_subject_ids"])
                oof = set(split["oof_subject_ids"])
                self.assertFalse(train & oof)
                self.assertEqual(train | oof, all_subjects)
                self.assertEqual(len(oof), expected_size)

    def test_materialized_roundtrip_never_recomputes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "ptt_formal.csv"
            written = materialize_formal_ptt_repeated_folds(
                self.records,
                target,
                output_root=directory,
            )
            loaded = load_formal_ptt_repeated_folds(target)
        self.assertEqual(written, loaded)
        self.assertTrue(
            all(
                row["runtime_split_recomputation_allowed"] == "false"
                for row in loaded
            )
        )


if __name__ == "__main__":
    unittest.main()

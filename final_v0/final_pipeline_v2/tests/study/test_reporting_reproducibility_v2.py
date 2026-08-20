from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from ppg_frailty.reporting.reproducibility import (
    FAIL,
    NOT_VERIFIABLE,
    PASS,
    audit_study_reproducibility,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _study(tmp_path: Path) -> tuple[SimpleNamespace, dict[tuple[str, int], Path]]:
    root = tmp_path / "study"
    split_path = root / "splits" / "folds.csv"
    split_path.parent.mkdir(parents=True)
    participants = ("p0", "p1", "p2", "p3")
    with split_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("repeat_index", "fold_index", "split_seed", "participant_id"),
        )
        writer.writeheader()
        for index, participant in enumerate(participants):
            writer.writerow(
                {
                    "repeat_index": 0,
                    "fold_index": index % 2,
                    "split_seed": 42,
                    "participant_id": participant,
                }
            )
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    registry_payload_sha = "a" * 64
    specifications = {
        "deep": ("outer_cv_repeat_seed_equals_split_seed", 42, (), True),
        "ensemble": ("final_refit_five_member_seeds", 42, (11, 12, 13, 14, 15), True),
        "classical": ("outer_cv_repeat_seed_equals_split_seed", 42, (), False),
        "comparator": ("cv_fixed_member0_seed_50042_comparator", 50042, (), False),
    }
    cases, case_records, cell_rows, subject_rows, history_rows, paths = [], [], [], [], [], {}
    for case_id, (policy, training_seed, members, iterative) in specifications.items():
        case_dir = root / "raw" / case_id
        artifact = case_dir / "attempts" / "attempt_002" / "experiment"
        (case_dir / "attempts" / "attempt_001").mkdir(parents=True)
        config = {
            "splits": {
                "path": "splits/folds.csv",
                "source_registry_file_sha256": "b" * 64,
                "source_registry_payload_sha256": registry_payload_sha,
            },
            "model": {
                "seed_policy": policy,
                **({"member_seeds": list(members)} if members else {}),
            },
            "evaluation": {"statistics": {"seed": 73}},
        }
        config_path = case_dir / "resolved_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        cases.append(
            {
                "case_id": case_id,
                "case_directory": f"raw/{case_id}",
                "resolved_config_path": f"raw/{case_id}/resolved_config.yaml",
            }
        )
        case_records.append(
            {
                "case_id": case_id,
                "status": "passed",
                "attempt": 2,
                "artifact_root": "attempts/attempt_002/experiment",
            }
        )
        for fold in (0, 1):
            oof_ids = tuple(participants[fold::2])
            train_ids = tuple(sorted(set(participants) - set(oof_ids)))
            model_seeds = members or (training_seed,)
            cell = {
                "status": "passed",
                "repeat_index": 0,
                "fold_index": fold,
                "split_seed": 42,
                "training_seed": training_seed,
                "training_orchestration_seed": 42,
                "member_training_seeds": list(members),
                "seed_policy": policy,
                "model_machine_id": case_id,
                "evaluation_policy": {"statistics": {"seed": 73}},
                "fitted_provenance": {
                    "fold_hash": split_sha,
                    "registry_hash": registry_payload_sha,
                    "fitted_participant_ids": list(train_ids),
                    "selected_epoch": 2 if iterative else None,
                },
                "frozen_model_run_provenance": {
                    "fold_hash": split_sha,
                    "random_seeds": list(model_seeds),
                    "seed_policy": policy,
                },
            }
            cell_dir = artifact / f"repeat_00_fold_{fold:02d}"
            run_path = cell_dir / "run_manifest.json"
            _write_json(run_path, {"cell": cell})
            _write_json(
                cell_dir / "experiment_result.json",
                {"provenance": {"manifest_hash": "c" * 64, "fold_hash": split_sha}},
            )
            paths[(case_id, fold)] = run_path
            cell_rows.append(
                {
                    "case_id": case_id,
                    "repeat": 0,
                    "fold": fold,
                    "split_seed": 42,
                    "training_seed": training_seed,
                }
            )
            subject_rows.extend(
                {"case_id": case_id, "repeat": 0, "fold": fold, "participant_id": value}
                for value in oof_ids
            )
            if iterative:
                for member, seed in enumerate(model_seeds):
                    for epoch in (1, 2):
                        history_rows.append(
                            {
                                "case_id": case_id,
                                "repeat": 0,
                                "fold": fold,
                                "member": member,
                                "epoch": epoch,
                                "training_seed": seed,
                                "epoch_rng_seed": seed + epoch * 1_000_000,
                            }
                        )
    collected = SimpleNamespace(
        root=root,
        plan={},
        manifest={"execution": {"repeats": [0], "folds": [0, 1]}, "cases": cases},
        case_records=tuple(case_records),
        cell_rows=tuple(cell_rows),
        subject_oof_rows=tuple(subject_rows),
        history_rows=tuple(history_rows),
    )
    return collected, paths


class ReproducibilityAuditTests(unittest.TestCase):
    def test_selected_single_ensemble_and_classical_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = audit_study_reproducibility(_study(Path(directory))[0])

        self.assertEqual(result.status, PASS)
        self.assertEqual(result.summary["observed_cell_count"], 8)
        self.assertEqual(len(result.split_rows), 2)
        self.assertTrue(
            all(row["excluded_attempts"] == ["attempt_001"] for row in result.case_rows)
        )
        ensemble = next(row for row in result.cell_rows if row["case_id"] == "ensemble")
        classical = next(row for row in result.cell_rows if row["case_id"] == "classical")
        self.assertEqual(ensemble["member_training_seeds"], [11, 12, 13, 14, 15])
        self.assertTrue(classical["member_seed_semantics"].startswith("N/A_single_model"))

    def test_current_cell_drift_fails_but_old_attempt_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            collected, paths = _study(Path(directory))
            payload = json.loads(paths[("deep", 0)].read_text(encoding="utf-8"))
            payload["cell"]["split_seed"] = 99
            payload["cell"]["fitted_provenance"]["training_seed"] = 999
            paths[("deep", 0)].write_text(json.dumps(payload), encoding="utf-8")
            old = collected.root / "raw/deep/attempts/attempt_001/run_manifest.json"
            _write_json(
                old,
                {"cell": {"repeat_index": 0, "fold_index": 0, "split_seed": 999}},
            )
            collected.history_rows = tuple(
                row for row in collected.history_rows
                if not (row["case_id"] == "deep" and row["fold"] == 0 and row["epoch"] == 2)
            )

            result = audit_study_reproducibility(collected)

        self.assertEqual(result.status, FAIL)
        codes = {row["code"] for row in result.issues}
        self.assertIn("cell_evidence_drift", codes)
        self.assertIn("cell_split_seed_drift", codes)
        self.assertIn("cross_case_split_drift", codes)
        self.assertIn("epoch_seed_roster_incomplete", codes)
        self.assertTrue(all("999" not in str(row) for row in result.cell_rows))

    def test_missing_historical_orchestration_seed_is_not_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            collected, paths = _study(Path(directory))
            payload = json.loads(paths[("classical", 0)].read_text(encoding="utf-8"))
            del payload["cell"]["training_orchestration_seed"]
            paths[("classical", 0)].write_text(json.dumps(payload), encoding="utf-8")

            result = audit_study_reproducibility(collected)

        self.assertEqual(result.status, NOT_VERIFIABLE)
        self.assertEqual(
            {row["code"] for row in result.issues}, {"seed_semantics_incomplete"}
        )

    def test_missing_selected_cell_is_not_counted_and_marks_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            collected, paths = _study(Path(directory))
            paths[("deep", 0)].unlink()
            collected.cell_rows = tuple(
                row for row in collected.cell_rows
                if not (row["case_id"] == "deep" and row["fold"] == 0)
            )
            collected.subject_oof_rows = tuple(
                row for row in collected.subject_oof_rows
                if not (row["case_id"] == "deep" and row["fold"] == 0)
            )

            result = audit_study_reproducibility(collected)

        deep = next(row for row in result.case_rows if row["case_id"] == "deep")
        self.assertEqual(result.status, FAIL)
        self.assertEqual(result.summary["observed_cell_count"], 7)
        self.assertEqual(deep["observed_cell_count"], 1)
        self.assertEqual(deep["audit_status"], FAIL)

    def test_invalid_manifest_and_missing_registry_fold_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            collected, _ = _study(Path(directory))
            empty = SimpleNamespace(**{**collected.__dict__, "manifest": {"cases": []}})
            self.assertEqual(audit_study_reproducibility(empty).status, FAIL)

            split_path = collected.root / "splits/folds.csv"
            rows = list(csv.DictReader(split_path.read_text(encoding="utf-8").splitlines()))
            with split_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(row for row in rows if row["fold_index"] == "0")
            result = audit_study_reproducibility(collected)

        self.assertEqual(result.status, FAIL)
        self.assertIn(
            "split_registry_planned_roster_missing",
            {row["code"] for row in result.issues},
        )

"""Focused contracts for Stage 5 OOM-safe artifact recovery."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ppg_frailty.study import ExecutionSpec, ResolvedCase, StudyRunner
from ppg_frailty.study.recovery import _externalize_legacy_sqi_payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class LegacySqiRecoveryTests(unittest.TestCase):
    def test_externalize_preserves_full_evidence_and_compacts_report_payloads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary)
            full_window_evidence = {
                "window_000": {
                    "direct": {
                        "state": "positive",
                        "reasons": ["rate_and_morph_passed"],
                        "q_rate": {
                            "state": "positive",
                            "score": 0.82,
                            "coverage": 0.91,
                            "threshold": 0.50,
                            "reasons": ["enough_valid_intervals"],
                            "components": {
                                "retained_imu_motion_energy": 0.173,
                                "valid_interval_count": 9,
                            },
                        },
                        "q_morph": {
                            "state": "positive",
                            "score": 0.74,
                            "coverage": 0.88,
                            "threshold": 0.65,
                            "reasons": [],
                            "components": {"template_correlation": 0.86},
                        },
                    },
                    "post_reduction": None,
                }
            }
            expected_full_evidence = copy.deepcopy(full_window_evidence)
            route_payload = {
                "schema_version": "ppg_frailty.route_artifacts.v2",
                "repeat_index": 0,
                "fold_index": 0,
                "rows": [
                    {
                        "record_id": "P001_B",
                        "participant_id": "P001",
                        "role": "B",
                        "route_artifact": {
                            "route_status": "retained_excellent",
                            "native_window_sqi_evidence": full_window_evidence,
                        },
                    }
                ],
            }
            quality_payload = {
                "schema_version": "ppg_frailty.quality_diagnostics.v2",
                "rows": [
                    {
                        "record_id": "P001_B",
                        "participant_id": "P001",
                        "role": "B",
                        "retained": True,
                        "route_artifact": copy.deepcopy(
                            route_payload["rows"][0]["route_artifact"]
                        ),
                    }
                ],
            }
            original_summary = {
                "status": "passed",
                "repeat_index": 0,
                "fold_index": 0,
                "scientific_scope": "frozen_5x5_scientific_benchmark",
                "quality_diagnostic_row_count": 1,
                "route_artifacts_row_count": 1,
            }
            metrics_payload = {
                "schema_version": "ppg_frailty.metrics_per_fold_seed.v2",
                "cells": [copy.deepcopy(original_summary)],
            }
            manifest_payload = {
                "schema_version": "ppg_frailty.run_manifest.v2",
                "status": "passed",
                "cell": copy.deepcopy(original_summary),
                "mandatory_artifacts": [
                    "run_manifest.json",
                    "metrics_per_fold_seed.json",
                    "quality_diagnostics.json",
                    "route_artifacts.json",
                ],
            }
            _write_json(cell / "route_artifacts.json", route_payload)
            _write_json(cell / "quality_diagnostics.json", quality_payload)
            _write_json(cell / "metrics_per_fold_seed.json", metrics_payload)
            _write_json(cell / "run_manifest.json", manifest_payload)

            summary, returned_quality = _externalize_legacy_sqi_payload(
                cell,
                config_id="stage5_case",
                config_hash="a" * 64,
            )

            evidence_path = cell / "route_window_sqi_evidence.jsonl.gz"
            self.assertTrue(evidence_path.is_file())
            with gzip.open(evidence_path, "rt", encoding="utf-8") as stream:
                evidence_rows = [json.loads(line) for line in stream]
            self.assertEqual(
                evidence_rows[0]["schema_version"],
                "ppg_frailty.route_window_sqi_evidence.v1",
            )
            self.assertEqual(evidence_rows[0]["record_type"], "header")
            self.assertFalse(evidence_rows[0]["report_consumed"])
            self.assertEqual(len(evidence_rows), 2)
            self.assertEqual(evidence_rows[1]["record_type"], "window_evidence")
            self.assertEqual(evidence_rows[1]["record_id"], "P001_B")
            self.assertEqual(
                evidence_rows[1]["evidence"],
                expected_full_evidence["window_000"],
            )

            compact_route = json.loads(
                (cell / "route_artifacts.json").read_text(encoding="utf-8")
            )["rows"][0]["route_artifact"]["native_window_sqi_evidence"]
            direct = compact_route["window_000"]["direct"]
            self.assertEqual(direct["q_rate"]["score"], 0.82)
            self.assertEqual(direct["q_morph"]["threshold"], 0.65)
            self.assertNotIn("components", direct["q_rate"])
            self.assertNotIn("components", direct["q_morph"])

            persisted_quality = json.loads(
                (cell / "quality_diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("route_artifact", persisted_quality["rows"][0])
            self.assertEqual(returned_quality, persisted_quality)
            self.assertTrue(persisted_quality["rows"][0]["retained"])

            evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            persisted_summary = json.loads(
                (cell / "metrics_per_fold_seed.json").read_text(encoding="utf-8")
            )["cells"][0]
            self.assertEqual(summary, persisted_summary)
            self.assertEqual(summary["config_id"], "stage5_case")
            self.assertEqual(summary["config_hash"], "a" * 64)
            self.assertEqual(summary["canonical_config_hash"], "a" * 64)
            self.assertEqual(summary["route_window_sqi_evidence_row_count"], 1)
            self.assertEqual(
                summary["route_window_sqi_evidence_sha256"], evidence_sha256
            )

            manifest = json.loads(
                (cell / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn(
                "route_window_sqi_evidence.jsonl.gz",
                manifest["mandatory_artifacts"],
            )
            self.assertEqual(manifest["cell"], summary)


class InterruptedStudyRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case_directory = self.root / "case"
        self.attempt = self.case_directory / "attempts" / "attempt_001"
        self.staging = self.attempt / ".experiment.staging.123"
        self.staging.mkdir(parents=True)
        self.config_path = self.root / "resolved_config.yaml"
        self.config_path.write_text("config_id: stage5_case\n", encoding="utf-8")
        self.case = ResolvedCase(
            case_id="stage5_case",
            config={"config_id": "stage5_case"},
            changed_values={},
            config_sha256="b" * 64,
            is_reference=False,
        )

    @staticmethod
    def _plan(
        *,
        repeats: tuple[int, ...] = tuple(range(5)),
        folds: tuple[int, ...] = tuple(range(5)),
        legacy_bridge: object | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            execution=ExecutionSpec(
                repeats=repeats,
                folds=folds,
                measure_operational_costs=False,
            ),
            legacy_bridge=legacy_bridge,
        )

    def test_canonical_complete_staging_is_finalized_and_indexed(self) -> None:
        runner = StudyRunner(pipeline_root=self.root)
        recovered_result = {
            "status": "passed",
            "scientific_scope": "frozen_5x5_scientific_benchmark",
            "config_id": "stage5_case",
            "config_hash": "b" * 64,
            "repeat_indices": list(range(5)),
            "fold_indices": list(range(5)),
            "output_dir": str(self.attempt / "experiment"),
            "cell_results": [],
            "metrics": {"passed_cell_count": 25},
        }

        def recover_side_effect(*args: object, **kwargs: object) -> object:
            del args
            Path(kwargs["output_dir"]).mkdir()
            return recovered_result

        with patch(
            "ppg_frailty.study.recovery.recover_completed_full_experiment_staging",
            side_effect=recover_side_effect,
        ) as recover:
            record = runner._recover_complete_interrupted_pass(
                self.case,
                self.config_path,
                self.case_directory,
                self._plan(),
            )

        self.assertIsNotNone(record)
        self.assertEqual(record["status"], "passed")
        self.assertEqual(record["attempt"], 1)
        self.assertEqual(record["artifact_root"], "attempts/attempt_001/experiment")
        self.assertTrue(record["recovered_from_complete_interrupted_staging"])
        self.assertFalse(record["recovery_index_only"])
        self.assertEqual(
            record["interrupted_staging_preserved"],
            "attempts/attempt_001/.experiment.staging.123",
        )
        self.assertTrue(self.staging.is_dir())
        self.assertTrue((self.attempt / "attempt_result.json").is_file())
        persisted = json.loads(
            (self.case_directory / "case_result.json").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted["config_sha256"], "b" * 64)
        self.assertEqual(persisted["result"]["metrics"]["passed_cell_count"], 25)
        recover.assert_called_once()
        call_args, call_kwargs = recover.call_args
        self.assertEqual(call_args, (self.config_path,))
        self.assertEqual(call_kwargs["interrupted_staging"], self.staging)
        self.assertEqual(call_kwargs["output_dir"], self.attempt / "experiment")
        self.assertEqual(call_kwargs["repeats"], tuple(range(5)))
        self.assertEqual(call_kwargs["folds"], tuple(range(5)))
        self.assertFalse(call_kwargs["measure_operational_costs"])
        self.assertTrue(callable(call_kwargs["progress_callback"]))

    def test_published_recovery_is_indexed_without_repeating_recovery(self) -> None:
        published = self.attempt / "experiment"
        published.mkdir()
        recovered_result = {
            "status": "passed",
            "scientific_scope": "frozen_5x5_scientific_benchmark",
            "config_id": "stage5_case",
            "config_hash": "b" * 64,
            "repeat_indices": list(range(5)),
            "fold_indices": list(range(5)),
            "output_dir": str(published),
            "cell_results": [],
            "metrics": {"passed_cell_count": 25},
        }
        runner = StudyRunner(pipeline_root=self.root)
        with (
            patch(
                "ppg_frailty.study.recovery."
                "validate_published_recovered_experiment",
                return_value=recovered_result,
            ) as validate,
            patch(
                "ppg_frailty.study.recovery."
                "recover_completed_full_experiment_staging"
            ) as recover,
        ):
            record = runner._recover_complete_interrupted_pass(
                self.case,
                self.config_path,
                self.case_directory,
                self._plan(),
            )

        self.assertIsNotNone(record)
        self.assertTrue(record["recovery_index_only"])
        self.assertEqual(record["artifact_root"], "attempts/attempt_001/experiment")
        validate.assert_called_once()
        recover.assert_not_called()

    def test_recovery_is_rejected_outside_canonical_full_5x5_runner(self) -> None:
        canonical_runner = StudyRunner(pipeline_root=self.root)
        custom_runner = StudyRunner(pipeline_root=self.root, executor=Mock())
        noncanonical_plans = (
            self._plan(repeats=(0,)),
            self._plan(folds=(0, 1, 2, 3)),
            self._plan(legacy_bridge=object()),
        )
        with patch(
            "ppg_frailty.study.recovery.recover_completed_full_experiment_staging"
        ) as recover:
            for plan in noncanonical_plans:
                with self.subTest(plan=plan):
                    self.assertIsNone(
                        canonical_runner._recover_complete_interrupted_pass(
                            self.case,
                            self.config_path,
                            self.case_directory,
                            plan,
                        )
                    )
            self.assertIsNone(
                custom_runner._recover_complete_interrupted_pass(
                    self.case,
                    self.config_path,
                    self.case_directory,
                    self._plan(),
                )
            )
        recover.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

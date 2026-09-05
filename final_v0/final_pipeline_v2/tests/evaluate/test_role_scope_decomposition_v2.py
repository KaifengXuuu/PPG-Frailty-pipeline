from __future__ import annotations

import hashlib
from importlib.util import find_spec
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ppg_frailty.evaluate.role_scope_decomposition import (
    load_role_scope_plan,
    run_role_scope_decomposition,
)
from ppg_frailty.training import read_oof_parquet, write_oof_parquet
from ppg_frailty.training.aggregation import (
    LINE_B_EQUAL_ROLE_FAMILIES,
    aggregate_hierarchy,
)
from ppg_frailty.training.oof import OofPredictionRow


HAS_PYARROW = find_spec("pyarrow") is not None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(HAS_PYARROW, "role-scope integration test requires pyarrow")
class RoleScopeDecompositionTest(unittest.TestCase):
    def _probability(self, label: int, participant_suffix: str, role: str) -> tuple[float, ...]:
        if role in {"S", "W"}:
            values = [0.025, 0.025, 0.025]
            values[label] = 0.95
            return tuple(values)
        if participant_suffix == "b":
            values = [0.10, 0.10, 0.10]
            values[label] = 0.80
            return tuple(values)
        values = [0.05, 0.05, 0.05]
        values[label] = 0.35
        values[(label + 1) % 3] = 0.60
        return tuple(values)

    def _file_rows(self, source: str, roles: tuple[str, ...]) -> tuple[OofPredictionRow, ...]:
        output = []
        for repeat in (0, 1):
            for label in (0, 1, 2):
                for suffix in ("a", "b"):
                    participant = f"p{label}{suffix}"
                    fold = (2 * label + (suffix == "b")) % 2
                    for role in roles:
                        output.append(
                            OofPredictionRow(
                                participant_id=participant,
                                file_id=f"{participant}_{role}_{repeat}",
                                role=role,
                                label=label,
                                probabilities=self._probability(label, suffix, role),
                                repeat=repeat,
                                fold=fold,
                                split_seed=42 + 10_000 * repeat,
                                training_seed=42,
                                config_hash=f"config-{source}",
                                manifest_hash="manifest-shared",
                                fold_hash=f"fold-{repeat}-{fold}",
                                preprocessing_hash="preprocessing-shared",
                                feature_hash="feature-shared",
                                model_hash=f"model-{source}-{repeat}-{fold}",
                                representation_mode="raw",
                                signal_route="canonical_raw",
                                quality_score=1.0,
                                retained=True,
                                level="file",
                                prediction_kind="single_model",
                                class_order=(0, 1, 2),
                                code_commit="test-commit",
                                data_schema_id="test-data-v1",
                                feature_schema_id="raw-eight-v1",
                                model_version="inception-test-v1",
                                aggregation_rule=LINE_B_EQUAL_ROLE_FAMILIES,
                                environment_hash="environment-test",
                                manifest_version="manifest-test-v1",
                                fold_registry_version="fold-test-v1",
                                artifact_reducer_name="identity",
                                artifact_reducer_version="identity-v1",
                                route_status="retained",
                                source_snapshot_hash="snapshot-shared",
                            )
                        )
        return tuple(output)

    def _write_source(
        self,
        study: Path,
        *,
        case_id: str,
        source: str,
        concrete_roles: list[str],
        role_families: list[str],
    ) -> tuple[Path, Path]:
        case = study / "raw" / case_id
        experiment = case / "attempts/attempt_001/experiment"
        experiment.mkdir(parents=True)
        config = {
            "schema_version": "ppg_frailty.pipeline_config.v2",
            "config_id": f"config-{source}",
            "roles": concrete_roles,
            "model": {"model_id": "InceptionTimeFull", "dropout": 0.5},
            "windows": {"raw_dl": {"length_s": 5.0, "hop_s": 2.5}},
            "training": {
                "classifier_role_families": role_families,
                "optimizer": "adamw",
                "class_count_basis": "row",
            },
        }
        (case / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        file_rows = self._file_rows(source, tuple(role_families))
        participant_rows = aggregate_hierarchy(
            file_rows,
            balance_line=LINE_B_EQUAL_ROLE_FAMILIES,
            quality_weighted=False,
            quality_weight_source="none",
        ).participant_rows
        file_path = write_oof_parquet(file_rows, experiment / "oof_file_predictions.parquet")
        subject_path = write_oof_parquet(
            participant_rows, experiment / "oof_subject_predictions.parquet"
        )
        return file_path, subject_path

    def _fixture(self, root: Path) -> tuple[Path, tuple[Path, ...]]:
        static_study = root / "static-study"
        all_study = root / "all-study"
        static_paths = self._write_source(
            static_study,
            case_id="rank5",
            source="static",
            concrete_roles=["B", "R1"],
            role_families=["B", "R"],
        )
        all_paths = self._write_source(
            all_study,
            case_id="rank3",
            source="all",
            concrete_roles=["B", "R1", "S1", "W1"],
            role_families=["B", "R", "S", "W"],
        )
        plan = {
            "schema_version": "ppg_frailty.role_scope_decomposition.v1",
            "study": {"study_id": "role-scope-test", "purpose": "test"},
            "sources": {
                "static_training": {
                    "name": "rank5",
                    "study_dir": str(static_study),
                    "case_id": "rank5",
                    "training_role_families": ["B", "R"],
                    "expected_files_per_participant": 2,
                },
                "all_role_training": {
                    "name": "rank3",
                    "study_dir": str(all_study),
                    "case_id": "rank3",
                    "training_role_families": ["B", "R", "S", "W"],
                    "expected_files_per_participant": 4,
                },
            },
            "aggregation_scopes": {
                "static": ["B", "R"],
                "all": ["B", "R", "S", "W"],
                "balance_line": LINE_B_EQUAL_ROLE_FAMILIES,
            },
            "inference": {
                "expected_participants": 6,
                "expected_repeats": [0, 1],
                "expected_class_order": [0, 1, 2],
                "metrics": ["balanced_accuracy", "macro_f1", "macro_roc_auc_ovr"],
                "bootstrap_resamples": 100,
                "permutation_resamples": 128,
                "seed": 42,
                "alpha": 0.05,
                "multiplicity_family": "role-scope-test",
                "require_complete_retention": True,
                "probability_tolerance": 1e-12,
            },
            "output": {
                "slug": "role-scope-test",
                "write_static_figures": False,
                "write_excel_workbook": False,
                "write_result_backup": False,
            },
        }
        plan_path = root / "plan.yaml"
        plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
        return plan_path, (*static_paths, *all_paths)

    def test_prediction_locked_three_cell_decomposition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, source_paths = self._fixture(root)
            before = {path: _sha256(path) for path in source_paths}
            plan = load_role_scope_plan(plan_path, pipeline_root=root)
            self.assertEqual(plan.static_aggregation_roles, ("B", "R"))

            output = run_role_scope_decomposition(
                plan_path,
                pipeline_root=root,
                output_root=root / "outputs",
            )
            self.assertEqual(before, {path: _sha256(path) for path in source_paths})
            summary = json.loads((output / "study_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["no_retraining"])
            self.assertFalse(summary["fourth_cell_available"])
            self.assertEqual(summary["dynamic_role_family_weight"], 0.5)
            self.assertLessEqual(summary["static_replay_max_abs_probability_error"], 1e-12)
            self.assertLessEqual(summary["all_role_replay_max_abs_probability_error"], 1e-12)

            cells = json.loads((output / "tables/factorial_cells.json").read_text())
            availability = {(row["cell_id"], row["metric"]): row["availability"] for row in cells}
            for metric in plan.metrics:
                self.assertEqual(
                    availability[("B_static_train_all_aggregate", metric)],
                    "unavailable_static_model_S_W_predictions_absent",
                )
                for cell in (
                    "A_static_train_static_aggregate",
                    "C_all_train_static_aggregate",
                    "D_all_train_all_aggregate",
                ):
                    self.assertEqual(availability[(cell, metric)], "available")

            contrasts = json.loads((output / "tables/contrasts.json").read_text())
            by_key = {(row["metric"], row["contrast_id"]): row for row in contrasts}
            for metric in plan.metrics:
                training = by_key[(metric, "training_side_at_static_aggregation")]
                aggregation = by_key[(metric, "aggregation_side_with_all_role_training")]
                total = by_key[(metric, "total_all_role_minus_static")]
                self.assertAlmostEqual(training["delta_percentage_points"], 0.0)
                self.assertAlmostEqual(
                    training["delta_percentage_points"]
                    + aggregation["delta_percentage_points"],
                    total["delta_percentage_points"],
                )

            derived = read_oof_parquet(
                output
                / "derived_oof/all_role_trained_static_aggregation_subject_predictions.parquet"
            )
            self.assertEqual(len(derived), 12)
            for required in (
                "STUDY_SUMMARY.md",
                "STUDY_SUMMARY.html",
                "REPORT_METHODS.md",
                "TEST_COMPONENTS.md",
                "tables/source_parameters.csv",
                "tables/test_components.csv",
                "outputs_index.json",
            ):
                self.assertTrue((output / required).is_file(), required)

    def test_non_role_config_difference_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, _ = self._fixture(root)
            config_path = root / "all-study/raw/rank3/resolved_config.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["model"]["dropout"] = 0.7
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differ outside"):
                run_role_scope_decomposition(
                    plan_path,
                    pipeline_root=root,
                    output_root=root / "outputs",
                )


if __name__ == "__main__":
    unittest.main()

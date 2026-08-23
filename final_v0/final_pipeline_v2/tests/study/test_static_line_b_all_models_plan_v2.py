"""Contracts for the explicit ordinary-model static Line B screening plan."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import unittest

from ppg_frailty.models import normalize_model_id
from ppg_frailty.study import (
    StudyRunner,
    load_study_plan,
    validate_canonical_expansion,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs" / "studies" / "static_line_b_all_models_v2.yaml"


class StaticLineBAllModelsPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = load_study_plan(PLAN)
        cls.expansion = validate_canonical_expansion(
            StudyRunner(pipeline_root=ROOT).expand(cls.plan)
        )

    def test_exact_ordinary_catalog_and_group_layout(self) -> None:
        self.assertEqual(self.plan.study.kind, "catalog_sweep")
        self.assertEqual(self.plan.catalog.balance_line, "line_b")
        self.assertEqual(self.plan.search.runtime_sampling, False)
        self.assertEqual(len(self.expansion.cases), 31)
        counts = Counter(case.output_group for case in self.expansion.cases)
        self.assertEqual(
            counts,
            {
                "raw": 15,
                "fusion": 6,
                "feature_vector": 9,
                "feature_matrix": 1,
            },
        )
        by_entry: dict[str, list[object]] = defaultdict(list)
        for case in self.expansion.cases:
            by_entry[str(case.catalog_entry)].append(case)
            self.assertEqual(
                case.output_group,
                case.config["representation_mode"],
            )
        self.assertEqual(len(by_entry), 11)
        self.assertEqual(len(by_entry["inception_matrix_small"]), 1)
        self.assertTrue(
            all(
                len(values) == 3
                for entry, values in by_entry.items()
                if entry != "inception_matrix_small"
            )
        )
        machines = {
            normalize_model_id(str(case.config["model"]["model_id"]))[1]
            for case in self.expansion.cases
        }
        self.assertEqual(len(machines), 11)
        self.assertFalse(any(int(case.config["model"]["ensemble_size"]) > 1 for case in self.expansion.cases))

    def test_shared_static_line_b_controls_are_identical(self) -> None:
        for case in self.expansion.cases:
            config = case.config
            self.assertEqual(config["roles"], ["B", "R1", "R2", "R3", "R4"])
            self.assertEqual(config["signal"]["internal_fs_hz"], 400.0)
            self.assertEqual(
                config["signal"]["imu"]["gravity_method"],
                "profile_a_lowpass_0p3hz",
            )
            self.assertEqual(config["quality"]["mode"], "off")
            self.assertFalse(config["artifact"]["motion_detector_enabled"])
            self.assertEqual(config["artifact"]["reducer"], "identity")
            self.assertEqual(
                config["aggregation"]["balance_line"],
                "line_b_equal_role_families",
            )
            self.assertEqual(config["training"]["fixed_epochs"], 10)
            self.assertEqual(config["windows"]["raw_dl"]["length_s"], 5.0)
            self.assertEqual(config["windows"]["raw_dl"]["hop_s"], 2.5)
            self.assertEqual(config["windows"]["engineering"]["length_s"], 10.0)
            self.assertEqual(config["windows"]["engineering"]["hop_s"], 2.0)

    def test_registered_sampling_profiles_do_not_mix_optimizer_changes(self) -> None:
        selected = [
            case
            for case in self.expansion.cases
            if case.catalog_entry in {"compact_cnn", "inception_full"}
        ]
        self.assertEqual(len(selected), 6)
        targets: dict[str, set[float]] = defaultdict(set)
        for case in selected:
            config = case.config
            targets[str(case.catalog_entry)].add(
                float(config["signal"]["dl_resampling"]["target_fs_hz"])
            )
            self.assertEqual(config["training"]["learning_rate"], 0.001)
            self.assertEqual(config["training"]["weight_decay"], 0.0001)
            identity = config["output"]["formal_ablation_materialization"]
            self.assertEqual(identity["family"], "fixed_kernel_samples")
            self.assertTrue(config["signal"]["dl_resampling"]["case_id"])
        self.assertEqual(
            targets,
            {
                "compact_cnn": {100.0, 200.0, 400.0},
                "inception_full": {100.0, 200.0, 400.0},
            },
        )
        varied = {
            str(row["parameter_path"]): row["case_values"]
            for row in self.expansion.varied_parameters
        }
        self.assertIn("signal.dl_resampling.target_fs_hz", varied)
        self.assertEqual(
            {
                float(varied["signal.dl_resampling.target_fs_hz"][case.case_id])
                for case in selected
            },
            {100.0, 200.0, 400.0},
        )

    def test_sparse_model_specific_parameters_reach_declared_architecture(self) -> None:
        by_id = {case.case_id: case.config for case in self.expansion.cases}
        logistic = by_id["logistic_regression__c10"]["model"]
        self.assertEqual((logistic["logistic_c"], logistic["architecture_parameters"]["C"]), (10.0, 10.0))
        svm = by_id["rbf_svm__c0p1_gamma0p001"]["model"]
        self.assertEqual((svm["svm_c"], svm["svm_gamma"]), (0.1, 0.001))
        trees = by_id["extra_trees__sqrt_leaf5"]["model"]
        self.assertEqual(
            (
                trees["extra_trees_max_features"],
                trees["extra_trees_min_samples_leaf"],
            ),
            ("sqrt", 5),
        )
        self.assertFalse(any("rocket" in case_id for case_id in by_id))
        shapeformer = by_id["shapeformer_osd__lr1e4_wd1e3"]
        self.assertEqual(shapeformer["model"]["input_fs_hz"], 400.0)
        self.assertEqual(shapeformer["training"]["learning_rate"], 0.0001)


if __name__ == "__main__":
    unittest.main()

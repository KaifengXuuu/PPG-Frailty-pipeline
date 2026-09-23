"""Paper-specific plans preserve executed case settings, not mutable templates."""

import json
from pathlib import Path

import pytest
import yaml

from ppg_frailty.study import StudyRunner, load_study_plan
from ppg_frailty.study.hyperparameter import load_hyperparameter_plan


ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "configs/studies/thesis"
HISTORY = ROOT.parent / "final_pipeline_v2/artifacts/studies/static_line_b_staged_v2"
SOURCES = {
    "representation_screening": (
        "20260823_075821_catalog_sweep_staged-static-01-representation-baselines-v2",
    ),
    "sqi_motion_routing": (
        "20260824_031024_catalog_sweep_staged-static-05-sqi-motion-compact-cnn-v6",
    ),
    "final_five_configurations": (
        "20260824_160517_catalog_sweep_final-case-all-roles-inception-architecture-comparison-v1",
        "20260824_175009_catalog_sweep_stage0-inception-small-no-gravity-supplement-v1",
        "20260824_111943_catalog_sweep_final-case-comparison-inception-full-v1",
    ),
}


@pytest.mark.parametrize("name,count", [
    ("representation_screening", 3),
    ("sqi_motion_routing", 4),
    ("final_five_configurations", 5),
])
def test_paper_cases_and_complete_grouped_resource(name, count):
    plan = load_study_plan(PLANS / f"{name}.yaml")
    expansion = StudyRunner(pipeline_root=ROOT, output_layout="v5").expand(plan)
    assert len(expansion.cases) == count
    assert plan.execution.repeats == plan.execution.folds == tuple(range(5))
    if name == "representation_screening":
        assert all(case.config["signal"]["peak_detector"]["detector_id"] == "aboy_project_v1"
                   for case in expansion.cases)
    elif name == "sqi_motion_routing":
        assert all(not case.config["artifact"]["denoiser_enabled"] for case in expansion.cases)
        assert {(case.config["quality"]["mode"], case.config["artifact"]["motion_detector_enabled"])
                for case in expansion.cases} == {("off", False), ("off", True), ("route", False), ("route", True)}
    else:
        rank4 = expansion.cases[3].config
        assert expansion.cases[3].case_id == "s1_163_v2_port_all_roles_modules_off"
        assert rank4["training"]["fixed_epochs"] == 15
        assert rank4["training"]["batch_size"] == 32
        assert rank4["training"]["class_count_basis"] == "participant"
        assert rank4["windows"]["raw_dl"]["cap_fraction_per_file"] == 0.9


@pytest.mark.parametrize("name", SOURCES)
def test_expanded_cases_equal_frozen_v2_configs_when_archive_available(name):
    """Local historical evidence is optional for a standalone V5 checkout."""
    sources = [HISTORY / directory for directory in SOURCES[name]]
    if not all((source / "study_manifest.json").is_file() for source in sources):
        pytest.skip("V2 historical run archive is not installed")
    expected = {}
    for source in sources:
        manifest = json.loads((source / "study_manifest.json").read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            expected[case["case_id"]] = yaml.safe_load(
                (source / case["resolved_config_path"]).read_text(encoding="utf-8")
            )
    expansion = StudyRunner(pipeline_root=ROOT, output_layout="v5").expand(
        load_study_plan(PLANS / f"{name}.yaml")
    )
    for case in expansion.cases:
        assert case.config == expected[case.case_id], case.case_id


@pytest.mark.parametrize("name,count", [("batch_learning_rate_search", 6), ("regularization_search", 9)])
def test_hyperparameter_plans_keep_resource_and_historical_detector(name, count):
    plan = load_hyperparameter_plan(PLANS / f"{name}.yaml")
    assert len(plan["candidates"]) == count
    assert plan["base"]["common_overrides"]["signal.peak_detector.detector_id"] == "aboy_project_v1"
    assert plan["base"]["common_overrides"]["signal.peak_detector.parameters"] == {}
    resource = plan["resource"]
    if count == 6:
        assert resource["screen_epochs"] == 5 and resource["screen_folds"] == [0]
        assert resource["promotion_epochs"] == 10 and resource["promote_count"] == 3
        assert resource["promotion_repeats"] == resource["promotion_folds"] == list(range(5))
    else:
        assert resource["epochs"] == 10
        assert resource["repeats"] == resource["folds"] == list(range(5))

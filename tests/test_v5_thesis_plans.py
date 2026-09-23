"""Paper-specific plans preserve executed case settings, not mutable templates."""

from pathlib import Path

import pytest
import yaml

from ppg_frailty.study import StudyRunner, load_study_plan
from ppg_frailty.study.hyperparameter import load_hyperparameter_plan


ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "configs/studies/thesis"
EXPECTED = yaml.safe_load((ROOT / "tests/fixtures/thesis_expected.yaml").read_text(encoding="utf-8"))


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


@pytest.mark.parametrize("name", EXPECTED["thesis_cases"])
def test_expanded_cases_equal_frozen_configuration(name):
    """Test every scientific setting without installing old experiment outputs."""
    expected = EXPECTED["thesis_cases"][name]
    expansion = StudyRunner(pipeline_root=ROOT, output_layout="v5").expand(
        load_study_plan(PLANS / f"{name}.yaml")
    )
    assert {case.case_id for case in expansion.cases} == set(expected)
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

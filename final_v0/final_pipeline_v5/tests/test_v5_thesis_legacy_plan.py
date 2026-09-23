"""Keep the historical raw bridge executable without changing its algorithms."""
from pathlib import Path

import pytest
import yaml

from ppg_frailty.legacy_bridge import resolve_legacy_bridge_profile
from ppg_frailty.study.expand import expand_study, load_study_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/studies/thesis/legacy_bridge_ablation.yaml"
HISTORY = ROOT.parent / (
    "final_pipeline_v2/artifacts/studies/static_line_b_staged_v2/"
    "20260821_162454_catalog_sweep_staged-static-03-centered-star-v1"
)
CASE_IDS = tuple(
    f"{model}__b{profile}_star_fixed10"
    for profile in range(8)
    for model in ("compact_cnn", "inception_full")
)


@pytest.fixture(scope="module")
def expansion():
    return expand_study(load_study_plan(PLAN), pipeline_root=ROOT)


def test_legacy_plan_retains_budget_and_data_only_execution(expansion):
    plan = expansion.plan
    assert tuple(case.case_id for case in expansion.cases) == CASE_IDS
    assert plan.execution.repeats == plan.execution.folds == (0, 1, 2, 3, 4)
    assert len(expansion.cases) * len(plan.execution.repeats) * len(plan.execution.folds) == 400
    assert plan.execution.preprocessing_cache.mode == "off"
    assert plan.output.root == "pipeline_output"
    assert not plan.report.write_html
    assert not plan.report.write_static_figures
    assert not plan.report.write_excel_workbook


def test_b3_uses_historical_ekf_and_bridge_owns_actual_training(expansion):
    bridge = expansion.plan.legacy_bridge
    assert bridge is not None
    configs = {case.case_id: case.config for case in expansion.cases}
    for definition in bridge.profiles:
        profile = resolve_legacy_bridge_profile(
            definition["profile_id"],
            protocol_design=bridge.design,
            profile_definition=definition,
            profile_definition_sha256=bridge.controls_sha256(definition),
        )
        training = profile.training_config()
        assert training.fixed_epochs == 10
        assert training.seed == 42
        assert training.class_count_basis == "row"
        assert not training.outer_labels_visible_to_trainer
        assert training.sampler == (
            "balance_line_weighted_v2" if profile.profile_id == "B5"
            else "exhaustive_shuffle_without_replacement"
        )
        assert (training.optimizer, training.batch_size) == (
            ("adam", 64) if profile.profile_id == "B6" else ("adamw", 32)
        )
        config = configs[definition["catalog_case_id"]]
        assert config["representation_mode"] == "raw"
        assert config["roles"] == ["B", "R1", "R2", "R3", "R4"]
        assert config["quality"]["mode"] == "off"
        assert not config["artifact"]["motion_detector_enabled"]
        assert not config["artifact"]["denoiser_enabled"]
        if profile.profile_id == "B3":
            assert profile.requires_calibrated_imu_views
            imu = config["signal"]["imu"]
            assert imu["gravity_method"] == "calibrated_roll_pitch_ekf"
            assert imu["comparison_method"] == "profile_a_lowpass_0p3hz"
            assert imu["process_covariance_diagonal_per_second"] == [5.0, 5.0, 0.05, 0.05, 0.05]
            assert imu["calibration_start_s"] == 5.0
            assert imu["calibration_stop_s"] == 100.0
        else:
            assert not profile.requires_calibrated_imu_views


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_execution_fields_match_actual_historical_configuration(expansion, case_id):
    source = HISTORY / "raw" / case_id / "resolved_config.yaml"
    if not source.is_file():
        pytest.skip("local V2 historical results are not included in this checkout")
    expected = yaml.safe_load(source.read_text(encoding="utf-8"))
    actual = next(case.config for case in expansion.cases if case.case_id == case_id)
    # Features are not computed by this raw bridge. Its old fixed-K metadata is
    # intentionally not restored; neither are disabled motion-bundle metadata.
    for field in ("roles", "signal", "windows", "quality", "training", "aggregation", "model", "manifest", "splits"):
        assert actual[field] == expected[field], (case_id, field)
    for field in ("reducer", "reducer_version", "motion_detector_enabled", "denoiser_enabled", "parameters"):
        assert actual["artifact"][field] == expected["artifact"][field], (case_id, field)


def test_bridge_effective_controls_match_historical_run(expansion):
    source = HISTORY / "study_plan.yaml"
    if not source.is_file():
        pytest.skip("local V2 historical study plan is unavailable")
    expected = load_study_plan(source).legacy_bridge
    actual = expansion.plan.legacy_bridge
    assert actual is not None and expected is not None
    assert actual.baseline_controls == expected.baseline_controls
    assert actual.factor_overrides == expected.factor_overrides
    assert actual.profiles == expected.profiles
    assert actual.budget == expected.budget

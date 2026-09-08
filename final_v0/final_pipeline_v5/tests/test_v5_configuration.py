from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
import shlex

import pytest
import ppg_frailty.v5.cli as v5_cli

from ppg_frailty.v5.configuration import (
    PRESETS,
    manual_cli_tokens,
    parameter_rows,
    parse_assignment,
    parse_module_assignment,
    preset_rows,
    resolve_configuration,
)
from ppg_frailty.study import ExecutionSpec, PreprocessingCacheSpec
from ppg_frailty.v5.cli import _execution, build_parser
from ppg_frailty.v5.sweep import build_parser as build_sweep_parser
from ppg_frailty.v5.output_contract import automatic_run_name


ROOT = Path(__file__).resolve().parents[1]


def _leaf_items(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict) and value:
        return {
            path: leaf
            for key, child in value.items()
            for path, leaf in _leaf_items(
                child, f"{prefix}.{key}" if prefix else str(key)
            ).items()
        }
    return {prefix: value}


def _leaf_paths(value: object, prefix: str = "") -> set[str]:
    return set(_leaf_items(value, prefix))


def test_all_parameter_catalog_covers_every_named_preset_leaf() -> None:
    rows = parameter_rows(ROOT, source_preset="all")
    by_path = {row["path"]: row for row in rows}
    assert len(rows) == len(by_path)
    for preset in PRESETS:
        config, _ = resolve_configuration(pipeline_root=ROOT, preset=preset)
        for path, value in _leaf_items(config).items():
            assert path in by_path
            assert preset in by_path[path]["applicable_presets"]
            assert by_path[path]["defaults_by_preset"][preset] == value


def test_all_parameter_catalog_covers_every_canonical_study_case_leaf() -> None:
    import yaml

    from ppg_frailty.study import load_study_plan
    from ppg_frailty.study.expand import expand_study

    rows = parameter_rows(ROOT, source_preset="all")
    paths = {row["path"] for row in rows}
    canonical_plan_count = 0
    for source in sorted((ROOT / "configs/studies").rglob("*.yaml")):
        declared = yaml.safe_load(source.read_text(encoding="utf-8"))
        if declared.get("schema_version") != "ppg_frailty.study_plan.v2":
            continue
        canonical_plan_count += 1
        expansion = expand_study(load_study_plan(source), pipeline_root=ROOT)
        for case in expansion.cases:
            assert _leaf_paths(case.config) <= paths
    assert canonical_plan_count == 19


def test_all_parameter_catalog_covers_independently_resolvable_modules() -> None:
    from ppg_frailty.module_registry import list_modules

    rows = parameter_rows(ROOT, source_preset="all")
    paths = {row["path"] for row in rows}
    tagged_modules = {
        selection for row in rows for selection in row["applicable_modules"]
    }
    resolved_modules: set[str] = set()
    for descriptor in list_modules():
        selection = f"{descriptor['family']}={descriptor['module_id']}"
        for preset in PRESETS:
            try:
                config, _ = resolve_configuration(
                    pipeline_root=ROOT,
                    preset=preset,
                    modules=(selection,),
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
            resolved_modules.add(selection)
            assert _leaf_paths(config) <= paths
        if selection in resolved_modules:
            assert selection in tagged_modules
    assert resolved_modules


def test_parameters_parser_exposes_complete_union_source() -> None:
    args = build_parser().parse_args(["parameters", "--source-preset", "all"])
    assert args.source_preset == "all"


@pytest.mark.parametrize(
    ("command", "headings", "example"),
    [
        ("modules", ("FAMILY", "MODULE_ID", "STATUS"), "model\tInceptionTimeSmall"),
        ("parameters", ("PATH", "TYPE", "RANGE", "CLI INPUT"), "training.batch_size"),
    ],
)
def test_catalog_help_prints_the_live_control_surface(
    command: str,
    headings: tuple[str, ...],
    example: str,
    capsys,
) -> None:
    with pytest.raises(SystemExit) as stopped:
        build_parser().parse_args([command, "--help"])
    output = capsys.readouterr().out
    assert stopped.value.code == 0
    assert all(heading in output for heading in headings)
    assert example in output


def test_automatic_name_uses_yaml_stem_and_utc() -> None:
    assert re.fullmatch(
        r"finalcase_\d{8}_\d{6}Z",
        automatic_run_name("configs/presets/finalcase.yaml"),
    )


def test_baseline_is_default_and_byte_identical_to_v2() -> None:
    rows = {row["name"]: row for row in preset_rows(ROOT)}
    assert rows["baseline"]["default"] is True
    source = ROOT.parent / "final_pipeline_v2/configs/reference_static_role_aware_v2.yaml"
    assert rows["baseline"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_named_preset_uses_schema_validation_not_a_duplicate_hash_gate(tmp_path: Path) -> None:
    preset_dir = tmp_path / "configs/presets"
    preset_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "configs/presets/registry.yaml", preset_dir / "registry.yaml")
    shutil.copy2(ROOT / "configs/presets/baseline.yaml", preset_dir / "baseline.yaml")
    with (preset_dir / "baseline.yaml").open("a", encoding="utf-8") as stream:
        stream.write("\n# unexpected local mutation\n")
    config, provenance = resolve_configuration(pipeline_root=tmp_path, preset="baseline")
    assert config["config_id"] == "reference_static_raw_compactcnn_role_aware_v2"
    assert provenance["resolved_config_sha256"]


def test_finalcase_is_exact_user_selected_rank2() -> None:
    config, provenance = resolve_configuration(
        pipeline_root=ROOT, preset="finalcase"
    )
    assert provenance["resolved_config_sha256"] == (
        "c3a4cc8b9c927f208d0f8476c9b3081dd9bfdbe058ae1b153c3d9024f77e3a79"
    )
    assert config["model"]["model_id"] == "InceptionTimeSmall"
    assert config["signal"]["imu"]["gravity_method"] == (
        "sensor_filter_only_no_gravity_removal"
    )
    assert config["signal"]["dl_resampling"]["target_fs_hz"] == 64.0
    assert config["aggregation"]["balance_line"] == "line_b_equal_role_families"
    assert config["roles"] == ["B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2"]


def test_preset_free_manual_cli_is_canonically_identical_to_finalcase() -> None:
    tokens = manual_cli_tokens(ROOT, source_preset="finalcase")
    assert "--preset" not in tokens
    assert "--config" not in tokens
    assert "--plan" not in tokens
    assignments = tuple(
        tokens[index + 1]
        for index, token in enumerate(tokens[:-1])
        if token == "--set"
    )
    unsets = tuple(
        tokens[index + 1]
        for index, token in enumerate(tokens[:-1])
        if token == "--unset"
    )
    manual, manual_provenance = resolve_configuration(
        pipeline_root=ROOT,
        manual=True,
        config_id=tokens[tokens.index("--config-id") + 1],
        assignments=assignments,
        unsets=unsets,
    )
    preset, preset_provenance = resolve_configuration(
        pipeline_root=ROOT, preset="finalcase"
    )
    assert manual == preset
    assert manual_provenance["resolved_config_sha256"] == (
        preset_provenance["resolved_config_sha256"]
    )


def test_manual_cli_generator_can_include_one_safe_fixed_run_name(capsys) -> None:
    args = build_parser().parse_args(
        ["manual-cli", "--source-preset", "finalcase", "--run-name", "finalcase_cli"]
    )
    assert v5_cli._dispatch(args) == 0
    command = capsys.readouterr().out.strip()
    assert " --run-name finalcase_cli" in command
    assert "--no-refit" not in command
    tokens = shlex.split(command)
    parsed = build_parser().parse_args(tokens[tokens.index("run") :])
    assert parsed.manual is True
    assert parsed.refit is False
    assert parsed.case_id == "tuned_all_roles__inception_small_no_gravity"
    assert parsed.continue_on_error is False
    assert parsed.preprocessing_cache_root == "cache/preprocessing"
    assert parsed.preprocessing_cache_namespaces == (
        "imu_calibration",
        "canonical_signal_views",
        "raw_windows",
    )

    unsafe = build_parser().parse_args(
        ["manual-cli", "--run-name", "../outside"]
    )
    with pytest.raises(ValueError, match="run name"):
        v5_cli._dispatch(unsafe)


@pytest.mark.parametrize(
    ("parser_factory", "command"),
    [
        (build_parser, ["run", "--preset", "finalcase"]),
        (build_sweep_parser, ["run", "--plan", "configs/studies/finalcase.yaml"]),
    ],
)
def test_refit_is_one_default_off_cli_flag(parser_factory, command: list[str]) -> None:
    assert parser_factory().parse_args(command).refit is False
    assert parser_factory().parse_args([*command, "--refit"]).refit is True
    with pytest.raises(SystemExit):
        parser_factory().parse_args([*command, "--no-refit"])


def test_config_run_resume_does_not_synthesize_a_new_run_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume = tmp_path / "pipeline_output" / "existing"
    resume.mkdir(parents=True)
    resolved = tmp_path / "resolved.yaml"
    resolved.write_text("config_id: test\n", encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        v5_cli,
        "_resolved",
        lambda _args: (
            {"training": {"device": "cuda"}},
            {"resolved_config_sha256": "a" * 64},
        ),
    )
    monkeypatch.setattr(v5_cli, "_resolved_file", lambda *_args: resolved)
    monkeypatch.setattr(v5_cli, "_plan", lambda *_args: object())
    monkeypatch.setattr(
        v5_cli,
        "automatic_run_name",
        lambda _source: (_ for _ in ()).throw(AssertionError("resume must not be renamed")),
    )

    def fake_run(_plan, **kwargs):
        captured.update(kwargs)
        return {"exit_code": 0}

    monkeypatch.setattr(v5_cli, "run_study", fake_run)
    args = build_parser().parse_args(
        ["run", "--preset", "finalcase", "--resume", str(resume)]
    )

    assert v5_cli._run_config(args) == 0
    assert captured["resume"] == resume.resolve()
    assert captured["run_name"] is None

    conflict = build_parser().parse_args(
        [
            "run",
            "--preset",
            "finalcase",
            "--resume",
            str(resume),
            "--run-name",
            "new_run",
        ]
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        v5_cli._run_config(conflict)


def test_cli_override_is_strict_and_hash_bound() -> None:
    config, provenance = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        assignments=("training.batch_size=32",),
    )
    assert config["training"]["batch_size"] == 32
    assert config["config_id"].startswith("v5_baseline_")
    assert provenance["assignments"] == [
        {"path": "training.batch_size", "value": 32}
    ]


def test_registered_module_selection_and_unknown_module() -> None:
    config, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        modules=("imu_gravity=sensor_filter_only_no_gravity_removal",),
    )
    assert config["signal"]["imu"]["gravity_method"] == (
        "sensor_filter_only_no_gravity_removal"
    )
    with pytest.raises(ValueError, match="unregistered module"):
        parse_module_assignment("imu_gravity=made_up")


def test_assignment_parser_preserves_yaml_types() -> None:
    assert parse_assignment("roles=[B, R1]") == ("roles", ["B", "R1"])
    assert parse_assignment("quality.mode=off") == ("quality.mode", "off")


def test_run_plan_execution_values_are_inherited_until_explicitly_overridden() -> None:
    source = ExecutionSpec(
        repeats=(1, 3),
        folds=(2,),
        jobs=4,
        continue_on_error=False,
        measure_operational_costs=True,
        preprocessing_cache=PreprocessingCacheSpec(
            mode="read_only",
            root="artifacts/custom_cache",
            namespaces=("raw_windows",),
        ),
    )
    inherited_args = build_parser().parse_args(
        ["run-plan", "--plan", "configs/studies/example.yaml"]
    )
    assert _execution(inherited_args, source) == source

    overridden_args = build_parser().parse_args(
        [
            "run-plan",
            "--plan",
            "configs/studies/example.yaml",
            "--repeats",
            "0",
            "--jobs",
            "2",
            "--preprocessing-cache-mode",
            "off",
        ]
    )
    overridden = _execution(overridden_args, source)
    assert overridden.repeats == (0,)
    assert overridden.folds == source.folds
    assert overridden.jobs == 2
    assert overridden.preprocessing_cache.mode == "off"
    assert overridden.preprocessing_cache.root == source.preprocessing_cache.root


def test_direct_cli_device_is_part_of_the_resolved_numerical_config() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--preset",
            "finalcase",
            "--device",
            "cpu",
            "--environment-policy",
            "record",
        ]
    )
    config, provenance = v5_cli._resolved_config(args)
    assert config["training"]["device"] == "cpu"
    assert {row["path"] for row in provenance["assignments"]} == {
        "training.device"
    }


def test_matching_cli_device_preserves_registered_finalcase_identity() -> None:
    inherited_args = build_parser().parse_args(["run", "--preset", "finalcase"])
    explicit_args = build_parser().parse_args(
        ["run", "--preset", "finalcase", "--device", "cuda"]
    )
    inherited, inherited_provenance = v5_cli._resolved_config(inherited_args)
    explicit, explicit_provenance = v5_cli._resolved_config(explicit_args)
    assert explicit == inherited
    assert explicit["config_id"] == inherited["config_id"]
    assert explicit_provenance["resolved_config_sha256"] == (
        inherited_provenance["resolved_config_sha256"]
    )
    assert explicit_provenance["assignments"] == []


def test_direct_cli_without_device_inherits_config_device(tmp_path: Path) -> None:
    args = build_parser().parse_args(["run", "--preset", "finalcase"])
    config, _ = v5_cli._resolved_config(args)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config_id: placeholder\n", encoding="utf-8")
    plan = v5_cli._plan_from_config_command(args, config_path, config)
    assert plan.execution.device == "cuda"


def test_public_output_root_is_an_explicit_execution_option() -> None:
    args = build_parser().parse_args(
        ["run", "--preset", "finalcase", "--output-root", "pipeline_output/nested"]
    )
    assert Path(args.output_root).parts[-2:] == ("pipeline_output", "nested")


def test_paths_resolve_relative_to_the_v5_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v5_cli, "PIPELINE_ROOT", tmp_path)
    assert v5_cli._path("pipeline_output/evidence") == (
        tmp_path / "pipeline_output/evidence"
    ).resolve()


def test_existing_path_resolution_is_shared_by_maintenance_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v5_cli, "PIPELINE_ROOT", tmp_path)
    run = tmp_path / "pipeline_output/run_a"
    run.mkdir(parents=True)
    (run / "study_manifest.json").write_text("{}\n", encoding="utf-8")
    assert v5_cli._path("pipeline_output/run_a", must_exist=True) == run.resolve()
    with pytest.raises(FileNotFoundError):
        v5_cli._path("pipeline_output/missing", must_exist=True)


def test_pretrained_infer_cli_has_one_explicit_manifest_contract() -> None:
    args = build_parser().parse_args(
        [
            "infer",
            "--model-config",
            "model_config/finalcase_run",
            "--case-id",
            "reference",
            "--input-manifest",
            "inputs/participant.yaml",
        ]
    )
    assert args.command == "infer"
    assert args.model_config == "model_config/finalcase_run"
    assert args.case_id == "reference"
    assert args.input_manifest == "inputs/participant.yaml"


def test_resolved_config_file_is_content_addressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v5_cli, "PIPELINE_ROOT", tmp_path)
    provenance = {"resolved_config_sha256": "a" * 64}
    path = v5_cli._resolved_file({"config_id": "reviewable"}, provenance)
    assert path.is_file()
    assert path == v5_cli._resolved_file({"config_id": "reviewable"}, provenance)

    path.write_text("nested:\n  second: 2\n  first: 1\nconfig_id: reviewable\n", encoding="utf-8")
    reordered = {"config_id": "reviewable", "nested": {"first": 1, "second": 2}}
    assert path == v5_cli._resolved_file(reordered, provenance)


def test_ppg_filter_and_local_prv_modules_map_to_runtime_config_ids() -> None:
    filtered, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        modules=("ppg_filter=butterworth_sos",),
    )
    assert filtered["signal"]["ppg_filter"]["family"] == "butterworth_sos"
    assert filtered["signal"]["ppg_filter"]["low_hz"] == 0.2

    local_prv, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="feature_vector",
        modules=("prv_backend=local",),
    )
    assert local_prv["features"]["prv_primary_backend"] == "local_manual"


@pytest.mark.parametrize("module_id", ["aura_hrv_analysis", "rhenan_hrv"])
def test_comparison_only_prv_backends_fail_closed(module_id: str) -> None:
    with pytest.raises(ValueError, match="comparison-only"):
        resolve_configuration(
            pipeline_root=ROOT,
            preset="feature_vector",
            modules=(f"prv_backend={module_id}",),
        )


def test_feature_group_selections_are_one_complete_canonical_replacement() -> None:
    config, provenance = resolve_configuration(
        pipeline_root=ROOT,
        preset="feature_vector",
        modules=(
            "feature_group=morphology",
            "feature_group=ppi_basic_rate",
            "feature_group=morphology",
        ),
    )
    assert config["features"]["enabled_groups"] == [
        "ppi_basic_rate",
        "morphology",
    ]
    assert config["features"]["registry_id"] != "feature_vector_282_v3"
    assert provenance["module_selections"] == [
        {"family": "feature_group", "module_id": "morphology"},
        {"family": "feature_group", "module_id": "ppi_basic_rate"},
        {"family": "feature_group", "module_id": "morphology"},
    ]

    with pytest.raises(ValueError, match="feature_group selections require"):
        resolve_configuration(
            pipeline_root=ROOT,
            preset="baseline",
            modules=("feature_group=morphology",),
        )


def test_shapeformer_balance_recomputes_derived_architecture_provenance() -> None:
    config, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        modules=(
            "model=ShapeFormerChannelSpecificOSD",
            "shapeformer_discovery_balance=class_window_balanced",
        ),
    )
    assert config["model"]["discovery_balance"] == "class_window_balanced"
    assert config["model"]["architecture_parameters"]["discovery_balance"] == (
        "class_window_balanced"
    )

    with pytest.raises(ValueError, match="channel-specific ShapeFormer"):
        resolve_configuration(
            pipeline_root=ROOT,
            preset="baseline",
            modules=(
                "shapeformer_discovery_balance=participant_file_balanced",
            ),
        )


@pytest.mark.parametrize(
    ("preset", "expected_policy"),
    [
        ("baseline", "denoise_then_compare_rate_exclude"),
        ("feature_vector", "denoise_then_extract_rate_features"),
        ("feature_matrix", "denoise_then_compare_rate_exclude"),
        ("fusion", "denoise_then_compare_rate_exclude"),
    ],
)
def test_artifact_policy_follows_representation(
    preset: str, expected_policy: str
) -> None:
    config, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset=preset,
        modules=("artifact=pca_bss",),
    )
    assert config["artifact"]["denoiser_enabled"] is True
    assert config["artifact"]["degraded_policy"] == expected_policy


def test_participant_class_counts_select_inverse_frequency_weighting() -> None:
    config, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        modules=("class_count_basis=participant",),
    )
    assert config["training"]["class_count_basis"] == "participant"
    assert config["training"]["class_weighting"] == "inverse_frequency"


@pytest.mark.parametrize(
    ("module_id", "training_balance", "hierarchy", "upper_operator"),
    [
        (
            "line_a_equal_files",
            "equal_files",
            ["window", "file", "participant"],
            "not_applicable",
        ),
        (
            "line_b_equal_role_families",
            "equal_role_families",
            ["window", "file", "role", "participant"],
            "ordinary_mean",
        ),
    ],
)
def test_aggregation_module_materializes_matched_training_and_reporting_line(
    module_id: str,
    training_balance: str,
    hierarchy: list[str],
    upper_operator: str,
) -> None:
    config, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        modules=(f"aggregation={module_id}",),
    )
    assert config["training"]["training_balance"] == training_balance
    assert config["aggregation"]["balance_line"] == module_id
    assert config["aggregation"]["hierarchy"] == hierarchy
    assert config["aggregation"]["file_to_role"] == upper_operator
    assert config["aggregation"]["quality_weight_source"] == "none"
    assert config["aggregation"]["quality_weight_levels"] == []


def test_quality_weight_source_materializes_its_runtime_levels() -> None:
    route, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        modules=(
            "quality_mode=route",
            "quality_weight_source=route_file_q_rate",
        ),
    )
    assert route["aggregation"]["quality_weighting"] is True
    assert route["aggregation"]["quality_weight_source"] == "route_file_q_rate"
    assert route["aggregation"]["quality_weight_levels"] == [
        "file_to_role",
        "role_to_participant",
    ]

    legacy, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        modules=(
            "window_quality_selection=legacy_per_file_top_fraction",
            "quality_weight_source=legacy_window_sqi",
        ),
    )
    assert legacy["aggregation"]["window_to_file"] == "quality_weighted_mean"
    assert legacy["aggregation"]["quality_weight_levels"] == [
        "window_to_file",
        "file_to_role",
        "role_to_participant",
    ]


def test_unique_catalog_model_updates_member_oof_and_same_model_is_noop() -> None:
    ensemble, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        modules=("model=InceptionTimeFullFiveMemberEnsemble",),
    )
    assert ensemble["output"]["write_member_oof"] is True
    assert ensemble["model"]["member_seeds"] == [
        50042,
        60042,
        70042,
        80042,
        90042,
    ]

    tuned, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="finalcase",
        modules=("model=InceptionTimeSmall",),
    )
    assert tuned["model"]["dropout"] == 0.5
    assert tuned["model"]["architecture_parameters"]["classifier_dropout"] == 0.5


def test_ambiguous_model_catalog_id_fails_closed() -> None:
    with pytest.raises(ValueError, match="2 complete catalog definitions"):
        resolve_configuration(
            pipeline_root=ROOT,
            preset="baseline",
            modules=("model=InceptionTimeFull",),
        )


@pytest.mark.parametrize(
    ("module_id", "fixed_epochs"),
    [("epoch_7_ablation", 7), ("epoch_15_ablation", 15)],
)
def test_safe_epoch_comparison_profiles_use_formal_catalog(
    module_id: str, fixed_epochs: int
) -> None:
    config, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        modules=(f"comparison_profile={module_id}",),
    )
    assert config["training"]["fixed_epochs"] == fixed_epochs
    assert config["training"]["epoch_profile"] == f"ablation_{fixed_epochs}"


def test_safe_direct_filter_and_line_b_comparison_profiles_are_complete() -> None:
    filtered, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        modules=("comparison_profile=direct_filter_0p5_to_5hz_ablation",),
    )
    assert filtered["signal"]["ppg_filter"]["low_hz"] == 0.5
    assert filtered["signal"]["ppg_filter"]["high_hz"] == 5.0
    assert filtered["signal"]["analysis_view"]["direct_source"] == (
        "x_filter_0p5_to_5hz"
    )

    line_b, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        modules=(
            "aggregation=line_a_equal_files",
            "comparison_profile=line_b_equal_role_families",
        ),
    )
    assert line_b["training"]["training_balance"] == "equal_role_families"
    assert line_b["aggregation"]["hierarchy"] == [
        "window",
        "file",
        "role",
        "participant",
    ]


def test_quality_diagnostics_comparison_requires_neutral_artifact_controls() -> None:
    config, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        modules=("comparison_profile=quality_diagnostics_only",),
    )
    assert config["quality"]["mode"] == "diagnostics_only"

    with pytest.raises(ValueError, match="requires motion and denoiser"):
        resolve_configuration(
            pipeline_root=ROOT,
            preset="baseline",
            modules=(
                "comparison_profile=quality_diagnostics_only",
                "denoiser_switch=enabled",
            ),
        )


@pytest.mark.parametrize(
    "module_id",
    [
        "imu_lpf_0p3hz_ablation",
        "fixed_kernel_samples_resampling_ablation",
        "fixed_kernel_samples_context_10s_400hz_ablation",
        "fixed_kernel_samples_dilation2_ablation",
    ],
)
def test_non_materializable_comparison_profiles_fail_closed(module_id: str) -> None:
    with pytest.raises(ValueError, match="cannot be selected safely"):
        resolve_configuration(
            pipeline_root=ROOT,
            preset="baseline",
            modules=(f"comparison_profile={module_id}",),
        )


def test_reused_motion_evidence_requires_bound_evidence_metadata() -> None:
    with pytest.raises(ValueError, match="requires evidence_path"):
        resolve_configuration(
            pipeline_root=ROOT,
            preset="baseline",
            modules=("motion_evidence=reused_frailty29_all29_bundle",),
        )

    config, _ = resolve_configuration(
        pipeline_root=ROOT,
        preset="baseline",
        modules=("motion_evidence=reused_frailty29_all29_bundle",),
        assignments=(
            "artifact.motion_detector.evidence_path=artifacts/motion.json",
            f"artifact.motion_detector.expected_evidence_sha256={'a' * 64}",
        ),
    )
    assert config["artifact"]["motion_detector_enabled"] is True
    assert config["artifact"]["motion_detector"]["evidence_path"] == (
        "artifacts/motion.json"
    )


@pytest.mark.parametrize(
    "module_id",
    ["sqi_only", "sqi_plus_motion_override", "historical_light_cnn_backup"],
)
def test_external_motion_evidence_options_fail_closed(module_id: str) -> None:
    with pytest.raises(ValueError, match="external or historical audit evidence"):
        resolve_configuration(
            pipeline_root=ROOT,
            preset="baseline",
            modules=(f"motion_evidence={module_id}",),
        )


def test_motion_finalist_bundle_is_self_contained_and_hash_bound() -> None:
    root = (
        ROOT
        / "artifacts/studies/static_line_b_staged_v2"
        / "20260820_225546_staged-static-05-pre-motion-ptt-v1"
        / "motion_internal"
    )
    evidence = root / "motion_internal_evidence.json"
    model = root / "final_all_internal/formal_motion_model.pt"
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == (
        "10f02a9d784e06471c7109ff8dc92d28f1a8d7753f8fdf179bebce5699fb446c"
    )
    assert hashlib.sha256(model.read_bytes()).hexdigest() == (
        "62a09c53fecf90dfb9388900df19efccc62facf9f72b221b09c7d06c999c6eca"
    )

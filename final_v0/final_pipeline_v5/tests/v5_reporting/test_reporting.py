from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import tempfile

import pytest
import yaml

from ppg_frailty.training import (
    OofPredictionRow,
    write_empty_oof_parquet,
    write_oof_parquet,
)
from ppg_frailty.v5.output_contract import PIPELINE_OUTPUT_ROOT, REPORT_OUTPUT_ROOT
from ppg_frailty.v5_reporting.analysis import build_analysis
from ppg_frailty.v5_reporting.collect import load_report_data
from ppg_frailty.v5_reporting.cli import _request as cli_request
from ppg_frailty.v5_reporting.cli import build_parser, main
from ppg_frailty.v5_reporting.contracts import (
    AnalysisProducts,
    ReportContractError,
    ReportRequest,
    RunSpec,
)
from ppg_frailty.v5_reporting.plots import generate_selected_figures
from ppg_frailty.v5_reporting.registry import (
    KNOWN_FIGURES,
    KNOWN_TABLES,
    resolve_selection,
)
from ppg_frailty.v5_reporting.validate import validate_report_data
from ppg_frailty.v5_reporting.writer import V5_ROOT, resolve_output_path, write_report


@contextmanager
def _workspace():
    REPORT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".v5-report-test-", dir=REPORT_OUTPUT_ROOT
    ) as raw:
        yield Path(raw)


def test_report_name_is_shared_pipeline_run_for_multiple_case_inputs():
    PIPELINE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".v5-pipeline-test-", dir=PIPELINE_OUTPUT_ROOT
    ) as raw:
        run = Path(raw)
        (run / "study_manifest.json").write_text("{}\n", encoding="utf-8")
        (run / "case_a").mkdir()
        (run / "case_b").mkdir()
        namespace = build_parser().parse_args(
            [
                "run",
                "--run",
                f"a={run / 'case_a'}",
                "--run",
                f"b={run / 'case_b'}",
            ]
        )
        request = cli_request(namespace)
        assert request.output_dir == REPORT_OUTPUT_ROOT / run.name


def _row(
    participant: str,
    label: int,
    probabilities: tuple[float, float, float],
    *,
    level: str,
    config_hash: str,
    prediction_kind: str = "single_model",
    member_index: int | None = None,
    training_seed: int | None = 42,
    member_training_seeds: tuple[int, ...] = (),
) -> OofPredictionRow:
    return OofPredictionRow(
        participant_id=participant,
        file_id=f"file-{participant}",
        role="R1",
        label=label,
        probabilities=probabilities,
        repeat=0,
        fold=0,
        split_seed=17,
        training_seed=training_seed,
        config_hash=config_hash,
        manifest_hash="shared-manifest",
        fold_hash="shared-fold",
        preprocessing_hash="preprocessing",
        feature_hash="feature",
        model_hash=f"model-{config_hash}",
        representation_mode="feature_vector",
        signal_route="direct",
        quality_score=1.0,
        retained=True,
        level=level,
        window_id=f"window-{participant}" if level == "window" else None,
        member_index=member_index,
        prediction_kind=prediction_kind,
        member_training_seeds=member_training_seeds,
        ensemble_base_model_id=("base" if prediction_kind != "single_model" else ""),
        class_order=(0, 1, 2),
        code_commit="commit",
        data_schema_id="data-v1",
        feature_schema_id="feature-v1",
        model_version="model-v1",
        aggregation_rule="mean",
        environment_hash="environment",
        manifest_version="manifest-v1",
        fold_registry_version="fold-registry-v1",
        artifact_reducer_name="identity",
        artifact_reducer_version="1",
        route_status="passed",
        source_snapshot_hash="source-snapshot",
    )


def _case(
    parent: Path,
    name: str,
    *,
    participants: tuple[str, ...] = ("p0", "p1", "p2"),
    independent: bool = False,
    ensemble: bool = False,
    threshold: float = 0.5,
) -> Path:
    root = parent / name
    root.mkdir()
    config_hash = f"config-{name}"
    base = ((0.78, 0.12, 0.10), (0.10, 0.80, 0.10), (0.10, 0.15, 0.75))
    layer_rows = {level: [] for level in ("window", "file", "role", "participant")}
    member_rows = []
    for index, participant in enumerate(participants):
        label = index % 3
        if ensemble:
            first = tuple(
                value + adjustment
                for value, adjustment in zip(base[label], (0.02, -0.01, -0.01))
            )
            second = tuple(
                value + adjustment
                for value, adjustment in zip(base[label], (-0.02, 0.01, 0.01))
            )
            average = tuple((a + b) / 2.0 for a, b in zip(first, second))
            for level in layer_rows:
                layer_rows[level].append(
                    _row(
                        participant,
                        label,
                        average,
                        level=level,
                        config_hash=config_hash,
                        prediction_kind="ensemble_average",
                        training_seed=None,
                        member_training_seeds=(101, 102),
                    )
                )
            member_rows.extend(
                (
                    _row(
                        participant,
                        label,
                        first,
                        level="participant",
                        config_hash=config_hash,
                        prediction_kind="ensemble_member",
                        member_index=0,
                        training_seed=101,
                    ),
                    _row(
                        participant,
                        label,
                        second,
                        level="participant",
                        config_hash=config_hash,
                        prediction_kind="ensemble_member",
                        member_index=1,
                        training_seed=102,
                    ),
                )
            )
        else:
            for level in layer_rows:
                layer_rows[level].append(
                    _row(
                        participant,
                        label,
                        base[label],
                        level=level,
                        config_hash=config_hash,
                    )
                )
    filenames = {
        "window": "oof_window_predictions.parquet",
        "file": "oof_file_predictions.parquet",
        "role": "oof_role_predictions.parquet",
        "participant": "oof_subject_predictions.parquet",
    }
    for level, rows in layer_rows.items():
        write_oof_parquet(rows, root / filenames[level])
    if member_rows:
        write_oof_parquet(member_rows, root / "oof_member_predictions.parquet")
    else:
        write_empty_oof_parquet(root / "oof_member_predictions.parquet", "single_model")
    manifest = {"evaluation_scope": "outer_oof", "independent_test": False}
    if independent:
        manifest = {
            "evaluation_scope": "independent_test",
            "independent_test": True,
            "independent_test_evidence": {
                "test_set_id": "held-out-site",
                "test_manifest_hash": "test-manifest",
                "training_roster_hash": "train-roster",
                "evaluation_roster_hash": "test-roster",
                "rosters_disjoint": True,
            },
        }
    (root / "evaluation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "config_id": name,
                "aggregation": {"threshold": threshold},
                "evaluation": {"statistics": {}},
            }
        ),
        encoding="utf-8",
    )
    return root


def _request(
    *runs: tuple[str, Path],
    mode: str = "single",
    reference: str | None = None,
    output: Path | None = None,
    factor_paths: tuple[str, ...] = (),
) -> ReportRequest:
    return ReportRequest(
        mode=mode,
        runs=tuple(RunSpec(name, path) for name, path in runs),
        output_dir=output,
        reference_case=reference,
        factor_paths=factor_paths,
        bootstrap_resamples=5,
        permutation_resamples=5,
    )


def test_registry_is_strict_and_explicit_outputs_are_exact():
    selection = resolve_selection(
        mode="single",
        presets=(),
        modules=("summary",),
        figures=("leaderboard",),
        tables=(),
    )
    assert selection.figures == ("leaderboard",)
    assert selection.tables == ()
    with pytest.raises(ReportContractError, match="unknown report module"):
        resolve_selection(
            mode="single",
            presets=(),
            modules=("typo",),
            figures=None,
            tables=None,
        )
    full_test = resolve_selection(
        mode="test", presets=("full",), modules=(), figures=(), tables=()
    )
    assert "hierarchy" in full_test.modules
    assert "ablation" not in full_test.modules
    assert "historical" not in full_test.modules


def test_five_layer_load_member_validation_and_test_fail_closed():
    with _workspace() as workspace:
        outer = _case(workspace, "outer", ensemble=True)
        request = _request(("outer", outer))
        data = load_report_data(request)
        report = validate_report_data(data, request)
        assert report.status == "passed"
        assert len(data.artifact_records) == 5
        assert len(data.layer_rows["member"]) == 6

        with pytest.raises(ReportContractError, match="rejects outer-OOF"):
            validate_report_data(data, _request(("outer", outer), mode="test"))

        test_root = _case(workspace, "test", independent=True)
        test_request = _request(("test", test_root), mode="test")
        test_data = load_report_data(test_request)
        assert validate_report_data(test_data, test_request).status == "passed"
        test_selection = resolve_selection(
            mode="test",
            presets=(),
            modules=("summary",),
            figures=(),
            tables=("case_summary", "classification_prediction_scores"),
        )
        test_products = build_analysis(test_data, test_request, test_selection)
        assert test_products.tables["case_summary"][0][
            "frailty_classification_evaluation_scope"
        ] == "independent_test_participant"
        assert {
            row["evaluation_id"]
            for row in test_products.tables["classification_prediction_scores"]
        } == {"participant_independent_test"}


def test_only_verified_single_model_v2_study_may_lack_member_artifact():
    with _workspace() as workspace:
        single = _case(workspace, "single")
        request = _request(("single", single))
        direct = load_report_data(request)
        without_member_record = tuple(
            row for row in direct.artifact_records if row.layer != "member"
        )
        with pytest.raises(ReportContractError, match="requires a member artifact"):
            validate_report_data(
                replace(direct, artifact_records=without_member_record), request
            )
        legacy = replace(
            direct,
            source_kind="v2_study",
            legacy_v2_cases=frozenset({"single"}),
            artifact_records=without_member_record,
        )
        report = validate_report_data(legacy, request)
        assert report.status == "passed_with_warnings"
        assert report.issues[0].code == (
            "legacy_v2_member_artifact_absent_non_ensemble"
        )

        ensemble = _case(workspace, "ensemble", ensemble=True)
        ensemble_data = load_report_data(_request(("ensemble", ensemble)))
        missing_ensemble_members = replace(
            ensemble_data,
            source_kind="v2_study",
            legacy_v2_cases=frozenset({"ensemble"}),
            artifact_records=tuple(
                row for row in ensemble_data.artifact_records if row.layer != "member"
            ),
            layer_rows={**ensemble_data.layer_rows, "member": ()},
        )
        with pytest.raises(ReportContractError, match="requires a member artifact"):
            validate_report_data(
                missing_ensemble_members,
                _request(("ensemble", ensemble)),
            )


def test_disabling_v2_compatibility_does_not_reject_a_v5_study():
    with _workspace() as workspace:
        study = workspace / "v5-study"
        study.mkdir()
        case = _case(study, "case")
        (case / "case_result.json").write_text(
            json.dumps(
                {
                    "case_id": "case",
                    "status": "passed",
                    "artifact_root": ".",
                    "result": {"cell_results": []},
                }
            ),
            encoding="utf-8",
        )
        (study / "study_plan.yaml").write_text(
            yaml.safe_dump({"execution": {"repeats": [0], "folds": [0]}}),
            encoding="utf-8",
        )
        (study / "study_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "ppg_frailty.study_manifest.v2",
                    "output_layout": "comparison/repeat/fold",
                    "cases": [
                        {
                            "case_id": "case",
                            "case_directory": "case",
                            "resolved_config_path": "case/resolved_config.yaml",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        request = replace(
            _request(("study", study)),
            allow_v2_compatibility=False,
        )

        data = load_report_data(request)

        assert data.source_kind == "v5_study"
        assert not data.legacy_v2_cases


def test_comparison_roster_and_ablation_contract_are_fail_closed():
    with _workspace() as workspace:
        reference = _case(workspace, "reference", threshold=0.5)
        candidate = _case(workspace, "candidate", threshold=0.7)
        request = _request(
            ("reference", reference),
            ("candidate", candidate),
            mode="ablation",
            reference="reference",
            factor_paths=("aggregation.threshold",),
        )
        data = load_report_data(request)
        assert validate_report_data(data, request).status == "passed"
        selection = resolve_selection(
            mode="ablation",
            presets=(),
            modules=("ablation",),
            figures=(),
            tables=("ablation_contract", "paired_participant_inference"),
        )
        products = build_analysis(data, request, selection)
        assert products.tables["ablation_contract"][0]["contract_status"] == "matched"
        assert {
            row["comparison_contract_status"]
            for row in products.tables["paired_participant_inference"]
        } == {"matched_complete_roster"}

        mismatch = _case(
            workspace,
            "mismatch",
            participants=("p0", "p1", "different"),
            threshold=0.7,
        )
        bad = _request(
            ("reference", reference),
            ("mismatch", mismatch),
            mode="comparison",
            reference="reference",
        )
        with pytest.raises(ReportContractError, match="paired roster mismatch"):
            validate_report_data(load_report_data(bad), bad)


def test_nondefault_calibration_bins_and_holm_alpha_are_effective():
    with _workspace() as workspace:
        reference = _case(workspace, "reference")
        candidate = _case(workspace, "candidate")
        request = replace(
            _request(
                ("reference", reference),
                ("candidate", candidate),
                mode="comparison",
                reference="reference",
            ),
            calibration_bins=7,
            alpha=0.20,
        )
        data = load_report_data(request)
        assert validate_report_data(data, request).status == "passed"
        selection = resolve_selection(
            mode="comparison",
            presets=(),
            modules=("calibration", "comparison"),
            figures=(),
            tables=("calibration_bins", "paired_participant_inference"),
        )
        products = build_analysis(data, request, selection)

        calibration = products.tables["calibration_bins"]
        assert len(calibration) == 2 * request.calibration_bins
        assert {row["bin_index"] for row in calibration} == set(
            range(request.calibration_bins)
        )

        tested = [
            row
            for row in products.tables["paired_participant_inference"]
            if row["raw_two_sided_p_value"] is not None
        ]
        assert tested
        assert {row["alpha"] for row in tested} == {request.alpha}
        assert all(
            row["reject_null_after_holm"]
            == (row["holm_adjusted_p_value"] <= request.alpha)
            for row in tested
        )


def test_fold_model_table_projects_v5_learned_checkpoint():
    with _workspace() as workspace:
        case = _case(workspace, "case")
        fold = case / "repeat_00" / "fold_00"
        fold.mkdir(parents=True)
        (fold / "run_manifest.json").write_text(
            json.dumps(
                {
                    "cell": {
                        "status": "passed",
                        "repeat_index": 0,
                        "fold_index": 0,
                        "learned_model_checkpoint": {
                            "schema_version": "ppg_frailty.fold_checkpoint.v1",
                            "manifest_path": "model_checkpoint/manifest.json",
                            "manifest_sha256": "manifest-sha",
                            "state_sha256": "state-sha",
                            "deployment_status": "research_only",
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        request = _request(("case", case))
        data = replace(load_report_data(request), source_kind="v5_study")
        selection = resolve_selection(
            mode="single",
            presets=(),
            modules=("audit",),
            figures=(),
            tables=("fold_model_parameters",),
        )

        row = build_analysis(data, request, selection).tables["fold_model_parameters"][0]

        assert row["learned_weight_checkpoint"] == "repeat_00/fold_00/model_checkpoint/manifest.json"
        assert row["checkpoint_schema"] == "ppg_frailty.fold_checkpoint.v1"
        assert row["checkpoint_manifest_sha256"] == "manifest-sha"
        assert row["checkpoint_state_sha256"] == "state-sha"
        assert row["checkpoint_deployment_status"] == "research_only"


def test_output_is_v5_confined_and_contains_dual_tables_and_indices():
    with _workspace() as workspace:
        case = _case(workspace, "case")
        output = workspace / "report"
        request = _request(("case", case), output=output)
        selection = resolve_selection(
            mode="single",
            presets=(),
            modules=("audit",),
            figures=(),
            tables=("input_artifacts", "participant_predictions"),
        )
        data = load_report_data(request)
        validation = validate_report_data(data, request)
        products = build_analysis(data, request, selection)
        assert KNOWN_TABLES <= set(products.tables)
        result = write_report(data, products, request, selection, validation)
        assert result == output
        assert (output / "tables/input_artifacts.csv").is_file()
        assert (output / "tables/input_artifacts.json").is_file()
        assert (output / "tables/report_tables.xlsx").is_file()
        assert (output / "STUDY_SUMMARY.html").is_file()
        assert (output / "STUDY_SUMMARY.md").is_file()
        assert (output / "analysis_manifest.json").is_file()
        index = json.loads((output / "outputs_index.json").read_text(encoding="utf-8"))
        indexed = {row["path"] for row in index["entries"]}
        assert "analysis_manifest.json" in indexed
        assert "tables/participant_predictions.csv" in indexed
        assert not (output / "figures").exists()
        with pytest.raises(ReportContractError, match="inside .*report_output"):
            resolve_output_path(Path("/tmp/v5-report-must-not-write"))
        with pytest.raises(ReportContractError, match="already exists"):
            resolve_output_path(output)


def test_figure_dispatch_does_not_append(monkeypatch: pytest.MonkeyPatch):
    with _workspace() as workspace:
        case = _case(workspace, "case")
        request = _request(("case", case))
        data = load_report_data(request)
        selection = resolve_selection(
            mode="single",
            presets=(),
            modules=("summary",),
            figures=("leaderboard",),
            tables=(),
        )
        products = AnalysisProducts(analysis=object(), tables={})
        called = []

        def fake(_collected, _analysis, directory, *, modules):
            called.append(tuple(modules))
            path = Path(directory) / "leaderboard.png"
            path.write_bytes(b"png")
            return (
                {
                    "figure": "leaderboard",
                    "status": "generated",
                    "path": "figures/leaderboard.png",
                    "reason": "",
                },
            )

        monkeypatch.setattr(
            "ppg_frailty.v5_reporting.plots.generate_static_figures", fake
        )
        statuses = generate_selected_figures(
            data,
            products,
            request,
            selection.figures,
            workspace / "figures",
        )
        assert called == [("leaderboard",)]
        assert tuple(row["figure"] for row in statuses) == ("leaderboard",)
        assert sorted(path.name for path in (workspace / "figures").iterdir()) == [
            "leaderboard.png"
        ]


def test_every_registered_figure_dispatches_to_a_real_artifact():
    with _workspace() as workspace:
        case = _case(workspace, "case")
        request = _request(("case", case))
        data = load_report_data(request)
        selection = resolve_selection(
            mode="single",
            presets=(),
            modules=("summary",),
            figures=(),
            tables=(),
        )
        products = build_analysis(data, request, selection)
        statuses = generate_selected_figures(
            data,
            products,
            request,
            tuple(sorted(KNOWN_FIGURES)),
            workspace / "figures",
        )

        assert {row["figure"] for row in statuses} == set(KNOWN_FIGURES)
        assert len(statuses) == len(KNOWN_FIGURES) == 36
        assert all(row["status"] in {"generated", "N/A"} for row in statuses)
        assert all((workspace / str(row["path"])).is_file() for row in statuses)


def test_cli_list_validate_and_run(capsys: pytest.CaptureFixture[str]):
    assert main(["list"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["modes"] == ["ablation", "comparison", "single", "test"]
    with _workspace() as workspace:
        case = _case(workspace, "case")
        common = [
            "--run",
            f"case={case}",
            "--module",
            "audit",
            "--figures",
            "none",
            "--tables",
            "input_artifacts",
            "--bootstrap-resamples",
            "5",
            "--permutation-resamples",
            "5",
        ]
        assert main(["validate", *common]) == 0
        validated = json.loads(capsys.readouterr().out)
        assert validated["status"] == "passed"
        output = workspace / "cli-report"
        assert main(["run", *common, "--output-dir", str(output)]) == 0
        completed = json.loads(capsys.readouterr().out)
        assert completed["status"] == "complete"
        assert (output / "analysis_manifest.json").is_file()
        (output / "tables/report_tables.xlsx").unlink()
        assert main(["export-excel", "--report-output", str(output)]) == 0
        exported = json.loads(capsys.readouterr().out)
        assert exported["status"] == "complete"
        assert (output / "tables/report_tables.xlsx").is_file()

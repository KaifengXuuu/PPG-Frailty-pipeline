"""Compose V2 analysis algorithms and V5-only audit/member tables."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import ppg_frailty.reporting.analyze as v2_analyze
from ppg_frailty.reporting.analyze import analyze_study
from ppg_frailty.reporting.components import (
    build_pipeline_test_component_rows,
    build_top_model_configuration_rows,
)
from ppg_frailty.reporting.conclusions import (
    classification_comparison_rows,
    classification_conclusion_rows,
    holm_adjust_paired_inference_rows,
    paired_repeat_deltas_against_reference,
)
from ppg_frailty.reporting.profiles import reporter_profile_rows
from ppg_frailty.reporting.reproducibility import audit_study_reproducibility
from ppg_frailty.training import evaluate_predictions

from .contracts import (
    AnalysisProducts,
    LoadedReportData,
    ReportRequest,
    ResolvedSelection,
)
from .registry import MODULE_BY_NAME
from .validate import changed_config_paths


def _config_value(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _statistics_policy(request: ReportRequest) -> dict[str, Any]:
    return {
        "cluster_unit": (
            "participant_with_all_independent_test_predictions"
            if request.mode == "test"
            else "participant_with_all_five_repeat_oof_predictions"
        ),
        "paired_exchange_unit": "participant",
        "multiplicity_correction": "holm_within_comparison_family",
        "affects_automatic_selection": False,
        "bootstrap_replicates": request.bootstrap_resamples,
        "paired_permutation_replicates": request.permutation_resamples,
        "seed": request.statistics_seed,
        "alpha": request.alpha,
    }


def _prepared_collected(
    data: LoadedReportData,
    request: ReportRequest,
    selection: ResolvedSelection,
) -> Any:
    collected = data.collected
    manifest = dict(collected.manifest)
    if request.reference_case is not None:
        manifest["reference_case_id"] = request.reference_case
        manifest["reference_case"] = request.reference_case
    cases = []
    for raw in manifest.get("cases", ()):
        if not isinstance(raw, Mapping):
            continue
        case = dict(raw)
        case["is_reference"] = str(case.get("case_id")) == request.reference_case
        if request.mode == "ablation":
            config = data.config_by_case.get(str(case.get("case_id")), {})
            case["changed_values"] = {path: _config_value(config, path) for path in request.factor_paths}
        cases.append(case)
    manifest["cases"] = tuple(cases)

    plan = dict(collected.plan)
    study = plan.get("study")
    study = dict(study) if isinstance(study, Mapping) else {}
    study.setdefault("study_id", request.comparison_family)
    plan["study"] = study
    if request.mode == "ablation":
        plan["axes"] = tuple({"path": path} for path in request.factor_paths)
    report = plan.get("report")
    report = dict(report) if isinstance(report, Mapping) else {}
    report.update(
        {
            "classification_tsne_random_state": request.statistics_seed,
            "classification_tsne_max_samples": 5000,
            "classification_roc_macro_grid_points": 201,
            "calibration_bins": request.calibration_bins,
        }
    )
    plan["report"] = report
    policy = _statistics_policy(request)
    resolved_by_case = {
        str(row.get("case_id")): row for row in collected.resolved_aggregation_configs if row.get("case_id") is not None
    }
    resolved = tuple(
        {
            **dict(resolved_by_case.get(case_id, {})),
            "case_id": case_id,
            "aggregation": dict(
                resolved_by_case.get(case_id, {}).get("aggregation", {})
                if isinstance(
                    resolved_by_case.get(case_id, {}).get("aggregation", {}),
                    Mapping,
                )
                else {}
            ),
            "evaluation_statistics": policy,
        }
        for case_id in data.case_ids
    )

    def requested(name: str) -> bool:
        spec = MODULE_BY_NAME[name]
        return (
            name in selection.modules
            or bool(set(spec.tables) & set(selection.tables))
            or bool(set(spec.figures) & set(selection.figures))
        )

    hierarchy = requested("hierarchy")
    learning = requested("learning")
    return replace(
        collected,
        plan=plan,
        manifest=manifest,
        resolved_aggregation_configs=resolved,
        window_oof_rows=collected.window_oof_rows if hierarchy else (),
        file_oof_rows=collected.file_oof_rows if hierarchy else (),
        role_oof_rows=collected.role_oof_rows if hierarchy else (),
        # Quality evidence also contributes scope/status fields in the common
        # case summary, so it must not vary with the selected plot modules.
        quality_rows=collected.quality_rows,
        history_rows=collected.history_rows if learning else (),
    )


def _scoped_value(value: Any) -> Any:
    if isinstance(value, str):
        return (
            value.replace(
                "available_recomputed_from_participant_outer_oof",
                "available_recomputed_from_independent_test",
            )
            .replace("outer_heldout_participant_oof", "independent_test_participant")
            .replace("participant_outer_oof", "participant_independent_test")
            .replace("outer-OOF", "independent-test")
            .replace("outer OOF", "independent test")
            .replace("outer_oof", "independent_test")
        )
    if isinstance(value, Mapping):
        return {key: _scoped_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_scoped_value(item) for item in value)
    if isinstance(value, list):
        return [_scoped_value(item) for item in value]
    return value


def _scope_analysis(analysis: Any, mode: str) -> Any:
    if mode != "test":
        return analysis
    changes = {item.name: _scoped_value(getattr(analysis, item.name)) for item in fields(analysis)}
    return replace(analysis, **changes)


def _run_v2_analysis(collected: Any, selection: ResolvedSelection) -> Any:
    """Avoid report-only t-SNE work unless it was explicitly selected."""

    wants_tsne = (
        "tsne" in selection.modules
        or "classification_prediction_tsne" in selection.figures
        or "classification_prediction_tsne" in selection.tables
    )
    if wants_tsne:
        return analyze_study(collected)
    original = v2_analyze.classification_tsne_rows
    v2_analyze.classification_tsne_rows = lambda *args, **kwargs: ()
    try:
        analysis = analyze_study(collected)
    finally:
        v2_analyze.classification_tsne_rows = original
    return replace(
        analysis,
        classification_diagnostic_status=tuple(
            {
                **dict(row),
                "prediction_tsne_status": "not_requested",
                "tsne_point_count": 0,
            }
            for row in analysis.classification_diagnostic_status
        ),
    )


def _flatten_config(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        output: list[tuple[str, Any]] = []
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            output.extend(_flatten_config(value[key], child))
        return output
    return [(prefix, list(value) if isinstance(value, tuple) else value)]


def _config_rows(data: LoadedReportData) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "case_id": case_id,
            "parameter_path": path,
            "value_json": json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        }
        for case_id, config in sorted(data.config_by_case.items())
        for path, value in _flatten_config(config)
    )


def _fold_model_rows(data: LoadedReportData) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for case_id, root in sorted(data.source_root_by_case.items()):
        for path in sorted(root.rglob("run_manifest.json"), key=str):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            cell = payload.get("cell") if isinstance(payload, Mapping) else None
            if not isinstance(cell, Mapping):
                continue
            fitted = cell.get("fitted_provenance")
            fitted = fitted if isinstance(fitted, Mapping) else {}
            factory = cell.get("model_factory_provenance")
            factory = factory if isinstance(factory, Mapping) else {}
            frozen = cell.get("frozen_model_run_provenance")
            frozen = frozen if isinstance(frozen, Mapping) else {}
            operational = cell.get("operational_metrics")
            operational = operational if isinstance(operational, Mapping) else {}
            checkpoint = cell.get("learned_model_checkpoint")
            checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
            checkpoint_path = str(checkpoint.get("manifest_path", "")).strip()
            if checkpoint_path and not Path(checkpoint_path).is_absolute():
                candidate = (path.parent / checkpoint_path).resolve()
                try:
                    checkpoint_path = candidate.relative_to(root.resolve()).as_posix()
                except ValueError:
                    # Preserve an unusual legacy declaration without turning a
                    # report projection into a filesystem policy decision.
                    pass
            rows.append(
                {
                    "case_id": case_id,
                    "repeat": cell.get("repeat_index"),
                    "fold": cell.get("fold_index"),
                    "status": cell.get("status", payload.get("status")),
                    "model_id": cell.get("model_machine_id", cell.get("model_id")),
                    "representation_mode": cell.get("representation_mode"),
                    "split_seed": cell.get("split_seed"),
                    "training_seed": cell.get("training_seed"),
                    "member_training_seeds": cell.get("member_training_seeds", ()),
                    "parameter_count": operational.get("parameter_count"),
                    "config_hash": cell.get("config_hash"),
                    "model_hash": cell.get("model_hash"),
                    "state_hash": fitted.get("state_hash"),
                    "fold_hash": fitted.get("fold_hash", cell.get("fold_hash")),
                    "preprocessing_hash": cell.get("preprocessing_hash"),
                    "feature_hash": cell.get("feature_hash"),
                    "architecture_parameters": frozen.get("architecture_parameters", {}),
                    "model_factory_provenance": factory,
                    "fitted_provenance": fitted,
                    "checkpoint_schema": checkpoint.get("schema_version", ""),
                    "learned_weight_checkpoint": checkpoint_path
                    or (
                        "not_persisted_by_v2_outer_cv"
                        if data.source_kind.startswith("v2_")
                        or case_id in data.legacy_v2_cases
                        else "not_declared"
                    ),
                    "checkpoint_manifest_sha256": checkpoint.get("manifest_sha256", ""),
                    "checkpoint_state_sha256": checkpoint.get("state_sha256", ""),
                    "checkpoint_deployment_status": checkpoint.get("deployment_status", ""),
                    "run_manifest_path": str(path),
                }
            )
    return tuple(rows)


def _member_metric_rows(data: LoadedReportData, request: ReportRequest) -> tuple[Mapping[str, Any], ...]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in data.layer_rows["member"]:
        key = (
            str(row["case_id"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["member_index"]),
            int(row["training_seed"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[Mapping[str, Any]] = []
    for key, rows in sorted(groups.items()):
        retained = [row for row in rows if bool(row["retained"])]
        base = {
            "case_id": key[0],
            "repeat": key[1],
            "fold": key[2],
            "member_index": key[3],
            "training_seed": key[4],
            "n_total": len(rows),
            "n_retained": len(retained),
            "coverage_rate": len(retained) / len(rows) if rows else None,
        }
        if not retained:
            output.append({**base, "status": "N/A_no_retained_predictions"})
            continue
        class_order = tuple(int(value) for value in retained[0]["class_order"])
        metrics = evaluate_predictions(
            np.asarray([int(row["label"]) for row in retained], dtype=np.int64),
            np.asarray([row["probabilities"] for row in retained], dtype=np.float64),
            class_order=class_order,
            n_total=len(rows),
            ece_bins=request.calibration_bins,
        )
        output.append({**base, "status": "available", **asdict(metrics)})
    return tuple(output)


def _ablation_rows(data: LoadedReportData, request: ReportRequest) -> tuple[Mapping[str, Any], ...]:
    if request.mode != "ablation":
        return ()
    reference_id = str(request.reference_case)
    reference = data.config_by_case[reference_id]
    return tuple(
        {
            "comparison_family": request.comparison_family,
            "reference_case_id": reference_id,
            "candidate_case_id": case_id,
            "declared_factor_paths": request.factor_paths,
            "observed_changed_paths": changed_config_paths(reference, config),
            "contract_status": "matched",
        }
        for case_id, config in sorted(data.config_by_case.items())
        if case_id != reference_id
    )


def _ordinary_audit_tables(
    data: LoadedReportData,
    request: ReportRequest,
    collected: Any,
    analysis: Any,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Return the evidence tables formerly assembled inside the V2 writer."""

    components = tuple(build_pipeline_test_component_rows(collected.root, collected.manifest))
    profiles = tuple(reporter_profile_rows(components))
    options = collected.plan.get("report", {})
    options = options if isinstance(options, Mapping) else {}
    top_configs = tuple(
        build_top_model_configuration_rows(
            collected.root,
            collected.manifest,
            analysis.predictive_leaderboard,
            top_k=int(options.get("detailed_configuration_top_k", 0)),
        )
    )
    audit = audit_study_reproducibility(collected)

    reference = request.reference_case
    if reference is None:
        raw = collected.manifest.get("reference_case_id", collected.manifest.get("reference_case"))
        reference = None if raw in (None, "") else str(raw)
    raw_repeats = collected.plan.get("execution", {}).get("repeats", ())
    repeats = tuple(int(value) for value in raw_repeats) if isinstance(raw_repeats, (list, tuple)) else ()
    case_ids = tuple(
        str(row["case_id"])
        for row in collected.manifest.get("cases", ())
        if isinstance(row, Mapping) and row.get("case_id") not in (None, "")
    )
    repeat_deltas = tuple(
        paired_repeat_deltas_against_reference(
            analysis.classification_prediction_scores,
            reference_case_id=reference,
            comparison_family=request.comparison_family,
            comparison_role="declared_reference_comparison",
            candidate_case_ids=case_ids,
            expected_repeats=repeats or None,
        )
        if reference is not None
        else ()
    )
    comparisons = tuple(
        classification_comparison_rows(
            analysis.case_summary,
            paired_inference=analysis.paired_participant_inference,
        )
    )
    conclusions = tuple(
        classification_conclusion_rows(
            comparisons,
            selected_case_id=None,
            selection_basis="manual review only; no automatic report selection",
            study_role=str(collected.plan.get("study", {}).get("decision_role", "study")),
            planned_case_count=int(collected.manifest.get("planned_case_count", 0) or 0),
            incomplete_case_count=len(analysis.incomplete_cases),
            inference_reference_case_ids=(reference,) if reference else (),
        )
    )
    return {
        "test_components": components,
        "reporter_profiles": profiles,
        "top_model_complete_configurations": top_configs,
        "varied_parameters": tuple(collected.varied_parameters),
        "controlled_parameters": tuple(collected.controlled_parameters),
        "pairwise_repeat_metric_deltas": repeat_deltas,
        "comparison_conclusions": comparisons,
        "selection_conclusions": conclusions,
        "reproducibility_summary": (audit.summary,),
        "reproducibility_cases": tuple(audit.case_rows),
        "reproducibility_cells": tuple(audit.cell_rows),
        "reproducibility_splits": tuple(audit.split_rows),
        "reproducibility_issues": tuple(audit.issues),
    }


def build_analysis(
    data: LoadedReportData,
    request: ReportRequest,
    selection: ResolvedSelection,
) -> AnalysisProducts:
    """Run existing algorithms once and expose normalized named tables."""

    collected = _prepared_collected(data, request, selection)
    analysis = _scope_analysis(_run_v2_analysis(collected, selection), request.mode)
    # The copied V2 analyzer correctly exposes raw permutation P values and uses
    # the same Holm implementation, but its application-level alpha was fixed at
    # 0.05.  Reapply only that final correction/decision layer so the V5 CLI
    # parameter is effective without changing resampling or metric algorithms.
    analysis = replace(
        analysis,
        paired_participant_inference=tuple(
            holm_adjust_paired_inference_rows(
                analysis.paired_participant_inference,
                alpha=request.alpha,
            )
        ),
    )
    tables: dict[str, tuple[Mapping[str, Any], ...]] = {
        item.name: tuple(getattr(analysis, item.name))
        for item in fields(analysis)
        if isinstance(getattr(analysis, item.name), tuple)
        and (not getattr(analysis, item.name) or isinstance(getattr(analysis, item.name)[0], Mapping))
    }
    tables.update(
        {
            "input_artifacts": tuple({**asdict(row), "path": str(row.path)} for row in data.artifact_records),
            "input_manifests": tuple(
                {"case_id": key, "manifest": value} for key, value in sorted(data.manifest_by_case.items())
            ),
            "resolved_config_parameters": _config_rows(data),
            "fold_model_parameters": _fold_model_rows(data),
            "window_predictions": data.layer_rows["window"],
            "file_predictions": data.layer_rows["file"],
            "role_predictions": data.layer_rows["role"],
            "participant_predictions": data.layer_rows["participant"],
            "member_predictions": data.layer_rows["member"],
            "case_records": tuple(collected.case_records),
            "cell_metrics_raw": tuple(collected.cell_rows),
            "training_history_raw": tuple(collected.history_rows),
            "quality_diagnostics_raw": tuple(collected.quality_rows),
            "preprocessing_cache": tuple(collected.preprocessing_cache_rows),
            "deployment_measurements": tuple(analysis.deployment_table),
            "ensemble_member_predictions": data.layer_rows["member"],
            "ensemble_member_metrics": _member_metric_rows(data, request),
            "ablation_contract": _ablation_rows(data, request),
        }
    )
    tables.update(_ordinary_audit_tables(data, request, collected, analysis))
    return AnalysisProducts(
        analysis=analysis,
        tables=tables,
        notes=tuple(analysis.notes),
    )


__all__ = ["build_analysis"]

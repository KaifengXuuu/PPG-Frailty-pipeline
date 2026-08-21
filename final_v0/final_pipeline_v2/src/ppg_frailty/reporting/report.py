"""Write complete, human-readable study reports and an output inventory."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .analyze import StudyAnalysis, analyze_study
from .collect import CollectedStudy, collect_study
from .plots import clear_static_figure_artifacts, generate_static_figures
from .reproducibility import (
    NOT_VERIFIABLE,
    ReproducibilityAudit,
    audit_study_reproducibility,
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", r"\|")


def _markdown_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
) -> list[str]:
    if not rows:
        return ["N/A — no rows were available.", ""]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_fmt(row.get(field)) for field, _ in columns)
            + " |"
        )
    lines.append("")
    return lines


def _study_info(collected: CollectedStudy) -> Mapping[str, Any]:
    value = collected.plan.get("study", {})
    return value if isinstance(value, Mapping) else {}


_LEGACY_BRIDGE_NUMERIC_COLUMNS = (
    ("numeric_profile_order", "Numeric order"),
    ("model", "Model"),
    ("profile", "Profile"),
    ("previous_numeric_profile", "Previous numeric profile"),
    ("numeric_comparison", "Predefined comparison"),
    ("BA_legacy_aggregation", "BA legacy W"),
    ("BA_line_a_aggregation", "BA Line A"),
    ("BA_v2_aggregation", "BA Line B"),
    ("delta_BA_legacy_aggregation", "Δ BA legacy W"),
    ("delta_BA_line_a_aggregation", "Δ BA Line A"),
    ("delta_BA_v2_aggregation", "Δ BA Line B"),
    ("macroF1_legacy_aggregation", "Macro-F1 legacy W"),
    ("macroF1_line_a_aggregation", "Macro-F1 Line A"),
    ("macroF1_v2_aggregation", "Macro-F1 Line B"),
    ("delta_macroF1_legacy_aggregation", "Δ Macro-F1 legacy W"),
    ("delta_macroF1_line_a_aggregation", "Δ Macro-F1 Line A"),
    ("delta_macroF1_v2_aggregation", "Δ Macro-F1 Line B"),
    ("worst_class_F1", "Worst-class F1 Line B"),
    ("delta_worst_class_F1", "Δ worst-class F1"),
    ("contrast_metrics_available", "Contrast available"),
    ("interpretation", "Interpretation"),
)


_LEGACY_BRIDGE_EXECUTION_COLUMNS = (
    ("execution_order", "Execution order"),
    ("model", "Model"),
    ("profile", "Profile"),
    ("previous_execution_profile", "Previous execution profile"),
    ("execution_transition", "Execution transition"),
    ("BA_legacy_aggregation", "BA legacy W"),
    ("BA_line_a_aggregation", "BA Line A"),
    ("BA_v2_aggregation", "BA Line B"),
    ("macroF1_legacy_aggregation", "Macro-F1 legacy W"),
    ("macroF1_line_a_aggregation", "Macro-F1 Line A"),
    ("macroF1_v2_aggregation", "Macro-F1 Line B"),
    ("worst_class_F1", "Worst-class F1 Line B"),
    ("execution_transition_is_ablation", "Transition is ablation"),
    ("interpretation", "Interpretation"),
)


def _is_stage3_centered_star(plan: Mapping[str, Any]) -> bool:
    bridge = plan.get("legacy_bridge")
    return isinstance(bridge, Mapping) and str(bridge.get("design", "")) == (
        "centered_star_v1"
    )


_STAGE3_STAR_ABSOLUTE_COLUMNS = (
    ("model", "Model"),
    ("profile", "Profile"),
    ("factor_id", "Factor"),
    ("native_aggregation_view", "Native endpoint"),
    ("native_balanced_accuracy", "Native BA"),
    ("native_macro_f1", "Native Macro-F1"),
    ("native_worst_class_f1", "Native worst-class F1"),
    ("BA_W_sensitivity", "BA W"),
    ("BA_A_sensitivity", "BA A"),
    ("BA_B_sensitivity", "BA B"),
    ("passed_cell_count", "Passed cells"),
    ("single_factor_audit", "Factor audit"),
    ("cross_model_profile_controls_match", "Cross-model controls match"),
)


_STAGE3_STAR_CONTRAST_COLUMNS = (
    ("model", "Model"),
    ("factor_id", "Factor"),
    ("reference_profile", "Reference"),
    ("variant_profile", "Variant"),
    ("reference_native_aggregation_view", "Reference endpoint"),
    ("variant_native_aggregation_view", "Variant endpoint"),
    ("delta_native_balanced_accuracy", "Native Δ BA"),
    ("delta_native_macro_f1", "Native Δ Macro-F1"),
    ("delta_native_worst_class_f1", "Native Δ worst-class F1"),
    ("delta_balanced_accuracy_W_sensitivity_only", "Sensitivity-only Δ BA W"),
    ("delta_balanced_accuracy_A_sensitivity_only", "Sensitivity-only Δ BA A"),
    ("delta_balanced_accuracy_B_sensitivity_only", "Sensitivity-only Δ BA B"),
    ("actual_changed_control_paths", "Actual changed paths"),
    ("single_factor_audit", "Factor audit"),
    ("seed_match", "Seeds match"),
    ("split_hash_match", "Split hashes match"),
    ("heldout_roster_hash_match", "Held-out rosters match"),
    ("contrast_metrics_available", "Available"),
    ("unavailable_reasons", "N/A reasons"),
    (
        "report_view_factor_training_controls_identical",
        "B0/B7 training controls identical",
    ),
    (
        "report_view_factor_window_oof_probabilities_identical",
        "B0/B7 window OOF identical",
    ),
    ("matched_window_oof_row_count", "B0/B7 matched window rows"),
    ("window_oof_probability_max_abs_diff", "B0/B7 max |probability diff|"),
    ("window_oof_identity_audit_status", "B0/B7 identity audit"),
)


_STAGE3_STAR_FOLD_COLUMNS = (
    ("model", "Model"),
    ("factor_id", "Factor"),
    ("reference_profile", "Reference"),
    ("variant_profile", "Variant"),
    ("repeat", "Repeat"),
    ("fold", "Fold"),
    ("delta_native_balanced_accuracy", "Native Δ BA"),
    ("delta_native_macro_f1", "Native Δ Macro-F1"),
    ("delta_native_worst_class_f1", "Native Δ worst-class F1"),
    ("contrast_metrics_available", "Available"),
    ("inference", "Inference"),
)


_STAGE3_STAR_EXECUTION_COLUMNS = (
    ("execution_order", "Execution order"),
    ("model", "Model"),
    ("profile", "Profile"),
    ("factor_id", "Factor"),
    ("native_aggregation_view", "Native endpoint"),
    ("native_balanced_accuracy", "Native BA"),
    ("native_macro_f1", "Native Macro-F1"),
    ("native_worst_class_f1", "Native worst-class F1"),
    ("execution_transition", "Scheduling transition"),
    ("execution_transition_is_ablation", "Transition is ablation"),
)

_STAGE3_STAR_TABLES = (
    (
        "stage3_star_absolute",
        "Stage 3 centered-star absolute endpoints",
        "Sixteen absolute model/profile endpoints. W/A/B are same-OOF sensitivity views; each row declares its native endpoint.",
        _STAGE3_STAR_ABSOLUTE_COLUMNS,
        "Sixteen absolute model/profile endpoints with native and W/A/B same-OOF metrics",
    ),
    (
        "stage3_star_contrasts",
        "Stage 3 centered-star contrasts",
        "Fourteen same-model B0→variant contrasts. Availability requires five passed cells plus matching seeds, split hashes, held-out rosters, native metrics, and exact factor paths; cross-model deltas are prohibited. B0/B7 also audits training-control and window-OOF identity.",
        _STAGE3_STAR_CONTRAST_COLUMNS,
        "Fourteen same-model B0-centered contrasts with factor and reproducibility audits",
    ),
    (
        "stage3_star_fold_contrasts",
        "Stage 3 centered-star matched-fold deltas",
        "The 14×5 fold deltas are descriptive only: no CI or significance claim. Seven contrasts within each model share the same correlated B0.",
        _STAGE3_STAR_FOLD_COLUMNS,
        "Seventy matched-fold descriptive deltas; no CI or significance inference",
    ),
    (
        "stage3_star_execution",
        "Stage 3 centered-star execution order",
        "Absolute scheduling rows only; neighbouring execution rows are not ablation contrasts.",
        _STAGE3_STAR_EXECUTION_COLUMNS,
        "Sixteen absolute results in execution order; no neighbouring execution deltas",
    ),
)

_REPRO_CASE_COLUMNS = (
    ("case_id", "Case"),
    ("selected_case_status", "Selected status"),
    ("selected_attempt", "Selected attempt"),
    ("excluded_attempts", "Excluded attempts"),
    ("planned_cell_count", "Planned cells"),
    ("observed_cell_count", "Observed cells"),
    ("declared_seed_policies", "Declared seed policy"),
    ("runtime_seed_policies", "Effective seed policy"),
    ("split_seeds", "Split seeds"),
    ("model_seeds", "Model seeds"),
    ("training_orchestration_seeds", "Orchestration seeds"),
    ("evaluation_statistics_seeds", "Evaluation seeds"),
    ("audit_status", "Status"),
)

_REPRO_CELL_COLUMNS = (
    ("case_id", "Case"),
    ("repeat", "Repeat"),
    ("fold", "Fold"),
    ("status", "Cell status"),
    ("selected_attempt", "Attempt"),
    ("declared_seed_policy", "Declared policy"),
    ("runtime_seed_policy", "Effective policy"),
    ("split_seed", "Split seed"),
    ("training_orchestration_seed", "Orchestration seed"),
    ("training_seed", "Training seed"),
    ("model_seed_roster", "Model/member seeds"),
    ("member_seed_semantics", "Member-seed semantics"),
    ("evaluation_statistics_seed", "Evaluation seed"),
    ("epoch_rng_seed_count", "Epoch RNG rows"),
    ("materialized_split_csv_sha256", "Split CSV SHA256"),
    ("split_identity_sha256", "Fold membership SHA256"),
    ("train_participant_count", "Train participants"),
    ("oof_participant_count", "OOF participants"),
    ("train_oof_overlap_count", "Train/OOF overlap"),
    ("audit_status", "Status"),
)

_REPRO_SPLIT_COLUMNS = (
    ("repeat", "Repeat"),
    ("fold", "Fold"),
    ("split_seed", "Split seed"),
    ("materialized_split_csv_sha256", "Split CSV SHA256"),
    ("declared_source_registry_json_file_sha256", "Declared authority JSON SHA256"),
    ("declared_source_registry_payload_sha256", "Declared authority payload SHA256"),
    ("train_participant_count", "Train participants"),
    ("oof_participant_count", "OOF participants"),
    ("train_oof_overlap_count", "Overlap"),
    ("case_ids", "Matching cases"),
    ("audit_status", "Status"),
)

_REPRO_ISSUE_COLUMNS = (
    ("severity", "Severity"),
    ("code", "Code"),
    ("case_id", "Case"),
    ("repeat", "Repeat"),
    ("fold", "Fold"),
    ("message", "Message"),
)


def _unavailable_reproducibility(message: str) -> ReproducibilityAudit:
    return ReproducibilityAudit(
        schema_version="ppg_frailty.reporting.reproducibility_audit.v1",
        status=NOT_VERIFIABLE,
        summary={
            "audit_status": NOT_VERIFIABLE,
            "scope": "audit_unavailable",
            "training_or_report_gate": False,
            "error_count": 0,
            "not_verifiable_count": 1,
        },
        case_rows=(),
        cell_rows=(),
        split_rows=(),
        issues=(
            {
                "severity": "not_verifiable",
                "code": "reproducibility_audit_unavailable",
                "case_id": None,
                "repeat": None,
                "fold": None,
                "message": message,
            },
        ),
    )


def _report_markdown(
    collected: CollectedStudy,
    analysis: StudyAnalysis,
    figures: Sequence[Mapping[str, Any]],
    reproducibility: ReproducibilityAudit | None = None,
) -> str:
    reproducibility = reproducibility or _unavailable_reproducibility(
        "seed/split evidence was not supplied to the report renderer"
    )
    study = _study_info(collected)
    manifest = collected.manifest
    execution = collected.plan.get("execution", {})
    axes = collected.plan.get("axes", ())
    catalog = collected.plan.get("catalog", {})
    search = collected.plan.get("search", {})
    is_catalog_sweep = study.get("kind") == "catalog_sweep"
    config_source_label = (
        f"Catalog: {catalog.get('path', 'N/A')} "
        f"(scope={catalog.get('scope', 'N/A')}, "
        f"balance={catalog.get('balance_line', 'N/A')})"
        if is_catalog_sweep and isinstance(catalog, Mapping)
        else f"Base pipeline config: {collected.plan.get('base_config', 'N/A')}"
    )
    lines = [
        f"# V2 study summary — {study.get('study_id', collected.root.name)}",
        "",
        "> This report is descriptive evidence for manual review. It does not "
        "automatically select a final use case or winner.",
        "",
        "## Scientific context",
        "",
        f"- Study kind: {study.get('kind', 'N/A')}",
        f"- Purpose: {study.get('purpose', 'N/A')}",
        f"- Position in use-case selection flow: {study.get('flow_position', 'N/A')}",
        f"- Decision role: {study.get('decision_role', 'N/A')}",
        f"- Thesis sections: {_fmt(study.get('thesis_sections', []))}",
        f"- {config_source_label}",
        f"- Reference case: {manifest.get('reference_case_id') or 'N/A'}",
        "",
        "## Run controls and completeness",
        "",
        f"- Repeats requested: {_fmt(execution.get('repeats', []))}",
        f"- Folds requested: {_fmt(execution.get('folds', []))}",
        f"- Case-level jobs requested: {execution.get('jobs', 'N/A')}",
        f"- Effective jobs: {manifest.get('effective_jobs', 'N/A')}",
        f"- Planned / passed / failed / not-run cases: "
        f"{manifest.get('planned_case_count', 'N/A')} / "
        f"{manifest.get('passed_case_count', 'N/A')} / "
        f"{manifest.get('failed_case_count', 'N/A')} / "
        f"{manifest.get('not_run_case_count', 'N/A')}",
        f"- Planned / reported / passed / failed / not-run cells: "
        f"{manifest.get('planned_cell_count', 'N/A')} / "
        f"{manifest.get('reported_cell_count', 'N/A')} / "
        f"{manifest.get('passed_cell_count', 'N/A')} / "
        f"{manifest.get('failed_cell_count', 'N/A')} / "
        f"{manifest.get('not_run_cell_count', 'N/A')}",
        f"- Resume-skipped passed cases: {manifest.get('resumed_case_count', 0)}",
        "",
        "## Seed and data-split reproducibility",
        "",
        f"- Audit status: **{reproducibility.status}**",
        f"- Scope: {reproducibility.summary.get('scope', 'N/A')}",
        f"- Planned / observed selected cells: "
        f"{reproducibility.summary.get('planned_cell_count', 'N/A')} / "
        f"{reproducibility.summary.get('observed_cell_count', 'N/A')}",
        f"- Split seeds by repeat: "
        f"{_fmt(reproducibility.summary.get('split_seed_by_repeat'))}",
        f"- Errors / not-verifiable items: "
        f"{reproducibility.summary.get('error_count', 0)} / "
        f"{reproducibility.summary.get('not_verifiable_count', 0)}",
        "- This is report-only evidence; it never gates training or report generation.",
        "",
    ]
    lines.extend(
        _markdown_table(
            reproducibility.case_rows,
            _REPRO_CASE_COLUMNS,
        )
    )
    lines.extend(
        [
            "<details><summary>Per-cell seed and split evidence</summary>",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            reproducibility.cell_rows,
            _REPRO_CELL_COLUMNS,
        )
    )
    lines.extend(["</details>", "", "### Frozen split roster", ""])
    lines.extend(
        _markdown_table(
            reproducibility.split_rows,
            _REPRO_SPLIT_COLUMNS,
        )
    )
    if reproducibility.issues:
        lines.extend(["### Reproducibility audit issues", ""])
        lines.extend(
            _markdown_table(
                reproducibility.issues,
                _REPRO_ISSUE_COLUMNS,
            )
        )
    lines.extend(["## Varied and controlled parameters", ""])
    if axes:
        lines.extend(
            [
                f"- {axis.get('path')}: values={_fmt(axis.get('values'))}; "
                f"reference={_fmt(axis.get('reference'))}"
                for axis in axes
                if isinstance(axis, Mapping)
            ]
        )
    elif is_catalog_sweep and isinstance(search, Mapping):
        lines.extend(
            [
                "- Explicit deterministic sparse catalog profiles; this is a "
                "screening comparison, not a single-factor causal ablation.",
                f"- Search method: {search.get('method', 'N/A')}",
                f"- Runtime parameter sampling: {search.get('runtime_sampling', 'N/A')}",
                f"- Profile-design seed: {search.get('selection_seed', 'N/A')}",
                f"- Interpretation: {search.get('interpretation', 'N/A')}",
            ]
        )
    else:
        lines.append("- No scientific axis: this is a single-config run.")
    lines.extend(
        [
            "",
            "The complete resolved varied/controlled tables are "
            "[varied_parameters.csv](tables/varied_parameters.csv) and "
            "[controlled_parameters.csv](tables/controlled_parameters.csv). "
            "Execution controls such as jobs are not scientific grid variables.",
            "",
            "<details><summary>Complete controlled-parameter list "
            f"({len(collected.controlled_parameters)} rows)</summary>",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            collected.controlled_parameters,
            (
                ("parameter_path", "Controlled parameter"),
                ("value", "Resolved value"),
            ),
        )
    )
    lines.extend(
        [
            "</details>",
            "",
            "## Predictive ranking",
            "",
            "Primary ranking is by participant-level, repeat-recomputed "
            "abstention-aware balanced accuracy, then participant coverage and "
            "abstention-aware Macro-F1. Conditional retained-only metrics remain "
            "visible but never lead the ranking; deployment measurements do not "
            "filter this table.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.predictive_leaderboard,
            (
                ("predictive_rank", "Rank"),
                ("case_id", "Case"),
                (
                    "participant_mean_abstention_aware_balanced_accuracy",
                    "Abstention-aware BA",
                ),
                (
                    "participant_mean_abstention_aware_macro_precision",
                    "Abstention-aware precision",
                ),
                (
                    "participant_mean_abstention_aware_macro_recall",
                    "Abstention-aware recall",
                ),
                (
                    "participant_mean_abstention_aware_macro_f1",
                    "Abstention-aware Macro-F1",
                ),
                ("participant_mean_coverage_rate", "Participant coverage"),
                ("abstention_count", "Abstentions"),
                ("abstention_counts_by_class", "Abstentions by class"),
                ("participant_mean_balanced_accuracy", "Conditional BA"),
                ("participant_mean_macro_f1", "Conditional Macro-F1"),
                (
                    "repeat_abstention_aware_balanced_accuracy_ci95_low",
                    "Aware BA CI95 low",
                ),
                (
                    "repeat_abstention_aware_balanced_accuracy_ci95_high",
                    "Aware BA CI95 high",
                ),
                (
                    "repeat_abstention_aware_macro_f1_ci95_low",
                    "Aware Macro-F1 CI95 low",
                ),
                (
                    "repeat_abstention_aware_macro_f1_ci95_high",
                    "Aware Macro-F1 CI95 high",
                ),
                ("balanced_accuracy_lcb95", "Conditional BA LCB95"),
                ("macro_f1_lcb95", "Conditional Macro-F1 LCB95"),
                (
                    "worst_fold_abstention_aware_balanced_accuracy",
                    "Aware worst-fold BA",
                ),
                ("worst_fold_balanced_accuracy", "Conditional worst-fold BA"),
                ("worst_class_recall", "Worst recall"),
                ("worst_class_f1", "Worst F1"),
                ("metric_source", "Source"),
                ("frailty_classification_evaluation_scope", "Frailty endpoint"),
                (
                    "auxiliary_motion_evidence_valid_outer_oof",
                    "Motion auxiliary outer-OOF",
                ),
                ("ranking_interpretation", "Interpretation"),
            ),
        )
    )
    if _is_stage3_centered_star(collected.plan):
        for name, title, notice, columns, _description in _STAGE3_STAR_TABLES:
            lines.extend([f"## {title}", "", notice, ""])
            lines.extend(_markdown_table(getattr(analysis, name), columns))
    elif isinstance(collected.plan.get("legacy_bridge"), Mapping):
        lines.extend(
            [
                "## Legacy/V2 bridge report A — numeric adjacent ablations (L0→L7)",
                "",
                "This is the causal-interpretation table: L0 is the baseline and "
                "the next seven rows are only the predefined adjacent profile "
                "contrasts L0→L1 through L6→L7. Deltas are never taken from run "
                "order.",
                "",
            ]
        )
        lines.extend(
            _markdown_table(
                analysis.legacy_bridge_numeric_ablation_report,
                _LEGACY_BRIDGE_NUMERIC_COLUMNS,
            )
        )
        lines.extend(
            [
                "## Legacy/V2 bridge report B — CompactCNN execution order",
                "",
                "This table lists absolute W/A/B metrics in the requested "
                "L7→L5→L6→L4→L3→L2→L1→L0 run order. It deliberately has no "
                "execution-order delta: L7→L5 and every other neighbouring run "
                "pair are scheduling transitions, not causal ablations.",
                "",
            ]
        )
        lines.extend(
            _markdown_table(
                analysis.legacy_bridge_execution_order_report,
                _LEGACY_BRIDGE_EXECUTION_COLUMNS,
            )
        )
    lines.extend(
        [
            "## Aggregation sensitivity from the same file-level OOF",
            "",
            "The declared-source row reproduces the aggregation used by the fitted "
            "model and, when eligible, the primary leaderboard. The other row "
            "reaggregates the same "
            "held-out file probabilities post hoc. It is not a separately retrained "
            "Line A/Line B experiment and is not selection evidence.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.aggregation_line_comparison,
            (
                ("case_id", "Case"),
                ("balance_line", "Aggregation view"),
                ("view_role", "Role"),
                ("participant_mean_balanced_accuracy", "Mean BA"),
                ("participant_mean_macro_f1", "Mean Macro-F1"),
                (
                    "line_a_minus_line_b_balanced_accuracy",
                    "Line A − Line B BA",
                ),
                ("line_a_minus_line_b_macro_f1", "Line A − Line B Macro-F1"),
                ("worst_class_recall", "Worst recall"),
                ("worst_class_f1", "Worst F1"),
                ("expected_calibration_error", "ECE"),
                ("repeat_count", "Repeats"),
                ("participant_oof_prediction_count", "Retained participant OOF n"),
                ("participant_oof_total_count", "All participant units n"),
                ("dropped_participant_oof_count", "Dropped participant units n"),
                ("file_oof_prediction_count", "All file OOF n"),
                ("dropped_file_oof_prediction_count", "Dropped files n"),
                ("source_replay_validation", "Source replay"),
                ("primary_ranking_eligible", "Primary ranking eligible"),
            ),
        )
    )
    lines.extend(
        [
            "## Parallel window/file/role-balanced participant views",
            "",
            "All three rows reuse the same fitted held-out OOF probabilities; they "
            "are not three training runs. `window_balanced_to_participant` gives "
            "every retained window equal report weight, Line A gives every file "
            "equal weight after window→file, and Line B gives every canonical role "
            "family equal weight after window→file→role. Only the declared training "
            "aggregation may support the primary leaderboard; the other views are "
            "post-hoc sensitivity plots.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.aggregation_view_comparison,
            (
                ("case_id", "Case"),
                ("aggregation_view", "Aggregation view"),
                ("evidence_role", "Evidence role"),
                ("participant_mean_balanced_accuracy", "Mean BA"),
                ("participant_mean_macro_f1", "Mean Macro-F1"),
                ("worst_class_recall", "Worst recall"),
                ("worst_class_f1", "Worst F1"),
                ("repeat_count", "Repeats"),
                ("participant_oof_prediction_count", "Participant OOF n"),
                ("primary_ranking_eligible", "Primary ranking eligible"),
            ),
        )
    )
    lines.extend(
        [
            "<details><summary>Hierarchy coverage: B/R1–R4 window/file views and "
            "B/R role-balanced view</summary>",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.aggregation_hierarchy_coverage,
            (
                ("case_id", "Case"),
                ("repeat", "Repeat"),
                ("aggregation_level", "Level"),
                ("aggregation_view", "View"),
                ("group_label", "Group"),
                ("oof_unit_count", "OOF units"),
                ("retained_oof_unit_count", "Retained units"),
                ("participant_count", "Participants"),
            ),
        )
    )
    lines.extend(["</details>", ""])
    lines.extend(
        [
            "## Worst-class F1 stability review",
            "",
            "This secondary view reorders complete cases by abstention-aware "
            "worst-class F1, then abstention-aware repeat variability. Conditional "
            "retained-only values remain visible for comparison.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.worst_class_f1_stability,
            (
                ("worst_class_f1_stability_rank", "Stability rank"),
                ("predictive_rank", "Aware-BA rank"),
                ("case_id", "Case"),
                ("abstention_aware_worst_class_f1", "Aware worst F1"),
                ("abstention_aware_worst_class_recall", "Aware worst recall"),
                (
                    "participant_mean_abstention_aware_balanced_accuracy",
                    "Aware mean BA",
                ),
                (
                    "repeat_abstention_aware_balanced_accuracy_population_sd",
                    "Aware repeat BA SD",
                ),
                ("worst_class_f1", "Worst F1"),
                ("worst_class_recall", "Worst recall"),
                ("participant_mean_balanced_accuracy", "Conditional mean BA"),
            ),
        )
    )
    lines.extend(["## Incomplete cases excluded from ranking", ""])
    lines.extend(
        _markdown_table(
            analysis.incomplete_cases,
            (
                ("case_id", "Case"),
                ("status", "Status"),
                ("incompleteness_reasons", "Reasons"),
                ("repeat_count", "Reported repeats"),
                ("expected_repeat_count", "Expected repeats"),
                ("passed_fold_cell_count", "Passed cells"),
                ("expected_fold_cell_count", "Expected cells"),
            ),
        )
    )
    lines.extend(
        [
            "## Deployment measurements (separate from predictive ranking)",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.deployment_table,
            (
                ("case_id", "Case"),
                ("parameter_count", "Parameters"),
                ("inference_cost", "Inference cost"),
                ("deployment_readiness", "Status"),
                ("reported_exclusion_reason", "Reported note"),
            ),
        )
    )
    lines.extend(
        [
            "## Route × role coverage and feature availability",
            "",
            "This table separates direct and processed rate paths, retained coverage, "
            "unavailable predictors, and reducer failures for each role/route state.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.route_role_coverage,
            (
                ("case_id", "Case"),
                ("role", "Role"),
                ("quality_tier", "Quality tier"),
                ("motion_state", "Motion"),
                ("route_state", "Route state"),
                ("signal_route", "Signal route"),
                ("retained_coverage", "Retained coverage"),
                ("abstention_rate", "Abstention"),
                ("abstention_reasons", "Abstention reasons"),
                ("direct_rate_record_count", "Direct"),
                ("processed_rate_record_count", "Processed"),
                ("unavailable_predictor_rate", "Unavailable predictors"),
                ("denoiser_attempt_count", "Denoiser attempts"),
                ("denoiser_success_count", "Denoiser successes"),
                ("reducer_failure_count", "Reducer failures"),
            ),
        )
    )
    lines.extend(
        [
            "## SQI state, score, and coverage provenance by each route",
            "",
            "Direct and post-denoiser coverage are reported separately so the "
            "configured minimum-coverage decision remains auditable.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.route_role_coverage,
            (
                ("case_id", "Case"),
                ("role", "Role"),
                ("quality_tier", "Tier"),
                ("direct_q_rate_states", "Direct Q_rate state"),
                ("mean_direct_q_rate_score", "Mean direct Q_rate"),
                ("mean_direct_q_rate_coverage", "Direct Q_rate coverage"),
                ("direct_q_morph_states", "Direct Q_morph state"),
                ("mean_direct_q_morph_score", "Mean direct Q_morph"),
                ("mean_direct_q_morph_coverage", "Direct Q_morph coverage"),
                ("post_q_rate_states", "Post Q_rate state"),
                ("mean_post_q_rate_score", "Mean post Q_rate"),
                ("mean_post_q_rate_coverage", "Post Q_rate coverage"),
            ),
        )
    )
    lines.extend(
        [
            "## Frozen motion evidence used by each route",
            "",
            "Frailty29 reuse is in-sample auxiliary motion-preprocessing evidence, "
            "not valid outer-OOF motion-detector evidence. The downstream frailty "
            "classification outcome is still evaluated on each outer held-out fold.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.route_role_coverage,
            (
                ("case_id", "Case"),
                ("role", "Role"),
                ("quality_tier", "Tier"),
                ("motion_state", "Motion"),
                ("mean_motion_record_probability", "Mean p(motion)"),
                ("mean_motion_threshold", "Threshold"),
                ("mean_motion_window_count", "Mean windows"),
                ("motion_evidence_sha256", "Evidence SHA-256"),
                ("motion_model_artifact_sha256", "Model SHA-256"),
                ("motion_training_scope", "Training scope"),
                ("motion_frailty29_relation", "Frailty29 relation"),
                (
                    "auxiliary_motion_evidence_valid_outer_oof",
                    "Valid outer-OOF motion evidence",
                ),
                ("denoiser_ids", "Denoiser"),
                ("denoiser_statuses", "Denoiser status"),
            ),
        )
    )
    lines.extend(
        [
            "## Quality-component distributions by route and role",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.quality_distributions,
            (
                ("case_id", "Case"),
                ("role", "Role"),
                ("route_state", "Route state"),
                ("component", "Component"),
                ("valid_count", "Valid n"),
                ("unavailable_rate", "Unavailable"),
                ("mean", "Mean"),
                ("population_sd", "SD"),
                ("minimum", "Min"),
                ("maximum", "Max"),
            ),
        )
    )
    failed = [
        row
        for row in collected.case_records
        if str(row.get("status")) not in {"passed"}
    ]
    lines.extend(["## Failed or incomplete cases", ""])
    lines.extend(
        _markdown_table(
            failed,
            (
                ("case_id", "Case"),
                ("status", "Status"),
                ("error_type", "Error type"),
                ("error", "Message"),
            ),
        )
    )
    lines.extend(["## Figure status", ""])
    lines.extend(
        _markdown_table(
            figures,
            (
                ("figure", "Figure"),
                ("status", "Status"),
                ("path", "Path"),
                ("reason", "Reason"),
            ),
        )
    )
    lines.extend(["## Limitations and N/A items", ""])
    if analysis.notes:
        lines.extend(f"- {note}" for note in analysis.notes)
    else:
        lines.append("- No collection limitation was recorded.")
    lines.extend(
        [
            "",
            "## Output navigation",
            "",
            "- [outputs_index.json](outputs_index.json): machine-readable inventory",
            "- [study_summary.json](study_summary.json): report context and tables",
            "- [tables/reproducibility_summary.csv](tables/reproducibility_summary.csv)",
            "- [tables/reproducibility_cases.csv](tables/reproducibility_cases.csv)",
            "- [tables/reproducibility_cells.csv](tables/reproducibility_cells.csv)",
            "- [tables/reproducibility_splits.csv](tables/reproducibility_splits.csv)",
            "- [tables/reproducibility_issues.csv](tables/reproducibility_issues.csv)",
            "- [tables/predictive_leaderboard.csv](tables/predictive_leaderboard.csv)",
            "- [tables/aggregation_line_comparison.csv](tables/aggregation_line_comparison.csv)",
            "- [tables/aggregation_line_repeat_metrics.csv](tables/aggregation_line_repeat_metrics.csv)",
            "- [tables/aggregation_line_per_class_metrics.csv](tables/aggregation_line_per_class_metrics.csv)",
            "- [tables/aggregation_view_comparison.csv](tables/aggregation_view_comparison.csv)",
            "- [tables/aggregation_view_confusion_matrices.csv](tables/aggregation_view_confusion_matrices.csv)",
            "- [tables/aggregation_hierarchy_coverage.csv](tables/aggregation_hierarchy_coverage.csv)",
            "- [tables/metric_distribution_summary.csv](tables/metric_distribution_summary.csv)",
            "- [tables/worst_class_f1_stability.csv](tables/worst_class_f1_stability.csv)",
            "- [tables/incomplete_cases.csv](tables/incomplete_cases.csv)",
            "- [tables/confusion_counts.csv](tables/confusion_counts.csv)",
            "- [tables/confusion_row_normalized.csv](tables/confusion_row_normalized.csv)",
            "- [tables/top_confusion_matrices/](tables/top_confusion_matrices/): top-case count and row-normalized CSVs",
            "- [tables/deployment_measurements.csv](tables/deployment_measurements.csv)",
            "- [figures/plot_status.json](figures/plot_status.json)",
            "",
        ]
    )
    if _is_stage3_centered_star(collected.plan):
        lines.extend(
            [f"- [tables/{name}.csv](tables/{name}.csv)" for name, *_ in _STAGE3_STAR_TABLES]
            + [""]
        )
    elif isinstance(collected.plan.get("legacy_bridge"), Mapping):
        lines.extend(
            [
                "- [tables/legacy_bridge_numeric_ablation_report.csv](tables/legacy_bridge_numeric_ablation_report.csv)",
                "- [tables/legacy_bridge_execution_order_report.csv](tables/legacy_bridge_execution_order_report.csv)",
                "",
            ]
        )
    return "\n".join(lines)


def _html_table(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]
) -> str:
    if not rows:
        return "<p><em>N/A — no rows were available.</em></p>"
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(_fmt(row.get(field)))}</td>"
                for field, _ in columns
            )
            + "</tr>"
        )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _report_html(
    collected: CollectedStudy,
    analysis: StudyAnalysis,
    figures: Sequence[Mapping[str, Any]],
    reproducibility: ReproducibilityAudit | None = None,
) -> str:
    reproducibility = reproducibility or _unavailable_reproducibility(
        "seed/split evidence was not supplied to the report renderer"
    )
    study = _study_info(collected)
    generated = [
        row
        for row in figures
        if str(row.get("path", "")).lower().endswith(".png")
    ]
    figure_html = "".join(
        f"<figure><img src='{html.escape(str(row['path']))}' alt='"
        f"{html.escape(str(row['figure']))}'><figcaption>"
        f"{html.escape(str(row['figure']))}"
        f"{(' — N/A: ' + html.escape(str(row.get('reason', '')))) if row.get('status') == 'N/A' else ''}"
        "</figcaption></figure>"
        for row in generated
    )
    limitations = "".join(f"<li>{html.escape(note)}</li>" for note in analysis.notes)
    bridge_html = ""
    if _is_stage3_centered_star(collected.plan):
        bridge_html = "".join(
            f"<h2>{html.escape(title)}</h2><p class='notice'>{html.escape(notice)}</p>"
            + _html_table(getattr(analysis, name), columns)
            for name, title, notice, columns, _description in _STAGE3_STAR_TABLES
        )
    elif isinstance(collected.plan.get("legacy_bridge"), Mapping):
        bridge_html = f"""
<h2>Legacy/V2 bridge report A — numeric adjacent ablations (L0→L7)</h2>
<p class="notice">This is the causal-interpretation table. L0 is the baseline;
the next seven rows are only L0→L1 through L6→L7. Deltas are never calculated
from execution order.</p>
{_html_table(
    analysis.legacy_bridge_numeric_ablation_report,
    _LEGACY_BRIDGE_NUMERIC_COLUMNS,
)}
<h2>Legacy/V2 bridge report B — CompactCNN execution order</h2>
<p class="notice">Absolute W/A/B metrics in
L7→L5→L6→L4→L3→L2→L1→L0 order. There is deliberately no execution-order
delta: L7→L5 and all other neighbouring runs are scheduling transitions, not
causal ablations.</p>
{_html_table(
    analysis.legacy_bridge_execution_order_report,
    _LEGACY_BRIDGE_EXECUTION_COLUMNS,
)}
"""
    reproducibility_issues = _html_table(
        reproducibility.issues,
        _REPRO_ISSUE_COLUMNS,
    ) if reproducibility.issues else "<p>No reproducibility issue recorded.</p>"
    reproducibility_html = f"""
<h2>Seed and data-split reproducibility — {html.escape(reproducibility.status)}</h2>
<p class="notice">Report-only evidence; this status never gates training or
report generation. Planned/observed selected cells:
{html.escape(_fmt(reproducibility.summary.get('planned_cell_count')))} /
{html.escape(_fmt(reproducibility.summary.get('observed_cell_count')))}.</p>
{_html_table(reproducibility.case_rows, _REPRO_CASE_COLUMNS)}
<details><summary>Per-cell seed and split evidence</summary>
{_html_table(reproducibility.cell_rows, _REPRO_CELL_COLUMNS)}</details>
<h3>Frozen split roster</h3>
{_html_table(reproducibility.split_rows, _REPRO_SPLIT_COLUMNS)}
<h3>Reproducibility audit issues</h3>{reproducibility_issues}
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>V2 study — {html.escape(str(study.get("study_id", collected.root.name)))}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1280px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{border:1px solid #ccc;padding:.4rem;text-align:left}}
th{{background:#f0f3f6}}img{{max-width:100%;height:auto}}figure{{margin:2rem 0}}
.notice{{padding:1rem;background:#fff5cc;border-left:4px solid #c49000}}
</style></head><body>
<h1>V2 study — {html.escape(str(study.get("study_id", collected.root.name)))}</h1>
<p class="notice">Descriptive manual-review report; no automatic winner is selected.</p>
<p><strong>Purpose:</strong> {html.escape(str(study.get("purpose", "N/A")))}</p>
<p><strong>Flow position:</strong> {html.escape(str(study.get("flow_position", "N/A")))}</p>
{reproducibility_html}
<h2>Predictive leaderboard</h2>
<p>Primary ranking uses participant-level, repeat-recomputed abstention-aware
balanced accuracy, then participant coverage and abstention-aware Macro-F1.
Conditional values use retained participants only and do not lead the ranking.</p>
{_html_table(analysis.predictive_leaderboard, (
    ("predictive_rank", "Rank"), ("case_id", "Case"),
    (
        "participant_mean_abstention_aware_balanced_accuracy",
        "Abstention-aware BA",
    ),
    (
        "participant_mean_abstention_aware_macro_precision",
        "Abstention-aware precision",
    ),
    (
        "participant_mean_abstention_aware_macro_recall",
        "Abstention-aware recall",
    ),
    (
        "participant_mean_abstention_aware_macro_f1",
        "Abstention-aware Macro-F1",
    ),
    ("participant_mean_coverage_rate", "Participant coverage"),
    ("abstention_count", "Abstentions"),
    ("abstention_counts_by_class", "Abstentions by class"),
    ("participant_mean_balanced_accuracy", "Conditional BA"),
    ("participant_mean_macro_f1", "Conditional Macro-F1"),
    ("repeat_abstention_aware_balanced_accuracy_ci95_low", "Aware BA CI95 low"),
    ("repeat_abstention_aware_balanced_accuracy_ci95_high", "Aware BA CI95 high"),
    ("repeat_abstention_aware_macro_f1_ci95_low", "Aware Macro-F1 CI95 low"),
    ("repeat_abstention_aware_macro_f1_ci95_high", "Aware Macro-F1 CI95 high"),
    ("balanced_accuracy_lcb95", "Conditional BA LCB95"),
    ("macro_f1_lcb95", "Conditional Macro-F1 LCB95"),
    ("worst_fold_abstention_aware_balanced_accuracy", "Aware worst-fold BA"),
    ("worst_fold_balanced_accuracy", "Conditional worst-fold BA"),
    ("worst_class_f1", "Worst F1"),
    ("frailty_classification_evaluation_scope", "Frailty endpoint"),
    ("auxiliary_motion_evidence_valid_outer_oof", "Motion auxiliary outer-OOF"),
    ("ranking_interpretation", "Interpretation"),
))}
{bridge_html}
<h2>Aggregation sensitivity from the same file-level OOF</h2>
<p class="notice">The declared-source row reproduces the aggregation used by
the fitted model and, when eligible, the primary leaderboard. The other row
reaggregates the same held-out file probabilities post hoc. It is not a separately retrained
Line A/Line B experiment and is not selection evidence.</p>
{_html_table(analysis.aggregation_line_comparison, (
    ("case_id", "Case"), ("balance_line", "Aggregation view"),
    ("view_role", "Role"),
    ("participant_mean_balanced_accuracy", "Mean BA"),
    ("participant_mean_macro_f1", "Mean Macro-F1"),
    ("line_a_minus_line_b_balanced_accuracy", "Line A - Line B BA"),
    ("line_a_minus_line_b_macro_f1", "Line A - Line B Macro-F1"),
    ("worst_class_recall", "Worst recall"),
    ("worst_class_f1", "Worst F1"),
    ("expected_calibration_error", "ECE"),
    ("repeat_count", "Repeats"),
    ("participant_oof_prediction_count", "Retained participant OOF n"),
    ("participant_oof_total_count", "All participant units n"),
    ("dropped_participant_oof_count", "Dropped participant units n"),
    ("file_oof_prediction_count", "All file OOF n"),
    ("dropped_file_oof_prediction_count", "Dropped files n"),
    ("source_replay_validation", "Source replay"),
    ("primary_ranking_eligible", "Primary ranking eligible"),
))}
<h2>Parallel window/file/role-balanced participant views</h2>
<p class="notice">These are three report views of the same fitted held-out OOF,
not three training runs. Equal-window and non-source Line A/Line B views are
post-hoc sensitivity only. Only the declared training aggregation may support
the primary leaderboard.</p>
{_html_table(analysis.aggregation_view_comparison, (
    ("case_id", "Case"), ("aggregation_view", "Aggregation view"),
    ("evidence_role", "Evidence role"),
    ("participant_mean_balanced_accuracy", "Mean BA"),
    ("participant_mean_macro_f1", "Mean Macro-F1"),
    ("worst_class_recall", "Worst recall"),
    ("worst_class_f1", "Worst F1"),
    ("repeat_count", "Repeats"),
    ("participant_oof_prediction_count", "Participant OOF n"),
    ("primary_ranking_eligible", "Primary ranking eligible"),
))}
<h3>Hierarchy coverage (B/R1–R4 and B/R)</h3>
{_html_table(analysis.aggregation_hierarchy_coverage, (
    ("case_id", "Case"), ("repeat", "Repeat"),
    ("aggregation_level", "Level"), ("aggregation_view", "View"),
    ("group_label", "Group"), ("oof_unit_count", "OOF units"),
    ("retained_oof_unit_count", "Retained units"),
    ("participant_count", "Participants"),
))}
<h2>Worst-class F1 stability review</h2>
<p>This secondary ordering uses abstention-aware worst-class F1 and
abstention-aware repeat variability; conditional retained-only values are
shown only for comparison.</p>
{_html_table(analysis.worst_class_f1_stability, (
    ("worst_class_f1_stability_rank", "Stability rank"),
    ("predictive_rank", "Aware-BA rank"), ("case_id", "Case"),
    ("abstention_aware_worst_class_f1", "Aware worst F1"),
    ("abstention_aware_worst_class_recall", "Aware worst recall"),
    ("participant_mean_abstention_aware_balanced_accuracy", "Aware mean BA"),
    ("repeat_abstention_aware_balanced_accuracy_population_sd", "Aware repeat BA SD"),
    ("worst_class_f1", "Worst F1"),
    ("participant_mean_balanced_accuracy", "Conditional mean BA"),
))}
<h2>Incomplete cases excluded from ranking</h2>
{_html_table(analysis.incomplete_cases, (
    ("case_id", "Case"), ("status", "Status"),
    ("incompleteness_reasons", "Reasons"),
    ("repeat_count", "Reported repeats"),
    ("expected_repeat_count", "Expected repeats"),
    ("passed_fold_cell_count", "Passed cells"),
    ("expected_fold_cell_count", "Expected cells"),
))}
<h2>Deployment measurements (not a predictive filter)</h2>
{_html_table(analysis.deployment_table, (
    ("case_id", "Case"), ("parameter_count", "Parameters"),
    ("inference_cost", "Inference cost"), ("deployment_readiness", "Status"),
))}
<h2>Route × role coverage and feature availability</h2>
{_html_table(analysis.route_role_coverage, (
    ("case_id", "Case"), ("role", "Role"),
    ("quality_tier", "Quality tier"), ("motion_state", "Motion"),
    ("route_state", "Route state"),
    ("signal_route", "Signal route"), ("retained_coverage", "Retained coverage"),
    ("abstention_rate", "Abstention"),
    ("abstention_reasons", "Abstention reasons"),
    ("direct_rate_record_count", "Direct"), ("processed_rate_record_count", "Processed"),
    ("unavailable_predictor_rate", "Unavailable predictors"),
    ("denoiser_attempt_count", "Denoiser attempts"),
    ("denoiser_success_count", "Denoiser successes"),
    ("reducer_failure_count", "Reducer failures"),
))}
<h2>SQI state, score, and coverage provenance by each route</h2>
<p>Direct and post-denoiser coverage are separate so the configured
minimum-coverage decision remains auditable.</p>
{_html_table(analysis.route_role_coverage, (
    ("case_id", "Case"), ("role", "Role"), ("quality_tier", "Tier"),
    ("direct_q_rate_states", "Direct Q_rate state"),
    ("mean_direct_q_rate_score", "Mean direct Q_rate"),
    ("mean_direct_q_rate_coverage", "Direct Q_rate coverage"),
    ("direct_q_morph_states", "Direct Q_morph state"),
    ("mean_direct_q_morph_score", "Mean direct Q_morph"),
    ("mean_direct_q_morph_coverage", "Direct Q_morph coverage"),
    ("post_q_rate_states", "Post Q_rate state"),
    ("mean_post_q_rate_score", "Mean post Q_rate"),
    ("mean_post_q_rate_coverage", "Post Q_rate coverage"),
))}
<h2>Frozen motion evidence used by each route</h2>
<p class="notice">Frailty29 reuse is in-sample auxiliary motion-preprocessing
evidence, not valid outer-OOF motion-detector evidence. Downstream frailty
classification is still evaluated on each outer held-out fold.</p>
{_html_table(analysis.route_role_coverage, (
    ("case_id", "Case"), ("role", "Role"), ("quality_tier", "Tier"),
    ("motion_state", "Motion"),
    ("mean_motion_record_probability", "Mean p(motion)"),
    ("mean_motion_threshold", "Threshold"),
    ("mean_motion_window_count", "Mean windows"),
    ("motion_evidence_sha256", "Evidence SHA-256"),
    ("motion_model_artifact_sha256", "Model SHA-256"),
    ("motion_training_scope", "Training scope"),
    ("motion_frailty29_relation", "Frailty29 relation"),
    ("auxiliary_motion_evidence_valid_outer_oof", "Valid outer-OOF motion evidence"),
    ("denoiser_ids", "Denoiser"), ("denoiser_statuses", "Denoiser status"),
))}
<h2>Quality-component distributions</h2>
{_html_table(analysis.quality_distributions, (
    ("case_id", "Case"), ("role", "Role"), ("route_state", "Route state"),
    ("component", "Component"), ("valid_count", "Valid n"),
    ("unavailable_rate", "Unavailable"), ("mean", "Mean"),
    ("population_sd", "SD"), ("minimum", "Min"), ("maximum", "Max"),
))}
<h2>Figures</h2>{figure_html or "<p><em>N/A — no generated figures.</em></p>"}
<h3>Figure status, including explicit N/A reasons</h3>
{_html_table(figures, (
    ("figure", "Figure"), ("status", "Status"),
    ("path", "Path"), ("reason", "Reason"),
))}
<h2>Limitations</h2><ul>{limitations or "<li>None recorded.</li>"}</ul>
<p>See <a href="outputs_index.json">outputs_index.json</a> for every artifact.</p>
</body></html>
"""


@dataclass(frozen=True)
class ReportResult:
    study_directory: Path
    summary_markdown: Path
    summary_html: Path | None
    output_index: Path
    table_count: int
    generated_figure_count: int
    na_figure_count: int

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def _index_entry(
    root: Path,
    path: Path,
    *,
    artifact_type: str,
    description: str,
    status: str = "available",
) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "type": artifact_type,
        "status": status,
        "description": description,
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else None
        ),
    }


def _safe_filename(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return cleaned[:100] or "case"


def _write_top_confusion_matrix_files(
    root: Path,
    tables: Path,
    analysis: StudyAnalysis,
) -> tuple[list[dict[str, Any]], int]:
    """Write top-case count and row-normalized matrices as standalone CSVs."""

    target = tables / "top_confusion_matrices"
    target.mkdir(exist_ok=True)
    for old in target.glob("*.csv"):
        if old.is_file() or old.is_symlink():
            old.unlink()
    matrices = {
        str(row.get("case_id")): row for row in analysis.confusion_matrices
    }
    entries: list[dict[str, Any]] = []
    written = 0
    for ranked in analysis.predictive_leaderboard:
        case_id = str(ranked["case_id"])
        matrix_row = matrices.get(case_id)
        if matrix_row is None:
            continue
        order = list(matrix_row.get("class_order", ()))
        matrix = matrix_row.get("confusion_matrix")
        if not isinstance(matrix, (list, tuple)) or len(matrix) != len(order):
            continue
        try:
            numeric = [[float(value) for value in row] for row in matrix]
        except (TypeError, ValueError):
            continue
        if any(len(row) != len(order) for row in numeric):
            continue
        rank = int(ranked["predictive_rank"])
        stem = f"rank_{rank:02d}_{_safe_filename(case_id)}"
        count_rows = [
            {
                "true_class": order[row_index],
                **{
                    f"predicted_{label}": numeric[row_index][column_index]
                    for column_index, label in enumerate(order)
                },
            }
            for row_index in range(len(order))
        ]
        normalized_rows: list[Mapping[str, Any]] = []
        for row_index, row in enumerate(numeric):
            total = sum(row)
            normalized_rows.append(
                {
                    "true_class": order[row_index],
                    **{
                        f"predicted_{label}": (
                            row[column_index] / total if total > 0.0 else None
                        )
                        for column_index, label in enumerate(order)
                    },
                }
            )
        outputs = (
            (
                target / f"{stem}_counts.csv",
                count_rows,
                f"Top-rank {rank} confusion counts for {case_id}",
            ),
            (
                target / f"{stem}_row_normalized.csv",
                normalized_rows,
                f"Top-rank {rank} row-normalized confusion matrix for {case_id}",
            ),
        )
        for path, rows, description in outputs:
            _write_csv(path, rows)
            entries.append(
                _index_entry(
                    root,
                    path,
                    artifact_type="table_csv",
                    description=description,
                )
            )
            written += 1
    return entries, written


def _artifact_type(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if relative == Path("study_plan.yaml"):
        return "study_plan"
    if relative == Path("study_manifest.json"):
        return "study_manifest"
    if relative == Path("study_run_result.json"):
        return "study_run_result"
    if relative == Path("progress_events.jsonl"):
        return "progress_log"
    if relative.parts and relative.parts[0] == "resolved_configs":
        return "resolved_config"
    if relative.parts and relative.parts[0] in {
        "cases",
        "raw",
        "fusion",
        "feature_vector",
        "feature_matrix",
    }:
        return "case_artifact"
    if relative.parts and relative.parts[0] == "tables":
        return "report_table"
    if relative.parts and relative.parts[0] == "figures":
        return "report_figure"
    if relative.name.startswith("STUDY_SUMMARY") or relative.name == "study_summary.json":
        return "study_summary"
    return "study_artifact"


def _complete_inventory(
    root: Path,
    generated_entries: Sequence[Mapping[str, Any]],
    *,
    output_index: Path,
) -> list[dict[str, Any]]:
    """Index every regular study artifact, not only generated report files."""

    generated = {
        str(row.get("path")): dict(row)
        for row in generated_entries
        if row.get("path")
    }
    inventory: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        if path == output_index or not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        known = generated.get(relative, {})
        inventory.append(
            {
                "path": relative,
                "type": known.get("type", _artifact_type(path, root)),
                "status": known.get("status", "available"),
                "description": known.get(
                    "description", "Study input, execution, case, or report artifact"
                ),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    inventory.append(
        {
            "path": output_index.name,
            "type": "output_index",
            "status": "available",
            "description": "Machine-readable complete inventory of this study folder",
            "bytes": None,
            "sha256": None,
            "self_hash_policy": "omitted_to_avoid_recursive_self_hash",
        }
    )
    return inventory


def generate_study_report(
    study_directory: str | Path,
    *,
    collected: CollectedStudy | None = None,
) -> ReportResult:
    """Collect, analyze, and report one existing study directory."""

    root = Path(study_directory).resolve()
    bundle = collected or collect_study(root)
    analysis = analyze_study(bundle)
    try:
        reproducibility = audit_study_reproducibility(bundle)
    except Exception as error:  # noqa: BLE001 - preserve the primary report.
        reproducibility = _unavailable_reproducibility(
            f"{type(error).__name__}: {error}"
        )
    tables = root / "tables"
    figures_dir = root / "figures"
    tables.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    table_payloads: tuple[tuple[str, Sequence[Mapping[str, Any]], str], ...] = (
        ("case_summary", analysis.case_summary, "One descriptive row per case"),
        (
            "metric_distribution_summary",
            analysis.metric_distribution_summary,
            "Repeat mean/SD/t-CI95/min/max by case and predictive metric",
        ),
        ("varied_parameters", bundle.varied_parameters, "Declared variables and resolved case values"),
        ("controlled_parameters", bundle.controlled_parameters, "Complete non-variable resolved parameter list"),
        ("predictive_leaderboard", analysis.predictive_leaderboard, "BA-ranked manual review table"),
        (
            "aggregation_line_comparison",
            analysis.aggregation_line_comparison,
            "Declared source aggregation plus non-selection post-hoc sensitivity from the same file OOF",
        ),
        (
            "aggregation_line_repeat_metrics",
            analysis.aggregation_line_repeat_metrics,
            "Per-repeat Line A/Line B metrics reaggregated from the same file OOF",
        ),
        (
            "aggregation_line_per_class_metrics",
            analysis.aggregation_line_per_class_metrics,
            "Per-repeat per-class Line A/Line B metrics reaggregated from the same file OOF",
        ),
        (
            "aggregation_view_comparison",
            analysis.aggregation_view_comparison,
            "Window/file/role-balanced participant views from the same fitted OOF; only the declared source is training evidence",
        ),
        (
            "aggregation_view_repeat_metrics",
            analysis.aggregation_view_repeat_metrics,
            "Per-repeat window/file/role-balanced participant metrics from the same fitted OOF",
        ),
        (
            "aggregation_view_per_class_metrics",
            analysis.aggregation_view_per_class_metrics,
            "All-class metrics for the three same-OOF aggregation report views",
        ),
        (
            "aggregation_view_confusion_matrices",
            analysis.aggregation_view_confusion_matrices,
            "Participant confusion matrices for each same-OOF aggregation report view",
        ),
        (
            "aggregation_hierarchy_coverage",
            analysis.aggregation_hierarchy_coverage,
            "Window/file B/R1-R4 and role B/R hierarchy populations",
        ),
        ("deployment_measurements", analysis.deployment_table, "Operational measurements, separate from ranking"),
        ("repeat_metrics", analysis.repeat_metrics, "Participant OOF or labeled cell fallback per repeat"),
        ("fold_metrics", analysis.fold_metrics, "Per repeat/fold cell metrics"),
        (
            "per_class_metrics",
            analysis.per_class_metrics,
            "Pooled participant OOF or labeled cell-fallback class metrics",
        ),
        (
            "confusion_matrices",
            analysis.confusion_matrices,
            "Pooled participant OOF or labeled cell-fallback confusion matrices",
        ),
        (
            "confusion_counts",
            analysis.confusion_counts,
            "Long-form pooled confusion counts",
        ),
        (
            "confusion_row_normalized",
            analysis.confusion_row_normalized,
            "Long-form row-normalized pooled confusion matrices",
        ),
        ("calibration_bins", analysis.calibration_bins, "Top-label participant OOF reliability bins"),
        ("paired_deltas", analysis.paired_deltas, "Repeat-paired deltas versus declared reference"),
        ("coverage", analysis.coverage, "Coverage and quality diagnostic counts"),
        (
            "route_role_coverage",
            analysis.route_role_coverage,
            "Tier, motion provenance, abstention, retained/direct/processed, and denoiser summaries by route and role",
        ),
        (
            "quality_distributions",
            analysis.quality_distributions,
            "Quality-component distributions by route and role",
        ),
        (
            "worst_class_f1_stability",
            analysis.worst_class_f1_stability,
            "Top-10 worst-class-F1 stability review",
        ),
        (
            "incomplete_cases",
            analysis.incomplete_cases,
            "Cases excluded from ranking because requested execution was incomplete",
        ),
        ("cell_metrics_raw", bundle.cell_rows, "Normalized raw cell metrics"),
        ("training_history_raw", bundle.history_rows, "Normalized training history"),
        (
            "quality_diagnostics_raw",
            bundle.quality_rows,
            "Report projection of quality diagnostics; full beat-level audits remain in each case quality_diagnostics.json",
        ),
        ("case_records", bundle.case_records, "Case pass/fail/resume records"),
        (
            "reproducibility_summary",
            (reproducibility.summary,),
            "Report-only seed and frozen-split consistency status",
        ),
        (
            "reproducibility_cases",
            reproducibility.case_rows,
            "Selected attempt and seed/split evidence by case",
        ),
        (
            "reproducibility_cells",
            reproducibility.cell_rows,
            "Seed and participant-split evidence for every selected cell",
        ),
        (
            "reproducibility_splits",
            reproducibility.split_rows,
            "Frozen split roster and cross-case membership hashes",
        ),
        (
            "reproducibility_issues",
            reproducibility.issues,
            "Contradictory or not-verifiable reproducibility evidence",
        ),
    )
    if _is_stage3_centered_star(bundle.plan):
        table_payloads += tuple(
            (name, getattr(analysis, name), description)
            for name, _title, _notice, _columns, description in _STAGE3_STAR_TABLES
        )
    elif isinstance(bundle.plan.get("legacy_bridge"), Mapping):
        table_payloads += (
            (
                "legacy_bridge_numeric_ablation_report",
                analysis.legacy_bridge_numeric_ablation_report,
                "CompactCNN L0 baseline plus seven predefined adjacent numeric-profile ablations",
            ),
            (
                "legacy_bridge_execution_order_report",
                analysis.legacy_bridge_execution_order_report,
                "CompactCNN absolute metrics in L7,L5,L6,L4,L3,L2,L1,L0 execution order; no causal execution deltas",
            ),
        )
    index: list[dict[str, Any]] = []
    table_file_count = 0
    for name, rows, description in table_payloads:
        csv_path = tables / f"{name}.csv"
        json_path = tables / f"{name}.json"
        _write_csv(csv_path, rows)
        _write_json(json_path, list(rows))
        status = "available" if rows else "N/A_no_rows"
        index.extend(
            (
                _index_entry(
                    root,
                    csv_path,
                    artifact_type="table_csv",
                    description=description,
                    status=status,
                ),
                _index_entry(
                    root,
                    json_path,
                    artifact_type="table_json",
                    description=description,
                    status=status,
                ),
            )
        )
        table_file_count += 2
    confusion_entries, confusion_file_count = _write_top_confusion_matrix_files(
        root,
        tables,
        analysis,
    )
    index.extend(confusion_entries)
    table_file_count += confusion_file_count
    write_figures = bool(bundle.plan.get("report", {}).get("write_static_figures", True))
    if not write_figures:
        clear_static_figure_artifacts(figures_dir)
    figures = (
        generate_static_figures(bundle, analysis, figures_dir)
        if write_figures
        else (
            {
                "figure": "all_static_figures",
                "status": "disabled",
                "path": "",
                "reason": "write_static_figures=false",
            },
        )
    )
    plot_status = figures_dir / "plot_status.json"
    _write_json(plot_status, list(figures))
    index.append(
        _index_entry(
            root,
            plot_status,
            artifact_type="figure_index",
            description="Generated/N/A status for every requested figure",
        )
    )
    for figure in figures:
        raw_path = figure.get("path")
        if raw_path and (root / str(raw_path)).is_file():
            index.append(
                _index_entry(
                    root,
                    root / str(raw_path),
                    artifact_type="figure" if figure["status"] == "generated" else "na_marker",
                    description=str(figure["figure"]),
                    status=str(figure["status"]),
                )
            )
    summary_payload = {
        "schema_version": "ppg_frailty.study_report.v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study_directory": str(root),
        "plan": bundle.plan,
        "manifest": bundle.manifest,
        "varied_parameters": bundle.varied_parameters,
        "controlled_parameters": bundle.controlled_parameters,
        "analysis": asdict(analysis),
        "reproducibility_audit": reproducibility.to_dict(),
        "figure_status": list(figures),
    }
    summary_json = root / "study_summary.json"
    _write_json(summary_json, summary_payload)
    index.append(
        _index_entry(
            root,
            summary_json,
            artifact_type="summary_json",
            description="Machine-readable complete study summary",
        )
    )
    markdown_path = root / "STUDY_SUMMARY.md"
    markdown_path.write_text(
        _report_markdown(bundle, analysis, figures, reproducibility), encoding="utf-8"
    )
    index.append(
        _index_entry(
            root,
            markdown_path,
            artifact_type="summary_markdown",
            description="Primary human-readable study summary",
        )
    )
    write_html = bool(bundle.plan.get("report", {}).get("write_html", True))
    html_path = root / "STUDY_SUMMARY.html" if write_html else None
    if html_path is not None:
        html_path.write_text(
            _report_html(bundle, analysis, figures, reproducibility), encoding="utf-8"
        )
        index.append(
            _index_entry(
                root,
                html_path,
                artifact_type="summary_html",
                description="Portable HTML summary with figures",
            )
        )
    output_index = root / "outputs_index.json"
    complete_index = _complete_inventory(
        root,
        index,
        output_index=output_index,
    )
    _write_json(
        output_index,
        {
            "schema_version": "ppg_frailty.study_output_index.v2",
            "study_directory": str(root),
            "inventory_scope": "all_regular_files_below_study_directory",
            "artifacts": complete_index,
        },
    )
    return ReportResult(
        study_directory=root,
        summary_markdown=markdown_path,
        summary_html=html_path,
        output_index=output_index,
        table_count=table_file_count,
        generated_figure_count=sum(
            row.get("status") == "generated" for row in figures
        ),
        na_figure_count=sum(row.get("status") == "N/A" for row in figures),
    )


__all__ = ["ReportResult", "generate_study_report"]

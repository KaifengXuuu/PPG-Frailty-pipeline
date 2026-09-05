"""Fail-closed reporting for interrupted studies without a root manifest.

This reporter is deliberately execution-only.  It inventories the declared
plan, recursive progress JSONL logs, and terminal case-result statuses.  It
never synthesizes a study manifest and never reads predictions to calculate
performance, ranking, confidence intervals, or P values.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..data.schema import CANONICAL_CLASS_NAMES
from ..motion_ids import FORMAL_MOTION_MODEL_ID
from .components import build_pipeline_test_component_rows
from .conclusions import (
    paired_inference_against_reference,
    paired_repeat_deltas_against_reference,
)
from .profiles import REPORTER_PROFILES, annotate_component_row
from .tabular import (
    html_column_definitions_block,
    markdown_column_definitions_block,
    write_excel_workbook_from_csv_directory,
    write_table_column_definitions,
)


_PASS_STATUSES = {"passed", "success", "complete", "completed"}
_FAIL_STATUSES = {"failed", "failed_closed", "error", "aborted", "killed"}
_FAILURE_WORDS = (
    "failed",
    "failure",
    "error",
    "exception",
    "traceback",
    "killed",
    "aborted",
)
_STUDY_TERMINAL_EVENTS = {
    "study_finished",
    "study_failed",
    "study_aborted",
    "motion_peak_study_finished",
}
_MAX_HUMAN_TABLE_COLUMNS = 8


_PER_CLASS_FIELDS = (
    "classifier_id",
    "evaluation_id",
    "aggregation_level",
    "class_label",
    "class_name",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "support",
    "predicted_support",
    "observation_count",
    "input_observation_count",
    "retained_observation_count",
    "excluded_observation_count",
    "precision",
    "sensitivity",
    "recall",
    "specificity",
    "balanced_accuracy_ovr",
    "f1",
    "roc_auc_ovr",
    "pr_auc_ovr",
    "probability_metric_applicability",
    "result_applicability",
    "case_execution_status",
    "metric_scope",
    "metric_source",
    "prediction_rule_source",
)

_PAIRWISE_REPEAT_FIELDS = (
    "comparison_family",
    "comparison_id",
    "comparison_role",
    "reference_case_id",
    "candidate_case_id",
    "repeat",
    "split_seed",
    "matched_participant_count",
    "matched_roster_sha256",
    "comparison_contract_source",
    "comparison_contract_status",
    "unavailable_reason",
    "difference_direction",
    "reference_balanced_accuracy",
    "candidate_balanced_accuracy",
    "balanced_accuracy_delta",
    "reference_macro_f1",
    "candidate_macro_f1",
    "macro_f1_delta",
    "reference_macro_roc_auc_ovr",
    "candidate_macro_roc_auc_ovr",
    "macro_roc_auc_ovr_delta",
    "automatic_selection",
)

_PAIRED_INFERENCE_FIELDS = (
    "comparison_family",
    "comparison_id",
    "reference_case_id",
    "candidate_case_id",
    "metric",
    "candidate_minus_reference",
    "participant_cluster_delta_ci95_low",
    "participant_cluster_delta_ci95_high",
    "bootstrap_resamples",
    "bootstrap_valid_resamples",
    "bootstrap_seed",
    "bootstrap_cluster_unit",
    "bootstrap_interval_method",
    "raw_two_sided_p_value",
    "n_resamples",
    "seed",
    "participant_count",
    "repeat_count",
    "exchange_unit",
    "test_method",
    "p_value_applicability",
    "comparison_contract_source",
    "comparison_contract_status",
    "unavailable_reason",
    "inference_role",
    "automatic_selection",
    "holm_adjusted_p_value",
    "holm_rank",
    "holm_family_size",
    "alpha",
    "reject_null_after_holm",
    "interpretation",
)


_PER_CLASS_ROSTER_DISPLAY_FIELDS = (
    "classifier_id",
    "evaluation_id",
    "aggregation_level",
    "class_label",
    "class_name",
    "case_execution_status",
    "result_applicability",
)
_PER_CLASS_LONG_COUNT_DISPLAY_FIELDS = (
    "classifier_id",
    "class_name",
    "count",
    "value",
    "result_applicability",
)
_PER_CLASS_LONG_METRIC_DISPLAY_FIELDS = (
    "classifier_id",
    "class_name",
    "metric",
    "value",
    "metric_scope",
    "result_applicability",
)
_PER_CLASS_LONG_PROBABILITY_DISPLAY_FIELDS = (
    "classifier_id",
    "class_name",
    "metric",
    "value",
    "probability_metric_applicability",
    "metric_source",
    "prediction_rule_source",
)
_PAIRWISE_CONTRACT_DISPLAY_FIELDS = (
    "candidate_case_id",
    "reference_case_id",
    "comparison_id",
    "comparison_family",
    "comparison_role",
    "comparison_contract_source",
    "comparison_contract_status",
    "unavailable_reason",
)
_PAIRWISE_ROSTER_DISPLAY_FIELDS = (
    "candidate_case_id",
    "reference_case_id",
    "comparison_id",
    "repeat",
    "split_seed",
    "matched_participant_count",
    "matched_roster_sha256",
    "comparison_contract_status",
)
_PAIRWISE_LONG_METRIC_DISPLAY_FIELDS = (
    "candidate_case_id",
    "reference_case_id",
    "comparison_id",
    "repeat",
    "metric",
    "reference_value",
    "candidate_value",
    "candidate_minus_reference",
)
_PAIRWISE_INTERPRETATION_DISPLAY_FIELDS = (
    "candidate_case_id",
    "reference_case_id",
    "comparison_id",
    "repeat",
    "metric",
    "difference_direction",
    "unavailable_reason",
    "automatic_selection",
)
_PAIRED_INFERENCE_CONTRACT_DISPLAY_FIELDS = (
    "candidate_case_id",
    "reference_case_id",
    "comparison_id",
    "comparison_family",
    "metric",
    "inference_role",
    "comparison_contract_status",
    "unavailable_reason",
)
_PAIRED_INFERENCE_EFFECT_DISPLAY_FIELDS = (
    "candidate_case_id",
    "reference_case_id",
    "comparison_id",
    "metric",
    "candidate_minus_reference",
    "participant_cluster_delta_ci95_low",
    "participant_cluster_delta_ci95_high",
    "p_value_applicability",
)
_PAIRED_INFERENCE_SUPPORT_DISPLAY_FIELDS = (
    "candidate_case_id",
    "reference_case_id",
    "comparison_id",
    "metric",
    "participant_count",
    "repeat_count",
    "exchange_unit",
    "test_method",
)
_PAIRED_INFERENCE_BOOTSTRAP_DISPLAY_FIELDS = (
    "candidate_case_id",
    "reference_case_id",
    "comparison_id",
    "metric",
    "bootstrap_resamples",
    "bootstrap_valid_resamples",
    "bootstrap_seed",
    "bootstrap_interval_method",
)
_PAIRED_INFERENCE_BOOTSTRAP_METHOD_DISPLAY_FIELDS = (
    "candidate_case_id",
    "reference_case_id",
    "comparison_id",
    "metric",
    "bootstrap_cluster_unit",
    "test_method",
    "p_value_applicability",
    "unavailable_reason",
)
_PAIRED_INFERENCE_P_DISPLAY_FIELDS = (
    "candidate_case_id",
    "reference_case_id",
    "comparison_id",
    "metric",
    "raw_two_sided_p_value",
    "holm_adjusted_p_value",
    "reject_null_after_holm",
    "p_value_applicability",
)
_PAIRED_INFERENCE_MULTIPLICITY_DISPLAY_FIELDS = (
    "candidate_case_id",
    "reference_case_id",
    "comparison_id",
    "metric",
    "holm_rank",
    "holm_family_size",
    "alpha",
    "automatic_selection",
)
_PAIRED_INFERENCE_AUDIT_DISPLAY_FIELDS = (
    "candidate_case_id",
    "reference_case_id",
    "comparison_id",
    "metric",
    "n_resamples",
    "seed",
    "exchange_unit",
    "interpretation",
)


@dataclass(frozen=True)
class IncompleteStudyReportResult:
    """Paths and status returned by one execution-only report rebuild."""

    study_directory: Path
    summary_markdown: Path
    summary_html: Path
    methods_markdown: Path
    interpretation_markdown: Path
    outputs_index: Path
    status: str
    table_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "incomplete_report_regenerated",
            "study_status": self.status,
            "report_scope": "execution_audit_only",
            "formal_result_available": False,
            "ranking_eligible": False,
            "inference_eligible": False,
            "selection_eligible": False,
            "study_dir": str(self.study_directory),
            "root_report": str(self.summary_markdown),
            "root_report_html": str(self.summary_html),
            "outputs_index": str(self.outputs_index),
        }


def is_incomplete_study_directory(study_directory: str | Path) -> bool:
    """Return true only for a materialized plan lacking its root manifest."""

    root = Path(study_directory)
    return (
        root.is_dir()
        and (root / "study_plan.yaml").is_file()
        and not (root / "study_manifest.json").exists()
    )


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-report.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def _atomic_json(path: Path, value: Any) -> Path:
    return _atomic_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
    )


def _atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-report.tmp")
    resolved_fields = list(
        dict.fromkeys(
            (
                *(str(value) for value in (fields or ())),
                *(str(key) for row in rows for key in row),
            )
        )
    )
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        if resolved_fields:
            writer = csv.DictWriter(
                stream,
                fieldnames=resolved_fields,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: _json_value(value)
                        if isinstance(value, (dict, list, tuple))
                        else value
                        for key, value in row.items()
                    }
                )
        else:
            stream.write("\n")
    temporary.replace(path)
    return path


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _case_id(row: Any) -> str | None:
    if not isinstance(row, Mapping):
        return None
    for key in ("case_id", "catalog_case_id", "config_id", "profile_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _plan_case_ids(plan: Mapping[str, Any]) -> list[str]:
    output: list[str] = []
    for key in ("cases", "candidates"):
        values = plan.get(key, ())
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            output.extend(
                value for row in values if (value := _case_id(row)) is not None
            )
    legacy = plan.get("legacy_bridge")
    if isinstance(legacy, Mapping):
        profiles = legacy.get("profiles", ())
        if isinstance(profiles, Sequence) and not isinstance(profiles, (str, bytes)):
            output.extend(
                value for row in profiles if (value := _case_id(row)) is not None
            )
    return list(dict.fromkeys(output))


def _axis_count(value: Any) -> int | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value) if value else None
    return None


def _planned_cells_per_case(plan: Mapping[str, Any]) -> int | None:
    for key in ("execution", "resource"):
        contract = plan.get(key)
        if not isinstance(contract, Mapping):
            continue
        repeat_count = _axis_count(contract.get("repeats"))
        fold_count = _axis_count(contract.get("folds"))
        if repeat_count is not None and fold_count is not None:
            return repeat_count * fold_count
    return None


def _planned_repeats(plan: Mapping[str, Any]) -> tuple[int, ...]:
    """Return only repeat indices explicitly persisted in the study contract."""

    for key in ("execution", "resource"):
        contract = plan.get(key)
        if not isinstance(contract, Mapping):
            continue
        for repeat_key in ("repeats", "promotion_repeats", "full_repeats"):
            values = contract.get(repeat_key)
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                try:
                    repeats = tuple(dict.fromkeys(int(value) for value in values))
                except (TypeError, ValueError):
                    continue
                if repeats:
                    return repeats
    legacy = plan.get("legacy_bridge")
    budget = legacy.get("budget") if isinstance(legacy, Mapping) else None
    values = budget.get("repeat_indices") if isinstance(budget, Mapping) else None
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        try:
            return tuple(dict.fromkeys(int(value) for value in values))
        except (TypeError, ValueError):
            pass
    return ()


def _legacy_case_id_map(plan: Mapping[str, Any]) -> dict[str, str]:
    """Map bridge runtime identities back to the classifier case roster."""

    legacy = plan.get("legacy_bridge")
    profiles = legacy.get("profiles", ()) if isinstance(legacy, Mapping) else ()
    output: dict[str, str] = {}
    if isinstance(profiles, Sequence) and not isinstance(profiles, (str, bytes)):
        for row in profiles:
            if not isinstance(row, Mapping):
                continue
            runtime_id = row.get("case_id")
            catalog_id = row.get("catalog_case_id", runtime_id)
            if runtime_id is not None and catalog_id is not None:
                output[str(runtime_id)] = str(catalog_id)
            if catalog_id is not None:
                output[str(catalog_id)] = str(catalog_id)
    return output


def _declared_classifier_case_ids(
    plan: Mapping[str, Any],
    component_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Recover classifier cases without treating unresolved placeholders as models."""

    output: list[str] = []
    containers: list[Mapping[str, Any]] = [plan]
    for key in ("manifest", "study_manifest"):
        value = plan.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    for container in containers:
        for key in ("cases", "candidates"):
            values = container.get(key, ())
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                output.extend(
                    case_id for row in values if (case_id := _case_id(row)) is not None
                )
    if not output:
        output.extend(
            str(row.get("catalog_case_id", row.get("case_id")))
            for row in (
                plan.get("legacy_bridge", {}).get("profiles", ())
                if isinstance(plan.get("legacy_bridge"), Mapping)
                else ()
            )
            if isinstance(row, Mapping)
            and row.get("catalog_case_id", row.get("case_id")) is not None
        )
    output.extend(
        str(row.get("participating_cases"))
        for row in component_rows
        if str(row.get("component_role", "")) == "classifier"
        and str(row.get("participating_cases", "")).strip()
    )
    id_map = _legacy_case_id_map(plan)
    return tuple(dict.fromkeys(id_map.get(value, value) for value in output))


def _declared_comparison_pairs(
    plan: Mapping[str, Any],
    classifier_case_ids: Sequence[str],
) -> tuple[dict[str, str], ...]:
    """Recover only explicitly declared pairwise classifier comparisons."""

    study = plan.get("study") if isinstance(plan.get("study"), Mapping) else {}
    study_id = str(study.get("study_id") or plan.get("study_id") or "study")
    decision_role = str(study.get("decision_role", "comparison"))
    id_map = _legacy_case_id_map(plan)

    def canonical(value: Any) -> str:
        raw = str(value)
        return id_map.get(raw, raw)

    pairs: list[dict[str, str]] = []

    def add(
        reference: Any,
        candidate: Any,
        *,
        family: str,
        role: str,
        source: str,
    ) -> None:
        if reference in (None, "") or candidate in (None, ""):
            return
        reference_id = canonical(reference)
        candidate_id = canonical(candidate)
        if reference_id == candidate_id:
            return
        pairs.append(
            {
                "comparison_family": family,
                "comparison_role": role,
                "reference_case_id": reference_id,
                "candidate_case_id": candidate_id,
                "comparison_contract_source": source,
            }
        )

    declared_containers: list[tuple[str, Mapping[str, Any]]] = [
        ("study_plan.yaml", plan)
    ]
    for key in ("manifest", "study_manifest"):
        value = plan.get(key)
        if isinstance(value, Mapping):
            declared_containers.append((f"study_plan.yaml:{key}", value))
    for container_source, container in declared_containers:
        for key in ("comparisons", "comparison_pairs"):
            values = container.get(key, ())
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            for row in values:
                if not isinstance(row, Mapping):
                    continue
                add(
                    row.get("reference_case_id"),
                    row.get(
                        "candidate_case_id",
                        row.get("variant_case_id"),
                    ),
                    family=str(
                        row.get("comparison_family", f"{study_id}__declared_pairs")
                    ),
                    role=str(
                        row.get(
                            "comparison_role",
                            "predeclared_pairwise_comparison",
                        )
                    ),
                    source=f"{container_source}:{key}",
                )

    legacy = plan.get("legacy_bridge")
    legacy = legacy if isinstance(legacy, Mapping) else {}
    centered = legacy.get("centered_comparisons", ())
    if (
        not pairs
        and isinstance(centered, Sequence)
        and not isinstance(centered, (str, bytes))
    ):
        for row in centered:
            if not isinstance(row, Mapping):
                continue
            model_id = str(row.get("model_id", "model"))
            add(
                row.get("reference_case_id"),
                row.get("variant_case_id", row.get("candidate_case_id")),
                family=f"{study_id}__centered_star__{model_id}",
                role="predeclared_centered_star_single_factor_ablation",
                source="study_plan.yaml:legacy_bridge.centered_comparisons",
            )
    if not pairs:
        profiles = legacy.get("profiles", ())
        if isinstance(profiles, Sequence) and not isinstance(profiles, (str, bytes)):
            for row in profiles:
                if not isinstance(row, Mapping):
                    continue
                add(
                    row.get("reference_case_id"),
                    row.get("catalog_case_id", row.get("case_id")),
                    family=f"{study_id}__profile_references",
                    role="predeclared_profile_reference_comparison",
                    source="study_plan.yaml:legacy_bridge.profiles.reference_case_id",
                )
    if not pairs and str(legacy.get("design", "")) in {
        "centered_star_v1",
        "field_driven_followup_v1",
    }:
        profiles = legacy.get("profiles", ())
        if isinstance(profiles, Sequence) and not isinstance(profiles, (str, bytes)):
            profile_rows = [row for row in profiles if isinstance(row, Mapping)]
            references: dict[str, str] = {}
            if legacy.get("design") == "centered_star_v1":
                references.update(
                    {
                        str(row.get("model_id", "model")): canonical(
                            row.get("catalog_case_id", row.get("case_id"))
                        )
                        for row in profile_rows
                        if str(row.get("profile_id", "")) == "B0"
                        and row.get("catalog_case_id", row.get("case_id"))
                        not in (None, "")
                    }
                )
            else:
                for row in profile_rows:
                    candidate = row.get("catalog_case_id", row.get("case_id"))
                    if candidate not in (None, ""):
                        references.setdefault(
                            str(row.get("model_id", "model")),
                            canonical(candidate),
                        )
            for row in profile_rows:
                model_id = str(row.get("model_id", "model"))
                candidate = row.get("catalog_case_id", row.get("case_id"))
                reference = references.get(model_id)
                if candidate in (None, "") or reference is None:
                    continue
                if canonical(candidate) == reference:
                    continue
                add(
                    reference,
                    candidate,
                    family=f"{study_id}__declared_{legacy.get('design')}__{model_id}",
                    role=(
                        "predeclared_centered_star_single_factor_ablation"
                        if legacy.get("design") == "centered_star_v1"
                        else "predeclared_field_driven_followup_comparison"
                    ),
                    source="study_plan.yaml:legacy_bridge.design_and_profiles",
                )
    if not pairs:
        adjacent = legacy.get("adjacent_comparisons", ())
        if isinstance(adjacent, Sequence) and not isinstance(adjacent, (str, bytes)):
            for value in adjacent:
                parts = str(value).split("->")
                if len(parts) != 2:
                    continue
                add(
                    parts[0],
                    parts[1],
                    family=f"{study_id}__legacy_adjacent_ablation",
                    role="predeclared_adjacent_numeric_ablation",
                    source="study_plan.yaml:legacy_bridge.adjacent_comparisons",
                )
    if not pairs:
        for container_key in ("cases", "candidates"):
            values = plan.get(container_key, ())
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            for row in values:
                if isinstance(row, Mapping):
                    add(
                        row.get("reference_case_id"),
                        _case_id(row),
                        family=f"{study_id}__case_references",
                        role="predeclared_case_reference_comparison",
                        source=f"study_plan.yaml:{container_key}.reference_case_id",
                    )
    if not pairs:
        reference_sources: list[tuple[Any, str]] = [
            (study.get("reference_case_id"), "study_plan.yaml:study.reference_case_id")
        ]
        reference_sources.extend(
            (
                container.get("reference_case_id"),
                f"{source}:reference_case_id",
            )
            for source, container in declared_containers[1:]
        )
        for reference, source in reference_sources:
            if reference in (None, ""):
                continue
            for candidate in classifier_case_ids:
                add(
                    reference,
                    candidate,
                    family=f"{study_id}__declared_reference",
                    role=f"predeclared_{decision_role}_reference_comparison",
                    source=source,
                )
            break

    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in pairs:
        key = (
            row["comparison_family"],
            row["reference_case_id"],
            row["candidate_case_id"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return tuple(unique)


def _incomplete_classifier_tables(
    plan: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand declared classifier contracts into explicit, non-numeric N/A rows."""

    classifier_ids = list(_declared_classifier_case_ids(plan, component_rows))
    pairs = list(_declared_comparison_pairs(plan, classifier_ids))
    for pair in pairs:
        classifier_ids.extend((pair["reference_case_id"], pair["candidate_case_id"]))
    classifier_ids = list(dict.fromkeys(classifier_ids))
    status_by_case = {
        str(row.get("case_id")): str(row.get("case_status", "unknown"))
        for row in case_rows
        if row.get("case_id") is not None
    }
    per_class_reason = "N/A_incomplete_study_no_formal_classifier_result"
    per_class_rows = [
        {
            "classifier_id": classifier_id,
            "evaluation_id": "participant_outer_oof",
            "aggregation_level": "participant",
            "class_label": class_label,
            "class_name": class_name,
            "true_positive": None,
            "false_positive": None,
            "true_negative": None,
            "false_negative": None,
            "support": None,
            "predicted_support": None,
            "observation_count": 0,
            "input_observation_count": 0,
            "retained_observation_count": 0,
            "excluded_observation_count": 0,
            "precision": None,
            "sensitivity": None,
            "recall": None,
            "specificity": None,
            "balanced_accuracy_ovr": None,
            "f1": None,
            "roc_auc_ovr": None,
            "pr_auc_ovr": None,
            "probability_metric_applicability": per_class_reason,
            "result_applicability": per_class_reason,
            "case_execution_status": status_by_case.get(classifier_id, "planned"),
            "metric_scope": "one_vs_rest_not_computable",
            "metric_source": "N/A_no_formal_classifier_evidence",
            "prediction_rule_source": "N/A_no_formal_classifier_evidence",
        }
        for classifier_id in classifier_ids
        for class_label, class_name in sorted(CANONICAL_CLASS_NAMES.items())
    ]

    unavailable_reason = (
        "incomplete_study_has_no_formal_root_manifest_or_eligible_matched_"
        "participant_oof_predictions"
    )
    repeats = _planned_repeats(plan)
    repeat_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    for pair in pairs:
        generated_repeat_rows = paired_repeat_deltas_against_reference(
            (),
            reference_case_id=pair["reference_case_id"],
            comparison_family=pair["comparison_family"],
            comparison_role=pair["comparison_role"],
            candidate_case_ids=(pair["candidate_case_id"],),
            expected_repeats=repeats or None,
        )
        for row in generated_repeat_rows:
            row.update(
                {
                    "comparison_contract_source": pair["comparison_contract_source"],
                    "comparison_contract_status": f"N/A_{unavailable_reason}",
                    "unavailable_reason": unavailable_reason,
                }
            )
            repeat_rows.append(row)
        for row in paired_inference_against_reference(
            (),
            reference_case_id=pair["reference_case_id"],
            comparison_family=pair["comparison_family"],
            inference_role=pair["comparison_role"],
            candidate_case_ids=(pair["candidate_case_id"],),
            expected_repeats=repeats or None,
        ):
            row.update(
                {
                    "bootstrap_resamples": None,
                    "bootstrap_valid_resamples": None,
                    "bootstrap_seed": None,
                    "bootstrap_interval_method": "N/A_no_bootstrap_executed",
                    "n_resamples": None,
                    "seed": None,
                    "comparison_contract_source": pair["comparison_contract_source"],
                    "comparison_contract_status": f"N/A_{unavailable_reason}",
                    "p_value_applicability": f"N/A_{unavailable_reason}",
                    "unavailable_reason": unavailable_reason,
                    "interpretation": (
                        "Declared pair retained as explicit N/A; the incomplete "
                        "reporter does not promote case fragments into formal "
                        "participant-cluster inference."
                    ),
                }
            )
            inference_rows.append(row)
    return per_class_rows, repeat_rows, inference_rows


def _progress_logs(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("progress_events.jsonl"))
        if "result_backup" not in path.parts
    ]


def _load_events(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for path in _progress_logs(root):
        relative = path.relative_to(root).as_posix()
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                malformed.append(
                    {
                        "source_progress_path": relative,
                        "line_number": line_number,
                        "timestamp_utc": "",
                        "event": "malformed_progress_line",
                        "case_id": "",
                        "repeat_index": None,
                        "fold_index": None,
                        "message": f"JSONDecodeError: {error}",
                        "classification": "malformed_progress_evidence",
                    }
                )
                continue
            if not isinstance(value, Mapping):
                malformed.append(
                    {
                        "source_progress_path": relative,
                        "line_number": line_number,
                        "timestamp_utc": "",
                        "event": "malformed_progress_line",
                        "case_id": "",
                        "repeat_index": None,
                        "fold_index": None,
                        "message": "progress JSON root is not a mapping",
                        "classification": "malformed_progress_evidence",
                    }
                )
                continue
            events.append(
                {
                    **dict(value),
                    "_source_progress_path": relative,
                    "_line_number": line_number,
                }
            )
    events.sort(
        key=lambda row: (
            str(row.get("timestamp_utc", "")),
            str(row["_source_progress_path"]),
            int(row["_line_number"]),
        )
    )
    return events, malformed


def _event_case_id(event: Mapping[str, Any]) -> str | None:
    value = event.get("case_id")
    return str(value) if value is not None and str(value).strip() else None


def _cell_key(event: Mapping[str, Any]) -> tuple[str, int, int] | None:
    case_id = _event_case_id(event)
    repeat = event.get("repeat", event.get("repeat_index"))
    fold = event.get("fold", event.get("fold_index"))
    if case_id is None or repeat is None or fold is None:
        return None
    try:
        return case_id, int(repeat), int(fold)
    except (TypeError, ValueError):
        return None


def _failure_message(event: Mapping[str, Any]) -> str | None:
    status = str(event.get("status", "")).strip().lower()
    message = str(event.get("message", event.get("error", ""))).strip()
    name = str(event.get("event", event.get("stage", ""))).strip().lower()
    combined = " ".join((status, name, message.lower()))
    if status in _FAIL_STATUSES:
        return message or f"status={status}"
    if any(word in combined for word in _FAILURE_WORDS):
        return message or name
    return None


def _failure_row(event: Mapping[str, Any], classification: str) -> dict[str, Any]:
    return {
        "source_progress_path": event.get("_source_progress_path", ""),
        "line_number": event.get("_line_number"),
        "timestamp_utc": event.get("timestamp_utc", ""),
        "event": event.get("event", event.get("stage", "")),
        "case_id": event.get("case_id", ""),
        "repeat_index": event.get("repeat", event.get("repeat_index")),
        "fold_index": event.get("fold", event.get("fold_index")),
        "message": event.get("message", event.get("error", "")),
        "classification": classification,
    }


def _load_case_results(root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("case_result.json")):
        if "result_backup" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, Mapping):
            continue
        case_id = str(payload.get("case_id") or path.parent.name)
        cells: list[tuple[int, int, str]] = []
        result = payload.get("result")
        values = result.get("cell_results", ()) if isinstance(result, Mapping) else ()
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for cell in values:
                if not isinstance(cell, Mapping):
                    continue
                try:
                    identity = (
                        int(cell.get("repeat_index", cell.get("repeat"))),
                        int(cell.get("fold_index", cell.get("fold"))),
                    )
                except (TypeError, ValueError):
                    continue
                cells.append((*identity, str(cell.get("status", "unknown"))))
        results[case_id] = {
            "status": str(payload.get("status", "unknown")),
            "path": path.relative_to(root).as_posix(),
            "cells": cells,
            "error": str(payload.get("error", "")),
        }
    return results


def _discovered_case_ids(
    root: Path,
    events: Sequence[Mapping[str, Any]],
    case_results: Mapping[str, Any],
) -> list[str]:
    values = [case_id for row in events if (case_id := _event_case_id(row))]
    values.extend(case_results)
    for path in root.rglob("resolved_config.yaml"):
        if "result_backup" not in path.parts:
            values.append(path.parent.name)
    return list(dict.fromkeys(values))


def _plan_case_rows(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for key in ("cases", "candidates"):
        values = plan.get(key, ())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for value in values:
            case_id = _case_id(value)
            if case_id is not None and isinstance(value, Mapping):
                rows.setdefault(case_id, value)
    legacy = plan.get("legacy_bridge")
    profiles = legacy.get("profiles", ()) if isinstance(legacy, Mapping) else ()
    if isinstance(profiles, Sequence) and not isinstance(profiles, (str, bytes)):
        for value in profiles:
            case_id = _case_id(value)
            if case_id is not None and isinstance(value, Mapping):
                rows.setdefault(case_id, value)
    return rows


def _component_execution_state(case_status: str | None) -> str:
    status = str(case_status or "planned").lower()
    if status == "complete":
        return "complete"
    if status == "not_started":
        return "not_started"
    if status.startswith("partial") or status == "failed":
        return "partial"
    return "planned"


def _planned_component_row(
    *,
    case_id: str,
    role: str,
    module_id: str,
    execution_state: str,
    configured_state: str,
    input_data: Mapping[str, Any],
    fixed_parameters: Mapping[str, Any],
    evidence_source: str,
    inventory_reason: str,
) -> dict[str, Any]:
    """Build one planned row while resolving reporter metadata when possible."""

    raw = {
        "participating_cases": case_id,
        "component_role": role,
        "module_id": module_id,
        # ``planned`` requests registry metadata without claiming execution. The
        # case-evidence state replaces this immediately after annotation.
        "execution_state": (
            "not_executed_disabled"
            if "disabled" in configured_state.lower()
            else "planned"
        ),
        "input_data": _json_value(input_data),
        "fixed_parameters": _json_value(fixed_parameters),
        "algorithm_kernel_description": "",
    }
    try:
        row = annotate_component_row(raw)
    except ValueError as error:
        row = {
            **raw,
            "reporter_profile_id": "audit_provenance_v1",
            "model_reporter_extension_id": "not_applicable",
            "algorithm_kernel_description": (
                "N/A — planned identity has no resolvable active registry binding"
            ),
            "algorithm_references": f"N/A — {error}",
            "reporter_binding_kind": "planned_unresolved",
            "reporter_binding_source": "incomplete_reporter_fail_closed",
        }
        inventory_reason = "; ".join(
            value for value in (inventory_reason, str(error)) if value
        )
    registered_id = str(row.get("registered_module_id", ""))
    if registered_id and registered_id != "not_applicable":
        row["module_id"] = registered_id
    row.update(
        {
            "execution_state": execution_state,
            "configured_state": configured_state,
            "execution_state_basis": (
                "case-level persisted evidence only; not an independent claim "
                "that this component executed"
            ),
            "evidence_source": evidence_source,
            "inventory_reason": inventory_reason,
        }
    )
    return row


def _plan_only_component_rows(
    plan: Mapping[str, Any],
    case_id: str,
    case: Mapping[str, Any] | None,
    execution_state: str,
) -> list[dict[str, Any]]:
    """Project only identities explicitly recoverable from the root plan."""

    case = case or {}
    base = plan.get("base") if isinstance(plan.get("base"), Mapping) else {}
    base_overrides = (
        base.get("common_overrides", {})
        if isinstance(base.get("common_overrides"), Mapping)
        else {}
    )
    case_overrides = (
        case.get("overrides", {}) if isinstance(case.get("overrides"), Mapping) else {}
    )
    overrides = {**dict(base_overrides), **dict(case_overrides)}
    fixed = {
        "plan_case": dict(case),
        "base": dict(base),
        "effective_declared_overrides": overrides,
        "resource": dict(plan.get("resource", {}))
        if isinstance(plan.get("resource"), Mapping)
        else {},
        "execution": dict(plan.get("execution", {}))
        if isinstance(plan.get("execution"), Mapping)
        else {},
    }
    unavailable_input = {
        "availability": "N/A",
        "reason": (
            "no persisted resolved_config.yaml for this case; dataset paths, "
            "channels and realized signal views cannot be recovered from the root plan"
        ),
    }
    source = "study_plan.yaml only; resolved case config unavailable"
    reason = (
        "planned identity only; fixed values are declarations, not execution evidence"
    )
    specifications: list[tuple[str, str, str]] = []

    model_id = (
        case.get("model_id")
        or case.get("catalog_entry")
        or base.get("model_id")
        or base.get("catalog_entry")
    )
    if model_id:
        specifications.append(("classifier", str(model_id), "planned_enabled"))
    representation = (
        case.get("representation_mode")
        or base.get("representation_mode")
        or case.get("output_group")
        or base.get("output_group")
    )
    if str(representation) in {"raw", "feature_vector", "feature_matrix", "fusion"}:
        specifications.append(
            ("representation", str(representation), "planned_enabled")
        )
    if "signal.imu.gravity_method" in overrides:
        specifications.append(
            (
                "imu_preprocessing",
                str(overrides["signal.imu.gravity_method"]),
                "planned_enabled",
            )
        )
    if "signal.peak_detector.detector_id" in overrides:
        specifications.append(
            (
                "peak_detector",
                str(overrides["signal.peak_detector.detector_id"]),
                "planned_enabled",
            )
        )
    signal_overrides = {
        key: value for key, value in overrides.items() if str(key).startswith("signal.")
    }
    if signal_overrides:
        specifications.append(
            (
                "signal_views_and_scaling",
                "planned_signal_views_and_scaling",
                "planned_enabled",
            )
        )
    window_overrides = {
        key: value
        for key, value in overrides.items()
        if str(key).startswith("windows.")
    }
    if window_overrides:
        specifications.append(
            ("window_planner", "planned_window_contract", "planned_enabled")
        )
    if "training.optimizer" in overrides:
        specifications.append(
            ("trainer", str(overrides["training.optimizer"]), "planned_enabled")
        )
    if "aggregation.balance_line" in overrides:
        specifications.append(
            (
                "aggregation",
                str(overrides["aggregation.balance_line"]),
                "planned_enabled",
            )
        )
    if "quality.mode" in overrides:
        mode = str(overrides["quality.mode"])
        specifications.append(
            (
                "sqi",
                f"quality_{mode}",
                "planned_disabled" if mode == "off" else "planned_enabled",
            )
        )
    if "artifact.motion_detector_enabled" in overrides:
        enabled = bool(overrides["artifact.motion_detector_enabled"])
        motion_id = str(
            overrides.get("artifact.motion_detector.model_id", FORMAL_MOTION_MODEL_ID)
        )
        specifications.append(
            (
                "motion_detector",
                motion_id,
                "planned_enabled" if enabled else "planned_disabled",
            )
        )
    if "artifact.reducer" in overrides:
        denoiser_enabled = bool(overrides.get("artifact.denoiser_enabled", False))
        specifications.append(
            (
                "denoiser",
                str(overrides["artifact.reducer"]),
                "planned_enabled" if denoiser_enabled else "planned_disabled",
            )
        )

    if not specifications:
        specifications.append(
            (
                "dataset_adapter",
                "not_available_from_study_plan",
                "planned_identity_unresolved",
            )
        )
    rows = [
        _planned_component_row(
            case_id=case_id,
            role=role,
            module_id=module_id,
            execution_state=execution_state,
            configured_state=configured_state,
            input_data=unavailable_input,
            fixed_parameters={
                **fixed,
                "component_specific_declared_values": (
                    signal_overrides
                    if role == "signal_views_and_scaling"
                    else window_overrides
                    if role == "window_planner"
                    else overrides
                ),
            },
            evidence_source=source,
            inventory_reason=reason,
        )
        for role, module_id, configured_state in specifications
    ]
    return rows


def _resolved_config_paths(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    candidates = sorted(
        (
            path
            for path in root.rglob("resolved_config.yaml")
            if "result_backup" not in path.parts
        ),
        key=lambda path: (len(path.relative_to(root).parts), path.as_posix()),
    )
    for path in candidates:
        paths.setdefault(path.parent.name, path)
    return paths


def _build_planned_component_inventory(
    root: Path,
    plan: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[Path]]:
    plan_cases = _plan_case_rows(plan)
    case_states = {
        str(row["case_id"]): _component_execution_state(str(row["case_status"]))
        for row in case_rows
    }
    resolved = _resolved_config_paths(root)
    case_ids = list(dict.fromkeys((*plan_cases, *case_states, *resolved)))
    rows: list[dict[str, Any]] = []
    evidence_paths: list[Path] = []
    for case_id in case_ids:
        state = case_states.get(case_id, "planned")
        path = resolved.get(case_id)
        if path is None:
            rows.extend(
                _plan_only_component_rows(plan, case_id, plan_cases.get(case_id), state)
            )
            continue
        relative = path.relative_to(root).as_posix()
        evidence_paths.append(path)
        try:
            resolved_rows = build_pipeline_test_component_rows(
                root,
                {"cases": [{"case_id": case_id, "resolved_config_path": relative}]},
            )
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            fallback = _plan_only_component_rows(
                plan, case_id, plan_cases.get(case_id), state
            )
            for row in fallback:
                row[
                    "inventory_reason"
                ] = f"resolved config could not be projected: {error}; " + str(
                    row["inventory_reason"]
                )
                row["evidence_source"] = relative
            rows.extend(fallback)
            continue
        for source in resolved_rows:
            row = dict(source)
            row.update(
                {
                    "execution_state": state,
                    "configured_state": str(source.get("execution_state", "N/A")),
                    "execution_state_basis": (
                        "case-level persisted evidence only; not an independent claim "
                        "that this component executed"
                    ),
                    "evidence_source": relative,
                    "inventory_reason": (
                        "input data and fixed parameters projected from persisted "
                        "resolved config; execution state is case-level only"
                    ),
                }
            )
            rows.append(row)
    return rows, evidence_paths


def _planned_reporter_profile_rows(
    component_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in component_rows:
        for field in ("reporter_profile_id", "model_reporter_extension_id"):
            profile_id = str(row.get(field, ""))
            if profile_id in REPORTER_PROFILES:
                grouped.setdefault(profile_id, []).append(row)
    output: list[dict[str, Any]] = []
    for profile_id, rows in sorted(grouped.items()):
        profile = REPORTER_PROFILES[profile_id]
        output.append(
            {
                **asdict(profile),
                "profile_kind": (
                    "model_or_module_extension"
                    if any(
                        str(row.get("model_reporter_extension_id", "")) == profile_id
                        for row in rows
                    )
                    else "endpoint_or_audit"
                ),
                "planned_components": sorted(
                    {
                        f"{row.get('participating_cases')}:{row.get('component_role')}:"
                        f"{row.get('module_id')}"
                        for row in rows
                    }
                ),
                "execution_states": sorted(
                    {str(row.get("execution_state", "planned")) for row in rows}
                ),
                "module_algorithm_references": sorted(
                    {
                        str(row.get("algorithm_references", ""))
                        for row in rows
                        if str(row.get("algorithm_references", "")).strip()
                    }
                ),
                "module_algorithm_summaries": sorted(
                    {
                        f"{row.get('participating_cases')}:{row.get('component_role')}:"
                        f"{row.get('module_id')} — "
                        f"{row.get('algorithm_kernel_description')}"
                        for row in rows
                        if str(row.get("algorithm_kernel_description", "")).strip()
                    }
                ),
                "execution_claim": (
                    "none; profile requirements are planned/audit metadata only"
                ),
            }
        )
    return output


def _project_unique_rows(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> list[dict[str, Any]]:
    """Project display-only columns and remove repeated semantic rows.

    The lossless CSV/JSON exports continue to use the original rows.  This
    helper exists only to keep the human report readable without silently
    changing the persisted evidence contract.
    """

    projected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        value = {field: row.get(field) for field in fields}
        identity = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if identity not in seen:
            seen.add(identity)
            projected.append(value)
    return projected


def _long_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity_fields: Sequence[str],
    metric_fields: Sequence[str],
    metric_key: str = "metric",
    value_key: str = "value",
    passthrough_fields: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Convert a wide endpoint block into one row per named metric."""

    return [
        {
            **{field: row.get(field) for field in identity_fields},
            metric_key: metric,
            value_key: row.get(metric),
            **{field: row.get(field) for field in passthrough_fields},
        }
        for row in rows
        for metric in metric_fields
    ]


def _per_class_display_tables(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, list[dict[str, Any]], tuple[str, ...]], ...]:
    """Return semantic per-class projections with at most eight columns."""

    roster = _project_unique_rows(rows, _PER_CLASS_ROSTER_DISPLAY_FIELDS)
    confusion = _long_metric_rows(
        rows,
        identity_fields=("classifier_id", "class_name"),
        metric_fields=(
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
            "support",
            "predicted_support",
        ),
        metric_key="count",
        passthrough_fields=("result_applicability",),
    )
    coverage = _long_metric_rows(
        rows,
        identity_fields=("classifier_id", "class_name"),
        metric_fields=(
            "observation_count",
            "input_observation_count",
            "retained_observation_count",
            "excluded_observation_count",
        ),
        metric_key="count",
        passthrough_fields=("result_applicability",),
    )
    performance = _long_metric_rows(
        rows,
        identity_fields=("classifier_id", "class_name"),
        metric_fields=(
            "precision",
            "sensitivity",
            "recall",
            "specificity",
            "balanced_accuracy_ovr",
            "f1",
        ),
        passthrough_fields=("metric_scope", "result_applicability"),
    )
    probability = _long_metric_rows(
        rows,
        identity_fields=("classifier_id", "class_name"),
        metric_fields=("roc_auc_ovr", "pr_auc_ovr"),
        passthrough_fields=(
            "probability_metric_applicability",
            "metric_source",
            "prediction_rule_source",
        ),
    )
    return (
        (
            "Declared class roster and applicability",
            roster,
            _PER_CLASS_ROSTER_DISPLAY_FIELDS,
        ),
        ("Per-class confusion counts", confusion, _PER_CLASS_LONG_COUNT_DISPLAY_FIELDS),
        (
            "Per-class observation coverage",
            coverage,
            _PER_CLASS_LONG_COUNT_DISPLAY_FIELDS,
        ),
        (
            "Per-class thresholded metrics",
            performance,
            _PER_CLASS_LONG_METRIC_DISPLAY_FIELDS,
        ),
        (
            "Per-class probability metrics",
            probability,
            _PER_CLASS_LONG_PROBABILITY_DISPLAY_FIELDS,
        ),
    )


def _pairwise_repeat_display_tables(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, list[dict[str, Any]], tuple[str, ...]], ...]:
    """Return contract, roster, and long-form repeat metric projections."""

    contract = _project_unique_rows(rows, _PAIRWISE_CONTRACT_DISPLAY_FIELDS)
    roster = _project_unique_rows(rows, _PAIRWISE_ROSTER_DISPLAY_FIELDS)
    metric_rows = [
        {
            "candidate_case_id": row.get("candidate_case_id"),
            "reference_case_id": row.get("reference_case_id"),
            "comparison_id": row.get("comparison_id"),
            "repeat": row.get("repeat"),
            "metric": metric,
            "reference_value": row.get(f"reference_{metric}"),
            "candidate_value": row.get(f"candidate_{metric}"),
            "candidate_minus_reference": row.get(f"{metric}_delta"),
            "difference_direction": row.get("difference_direction"),
            "unavailable_reason": row.get("unavailable_reason"),
            "automatic_selection": row.get("automatic_selection"),
        }
        for row in rows
        for metric in (
            "balanced_accuracy",
            "macro_f1",
            "macro_roc_auc_ovr",
        )
    ]
    return (
        ("Declared comparison contracts", contract, _PAIRWISE_CONTRACT_DISPLAY_FIELDS),
        ("Repeat and matched-roster audit", roster, _PAIRWISE_ROSTER_DISPLAY_FIELDS),
        (
            "Repeat metric differences",
            metric_rows,
            _PAIRWISE_LONG_METRIC_DISPLAY_FIELDS,
        ),
        (
            "Repeat metric applicability and interpretation",
            metric_rows,
            _PAIRWISE_INTERPRETATION_DISPLAY_FIELDS,
        ),
    )


def _paired_inference_display_tables(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, list[dict[str, Any]], tuple[str, ...]], ...]:
    """Return narrow participant-inference projections without losing raw fields."""

    return (
        (
            "Inference contracts and applicability",
            _project_unique_rows(rows, _PAIRED_INFERENCE_CONTRACT_DISPLAY_FIELDS),
            _PAIRED_INFERENCE_CONTRACT_DISPLAY_FIELDS,
        ),
        (
            "Effect estimates and participant-cluster intervals",
            _project_unique_rows(rows, _PAIRED_INFERENCE_EFFECT_DISPLAY_FIELDS),
            _PAIRED_INFERENCE_EFFECT_DISPLAY_FIELDS,
        ),
        (
            "Inference sample support and exchange unit",
            _project_unique_rows(rows, _PAIRED_INFERENCE_SUPPORT_DISPLAY_FIELDS),
            _PAIRED_INFERENCE_SUPPORT_DISPLAY_FIELDS,
        ),
        (
            "Bootstrap contract",
            _project_unique_rows(rows, _PAIRED_INFERENCE_BOOTSTRAP_DISPLAY_FIELDS),
            _PAIRED_INFERENCE_BOOTSTRAP_DISPLAY_FIELDS,
        ),
        (
            "Bootstrap method and applicability",
            _project_unique_rows(
                rows, _PAIRED_INFERENCE_BOOTSTRAP_METHOD_DISPLAY_FIELDS
            ),
            _PAIRED_INFERENCE_BOOTSTRAP_METHOD_DISPLAY_FIELDS,
        ),
        (
            "Paired P values and Holm adjustment",
            _project_unique_rows(rows, _PAIRED_INFERENCE_P_DISPLAY_FIELDS),
            _PAIRED_INFERENCE_P_DISPLAY_FIELDS,
        ),
        (
            "Multiplicity and selection audit",
            _project_unique_rows(rows, _PAIRED_INFERENCE_MULTIPLICITY_DISPLAY_FIELDS),
            _PAIRED_INFERENCE_MULTIPLICITY_DISPLAY_FIELDS,
        ),
        (
            "Permutation and interpretation audit",
            _project_unique_rows(rows, _PAIRED_INFERENCE_AUDIT_DISPLAY_FIELDS),
            _PAIRED_INFERENCE_AUDIT_DISPLAY_FIELDS,
        ),
    )


def _execution_summary_display_tables(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, list[dict[str, Any]], tuple[str, ...]], ...]:
    """Split the lossless execution summary into focused human views."""

    schemas = (
        (
            "Study identity and terminal state",
            (
                "study_id",
                "study_status",
                "report_scope",
                "source_manifest_status",
                "study_terminal_event_present",
                "last_event",
                "last_event_timestamp_utc",
                "limitations",
            ),
        ),
        (
            "Formal-evidence eligibility",
            (
                "study_id",
                "formal_result_available",
                "ranking_eligible",
                "inference_eligible",
                "selection_eligible",
            ),
        ),
        (
            "Declared case and reporter inventory",
            (
                "study_id",
                "planned_case_count",
                "complete_case_count",
                "incomplete_case_count",
                "complete_case_ids",
                "planned_component_row_count",
                "planned_reporter_profile_count",
                "declared_classifier_count",
            ),
        ),
        (
            "Declared comparisons and progress evidence",
            (
                "study_id",
                "declared_pairwise_comparison_count",
                "progress_log_count",
                "malformed_progress_line_count",
            ),
        ),
        (
            "Fold-cell execution counts",
            (
                "study_id",
                "planned_cell_count",
                "passed_cell_count",
                "failed_closed_cell_count",
                "started_without_terminal_event_cell_count",
                "not_started_cell_count",
            ),
        ),
    )
    return tuple(
        (title, _project_unique_rows(rows, fields), fields) for title, fields in schemas
    )


def _component_display_tables(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, list[dict[str, Any]], tuple[str, ...]], ...]:
    """Keep model/module identity first while separating component concerns."""

    schemas = (
        (
            "Participation, state, and reporter binding",
            (
                "module_id",
                "component_role",
                "participating_cases",
                "execution_state",
                "configured_state",
                "reporter_profile_id",
                "model_reporter_extension_id",
            ),
        ),
        (
            "Input data and fixed parameters",
            (
                "module_id",
                "component_role",
                "input_data",
                "fixed_parameters",
                "evidence_source",
                "inventory_reason",
            ),
        ),
        (
            "Algorithm kernel and literature",
            (
                "module_id",
                "component_role",
                "algorithm_kernel_description",
                "algorithm_references",
                "reporter_binding_kind",
                "reporter_binding_source",
            ),
        ),
        (
            "Execution-evidence interpretation",
            (
                "module_id",
                "participating_cases",
                "execution_state",
                "execution_state_basis",
                "evidence_source",
                "inventory_reason",
            ),
        ),
    )
    return tuple(
        (title, _project_unique_rows(rows, fields), fields) for title, fields in schemas
    )


def _reporter_profile_display_tables(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, list[dict[str, Any]], tuple[str, ...]], ...]:
    """Split planned reporter profiles without changing their audit export."""

    schemas = (
        (
            "Profile identity and planned components",
            (
                "profile_id",
                "title",
                "profile_kind",
                "planned_components",
                "execution_states",
                "execution_claim",
            ),
        ),
        (
            "Required outputs",
            ("profile_id", "required_tables", "required_figures"),
        ),
        (
            "Methods, limitations, and provenance",
            (
                "profile_id",
                "algorithm_summary",
                "statistical_methods",
                "limitations",
                "literature",
                "module_algorithm_references",
                "module_algorithm_summaries",
            ),
        ),
    )
    return tuple(
        (title, _project_unique_rows(rows, fields), fields) for title, fields in schemas
    )


def _incomplete_case_display_tables(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, list[dict[str, Any]], tuple[str, ...]], ...]:
    """Split incomplete-case status, cell counts, and eligibility."""

    schemas = (
        (
            "Case status and terminal evidence",
            (
                "case_id",
                "planned",
                "case_status",
                "case_result_status",
                "case_result_path",
                "last_event",
                "last_timestamp_utc",
                "failure_reason",
            ),
        ),
        (
            "Case fold-cell counts",
            (
                "case_id",
                "expected_cell_count",
                "passed_cell_count",
                "failed_closed_cell_count",
                "started_without_terminal_event_cell_count",
                "not_started_cell_count",
            ),
        ),
        (
            "Case formal-evidence eligibility",
            (
                "case_id",
                "formal_result_available",
                "ranking_eligible",
                "inference_eligible",
                "selection_eligible",
            ),
        ),
    )
    return tuple(
        (title, _project_unique_rows(rows, fields), fields) for title, fields in schemas
    )


def _failure_event_display_tables(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, list[dict[str, Any]], tuple[str, ...]], ...]:
    """Split failure location from its source/message payload."""

    schemas = (
        (
            "Failure event identities",
            (
                "case_id",
                "event",
                "classification",
                "timestamp_utc",
                "repeat_index",
                "fold_index",
                "line_number",
            ),
        ),
        (
            "Failure sources and messages",
            (
                "case_id",
                "source_progress_path",
                "event",
                "message",
                "classification",
            ),
        ),
    )
    return tuple(
        (title, _project_unique_rows(rows, fields), fields) for title, fields in schemas
    )


def _resolved_table_fields(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None
) -> list[str]:
    """Resolve an explicit projection strictly, or infer columns from rows."""

    if fields is not None:
        return list(dict.fromkeys(str(value) for value in fields))
    return list(dict.fromkeys(str(key) for row in rows for key in row))


def _markdown_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] | None = None,
) -> str:
    resolved_fields = _resolved_table_fields(rows, fields)
    if len(resolved_fields) > _MAX_HUMAN_TABLE_COLUMNS:
        raise ValueError(
            f"human-facing incomplete-study table has {len(resolved_fields)} columns; "
            f"maximum is {_MAX_HUMAN_TABLE_COLUMNS}"
        )
    if not resolved_fields:
        return "N/A — no rows."
    output = [
        "| " + " | ".join(resolved_fields) + " |",
        "|" + "|".join("---" for _ in resolved_fields) + "|",
    ]
    for row in rows:
        rendered = {
            field: "N/A" if row.get(field) is None else str(row.get(field, ""))
            for field in resolved_fields
        }
        output.append(
            "| "
            + " | ".join(
                rendered[field].replace("|", r"\|").replace("\n", " ")
                for field in resolved_fields
            )
            + " |"
        )
    output.extend(("", markdown_column_definitions_block(resolved_fields)))
    return "\n".join(output)


def _html_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] | None = None,
) -> str:
    resolved_fields = _resolved_table_fields(rows, fields)
    if len(resolved_fields) > _MAX_HUMAN_TABLE_COLUMNS:
        raise ValueError(
            f"human-facing incomplete-study table has {len(resolved_fields)} columns; "
            f"maximum is {_MAX_HUMAN_TABLE_COLUMNS}"
        )
    if not resolved_fields:
        return "<p>N/A — no rows.</p>"
    header = "".join(f"<th>{html_escape(field)}</th>" for field in resolved_fields)
    body = "".join(
        "<tr>"
        + "".join(
            "<td>"
            + html_escape("N/A" if row.get(field) is None else str(row.get(field, "")))
            + "</td>"
            for field in resolved_fields
        )
        + "</tr>"
        for row in rows
    )
    return (
        f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
        + html_column_definitions_block(resolved_fields)
    )


def _markdown_display_tables(
    tables: Sequence[tuple[str, Sequence[Mapping[str, Any]], Sequence[str]]],
) -> str:
    """Render named semantic sub-tables for the Markdown report."""

    return "\n\n".join(
        f"### {title}\n\n{_markdown_table(rows, fields=fields)}"
        for title, rows, fields in tables
    )


def _html_display_tables(
    tables: Sequence[tuple[str, Sequence[Mapping[str, Any]], Sequence[str]]],
) -> str:
    """Render the same semantic sub-tables for the HTML report."""

    return "".join(
        f"<h3>{html_escape(title)}</h3>" + _html_table(rows, fields=fields)
        for title, rows, fields in tables
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_audit(root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    events, malformed = _load_events(root)
    case_results = _load_case_results(root)
    declared_ids = _plan_case_ids(plan)
    discovered_ids = _discovered_case_ids(root, events, case_results)
    case_ids = list(dict.fromkeys((*declared_ids, *discovered_ids)))
    expected_per_case = _planned_cells_per_case(plan)
    event_case_total = max(
        (
            int(row.get("total", 0) or 0)
            for row in events
            if str(row.get("event")) == "study_started"
        ),
        default=0,
    )
    planned_case_count = max(len(declared_ids), event_case_total, len(case_ids))
    while len(case_ids) < planned_case_count:
        case_ids.append(f"__unresolved_planned_case_{len(case_ids) + 1:03d}")

    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    case_events: dict[str, list[Mapping[str, Any]]] = {
        case_id: [] for case_id in case_ids
    }
    failure_rows = list(malformed)
    for event in events:
        case_id = _event_case_id(event)
        if case_id is not None:
            case_events.setdefault(case_id, []).append(event)
        key = _cell_key(event)
        name = str(event.get("event", event.get("stage", "")))
        if key is not None and name == "cell_start":
            cells.setdefault(key, {})["start"] = event
        if key is not None and name == "cell_complete":
            failure = _failure_message(event)
            cells.setdefault(key, {}).update(
                {"complete": event, "status": "failed_closed" if failure else "passed"}
            )
            if failure:
                failure_rows.append(_failure_row(event, "failed_closed_cell"))
        elif name != "cell_complete" and (failure := _failure_message(event)):
            classification = "study_failure" if case_id is None else "case_failure"
            failure_rows.append(_failure_row(event, classification))

    for case_id, result in case_results.items():
        result_status = str(result.get("status", "unknown")).lower()
        if result_status not in _PASS_STATUSES:
            failure_rows.append(
                {
                    "source_progress_path": result.get("path", ""),
                    "line_number": None,
                    "timestamp_utc": "",
                    "event": "case_result_failure",
                    "case_id": case_id,
                    "repeat_index": None,
                    "fold_index": None,
                    "message": result.get("error") or f"status={result_status}",
                    "classification": "case_result_failure",
                }
            )

    for case_id, result in case_results.items():
        for repeat, fold, status in result["cells"]:
            cells.setdefault((case_id, repeat, fold), {}).update(
                {
                    "status": "passed"
                    if status.lower() in _PASS_STATUSES
                    else "failed_closed",
                    "case_result_path": result["path"],
                }
            )

    event_expected: dict[str, int] = {}
    for case_id, values in case_events.items():
        totals = [
            int(row.get("total", 0) or 0)
            for row in values
            if str(row.get("event")) in {"run_start", "cell_start", "cell_complete"}
        ]
        event_expected[case_id] = max(totals, default=0)

    case_rows: list[dict[str, Any]] = []
    complete_case_ids: list[str] = []
    for case_id in case_ids:
        identities = {key: value for key, value in cells.items() if key[0] == case_id}
        passed = sum(row.get("status") == "passed" for row in identities.values())
        failed = sum(
            row.get("status") == "failed_closed" for row in identities.values()
        )
        started = [
            (key, row)
            for key, row in identities.items()
            if row.get("start") is not None and row.get("status") is None
        ]
        expected = expected_per_case or event_expected.get(case_id) or len(identities)
        not_started = max(0, expected - passed - failed - len(started))
        result = case_results.get(case_id, {})
        result_status = str(result.get("status", "not_available"))
        explicit_pass = result_status.lower() in _PASS_STATUSES or any(
            str(row.get("event")) == "case_finished"
            and str(row.get("message", "")).lower() in _PASS_STATUSES
            for row in case_events.get(case_id, ())
        )
        complete = bool(
            explicit_pass and expected and passed == expected and failed == 0
        )
        if complete:
            status = "complete"
            complete_case_ids.append(case_id)
        elif failed:
            status = "partial_failed"
        elif passed or started:
            status = "partial_interrupted"
        elif result_status not in {"not_available", *tuple(_PASS_STATUSES)}:
            status = "failed"
        else:
            status = "not_started"
        related_failures = [
            str(row["message"])
            for row in failure_rows
            if str(row.get("case_id", "")) == case_id and str(row.get("message", ""))
        ]
        if related_failures:
            reason = "; ".join(dict.fromkeys(related_failures))
        elif status == "not_started":
            reason = "not_started_before_study_termination"
        elif not complete:
            reason = "process_terminated_without_terminal_event"
        else:
            reason = ""
        values = case_events.get(case_id, ())
        last = values[-1] if values else {}
        case_rows.append(
            {
                "case_id": case_id,
                "planned": case_id in declared_ids
                or case_id.startswith("__unresolved"),
                "case_status": status,
                "expected_cell_count": expected,
                "passed_cell_count": passed,
                "failed_closed_cell_count": failed,
                "started_without_terminal_event_cell_count": len(started),
                "not_started_cell_count": not_started,
                "case_result_status": result_status,
                "case_result_path": result.get("path", ""),
                "last_event": last.get("event", ""),
                "last_timestamp_utc": last.get("timestamp_utc", ""),
                "failure_reason": reason,
                "formal_result_available": False,
                "ranking_eligible": False,
                "inference_eligible": False,
                "selection_eligible": False,
            }
        )
        for key, cell in started:
            start = cell["start"]
            failure_rows.append(
                {
                    **_failure_row(start, "started_without_terminal_event"),
                    "message": "process_terminated_without_cell_terminal_event",
                }
            )

    terminal_events = [
        row for row in events if str(row.get("event")) in _STUDY_TERMINAL_EVENTS
    ]
    if not terminal_events:
        last = events[-1] if events else {}
        failure_rows.append(
            {
                "source_progress_path": last.get("_source_progress_path", ""),
                "line_number": last.get("_line_number"),
                "timestamp_utc": last.get("timestamp_utc", ""),
                "event": "missing_study_terminal_event",
                "case_id": last.get("case_id", ""),
                "repeat_index": last.get("repeat", last.get("repeat_index")),
                "fold_index": last.get("fold", last.get("fold_index")),
                "message": "process_terminated_without_study_terminal_event",
                "classification": "study_interruption",
            }
        )

    incomplete_rows = [row for row in case_rows if row["case_status"] != "complete"]
    passed_cells = sum(int(row["passed_cell_count"]) for row in case_rows)
    failed_cells = sum(int(row["failed_closed_cell_count"]) for row in case_rows)
    started_cells = sum(
        int(row["started_without_terminal_event_cell_count"]) for row in case_rows
    )
    not_started_cells = sum(int(row["not_started_cell_count"]) for row in case_rows)
    planned_cells = sum(int(row["expected_cell_count"]) for row in case_rows)
    if failed_cells and started_cells:
        study_status = "incomplete_failed_and_interrupted"
    elif failed_cells:
        study_status = "incomplete_failed_unfinalized"
    elif started_cells or not terminal_events:
        study_status = "incomplete_interrupted"
    else:
        study_status = "incomplete_unfinalized"
    study = plan.get("study") if isinstance(plan.get("study"), Mapping) else {}
    study_id = str(study.get("study_id") or plan.get("study_id") or root.name)
    last_event = events[-1] if events else {}
    summary = {
        "schema_version": "ppg_frailty.incomplete_study_execution_audit.v1",
        "study_id": study_id,
        "study_status": study_status,
        "report_scope": "execution_audit_only",
        "source_manifest_status": "absent_unfinalized",
        "formal_result_available": False,
        "ranking_eligible": False,
        "inference_eligible": False,
        "selection_eligible": False,
        "planned_case_count": planned_case_count,
        "complete_case_count": len(complete_case_ids),
        "incomplete_case_count": len(incomplete_rows),
        "complete_case_ids": complete_case_ids,
        "planned_cell_count": planned_cells,
        "passed_cell_count": passed_cells,
        "failed_closed_cell_count": failed_cells,
        "started_without_terminal_event_cell_count": started_cells,
        "not_started_cell_count": not_started_cells,
        "study_terminal_event_present": bool(terminal_events),
        "progress_log_count": len(_progress_logs(root)),
        "malformed_progress_line_count": len(malformed),
        "last_event": last_event.get("event", ""),
        "last_event_timestamp_utc": last_event.get("timestamp_utc", ""),
        "limitations": (
            "No formal root study_manifest.json exists. Execution fragments are "
            "not converted into performance, ranking, inference, or selection evidence."
        ),
    }
    return {
        "summary": summary,
        "case_rows": case_rows,
        "incomplete_cases": incomplete_rows,
        "failure_events": failure_rows,
        "events": events,
        "case_results": case_results,
    }


def _methods_markdown(
    summary: Mapping[str, Any],
    component_rows: Sequence[Mapping[str, Any]],
    reporter_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Incomplete-study reporter methods",
        "",
        "This is a fail-closed execution audit, not a model-performance report.",
        "",
        "- Inputs: the persisted root `study_plan.yaml`, recursive "
        "`progress_events.jsonl` files, case-result status/cell identities, and "
        "persisted resolved case configs when present.",
        "- Unit of execution counting: unique `(case_id, repeat, fold)` identities.",
        "- A non-empty failure/error/exception/killed/abort status or message is "
        "classified as failed-closed; a started cell without a terminal cell event "
        "is classified as interrupted.",
        "- Missing root `study_manifest.json` remains missing. The reporter never "
        "synthesizes or repairs a manifest.",
        "- Planned component rows use case-level states (`planned`, `not_started`, "
        "`partial`, or `complete`). They do not independently claim that every listed "
        "module executed. `configured_state` preserves the declared switch state.",
        "- Resolved configs supply actual persisted input descriptions and fixed "
        "parameters. When absent, input data are explicitly N/A and plan declarations "
        "are retained without inventing dataset paths or runtime values.",
        "- Predictions, model scores, class metrics, confidence intervals, rankings, "
        "and P values are not read or calculated.",
        "- Every classifier identity declared by the persisted plan is nevertheless "
        "expanded into the three canonical frailty classes in "
        "`classifier_per_class_results`; confusion counts and metrics are explicit "
        "N/A/NULL. Zero observation counts mean no formal OOF rows were admitted; they "
        "are audit counts, never zero-valued surrogate performance results.",
        "- Every explicitly declared reference/candidate pair is expanded over every "
        "planned repeat in `pairwise_repeat_metric_deltas`, and over BA, macro-F1 and "
        "macro ROC-AUC in `paired_participant_inference`. All estimates, differences, "
        "CIs and P values remain N/A/NULL because matched participant OOF evidence is "
        "not formally available.",
        "- Registered participant-cluster CI semantics (not executed here): resample "
        "participant IDs with replacement within true-class strata, carry all OOF "
        "repeat rows for every sampled participant, recompute the metric in each "
        "repeat, average repeats equally, and take the 2.5th/97.5th percentiles. A "
        "paired CI must apply the identical sampled-participant multiplicities to "
        "reference and candidate. Such a CI is conditional on this participant "
        "dataset, frozen splits and persisted predictions; it does not include "
        "dataset-shift or model-selection uncertainty.",
        "- A completed case inside an incomplete multi-case study remains execution "
        "evidence only and is not promoted into a cross-case conclusion.",
        "",
        f"Formal result available: **{str(summary['formal_result_available']).lower()}**.",
        "Ranking/inference/selection eligible: **false/false/false**.",
        "",
        f"Planned component rows inventoried: **{len(component_rows)}**.",
        "",
        "## Planned model/module reporter profiles",
        "",
        "The following algorithms, literature bindings and reporter requirements come "
        "from the planned component identities. Their execution states remain explicit; "
        "this section is not performance evidence.",
        "",
    ]
    if not reporter_rows:
        lines.extend(
            [
                "N/A — no reporter profile could be resolved from persisted planned identities.",
                "",
            ]
        )
    for row in reporter_rows:
        references = list(
            dict.fromkeys(
                [
                    *[
                        str(value)
                        for value in row.get("module_algorithm_references", ())
                    ],
                    *[str(value) for value in row.get("literature", ())],
                ]
            )
        )
        lines.extend(
            [
                f"### {row['title']} (`{row['profile_id']}`)",
                "",
                str(row["algorithm_summary"]),
                "",
                "Planned components: "
                + ", ".join(f"`{value}`" for value in row["planned_components"]),
                "",
                "Execution states: "
                + ", ".join(f"`{value}`" for value in row["execution_states"]),
                "",
                "Module/model algorithm bindings:",
                "",
                *(
                    [
                        f"- {value}"
                        for value in row.get("module_algorithm_summaries", ())
                    ]
                    or ["- N/A — no separate module algorithm summary is available."]
                ),
                "",
                "Algorithm/literature provenance:",
                "",
                *(
                    [f"- {value}" for value in references]
                    if references
                    else ["- N/A — no separate literature source is declared."]
                ),
                "",
            ]
        )
    lines.extend(
        [
            "Limitation: when the progress log has no explicit process-level exception or "
            "terminal event, the reporter can identify interruption but cannot infer why "
            "the process ended.",
            "",
        ]
    )
    return "\n".join(lines)


def _interpretation_markdown(
    summary: Mapping[str, Any], incomplete_rows: Sequence[Mapping[str, Any]]
) -> str:
    failed = [row for row in incomplete_rows if int(row["failed_closed_cell_count"])]
    return "\n".join(
        (
            "# Incomplete-study result interpretation",
            "",
            f"Study status: **{summary['study_status']}**.",
            "",
            f"The plan declared {summary['planned_case_count']} cases and "
            f"{summary['planned_cell_count']} fold cells. "
            f"{summary['complete_case_count']} cases completed; "
            f"{summary['incomplete_case_count']} did not.",
            "",
            f"Observed execution: {summary['passed_cell_count']} passed cells, "
            f"{summary['failed_closed_cell_count']} failed-closed cells, "
            f"{summary['started_without_terminal_event_cell_count']} started cells without "
            f"a terminal event, and {summary['not_started_cell_count']} unstarted cells.",
            "",
            "No formal model/module comparison exists for this directory. Declared "
            "classifier pairs are retained only as explicit N/A audit rows; no "
            "performance ranking, confidence interval, P value, winner, or selected "
            "configuration is reported.",
            "",
            (
                "Cases with explicit failed-closed cells: "
                + ", ".join(str(row["case_id"]) for row in failed)
                + "."
                if failed
                else "No explicit failed-closed cell was recorded; the evidence only "
                "establishes interruption/unfinalized execution."
            ),
            "",
        )
    )


def generate_incomplete_study_report(
    study_directory: str | Path,
) -> IncompleteStudyReportResult:
    """Build an execution-only report for a study lacking its root manifest."""

    root = Path(study_directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if (root / "study_manifest.json").exists():
        raise ValueError(
            "incomplete reporter refuses a study with a root manifest; use its formal reporter"
        )
    plan_path = root / "study_plan.yaml"
    if not plan_path.is_file():
        raise FileNotFoundError("incomplete reporting requires root study_plan.yaml")
    plan = _mapping(yaml.safe_load(plan_path.read_text(encoding="utf-8")), "study plan")
    audit = _build_audit(root, plan)
    summary = audit["summary"]
    case_rows = audit["case_rows"]
    incomplete_rows = audit["incomplete_cases"]
    failure_rows = audit["failure_events"]
    summary_rows = [summary]
    component_rows, component_input_paths = _build_planned_component_inventory(
        root, plan, case_rows
    )
    reporter_rows = _planned_reporter_profile_rows(component_rows)
    (
        classifier_per_class_rows,
        pairwise_repeat_rows,
        paired_inference_rows,
    ) = _incomplete_classifier_tables(plan, case_rows, component_rows)
    summary["planned_component_row_count"] = len(component_rows)
    summary["planned_reporter_profile_count"] = len(reporter_rows)
    summary["declared_classifier_count"] = len(
        {str(row["classifier_id"]) for row in classifier_per_class_rows}
    )
    summary["declared_pairwise_comparison_count"] = len(
        {
            (str(row["comparison_family"]), str(row["comparison_id"]))
            for row in paired_inference_rows
        }
    )

    tables = root / "tables"
    tables.mkdir(exist_ok=True)
    table_specs = (
        ("execution_completeness", summary_rows, None),
        ("incomplete_cases", incomplete_rows, None),
        ("failure_events", failure_rows, None),
        ("test_components", component_rows, None),
        ("reporter_profiles", reporter_rows, None),
        (
            "classifier_per_class_results",
            classifier_per_class_rows,
            _PER_CLASS_FIELDS,
        ),
        (
            "pairwise_repeat_metric_deltas",
            pairwise_repeat_rows,
            _PAIRWISE_REPEAT_FIELDS,
        ),
        (
            "paired_participant_inference",
            paired_inference_rows,
            _PAIRED_INFERENCE_FIELDS,
        ),
    )
    generated: list[tuple[Path, str]] = []
    for name, rows, fields in table_specs:
        generated.append(
            (
                _atomic_csv(tables / f"{name}.csv", rows, fields=fields),
                "table_csv",
            )
        )
        generated.append((_atomic_json(tables / f"{name}.json", rows), "table_json"))
    # The incomplete reporter may coexist with runner-written root CSVs. Every
    # root table CSV receives one workbook sheet, not only tables created here.
    csv_paths = sorted(tables.glob("*.csv"), key=lambda path: path.name)
    generated_paths = {path.resolve() for path, _role in generated}
    generated.extend(
        (path, "retained_root_table_csv")
        for path in csv_paths
        if path.resolve() not in generated_paths
    )
    for definition_path in write_table_column_definitions(
        tables,
        csv_directory=tables,
    ):
        generated.append((definition_path, "table_column_definitions"))
    workbook_tmp = tables / ".report_tables.incomplete-report.tmp.xlsx"
    write_excel_workbook_from_csv_directory(workbook_tmp, tables)
    workbook = tables / "report_tables.xlsx"
    workbook_tmp.replace(workbook)
    generated.append((workbook, "table_workbook"))

    execution_summary_display_tables = _execution_summary_display_tables(summary_rows)
    component_display_tables = _component_display_tables(component_rows)
    reporter_profile_display_tables = _reporter_profile_display_tables(reporter_rows)
    incomplete_case_display_tables = _incomplete_case_display_tables(incomplete_rows)
    failure_event_display_tables = _failure_event_display_tables(failure_rows)
    component_table = _markdown_display_tables(component_display_tables)
    test_components = _atomic_text(
        root / "TEST_COMPONENTS.md",
        "# Planned test models, modules, inputs, and fixed parameters\n\n"
        "This fail-closed inventory is reconstructed only from persisted study-plan, "
        "resolved-config, case-result, and progress evidence. `execution_state` is a "
        "case-level state and does not independently claim that each component ran. "
        "Missing input/config evidence is explicitly N/A.\n\n" + component_table + "\n",
    )
    methods = _atomic_text(
        root / "REPORT_METHODS.md",
        _methods_markdown(summary, component_rows, reporter_rows),
    )
    interpretation = _atomic_text(
        root / "RESULT_INTERPRETATION.md",
        _interpretation_markdown(summary, incomplete_rows),
    )
    generated.extend(
        (
            (test_components, "planned_component_inventory_markdown"),
            (methods, "methods"),
            (interpretation, "interpretation"),
        )
    )

    flags = [
        {
            "formal_result_available": False,
            "ranking_eligible": False,
            "inference_eligible": False,
            "selection_eligible": False,
            "source_manifest_status": "absent_unfinalized",
        }
    ]
    per_class_display_tables = _per_class_display_tables(classifier_per_class_rows)
    pairwise_repeat_display_tables = _pairwise_repeat_display_tables(
        pairwise_repeat_rows
    )
    paired_inference_display_tables = _paired_inference_display_tables(
        paired_inference_rows
    )
    markdown = "\n".join(
        (
            f"# {summary['study_id']}",
            "",
            f"Status: **{summary['study_status']}**",
            "",
            "> Fail-closed execution audit only. This directory has no formal root "
            "`study_manifest.json`; no performance, ranking, inference, or selection "
            "result is produced.",
            "",
            "## Formal-evidence eligibility",
            "",
            _markdown_table(flags),
            "",
            "## Execution completeness",
            "",
            _markdown_display_tables(execution_summary_display_tables),
            "",
            "## Planned models/modules, inputs, and fixed parameters",
            "",
            "The identical inventory is available in [TEST_COMPONENTS.md](TEST_COMPONENTS.md) "
            "and machine-readable CSV/JSON. States are case-level evidence only and never "
            "an inferred component-execution claim.",
            "",
            component_table,
            "",
            "## Planned reporter profiles and provenance",
            "",
            _markdown_display_tables(reporter_profile_display_tables),
            "",
            "## Classifier per-class results (explicit N/A)",
            "",
            "The roster is expanded to all three canonical frailty classes. Numeric "
            "cells are unavailable because this directory has no formal root manifest; "
            "zero observations are audit counts, not performance estimates.",
            "",
            _markdown_display_tables(per_class_display_tables),
            "",
            "## Pairwise per-repeat metric differences (explicit N/A)",
            "",
            (
                "Each plan-declared pair is shown for every planned repeat. No pair is "
                "invented when the plan declares no reference/comparison contract."
            ),
            "",
            _markdown_display_tables(pairwise_repeat_display_tables),
            "",
            "## Paired participant-cluster inference (explicit N/A)",
            "",
            "Each declared pair has separate BA, macro-F1 and macro ROC-AUC rows. "
            "The participant-cluster estimates and CIs remain NULL because no eligible "
            "matched participant OOF contract exists in this incomplete study.",
            "",
            _markdown_display_tables(paired_inference_display_tables),
            "",
            "## Incomplete cases",
            "",
            _markdown_display_tables(incomplete_case_display_tables),
            "",
            "## Failure and interruption events",
            "",
            _markdown_display_tables(failure_event_display_tables),
            "",
            "## Interpretation",
            "",
            "See [RESULT_INTERPRETATION.md](RESULT_INTERPRETATION.md).",
            "",
            "## Methods",
            "",
            "See [REPORT_METHODS.md](REPORT_METHODS.md).",
            "",
        )
    )
    summary_markdown = _atomic_text(root / "STUDY_SUMMARY.md", markdown)
    html = (
        "<!doctype html><meta charset='utf-8'><title>"
        + html_escape(str(summary["study_id"]))
        + "</title><style>body{font-family:sans-serif;max-width:1200px;margin:auto}"
        "table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #bbb;"
        "padding:4px;vertical-align:top}blockquote{border-left:4px solid #b00;"
        "padding-left:1em}</style><h1>"
        + html_escape(str(summary["study_id"]))
        + "</h1><p>Status: <strong>"
        + html_escape(str(summary["study_status"]))
        + "</strong></p><blockquote>Fail-closed execution audit only. No formal "
        "performance, ranking, inference, or selection result exists.</blockquote>"
        "<h2>Formal-evidence eligibility</h2>"
        + _html_table(flags)
        + "<h2>Execution completeness</h2>"
        + _html_display_tables(execution_summary_display_tables)
        + "<h2>Planned models/modules, inputs, and fixed parameters</h2>"
        + "<p>Case-level states do not independently claim component execution.</p>"
        + _html_display_tables(component_display_tables)
        + "<h2>Planned reporter profiles and provenance</h2>"
        + _html_display_tables(reporter_profile_display_tables)
        + "<h2>Classifier per-class results (explicit N/A)</h2>"
        + "<p>The declared classifier roster is expanded to all three canonical "
        "frailty classes; no performance value is inferred.</p>"
        + _html_display_tables(per_class_display_tables)
        + "<h2>Pairwise per-repeat metric differences (explicit N/A)</h2>"
        + "<p>Only plan-declared pairs are expanded over planned repeats.</p>"
        + _html_display_tables(pairwise_repeat_display_tables)
        + "<h2>Paired participant-cluster inference (explicit N/A)</h2>"
        + "<p>BA, macro-F1 and macro ROC-AUC inference remains unavailable without "
        "eligible matched participant OOF evidence.</p>"
        + _html_display_tables(paired_inference_display_tables)
        + "<h2>Incomplete cases</h2>"
        + _html_display_tables(incomplete_case_display_tables)
        + "<h2>Failure and interruption events</h2>"
        + _html_display_tables(failure_event_display_tables)
        + "<p>See REPORT_METHODS.md and RESULT_INTERPRETATION.md.</p>"
    )
    summary_html = _atomic_text(root / "STUDY_SUMMARY.html", html)
    generated.extend(
        ((summary_markdown, "summary_markdown"), (summary_html, "summary_html"))
    )

    input_paths: list[Path] = [plan_path, *_progress_logs(root)]
    input_paths.extend(
        root / str(row["path"])
        for row in audit["case_results"].values()
        if row.get("path")
    )
    input_paths.extend(component_input_paths)
    index_path = root / "outputs_index.json"
    indexed_paths: set[Path] = set()
    entries: list[dict[str, Any]] = []
    for path, role in [
        *((path, "input_evidence") for path in dict.fromkeys(input_paths)),
        *generated,
    ]:
        resolved_path = path.resolve()
        if resolved_path in indexed_paths:
            continue
        indexed_paths.add(resolved_path)
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    entries.append(
        {
            "path": index_path.name,
            "role": "output_index",
            "bytes": None,
            "sha256": None,
            "self_hash_policy": "omitted_to_avoid_recursive_self_hash",
        }
    )
    _atomic_json(
        index_path,
        {
            "schema_version": "ppg_frailty.incomplete_study_output_index.v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "study_directory": str(root),
            "inventory_scope": "incomplete_report_inputs_and_outputs_only",
            "artifacts": entries,
        },
    )
    return IncompleteStudyReportResult(
        study_directory=root,
        summary_markdown=summary_markdown,
        summary_html=summary_html,
        methods_markdown=methods,
        interpretation_markdown=interpretation,
        outputs_index=index_path,
        status=str(summary["study_status"]),
        table_count=len(csv_paths),
    )


__all__ = [
    "IncompleteStudyReportResult",
    "generate_incomplete_study_report",
    "is_incomplete_study_directory",
]

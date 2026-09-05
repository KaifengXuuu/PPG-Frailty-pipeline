"""Write complete, human-readable study reports and an output inventory."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .analyze import StudyAnalysis, analyze_study
from .collect import CollectedStudy, collect_study
from .components import (
    TEST_COMPONENT_VIEW_SCHEMAS,
    TOP_MODEL_CONFIGURATION_COLUMNS,
    build_pipeline_test_component_rows,
    build_top_model_configuration_rows,
    markdown_test_component_table,
    write_test_component_markdown,
)
from .conclusions import (
    DEFAULT_REPORTING_RANDOM_SEED,
    classification_comparison_rows,
    classification_conclusion_rows,
    holm_adjust_paired_inference_rows,
    paired_inference_against_reference,
    paired_repeat_deltas_against_reference,
    write_result_interpretation,
)
from .plots import (
    FIGURE_TABLE_SOURCES,
    clear_static_figure_artifacts,
    generate_static_figures,
)
from .reproducibility import (
    NOT_VERIFIABLE,
    ReproducibilityAudit,
    audit_study_reproducibility,
)
from .profiles import (
    REPORTER_PROFILE_VIEW_SCHEMAS,
    markdown_reporter_profile_tables,
    reporter_profile_rows,
    required_figure_modules,
    write_reporter_methods,
)
from .tabular import (
    ReportTable,
    compact_rows,
    format_mean_sd,
    html_column_definitions_block,
    markdown_column_definitions_block,
    write_csv,
    write_excel_workbook_from_csv_directory,
    write_table_column_definitions,
)


_MAX_HUMAN_FACING_REPORT_COLUMNS = 8


class _ReportJSONEncoder(json.JSONEncoder):
    """Encode report rows incrementally without cloning the whole payload."""

    def default(self, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            return value.item()
        return str(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = _ReportJSONEncoder(
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            for chunk in encoder.iterencode(value):
                stream.write(chunk)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        write_csv(temporary, rows)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if value != 0.0 and abs(value) < 1e-4:
            return f"{value:.3e}"
        return f"{value:.4f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", r"\|")


def _markdown_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[
        tuple[str | tuple[str, str] | tuple[str, str, bool], str]
    ],
) -> list[str]:
    if len(columns) > _MAX_HUMAN_FACING_REPORT_COLUMNS:
        raise ValueError(
            f"human-facing Markdown table has {len(columns)} columns; "
            f"maximum is {_MAX_HUMAN_FACING_REPORT_COLUMNS}"
        )
    definition_columns = [field for field, _label in columns]
    definition_labels = [label for _field, label in columns]
    definition_block = markdown_column_definitions_block(
        definition_columns,
        display_labels=definition_labels,
    )
    if not rows:
        return ["N/A — no rows were available.", "", definition_block, ""]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    format_mean_sd(
                        row.get(field[0]),
                        row.get(field[1]),
                        percent=True if len(field) == 2 else field[2],
                    )
                    if isinstance(field, tuple)
                    else _fmt(row.get(field))
                )
                for field, _ in columns
            )
            + " |"
        )
    lines.extend(("", definition_block, ""))
    return lines


def _study_info(collected: CollectedStudy) -> Mapping[str, Any]:
    value = collected.plan.get("study", {})
    return value if isinstance(value, Mapping) else {}


def _preprocessing_cache_overview(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "cell_layer_rows": len(rows),
        "cell_count": len(
            {
                (row.get("case_id"), row.get("repeat"), row.get("fold"))
                for row in rows
            }
        ),
        "event_count": sum(int(row.get("event_count", 0) or 0) for row in rows),
        "hit_count": sum(int(row.get("hit_count", 0) or 0) for row in rows),
        "write_count": sum(int(row.get("write_count", 0) or 0) for row in rows),
        "bypass_count": sum(int(row.get("bypass_count", 0) or 0) for row in rows),
        "logical_array_bytes": sum(
            int(row.get("logical_array_bytes", 0) or 0) for row in rows
        ),
        "elapsed_seconds": sum(
            float(row.get("elapsed_seconds", 0.0) or 0.0) for row in rows
        ),
    }


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


_LEGACY_BRIDGE_NUMERIC_REPORT_TABLES = (
    (
        "Contrast identity and interpretation",
        (
            ("model", "Model"),
            ("profile", "Profile"),
            ("numeric_profile_order", "Numeric order"),
            ("previous_numeric_profile", "Previous numeric profile"),
            ("numeric_comparison", "Predefined comparison"),
            ("contrast_metrics_available", "Contrast available"),
            ("interpretation", "Interpretation"),
        ),
    ),
    (
        "Absolute balanced accuracy by aggregation",
        (
            ("model", "Model"),
            ("profile", "Profile"),
            ("BA_legacy_aggregation", "BA legacy W"),
            ("BA_line_a_aggregation", "BA Line A"),
            ("BA_v2_aggregation", "BA Line B"),
        ),
    ),
    (
        "Balanced-accuracy adjacent deltas",
        (
            ("model", "Model"),
            ("profile", "Profile"),
            ("numeric_comparison", "Predefined comparison"),
            ("delta_BA_legacy_aggregation", "Δ BA legacy W"),
            ("delta_BA_line_a_aggregation", "Δ BA Line A"),
            ("delta_BA_v2_aggregation", "Δ BA Line B"),
            ("contrast_metrics_available", "Contrast available"),
        ),
    ),
    (
        "Absolute Macro-F1 by aggregation",
        (
            ("model", "Model"),
            ("profile", "Profile"),
            ("macroF1_legacy_aggregation", "Macro-F1 legacy W"),
            ("macroF1_line_a_aggregation", "Macro-F1 Line A"),
            ("macroF1_v2_aggregation", "Macro-F1 Line B"),
            ("worst_class_F1", "Worst-class F1 Line B"),
        ),
    ),
    (
        "Macro-F1 and worst-class adjacent deltas",
        (
            ("model", "Model"),
            ("profile", "Profile"),
            ("numeric_comparison", "Predefined comparison"),
            ("delta_macroF1_legacy_aggregation", "Δ Macro-F1 legacy W"),
            ("delta_macroF1_line_a_aggregation", "Δ Macro-F1 Line A"),
            ("delta_macroF1_v2_aggregation", "Δ Macro-F1 Line B"),
            ("delta_worst_class_F1", "Δ worst-class F1"),
            ("contrast_metrics_available", "Contrast available"),
        ),
    ),
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


_LEGACY_BRIDGE_EXECUTION_REPORT_TABLES = (
    (
        "Execution identity and scheduling interpretation",
        (
            ("model", "Model"),
            ("profile", "Profile"),
            ("execution_order", "Execution order"),
            ("previous_execution_profile", "Previous execution profile"),
            ("execution_transition", "Execution transition"),
            ("execution_transition_is_ablation", "Transition is ablation"),
            ("interpretation", "Interpretation"),
        ),
    ),
    (
        "Execution-order balanced accuracy",
        (
            ("model", "Model"),
            ("profile", "Profile"),
            ("execution_order", "Execution order"),
            ("BA_legacy_aggregation", "BA legacy W"),
            ("BA_line_a_aggregation", "BA Line A"),
            ("BA_v2_aggregation", "BA Line B"),
        ),
    ),
    (
        "Execution-order Macro-F1 and worst-class F1",
        (
            ("model", "Model"),
            ("profile", "Profile"),
            ("execution_order", "Execution order"),
            ("macroF1_legacy_aggregation", "Macro-F1 legacy W"),
            ("macroF1_line_a_aggregation", "Macro-F1 Line A"),
            ("macroF1_v2_aggregation", "Macro-F1 Line B"),
            ("worst_class_F1", "Worst-class F1 Line B"),
        ),
    ),
)


_DENOISER_HR_SUMMARY_COLUMNS = (
    ("case_id", "Case"),
    ("denoiser_id", "Denoiser"),
    ("outer_partition", "Partition"),
    ("role_scope", "Role"),
    ("attempted_record_count", "Attempts"),
    ("successful_reducer_record_count", "Reducer successes"),
    ("reducer_failure_count", "Reducer failures"),
    ("reducer_failure_rate", "Reducer failure rate"),
    ("post_q_rate_pass_count", "Post Q_rate passes"),
    ("post_q_rate_pass_rate", "Post Q_rate pass rate"),
    ("post_q_rate_recovery_eligible_count", "Q_rate recovery eligible"),
    ("post_q_rate_recovery_count", "Q_rate recovered"),
    ("post_q_rate_recovery_rate", "Q_rate recovery rate"),
    ("paired_hr_record_count", "Paired HR records"),
    ("paired_participant_count", "Participants"),
    ("paired_ppi_record_count", "Paired PPI records"),
    ("paired_ppi_participant_count", "PPI participants"),
    (
        (
            "participant_macro_direct_hr_bpm",
            "participant_sd_direct_hr_bpm",
            False,
        ),
        "Direct HR bpm",
    ),
    (
        (
            "participant_macro_post_denoise_hr_bpm",
            "participant_sd_post_denoise_hr_bpm",
            False,
        ),
        "Post-denoise HR bpm",
    ),
    (
        (
            "participant_macro_post_minus_direct_hr_bpm",
            "participant_sd_post_minus_direct_hr_bpm",
            False,
        ),
        "Post − direct bpm",
    ),
    (
        (
            "participant_macro_absolute_hr_change_bpm",
            "participant_sd_absolute_hr_change_bpm",
            False,
        ),
        "Absolute ΔHR bpm",
    ),
    (
        (
            "participant_macro_direct_median_ppi_ms",
            "participant_sd_direct_median_ppi_ms",
            False,
        ),
        "Direct median PPI ms",
    ),
    (
        (
            "participant_macro_post_denoise_median_ppi_ms",
            "participant_sd_post_denoise_median_ppi_ms",
            False,
        ),
        "Post-denoise median PPI ms",
    ),
    (
        (
            "participant_macro_post_minus_direct_ppi_ms",
            "participant_sd_post_minus_direct_ppi_ms",
            False,
        ),
        "Post − direct PPI ms",
    ),
    (
        (
            "participant_macro_ppi_endpoint_error_ms",
            "participant_sd_ppi_endpoint_error_ms",
            False,
        ),
        "Absolute PPI endpoint error ms",
    ),
    ("endpoint_reference", "Endpoint reference"),
)


_DENOISER_HR_RECORD_COLUMNS = (
    ("case_id", "Case"),
    ("repeat", "Repeat"),
    ("fold", "Fold"),
    ("outer_partition", "Partition"),
    ("participant_id", "Participant"),
    ("record_id", "Record"),
    ("role", "Role"),
    ("denoiser_id", "Denoiser"),
    ("denoiser_status", "Status"),
    ("direct_hr_bpm", "Direct HR"),
    ("post_denoise_hr_bpm", "Post HR"),
    ("post_minus_direct_hr_bpm", "ΔHR"),
    ("direct_median_valid_ppi_ms", "Direct median PPI ms"),
    ("post_denoise_median_valid_ppi_ms", "Post median PPI ms"),
    ("post_minus_direct_ppi_ms", "ΔPPI ms"),
    ("absolute_post_minus_direct_ppi_ms", "Absolute ΔPPI ms"),
    ("direct_valid_ppi_count", "Direct PPI n"),
    ("post_denoise_valid_ppi_count", "Post PPI n"),
    ("direct_q_rate_state", "Direct Q_rate"),
    ("post_q_rate_state", "Post Q_rate"),
    ("post_q_rate_recovery_eligible", "Recovery eligible"),
    ("post_q_rate_recovered", "Q_rate recovered"),
    ("reducer_failed", "Reducer failed"),
    ("retained_for_classifier", "CNN retained"),
)


_DENOISER_HR_SUMMARY_REPORT_TABLES = (
    (
        "Denoiser reducer outcomes",
        (
            ("case_id", "Case"),
            ("denoiser_id", "Denoiser"),
            ("outer_partition", "Partition"),
            ("role_scope", "Role"),
            ("attempted_record_count", "Attempts"),
            ("successful_reducer_record_count", "Reducer successes"),
            ("reducer_failure_count", "Reducer failures"),
            ("reducer_failure_rate", "Reducer failure rate"),
        ),
    ),
    (
        "Post-denoiser Q_rate pass",
        (
            ("case_id", "Case"),
            ("denoiser_id", "Denoiser"),
            ("outer_partition", "Partition"),
            ("role_scope", "Role"),
            ("post_q_rate_pass_count", "Post Q_rate passes"),
            ("post_q_rate_pass_rate", "Post Q_rate pass rate"),
        ),
    ),
    (
        "Post-denoiser Q_rate recovery",
        (
            ("case_id", "Case"),
            ("denoiser_id", "Denoiser"),
            ("outer_partition", "Partition"),
            ("role_scope", "Role"),
            ("post_q_rate_recovery_eligible_count", "Recovery eligible"),
            ("post_q_rate_recovery_count", "Q_rate recovered"),
            ("post_q_rate_recovery_rate", "Q_rate recovery rate"),
        ),
    ),
    (
        "Denoiser endpoint-pairing support",
        (
            ("case_id", "Case"),
            ("denoiser_id", "Denoiser"),
            ("outer_partition", "Partition"),
            ("role_scope", "Role"),
            ("paired_hr_record_count", "Paired HR records"),
            ("paired_participant_count", "HR participants"),
            ("paired_ppi_record_count", "Paired PPI records"),
            ("paired_ppi_participant_count", "PPI participants"),
        ),
    ),
    (
        "Denoiser HR endpoints",
        (
            ("case_id", "Case"),
            ("denoiser_id", "Denoiser"),
            ("outer_partition", "Partition"),
            ("role_scope", "Role"),
            (
                (
                    "participant_macro_direct_hr_bpm",
                    "participant_sd_direct_hr_bpm",
                    False,
                ),
                "Direct HR bpm",
            ),
            (
                (
                    "participant_macro_post_denoise_hr_bpm",
                    "participant_sd_post_denoise_hr_bpm",
                    False,
                ),
                "Post-denoise HR bpm",
            ),
            (
                (
                    "participant_macro_post_minus_direct_hr_bpm",
                    "participant_sd_post_minus_direct_hr_bpm",
                    False,
                ),
                "Post − direct bpm",
            ),
            (
                (
                    "participant_macro_absolute_hr_change_bpm",
                    "participant_sd_absolute_hr_change_bpm",
                    False,
                ),
                "Absolute ΔHR bpm",
            ),
        ),
    ),
    (
        "Denoiser PPI endpoints",
        (
            ("case_id", "Case"),
            ("denoiser_id", "Denoiser"),
            ("outer_partition", "Partition"),
            ("role_scope", "Role"),
            (
                (
                    "participant_macro_direct_median_ppi_ms",
                    "participant_sd_direct_median_ppi_ms",
                    False,
                ),
                "Direct median PPI ms",
            ),
            (
                (
                    "participant_macro_post_denoise_median_ppi_ms",
                    "participant_sd_post_denoise_median_ppi_ms",
                    False,
                ),
                "Post-denoise median PPI ms",
            ),
            (
                (
                    "participant_macro_post_minus_direct_ppi_ms",
                    "participant_sd_post_minus_direct_ppi_ms",
                    False,
                ),
                "Post − direct PPI ms",
            ),
            (
                (
                    "participant_macro_ppi_endpoint_error_ms",
                    "participant_sd_ppi_endpoint_error_ms",
                    False,
                ),
                "Absolute PPI endpoint error ms",
            ),
        ),
    ),
    (
        "Denoiser endpoint provenance",
        (
            ("case_id", "Case"),
            ("denoiser_id", "Denoiser"),
            ("outer_partition", "Partition"),
            ("role_scope", "Role"),
            ("endpoint_reference", "Endpoint reference"),
        ),
    ),
)


_DENOISER_HR_RECORD_REPORT_TABLES = (
    (
        "Paired denoiser-record identity",
        (
            ("case_id", "Case"),
            ("repeat", "Repeat"),
            ("fold", "Fold"),
            ("outer_partition", "Partition"),
            ("participant_id", "Participant"),
            ("record_id", "Record"),
            ("role", "Role"),
            ("denoiser_id", "Denoiser"),
        ),
    ),
    (
        "Paired denoiser-record status",
        (
            ("case_id", "Case"),
            ("participant_id", "Participant"),
            ("record_id", "Record"),
            ("denoiser_id", "Denoiser"),
            ("denoiser_status", "Status"),
            ("reducer_failed", "Reducer failed"),
            ("retained_for_classifier", "CNN retained"),
        ),
    ),
    (
        "Paired denoiser-record HR",
        (
            ("case_id", "Case"),
            ("participant_id", "Participant"),
            ("record_id", "Record"),
            ("direct_hr_bpm", "Direct HR"),
            ("post_denoise_hr_bpm", "Post HR"),
            ("post_minus_direct_hr_bpm", "ΔHR"),
            ("direct_valid_ppi_count", "Direct PPI n"),
            ("post_denoise_valid_ppi_count", "Post PPI n"),
        ),
    ),
    (
        "Paired denoiser-record PPI",
        (
            ("case_id", "Case"),
            ("participant_id", "Participant"),
            ("record_id", "Record"),
            ("direct_median_valid_ppi_ms", "Direct median PPI ms"),
            ("post_denoise_median_valid_ppi_ms", "Post median PPI ms"),
            ("post_minus_direct_ppi_ms", "ΔPPI ms"),
            ("absolute_post_minus_direct_ppi_ms", "Absolute ΔPPI ms"),
        ),
    ),
    (
        "Paired denoiser-record Q_rate",
        (
            ("case_id", "Case"),
            ("participant_id", "Participant"),
            ("record_id", "Record"),
            ("direct_q_rate_state", "Direct Q_rate"),
            ("post_q_rate_state", "Post Q_rate"),
            ("post_q_rate_recovery_eligible", "Recovery eligible"),
            ("post_q_rate_recovered", "Q_rate recovered"),
        ),
    ),
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
    (
        "window_oof_probability_max_abs_diff",
        "B0/B7 max absolute probability diff",
    ),
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


_STAGE3_STAR_WITHIN_MODEL_COLUMNS = (
    ("profile", "Profile"),
    ("factor_id", "Factor"),
    ("native_aggregation_view", "Native endpoint"),
    (
        ("native_balanced_accuracy", "native_balanced_accuracy_sd"),
        "BA mean ± SD (%)",
    ),
    (
        ("delta_vs_B0_balanced_accuracy", "delta_vs_B0_balanced_accuracy_sd"),
        "Δ BA vs B0 mean ± SD (pp)",
    ),
    (("native_macro_f1", "native_macro_f1_sd"), "Macro-F1 mean ± SD (%)"),
    (
        ("delta_vs_B0_macro_f1", "delta_vs_B0_macro_f1_sd"),
        "Δ Macro-F1 vs B0 mean ± SD (pp)",
    ),
    (
        ("native_worst_class_f1", "native_worst_class_f1_sd"),
        "Worst-class F1 mean ± SD (%)",
    ),
    (
        ("delta_vs_B0_worst_class_f1", "delta_vs_B0_worst_class_f1_sd"),
        "Δ worst-class F1 vs B0 mean ± SD (pp)",
    ),
    ("repeat_count", "Repeats"),
    ("changed_control_paths", "Changed controls"),
    ("single_factor_audit", "Factor audit"),
    ("contrast_metrics_available", "Available"),
)


_STAGE3_STAR_CROSS_MODEL_COLUMNS = (
    ("profile", "Profile"),
    ("factor_id", "Factor"),
    ("native_aggregation_view", "Native endpoint"),
    (
        ("inception_balanced_accuracy", "inception_balanced_accuracy_sd"),
        "InceptionTime BA mean ± SD (%)",
    ),
    (
        ("cnn_balanced_accuracy", "cnn_balanced_accuracy_sd"),
        "CNN BA mean ± SD (%)",
    ),
    (
        (
            "inception_minus_cnn_balanced_accuracy",
            "inception_minus_cnn_balanced_accuracy_sd",
        ),
        "InceptionTime − CNN Δ BA mean ± SD (pp)",
    ),
    (
        ("inception_macro_f1", "inception_macro_f1_sd"),
        "InceptionTime Macro-F1 mean ± SD (%)",
    ),
    (("cnn_macro_f1", "cnn_macro_f1_sd"), "CNN Macro-F1 mean ± SD (%)"),
    (
        ("inception_minus_cnn_macro_f1", "inception_minus_cnn_macro_f1_sd"),
        "InceptionTime − CNN Δ Macro-F1 mean ± SD (pp)",
    ),
    (
        ("inception_worst_class_f1", "inception_worst_class_f1_sd"),
        "InceptionTime worst-class F1 mean ± SD (%)",
    ),
    (
        ("cnn_worst_class_f1", "cnn_worst_class_f1_sd"),
        "CNN worst-class F1 mean ± SD (%)",
    ),
    (
        (
            "inception_minus_cnn_worst_class_f1",
            "inception_minus_cnn_worst_class_f1_sd",
        ),
        "InceptionTime − CNN Δ worst-class F1 mean ± SD (pp)",
    ),
    ("repeat_count", "Paired repeats"),
    ("cross_model_profile_controls_match", "Controls match"),
    ("comparison_metrics_available", "Available"),
)


_STAGE3_STAR_WITHIN_MODEL_REPORT_TABLES = (
    (
        "Absolute within-model performance",
        (
            ("profile", "Profile"),
            ("factor_id", "Factor"),
            ("native_aggregation_view", "Native endpoint"),
            (
                ("native_balanced_accuracy", "native_balanced_accuracy_sd"),
                "BA mean ± SD (%)",
            ),
            (("native_macro_f1", "native_macro_f1_sd"), "Macro-F1 mean ± SD (%)"),
            (
                ("native_worst_class_f1", "native_worst_class_f1_sd"),
                "Worst-class F1 mean ± SD (%)",
            ),
            ("repeat_count", "Repeats"),
        ),
    ),
    (
        "B0-centred within-model deltas",
        (
            ("profile", "Profile"),
            ("factor_id", "Factor"),
            (
                (
                    "delta_vs_B0_balanced_accuracy",
                    "delta_vs_B0_balanced_accuracy_sd",
                ),
                "Δ BA vs B0 mean ± SD (pp)",
            ),
            (
                ("delta_vs_B0_macro_f1", "delta_vs_B0_macro_f1_sd"),
                "Δ Macro-F1 vs B0 mean ± SD (pp)",
            ),
            (
                (
                    "delta_vs_B0_worst_class_f1",
                    "delta_vs_B0_worst_class_f1_sd",
                ),
                "Δ worst-class F1 vs B0 mean ± SD (pp)",
            ),
            ("repeat_count", "Repeats"),
            ("contrast_metrics_available", "Available"),
        ),
    ),
    (
        "Within-model factor audit",
        (
            ("profile", "Profile"),
            ("factor_id", "Factor"),
            ("changed_control_paths", "Changed controls"),
            ("single_factor_audit", "Factor audit"),
            ("contrast_metrics_available", "Available"),
        ),
    ),
)


_STAGE3_STAR_CROSS_MODEL_REPORT_TABLES = (
    (
        "Cross-model balanced accuracy",
        (
            ("profile", "Profile"),
            ("factor_id", "Factor"),
            ("native_aggregation_view", "Native endpoint"),
            (
                ("inception_balanced_accuracy", "inception_balanced_accuracy_sd"),
                "InceptionTime BA mean ± SD (%)",
            ),
            (
                ("cnn_balanced_accuracy", "cnn_balanced_accuracy_sd"),
                "CNN BA mean ± SD (%)",
            ),
            (
                (
                    "inception_minus_cnn_balanced_accuracy",
                    "inception_minus_cnn_balanced_accuracy_sd",
                ),
                "InceptionTime − CNN Δ BA mean ± SD (pp)",
            ),
            ("repeat_count", "Paired repeats"),
        ),
    ),
    (
        "Cross-model Macro-F1",
        (
            ("profile", "Profile"),
            ("factor_id", "Factor"),
            ("native_aggregation_view", "Native endpoint"),
            (
                ("inception_macro_f1", "inception_macro_f1_sd"),
                "InceptionTime Macro-F1 mean ± SD (%)",
            ),
            (("cnn_macro_f1", "cnn_macro_f1_sd"), "CNN Macro-F1 mean ± SD (%)"),
            (
                (
                    "inception_minus_cnn_macro_f1",
                    "inception_minus_cnn_macro_f1_sd",
                ),
                "InceptionTime − CNN Δ Macro-F1 mean ± SD (pp)",
            ),
            ("repeat_count", "Paired repeats"),
        ),
    ),
    (
        "Cross-model worst-class F1",
        (
            ("profile", "Profile"),
            ("factor_id", "Factor"),
            ("native_aggregation_view", "Native endpoint"),
            (
                ("inception_worst_class_f1", "inception_worst_class_f1_sd"),
                "InceptionTime worst-class F1 mean ± SD (%)",
            ),
            (
                ("cnn_worst_class_f1", "cnn_worst_class_f1_sd"),
                "CNN worst-class F1 mean ± SD (%)",
            ),
            (
                (
                    "inception_minus_cnn_worst_class_f1",
                    "inception_minus_cnn_worst_class_f1_sd",
                ),
                "InceptionTime − CNN Δ worst-class F1 mean ± SD (pp)",
            ),
            ("repeat_count", "Paired repeats"),
        ),
    ),
    (
        "Cross-model control and applicability audit",
        (
            ("profile", "Profile"),
            ("factor_id", "Factor"),
            ("native_aggregation_view", "Native endpoint"),
            ("repeat_count", "Paired repeats"),
            ("cross_model_profile_controls_match", "Controls match"),
            ("comparison_metrics_available", "Available"),
        ),
    ),
)


_STAGE3_STAR_ABSOLUTE_REPORT_TABLES = (
    (
        "Native absolute endpoints",
        (
            ("model", "Model"),
            ("profile", "Profile"),
            ("factor_id", "Factor"),
            ("native_aggregation_view", "Native endpoint"),
            ("native_balanced_accuracy", "Native BA"),
            ("native_macro_f1", "Native Macro-F1"),
            ("native_worst_class_f1", "Native worst-class F1"),
            ("passed_cell_count", "Passed cells"),
        ),
    ),
    (
        "Absolute aggregation-sensitivity views",
        (
            ("model", "Model"),
            ("profile", "Profile"),
            ("factor_id", "Factor"),
            ("BA_W_sensitivity", "BA W"),
            ("BA_A_sensitivity", "BA A"),
            ("BA_B_sensitivity", "BA B"),
        ),
    ),
    (
        "Absolute endpoint factor audit",
        (
            ("model", "Model"),
            ("profile", "Profile"),
            ("factor_id", "Factor"),
            ("passed_cell_count", "Passed cells"),
            ("single_factor_audit", "Factor audit"),
            ("cross_model_profile_controls_match", "Cross-model controls match"),
        ),
    ),
)


_STAGE3_STAR_CONTRAST_REPORT_TABLES = (
    (
        "Contrast endpoint identity",
        (
            ("model", "Model"),
            ("factor_id", "Factor"),
            ("reference_profile", "Reference"),
            ("variant_profile", "Variant"),
            ("reference_native_aggregation_view", "Reference endpoint"),
            ("variant_native_aggregation_view", "Variant endpoint"),
        ),
    ),
    (
        "Native contrast deltas",
        (
            ("model", "Model"),
            ("factor_id", "Factor"),
            ("reference_profile", "Reference"),
            ("variant_profile", "Variant"),
            ("delta_native_balanced_accuracy", "Native Δ BA"),
            ("delta_native_macro_f1", "Native Δ Macro-F1"),
            ("delta_native_worst_class_f1", "Native Δ worst-class F1"),
            ("contrast_metrics_available", "Available"),
        ),
    ),
    (
        "Aggregation-sensitivity contrast deltas",
        (
            ("model", "Model"),
            ("factor_id", "Factor"),
            ("reference_profile", "Reference"),
            ("variant_profile", "Variant"),
            ("delta_balanced_accuracy_W_sensitivity_only", "Sensitivity Δ BA W"),
            ("delta_balanced_accuracy_A_sensitivity_only", "Sensitivity Δ BA A"),
            ("delta_balanced_accuracy_B_sensitivity_only", "Sensitivity Δ BA B"),
            ("contrast_metrics_available", "Available"),
        ),
    ),
    (
        "Contrast reproducibility audit",
        (
            ("model", "Model"),
            ("factor_id", "Factor"),
            ("reference_profile", "Reference"),
            ("variant_profile", "Variant"),
            ("seed_match", "Seeds match"),
            ("split_hash_match", "Split hashes match"),
            ("heldout_roster_hash_match", "Held-out rosters match"),
            ("contrast_metrics_available", "Available"),
        ),
    ),
    (
        "Contrast factor and availability audit",
        (
            ("model", "Model"),
            ("factor_id", "Factor"),
            ("reference_profile", "Reference"),
            ("variant_profile", "Variant"),
            ("actual_changed_control_paths", "Actual changed paths"),
            ("single_factor_audit", "Factor audit"),
            ("unavailable_reasons", "N/A reasons"),
        ),
    ),
    (
        "B0/B7 window-OOF identity audit",
        (
            ("model", "Model"),
            ("variant_profile", "Variant"),
            (
                "report_view_factor_training_controls_identical",
                "Training controls identical",
            ),
            (
                "report_view_factor_window_oof_probabilities_identical",
                "Window OOF identical",
            ),
            ("matched_window_oof_row_count", "Matched window rows"),
            (
                "window_oof_probability_max_abs_diff",
                "Max absolute probability diff",
            ),
            ("window_oof_identity_audit_status", "Identity audit"),
        ),
    ),
)


_STAGE3_STAR_FOLD_REPORT_TABLES = (
    (
        "Matched-fold metric deltas",
        (
            ("model", "Model"),
            ("factor_id", "Factor"),
            ("repeat", "Repeat"),
            ("fold", "Fold"),
            ("delta_native_balanced_accuracy", "Native Δ BA"),
            ("delta_native_macro_f1", "Native Δ Macro-F1"),
            ("delta_native_worst_class_f1", "Native Δ worst-class F1"),
            ("contrast_metrics_available", "Available"),
        ),
    ),
    (
        "Matched-fold pairing and inference",
        (
            ("model", "Model"),
            ("factor_id", "Factor"),
            ("reference_profile", "Reference"),
            ("variant_profile", "Variant"),
            ("repeat", "Repeat"),
            ("fold", "Fold"),
            ("contrast_metrics_available", "Available"),
            ("inference", "Inference"),
        ),
    ),
)


_STAGE3_STAR_EXECUTION_REPORT_TABLES = (
    (
        "Execution-order absolute performance",
        (
            ("model", "Model"),
            ("profile", "Profile"),
            ("factor_id", "Factor"),
            ("execution_order", "Execution order"),
            ("native_aggregation_view", "Native endpoint"),
            ("native_balanced_accuracy", "Native BA"),
            ("native_macro_f1", "Native Macro-F1"),
            ("native_worst_class_f1", "Native worst-class F1"),
        ),
    ),
    (
        "Execution-order scheduling audit",
        (
            ("model", "Model"),
            ("profile", "Profile"),
            ("factor_id", "Factor"),
            ("execution_order", "Execution order"),
            ("execution_transition", "Scheduling transition"),
            ("execution_transition_is_ablation", "Transition is ablation"),
        ),
    ),
)


_STAGE3_STAR_TABLES = (
    (
        "stage3_star_inception_comparison",
        "Stage 3 InceptionTime B0–B7 comparison",
        "One InceptionTime table: B0 is the baseline and B1–B7 are paired to B0 by repeat. Values are native participant-OOF repeat mean ± population SD. B2 and B6 are declared coupled bundles; B7 is a reporting-aggregation ablation.",
        _STAGE3_STAR_WITHIN_MODEL_REPORT_TABLES,
        "InceptionTime B0-B7 native repeat mean/SD and paired B0-centered deltas",
    ),
    (
        "stage3_star_cnn_comparison",
        "Stage 3 CompactCNN B0–B7 comparison",
        "One CompactCNN table: B0 is the baseline and B1–B7 are paired to B0 by repeat. Values are native participant-OOF repeat mean ± population SD. B2 and B6 are declared coupled bundles; B7 is a reporting-aggregation ablation.",
        _STAGE3_STAR_WITHIN_MODEL_REPORT_TABLES,
        "CompactCNN B0-B7 native repeat mean/SD and paired B0-centered deltas",
    ),
    (
        "stage3_star_model_comparison",
        "Stage 3 B0–B7 InceptionTime versus CompactCNN",
        "Each row horizontally matches the two models under the same B-profile, repeat split and native endpoint. InceptionTime − CNN is a descriptive matched architecture comparison, not one of the fourteen B0-centered ablations and carries no significance claim.",
        _STAGE3_STAR_CROSS_MODEL_REPORT_TABLES,
        "B0-B7 side-by-side InceptionTime and CompactCNN repeat mean/SD with paired descriptive model deltas",
    ),
    (
        "stage3_star_absolute",
        "Stage 3 centered-star detailed absolute endpoints",
        "Sixteen absolute model/profile endpoints. W/A/B are same-OOF sensitivity views; each row declares its native endpoint.",
        _STAGE3_STAR_ABSOLUTE_REPORT_TABLES,
        "Sixteen absolute model/profile endpoints with native and W/A/B same-OOF metrics",
    ),
    (
        "stage3_star_contrasts",
        "Stage 3 centered-star detailed contrast audit",
        "Fourteen same-model B0→variant contrasts. Availability requires all declared repeat×fold cells plus matching seeds, split hashes, held-out rosters, native metrics, and exact factor paths. B0/B7 also audits training-control and window-OOF identity.",
        _STAGE3_STAR_CONTRAST_REPORT_TABLES,
        "Fourteen same-model B0-centered contrasts with factor and reproducibility audits",
    ),
    (
        "stage3_star_fold_contrasts",
        "Stage 3 centered-star detailed matched-fold deltas",
        "Every declared repeat×fold delta is descriptive only: no CI or significance claim. Seven contrasts within each model share the same correlated B0.",
        _STAGE3_STAR_FOLD_REPORT_TABLES,
        "All matched repeat/fold descriptive deltas; no CI or significance inference",
    ),
    (
        "stage3_star_execution",
        "Stage 3 centered-star execution order",
        "Absolute scheduling rows only; neighbouring execution rows are not ablation contrasts.",
        _STAGE3_STAR_EXECUTION_REPORT_TABLES,
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


_REPRO_CASE_REPORT_TABLES = (
    (
        "Case execution and audit status",
        (
            ("case_id", "Case"),
            ("selected_case_status", "Selected status"),
            ("selected_attempt", "Selected attempt"),
            ("excluded_attempts", "Excluded attempts"),
            ("planned_cell_count", "Planned cells"),
            ("observed_cell_count", "Observed cells"),
            ("audit_status", "Status"),
        ),
    ),
    (
        "Case seed policies and observed seeds",
        (
            ("case_id", "Case"),
            ("declared_seed_policies", "Declared seed policy"),
            ("runtime_seed_policies", "Effective seed policy"),
            ("split_seeds", "Split seeds"),
            ("model_seeds", "Model seeds"),
            ("training_orchestration_seeds", "Orchestration seeds"),
            ("evaluation_statistics_seeds", "Evaluation seeds"),
        ),
    ),
)


_REPRO_CELL_REPORT_TABLES = (
    (
        "Cell execution and audit status",
        (
            ("case_id", "Case"),
            ("repeat", "Repeat"),
            ("fold", "Fold"),
            ("status", "Cell status"),
            ("selected_attempt", "Attempt"),
            ("audit_status", "Status"),
        ),
    ),
    (
        "Cell seed policy",
        (
            ("case_id", "Case"),
            ("repeat", "Repeat"),
            ("fold", "Fold"),
            ("declared_seed_policy", "Declared policy"),
            ("runtime_seed_policy", "Effective policy"),
            ("member_seed_semantics", "Member-seed semantics"),
        ),
    ),
    (
        "Cell effective seeds",
        (
            ("case_id", "Case"),
            ("repeat", "Repeat"),
            ("fold", "Fold"),
            ("split_seed", "Split seed"),
            ("training_orchestration_seed", "Orchestration seed"),
            ("training_seed", "Training seed"),
            ("model_seed_roster", "Model/member seeds"),
            ("evaluation_statistics_seed", "Evaluation seed"),
        ),
    ),
    (
        "Cell split identity and participant counts",
        (
            ("case_id", "Case"),
            ("repeat", "Repeat"),
            ("fold", "Fold"),
            ("materialized_split_csv_sha256", "Split CSV SHA256"),
            ("split_identity_sha256", "Fold membership SHA256"),
            ("train_participant_count", "Train participants"),
            ("oof_participant_count", "OOF participants"),
            ("train_oof_overlap_count", "Train/OOF overlap"),
        ),
    ),
    (
        "Cell epoch RNG audit",
        (
            ("case_id", "Case"),
            ("repeat", "Repeat"),
            ("fold", "Fold"),
            ("epoch_rng_seed_count", "Epoch RNG rows"),
            ("audit_status", "Status"),
        ),
    ),
)


_REPRO_SPLIT_REPORT_TABLES = (
    (
        "Frozen split identity and matching cases",
        (
            ("repeat", "Repeat"),
            ("fold", "Fold"),
            ("split_seed", "Split seed"),
            ("case_ids", "Matching cases"),
            ("audit_status", "Status"),
        ),
    ),
    (
        "Frozen split registry hashes",
        (
            ("repeat", "Repeat"),
            ("fold", "Fold"),
            ("materialized_split_csv_sha256", "Split CSV SHA256"),
            (
                "declared_source_registry_json_file_sha256",
                "Declared authority JSON SHA256",
            ),
            (
                "declared_source_registry_payload_sha256",
                "Declared authority payload SHA256",
            ),
        ),
    ),
    (
        "Frozen split participant counts",
        (
            ("repeat", "Repeat"),
            ("fold", "Fold"),
            ("train_participant_count", "Train participants"),
            ("oof_participant_count", "OOF participants"),
            ("train_oof_overlap_count", "Overlap"),
            ("audit_status", "Status"),
        ),
    ),
)


# Human-facing classifier tables are deliberately narrow.  The lossless
# ``comparison_conclusions`` and ``predictive_leaderboard`` exports retain all
# fields; these shared projections keep Markdown and HTML from independently
# growing another wide, denormalized result table.
_COMPARISON_REPORT_TABLES = (
    (
        "Performance",
        (
            ("case_id", "Case"),
            ("rank", "Rank"),
            ("balanced_accuracy_mean_sd_percent", "BA mean ± SD (%)"),
            ("macro_f1_mean_sd_percent", "Macro-F1 mean ± SD (%)"),
            (
                "macro_roc_auc_ovr_mean_sd_percent",
                "Macro ROC-AUC mean ± SD (%)",
            ),
            ("macro_pr_auc_ovr_mean_sd_percent", "Macro PR-AUC mean ± SD (%)"),
            ("worst_fold_balanced_accuracy_percent", "Worst-fold BA (%)"),
            ("worst_class_f1_percent", "Worst-class F1 (%)"),
        ),
    ),
    (
        "Confidence intervals",
        (
            ("case_id", "Case"),
            ("rank", "Rank"),
            (
                "balanced_accuracy_participant_cluster_ci95_percent",
                "BA participant-cluster 95% CI (%)",
            ),
            (
                "macro_f1_participant_cluster_ci95_percent",
                "Macro-F1 participant-cluster 95% CI (%)",
            ),
            (
                "macro_roc_auc_ovr_participant_cluster_ci95_percent",
                "ROC-AUC participant-cluster 95% CI (%)",
            ),
            ("participant_cluster_ci_applicability", "Cluster CI status"),
            ("participant_cluster_ci_reason", "Cluster CI reason"),
        ),
    ),
    (
        "Paired inference",
        (
            ("case_id", "Candidate"),
            ("paired_reference_case_id", "Reference"),
            ("ba_paired_delta_cluster_ci95_percent", "ΔBA cluster 95% CI (%)"),
            ("ba_holm_adjusted_p", "BA Holm P"),
            ("f1_paired_delta_cluster_ci95_percent", "ΔF1 cluster 95% CI (%)"),
            ("f1_holm_adjusted_p", "F1 Holm P"),
            (
                "roc_auc_paired_delta_cluster_ci95_percent",
                "ΔROC-AUC cluster 95% CI (%)",
            ),
            ("inference_role", "P-value role"),
        ),
    ),
)


_PREDICTIVE_REPORT_TABLES = (
    (
        "Predictive performance",
        (
            ("case_id", "Case"),
            ("predictive_rank", "Rank"),
            (
                (
                    "participant_mean_abstention_aware_balanced_accuracy",
                    "repeat_abstention_aware_balanced_accuracy_sample_sd",
                ),
                "Abstention-aware BA, mean ± SD (%)",
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
                (
                    "participant_mean_abstention_aware_macro_f1",
                    "repeat_abstention_aware_macro_f1_sample_sd",
                ),
                "Abstention-aware Macro-F1, mean ± SD (%)",
            ),
            (
                (
                    "participant_mean_balanced_accuracy",
                    "repeat_balanced_accuracy_sample_sd",
                ),
                "Conditional BA, mean ± SD (%)",
            ),
            (
                ("participant_mean_macro_f1", "repeat_macro_f1_sample_sd"),
                "Conditional Macro-F1, mean ± SD (%)",
            ),
        ),
    ),
    (
        "Coverage and abstention",
        (
            ("case_id", "Case"),
            ("predictive_rank", "Rank"),
            ("participant_mean_coverage_rate", "Participant coverage"),
            ("abstention_count", "Abstentions"),
            ("abstention_counts_by_class", "Abstentions by class"),
            ("metric_source", "Source"),
            ("frailty_classification_evaluation_scope", "Frailty endpoint"),
            (
                "auxiliary_motion_evidence_valid_outer_oof",
                "Motion auxiliary outer-OOF",
            ),
        ),
    ),
    (
        "Robustness and worst-case performance",
        (
            ("case_id", "Case"),
            ("predictive_rank", "Rank"),
            ("balanced_accuracy_lcb95", "Conditional BA LCB95"),
            ("macro_f1_lcb95", "Conditional Macro-F1 LCB95"),
            (
                "worst_fold_abstention_aware_balanced_accuracy",
                "Aware worst-fold BA",
            ),
            ("worst_fold_balanced_accuracy", "Conditional worst-fold BA"),
            ("worst_class_recall", "Worst recall"),
            ("worst_class_f1", "Worst F1"),
        ),
    ),
)


# The rows behind these projections remain lossless in CSV/JSON.  Keep the
# report views narrow and keyed so a reader can join them without relying on
# table position.
_PER_CLASS_REPORT_TABLES = (
    (
        "Per-class discrimination",
        (
            ("classifier_id", "Classifier"),
            ("evaluation_id", "Evaluation"),
            ("aggregation_level", "Level"),
            ("class_name", "Class"),
            ("balanced_accuracy_ovr", "One-vs-rest BA"),
            ("f1", "F1"),
            ("roc_auc_ovr", "One-vs-rest ROC-AUC"),
            ("pr_auc_ovr", "One-vs-rest PR-AUC"),
        ),
    ),
    (
        "Per-class sensitivity, precision, and specificity",
        (
            ("classifier_id", "Classifier"),
            ("evaluation_id", "Evaluation"),
            ("aggregation_level", "Level"),
            ("class_name", "Class"),
            ("precision", "Precision"),
            ("sensitivity", "Sensitivity / recall"),
            ("specificity", "Specificity"),
            ("result_applicability", "Result applicability"),
        ),
    ),
    (
        "Per-class confusion counts",
        (
            ("classifier_id", "Classifier"),
            ("evaluation_id", "Evaluation"),
            ("class_name", "Class"),
            ("true_positive", "TP"),
            ("false_positive", "FP"),
            ("true_negative", "TN"),
            ("false_negative", "FN"),
        ),
    ),
    (
        "Per-class support and retention",
        (
            ("classifier_id", "Classifier"),
            ("evaluation_id", "Evaluation"),
            ("class_name", "Class"),
            ("support", "Support"),
            ("predicted_support", "Predicted support"),
            ("input_observation_count", "Input observations"),
            ("retained_observation_count", "Retained observations"),
            ("excluded_observation_count", "Excluded observations"),
        ),
    ),
    (
        "Per-class applicability and provenance",
        (
            ("classifier_id", "Classifier"),
            ("evaluation_id", "Evaluation"),
            ("class_name", "Class"),
            ("metric_scope", "Metric scope"),
            ("case_execution_status", "Case status"),
            ("probability_metric_applicability", "ROC/PR applicability"),
            ("metric_source", "Source"),
        ),
    ),
)


_PAIRED_INFERENCE_REPORT_TABLES = (
    (
        "Paired effects and confidence intervals",
        (
            ("candidate_case_id", "Candidate"),
            ("reference_case_id", "Reference"),
            ("metric", "Metric"),
            ("candidate_minus_reference", "Candidate − reference"),
            ("participant_cluster_delta_ci95_low", "Cluster CI low"),
            ("participant_cluster_delta_ci95_high", "Cluster CI high"),
            ("comparison_family", "Family"),
            ("inference_role", "Inference role"),
        ),
    ),
    (
        "Paired P values",
        (
            ("candidate_case_id", "Candidate"),
            ("reference_case_id", "Reference"),
            ("metric", "Metric"),
            ("raw_two_sided_p_value", "Raw P"),
            ("holm_adjusted_p_value", "Holm P"),
            ("reject_null_after_holm", "Holm P ≤ 0.05"),
            ("p_value_applicability", "P applicability"),
        ),
    ),
    (
        "Paired resampling audit",
        (
            ("candidate_case_id", "Candidate"),
            ("reference_case_id", "Reference"),
            ("metric", "Metric"),
            ("participant_count", "Participants"),
            ("repeat_count", "Repeats"),
            ("n_resamples", "Permutations"),
            ("bootstrap_resamples", "Bootstrap draws"),
            ("bootstrap_seed", "Bootstrap seed"),
        ),
    ),
    (
        "Paired comparison applicability",
        (
            ("candidate_case_id", "Candidate"),
            ("reference_case_id", "Reference"),
            ("metric", "Metric"),
            ("comparison_family", "Family"),
            ("inference_role", "Inference role"),
            ("comparison_contract_status", "Roster contract"),
            ("p_value_applicability", "P applicability"),
        ),
    ),
)


_PAIRWISE_REPEAT_DELTA_REPORT_TABLES = (
    (
        "Per-repeat balanced-accuracy differences",
        (
            ("candidate_case_id", "Candidate"),
            ("reference_case_id", "Reference"),
            ("repeat", "Repeat"),
            ("split_seed", "Split seed"),
            ("reference_balanced_accuracy", "Reference BA"),
            ("candidate_balanced_accuracy", "Candidate BA"),
            ("balanced_accuracy_delta", "ΔBA"),
        ),
    ),
    (
        "Per-repeat Macro-F1 differences",
        (
            ("candidate_case_id", "Candidate"),
            ("reference_case_id", "Reference"),
            ("repeat", "Repeat"),
            ("split_seed", "Split seed"),
            ("reference_macro_f1", "Reference Macro-F1"),
            ("candidate_macro_f1", "Candidate Macro-F1"),
            ("macro_f1_delta", "ΔMacro-F1"),
        ),
    ),
    (
        "Per-repeat ROC-AUC differences",
        (
            ("candidate_case_id", "Candidate"),
            ("reference_case_id", "Reference"),
            ("repeat", "Repeat"),
            ("split_seed", "Split seed"),
            ("reference_macro_roc_auc_ovr", "Reference Macro ROC-AUC"),
            ("candidate_macro_roc_auc_ovr", "Candidate Macro ROC-AUC"),
            ("macro_roc_auc_ovr_delta", "ΔMacro ROC-AUC"),
        ),
    ),
    (
        "Per-repeat roster audit",
        (
            ("candidate_case_id", "Candidate"),
            ("reference_case_id", "Reference"),
            ("repeat", "Repeat"),
            ("comparison_family", "Family"),
            ("comparison_role", "Comparison role"),
            ("matched_participant_count", "Matched participants"),
            ("comparison_contract_status", "Roster contract"),
            ("matched_roster_sha256", "Roster SHA-256"),
        ),
    ),
)


_AGGREGATION_LINE_REPORT_TABLES = (
    (
        "Aggregation-view performance",
        (
            ("case_id", "Case"),
            ("balance_line", "Aggregation view"),
            ("view_role", "Role"),
            ("participant_mean_balanced_accuracy", "Mean BA"),
            ("participant_mean_macro_f1", "Mean Macro-F1"),
            ("worst_class_recall", "Worst recall"),
            ("worst_class_f1", "Worst F1"),
            ("expected_calibration_error", "ECE"),
        ),
    ),
    (
        "Line-A minus Line-B sensitivity deltas",
        (
            ("case_id", "Case"),
            ("balance_line", "Aggregation view"),
            ("view_role", "Role"),
            ("line_a_minus_line_b_balanced_accuracy", "Line A − Line B BA"),
            ("line_a_minus_line_b_macro_f1", "Line A − Line B Macro-F1"),
            ("repeat_count", "Repeats"),
            ("primary_ranking_eligible", "Primary ranking eligible"),
        ),
    ),
    (
        "Aggregation OOF retention",
        (
            ("case_id", "Case"),
            ("balance_line", "Aggregation view"),
            ("view_role", "Role"),
            ("participant_oof_prediction_count", "Retained participant OOF n"),
            ("participant_oof_total_count", "All participant units n"),
            ("dropped_participant_oof_count", "Dropped participant units n"),
            ("file_oof_prediction_count", "All file OOF n"),
            ("dropped_file_oof_prediction_count", "Dropped files n"),
        ),
    ),
    (
        "Aggregation replay and ranking applicability",
        (
            ("case_id", "Case"),
            ("balance_line", "Aggregation view"),
            ("view_role", "Role"),
            ("source_replay_validation", "Source replay"),
            ("primary_ranking_eligible", "Primary ranking eligible"),
        ),
    ),
)


_AGGREGATION_VIEW_REPORT_TABLES = (
    (
        "Parallel aggregation-view performance",
        (
            ("case_id", "Case"),
            ("aggregation_view", "Aggregation view"),
            ("evidence_role", "Evidence role"),
            ("participant_mean_balanced_accuracy", "Mean BA"),
            ("participant_mean_macro_f1", "Mean Macro-F1"),
            ("worst_class_recall", "Worst recall"),
            ("worst_class_f1", "Worst F1"),
            ("repeat_count", "Repeats"),
        ),
    ),
    (
        "Parallel aggregation-view support and applicability",
        (
            ("case_id", "Case"),
            ("aggregation_view", "Aggregation view"),
            ("evidence_role", "Evidence role"),
            ("participant_oof_prediction_count", "Participant OOF n"),
            ("primary_ranking_eligible", "Primary ranking eligible"),
        ),
    ),
)


_AGGREGATION_HIERARCHY_REPORT_TABLES = (
    (
        "Hierarchy OOF-unit retention",
        (
            ("case_id", "Case"),
            ("repeat", "Repeat"),
            ("aggregation_level", "Level"),
            ("aggregation_view", "View"),
            ("group_label", "Group"),
            ("oof_unit_count", "OOF units"),
            ("retained_oof_unit_count", "Retained units"),
            ("dropped_oof_unit_count", "Dropped units"),
        ),
    ),
    (
        "Hierarchy participant retention",
        (
            ("case_id", "Case"),
            ("repeat", "Repeat"),
            ("aggregation_view", "View"),
            ("group_label", "Group"),
            ("retained_coverage", "Retained coverage"),
            ("total_participant_count", "All participants"),
            ("retained_participant_count", "Retained participants"),
            ("dropped_participant_count", "Dropped participants"),
        ),
    ),
)


_WORST_CLASS_STABILITY_REPORT_TABLES = (
    (
        "Abstention-aware worst-class stability",
        (
            ("case_id", "Case"),
            ("worst_class_f1_stability_rank", "Stability rank"),
            ("predictive_rank", "Aware-BA rank"),
            ("abstention_aware_worst_class_f1", "Aware worst F1"),
            ("abstention_aware_worst_class_recall", "Aware worst recall"),
            (
                (
                    "participant_mean_abstention_aware_balanced_accuracy",
                    "repeat_abstention_aware_balanced_accuracy_population_sd",
                ),
                "Aware BA, mean ± SD (%)",
            ),
        ),
    ),
    (
        "Conditional worst-class stability",
        (
            ("case_id", "Case"),
            ("worst_class_f1_stability_rank", "Stability rank"),
            ("predictive_rank", "Aware-BA rank"),
            ("worst_class_f1", "Worst F1"),
            ("worst_class_recall", "Worst recall"),
            (
                (
                    "participant_mean_balanced_accuracy",
                    "repeat_balanced_accuracy_population_sd",
                ),
                "Conditional BA, mean ± SD (%)",
            ),
        ),
    ),
)


_ROUTE_ROLE_COVERAGE_REPORT_TABLES = (
    (
        "Route identity and input files",
        (
            ("case_id", "Case"),
            ("evaluation_partition", "Partition"),
            ("role", "Role"),
            ("quality_tier", "Quality tier"),
            ("motion_state", "Motion"),
            ("route_state", "Route state"),
            ("signal_route", "Signal route"),
            ("record_count", "Files"),
        ),
    ),
    (
        "Route file retention",
        (
            ("case_id", "Case"),
            ("evaluation_partition", "Partition"),
            ("role", "Role"),
            ("route_state", "Route state"),
            ("record_count", "Files"),
            ("retained_record_count", "Retained files"),
            ("dropped_record_count", "Dropped files"),
            ("retained_coverage", "Retained coverage"),
        ),
    ),
    (
        "Route abstention and rate availability",
        (
            ("case_id", "Case"),
            ("role", "Role"),
            ("route_state", "Route state"),
            ("abstention_rate", "Abstention"),
            ("abstention_reasons", "Abstention reasons"),
            ("direct_rate_record_count", "Direct"),
            ("processed_rate_record_count", "Processed"),
            ("unavailable_predictor_rate", "Unavailable predictors"),
        ),
    ),
    (
        "Denoiser record outcomes",
        (
            ("case_id", "Case"),
            ("role", "Role"),
            ("route_state", "Route state"),
            ("denoiser_attempt_count", "Denoiser attempts"),
            ("denoiser_success_count", "Denoiser successes"),
            ("reducer_failure_count", "Reducer failures"),
            ("reducer_failure_rate", "Reducer failure rate"),
            ("denoiser_requested_cell_count", "Denoiser cells"),
        ),
    ),
    (
        "Denoiser cell outcomes",
        (
            ("case_id", "Case"),
            ("role", "Role"),
            ("route_state", "Route state"),
            ("denoiser_requested_cell_count", "Denoiser cells"),
            ("denoiser_failed_cell_count", "Failed denoiser cells"),
            ("denoiser_cell_failure_rate", "Denoiser cell failure rate"),
            ("post_q_rate_pass_cell_count", "Post-Q pass cells"),
            ("post_q_rate_pass_cell_rate", "Post-Q pass cell rate"),
        ),
    ),
    (
        "Post-denoiser Q_rate record recovery",
        (
            ("case_id", "Case"),
            ("role", "Role"),
            ("route_state", "Route state"),
            ("post_q_rate_pass_rate", "Post Q_rate pass rate"),
            ("post_q_rate_recovery_eligible_count", "Recovery eligible"),
            ("post_q_rate_recovery_count", "Q_rate recovered"),
            ("post_q_rate_recovery_rate", "Q_rate recovery rate"),
        ),
    ),
    (
        "Post-denoiser Q_rate cell recovery",
        (
            ("case_id", "Case"),
            ("role", "Role"),
            ("route_state", "Route state"),
            (
                "post_q_rate_recovery_eligible_cell_count",
                "Recovery-eligible cells",
            ),
            ("post_q_rate_recovered_cell_count", "Recovered cells"),
            ("post_q_rate_recovery_cell_rate", "Cell recovery rate"),
        ),
    ),
)


_SQI_PROVENANCE_REPORT_TABLES = (
    (
        "Direct Q_rate evidence",
        (
            ("case_id", "Case"),
            ("role", "Role"),
            ("quality_tier", "Tier"),
            ("direct_q_rate_states", "Direct Q_rate state"),
            ("mean_direct_q_rate_score", "Mean direct Q_rate"),
            ("mean_direct_q_rate_coverage", "Direct Q_rate coverage"),
        ),
    ),
    (
        "Direct Q_morph evidence",
        (
            ("case_id", "Case"),
            ("role", "Role"),
            ("quality_tier", "Tier"),
            ("direct_q_morph_states", "Direct Q_morph state"),
            ("mean_direct_q_morph_score", "Mean direct Q_morph"),
            ("mean_direct_q_morph_coverage", "Direct Q_morph coverage"),
        ),
    ),
    (
        "Post-denoiser Q_rate evidence",
        (
            ("case_id", "Case"),
            ("role", "Role"),
            ("quality_tier", "Tier"),
            ("post_q_rate_states", "Post Q_rate state"),
            ("mean_post_q_rate_score", "Mean post Q_rate"),
            ("mean_post_q_rate_coverage", "Post Q_rate coverage"),
        ),
    ),
)


_MOTION_EVIDENCE_REPORT_TABLES = (
    (
        "Frozen motion scores and route",
        (
            ("case_id", "Case"),
            ("role", "Role"),
            ("quality_tier", "Tier"),
            ("motion_state", "Motion"),
            ("mean_motion_record_probability", "Mean p(motion)"),
            ("mean_motion_threshold", "Threshold"),
            ("mean_motion_window_count", "Mean windows"),
            ("denoiser_ids", "Denoiser"),
        ),
    ),
    (
        "Frozen motion artifact provenance",
        (
            ("case_id", "Case"),
            ("role", "Role"),
            ("quality_tier", "Tier"),
            ("motion_state", "Motion"),
            ("motion_evidence_sha256", "Evidence SHA-256"),
            ("motion_model_artifact_sha256", "Model SHA-256"),
            ("motion_training_scope", "Training scope"),
            ("motion_frailty29_relation", "Frailty29 relation"),
        ),
    ),
    (
        "Frozen motion applicability and denoiser status",
        (
            ("case_id", "Case"),
            ("role", "Role"),
            ("quality_tier", "Tier"),
            ("motion_state", "Motion"),
            (
                "auxiliary_motion_evidence_valid_outer_oof",
                "Valid outer-OOF motion evidence",
            ),
            ("denoiser_ids", "Denoiser"),
            ("denoiser_statuses", "Denoiser status"),
        ),
    ),
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


def _fail_closed_pairwise_rows_for_frozen_registry(
    rows: Sequence[Mapping[str, Any]],
    *,
    reason: str,
) -> list[dict[str, Any]]:
    """Retain every declared pair row while invalidating numeric inference.

    Pairwise tables are part of the report contract even when the authoritative
    frozen split registry cannot be verified.  This helper therefore preserves
    pair/family/role provenance and the planned row roster, but removes every
    numeric comparison result instead of allowing an observed OOF roster to act
    as an implicit replacement registry.
    """

    unavailable = f"N/A_{reason}"
    output: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        for field in tuple(row):
            if (
                field in {
                    "candidate_minus_reference",
                    "participant_cluster_delta_ci95_low",
                    "participant_cluster_delta_ci95_high",
                    "bootstrap_valid_resamples",
                    "raw_two_sided_p_value",
                    "n_resamples",
                    "holm_adjusted_p_value",
                    "holm_rank",
                    "holm_family_size",
                    "reject_null_after_holm",
                    "participant_count",
                    "repeat_count",
                    "split_seed",
                    "matched_participant_count",
                    "matched_roster_sha256",
                }
                or field.startswith("reference_balanced_accuracy")
                or field.startswith("candidate_balanced_accuracy")
                or field.startswith("balanced_accuracy_delta")
                or field.startswith("reference_macro_f1")
                or field.startswith("candidate_macro_f1")
                or field.startswith("macro_f1_delta")
                or field.startswith("reference_macro_roc_auc_ovr")
                or field.startswith("candidate_macro_roc_auc_ovr")
                or field.startswith("macro_roc_auc_ovr_delta")
            ):
                row[field] = None
        row["comparison_contract_status"] = unavailable
        row["frozen_split_registry_status"] = unavailable
        if "p_value_applicability" in row:
            row["p_value_applicability"] = unavailable
        if "test_method" in row:
            row["test_method"] = unavailable
        if "interpretation" in row:
            row["interpretation"] = (
                "N/A: numeric comparison was not computed because the frozen "
                "split registry could not be verified; the declared pair row "
                "is retained for audit completeness."
            )
        output.append(row)
    return output


def _report_markdown(
    collected: CollectedStudy,
    analysis: StudyAnalysis,
    figures: Sequence[Mapping[str, Any]],
    reproducibility: ReproducibilityAudit | None = None,
    test_component_rows: Sequence[Mapping[str, Any]] = (),
    reporter_profiles: Sequence[Mapping[str, Any]] = (),
    comparison_conclusions: Sequence[Mapping[str, Any]] = (),
    selection_conclusions: Sequence[Mapping[str, Any]] = (),
    paired_inference_rows: Sequence[Mapping[str, Any]] = (),
    pairwise_repeat_deltas: Sequence[Mapping[str, Any]] = (),
    top_model_configuration_rows: Sequence[Mapping[str, Any]] = (),
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
        "## Test models, modules, inputs, and fixed parameters",
        "",
        "The identical standalone table is in "
        "[TEST_COMPONENTS.md](TEST_COMPONENTS.md); machine-readable copies are "
        "`tables/test_components.csv` and `.json`. Input data are reported as "
        "dataset/path, signal view, channels, units, rate, and windows—not hashes.",
        "",
        markdown_test_component_table(test_component_rows),
        "",
        "## Model/module-owned reporter contracts and literature",
        "",
        "The complete generated methods record is in "
        "[REPORT_METHODS.md](REPORT_METHODS.md). Profiles are selected from the "
        "persisted component identities and affect presentation only—not training, "
        "predictions, thresholds, or ranking.",
        "",
    ]
    lines.extend((markdown_reporter_profile_tables(reporter_profiles), ""))
    lines.extend([
        "## Comprehensive comparison and confidence-qualified conclusion",
        "",
        "P values are null-hypothesis tail probabilities, not the probability "
        "that a model is best. Repeat Student-t CIs and participant-cluster "
        "bootstrap CIs are kept separate. A participant-cluster CI resamples "
        "participant IDs with replacement within true-class strata; each sampled "
        "participant carries all of its repeat OOF predictions, the metric is "
        "recomputed within repeat and repeats are averaged equally. Its 95% bounds "
        "are the 2.5th/97.5th percentiles. For paired CIs, the identical sampled "
        "participant multiset is applied to candidate and reference before taking "
        "candidate − reference. This is participant-sampling uncertainty "
        "conditional on the frozen dataset/folds/predictions; it excludes dataset "
        "shift and model-selection uncertainty. The lossless table and full narrative "
        "are in [RESULT_INTERPRETATION.md](RESULT_INTERPRETATION.md) and "
        "`tables/comparison_conclusions.json`.",
        "",
    ])
    for title, columns in _COMPARISON_REPORT_TABLES:
        lines.extend([f"### {title}", ""])
        lines.extend(_markdown_table(comparison_conclusions, columns))
    lines.extend(["### Conclusions by evidence angle", ""])
    lines.extend(
        _markdown_table(
            selection_conclusions,
            (
                ("angle", "Angle"),
                ("leading_or_selected_case", "Case"),
                ("finding", "Finding"),
                ("confidence", "Confidence"),
                ("selection_effect", "Selection effect"),
            ),
        )
    )
    lines.extend([
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
    ])
    for title, columns in _REPRO_CASE_REPORT_TABLES:
        lines.extend([f"### {title}", ""])
        lines.extend(_markdown_table(reproducibility.case_rows, columns))
    lines.extend(
        [
            "<details><summary>Per-cell seed and split evidence</summary>",
            "",
        ]
    )
    for title, columns in _REPRO_CELL_REPORT_TABLES:
        lines.extend([f"#### {title}", ""])
        lines.extend(_markdown_table(reproducibility.cell_rows, columns))
    lines.extend(["</details>", "", "### Frozen split roster", ""])
    for title, columns in _REPRO_SPLIT_REPORT_TABLES:
        lines.extend([f"#### {title}", ""])
        lines.extend(_markdown_table(reproducibility.split_rows, columns))
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
            "filter this table. Probability metrics and participant-cluster CIs "
            "are reported once in the comprehensive performance/CI tables above "
            "instead of being duplicated here.",
            "",
        ]
    )
    for title, columns in _PREDICTIVE_REPORT_TABLES:
        lines.extend([f"### {title}", ""])
        lines.extend(_markdown_table(analysis.predictive_leaderboard, columns))
    if top_model_configuration_rows:
        lines.extend(
            [
                "## Top-ranked model complete resolved configurations",
                "",
                "This is an audit projection of the complete persisted resolved YAML "
                "for the configured number of top predictive-ranked models (five for "
                "Stage 0). Ranking is copied from the participant-level predictive "
                "leaderboard; this table does not perform another selection. Ordered "
                "lists such as roles and channels remain intact. Provenance hashes are "
                "excluded here and remain available in the manifests and resolved files.",
                "",
                "<details><summary>Expand complete Top-model configuration table "
                f"({len(top_model_configuration_rows)} parameter rows)</summary>",
                "",
            ]
        )
        lines.extend(
            _markdown_table(
                top_model_configuration_rows,
                TOP_MODEL_CONFIGURATION_COLUMNS,
            )
        )
        lines.extend(["</details>", ""])
    lines.extend(
        [
            "## Per-class classifier results",
            "",
            "Every classifier with persisted participant OOF probabilities is "
            "listed for every declared class. Hard-label metrics use the "
            "persisted decision rule; ROC/PR metrics use the matching class "
            "probability in a one-vs-rest calculation. Routes with abstentions "
            "also receive full-roster rows where an abstention is a false "
            "negative for its true class; ROC/PR are N/A for that scope because "
            "no probability is assigned to an abstained endpoint. Their one-vs-rest "
            "BA is descriptive and must not be averaged to reconstruct the global "
            "abstention-aware BA, whose registered definition is macro recall. Missing "
            "classifier OOF remains explicit rather than being omitted.",
            "",
        ]
    )
    per_class_rows = getattr(analysis, "classifier_per_class_results", ())
    for title, columns in _PER_CLASS_REPORT_TABLES:
        lines.extend([f"### {title}", ""])
        lines.extend(_markdown_table(per_class_rows, columns))
    lines.extend(
        [
            "## Repeat-level predictive distributions",
            "",
            "Mean and sample SD are shown in one percentage column and the "
            "two-sided repeat-level Student-t 95% CI is shown beside it. "
            "Lossless bounds, range, mean, and SD remain in the matching JSON table.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            compact_rows(getattr(analysis, "metric_distribution_summary", ())),
            (
                ("case_id", "Case"),
                ("metric", "Metric"),
                ("n", "Repeats"),
                ("mean_sd", "Mean ± SD (%)"),
                ("ci95", "Repeat 95% CI (%)"),
                ("metric_source", "Source"),
            ),
        )
    )
    lines.extend(
        [
            "<details><summary>Per-class repeat distributions</summary>",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            compact_rows(
                getattr(analysis, "per_class_metric_distribution_summary", ())
            ),
            (
                ("case_id", "Case"),
                ("class_name", "Class"),
                ("metric", "Metric"),
                ("n", "Repeats"),
                ("mean_sd", "Mean ± SD (%)"),
                ("ci95", "Repeat 95% CI (%)"),
            ),
        )
    )
    lines.extend(["</details>", ""])
    lines.extend(
        [
            "## Paired participant-cluster inference",
            "",
            "Each candidate is compared with the declared reference on the exact "
            "participant/repeat/fold/split roster. P values are two-sided "
            "participant-cluster permutation results; Holm adjustment is applied "
            "separately within BA and Macro-F1. BA, Macro-F1 and macro ROC-AUC "
            "also report shared-draw participant-cluster bootstrap CIs for the "
            "candidate-minus-reference difference; ROC-AUC P is N/A. These comparisons do not select a "
            "winner and do not turn this representation screen into a causal ablation.",
            "",
        ]
    )
    for title, columns in _PAIRED_INFERENCE_REPORT_TABLES:
        lines.extend([f"### {title}", ""])
        lines.extend(_markdown_table(paired_inference_rows, columns))
    lines.extend(
        [
            "### Matched per-repeat differences for every declared pair",
            "",
            "Each row keeps the exact participant/fold/split roster and reports "
            "candidate − reference. Only pairs registered by the study design "
            "or explicitly labeled post-hoc by the reporter are included; an "
            "architecture comparison is not relabelled as an ablation.",
            "",
        ]
    )
    for title, columns in _PAIRWISE_REPEAT_DELTA_REPORT_TABLES:
        lines.extend([f"#### {title}", ""])
        lines.extend(_markdown_table(pairwise_repeat_deltas, columns))
    if _is_stage3_centered_star(collected.plan):
        for name, title, notice, schemas, _description in _STAGE3_STAR_TABLES:
            lines.extend([f"## {title}", "", notice, ""])
            for view_title, columns in schemas:
                lines.extend([f"### {view_title}", ""])
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
        for view_title, columns in _LEGACY_BRIDGE_NUMERIC_REPORT_TABLES:
            lines.extend([f"### {view_title}", ""])
            lines.extend(
                _markdown_table(
                    analysis.legacy_bridge_numeric_ablation_report,
                    columns,
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
        for view_title, columns in _LEGACY_BRIDGE_EXECUTION_REPORT_TABLES:
            lines.extend([f"### {view_title}", ""])
            lines.extend(
                _markdown_table(
                    analysis.legacy_bridge_execution_order_report,
                    columns,
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
    for title, columns in _AGGREGATION_LINE_REPORT_TABLES:
        lines.extend([f"### {title}", ""])
        lines.extend(_markdown_table(analysis.aggregation_line_comparison, columns))
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
    for title, columns in _AGGREGATION_VIEW_REPORT_TABLES:
        lines.extend([f"### {title}", ""])
        lines.extend(_markdown_table(analysis.aggregation_view_comparison, columns))
    lines.extend(
        [
            "<details><summary>Hierarchy coverage: B/R1–R4 window/file views and "
            "B/R role-balanced view</summary>",
            "",
        ]
    )
    for title, columns in _AGGREGATION_HIERARCHY_REPORT_TABLES:
        lines.extend([f"#### {title}", ""])
        lines.extend(
            _markdown_table(analysis.aggregation_hierarchy_coverage, columns)
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
    for title, columns in _WORST_CLASS_STABILITY_REPORT_TABLES:
        lines.extend([f"### {title}", ""])
        lines.extend(_markdown_table(analysis.worst_class_f1_stability, columns))
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
    cache_overview = _preprocessing_cache_overview(
        collected.preprocessing_cache_rows
    )
    lines.extend(
        [
            "## Preprocessing-cache operational audit",
            "",
            "Cache reuse is operational provenance only and does not affect "
            "predictions, ranking, labels, fold-fitted artifacts, or route masks. "
            "The lossless per-cell/per-layer audit is in "
            "[`tables/preprocessing_cache.csv`](tables/preprocessing_cache.csv) "
            "and [`tables/preprocessing_cache.json`](tables/preprocessing_cache.json).",
            "",
            f"- Cells / cell-layer rows / events: "
            f"{cache_overview['cell_count']} / "
            f"{cache_overview['cell_layer_rows']} / "
            f"{cache_overview['event_count']}",
            f"- Hits / writes / bypasses: {cache_overview['hit_count']} / "
            f"{cache_overview['write_count']} / "
            f"{cache_overview['bypass_count']}",
            f"- Logical materialized array bytes / cache-operation seconds: "
            f"{cache_overview['logical_array_bytes']} / "
            f"{float(cache_overview['elapsed_seconds']):.4f}",
            "",
        ]
    )
    lines.extend(
        [
            "## Route × role coverage and feature availability",
            "",
            "This formal table uses outer-held-out (`outer_oof`) records only; "
            "outer-training route rows remain in the source artifacts but are not "
            "mixed into validation coverage. It separates direct and processed rate "
            "paths, retained coverage, unavailable predictors, and reducer failures "
            "for each role/route state.",
            "",
        ]
    )
    for title, columns in _ROUTE_ROLE_COVERAGE_REPORT_TABLES:
        lines.extend([f"### {title}", ""])
        lines.extend(_markdown_table(analysis.route_role_coverage, columns))
    lines.extend(
        [
            "## SQI state, score, and coverage provenance by each route",
            "",
            "Direct and post-denoiser coverage are reported separately so the "
            "configured minimum-coverage decision remains auditable.",
            "",
        ]
    )
    for title, columns in _SQI_PROVENANCE_REPORT_TABLES:
        lines.extend([f"### {title}", ""])
        lines.extend(_markdown_table(analysis.route_role_coverage, columns))
    lines.extend(
        [
            "## Denoiser paired HR/PPI endpoint audit",
            "",
            "HR is calculated as `60 / median(valid PPI seconds)` from the same "
            "registered peak detector before and after the single denoiser attempt. "
            "Rows are paired within recording and averaged within participant before "
            "the participant-macro summary. Use the `outer_oof` rows for the primary "
            "held-out comparison; outer-train rows remain audit-only. HR/PPI endpoint "
            "error here is absolute post-denoise minus same-record direct-PPG change; "
            "Frailty29 has no ECG reference, so it is not physiological accuracy.",
            "",
        ]
    )
    for title, columns in _DENOISER_HR_SUMMARY_REPORT_TABLES:
        lines.extend([f"### {title}", ""])
        lines.extend(_markdown_table(analysis.denoiser_hr_comparison, columns))
    lines.extend(
        [
            "<details><summary>Per-record paired denoiser HR evidence</summary>",
            "",
        ]
    )
    for title, columns in _DENOISER_HR_RECORD_REPORT_TABLES:
        lines.extend([f"#### {title}", ""])
        lines.extend(_markdown_table(analysis.denoiser_hr_record_pairs, columns))
    lines.extend(["</details>", ""])
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
    for title, columns in _MOTION_EVIDENCE_REPORT_TABLES:
        lines.extend([f"### {title}", ""])
        lines.extend(_markdown_table(analysis.route_role_coverage, columns))
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
                (("mean", "population_sd", False), "Mean ± SD"),
            ),
        )
    )
    lines.extend(
        [
            "## Classification score, t-SNE, and ROC–AUC diagnostics",
            "",
            "Every classifier with persisted participant OOF probabilities is "
            "represented in the three paired figure modules. t-SNE embeds the "
            "prediction-probability vector, not hidden features. Multiclass "
            "frailty decisions use argmax and therefore have no single scalar "
            "threshold.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            getattr(analysis, "classification_diagnostic_status", ()),
            (
                ("classifier_id", "Classifier"),
                ("prediction_score_status", "Score/threshold"),
                ("prediction_tsne_status", "Prediction t-SNE"),
                ("roc_auc_curve_status", "ROC–AUC curve"),
                ("prediction_row_count", "OOF rows"),
                ("tsne_point_count", "t-SNE points"),
                ("roc_curve_point_count", "ROC points"),
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
            "- [tables/repeat_per_class_metrics.csv](tables/repeat_per_class_metrics.csv)",
            "- [tables/per_class_metric_distribution_summary.csv](tables/per_class_metric_distribution_summary.csv)",
            "- [tables/classification_prediction_scores.csv](tables/classification_prediction_scores.csv)",
            "- [tables/classification_prediction_tsne.csv](tables/classification_prediction_tsne.csv)",
            "- [tables/classification_roc_curves.csv](tables/classification_roc_curves.csv)",
            "- [tables/classification_diagnostic_status.csv](tables/classification_diagnostic_status.csv)",
            "- [tables/table_figure_pairs.csv](tables/table_figure_pairs.csv)",
            "- [tables/report_tables.xlsx](tables/report_tables.xlsx): one table per worksheet",
            "- [tables/TABLE_COLUMN_DEFINITIONS.md](tables/TABLE_COLUMN_DEFINITIONS.md): every table column definition and formula",
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
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[
        tuple[str | tuple[str, str] | tuple[str, str, bool], str]
    ],
) -> str:
    if len(columns) > _MAX_HUMAN_FACING_REPORT_COLUMNS:
        raise ValueError(
            f"human-facing HTML table has {len(columns)} columns; "
            f"maximum is {_MAX_HUMAN_FACING_REPORT_COLUMNS}"
        )
    definition_columns = [field for field, _label in columns]
    definition_labels = [label for _field, label in columns]
    definitions = html_column_definitions_block(
        definition_columns,
        display_labels=definition_labels,
    )
    if not rows:
        return "<p><em>N/A — no rows were available.</em></p>" + definitions
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(format_mean_sd(row.get(field[0]), row.get(field[1]), percent=True if len(field) == 2 else field[2]) if isinstance(field, tuple) else _fmt(row.get(field)))}</td>"
                for field, _ in columns
            )
            + "</tr>"
        )
    return (
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
        + definitions
    )


def _html_semantic_tables(
    rows: Sequence[Mapping[str, Any]],
    schemas: Sequence[
        tuple[
            str,
            Sequence[
                tuple[str | tuple[str, str] | tuple[str, str, bool], str]
            ],
        ]
    ],
    *,
    heading_level: int = 3,
) -> str:
    """Render the same named narrow-table schemas used by Markdown."""

    if heading_level not in {2, 3, 4, 5, 6}:
        raise ValueError("semantic-table heading level must be between 2 and 6")
    fragments: list[str] = []
    for title, columns in schemas:
        if len(columns) > _MAX_HUMAN_FACING_REPORT_COLUMNS:
            raise ValueError(
                f"human-facing report table {title!r} has {len(columns)} "
                f"columns; maximum is {_MAX_HUMAN_FACING_REPORT_COLUMNS}"
            )
        fragments.extend(
            (
                f"<h{heading_level}>{html.escape(title)}</h{heading_level}>",
                _html_table(rows, columns),
            )
        )
    return "".join(fragments)


def _report_html(
    collected: CollectedStudy,
    analysis: StudyAnalysis,
    figures: Sequence[Mapping[str, Any]],
    reproducibility: ReproducibilityAudit | None = None,
    test_component_rows: Sequence[Mapping[str, Any]] = (),
    reporter_profiles: Sequence[Mapping[str, Any]] = (),
    comparison_conclusions: Sequence[Mapping[str, Any]] = (),
    selection_conclusions: Sequence[Mapping[str, Any]] = (),
    paired_inference_rows: Sequence[Mapping[str, Any]] = (),
    pairwise_repeat_deltas: Sequence[Mapping[str, Any]] = (),
    top_model_configuration_rows: Sequence[Mapping[str, Any]] = (),
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
            + _html_semantic_tables(getattr(analysis, name), schemas)
            for name, title, notice, schemas, _description in _STAGE3_STAR_TABLES
        )
    elif isinstance(collected.plan.get("legacy_bridge"), Mapping):
        bridge_html = f"""
<h2>Legacy/V2 bridge report A — numeric adjacent ablations (L0→L7)</h2>
<p class="notice">This is the causal-interpretation table. L0 is the baseline;
the next seven rows are only L0→L1 through L6→L7. Deltas are never calculated
from execution order.</p>
{_html_semantic_tables(
    analysis.legacy_bridge_numeric_ablation_report,
    _LEGACY_BRIDGE_NUMERIC_REPORT_TABLES,
)}
<h2>Legacy/V2 bridge report B — CompactCNN execution order</h2>
<p class="notice">Absolute W/A/B metrics in
L7→L5→L6→L4→L3→L2→L1→L0 order. There is deliberately no execution-order
delta: L7→L5 and all other neighbouring runs are scheduling transitions, not
causal ablations.</p>
{_html_semantic_tables(
    analysis.legacy_bridge_execution_order_report,
    _LEGACY_BRIDGE_EXECUTION_REPORT_TABLES,
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
{_html_semantic_tables(reproducibility.case_rows, _REPRO_CASE_REPORT_TABLES)}
<details><summary>Per-cell seed and split evidence</summary>
{_html_semantic_tables(
    reproducibility.cell_rows,
    _REPRO_CELL_REPORT_TABLES,
    heading_level=4,
)}</details>
<h3>Frozen split roster</h3>
{_html_semantic_tables(
    reproducibility.split_rows,
    _REPRO_SPLIT_REPORT_TABLES,
    heading_level=4,
)}
<h3>Reproducibility audit issues</h3>{reproducibility_issues}
"""
    component_tables_html = _html_semantic_tables(
        test_component_rows,
        TEST_COMPONENT_VIEW_SCHEMAS,
    )
    reporter_profile_tables_html = _html_semantic_tables(
        reporter_profiles,
        REPORTER_PROFILE_VIEW_SCHEMAS,
    )
    comparison_tables_html = _html_semantic_tables(
        comparison_conclusions,
        _COMPARISON_REPORT_TABLES,
    )
    predictive_tables_html = _html_semantic_tables(
        analysis.predictive_leaderboard,
        _PREDICTIVE_REPORT_TABLES,
    )
    top_model_configuration_html = (
        "<h2>Top-ranked model complete resolved configurations</h2>"
        "<p>Complete persisted resolved-YAML audit projection for the configured "
        "number of top predictive-ranked models (five for Stage 0). Ranking is "
        "copied from the participant-level predictive leaderboard and performs no "
        "additional selection. Ordered channel/role lists remain intact; provenance "
        "hashes stay in the source manifests and resolved configuration files.</p>"
        "<details><summary>Expand complete Top-model configuration table "
        f"({len(top_model_configuration_rows)} parameter rows)</summary>"
        + _html_table(
            top_model_configuration_rows,
            TOP_MODEL_CONFIGURATION_COLUMNS,
        )
        + "</details>"
        if top_model_configuration_rows
        else ""
    )
    per_class_tables_html = _html_semantic_tables(
        getattr(analysis, "classifier_per_class_results", ()),
        _PER_CLASS_REPORT_TABLES,
    )
    paired_inference_tables_html = _html_semantic_tables(
        paired_inference_rows,
        _PAIRED_INFERENCE_REPORT_TABLES,
    )
    pairwise_repeat_delta_tables_html = _html_semantic_tables(
        pairwise_repeat_deltas,
        _PAIRWISE_REPEAT_DELTA_REPORT_TABLES,
        heading_level=4,
    )
    aggregation_line_tables_html = _html_semantic_tables(
        analysis.aggregation_line_comparison,
        _AGGREGATION_LINE_REPORT_TABLES,
    )
    aggregation_view_tables_html = _html_semantic_tables(
        analysis.aggregation_view_comparison,
        _AGGREGATION_VIEW_REPORT_TABLES,
    )
    aggregation_hierarchy_tables_html = _html_semantic_tables(
        analysis.aggregation_hierarchy_coverage,
        _AGGREGATION_HIERARCHY_REPORT_TABLES,
        heading_level=4,
    )
    worst_class_stability_tables_html = _html_semantic_tables(
        analysis.worst_class_f1_stability,
        _WORST_CLASS_STABILITY_REPORT_TABLES,
    )
    route_role_coverage_tables_html = _html_semantic_tables(
        analysis.route_role_coverage,
        _ROUTE_ROLE_COVERAGE_REPORT_TABLES,
    )
    sqi_provenance_tables_html = _html_semantic_tables(
        analysis.route_role_coverage,
        _SQI_PROVENANCE_REPORT_TABLES,
    )
    denoiser_hr_summary_tables_html = _html_semantic_tables(
        getattr(analysis, "denoiser_hr_comparison", ()),
        _DENOISER_HR_SUMMARY_REPORT_TABLES,
    )
    denoiser_hr_record_tables_html = _html_semantic_tables(
        getattr(analysis, "denoiser_hr_record_pairs", ()),
        _DENOISER_HR_RECORD_REPORT_TABLES,
        heading_level=4,
    )
    motion_evidence_tables_html = _html_semantic_tables(
        analysis.route_role_coverage,
        _MOTION_EVIDENCE_REPORT_TABLES,
    )
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
<h2>Test models, modules, inputs, and fixed parameters</h2>
<p>The identical Markdown table is in <a href="TEST_COMPONENTS.md">TEST_COMPONENTS.md</a>.
Input data are named directly rather than represented by hashes.</p>
{component_tables_html}
<h2>Model/module-owned reporter contracts and literature</h2>
<p>See <a href="REPORT_METHODS.md">REPORT_METHODS.md</a>. These profiles affect
presentation only and are resolved from persisted component identities.</p>
{reporter_profile_tables_html}
<h2>Comprehensive comparison and confidence-qualified conclusion</h2>
<p>P values are null-hypothesis tail probabilities, not posterior probabilities.
Repeat t-CIs and participant-cluster bootstrap CIs remain separately labeled.
Cluster CIs resample participant IDs within true-class strata and carry all repeat
OOF predictions; paired CIs apply the same draw to both classifiers before taking
candidate minus reference. Bounds are the 2.5th and 97.5th percentiles.
They are conditional on the frozen dataset/folds/predictions and do not include
dataset shift or model-selection uncertainty.
See <a href="RESULT_INTERPRETATION.md">RESULT_INTERPRETATION.md</a>.</p>
{comparison_tables_html}
<h3>Conclusions by evidence angle</h3>
{_html_table(selection_conclusions, (
    ("angle", "Angle"), ("leading_or_selected_case", "Case"),
    ("finding", "Finding"), ("confidence", "Confidence"),
    ("selection_effect", "Selection effect"),
))}
{reproducibility_html}
<h2>Predictive leaderboard</h2>
<p>Primary ranking uses participant-level, repeat-recomputed abstention-aware
balanced accuracy, then participant coverage and abstention-aware Macro-F1.
Conditional values use retained participants only and do not lead the ranking.
Probability metrics and participant-cluster CIs are reported once in the
comprehensive performance/CI tables above instead of being duplicated here.</p>
{predictive_tables_html}
{top_model_configuration_html}
<h2>Per-class classifier results</h2>
<p>Every classifier with persisted participant OOF probabilities is shown for
every class. Hard-label metrics preserve the persisted decision rule; ROC/PR
metrics are one-vs-rest probability metrics. If a route abstains, a separate
full-roster scope counts abstentions as false negatives for their true class;
ROC/PR remain N/A in that scope. Its one-vs-rest BA is descriptive; global
abstention-aware BA remains the registered macro recall.</p>
{per_class_tables_html}
<h2>Repeat-level predictive distributions</h2>
<p>Mean and sample SD share one percentage column; the adjacent interval is the
two-sided repeat-level Student-t 95% CI. Lossless numeric bounds remain in the
matching JSON table.</p>
{_html_table(compact_rows(getattr(analysis, "metric_distribution_summary", ())), (
    ("case_id", "Case"), ("metric", "Metric"), ("n", "Repeats"),
    ("mean_sd", "Mean ± SD (%)"), ("ci95", "Repeat 95% CI (%)"),
    ("metric_source", "Source"),
))}
<details><summary>Per-class repeat distributions</summary>
{_html_table(compact_rows(getattr(analysis, "per_class_metric_distribution_summary", ())), (
    ("case_id", "Case"), ("class_name", "Class"), ("metric", "Metric"),
    ("n", "Repeats"), ("mean_sd", "Mean ± SD (%)"),
    ("ci95", "Repeat 95% CI (%)"),
))}</details>
<h2>Paired participant-cluster inference</h2>
<p>Each candidate is paired to the declared reference on the exact frozen
participant/repeat/fold/split roster. Shared-draw participant-cluster bootstrap
CIs cover BA, macro-F1 and macro ROC-AUC deltas. Raw/Holm permutation P values
cover BA and macro-F1 only and are comparison evidence, not posterior confidence.</p>
{paired_inference_tables_html}
<h3>Matched per-repeat differences for every declared pair</h3>
<p>Rows use candidate minus reference on an identical participant/fold/split
roster. Design-registered pairs and explicitly labeled post-hoc model
comparisons are kept distinct.</p>
{pairwise_repeat_delta_tables_html}
{bridge_html}
<h2>Aggregation sensitivity from the same file-level OOF</h2>
<p class="notice">The declared-source row reproduces the aggregation used by
the fitted model and, when eligible, the primary leaderboard. The other row
reaggregates the same held-out file probabilities post hoc. It is not a separately retrained
Line A/Line B experiment and is not selection evidence.</p>
{aggregation_line_tables_html}
<h2>Parallel window/file/role-balanced participant views</h2>
<p class="notice">These are three report views of the same fitted held-out OOF,
not three training runs. Equal-window and non-source Line A/Line B views are
post-hoc sensitivity only. Only the declared training aggregation may support
the primary leaderboard.</p>
{aggregation_view_tables_html}
<h3>Hierarchy coverage (B/R1–R4 and B/R)</h3>
{aggregation_hierarchy_tables_html}
<h2>Worst-class F1 stability review</h2>
<p>This secondary ordering uses abstention-aware worst-class F1 and
abstention-aware repeat variability; conditional retained-only values are
shown only for comparison.</p>
{worst_class_stability_tables_html}
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
<h2>Preprocessing-cache operational audit</h2>
<p>Cache reuse is operational provenance only and does not affect predictions,
ranking, labels, fold-fitted artifacts, or route masks. See
<a href="tables/preprocessing_cache.csv">CSV</a> and
<a href="tables/preprocessing_cache.json">JSON</a> for lossless per-cell/per-layer
hit, write, bypass, logical-byte, and elapsed-time evidence.</p>
{_html_table((_preprocessing_cache_overview(getattr(collected, "preprocessing_cache_rows", ())),), (
    ("cell_count", "Cells"), ("cell_layer_rows", "Cell-layer rows"),
    ("event_count", "Events"), ("hit_count", "Hits"),
    ("write_count", "Writes"), ("bypass_count", "Bypasses"),
    ("logical_array_bytes", "Logical array bytes"),
    ("elapsed_seconds", "Cache-operation seconds"),
))}
<h2>Route × role coverage and feature availability</h2>
<p>This formal table uses outer-held-out (<code>outer_oof</code>) records only;
outer-training route rows remain audit-only in the source artifacts.</p>
{route_role_coverage_tables_html}
<h2>SQI state, score, and coverage provenance by each route</h2>
<p>Direct and post-denoiser coverage are separate so the configured
minimum-coverage decision remains auditable.</p>
{sqi_provenance_tables_html}
<h2>Denoiser paired HR/PPI endpoint audit</h2>
<p>HR is 60 / median(valid PPI seconds) from the same registered detector
before and after one denoiser attempt. Same-record pairs are averaged within
participant before participant-macro summaries. Outer-OOF rows are primary;
outer-train rows remain audit-only. HR/PPI endpoint error is absolute
post-denoise minus same-record direct-PPG change; because Frailty29 has no ECG
reference, this is endpoint drift rather than physiological accuracy.</p>
{denoiser_hr_summary_tables_html}
<details><summary>Per-record paired denoiser HR evidence</summary>
{denoiser_hr_record_tables_html}</details>
<h2>Frozen motion evidence used by each route</h2>
<p class="notice">Frailty29 reuse is in-sample auxiliary motion-preprocessing
evidence, not valid outer-OOF motion-detector evidence. Downstream frailty
classification is still evaluated on each outer held-out fold.</p>
{motion_evidence_tables_html}
<h2>Quality-component distributions</h2>
{_html_table(analysis.quality_distributions, (
    ("case_id", "Case"), ("role", "Role"), ("route_state", "Route state"),
    ("component", "Component"), ("valid_count", "Valid n"),
    ("unavailable_rate", "Unavailable"),
    (("mean", "population_sd", False), "Mean ± SD"),
))}
<h2>Classification score, t-SNE, and ROC–AUC diagnostics</h2>
<p>Every classifier with persisted participant OOF probabilities is represented
in the three paired figure modules. t-SNE embeds prediction probabilities, not
hidden features. Multiclass frailty decisions use argmax and have no single
scalar threshold.</p>
{_html_table(getattr(analysis, "classification_diagnostic_status", ()), (
    ("classifier_id", "Classifier"),
    ("prediction_score_status", "Score/threshold"),
    ("prediction_tsne_status", "Prediction t-SNE"),
    ("roc_auc_curve_status", "ROC–AUC curve"),
    ("prediction_row_count", "OOF rows"),
    ("tsne_point_count", "t-SNE points"),
    ("roc_curve_point_count", "ROC points"),
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
        return {
            "study_directory": str(self.study_directory),
            "summary_markdown": str(self.summary_markdown),
            "summary_html": (
                None if self.summary_html is None else str(self.summary_html)
            ),
            "output_index": str(self.output_index),
            "table_count": self.table_count,
            "generated_figure_count": self.generated_figure_count,
            "na_figure_count": self.na_figure_count,
        }


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
            _file_sha256(path)
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
                "sha256": _file_sha256(path),
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
    test_component_rows = build_pipeline_test_component_rows(root, bundle.manifest)
    profile_rows = reporter_profile_rows(test_component_rows)
    report_options = bundle.plan.get("report", {})
    report_options = report_options if isinstance(report_options, Mapping) else {}
    detailed_configuration_top_k = int(
        report_options.get("detailed_configuration_top_k", 0)
    )
    top_model_configuration_rows = build_top_model_configuration_rows(
        root,
        bundle.manifest,
        analysis.predictive_leaderboard,
        top_k=detailed_configuration_top_k,
    )
    manifest_case_ids = tuple(
        str(row["case_id"])
        for row in bundle.manifest.get("cases", ())
        if isinstance(row, Mapping) and row.get("case_id") not in (None, "")
    )
    raw_expected_repeats = bundle.plan.get("execution", {}).get("repeats", ())
    expected_repeats = (
        tuple(int(value) for value in raw_expected_repeats)
        if isinstance(raw_expected_repeats, (list, tuple))
        else ()
    )
    expected_membership: dict[tuple[str, int], tuple[int, int]] = {}
    expected_membership_valid = True
    for split_row in reproducibility.split_rows:
        try:
            repeat = int(split_row["repeat"])
            fold = int(split_row["fold"])
            split_seed = int(split_row["split_seed"])
            participant_ids = split_row["oof_participant_ids"]
        except (KeyError, TypeError, ValueError):
            expected_membership_valid = False
            continue
        if not isinstance(participant_ids, (list, tuple)):
            expected_membership_valid = False
            continue
        for participant_id in participant_ids:
            key = (str(participant_id), repeat)
            value = (fold, split_seed)
            previous = expected_membership.setdefault(key, value)
            if previous != value:
                expected_membership_valid = False
    reproducibility_status = str(
        reproducibility.summary.get("audit_status", NOT_VERIFIABLE)
    )
    frozen_membership = (
        expected_membership
        if expected_membership
        and expected_membership_valid
        and reproducibility_status == "PASS"
        else None
    )
    frozen_registry_failure_reason = (
        None
        if frozen_membership is not None
        else "frozen_split_registry_not_verifiable"
    )
    # Empty expected membership is an intentional fail-closed sentinel: the
    # shared helpers retain the full declared pair/repeat schema but stop before
    # bootstrap or permutation work.  The sentinel reason is made explicit
    # below after all ordinary/centered/cross-model rows have been assembled.
    frozen_membership_guard: Mapping[tuple[str, int], tuple[int, int]] = (
        frozen_membership if frozen_membership is not None else {}
    )
    declared_inference = [
        {
            **dict(row),
            "inference_role": "declared_reference_confirmatory",
        }
        for row in analysis.paired_participant_inference
    ]
    declared_repeat_deltas: list[dict[str, Any]] = []
    additional_pairwise_inference: list[dict[str, Any]] = []
    ordinary_reference_id = bundle.manifest.get("reference_case_id")
    if ordinary_reference_id not in (None, ""):
        ordinary_reference_id = str(ordinary_reference_id)
        observed_reference_membership: dict[
            tuple[str, int], tuple[int, int]
        ] = {}
        ordinary_roster_valid = True
        for row in analysis.classification_prediction_scores:
            if str(row.get("classifier_id")) != ordinary_reference_id:
                continue
            try:
                key = (str(row["participant_id"]), int(row["repeat"]))
                value = (int(row["fold"]), int(row["split_seed"]))
            except (KeyError, TypeError, ValueError):
                ordinary_roster_valid = False
                continue
            previous = observed_reference_membership.setdefault(key, value)
            if previous != value:
                ordinary_roster_valid = False
        ordinary_frozen_roster_pass = bool(
            frozen_membership is not None
            and ordinary_roster_valid
            and observed_reference_membership == frozen_membership
        )
        if not ordinary_frozen_roster_pass:
            reason = (
                "frozen_split_registry_not_verifiable"
                if frozen_membership is None
                else "reference_frozen_split_registry_roster_mismatch"
            )
            declared_inference = _fail_closed_pairwise_rows_for_frozen_registry(
                declared_inference,
                reason=reason,
            )
        declared_repeat_deltas.extend(
            paired_repeat_deltas_against_reference(
                analysis.classification_prediction_scores,
                reference_case_id=ordinary_reference_id,
                comparison_family=(
                    f"{bundle.plan.get('study', {}).get('study_id')}__"
                    "declared_reference"
                ),
                comparison_role="declared_reference_comparison",
                candidate_case_ids=manifest_case_ids,
                expected_repeats=expected_repeats or None,
                expected_membership=frozen_membership_guard,
            )
        )
    centered_comparisons = [
        dict(row)
        for row in bundle.plan.get("legacy_bridge", {}).get(
            "centered_comparisons", ()
        )
        if isinstance(row, Mapping)
    ]
    if centered_comparisons:
        profiles = [
            dict(row)
            for row in bundle.plan.get("legacy_bridge", {}).get("profiles", ())
            if isinstance(row, Mapping)
        ]
        profile_to_catalog = {
            str(row["case_id"]): str(row["catalog_case_id"])
            for row in profiles
        }
        comparison_groups: dict[tuple[str, str], set[str]] = {}
        for comparison in centered_comparisons:
            model_id = str(comparison["model_id"])
            reference_id = profile_to_catalog.get(
                str(comparison["reference_case_id"]),
                str(comparison["reference_case_id"]),
            )
            variant_id = profile_to_catalog.get(
                str(comparison["variant_case_id"]),
                str(comparison["variant_case_id"]),
            )
            comparison_groups.setdefault((model_id, reference_id), {reference_id}).add(
                variant_id
            )
        declared_inference = []
        declared_repeat_deltas = []
        inference_seed = int(
            bundle.plan.get("search", {}).get(
                "selection_seed", DEFAULT_REPORTING_RANDOM_SEED
            )
        )
        for (model_id, reference_id), catalog_ids in sorted(
            comparison_groups.items()
        ):
            selected_predictions = [
                row
                for row in analysis.classification_prediction_scores
                if str(row.get("classifier_id")) in catalog_ids
            ]
            declared_inference.extend(
                paired_inference_against_reference(
                    selected_predictions,
                    reference_case_id=reference_id,
                    comparison_family=(
                        f"{bundle.plan.get('study', {}).get('study_id')}__"
                        f"{model_id}__reference_{reference_id}"
                    ),
                    inference_role="declared_reference_confirmatory",
                    candidate_case_ids=tuple(catalog_ids),
                    expected_repeats=expected_repeats or None,
                    expected_membership=frozen_membership_guard,
                    seed=inference_seed,
                )
            )
            declared_repeat_deltas.extend(
                paired_repeat_deltas_against_reference(
                    selected_predictions,
                    reference_case_id=reference_id,
                    comparison_family=(
                        f"{bundle.plan.get('study', {}).get('study_id')}__"
                        f"{model_id}__reference_{reference_id}"
                    ),
                    comparison_role="predeclared_same_model_centered_ablation",
                    candidate_case_ids=tuple(catalog_ids),
                    expected_repeats=expected_repeats or None,
                    expected_membership=frozen_membership_guard,
                )
            )
        profiles_by_key = {
            (str(row.get("profile_id")), str(row.get("model_id"))): row
            for row in profiles
        }
        for profile_id in sorted(
            {str(row.get("profile_id")) for row in profiles}
        ):
            compact = profiles_by_key.get((profile_id, "CompactCNN1D"))
            inception = profiles_by_key.get((profile_id, "InceptionTimeFull"))
            if compact is None or inception is None:
                continue
            compact_id = str(compact["catalog_case_id"])
            inception_id = str(inception["catalog_case_id"])
            cross_predictions = [
                row
                for row in analysis.classification_prediction_scores
                if str(row.get("classifier_id")) in {compact_id, inception_id}
            ]
            declared_repeat_deltas.extend(
                paired_repeat_deltas_against_reference(
                    cross_predictions,
                    reference_case_id=compact_id,
                    comparison_family=(
                        f"{bundle.plan.get('study', {}).get('study_id')}__"
                        "matched_architecture_comparison"
                    ),
                    comparison_role=(
                        "post_hoc_matched_architecture_model_comparison_not_ablation"
                    ),
                    candidate_case_ids=(compact_id, inception_id),
                    expected_repeats=expected_repeats or None,
                    expected_membership=frozen_membership_guard,
                )
            )
            additional_pairwise_inference.extend(
                paired_inference_against_reference(
                    cross_predictions,
                    reference_case_id=compact_id,
                    comparison_family=(
                        f"{bundle.plan.get('study', {}).get('study_id')}__"
                        "matched_architecture_comparison"
                    ),
                    inference_role=(
                        "exploratory_post_hoc_matched_architecture_comparison_not_ablation"
                    ),
                    candidate_case_ids=(compact_id, inception_id),
                    expected_repeats=expected_repeats or None,
                    expected_membership=frozen_membership_guard,
                    seed=inference_seed,
                )
            )
    if frozen_registry_failure_reason is not None:
        declared_inference = _fail_closed_pairwise_rows_for_frozen_registry(
            declared_inference,
            reason=frozen_registry_failure_reason,
        )
        additional_pairwise_inference = (
            _fail_closed_pairwise_rows_for_frozen_registry(
                additional_pairwise_inference,
                reason=frozen_registry_failure_reason,
            )
        )
        declared_repeat_deltas = _fail_closed_pairwise_rows_for_frozen_registry(
            declared_repeat_deltas,
            reason=frozen_registry_failure_reason,
        )
    report_pairwise_inference = holm_adjust_paired_inference_rows(
        [*declared_inference, *additional_pairwise_inference]
    )
    comparison_conclusion_rows = classification_comparison_rows(
        analysis.case_summary,
        paired_inference=declared_inference,
    )
    selection_conclusion_rows = classification_conclusion_rows(
        comparison_conclusion_rows,
        selected_case_id=None,
        selection_basis="manual review only; no automatic selection in ordinary study reporter",
        study_role=str(bundle.plan.get("study", {}).get("decision_role", "study")),
        planned_case_count=int(bundle.manifest.get("planned_case_count", 0) or 0),
        incomplete_case_count=len(analysis.incomplete_cases),
        inference_reference_case_ids=tuple(
            sorted(
                {
                    str(row["reference_case_id"])
                    for row in declared_inference
                    if row.get("reference_case_id") not in (None, "")
                }
            )
        ),
    )
    table_payloads: tuple[tuple[str, Sequence[Mapping[str, Any]], str], ...] = (
        (
            "test_components",
            test_component_rows,
            "All participating models/modules with named input data and complete fixed parameters",
        ),
        *(
            (
                (
                    "top_model_complete_configurations",
                    top_model_configuration_rows,
                    "Complete long-form resolved configuration for the configured top predictive-ranked models",
                ),
            )
            if detailed_configuration_top_k > 0
            else ()
        ),
        (
            "reporter_profiles",
            profile_rows,
            "Model/module-owned reporter contracts, methods, required outputs, and literature",
        ),
        (
            "comparison_conclusions",
            comparison_conclusion_rows,
            "Comprehensive BA/F1/ROC-AUC/SD/CI/P comparison table",
        ),
        (
            "selection_conclusions",
            selection_conclusion_rows,
            "Evidence-angle conclusions and explicitly qualified selection confidence",
        ),
        (
            "pairwise_repeat_metric_deltas",
            declared_repeat_deltas,
            "One matched row per declared comparison and repeat with BA, macro-F1, and macro ROC-AUC candidate-minus-reference differences",
        ),
        ("case_summary", analysis.case_summary, "One descriptive row per case"),
        (
            "classification_prediction_scores",
            analysis.classification_prediction_scores,
            "Per-classifier persisted OOF prediction probabilities, confidence, and decision-threshold provenance",
        ),
        (
            "classification_prediction_tsne",
            analysis.classification_prediction_tsne,
            "Deterministic report-only t-SNE coordinates from persisted OOF probability vectors",
        ),
        (
            "classification_roc_curves",
            analysis.classification_roc_curves,
            "Empirical one-vs-rest and macro-average OOF ROC curve coordinates with AUC",
        ),
        (
            "classification_diagnostic_status",
            analysis.classification_diagnostic_status,
            "Per-classifier availability audit for score, prediction-space t-SNE, and ROC-AUC figures",
        ),
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
            "classifier_per_class_results",
            analysis.classifier_per_class_results,
            "Every classifier/evaluation/level/class one-vs-rest result recomputed from persisted prediction probabilities and decision labels",
        ),
        (
            "repeat_per_class_metrics",
            analysis.repeat_per_class_metrics,
            "Per-repeat participant OOF one-vs-rest BA/F1/ROC-AUC/PR-AUC by class",
        ),
        (
            "per_class_metric_distribution_summary",
            analysis.per_class_metric_distribution_summary,
            "Repeat mean/SD by case, class, and one-vs-rest predictive metric",
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
        (
            "paired_deltas",
            analysis.paired_deltas,
            "Legacy descriptive repeat-paired deltas; the strict canonical complete-roster table is pairwise_repeat_metric_deltas",
        ),
        (
            "paired_participant_inference",
            report_pairwise_inference,
            "Canonical participant-cluster CI/P table for every declared ablation and explicitly labeled post-hoc matched-model comparison; Holm is applied over each complete family × metric",
        ),
        ("coverage", analysis.coverage, "Coverage and quality diagnostic counts"),
        (
            "route_role_coverage",
            analysis.route_role_coverage,
            "Tier, motion provenance, abstention, retained/direct/processed, and denoiser summaries by route and role",
        ),
        (
            "denoiser_hr_comparison",
            analysis.denoiser_hr_comparison,
            "Participant-macro paired direct/post-denoiser PPG HR/PPI endpoint audit, Q_rate recovery, and reducer failures",
        ),
        (
            "denoiser_hr_record_pairs",
            analysis.denoiser_hr_record_pairs,
            "Per-record paired direct/post-denoiser PPG heart-rate evidence",
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
        (
            "preprocessing_cache",
            bundle.preprocessing_cache_rows,
            "Operational per-cell/per-layer preprocessing-cache hits, writes, bypasses, logical bytes, and elapsed time",
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
    declared_table_names = {name for name, _rows, _description in table_payloads}
    profile_required_tables = {
        str(table_name)
        for profile in profile_rows
        for table_name in profile.get("required_tables", ())
    }
    missing_profile_tables = profile_required_tables - declared_table_names
    if missing_profile_tables:
        raise ValueError(
            "reporter profile requires unregistered ordinary-report tables: "
            f"{sorted(missing_profile_tables)}"
        )
    index: list[dict[str, Any]] = []
    table_file_count = 0
    use_compact_tables = bool(report_options.get("compact_mean_sd", True))
    report_tables = [
        ReportTable(
            name=name,
            rows=rows,
            description=description,
            compact=use_compact_tables,
        )
        for name, rows, description in table_payloads
    ]
    for name, rows, description in table_payloads:
        csv_path = tables / f"{name}.csv"
        json_path = tables / f"{name}.json"
        _write_csv(csv_path, compact_rows(rows) if use_compact_tables else rows)
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
    write_figures = bool(report_options.get("write_static_figures", True))
    raw_figure_modules = report_options.get("figure_modules", ("all",))
    figure_modules = (
        (raw_figure_modules,)
        if isinstance(raw_figure_modules, str)
        else tuple(raw_figure_modules)
    )
    if "all" not in figure_modules:
        figure_modules = tuple(
            dict.fromkeys(
                (*figure_modules, *required_figure_modules(test_component_rows))
            )
        )
    if not write_figures:
        clear_static_figure_artifacts(figures_dir)
    figures = (
        generate_static_figures(
            bundle,
            analysis,
            figures_dir,
            modules=figure_modules,
        )
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
    generated_or_na_figure_names = {
        str(row.get("figure")) for row in figures if row.get("figure")
    }
    missing_profile_figures = set(
        required_figure_modules(test_component_rows)
    ) - generated_or_na_figure_names
    if missing_profile_figures:
        raise ValueError(
            "reporter profile figure contract was silently omitted: "
            f"{sorted(missing_profile_figures)}"
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
    table_names = {table.name for table in report_tables}
    table_figure_pairs: list[dict[str, Any]] = []
    for figure in figures:
        figure_name = str(figure.get("figure", ""))
        for table_name in FIGURE_TABLE_SOURCES.get(figure_name, ()):
            table_figure_pairs.append(
                {
                    "table": table_name,
                    "table_status": (
                        "available" if table_name in table_names else "not_registered"
                    ),
                    "figure": figure_name,
                    "figure_status": figure.get("status"),
                    "figure_path": figure.get("path", ""),
                    "reason": figure.get("reason", ""),
                }
            )
    pair_table = ReportTable(
        name="table_figure_pairs",
        rows=table_figure_pairs,
        description="Modular source-table to figure pairing registry",
        compact=False,
    )
    report_tables.append(pair_table)
    pair_csv = tables / "table_figure_pairs.csv"
    pair_json = tables / "table_figure_pairs.json"
    _write_csv(pair_csv, table_figure_pairs)
    _write_json(pair_json, table_figure_pairs)
    index.extend(
        (
            _index_entry(
                root,
                pair_csv,
                artifact_type="table_csv",
                description=pair_table.description,
                status="available" if table_figure_pairs else "N/A_no_rows",
            ),
            _index_entry(
                root,
                pair_json,
                artifact_type="table_json",
                description=pair_table.description,
                status="available" if table_figure_pairs else "N/A_no_rows",
            ),
        )
    )
    table_file_count += 2
    definition_csv, definition_json, definition_markdown = (
        write_table_column_definitions(
            tables,
            csv_directory=tables,
        )
    )
    for definition_path, artifact_type in (
        (definition_csv, "table_column_definitions_csv"),
        (definition_json, "table_column_definitions_json"),
        (definition_markdown, "table_column_definitions_markdown"),
    ):
        index.append(
            _index_entry(
                root,
                definition_path,
                artifact_type=artifact_type,
                description=(
                    "Central definition and calculation-formula catalog for "
                    "every persisted root report-table column"
                ),
            )
        )
    table_file_count += 3
    if bool(report_options.get("write_excel_workbook", True)):
        workbook = tables / "report_tables.xlsx"
        # Build only after the final root CSV (table_figure_pairs.csv) exists;
        # the persisted CSV directory is the authoritative one-sheet-per-table
        # workbook roster.
        write_excel_workbook_from_csv_directory(workbook, tables)
        index.append(
            _index_entry(
                root,
                workbook,
                artifact_type="table_workbook",
                description="One worksheet per persisted root CSV report table",
            )
        )
        table_file_count += 1
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
        "table_figure_pairs": table_figure_pairs,
        "test_components": test_component_rows,
        "top_model_complete_configurations": top_model_configuration_rows,
        "reporter_profiles": profile_rows,
        "comparison_conclusions": comparison_conclusion_rows,
        "selection_conclusions": selection_conclusion_rows,
        "pairwise_participant_cluster_inference": report_pairwise_inference,
        "pairwise_repeat_metric_deltas": declared_repeat_deltas,
        "preprocessing_cache": bundle.preprocessing_cache_rows,
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
    component_markdown = write_test_component_markdown(root, test_component_rows)
    index.append(
        _index_entry(
            root,
            component_markdown,
            artifact_type="test_component_contract",
            description=(
                "Standalone copy of the report's model/module, input-data, and "
                "fixed-parameter table"
            ),
        )
    )
    methods_markdown = write_reporter_methods(root, test_component_rows)
    index.append(
        _index_entry(
            root,
            methods_markdown,
            artifact_type="reporter_methods_contract",
            description="Model/module-owned reporter methods and literature",
        )
    )
    interpretation_markdown = write_result_interpretation(
        root,
        comparison_rows=comparison_conclusion_rows,
        conclusion_rows=selection_conclusion_rows,
        paired_inference=report_pairwise_inference,
        split_classification_comparison=True,
    )
    index.append(
        _index_entry(
            root,
            interpretation_markdown,
            artifact_type="result_interpretation",
            description="Confidence-qualified comparison conclusions",
        )
    )
    markdown_path = root / "STUDY_SUMMARY.md"
    markdown_path.write_text(
        _report_markdown(
            bundle,
            analysis,
            figures,
            reproducibility,
            test_component_rows,
            profile_rows,
            comparison_conclusion_rows,
            selection_conclusion_rows,
            report_pairwise_inference,
            declared_repeat_deltas,
            top_model_configuration_rows,
        ),
        encoding="utf-8",
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
            _report_html(
                bundle,
                analysis,
                figures,
                reproducibility,
                test_component_rows,
                profile_rows,
                comparison_conclusion_rows,
                selection_conclusion_rows,
                report_pairwise_inference,
                declared_repeat_deltas,
                top_model_configuration_rows,
            ),
            encoding="utf-8",
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

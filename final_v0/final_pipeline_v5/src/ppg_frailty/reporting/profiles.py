"""Model-owned report profiles shared by every analysis entry point."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..module_registry import component_reporter_binding
from .tabular import markdown_column_definitions_block


@dataclass(frozen=True)
class ReporterProfile:
    profile_id: str
    title: str
    algorithm_summary: str
    statistical_methods: tuple[str, ...]
    required_tables: tuple[str, ...]
    required_figures: tuple[str, ...]
    literature: tuple[str, ...]
    limitations: tuple[str, ...] = ()


REPORTER_PROFILE_VIEW_SCHEMAS = (
    (
        "Profile identity",
        (
            ("profile_id", "Profile"),
            ("title", "Title"),
            ("profile_kind", "Kind"),
            ("participating_components", "Components"),
        ),
    ),
    ("Required outputs", (("profile_id", "Profile"), ("required_tables", "Tables"), ("required_figures", "Figures"))),
    (
        "Methods and provenance",
        (
            ("profile_id", "Profile"),
            ("algorithm_summary", "Algorithm"),
            ("statistical_methods", "Statistics"),
            ("literature", "Literature"),
            ("limitations", "Limitations"),
        ),
    ),
)

_CLASSIFICATION_TABLES = ("case_summary", "repeat_metrics", "classifier_per_class_results")
_CLASSIFICATION_FIGURES = ("leaderboard", "stability", "classification_roc_auc_curves", "confusion_matrices")
_LEARNING_FIGURES = ("learning_curves", "top_learning_curves", "balanced_accuracy_learning_curves")
_PROFILES = {
    "inceptiontime_single_network_model_v1": (
        "InceptionTime single network",
        _CLASSIFICATION_TABLES + ("training_history_raw",),
        _CLASSIFICATION_FIGURES + _LEARNING_FIGURES,
    ),
    "inceptiontime_matrix_model_v1": (
        "Matrix InceptionTime",
        _CLASSIFICATION_TABLES + ("training_history_raw",),
        _CLASSIFICATION_FIGURES + _LEARNING_FIGURES,
    ),
    "compactcnn_model_v1": (
        "CompactCNN1D",
        _CLASSIFICATION_TABLES + ("training_history_raw",),
        _CLASSIFICATION_FIGURES + _LEARNING_FIGURES,
    ),
    "logistic_l2_model_v1": ("L2 logistic regression", _CLASSIFICATION_TABLES, _CLASSIFICATION_FIGURES),
    "inceptiontime_probability_ensemble_model_v1": (
        "InceptionTime probability ensemble",
        _CLASSIFICATION_TABLES + ("ensemble_member_metrics",),
        _CLASSIFICATION_FIGURES + ("ensemble_member_metrics",),
    ),
    "rbf_svm_model_v1": ("RBF support-vector classifier", _CLASSIFICATION_TABLES, _CLASSIFICATION_FIGURES),
    "extra_trees_model_v1": ("Extra-trees classifier", _CLASSIFICATION_TABLES, _CLASSIFICATION_FIGURES),
    "shapeformer_model_v1": ("ShapeFormer project adaptation", _CLASSIFICATION_TABLES, _CLASSIFICATION_FIGURES),
    "file_bag_fusion_model_v1": ("File-bag fusion classifier", _CLASSIFICATION_TABLES, _CLASSIFICATION_FIGURES),
    "multiclass_participant_oof_v1": ("Multiclass participant OOF", _CLASSIFICATION_TABLES, _CLASSIFICATION_FIGURES),
    "binary_motion_window_file_v1": ("Binary motion window/file evaluation", ("test_components",), ()),
    "motion_route_component_v1": (
        "Motion routing",
        ("route_role_coverage", "quality_distributions"),
        ("route_role_coverage", "quality_distributions"),
    ),
    "beat_detector_recording_v1": ("Beat detector recording endpoints", ("test_components",), ()),
    "beat_detector_legacy_persisted_v1": ("Legacy beat detector endpoints", ("test_components",), ()),
    "stage5_ecg_ppg_denoiser_v1": (
        "ECG–PPG denoiser endpoints",
        ("denoiser_hr_comparison", "denoiser_hr_record_pairs"),
        ("denoiser_hr_comparison",),
    ),
    "frailty_denoiser_route_v1": (
        "Denoiser in frailty routing",
        ("denoiser_hr_comparison", "coverage"),
        ("denoiser_hr_comparison", "coverage"),
    ),
    "sqi_route_coverage_v1": (
        "SQI routing and coverage",
        ("route_role_coverage", "quality_distributions", "coverage"),
        ("route_role_coverage", "quality_distributions", "coverage"),
    ),
    "audit_provenance_v1": ("Configuration and provenance audit", ("test_components", "reproducibility_summary"), ()),
}


def _profile(profile_id: str, title: str, tables: tuple[str, ...], figures: tuple[str, ...]) -> ReporterProfile:
    return ReporterProfile(
        profile_id,
        title,
        "The persisted module configuration and prediction artifacts determine this presentation; reporting never refits or changes predictions.",
        (
            "Descriptive summaries use persisted evaluation units; registered inferential modules retain their declared resampling unit and seed.",
        ),
        tables,
        figures,
        (),
        ("Interpret results only for the persisted evaluation scope and roster.",),
    )


REPORTER_PROFILES = {key: _profile(key, *value) for key, value in _PROFILES.items()}
_ALIASES = {"denoiser_ecg_ppg_endpoint_v1": "stage5_ecg_ppg_denoiser_v1"}


def _profile_id(role: str) -> str:
    role = role.lower()
    if role.startswith("classifier"):
        return "multiclass_participant_oof_v1"
    if role.startswith("motion"):
        return "binary_motion_window_file_v1"
    if role == "denoiser":
        return "stage5_ecg_ppg_denoiser_v1"
    if role == "sqi":
        return "sqi_route_coverage_v1"
    return "audit_provenance_v1"


def annotate_component_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    state = str(output.get("execution_state", "")).lower()
    inactive = any(word in state for word in ("disabled", "not_executed", "not_run"))
    profile_id = _ALIASES.get(
        str(output.get("reporter_profile_id", "")).strip(), str(output.get("reporter_profile_id", "")).strip()
    )
    profile_id = profile_id or (
        "audit_provenance_v1" if inactive else _profile_id(str(output.get("component_role", "")))
    )
    if profile_id not in REPORTER_PROFILES:
        raise ValueError(f"unknown reporter profile: {profile_id}")
    binding = component_reporter_binding(
        str(output.get("component_role", "")), str(output.get("module_id", "")), active=not inactive
    )
    extension = str(binding["reporter_extension_id"])
    if extension != "not_applicable" and extension not in REPORTER_PROFILES:
        raise ValueError(f"unknown reporter extension: {extension}")
    output.update(
        {
            "reporter_profile_id": profile_id,
            "model_reporter_extension_id": extension,
            "algorithm_kernel_description": output.get("algorithm_kernel_description") or binding["algorithm_summary"],
            "algorithm_references": "; ".join(binding["references"]) or "N/A — component was not executed",
            "registered_module_id": binding["registered_module_id"],
            "registered_module_family": binding["registered_module_family"],
            "reporter_binding_kind": binding["reporter_binding_kind"],
            "reporter_binding_source": binding["reporter_binding_source"],
        }
    )
    return output


def annotate_component_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [annotate_component_row(row) for row in rows]


def reporter_profile_rows(component_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in annotate_component_rows(component_rows):
        grouped.setdefault(str(row["reporter_profile_id"]), []).append(row)
        if row["model_reporter_extension_id"] in REPORTER_PROFILES:
            grouped.setdefault(str(row["model_reporter_extension_id"]), []).append(row)
    return [
        {
            **asdict(REPORTER_PROFILES[key]),
            "profile_kind": "model_or_module_extension"
            if any(r.get("model_reporter_extension_id") == key for r in rows)
            else "endpoint_or_module",
            "participating_components": sorted({f"{r.get('component_role')}:{r.get('module_id')}" for r in rows}),
            "module_references": sorted({str(r.get("algorithm_references", "")) for r in rows}),
            "presentation_only": True,
            "changes_training_or_predictions": False,
        }
        for key, rows in sorted(grouped.items())
    ]


def required_figure_modules(component_rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    rows = reporter_profile_rows(component_rows)
    return tuple(sorted({figure for row in rows for figure in row["required_figures"]}))


def _cell(value: Any) -> str:
    return "; ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)


def markdown_reporter_profile_tables(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "N/A — no reporter profile was selected."
    sections = []
    for title, schema in REPORTER_PROFILE_VIEW_SCHEMAS:
        fields, labels = zip(*schema, strict=True)
        lines = [f"### {title}", "", "| " + " | ".join(labels) + " |", "|" + "|".join("---" for _ in fields) + "|"]
        lines += [
            "| "
            + " | ".join(_cell(row.get(field, "")).replace("|", r"\|").replace("\n", " ") for field in fields)
            + " |"
            for row in rows
        ]
        lines += ["", markdown_column_definitions_block(fields, display_labels=labels)]
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def reporter_methods_markdown(component_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Reporter methods and literature",
        "",
        "Profiles select presentation only; training and predictions remain unchanged.",
        "",
    ]
    for row in reporter_profile_rows(component_rows):
        lines += [
            f"## {row['title']} (`{row['profile_id']}`)",
            "",
            str(row["algorithm_summary"]),
            "",
            "Required tables: " + ", ".join(row["required_tables"]),
            "Required figures: " + (", ".join(row["required_figures"]) or "none"),
            "",
        ]
    return "\n".join(lines)


def write_reporter_methods(root: str | Path, component_rows: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(root) / "REPORT_METHODS.md"
    target.write_text(reporter_methods_markdown(component_rows), encoding="utf-8")
    return target


__all__ = [
    "REPORTER_PROFILES",
    "REPORTER_PROFILE_VIEW_SCHEMAS",
    "ReporterProfile",
    "annotate_component_row",
    "annotate_component_rows",
    "markdown_reporter_profile_tables",
    "reporter_methods_markdown",
    "reporter_profile_rows",
    "required_figure_modules",
    "write_reporter_methods",
]

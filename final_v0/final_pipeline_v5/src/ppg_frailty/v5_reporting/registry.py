"""Declarative table/figure modules used by CLI, Dash and the writer."""

from __future__ import annotations

from collections.abc import Iterable

from ppg_frailty.reporting.plots import STATIC_FIGURE_NAMES

from .contracts import ModuleSpec, REPORT_MODES, ReportContractError, ResolvedSelection

ALL = frozenset(REPORT_MODES)
OOF = frozenset({"single", "comparison", "ablation"})


def _spec(
        name: str,
        tables: str = "",
        figures: str = "",
        *,
        modes: frozenset[str] = ALL,
        dependencies: tuple[str, ...] = (),
) -> ModuleSpec:
    return ModuleSpec(
        name,
        modes,
        tuple(tables.split()),
        tuple(figures.split()),
        dependencies,
    )


MODULES = (
    _spec(
        "audit",
        "input_artifacts input_manifests resolved_config_parameters "
        "fold_model_parameters test_components reporter_profiles "
        "top_model_complete_configurations varied_parameters controlled_parameters "
        "reproducibility_summary reproducibility_cases reproducibility_cells "
        "reproducibility_splits reproducibility_issues case_records cell_metrics_raw",
    ),
    _spec(
        "predictions",
        "window_predictions file_predictions role_predictions "
        "participant_predictions member_predictions",
    ),
    _spec(
        "summary",
        "case_summary metric_distribution_summary repeat_metrics fold_metrics "
        "predictive_leaderboard incomplete_cases worst_class_f1_stability "
        "comparison_conclusions selection_conclusions",
        "leaderboard stability macro_f1_stability",
    ),
    _spec(
        "prediction_scores",
        "classification_prediction_scores classification_diagnostic_status",
        "classification_prediction_scores",
    ),
    _spec(
        "roc_auc",
        "classification_roc_curves classification_diagnostic_status",
        "classification_roc_auc_curves roc_pr_auc_stability",
    ),
    _spec(
        "tsne",
        "classification_prediction_tsne classification_diagnostic_status",
        "classification_prediction_tsne",
    ),
    _spec(
        "confusion",
        "confusion_matrices confusion_counts confusion_row_normalized",
        "confusion_matrices confusion_matrices_row_normalized",
    ),
    _spec("calibration", "calibration_bins", "calibration"),
    _spec(
        "per_class",
        "per_class_metrics classifier_per_class_results repeat_per_class_metrics "
        "per_class_metric_distribution_summary",
        "per_class per_class_metric_stability worst_class_f1_stability",
    ),
    _spec(
        "learning",
        "training_history_raw",
        "learning_curves top_learning_curves balanced_accuracy_learning_curves "
        "top_balanced_accuracy_learning_curves",
    ),
    _spec("coverage", "coverage", "coverage"),
    _spec(
        "hierarchy",
        "aggregation_line_comparison aggregation_line_repeat_metrics "
        "aggregation_line_per_class_metrics aggregation_view_comparison "
        "aggregation_view_repeat_metrics aggregation_view_per_class_metrics "
        "aggregation_view_confusion_matrices aggregation_hierarchy_coverage",
        "aggregation_view_metrics aggregation_hierarchy_coverage "
        "aggregation_view_confusion_matrices "
        "aggregation_view_confusion_matrices_row_normalized aggregation_view_per_class",
    ),
    _spec(
        "quality",
        "route_role_coverage quality_distributions denoiser_hr_comparison "
        "denoiser_hr_record_pairs quality_diagnostics_raw",
        "route_role_coverage quality_distributions denoiser_hr_comparison",
    ),
    _spec(
        "comparison",
        "paired_deltas paired_participant_inference pairwise_repeat_metric_deltas",
        "paired_deltas fold_heatmap",
        modes=frozenset({"comparison", "ablation"}),
        dependencies=("summary", ),
    ),
    _spec(
        "ablation",
        "ablation_contract paired_deltas paired_participant_inference",
        "ablation_sensitivity_metrics parameter_effects parameter_interaction",
        modes=frozenset({"ablation"}),
        dependencies=("comparison", ),
    ),
    _spec(
        "ensemble",
        "ensemble_member_predictions ensemble_member_metrics",
        "ensemble_member_metrics",
    ),
    _spec("operations", "deployment_measurements preprocessing_cache"),
    _spec(
        "historical",
        "legacy_bridge_numeric_ablation_report legacy_bridge_execution_order_report "
        "stage3_star_absolute stage3_star_contrasts stage3_star_fold_contrasts "
        "stage3_star_execution stage3_star_inception_comparison "
        "stage3_star_cnn_comparison stage3_star_model_comparison",
        "legacy_bridge_numeric_ablation_report legacy_bridge_execution_order_report "
        "stage3_star_model_deltas stage3_star_fold_delta_heatmap",
        modes=OOF,
    ),
)

MODULE_BY_NAME = {module.name: module for module in MODULES}
PRESETS = {
    "minimal": ("audit", "summary", "roc_auc", "confusion"),
    "classification": (
        "audit",
        "summary",
        "prediction_scores",
        "roc_auc",
        "confusion",
        "calibration",
        "per_class",
        "learning",
        "coverage",
        "hierarchy",
    ),
    "test": (
        "audit",
        "summary",
        "prediction_scores",
        "roc_auc",
        "confusion",
        "calibration",
        "per_class",
        "coverage",
    ),
    "comparison": ("audit", "summary", "roc_auc", "confusion", "comparison"),
    "ablation": (
        "audit",
        "summary",
        "roc_auc",
        "confusion",
        "comparison",
        "ablation",
    ),
    "ensemble": ("audit", "summary", "ensemble"),
    "full": (),
}
KNOWN_TABLES = frozenset(table for module in MODULES for table in module.tables)
KNOWN_FIGURES = frozenset(STATIC_FIGURE_NAMES) | {"ensemble_member_metrics"}


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def resolve_selection(
    *,
    mode: str,
    presets: Iterable[str],
    modules: Iterable[str],
    figures: tuple[str, ...] | None,
    tables: tuple[str, ...] | None,
) -> ResolvedSelection:
    requested_presets, requested_modules = _unique(presets), _unique(modules)
    if unknown := set(requested_presets) - PRESETS.keys():
        raise ReportContractError(f"unknown report preset(s): {sorted(unknown)}")
    if unknown := set(requested_modules) - MODULE_BY_NAME.keys():
        raise ReportContractError(f"unknown report module(s): {sorted(unknown)}")
    if not requested_presets and not requested_modules:
        requested_presets = ({
            "single": "minimal",
            "comparison": "comparison",
            "ablation": "ablation",
            "test": "test",
        }[mode], )
    expanded = []
    for preset in requested_presets:
        expanded.extend((module.name for module in MODULES
                         if mode in module.modes) if preset == "full" else PRESETS[preset])
    expanded.extend(requested_modules)

    resolved: list[str] = []

    def add(name: str) -> None:
        for dependency in MODULE_BY_NAME[name].dependencies:
            add(dependency)
        if name not in resolved:
            resolved.append(name)

    for name in _unique(expanded):
        add(name)
    unsupported = [name for name in resolved if mode not in MODULE_BY_NAME[name].modes]
    if unsupported:
        raise ReportContractError(f"module(s) unavailable in {mode!r} mode: {unsupported}")

    defaults = lambda attr: _unique(  # noqa: E731
        value for name in resolved for value in getattr(MODULE_BY_NAME[name], attr))
    chosen_tables = defaults("tables") if tables is None else _unique(tables)
    chosen_figures = defaults("figures") if figures is None else _unique(figures)
    allowed_tables = {value for spec in MODULES if mode in spec.modes for value in spec.tables}
    allowed_figures = {value for spec in MODULES if mode in spec.modes for value in spec.figures}
    if unknown := set(chosen_tables) - allowed_tables:
        raise ReportContractError(f"table(s) unavailable in {mode!r} mode: {sorted(unknown)}")
    if unknown := set(chosen_figures) - allowed_figures:
        raise ReportContractError(f"figure(s) unavailable in {mode!r} mode: {sorted(unknown)}")
    return ResolvedSelection(tuple(resolved), chosen_tables, chosen_figures)


def validate_registry() -> None:
    """Compatibility hook; the registry is constructed from one typed source."""


__all__ = [
    "KNOWN_FIGURES",
    "KNOWN_TABLES",
    "MODULES",
    "MODULE_BY_NAME",
    "PRESETS",
    "resolve_selection",
    "validate_registry",
]

"""Model/module-owned reporter profiles and literature provenance.

Reporter profiles are presentation contracts.  They select report tables,
figures, methods text, and citations from persisted results; they never alter
training, prediction, thresholds, ranking, or study eligibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..module_registry import component_reporter_binding
from .tabular import markdown_column_definitions_block


@dataclass(frozen=True)
class ReporterProfile:
    """One reusable, model/module-owned reporting contract."""

    profile_id: str
    title: str
    algorithm_summary: str
    statistical_methods: tuple[str, ...]
    required_tables: tuple[str, ...]
    required_figures: tuple[str, ...]
    literature: tuple[str, ...]
    limitations: tuple[str, ...] = ()


# ``reporter_profile_rows`` remains the lossless machine-facing schema.  These
# views are the shared projection for reports that choose to show the profile
# registry as tables instead of linking to REPORT_METHODS.md.
REPORTER_PROFILE_VIEW_SCHEMAS: tuple[
    tuple[str, tuple[tuple[str, str], ...]], ...
] = (
    (
        "Profile identity and participating components",
        (
            ("profile_id", "Profile ID"),
            ("title", "Profile title"),
            ("profile_kind", "Profile kind"),
            ("participating_components", "Participating components"),
            ("presentation_only", "Presentation only"),
            ("changes_training_or_predictions", "Changes training/predictions"),
        ),
    ),
    (
        "Required outputs",
        (
            ("profile_id", "Profile ID"),
            ("required_tables", "Required tables"),
            ("required_figures", "Required figures"),
        ),
    ),
    (
        "Methods, limitations, and provenance",
        (
            ("profile_id", "Profile ID"),
            ("algorithm_summary", "Algorithm summary"),
            ("statistical_methods", "Statistical/reporting methods"),
            ("limitations", "Limitations"),
            ("literature", "Profile literature"),
            ("module_references", "Module references"),
        ),
    ),
)

_MAX_HUMAN_FACING_PROFILE_COLUMNS = 8


_CLASSIFICATION_LITERATURE = (
    "Brodersen et al. (2010), balanced accuracy, DOI:10.1109/ICPR.2010.764",
    "Sokolova & Lapalme (2009), classification measures including F-score, DOI:10.1016/j.ipm.2009.03.002",
    "Fawcett (2006), ROC analysis, DOI:10.1016/j.patrec.2005.10.010",
    "Efron & Tibshirani (1993), An Introduction to the Bootstrap, DOI:10.1007/978-1-4899-4541-9",
    "Holm (1979), sequentially rejective multiple testing, DOI:10.2307/4615733",
)


REPORTER_PROFILES: Mapping[str, ReporterProfile] = {
    "inceptiontime_single_network_model_v1": ReporterProfile(
        profile_id="inceptiontime_single_network_model_v1",
        title="InceptionTime single-network model extension",
        algorithm_summary=(
            "The project Full/Small routes use one InceptionTime-style network "
            "with bottlenecked parallel temporal convolutions and residual blocks; "
            "they are not the original five-member probability ensemble."
        ),
        statistical_methods=(
            "Epoch loss and held-out participant BA learning curves are reported for diagnostic review only; fixed epochs are not selected from outer-test labels.",
        ),
        required_tables=("training_history_raw", "test_components"),
        required_figures=(
            "learning_curves",
            "top_learning_curves",
            "balanced_accuracy_learning_curves",
            "top_balanced_accuracy_learning_curves",
        ),
        literature=(),
        limitations=(
            "Results must be described as a single-network adaptation unless an explicit ensemble module was executed.",
        ),
    ),
    "inceptiontime_matrix_model_v1": ReporterProfile(
        profile_id="inceptiontime_matrix_model_v1",
        title="Matrix InceptionTime model extension",
        algorithm_summary=(
            "Project adaptation applies an InceptionTime-style single network to "
            "the ordered feature-matrix axis and preserves the persisted matrix mask/shape contract."
        ),
        statistical_methods=(
            "Matrix shape, mask, feature count and temporal K are reported from the resolved input contract.",
            "Deep learning curves are diagnostic and do not alter fixed-epoch evaluation.",
        ),
        required_tables=("training_history_raw", "test_components"),
        required_figures=(
            "learning_curves",
            "top_learning_curves",
            "balanced_accuracy_learning_curves",
        ),
        literature=(),
        limitations=("This matrix adaptation is project-specific, not an original-paper input route.",),
    ),
    "compactcnn_model_v1": ReporterProfile(
        profile_id="compactcnn_model_v1",
        title="CompactCNN1D model extension",
        algorithm_summary=(
            "Project CompactCNN1D uses three temporal convolution stages and a "
            "pooled classification head over the persisted raw-channel tensor."
        ),
        statistical_methods=(
            "Training and held-out participant BA learning curves are diagnostic; fixed epochs remain frozen.",
        ),
        required_tables=("training_history_raw", "test_components"),
        required_figures=(
            "learning_curves",
            "top_learning_curves",
            "balanced_accuracy_learning_curves",
        ),
        literature=(),
    ),
    "logistic_l2_model_v1": ReporterProfile(
        profile_id="logistic_l2_model_v1",
        title="L2 logistic-regression model extension",
        algorithm_summary=(
            "One L2-regularized multinomial logistic estimator is fitted per "
            "outer-training fold on the engineered feature vector."
        ),
        statistical_methods=(
            "Estimator coefficients/configuration and outer-OOF endpoints are reported; epoch/learning-curve sections are not applicable.",
        ),
        required_tables=("test_components", "case_summary"),
        required_figures=(),
        literature=(),
    ),
    "inceptiontime_probability_ensemble_model_v1": ReporterProfile(
        profile_id="inceptiontime_probability_ensemble_model_v1",
        title="InceptionTime probability-ensemble model extension",
        algorithm_summary=(
            "Persisted independently seeded member networks are reported separately, "
            "and their class probabilities are combined by the configured exact mean."
        ),
        statistical_methods=(
            "Member rosters and member-level training histories are provenance; outer-held-out ensemble probabilities drive classification endpoints.",
        ),
        required_tables=("training_history_raw", "test_components"),
        required_figures=(
            "learning_curves",
            "balanced_accuracy_learning_curves",
        ),
        literature=(),
        limitations=(
            "The historical five-member name does not establish that five members ran; the persisted member roster is authoritative.",
        ),
    ),
    "rbf_svm_model_v1": ReporterProfile(
        profile_id="rbf_svm_model_v1",
        title="RBF-SVM model extension",
        algorithm_summary=(
            "One RBF-kernel support-vector classifier is fitted per outer-training "
            "fold after fold-local feature preprocessing."
        ),
        statistical_methods=(
            "Persisted estimator configuration and outer-OOF endpoints are reported; epoch curves are not applicable.",
        ),
        required_tables=("test_components", "case_summary"),
        required_figures=(),
        literature=(),
    ),
    "extra_trees_model_v1": ReporterProfile(
        profile_id="extra_trees_model_v1",
        title="Extra Trees model extension",
        algorithm_summary=(
            "One extremely randomized tree ensemble is fitted per outer-training "
            "fold after fold-local feature imputation."
        ),
        statistical_methods=(
            "Persisted estimator/tree configuration and outer-OOF endpoints are reported; epoch curves are not applicable.",
        ),
        required_tables=("test_components", "case_summary"),
        required_figures=(),
        literature=(),
    ),
    "shapeformer_model_v1": ReporterProfile(
        profile_id="shapeformer_model_v1",
        title="ShapeFormer-family model extension",
        algorithm_summary=(
            "The exact registered discovery and downstream variant, including its "
            "outer-training-only shapelet bank, is reported from persisted parameters."
        ),
        statistical_methods=(
            "Discovery provenance and training curves are diagnostic and never permit outer-test-guided shapelet or epoch selection.",
        ),
        required_tables=("training_history_raw", "test_components"),
        required_figures=(
            "learning_curves",
            "balanced_accuracy_learning_curves",
        ),
        literature=(),
        limitations=(
            "Experimental and legacy variants must not be relabelled as literature-parity ShapeFormer routes.",
        ),
    ),
    "file_bag_fusion_model_v1": ReporterProfile(
        profile_id="file_bag_fusion_model_v1",
        title="File-bag fusion model extension",
        algorithm_summary=(
            "A registered raw-window encoder is pooled at file level and concatenated "
            "once with the engineered file vector before classification."
        ),
        statistical_methods=(
            "Training curves and the raw/engineered input contract are reported without changing the persisted file-level fusion hierarchy.",
        ),
        required_tables=("training_history_raw", "test_components"),
        required_figures=(
            "learning_curves",
            "balanced_accuracy_learning_curves",
        ),
        literature=(),
    ),
    "multiclass_participant_oof_v1": ReporterProfile(
        profile_id="multiclass_participant_oof_v1",
        title="Multiclass frailty classifier",
        algorithm_summary=(
            "Outer-held-out probabilities are aggregated to participant OOF; "
            "BA, macro-F1, one-vs-rest ROC/PR AUC, per-class metrics, confusion, "
            "calibration, repeat variability, confidence intervals and paired "
            "inference are reported from persisted predictions."
        ),
        statistical_methods=(
            "Participant-level metrics are recomputed separately within each repeat.",
            "Repeat summaries use mean, sample/population SD and two-sided Student-t 95% CI.",
            "Student-t intervals are descriptive across repeats and are not clipped to [0,1]; they rely on a small-sample parametric approximation even though repeated-CV estimates are correlated.",
            "BA, macro-F1 and macro one-vs-rest ROC-AUC use class-stratified participant-cluster percentile bootstrap resampling: a sampled participant carries every repeat OOF prediction; count and seed are configurable (default 10,000) and persisted.",
            "Every declared comparison reports a paired participant-cluster bootstrap CI for candidate-minus-reference BA, macro-F1 and macro ROC-AUC, using the same participant draw for both classifiers.",
            "Declared-reference BA and macro-F1 additionally use paired participant-cluster permutations with Holm correction; ROC-AUC P is explicitly N/A until a separate test is registered; count and seed are configurable (default 100,000) and persisted.",
            "If two routes change the retained participant roster, paired participant-cluster inference remains explicit N/A until a common-retained, route-specific conditional, or full-roster abstention-aware estimand is registered; the reporter does not substitute one after seeing results.",
            "Every displayed/exported table column has a generated definition and formula; identifiers/provenance fields are explicitly marked as non-arithmetic.",
            "Permutation tables persist the implementation version and NumPy row-wise int8 RNG contract so same-seed historical rebuilds remain auditable.",
            "Multiclass ROC uses empirical one-vs-rest curves plus a macro-average interpolation. The ROC figure pools persisted participant-repeat OOF rows for visualization; inferential AUC summaries remain repeat-wise and preserve their dependence labels. t-SNE is report-only on probability vectors.",
        ),
        required_tables=(
            "case_summary",
            "metric_distribution_summary",
            "repeat_metrics",
            "repeat_per_class_metrics",
            "per_class_metrics",
            "classifier_per_class_results",
            "confusion_matrices",
            "classification_prediction_scores",
            "classification_prediction_tsne",
            "classification_roc_curves",
            "classification_diagnostic_status",
            "paired_participant_inference",
            "pairwise_repeat_metric_deltas",
            "comparison_conclusions",
        ),
        required_figures=(
            "classification_prediction_scores",
            "classification_prediction_tsne",
            "classification_roc_auc_curves",
            "leaderboard",
            "stability",
            "macro_f1_stability",
            "roc_pr_auc_stability",
            "per_class_metric_stability",
            "confusion_matrices",
            "confusion_matrices_row_normalized",
            "per_class",
            "calibration",
        ),
        literature=_CLASSIFICATION_LITERATURE,
        limitations=(
            "A P value is a null-model tail probability, not the probability that a model is correct or superior.",
            "Overlapping marginal 95% CIs are descriptive and are not a paired significance test.",
            "Tuning/screening evidence is not an untouched independent final test.",
        ),
    ),
    "binary_motion_window_file_v1": ReporterProfile(
        profile_id="binary_motion_window_file_v1",
        title="Binary motion detector",
        algorithm_summary=(
            "Frozen motion probabilities and thresholds are evaluated separately "
            "at window and file level with BA, macro-F1, sensitivity, specificity, "
            "ROC AUC, PR AUC, confusion matrices, score distributions and ROC curves."
        ),
        statistical_methods=(
            "Grouped-OOF rows preserve participant groups; frozen cross-dataset rows are never recalibrated.",
            "Window metrics weight windows; file metrics aggregate window probability within physical file before one threshold application.",
            "Primary detector endpoints are reported as participant-macro mean ± between-participant sample SD with a participant percentile-bootstrap 95% CI.",
            "When both persisted training-source models predict an identical target roster, the retrospective exploratory comparison uses a two-sided participant-paired Monte-Carlo sign-flip test and Holm correction across six endpoints within target and level.",
            "Each class receives TP/FP/TN/FN, precision, sensitivity, specificity, one-vs-rest BA/F1/ROC-AUC/PR-AUC at both registered evaluation levels.",
        ),
        required_tables=(
            "motion_detector_balanced_accuracy",
            "motion_detector_macro_f1",
            "motion_detector_sensitivity",
            "motion_detector_specificity",
            "motion_detector_roc_auc",
            "motion_detector_pr_auc",
            "motion_detector_worst_fold_ba",
            "motion_detector_window_confusion",
            "motion_detector_file_confusion",
            "motion_detector_score_distributions",
            "motion_detector_roc_curves",
            "motion_detector_per_class_results",
            "motion_detector_per_class_performance",
            "motion_detector_per_class_discrimination",
            "motion_detector_training_source_inference",
            "inference_configuration",
        ),
        required_figures=(
            "motion_detector_metrics",
            "motion_internal_confusion_matrix",
            "motion_ptt_confusion_matrix",
            "frailty29_trained_window_score_distribution",
            "frailty29_trained_file_score_distribution",
            "frailty29_trained_window_prediction_tsne",
            "frailty29_trained_file_prediction_tsne",
            "frailty29_trained_window_roc_auc_curve",
            "frailty29_trained_file_roc_auc_curve",
        ),
        literature=(
            "Brodersen et al. (2010), balanced accuracy, DOI:10.1109/ICPR.2010.764",
            "Fawcett (2006), ROC analysis, DOI:10.1016/j.patrec.2005.10.010",
            "Efron & Tibshirani (1993), An Introduction to the Bootstrap, DOI:10.1007/978-1-4899-4541-9",
            "Phipson & Smyth (2010), permutation P-value plus-one correction, DOI:10.2202/1544-6115.1585",
            "Holm (1979), sequentially rejective multiple testing, DOI:10.2307/4615733",
        ),
        limitations=(
            "Protocol activity state is a proxy label, not window-wise manually adjudicated motion-artifact ground truth.",
        ),
    ),
    "motion_route_component_v1": ReporterProfile(
        profile_id="motion_route_component_v1",
        title="Frozen motion component inside frailty routing",
        algorithm_summary=(
            "A persisted frozen motion model supplies low/high-motion routing "
            "state inside a frailty-classification fold. This profile reports "
            "route coverage and quality effects; it does not relabel those "
            "frailty outcomes as binary motion-detector validation."
        ),
        statistical_methods=(
            "Report retained/excluded windows, files and participants plus route/role coverage from held-out frailty-fold artifacts.",
            "Binary motion BA/F1/ROC/PR endpoints remain exclusive to Stage5-pre motion-reference evidence.",
        ),
        required_tables=(
            "coverage",
            "route_role_coverage",
            "quality_distributions",
        ),
        required_figures=(
            "coverage",
            "route_role_coverage",
            "quality_distributions",
        ),
        literature=(),
        limitations=(
            "Frailty routing outcomes do not constitute new motion ground-truth labels.",
        ),
    ),
    "beat_detector_recording_v1": ReporterProfile(
        profile_id="beat_detector_recording_v1",
        title="PPG beat detector",
        algorithm_summary=(
            "Detected PPG beats are aligned to ECG reference beats with recording-"
            "local lag updates and one-to-one ±150 ms matching; sensitivity, PPV, "
            "F1, interval RMSE and runtime are summarized per recording."
        ),
        statistical_methods=(
            "Lag is re-estimated every 300 s; each PPG beat can match at most one reference beat.",
            "Recording distributions use median/IQR and 10th/90th-percentile boxplot whiskers.",
            "F1, sensitivity, PPV, IBI–PPI RMSE and execution-time percentage use the same two-sided Wilcoxon rank-sum procedure.",
            "All selected endpoints, channels and reference-comparator contrasts form one Holm–Sidak step-down family; historical reports label endpoints absent from their resolved plan as retrospective supplements.",
            "The source-faithful rank-sum procedure is unpaired even when identical subject-recordings are available; it must not be described as a paired signed-rank test.",
        ),
        required_tables=(
            "static_peak_detector_recording_metrics",
            "static_peak_detector_distribution_statistics",
            "static_peak_detector_rank_sum_holm_sidak",
            "static_peak_detector_significance_summary",
        ),
        required_figures=(
            "static_peak_detector_f1",
            "static_peak_detector_sensitivity",
            "static_peak_detector_ppv",
            "static_peak_detector_interval_rmse",
            "static_peak_detector_runtime",
        ),
        literature=(
            "Charlton et al. (2025), MSPTDfast (v.2), DOI:10.1088/1361-6579/adb89e",
            "Wilcoxon (1945), rank-based two-sample and paired comparisons, DOI:10.2307/3001968",
            "Holm (1979), sequentially rejective multiple testing, DOI:10.2307/4615733",
            "Šidák (1967), simultaneous multiplicity adjustment, DOI:10.1080/01621459.1967.10482935",
        ),
        limitations=(
            "scipy.stats.ranksums does not use within-record pairing or apply a tie correction.",
            "Execution time is a single sequential local wall-time measurement per detector/recording/channel and is hardware/load specific.",
        ),
    ),
    "beat_detector_legacy_persisted_v1": ReporterProfile(
        profile_id="beat_detector_legacy_persisted_v1",
        title="Historical PPG beat-detector report",
        algorithm_summary=(
            "Historical detector outputs are regenerated only with the lag window, "
            "beat tolerance, aggregation and metrics persisted in that run's "
            "resolved_plan.yaml; the later 300 s/±150 ms contract is not back-applied."
        ),
        statistical_methods=(
            "Exact historical validation settings are displayed from resolved_plan.yaml.",
            "Historical summaries remain historical evidence and are not relabeled as current-contract results.",
        ),
        required_tables=("static_peak_detector_summary",),
        required_figures=(
            "static_peak_detector_f1",
            "static_peak_detector_sensitivity",
            "static_peak_detector_ppv",
            "static_peak_detector_interval_rmse",
            "static_peak_detector_runtime",
        ),
        literature=(
            "Current comparison paper retained for context only: Charlton et al. (2025), DOI:10.1088/1361-6579/adb89e",
        ),
        limitations=(
            "Do not pool this profile with current-contract 300 s/±150 ms recording-level results.",
        ),
    ),
    "stage5_ecg_ppg_denoiser_v1": ReporterProfile(
        profile_id="stage5_ecg_ppg_denoiser_v1",
        title="Motion-artifact denoiser",
        algorithm_summary=(
            "Each reducer is assessed by re-detecting PPG beats, lag-aligning them "
            "to ECG annotations, and reporting subject-macro PPI–RR RMSE, beat "
            "sensitivity/PPV/F1, attempted/passed/failed coverage and runtime."
        ),
        statistical_methods=(
            "Static and dynamic activity groups are separate five-column result tables, each ordered by subject-macro PPI–RR RMSE ascending across RED and IR rows.",
            "The visible result columns are denoiser, optical channel, RMSE mean ± SD, F1 mean ± SD, and RMSE P versus identity; full endpoint and CI evidence remains machine-auditable without widening the main tables.",
            "Participant-macro means and sample SD (ddof=1) are computed across evaluable subjects; this SD is between-subject dispersion, not training-repeat variability.",
            "Absolute endpoint CI95 uses participant percentile bootstrap resampling.",
            "The 2026-08-24 retrospective exploratory supplement compares every reducer with the configured reference (default: identity) on identical successful segment keys using a two-sided participant-paired Monte-Carlo sign-flip test; Holm correction is applied across reducers separately for each activity, channel and endpoint.",
        ),
        required_tables=(
            "denoiser_static",
            "denoiser_dynamic",
            "denoiser_coverage",
            "denoiser_paired_inference",
            "inference_configuration",
        ),
        required_figures=(
            "denoiser_interval_rmse",
            "denoiser_beat_f1",
            "denoiser_beat_sensitivity",
            "denoiser_beat_ppv",
            "denoiser_runtime",
        ),
        literature=(
            "Charlton et al. (2025), beat assessment context, DOI:10.1088/1361-6579/adb89e",
            "Efron & Tibshirani (1993), An Introduction to the Bootstrap, DOI:10.1007/978-1-4899-4541-9",
            "Phipson & Smyth (2010), permutation P-value plus-one correction, DOI:10.2202/1544-6115.1585",
            "Holm (1979), sequentially rejective multiple testing, DOI:10.2307/4615733",
        ),
        limitations=(
            "All-failed subjects/reducers remain visible as N/A and are not converted to zero-valued physiology endpoints.",
            "The default identity reference was selected after the original plan was resolved; P values are exploratory and cannot be relabeled as prespecified confirmatory evidence.",
        ),
    ),
    "frailty_denoiser_route_v1": ReporterProfile(
        profile_id="frailty_denoiser_route_v1",
        title="Denoiser inside frailty routing",
        algorithm_summary=(
            "The configured reducer is applied only on eligible routed windows; "
            "the report pairs direct and post-denoise PPG HR/PPI endpoints, "
            "Q_rate recovery, reducer failures and downstream coverage."
        ),
        statistical_methods=(
            "Direct/post endpoint rows remain paired by physical record and are summarized participant-macro.",
            "Reducer failure and missing endpoint counts remain explicit and are not converted into zero-valued physiology metrics.",
        ),
        required_tables=(
            "denoiser_hr_comparison",
            "denoiser_hr_record_pairs",
            "route_role_coverage",
            "coverage",
        ),
        required_figures=(
            "denoiser_hr_comparison",
            "route_role_coverage",
            "coverage",
        ),
        literature=(),
        limitations=(
            "This is a downstream frailty-route audit, not the PTT ECG-reference denoiser benchmark.",
        ),
    ),
    "sqi_route_coverage_v1": ReporterProfile(
        profile_id="sqi_route_coverage_v1",
        title="SQI and routing",
        algorithm_summary=(
            "Q_rate, Q_morph, frozen motion state and optional denoiser outcome "
            "determine Excellent/Acceptable/Unfit routing; reports preserve "
            "conditional and abstention-aware endpoints plus coverage."
        ),
        statistical_methods=(
            "Conditional metrics use retained participants; abstention-aware metrics count excluded endpoints explicitly.",
            "Coverage and class-specific abstention counts accompany every performance result.",
        ),
        required_tables=(
            "route_role_coverage",
            "quality_distributions",
            "coverage",
            "denoiser_hr_comparison",
        ),
        required_figures=(
            "coverage",
            "route_role_coverage",
            "quality_distributions",
            "denoiser_hr_comparison",
        ),
        literature=(),
        limitations=(
            "This SQI decision tree and its thresholds are project algorithms unless a component row names a separate source.",
        ),
    ),
    "audit_provenance_v1": ReporterProfile(
        profile_id="audit_provenance_v1",
        title="Configuration and provenance audit",
        algorithm_summary=(
            "Persisted resolved configuration, input data, seeds, splits, status "
            "and artifact inventory are projected without changing the experiment."
        ),
        statistical_methods=(),
        required_tables=("test_components", "reproducibility_summary"),
        required_figures=(),
        literature=(),
    ),
}


_REPORTER_PROFILE_ALIASES: Mapping[str, str] = {
    # Historical persisted/report-facing name accepted on rebuild. New reports
    # emit the more precise Stage5 endpoint name.
    "denoiser_ecg_ppg_endpoint_v1": "stage5_ecg_ppg_denoiser_v1",
}


def _profile_id(component_role: str) -> str:
    role = str(component_role).strip().lower()
    if role == "classifier" or role == "classifier_tuning_candidate":
        return "multiclass_participant_oof_v1"
    if role.startswith("motion_detector") or role == "motion_threshold":
        return "binary_motion_window_file_v1"
    if role in {"peak_detector", "peak_validation"}:
        # A detector used inside a classifier is provenance, not a beat-endpoint
        # study. Motion/peak studies explicitly override this with the persisted
        # legacy or current validation contract.
        return "audit_provenance_v1"
    if role == "denoiser":
        return "stage5_ecg_ppg_denoiser_v1"
    if role == "sqi":
        return "sqi_route_coverage_v1"
    return "audit_provenance_v1"


def annotate_component_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Attach registry-owned reporter metadata or fail closed when active."""

    output = dict(row)
    explicit_profile_id = str(output.get("reporter_profile_id", "")).strip()
    execution_state = str(output.get("execution_state", "")).strip().lower()
    inactive = any(
        marker in execution_state
        for marker in (
            "disabled",
            "configured_not_executed",
            "not_executed",
            "not_run",
        )
    )
    profile_id = explicit_profile_id or (
        "audit_provenance_v1"
        if inactive
        else _profile_id(str(output.get("component_role", "")))
    )
    profile_id = _REPORTER_PROFILE_ALIASES.get(profile_id, profile_id)
    if profile_id not in REPORTER_PROFILES:
        raise ValueError(f"unknown reporter profile: {profile_id}")
    output["reporter_profile_id"] = profile_id
    binding = component_reporter_binding(
        str(output.get("component_role", "")),
        str(output.get("module_id", "")),
        active=not inactive,
    )
    extension_id = str(binding["reporter_extension_id"])
    if extension_id != "not_applicable" and extension_id not in REPORTER_PROFILES:
        raise ValueError(
            "module registry names unknown reporter extension: "
            f"{extension_id}"
        )
    # Keep the historical column name for report-schema compatibility. Its
    # value now comes from the model/module registry for every active module.
    output["model_reporter_extension_id"] = extension_id
    if not str(output.get("algorithm_kernel_description", "")).strip():
        output["algorithm_kernel_description"] = str(binding["algorithm_summary"])
    references = tuple(str(value) for value in binding["references"])
    output["algorithm_references"] = (
        "; ".join(references)
        if references
        else "N/A — component was not executed"
    )
    output["registered_module_id"] = str(binding["registered_module_id"])
    output["registered_module_family"] = str(binding["registered_module_family"])
    output["reporter_binding_kind"] = str(binding["reporter_binding_kind"])
    output["reporter_binding_source"] = str(binding["reporter_binding_source"])
    return output


def annotate_component_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [annotate_component_row(row) for row in rows]


def reporter_profile_rows(
    component_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one auditable row per profile actually present in a report."""

    annotated = annotate_component_rows(component_rows)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in annotated:
        grouped.setdefault(str(row["reporter_profile_id"]), []).append(row)
        extension_id = str(row.get("model_reporter_extension_id", ""))
        if extension_id in REPORTER_PROFILES:
            grouped.setdefault(extension_id, []).append(row)
    output: list[dict[str, Any]] = []
    for profile_id, rows in sorted(grouped.items()):
        profile = REPORTER_PROFILES[profile_id]
        module_references = sorted(
            {
                str(row["algorithm_references"])
                for row in rows
                if str(row.get("algorithm_references", "")).strip()
            }
        )
        output.append(
            {
                **asdict(profile),
                "profile_kind": (
                    "model_or_module_extension"
                    if any(
                        str(row.get("model_reporter_extension_id", "")) == profile_id
                        and str(row.get("reporter_binding_kind", "")) == "extension"
                        for row in rows
                    )
                    else "endpoint_or_module"
                ),
                "participating_components": sorted(
                    {
                        f"{row.get('component_role')}:{row.get('module_id')}"
                        for row in rows
                    }
                ),
                "module_references": module_references,
                "presentation_only": True,
                "changes_training_or_predictions": False,
            }
        )
    return output


def required_figure_modules(
    component_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    annotated = annotate_component_rows(component_rows)
    profile_ids = {
        str(row["reporter_profile_id"])
        for row in annotated
    }
    profile_ids.update(
        extension_id
        for row in annotated
        if (
            extension_id := str(row.get("model_reporter_extension_id", ""))
        )
        in REPORTER_PROFILES
    )
    return tuple(
        sorted(
            {
                figure
                for profile_id in profile_ids
                for figure in REPORTER_PROFILES[profile_id].required_figures
            }
        )
    )


def _profile_table_cell(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    return str(value)


def markdown_reporter_profile_tables(
    profile_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Render the lossless profile rows as separate semantic narrow tables."""

    if not profile_rows:
        return "N/A — no reporter profile was selected."
    lines: list[str] = []
    for title, schema in REPORTER_PROFILE_VIEW_SCHEMAS:
        if len(schema) > _MAX_HUMAN_FACING_PROFILE_COLUMNS:
            raise ValueError(
                f"human-facing reporter-profile table {title!r} has "
                f"{len(schema)} columns; maximum is "
                f"{_MAX_HUMAN_FACING_PROFILE_COLUMNS}"
            )
        headings = [label for _field, label in schema]
        lines.extend(
            (
                f"### {title}",
                "",
                "| " + " | ".join(headings) + " |",
                "|" + "|".join("---" for _ in headings) + "|",
            )
        )
        for row in profile_rows:
            values = [
                _profile_table_cell(row.get(field, ""))
                .replace("|", r"\|")
                .replace("\n", " ")
                for field, _label in schema
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines.extend(
            (
                "",
                markdown_column_definitions_block(
                    [field for field, _label in schema],
                    display_labels=[label for _field, label in schema],
                ),
                "",
            )
        )
    return "\n".join(lines).rstrip()


def reporter_methods_markdown(
    component_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Render the canonical methods/literature section shared by all reporters."""

    lines = [
        "# Reporter methods and literature",
        "",
        "These profiles are selected from the persisted model/module identities.",
        "They control presentation only and never modify fitted models, predictions,",
        "thresholds, ranking evidence, or eligibility.",
        "",
    ]
    for row in reporter_profile_rows(component_rows):
        lines.extend(
            [
                f"## {row['title']} (`{row['profile_id']}`)",
                "",
                str(row["algorithm_summary"]),
                "",
                "Components: "
                + ", ".join(f"`{value}`" for value in row["participating_components"]),
                "",
                "Required tables: "
                + ", ".join(f"`{value}`" for value in row["required_tables"]),
                "",
                "Required figures: "
                + (
                    ", ".join(f"`{value}`" for value in row["required_figures"])
                    or "none"
                ),
                "",
                "Statistical/reporting methods:",
                "",
                *(
                    [f"- {value}" for value in row["statistical_methods"]]
                    or ["- No additional statistical method is attached to this audit-only profile."]
                ),
                "",
                "Algorithm and literature provenance:",
                "",
                *[
                    f"- {value}"
                    for value in dict.fromkeys(
                        [*row["module_references"], *row["literature"]]
                    )
                ],
                "",
            ]
        )
        if row["limitations"]:
            lines.extend(
                ["Limitations:", "", *[f"- {value}" for value in row["limitations"]], ""]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_reporter_methods(
    root: str | Path,
    component_rows: Sequence[Mapping[str, Any]],
) -> Path:
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

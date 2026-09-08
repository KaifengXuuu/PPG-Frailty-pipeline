"""Public training, evaluation, OOF and bundle facade.

English: Stable names in this package enforce frozen outer-fold membership,
outer-label-blind epoch selection, hierarchy aggregation, one-factor ablations
and integrity-checked deployable bundles for both the CLI and unit tests.

中文：本包通过稳定名称强制执行冻结 outer-fold 成员、outer 标签不可见的 epoch
选择、层级聚合、单因素消融，以及供 CLI 与单元测试使用的完整性校验部署 bundle。
"""
from importlib import import_module


_EXPORT_GROUPS = {
    "ablation": "AblationCase PairedComparison paired_subject_deltas run_ablation_matrix",
    "aggregation": """BALANCE_LINES LINE_A_EQUAL_FILES LINE_B_EQUAL_ROLE_FAMILIES
        QUALITY_WEIGHT_SOURCE_LEGACY_WINDOW QUALITY_WEIGHT_SOURCE_NONE QUALITY_WEIGHT_SOURCE_ROUTE_FILE
        QUALITY_WEIGHT_SOURCES CoverageSummary ExperimentIdentity HierarchyAggregation aggregate_hierarchy
        aggregate_oof_rows aggregation_rule_for_training_balance canonical_role_family experiment_identity""",
    "bundle": """BUNDLE_FORMAT_VERSION FINAL_BUNDLE_PARITY_ATOL REQUIRED_METADATA FinalRefitBinding
        FinalRefitExecution FinalRefitPlan LoadedBundle canonical_input_spec_payload current_runtime_environment
        execute_full_cohort_refit assert_golden_parity assert_repeated_bundle_parity load_bundle
        materialize_final_refit_binding input_spec_sha256 predict_bundle predict_bundle_raw save_bundle
        save_final_refit_bundle validate_bundle_metadata""",
    "datasets": """FeatureMatrixDataset FeatureVectorDataset FileBagDataset RawWindowDataset SampleIdentity
        collate_samples""",
    "evaluator": """AbstentionAwareEvaluationMetrics AbstentionAwarePerClassMetrics EvaluationMetrics PairedDeltaSummary
        PerClassMetrics RepeatMetricSummary evaluate_predictions evaluate_predictions_with_abstentions
        paired_fold_seed_deltas predict_torch_dataset summarize_repeat_metric""",
    "oof": """OOF_SCHEMA_VERSION OofPredictionRow OofWriter read_oof_parquet_metadata read_oof_parquet
        validate_expected_oof_roster validate_formal_oof validate_unique_subject_oof write_oof_parquet
        write_empty_oof_parquet""",
    "operational": """CPU_BATCH1_MEASURED_RUNS CPU_BATCH1_WARMUP_RUNS OperationalMetrics measure_bundle_bytes
        measure_cpu_batch1_operational_metrics model_parameter_count""",
    "statistics": """CLUSTER_BOOTSTRAP_IMPLEMENTATION_VERSION CLUSTER_BOOTSTRAP_RNG_CONTRACT
        DEFAULT_BOOTSTRAP_RESAMPLES DEFAULT_PERMUTATION_RESAMPLES ClusterBootstrapResult ComparisonArchive ConfigMetrics
        HolmResult ManualFinalSelection PairedClusterBootstrapResult PairedPermutationResult ParticipantPrediction
        build_config_metrics_from_predictions_and_fold_summaries holm_adjust holm_adjust_by_family_metric
        paired_participant_cluster_bootstrap paired_participant_permutation participant_cluster_bootstrap rank_top10
        read_verified_manual_selections verify_comparison_archive write_comparison_archive""",
    "trainer": """OPTIMIZER_PARAMETER_DEFAULTS TRAINING_CACHE_POLICIES TRAINING_CLASS_COUNT_BASES DEEP_EPOCH_CONFIG_IDS
        TRAINING_CLASS_WEIGHTINGS TRAINING_LOSSES TRAINING_OPTIMIZERS TRAINING_SAMPLERS FittedObjectProvenance
        FullCohortRefitScope FrozenOuterSplit InnerGroupedSplit TrainingConfig TrainingResult UnifiedTrainer
        build_inner_grouped_split dataset_binding_hash configured_class_weight_vector configured_row_sampling_weights
        normalize_participant_window_quota materialize_all_deep_epoch_configs materialize_deep_epoch_config
        outer_train_effective_number_weights outer_train_class_counts outer_train_participant_class_counts
        participant_file_window_sampling_weights participant_window_sampling_weights resolve_optimizer_parameters
        subject_epoch_sampling_indices outer_train_window_inverse_frequency_weights model_member_state_hashes
        validate_dataset_identity_coherence""",
}
_EXPORTS = {name: module for module, names in _EXPORT_GROUPS.items() for name in names.split()}
__all__ = sorted(_EXPORTS)

def __getattr__(name: str):
    """Load facade members on first use and then cache them in this module."""
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

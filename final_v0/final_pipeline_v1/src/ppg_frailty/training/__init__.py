"""Public training, evaluation, OOF and bundle facade.

English: Stable names in this package enforce frozen outer-fold membership,
outer-label-blind epoch selection, hierarchy aggregation, one-factor ablations
and integrity-checked deployable bundles for both the CLI and unit tests.

中文：本包通过稳定名称强制执行冻结 outer-fold 成员、outer 标签不可见的 epoch
选择、层级聚合、单因素消融，以及供 CLI 与单元测试使用的完整性校验部署 bundle。
"""

from .ablation import AblationCase, PairedComparison, paired_subject_deltas, run_ablation_matrix
from .aggregation import (
    CoverageSummary,
    ExperimentIdentity,
    HierarchyAggregation,
    aggregate_hierarchy,
    aggregate_oof_rows,
    experiment_identity,
)
from .bundle import (
    BUNDLE_FORMAT_VERSION,
    REQUIRED_METADATA,
    LoadedBundle,
    assert_golden_parity,
    assert_repeated_bundle_parity,
    load_bundle,
    predict_bundle,
    predict_bundle_raw,
    save_bundle,
    validate_bundle_metadata,
)
from .datasets import (
    FeatureMatrixDataset,
    FeatureVectorDataset,
    FileBagDataset,
    RawWindowDataset,
    SampleIdentity,
    collate_samples,
)
from .evaluator import (
    EvaluationMetrics,
    PairedDeltaSummary,
    PerClassMetrics,
    RepeatMetricSummary,
    evaluate_predictions,
    paired_fold_seed_deltas,
    predict_torch_dataset,
    summarize_repeat_metric,
)
from .oof import (
    OofPredictionRow,
    OofWriter,
    validate_expected_oof_roster,
    validate_formal_oof,
    validate_unique_subject_oof,
    write_oof_parquet,
)
from .trainer import (
    FittedObjectProvenance,
    FrozenOuterSplit,
    InnerGroupedSplit,
    TrainingConfig,
    TrainingResult,
    UnifiedTrainer,
    dataset_binding_hash,
    validate_dataset_identity_coherence,
)

__all__ = [
    "AblationCase",
    "BUNDLE_FORMAT_VERSION",
    "CoverageSummary",
    "EvaluationMetrics",
    "ExperimentIdentity",
    "FeatureMatrixDataset",
    "FeatureVectorDataset",
    "FileBagDataset",
    "FittedObjectProvenance",
    "FrozenOuterSplit",
    "HierarchyAggregation",
    "InnerGroupedSplit",
    "LoadedBundle",
    "OofPredictionRow",
    "OofWriter",
    "PairedDeltaSummary",
    "PerClassMetrics",
    "PairedComparison",
    "RawWindowDataset",
    "REQUIRED_METADATA",
    "RepeatMetricSummary",
    "SampleIdentity",
    "TrainingConfig",
    "TrainingResult",
    "UnifiedTrainer",
    "aggregate_hierarchy",
    "aggregate_oof_rows",
    "assert_golden_parity",
    "assert_repeated_bundle_parity",
    "collate_samples",
    "evaluate_predictions",
    "experiment_identity",
    "load_bundle",
    "paired_subject_deltas",
    "paired_fold_seed_deltas",
    "predict_bundle",
    "predict_bundle_raw",
    "predict_torch_dataset",
    "run_ablation_matrix",
    "save_bundle",
    "summarize_repeat_metric",
    "dataset_binding_hash",
    "validate_bundle_metadata",
    "validate_dataset_identity_coherence",
    "validate_expected_oof_roster",
    "validate_formal_oof",
    "validate_unique_subject_oof",
    "write_oof_parquet",
]

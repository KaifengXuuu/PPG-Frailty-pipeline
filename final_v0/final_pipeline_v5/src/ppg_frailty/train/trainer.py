"""统一 Trainer 规范重导出 / Canonical UnifiedTrainer re-exports.

中文：dataset identity 与 binding hash 同样属于训练防泄漏合同，必须从 canonical
门面可见。English: Dataset identity and binding hash are part of the leak guard.
"""

from ..training.trainer import (
    FittedObjectProvenance,
    FrozenOuterSplit,
    InnerGroupedSplit,
    TrainingConfig,
    TrainingResult,
    UnifiedTrainer,
    dataset_binding_hash,
    dataset_identities,
    participant_file_window_sampling_weights,
    dataset_participant_ids,
    validate_dataset_identity_coherence,
)

__all__ = [
    "FittedObjectProvenance",
    "FrozenOuterSplit",
    "InnerGroupedSplit",
    "TrainingConfig",
    "TrainingResult",
    "UnifiedTrainer",
    "dataset_binding_hash",
    "dataset_identities",
    "dataset_participant_ids",
    "participant_file_window_sampling_weights",
    "validate_dataset_identity_coherence",
]

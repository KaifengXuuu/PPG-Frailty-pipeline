"""数据身份、QC、冻结分折、窗口与缓存 / Data contracts and safeguards.

中文：本子包只公开规范入口。所有内部 recording 都由版本化 manifest 驱动，
所有 outer fold 都来自物化表，任何缓存都由完整 provenance 内容寻址。

English: This package exposes only canonical entry points. Internal recordings are
manifest-driven, outer folds are materialized rather than regenerated, and caches
are addressed by complete provenance identities.
"""

from .cache import CacheIdentity, ContentAddressedCache
from .external_manifest import (
    ExternalManifestError,
    ExternalRecord,
    PttSynchronizedSignals,
    adapt_ptt_synchronized_channels,
    audit_external_manifest,
    build_external_manifest,
    load_external_manifest,
)
from .external_folds import (
    PTT_FORMAL_ALGORITHM,
    PTT_FORMAL_FOLD_SIZES,
    PTT_FORMAL_REGISTRY_ID,
    PTT_FORMAL_REPEAT_SEEDS,
    build_formal_ptt_fold_rows,
    load_formal_ptt_repeated_folds,
    materialize_formal_ptt_repeated_folds,
    resolve_formal_ptt_split,
    validate_formal_ptt_fold_rows,
)
from .folds import (
    FrozenFoldAudit,
    FrozenFoldRegistry,
    load_frozen_memberships,
    load_m2_frozen_registry,
    materialize_assignments,
    materialize_fold_csvs,
    resolve_outer_fold,
)
from .manifest import (
    audit_manifest,
    build_internal_manifest,
    load_internal_manifest,
    load_manifest,
    load_m2_internal_manifest,
    manifest_summary,
)
from .qc import (
    QCAssessment,
    QCThresholds,
    RecordingQCAdmission,
    assess_manifest_record,
    assess_numeric_record,
    physical_recording_qc_profile_v2,
    physical_recording_qc_thresholds_v2,
    require_recording_qc_pass,
)
from .schema import (
    DEFAULT_CLASSIFIER_ROLE_FAMILIES,
    ROLE_FAMILIES,
    FoldAssignment,
    QCReason,
    QCStatus,
    canonicalize_role_family,
    is_default_classifier_role,
    validate_manifest_row,
)
from .windows import WindowPlan, WindowSlice, extract_window

__all__ = [
    "CacheIdentity",
    "ContentAddressedCache",
    "DEFAULT_CLASSIFIER_ROLE_FAMILIES",
    "ExternalManifestError",
    "ExternalRecord",
    "PttSynchronizedSignals",
    "PTT_FORMAL_ALGORITHM",
    "PTT_FORMAL_FOLD_SIZES",
    "PTT_FORMAL_REGISTRY_ID",
    "PTT_FORMAL_REPEAT_SEEDS",
    "FoldAssignment",
    "FrozenFoldAudit",
    "FrozenFoldRegistry",
    "QCAssessment",
    "QCReason",
    "QCStatus",
    "QCThresholds",
    "RecordingQCAdmission",
    "ROLE_FAMILIES",
    "WindowPlan",
    "WindowSlice",
    "audit_external_manifest",
    "adapt_ptt_synchronized_channels",
    "audit_manifest",
    "assess_numeric_record",
    "assess_manifest_record",
    "build_external_manifest",
    "build_formal_ptt_fold_rows",
    "build_internal_manifest",
    "canonicalize_role_family",
    "extract_window",
    "is_default_classifier_role",
    "load_external_manifest",
    "load_frozen_memberships",
    "load_internal_manifest",
    "load_manifest",
    "load_m2_frozen_registry",
    "load_m2_internal_manifest",
    "load_formal_ptt_repeated_folds",
    "manifest_summary",
    "materialize_assignments",
    "materialize_fold_csvs",
    "materialize_formal_ptt_repeated_folds",
    "physical_recording_qc_profile_v2",
    "physical_recording_qc_thresholds_v2",
    "resolve_formal_ptt_split",
    "resolve_outer_fold",
    "require_recording_qc_pass",
    "validate_formal_ptt_fold_rows",
    "validate_manifest_row",
]

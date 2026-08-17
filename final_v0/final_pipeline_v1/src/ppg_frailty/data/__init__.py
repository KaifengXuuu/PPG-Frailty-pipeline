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
    audit_external_manifest,
    build_external_manifest,
    load_external_manifest,
    load_provisional_external_split,
    materialize_provisional_external_grouped_split,
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
from .qc import QCAssessment, QCThresholds, assess_numeric_record
from .schema import FoldAssignment, QCReason, QCStatus, validate_manifest_row
from .windows import WindowPlan, WindowSlice, extract_window

__all__ = [
    "CacheIdentity",
    "ContentAddressedCache",
    "ExternalManifestError",
    "ExternalRecord",
    "FoldAssignment",
    "FrozenFoldAudit",
    "FrozenFoldRegistry",
    "QCAssessment",
    "QCReason",
    "QCStatus",
    "QCThresholds",
    "WindowPlan",
    "WindowSlice",
    "audit_external_manifest",
    "audit_manifest",
    "assess_numeric_record",
    "build_external_manifest",
    "build_internal_manifest",
    "extract_window",
    "load_external_manifest",
    "load_frozen_memberships",
    "load_internal_manifest",
    "load_manifest",
    "load_m2_frozen_registry",
    "load_m2_internal_manifest",
    "load_provisional_external_split",
    "manifest_summary",
    "materialize_assignments",
    "materialize_fold_csvs",
    "materialize_provisional_external_grouped_split",
    "resolve_outer_fold",
    "validate_manifest_row",
]

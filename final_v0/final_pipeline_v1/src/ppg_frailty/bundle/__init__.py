"""规范可部署 bundle 门面 / Canonical deployable-bundle facade.

中文：集中公开严格 schema、完整性加载与 raw adapter 推理。
English: Expose strict schema, integrity loading and raw-adapter inference together.
"""

from .infer import infer_raw_record
from .load import LoadedBundle, load_bundle
from .save import REQUIRED_V1_METADATA, save_bundle_strict, validate_bundle_metadata
from .schema import BUNDLE_SCHEMA_VERSION
from ..training.bundle import predict_bundle_raw

__all__ = [
    "BUNDLE_SCHEMA_VERSION", "LoadedBundle", "REQUIRED_V1_METADATA", "infer_raw_record",
    "load_bundle", "predict_bundle_raw", "save_bundle_strict", "validate_bundle_metadata",
]

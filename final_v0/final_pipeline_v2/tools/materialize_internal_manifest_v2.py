#!/usr/bin/env python3
"""物化 V2 内部 manifest / Materialize the V2 internal manifest.

中文：唯一输入是 M2 冻结清单。本工具逐一重算 261 个来源文件的 SHA-256，
保留 FRAILTY-STATUS 2/3 的标签记录来源，并把 Young 明确记录为 cohort override，
绝不为 Young 虚构 FRAILTY 标签记录。

English: The frozen M2 manifest is the sole authority. This tool re-hashes all
261 recordings, preserves FRAILTY-STATUS 2/3 label-record provenance, and records
Young as a cohort override without inventing a FRAILTY label record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]
SRC_ROOT = PIPELINE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ppg_frailty.data import audit_manifest, build_internal_manifest  # noqa: E402
from ppg_frailty.data.manifest import (  # noqa: E402
    M2_FILE_MANIFEST,
    M2_FILE_MANIFEST_SHA256,
)
from ppg_frailty.provenance import atomic_write_json, sha256_file  # noqa: E402


OUTPUT_PATH = PIPELINE_ROOT / "manifests/internal_records_v2.csv"
REPORT_PATH = PIPELINE_ROOT / "reports/internal_manifest_v2_report.json"


def _artifact(path: Path) -> dict[str, object]:
    """返回已物化文件身份 / Return materialized artifact identity."""

    return {
        "path": path.relative_to(PIPELINE_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    """重哈希来源并原子发布 manifest/report / Verify and publish outputs."""

    if PIPELINE_ROOT.name != "final_pipeline_v2":
        raise SystemExit("materializer root is not final_pipeline_v2")
    source = REPOSITORY_ROOT / M2_FILE_MANIFEST
    incomplete = {
        "schema_version": "ppg_frailty.internal_manifest_materialization.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "materializing_incomplete_fail_closed",
        "source_manifest_path": M2_FILE_MANIFEST.as_posix(),
        "source_manifest_sha256_expected": M2_FILE_MANIFEST_SHA256,
    }
    atomic_write_json(REPORT_PATH, incomplete, root=PIPELINE_ROOT)
    rows = build_internal_manifest(source, OUTPUT_PATH)
    summary = audit_manifest(rows)
    payload = {
        "schema_version": "ppg_frailty.internal_manifest_materialization.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "passed",
        "source_manifest_path": M2_FILE_MANIFEST.as_posix(),
        "source_manifest_sha256": sha256_file(source),
        "source_manifest_sha256_expected": M2_FILE_MANIFEST_SHA256,
        "all_261_source_hashes_verified": True,
        "young_label_semantics": {
            "class_source": "cohort_override_young",
            "label_record_id": (
                "preserved_from_m2_optional_cohort_source_record_"
                "not_frailty_status"
            ),
        },
        "summary": summary,
        "generated_artifact": _artifact(OUTPUT_PATH),
        "producer_sha256": sha256_file(__file__),
    }
    atomic_write_json(REPORT_PATH, payload, root=PIPELINE_ROOT)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

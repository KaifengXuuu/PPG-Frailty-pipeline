#!/usr/bin/env python3
"""运行 M1 V3 当前合同验证入口；run the current M1 V3 contract validator.

中文
----
首版 V3 validator 会扫描含迁移叙述的 registry，因此把迁移表中的旧字段原名误报
为活动合同残留。本入口保持首版全部结构/交叉检查，只把 routing registry 指向不含
迁移叙述的 active 视图，并生成带 CURRENT 后缀的权威报告和完整性树。

English
-------
The first V3 validator scanned a registry that also documented legacy migration,
so legacy names in that metadata were falsely reported as active fields. This
entry retains every structural/cross check, points routing validation at the
metadata-free active registry, and writes authoritative CURRENT reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import validate_m1_contracts_v3 as base


# 中文：CURRENT 文件独立生成；不覆盖首版失败证据。
# English: Generate separate CURRENT files and preserve the first-run evidence.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PACKAGE_ROOT / "M1_CONTRACT_VERIFICATION_V3_CURRENT.json"
TREE_PATH = PACKAGE_ROOT / "M1_PACKAGE_TREE_V3_CURRENT.md"
ACTIVE_ROUTING_REGISTRY = "registries_v3/quality_routing_registry_v3_active.json"
CURRENT_FILES = (
    "00_CURRENT_STATUS_V3.md",
    "00_CURRENT_STATUS_V3_1.md",
    "06_SEQUENTIAL_SQI_MOTION_ROUTING_V3.md",
    *base.SCHEMA_FILES,
    "registries_v2/platform_profiles_v2.json",
    ACTIVE_ROUTING_REGISTRY,
    "registries_v2/feature_extractor_registry_v2.json",
    "registries_v2/classifier_registry_v2.json",
    *base.EXAMPLE_FILES,
    "tools/validate_m1_contracts_v3.py",
    "tools/validate_m1_contracts_v3_current.py",
    "tools/validate_m1_v3_routing_invariants.py",
)


def configure_base_validator() -> None:
    """将首版检查绑定到 active registry；bind base checks to the active registry."""

    base.REGISTRY_FILES = (
        "registries_v2/platform_profiles_v2.json",
        ACTIVE_ROUTING_REGISTRY,
        "registries_v2/feature_extractor_registry_v2.json",
        "registries_v2/classifier_registry_v2.json",
    )
    base.CURRENT_FILES = CURRENT_FILES


def validate_contracts(require_generated: bool = False) -> dict[str, Any]:
    """运行全部 V3 检查并追加 CURRENT 完整性门；run V3 checks plus CURRENT gates."""

    configure_base_validator()
    # 中文：首版生成文件不属于 CURRENT 入口，故关闭其 require_generated 再检查本入口文件。
    # English: Ignore first-entry generated files, then check CURRENT generated files here.
    report = base.validate_contracts(require_generated=False)
    extra_failures: list[dict[str, str]] = []
    if require_generated:
        for path in (REPORT_PATH, TREE_PATH):
            if not path.is_file():
                extra_failures.append(
                    {
                        "file": path.name,
                        "rule": "missing_generated_file",
                        "detail": "Run current validator with --write-report",
                    }
                )
    report["failures"].extend(extra_failures)
    report["failure_count"] = len(report["failures"])
    report["status"] = "pass" if not report["failures"] else "fail"
    report["validator_revision"] = "current_active_registry_v1"
    report["routing_registry"] = ACTIVE_ROUTING_REGISTRY
    return report


def render_tree() -> str:
    """渲染 CURRENT 权威文件树；render the CURRENT authority tree."""

    lines = [
        "# M1 V3 CURRENT 权威文件树与完整性",
        "",
        "> 由 `tools/validate_m1_contracts_v3_current.py --write-report` 生成。",
        "",
        "| File | Bytes | SHA-256 | Content |",
        "|---|---:|---|---|",
    ]
    for relative_path in CURRENT_FILES:
        path = PACKAGE_ROOT / relative_path
        payload = path.read_bytes()
        lines.append(
            f"| `{relative_path}` | {len(payload)} | `{base.sha256_bytes(payload)}` | "
            f"{base.describe(relative_path, payload)} |"
        )
    lines.extend(
        [
            f"| `{REPORT_PATH.name}` | self | intentionally omitted | CURRENT machine verification |",
            f"| `{TREE_PATH.name}` | self | intentionally omitted | CURRENT integrity tree |",
            "",
            f"- CURRENT authority files including generated indexes: **{len(CURRENT_FILES) + 2}**.",
            "- 首版 V3 validator/迁移 registry 保留为历史，CURRENT 使用 active registry。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析命令行；parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true", help="Write CURRENT report and tree inside M1.")
    return parser.parse_args()


def main() -> int:
    """运行 CURRENT 验证并返回确定性退出码；run CURRENT validation."""

    args = parse_args()
    base.checked_relative(PACKAGE_ROOT)
    if args.write_report:
        preflight = validate_contracts(require_generated=False)
        # 中文：使用首版的安全原子写工具，但写入目标固定为本入口路径。
        # English: Reuse the safe atomic writer with paths pinned by this entry.
        base.atomic_write_json(
            REPORT_PATH,
            {
                "contract_version": base.CONTRACT_VERSION,
                "status": "generating",
                "preflight_status": preflight["status"],
            },
        )
        base.atomic_write_text(TREE_PATH, render_tree())
        report = validate_contracts(require_generated=True)
        report["preflight_status"] = preflight["status"]
        base.atomic_write_json(REPORT_PATH, report)
    else:
        report = validate_contracts(require_generated=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


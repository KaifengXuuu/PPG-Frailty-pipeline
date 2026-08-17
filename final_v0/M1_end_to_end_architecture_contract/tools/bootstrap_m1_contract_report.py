#!/usr/bin/env python3
"""首次生成 M1 报告与包树；bootstrap the first M1 report and package tree.

中文：主验证器在只读验证时要求两个生成文件存在。本工具只解决第一次生成时的
自举顺序，随后主验证器的默认只读模式和 ``--write-report`` 模式都可正常使用。

English: The main validator requires both generated files during read-only verification.
This helper resolves only the first-generation ordering; afterwards both the default mode
and ``--write-report`` mode of the main validator work normally.
"""

from __future__ import annotations

import json

from validate_m1_contracts import (
    CONTRACT_VERSION,
    REPORT_PATH,
    TREE_PATH,
    atomic_write_json,
    atomic_write_text,
    render_tree,
    validate_contracts,
)


def main() -> int:
    """生成自举文件并执行最终验证；write bootstrap files and run final validation."""

    preflight = validate_contracts(require_generated=False)
    # 中文：先创建合法占位报告，避免“报告尚未生成却必须存在”的循环。
    # English: Seed a valid placeholder to break the first-generation existence cycle.
    atomic_write_json(
        REPORT_PATH,
        {
            "contract_version": CONTRACT_VERSION,
            "status": "generating",
            "preflight_status": preflight["status"],
        },
    )
    # 中文：包树明确排除报告和树自身的哈希，因此最终重写不会产生漂移。
    # English: The tree excludes report/tree self-hashes, so the final rewrite is stable.
    atomic_write_text(TREE_PATH, render_tree())
    report = validate_contracts(require_generated=True)
    report["preflight_status"] = preflight["status"]
    atomic_write_json(REPORT_PATH, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


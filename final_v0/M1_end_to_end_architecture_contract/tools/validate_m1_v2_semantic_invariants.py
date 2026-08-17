#!/usr/bin/env python3
"""验证 M1 V2 的运行时语义不变量；validate M1 V2 runtime semantic invariants.

中文
----
JSON Schema 负责结构，本工具补充状态机、概率和、动作 owner、provider fallback、
locked bundle、hash 与相对路径语义。默认模式只读；`--write-report` 仅在
M1 包内写入确定性报告。工具不安装依赖、不运行模型、不访问网络。

English
-------
JSON Schema covers structure. This tool supplements state-machine, probability-sum,
action-owner, provider-fallback, locked-bundle, hash, and relative-path semantics.
Default mode is read-only; `--write-report` writes one deterministic report inside
the M1 package. It installs nothing, runs no model, and accesses no network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Callable


# 中文：固定写入 final_v0/M1 包，cwd 不能重定向。
# English: Pin writes to the final_v0/M1 package; cwd cannot redirect them.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PACKAGE_ROOT / "M1_SEMANTIC_INVARIANTS_V2.json"
NO_RESULT_STATUSES = {"invalid_input", "insufficient_quality", "processing_lag", "runtime_error"}
LABELS = ("pre_frail", "robust_non_frail", "young")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


def checked_relative(path: Path) -> None:
    """拒绝工具写出 M1 包；reject writes outside the M1 package."""

    try:
        path.resolve().relative_to(PACKAGE_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Path escapes M1 package: {path}") from exc


def atomic_write_json(path: Path, data: Any) -> None:
    """原子写严格 JSON；atomically write strict JSON."""

    checked_relative(path)
    payload = (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def validate_bundle_relative_path(value: object) -> list[str]:
    """检查 bundle 内 POSIX 相对路径；validate a bundle-internal POSIX relative path."""

    if not isinstance(value, str) or not value:
        return ["path_empty_or_not_string"]
    # 中文：反斜杠和 drive letter 在 Linux Path 下可能被误当普通字符，需显式拒绝。
    # English: Explicitly reject backslashes and drives that Linux Path may treat as text.
    if value.startswith("/") or value.startswith("\\") or "\\" in value or DRIVE_PATTERN.match(value):
        return ["path_not_posix_relative"]
    if ".." in PurePosixPath(value).parts:
        return ["path_traversal"]
    return []


def validate_output(result: dict[str, Any]) -> list[str]:
    """检查输出状态机、概率与动作 owner；validate output state, probabilities, and action owner."""

    errors: list[str] = []
    status = result.get("status")
    probabilities = result.get("probabilities")
    label = result.get("predicted_label")
    confidence = result.get("confidence")

    if status == "ok":
        if not isinstance(probabilities, dict) or set(probabilities) != set(LABELS):
            errors.append("ok_requires_three_probabilities")
        else:
            values = [probabilities[name] for name in LABELS]
            if not all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 for value in values):
                errors.append("probability_range_or_finite")
            elif not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
                errors.append("probability_sum")
        if label not in LABELS:
            errors.append("ok_requires_label")
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            errors.append("ok_requires_confidence")
    elif status in NO_RESULT_STATUSES:
        if probabilities is not None or label is not None or confidence is not None:
            errors.append("no_result_must_clear_prediction")

    quality = result.get("quality_summary", {})
    owner = quality.get("action_owner")
    counts = quality.get("action_counts", {})
    if owner == "sqi" and counts.get("COARSE_REPLACE", 0) != 0:
        errors.append("sqi_owner_cannot_replace")
    if owner == "coarse_denoise" and (counts.get("SQI_DROP", 0) != 0 or counts.get("SQI_WEIGHT", 0) != 0):
        errors.append("coarse_owner_cannot_apply_sqi_action")
    return errors


def validate_config(config: dict[str, Any]) -> list[str]:
    """检查 provider chain 与 locked artifact；validate provider chain and locked artifacts."""

    errors: list[str] = []
    runtime = config.get("runtime", {})
    engine = str(runtime.get("engine", ""))
    providers = runtime.get("execution_providers", [])
    if "onnxruntime" in engine and (not isinstance(providers, list) or not providers or providers[-1] != "CPUExecutionProvider"):
        errors.append("onnx_provider_chain_requires_cpu_last")
    if runtime.get("device") == "accelerator_with_cpu_fallback" and len(providers) < 2:
        errors.append("accelerator_requires_primary_and_cpu")

    if config.get("deployment_state") == "deploy_locked":
        artifacts = config.get("artifacts", {})
        files = artifacts.get("files", [])
        if artifacts.get("status") != "locked" or not isinstance(files, list) or not files:
            errors.append("locked_bundle_requires_files")
        for artifact in files if isinstance(files, list) else []:
            errors.extend(validate_bundle_relative_path(artifact.get("relative_path")))
            digest = artifact.get("sha256")
            if not isinstance(digest, str) or HASH_PATTERN.fullmatch(digest) is None:
                errors.append("artifact_sha256")
            size = artifact.get("bytes")
            if not isinstance(size, int) or size < 0:
                errors.append("artifact_bytes")
        thresholds = config.get("thresholds", {})
        if not thresholds:
            errors.append("locked_thresholds_required")
        for name, value in thresholds.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"locked_threshold_invalid:{name}")
    return errors


def base_output() -> dict[str, Any]:
    """生成最小有效 ok 输出；build a minimal semantically valid ok output."""

    return {
        "status": "ok",
        "probabilities": {"pre_frail": 0.2, "robust_non_frail": 0.7, "young": 0.1},
        "predicted_label": "robust_non_frail",
        "confidence": 0.7,
        "quality_summary": {
            "action_owner": "sqi",
            "action_counts": {"KEEP_RAW": 1, "SQI_DROP": 0, "SQI_WEIGHT": 0, "COARSE_REPLACE": 0},
        },
    }


def base_locked_config() -> dict[str, Any]:
    """生成最小有效 locked config；build a minimal semantically valid locked config."""

    return {
        "deployment_state": "deploy_locked",
        "runtime": {
            "engine": "hybrid_python_onnxruntime",
            "device": "accelerator_with_cpu_fallback",
            "execution_providers": ["VerifiedVendorEP", "CPUExecutionProvider"],
        },
        "thresholds": {"motion_probability": 0.5, "sqi_accept": 0.6},
        "artifacts": {
            "status": "locked",
            "files": [
                {
                    "relative_path": "artifacts/model.onnx",
                    "sha256": hashlib.sha256(b"fixture").hexdigest(),
                    "bytes": 7,
                }
            ],
        },
    }


def run_tests() -> dict[str, Any]:
    """运行正反例；run positive and negative semantic fixtures."""

    tests: list[tuple[str, Callable[[], list[str]], bool]] = []
    tests.append(("valid_ok_output", lambda: validate_output(base_output()), True))

    invalid_ok = base_output()
    invalid_ok["probabilities"] = None
    tests.append(("reject_ok_without_probabilities", lambda: validate_output(invalid_ok), False))

    bad_sum = base_output()
    bad_sum["probabilities"] = {"pre_frail": 0.2, "robust_non_frail": 0.2, "young": 0.2}
    tests.append(("reject_probability_sum", lambda: validate_output(bad_sum), False))

    valid_no_result = base_output()
    valid_no_result.update({"status": "insufficient_quality", "probabilities": None, "predicted_label": None, "confidence": None})
    tests.append(("valid_explicit_no_result", lambda: validate_output(valid_no_result), True))

    stale_no_result = base_output()
    stale_no_result["status"] = "processing_lag"
    tests.append(("reject_stale_prediction_on_lag", lambda: validate_output(stale_no_result), False))

    bad_sqi_owner = base_output()
    bad_sqi_owner["quality_summary"]["action_counts"]["COARSE_REPLACE"] = 1
    tests.append(("reject_dual_quality_action", lambda: validate_output(bad_sqi_owner), False))

    tests.append(("valid_locked_bundle", lambda: validate_config(base_locked_config()), True))

    absolute = base_locked_config()
    absolute["artifacts"]["files"][0]["relative_path"] = "/etc/passwd"
    tests.append(("reject_absolute_artifact_path", lambda: validate_config(absolute), False))

    traversal = base_locked_config()
    traversal["artifacts"]["files"][0]["relative_path"] = "artifacts/../outside.onnx"
    tests.append(("reject_artifact_traversal", lambda: validate_config(traversal), False))

    missing_external = base_locked_config()
    missing_external["artifacts"]["files"] = []
    tests.append(("reject_locked_bundle_without_files", lambda: validate_config(missing_external), False))

    null_threshold = base_locked_config()
    null_threshold["thresholds"]["sqi_accept"] = None
    tests.append(("reject_locked_null_threshold", lambda: validate_config(null_threshold), False))

    missing_cpu = base_locked_config()
    missing_cpu["runtime"]["execution_providers"] = ["VerifiedVendorEP"]
    tests.append(("reject_accelerator_without_cpu_fallback", lambda: validate_config(missing_cpu), False))

    failures: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for name, check, expected_valid in tests:
        errors = check()
        observed_valid = not errors
        passed = observed_valid == expected_valid
        details.append(
            {
                "name": name,
                "expected_valid": expected_valid,
                "observed_valid": observed_valid,
                "errors": errors,
                "status": "pass" if passed else "fail",
            }
        )
        if not passed:
            failures.append(details[-1])

    return {
        "contract_version": "m1.architecture.v2",
        "validator": "semantic_invariants_v1",
        "status": "pass" if not failures else "fail",
        "test_count": len(tests),
        "passed_test_count": len(tests) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "tests": details,
        "scope": [
            "output_state_machine",
            "probability_sum",
            "quality_action_owner",
            "provider_cpu_fallback",
            "locked_bundle_artifacts",
            "threshold_finiteness",
            "artifact_path_containment",
        ],
        "model_execution": "not_run",
    }


def parse_args() -> argparse.Namespace:
    """解析 CLI；parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true", help="Write M1_SEMANTIC_INVARIANTS_V2.json inside the M1 package.")
    return parser.parse_args()


def main() -> int:
    """执行语义测试；run semantic tests."""

    args = parse_args()
    report = run_tests()
    if args.write_report:
        atomic_write_json(REPORT_PATH, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


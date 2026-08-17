#!/usr/bin/env python3
"""验证 M1 V3 顺序路由语义；validate M1 V3 sequential routing semantics.

中文
----
本工具使用零第三方依赖的正反例 fixture 检查 SQI-first、可选 Motion、high-quality
bypass、run-level 手动互斥策略、fail-closed、FeatureBlock 来源、动作守恒和
coverage。默认只读；`--write-report` 仅在 M1 包内原子写入确定性报告。

English
-------
This zero-dependency fixture suite checks SQI-first routing, optional Motion,
high-quality bypass, run-level mutually exclusive policy, fail-closed behavior,
FeatureBlock provenance, action conservation, and coverage. Default mode is
read-only; `--write-report` atomically writes one deterministic M1 report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable


# 中文：固定报告位置，避免 cwd 将写入引向工作区其他位置。
# English: Pin the report path so cwd cannot redirect writes elsewhere.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PACKAGE_ROOT / "M1_ROUTING_INVARIANTS_V3.json"
POLICIES = {"drop", "denoise_then_extract_features"}
QUALITY_STATES = {"high", "low", "unrecoverable", "unknown"}
MOTION_STATES = {"static", "motion", "not_evaluated", "unknown"}
DENOISE_ACTIONS = {
    "DENOISE_LOW_QUALITY_THEN_EXTRACT",
    "DENOISE_MOTION_THEN_EXTRACT",
    "DENOISE_LOW_QUALITY_AND_MOTION_THEN_EXTRACT",
}
POLICY_DROP_ACTIONS = {
    "POLICY_DROP_LOW_QUALITY",
    "POLICY_DROP_MOTION",
    "POLICY_DROP_LOW_QUALITY_AND_MOTION",
}
FORCED_DROP_ACTIONS = {"FORCE_DROP_INVALID", "FORCE_DROP_UNRECOVERABLE"}
ALL_ACTIONS = (
    FORCED_DROP_ACTIONS
    | POLICY_DROP_ACTIONS
    | DENOISE_ACTIONS
    | {"RETURN_HIGH_QUALITY_TO_FEATURES", "FAILURE_NO_RESULT"}
)


def checked_relative(path: Path) -> None:
    """拒绝写出 M1 包；reject writes outside the M1 package."""

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


def action_for_reason(prefix: str, sqi_state: str, motion_state: str) -> str:
    """为 low/motion/both 生成唯一动作；map low/motion/both to one action."""

    low = sqi_state == "low"
    motion = motion_state == "motion"
    if low and motion:
        suffix = "LOW_QUALITY_AND_MOTION"
    elif low:
        suffix = "LOW_QUALITY"
    elif motion:
        suffix = "MOTION"
    else:
        raise ValueError("degraded route requires low SQI or motion")
    return f"{prefix}_{suffix}" + ("_THEN_EXTRACT" if prefix == "DENOISE" else "")


def route_window(
    *,
    valid: bool,
    sqi_state: str,
    motion_enabled: bool,
    motion_state: str,
    manual_policy: str,
    denoiser_ok: bool = True,
    feature_ok: bool = True,
) -> dict[str, Any]:
    """执行合同状态机 fixture；execute the contract state-machine fixture.

    中文：这是合同 oracle，不是生产算法。它明确检查模块调用与输出来源，防止
    high/drop 路线误启 denoiser 或失败后静默回退 raw。
    English: This is a contract oracle, not a production algorithm. It tracks calls
    and provenance to prevent denoiser use on high/drop paths or raw fallback.
    """

    if sqi_state not in QUALITY_STATES:
        raise ValueError("unknown sqi_state")
    if motion_state not in MOTION_STATES:
        raise ValueError("unknown motion_state")
    if manual_policy not in POLICIES:
        raise ValueError("manual_policy must be one frozen V3 value")
    if motion_enabled and motion_state == "not_evaluated":
        raise ValueError("enabled motion detector cannot be not_evaluated")
    if not motion_enabled and motion_state != "not_evaluated":
        raise ValueError("disabled motion detector must be not_evaluated")

    base = {
        "action": None,
        "reason": None,
        "denoiser_executed": False,
        "feature_extractor_executed": False,
        "feature_source": "none",
        "features_present": False,
        "terminal_status": None,
    }
    if not valid:
        base.update(
            action="FORCE_DROP_INVALID",
            reason="INVALID_INPUT",
            terminal_status="expected_abstention",
        )
        return base

    # 中文：SQI 是所有 valid 窗口的必做一级门；unknown 不能被当作 high。
    # English: SQI is mandatory for every valid window; unknown never means high.
    if sqi_state == "unknown":
        base.update(action="FAILURE_NO_RESULT", reason="SQI_UNAVAILABLE", terminal_status="failure")
        return base
    if motion_enabled and motion_state == "unknown":
        base.update(action="FAILURE_NO_RESULT", reason="MOTION_UNAVAILABLE", terminal_status="failure")
        return base
    if sqi_state == "unrecoverable":
        base.update(
            action="FORCE_DROP_UNRECOVERABLE",
            reason="UNRECOVERABLE_QUALITY",
            terminal_status="expected_abstention",
        )
        return base

    # 中文：只有 high 且非 motion 可绕过去噪；feature 失败仍必须显式失败。
    # English: Only high and non-motion bypasses denoising; feature failure is explicit.
    if sqi_state == "high" and motion_state in {"static", "not_evaluated"}:
        if not feature_ok:
            base.update(
                action="FAILURE_NO_RESULT",
                reason="FEATURE_EXTRACTION_FAILURE",
                feature_extractor_executed=True,
                terminal_status="failure",
            )
            return base
        base.update(
            action="RETURN_HIGH_QUALITY_TO_FEATURES",
            reason="HIGH_QUALITY_NON_MOTION",
            feature_extractor_executed=True,
            feature_source="preprocessed_raw",
            features_present=True,
            terminal_status="success",
        )
        return base

    # 中文：到这里必然是 low SQI、motion，或两者同时发生。
    # English: Reaching this branch means low SQI, motion, or both.
    if manual_policy == "drop":
        base.update(
            action=action_for_reason("POLICY_DROP", sqi_state, motion_state),
            reason="LOW_QUALITY_AND_MOTION" if sqi_state == "low" and motion_state == "motion" else (
                "LOW_QUALITY" if sqi_state == "low" else "MOTION"
            ),
            terminal_status="expected_abstention",
        )
        return base

    base["denoiser_executed"] = True
    if not denoiser_ok:
        base.update(action="FAILURE_NO_RESULT", reason="DENOISER_FAILURE", terminal_status="failure")
        return base
    base["feature_extractor_executed"] = True
    if not feature_ok:
        base.update(action="FAILURE_NO_RESULT", reason="FEATURE_EXTRACTION_FAILURE", terminal_status="failure")
        return base
    base.update(
        action=action_for_reason("DENOISE", sqi_state, motion_state),
        reason="LOW_QUALITY_AND_MOTION" if sqi_state == "low" and motion_state == "motion" else (
            "LOW_QUALITY" if sqi_state == "low" else "MOTION"
        ),
        feature_source="denoiser_features",
        features_present=True,
        terminal_status="success",
    )
    return base


def validate_route_shape(result: dict[str, Any]) -> list[str]:
    """检查单窗结果内部一致性；validate internal per-window consistency."""

    errors: list[str] = []
    action = result.get("action")
    if action not in ALL_ACTIONS:
        errors.append("unknown_or_missing_terminal_action")
    if action == "RETURN_HIGH_QUALITY_TO_FEATURES":
        if result.get("denoiser_executed") is not False:
            errors.append("high_quality_must_bypass_denoiser")
        if result.get("feature_source") != "preprocessed_raw" or result.get("features_present") is not True:
            errors.append("high_quality_feature_source")
    if action in POLICY_DROP_ACTIONS | FORCED_DROP_ACTIONS:
        if result.get("denoiser_executed") or result.get("feature_extractor_executed"):
            errors.append("drop_must_call_no_downstream_module")
        if result.get("features_present") or result.get("feature_source") != "none":
            errors.append("drop_must_clear_features")
    if action in DENOISE_ACTIONS:
        if result.get("denoiser_executed") is not True or result.get("feature_extractor_executed") is not True:
            errors.append("denoise_success_requires_both_calls")
        if result.get("feature_source") != "denoiser_features" or result.get("features_present") is not True:
            errors.append("denoise_feature_source")
    if action == "FAILURE_NO_RESULT":
        if result.get("features_present") or result.get("feature_source") != "none":
            errors.append("failure_must_clear_features")
    return errors


def validate_run_policy(policies: list[str]) -> list[str]:
    """检查同一 run 的策略不可逐窗变化；ensure one immutable policy per run."""

    if not policies or any(policy not in POLICIES for policy in policies):
        return ["run_policy_missing_or_invalid"]
    if len(set(policies)) != 1:
        return ["window_level_policy_switch_forbidden"]
    return []


def validate_routing_summary(summary: dict[str, Any]) -> list[str]:
    """检查动作计数与 coverage 守恒；validate action-count and coverage conservation."""

    errors: list[str] = []
    scheduled = summary.get("scheduled_window_count")
    counts = summary.get("action_counts", {})
    if not isinstance(scheduled, int) or scheduled < 0:
        return ["scheduled_count_invalid"]
    if set(counts) != ALL_ACTIONS:
        errors.append("action_count_key_set")
        return errors
    if not all(isinstance(value, int) and value >= 0 for value in counts.values()):
        errors.append("action_count_value")
        return errors
    if sum(counts.values()) != scheduled:
        errors.append("terminal_action_conservation")

    usable = counts["RETURN_HIGH_QUALITY_TO_FEATURES"] + sum(counts[name] for name in DENOISE_ACTIONS)
    dropped = sum(counts[name] for name in FORCED_DROP_ACTIONS | POLICY_DROP_ACTIONS)
    failures = counts["FAILURE_NO_RESULT"]
    if summary.get("usable_feature_window_count") != usable:
        errors.append("usable_feature_count")
    if summary.get("dropped_window_count") != dropped:
        errors.append("dropped_count")
    if summary.get("failure_window_count") != failures:
        errors.append("failure_count")
    expected_coverage = 0.0 if scheduled == 0 else usable / scheduled
    coverage = summary.get("window_coverage")
    if not isinstance(coverage, (int, float)) or not math.isclose(coverage, expected_coverage, abs_tol=1e-12):
        errors.append("window_coverage")
    time_coverage = summary.get("time_coverage")
    if not isinstance(time_coverage, (int, float)) or not 0 <= time_coverage <= 1:
        errors.append("time_coverage_range")
    return errors


def base_summary() -> dict[str, Any]:
    """生成守恒的 summary fixture；build a conserved routing summary fixture."""

    counts = {name: 0 for name in ALL_ACTIONS}
    counts.update(
        {
            "RETURN_HIGH_QUALITY_TO_FEATURES": 4,
            "POLICY_DROP_LOW_QUALITY": 2,
            "DENOISE_MOTION_THEN_EXTRACT": 1,
            "FAILURE_NO_RESULT": 1,
        }
    )
    return {
        "scheduled_window_count": 8,
        "action_counts": counts,
        "usable_feature_window_count": 5,
        "dropped_window_count": 2,
        "failure_window_count": 1,
        "window_coverage": 0.625,
        "time_coverage": 0.58,
    }


def route_check(kwargs: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """运行一个路由 fixture 并比较关键字段；run and compare one route fixture."""

    try:
        result = route_window(**kwargs)
    except ValueError as exc:
        return [f"unexpected_exception:{exc}"]
    errors = validate_route_shape(result)
    for key, value in expected.items():
        if result.get(key) != value:
            errors.append(f"{key}:expected={value}:observed={result.get(key)}")
    return errors


def rejection_check(kwargs: dict[str, Any]) -> list[str]:
    """期望非法状态组合被拒绝；expect an invalid state/config combination to fail."""

    try:
        route_window(**kwargs)
    except ValueError:
        return []
    return ["invalid_combination_was_accepted"]


def run_tests() -> dict[str, Any]:
    """运行正反例测试集；run positive and negative fixture tests."""

    tests: list[tuple[str, Callable[[], list[str]], bool]] = []
    common = {"valid": True, "denoiser_ok": True, "feature_ok": True}

    def add_route(name: str, kwargs: dict[str, Any], expected: dict[str, Any]) -> None:
        tests.append((name, lambda k=kwargs, e=expected: route_check(k, e), True))

    add_route(
        "high_motion_disabled_bypasses_denoiser",
        {**common, "sqi_state": "high", "motion_enabled": False, "motion_state": "not_evaluated", "manual_policy": "drop"},
        {"action": "RETURN_HIGH_QUALITY_TO_FEATURES", "denoiser_executed": False, "feature_source": "preprocessed_raw"},
    )
    add_route(
        "high_static_bypasses_denoiser",
        {**common, "sqi_state": "high", "motion_enabled": True, "motion_state": "static", "manual_policy": "denoise_then_extract_features"},
        {"action": "RETURN_HIGH_QUALITY_TO_FEATURES", "denoiser_executed": False},
    )
    add_route(
        "high_motion_drop",
        {**common, "sqi_state": "high", "motion_enabled": True, "motion_state": "motion", "manual_policy": "drop"},
        {"action": "POLICY_DROP_MOTION", "features_present": False},
    )
    add_route(
        "high_motion_denoise",
        {**common, "sqi_state": "high", "motion_enabled": True, "motion_state": "motion", "manual_policy": "denoise_then_extract_features"},
        {"action": "DENOISE_MOTION_THEN_EXTRACT", "feature_source": "denoiser_features"},
    )
    add_route(
        "low_static_drop",
        {**common, "sqi_state": "low", "motion_enabled": True, "motion_state": "static", "manual_policy": "drop"},
        {"action": "POLICY_DROP_LOW_QUALITY", "denoiser_executed": False},
    )
    add_route(
        "low_static_denoise",
        {**common, "sqi_state": "low", "motion_enabled": True, "motion_state": "static", "manual_policy": "denoise_then_extract_features"},
        {"action": "DENOISE_LOW_QUALITY_THEN_EXTRACT", "features_present": True},
    )
    add_route(
        "low_motion_drop_preserves_both_reasons",
        {**common, "sqi_state": "low", "motion_enabled": True, "motion_state": "motion", "manual_policy": "drop"},
        {"action": "POLICY_DROP_LOW_QUALITY_AND_MOTION", "reason": "LOW_QUALITY_AND_MOTION"},
    )
    add_route(
        "low_motion_denoise_preserves_both_reasons",
        {**common, "sqi_state": "low", "motion_enabled": True, "motion_state": "motion", "manual_policy": "denoise_then_extract_features"},
        {"action": "DENOISE_LOW_QUALITY_AND_MOTION_THEN_EXTRACT", "reason": "LOW_QUALITY_AND_MOTION"},
    )
    add_route(
        "invalid_forced_drop",
        {**common, "valid": False, "sqi_state": "high", "motion_enabled": True, "motion_state": "motion", "manual_policy": "denoise_then_extract_features"},
        {"action": "FORCE_DROP_INVALID", "denoiser_executed": False},
    )
    add_route(
        "unrecoverable_forced_drop",
        {**common, "sqi_state": "unrecoverable", "motion_enabled": True, "motion_state": "motion", "manual_policy": "denoise_then_extract_features"},
        {"action": "FORCE_DROP_UNRECOVERABLE", "denoiser_executed": False},
    )
    add_route(
        "sqi_unknown_fail_closed",
        {**common, "sqi_state": "unknown", "motion_enabled": False, "motion_state": "not_evaluated", "manual_policy": "drop"},
        {"action": "FAILURE_NO_RESULT", "reason": "SQI_UNAVAILABLE"},
    )
    add_route(
        "motion_unknown_fail_closed",
        {**common, "sqi_state": "high", "motion_enabled": True, "motion_state": "unknown", "manual_policy": "drop"},
        {"action": "FAILURE_NO_RESULT", "reason": "MOTION_UNAVAILABLE"},
    )
    add_route(
        "denoiser_failure_no_raw_fallback",
        {**common, "sqi_state": "low", "motion_enabled": True, "motion_state": "static", "manual_policy": "denoise_then_extract_features", "denoiser_ok": False},
        {"action": "FAILURE_NO_RESULT", "feature_source": "none", "features_present": False},
    )
    add_route(
        "raw_feature_failure_is_explicit",
        {**common, "sqi_state": "high", "motion_enabled": True, "motion_state": "static", "manual_policy": "drop", "feature_ok": False},
        {"action": "FAILURE_NO_RESULT", "reason": "FEATURE_EXTRACTION_FAILURE"},
    )
    add_route(
        "denoised_feature_failure_is_explicit",
        {**common, "sqi_state": "low", "motion_enabled": True, "motion_state": "static", "manual_policy": "denoise_then_extract_features", "feature_ok": False},
        {"action": "FAILURE_NO_RESULT", "reason": "FEATURE_EXTRACTION_FAILURE", "denoiser_executed": True},
    )

    tests.append(
        (
            "reject_disabled_motion_reported_static",
            lambda: rejection_check({**common, "sqi_state": "high", "motion_enabled": False, "motion_state": "static", "manual_policy": "drop"}),
            True,
        )
    )
    tests.append(
        (
            "reject_enabled_motion_not_evaluated",
            lambda: rejection_check({**common, "sqi_state": "high", "motion_enabled": True, "motion_state": "not_evaluated", "manual_policy": "drop"}),
            True,
        )
    )
    tests.append(
        (
            "reject_third_manual_policy",
            lambda: rejection_check({**common, "sqi_state": "low", "motion_enabled": False, "motion_state": "not_evaluated", "manual_policy": "auto_best"}),
            True,
        )
    )
    tests.append(("valid_run_level_policy_lock", lambda: validate_run_policy(["drop", "drop", "drop"]), True))
    tests.append(("reject_window_level_policy_switch", lambda: validate_run_policy(["drop", "denoise_then_extract_features"]), False))
    tests.append(("valid_action_and_coverage_conservation", lambda: validate_routing_summary(base_summary()), True))

    bad_total = base_summary()
    bad_total["scheduled_window_count"] = 9
    tests.append(("reject_action_count_nonconservation", lambda: validate_routing_summary(bad_total), False))
    bad_coverage = base_summary()
    bad_coverage["window_coverage"] = 0.9
    tests.append(("reject_coverage_recomputed_after_drop", lambda: validate_routing_summary(bad_coverage), False))
    bad_usable = base_summary()
    bad_usable["usable_feature_window_count"] = 7
    tests.append(("reject_stale_or_fabricated_feature_count", lambda: validate_routing_summary(bad_usable), False))

    failures: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for name, check, expected_valid in tests:
        errors = check()
        observed_valid = not errors
        passed = observed_valid == expected_valid
        detail = {
            "name": name,
            "expected_valid": expected_valid,
            "observed_valid": observed_valid,
            "errors": errors,
            "status": "pass" if passed else "fail",
        }
        details.append(detail)
        if not passed:
            failures.append(detail)

    return {
        "contract_version": "m1.architecture.v3",
        "validator": "sequential_sqi_motion_routing_invariants_v1",
        "status": "pass" if not failures else "fail",
        "test_count": len(tests),
        "passed_test_count": len(tests) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "tests": details,
        "scope": [
            "sqi_required_first_stage",
            "motion_optional_not_evaluated_semantics",
            "motion_overrides_high_sqi_bypass",
            "high_quality_denoiser_bypass",
            "run_level_manual_policy_exclusivity",
            "drop_clears_features",
            "denoiser_failure_no_raw_fallback",
            "quality_motion_reason_preservation",
            "terminal_action_conservation",
            "coverage_denominator_frozen_before_routing"
        ],
        "model_execution": "not_run_contract_fixtures_only"
    }


def parse_args() -> argparse.Namespace:
    """解析命令行；parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true", help="Write M1_ROUTING_INVARIANTS_V3.json inside M1.")
    return parser.parse_args()


def main() -> int:
    """运行测试并返回确定性退出码；run tests and return a deterministic status."""

    args = parse_args()
    report = run_tests()
    if args.write_report:
        atomic_write_json(REPORT_PATH, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


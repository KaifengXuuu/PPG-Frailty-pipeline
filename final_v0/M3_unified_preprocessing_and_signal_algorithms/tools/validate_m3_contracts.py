#!/usr/bin/env python3
"""验证 M3 合同、权威绑定、机器证据与正式测试报告。

Validate M3 contracts, authority bindings, machine evidence, and the formal test
report.  The validator is deliberately fail-closed: a stale test snapshot, a changed
M1/M2 authority artifact, an undeclared reason code, or a future-active implementation
outside the M3 source boundary cannot pass silently.

中文：``--write-report`` 仅写本 M3 包内的验证报告和包树；缺少并行构建中的
schema/registry/example 时记录为 ``wait``，而不是抛异常或伪造成功。所有 JSON
均使用拒绝 NaN/Infinity、重复键和非标准常量的严格解析/序列化。

English: ``--write-report`` writes only the package-local verification report and
package tree.  Concurrently produced schemas, registries, and examples are reported as
``wait`` while absent instead of crashing or being treated as passes.  JSON parsing and
writing reject NaN/Infinity, duplicate keys, and non-standard constants.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# 中文：所有写入都由 PACKAGE_ROOT 约束；外部 M1/M2 与根代码仅只读。
# English: PACKAGE_ROOT constrains every write; M1/M2 and root code are read-only.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = FINAL_ROOT.parent
M1_ROOT = FINAL_ROOT / "M1_end_to_end_architecture_contract"
M2_ROOT = FINAL_ROOT / "M2_data_manifest_and_evaluation_protocol"
REPORT_PATH = PACKAGE_ROOT / "M3_CONTRACT_VERIFICATION.json"
TREE_PATH = PACKAGE_ROOT / "M3_PACKAGE_TREE.md"

REPORT_SCHEMA_VERSION = "m3.contract_verification.v1"
REPORT_ID = "m3_contract_verification_v1"
VALIDATOR_REVISION = "m3_contract_validator_v1"
MIN_FORMAL_TESTS = 38

DATASET_VERSION_ID = "frailty3_m2_20260815_a054800abda272f6"
FOLD_REGISTRY_ID = "frailty3_future_corrected_sgkf5_v2"
FOLD_PAYLOAD_SHA256 = (
    "0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46"
)
M2_MANIFEST_SHA256 = (
    "bd429ae9c56974ba9ffcb924dfbad0ed930f7d2d47418365754a1929ada06e90"
)
SEEDS = [42, 10042, 20042, 30042, 40042]
ROLES = ["B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2"]
CLASS_NAMES = {"pre_frail", "robust_non_frail", "young"}

# 中文：这些 SHA 是 M3 开始时冻结的上游权威快照，不是运行时自我声明。
# English: These digests freeze upstream authority snapshots at the M3 boundary.
UPSTREAM_FILE_SHA256 = {
    "M1_CONTRACT_VERIFICATION_V3_CURRENT.json": (
        "2bb9e0be3833b35bc3ce838a52d900d118fd1455a149295912aa6a25ac553d85"
    ),
    "quality_routing_registry_v3_active.json": (
        "0d5643de7a19f9c612fdbd29afa31273a37fc91fbc716727092a82a67fa6411e"
    ),
    "M2_BUILD_REPORT.json": (
        "9c7903b4ce0594a1cb53be2835aeba36453f04835b95d45bd6935d63da6d8a8e"
    ),
    "M2_CONTRACT_VERIFICATION.json": (
        "a8cf5f0ac60635a6c81577627a7e1f7331b9b973053d68ddd860c0ac4bcd7cec"
    ),
    "frailty3_future_corrected_sgkf5_v2.json": (
        "c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c"
    ),
}

# 中文：核心文件缺失是失败；并行合同文件缺失先标记 wait，最终仍必须落盘。
# English: Missing core files fail; concurrent contract files wait until finalized.
CORE_REQUIRED_FILES = (
    "README.md",
    "M3_BUILD_REPORT.json",
    "M3_REFERENCE_TEST_RESULTS.json",
    "registries/preprocessing_profiles_v1.json",
    "registries/physiology_algorithms_v1.json",
    "registries/reason_codes_v1.json",
    "schemas/preprocessing_profile.schema.json",
    "evidence/ekf_lpf_frailty3_role_proxy.json",
    "evidence/ekf_lpf_synthetic_comparison.json",
    "evidence/filter_response_comparison.json",
    "evidence/frailty3_signal_integrity_summary.json",
    "evidence/historical_preprocessing_crosswalk_v1.json",
    "evidence/legacy_peak_parity.json",
    "fixtures/reference_fixture_manifest.json",
    "fixtures/ppg_reference_v1.npy",
    "fixtures/ppg_expected_peaks_v1.npy",
    "fixtures/imu_reference_v1.npy",
    "src/m3_signal_core/__init__.py",
    "src/m3_signal_core/contracts.py",
    "src/m3_signal_core/fold_contract.py",
    "src/m3_signal_core/imu.py",
    "src/m3_signal_core/imu_math.py",
    "src/m3_signal_core/imu_runtime.py",
    "src/m3_signal_core/physiology.py",
    "src/m3_signal_core/ppg.py",
    "src/m3_signal_core/quality.py",
    "src/m3_signal_core/reference_evaluation.py",
    "src/m3_signal_core/registry.py",
    "src/m3_signal_core/scaling.py",
    "tests/test_contract_edges.py",
    "tests/test_fold_reference.py",
    "tests/test_imu_physiology.py",
    "tests/test_quality_ppg_scaling.py",
    "tools/build_m3_core_evidence.py",
    "tools/build_m3_frailty_imu_proxy.py",
    "tools/build_m3_reference_fixtures.py",
    "tools/run_m3_reference_tests.py",
    "tools/validate_m3_contracts.py",
)

CONCURRENT_REQUIRED_FILES = (
    "schemas/preprocessing_result.schema.json",
    "schemas/fold_fitted_artifact.schema.json",
    "schemas/physiology_result.schema.json",
    "schemas/module_binding.schema.json",
    "schemas/reference_fixture_manifest.schema.json",
    "registries/module_bindings_v1.json",
    "registries/historical_preprocessing_crosswalk_v1.json",
    "registries/feature_schemas_v1.json",
    "registries/status_mapping_v1.json",
    "examples/m1_pipeline_config_m3_offline.json",
    "examples/m1_pipeline_config_m3_mobile.json",
    "examples/m2_result_provenance_m3_bound.json",
)

EXPECTED_SCHEMA_IDS = {
    "schemas/preprocessing_profile.schema.json": "m3.preprocessing_profile.schema.v1",
    "schemas/preprocessing_result.schema.json": "m3.preprocessing_result.v1",
    "schemas/fold_fitted_artifact.schema.json": "m3.fold_fitted_artifact.v1",
    "schemas/physiology_result.schema.json": "m3.physiology_result.v1",
    "schemas/module_binding.schema.json": "m3.module_binding.v1",
    "schemas/reference_fixture_manifest.schema.json": (
        "m3.reference_fixture_manifest.v1"
    ),
}

EXPECTED_REGISTRY_IDENTITIES = {
    "registries/preprocessing_profiles_v1.json": (
        "registry_id",
        "m3_preprocessing_profiles_corrected_v1",
    ),
    "registries/physiology_algorithms_v1.json": (
        "registry_id",
        "m3_physiology_corrected_v1",
    ),
    "registries/reason_codes_v1.json": (
        "registry_id",
        "m3_reason_codes_corrected_v1",
    ),
    "registries/module_bindings_v1.json": (
        "registry_id",
        "m3_module_bindings_v1",
    ),
    "registries/historical_preprocessing_crosswalk_v1.json": (
        "registry_id",
        "m3_historical_preprocessing_crosswalk_v1",
    ),
    "registries/feature_schemas_v1.json": (
        "registry_id",
        "m3_feature_schemas_v1",
    ),
    "registries/status_mapping_v1.json": (
        "registry_id",
        "m3_status_mapping_v1",
    ),
}

EXPECTED_REGISTRY_SCHEMA_VERSIONS = {
    "registries/reason_codes_v1.json": "m3.reason_codes.v1",
    "registries/module_bindings_v1.json": "m3.module_binding.v1",
    "registries/historical_preprocessing_crosswalk_v1.json": (
        "m3.historical_preprocessing_crosswalk.v1"
    ),
    "registries/feature_schemas_v1.json": "m3.feature_schemas.v1",
    "registries/status_mapping_v1.json": "m3.status_mapping.v1",
}

# 中文：状态码不是任意错误文本；这些运行时状态必须在 registry 中有定义。
# English: Runtime states are controlled codes, not arbitrary exception text.
REQUIRED_RUNTIME_REASON_CODES = {
    "initialization_pending",
    "prediction_only",
    "no_estimate",
    "zero_iqr_channel_requires_no_estimate",
}


class StrictJsonError(ValueError):
    """表示非严格 JSON / Mark non-strict JSON input."""


@dataclass
class Audit:
    """聚合验证结果且区分失败与等待 / Collect failures separately from waits."""

    checks: list[dict[str, str]] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    waiting: list[dict[str, str]] = field(default_factory=list)

    def check(self, condition: bool, rule: str, detail: str = "") -> bool:
        """记录确定性 pass/fail / Record a deterministic pass or failure."""

        status = "pass" if bool(condition) else "fail"
        item = {"rule": str(rule), "status": status, "detail": str(detail)}
        self.checks.append(item)
        if not condition:
            self.failures.append({"rule": str(rule), "detail": str(detail)})
        return bool(condition)

    def wait(self, rule: str, detail: str) -> None:
        """记录并行产物等待项 / Record a concurrent-artifact wait item."""

        item = {"rule": str(rule), "status": "wait", "detail": str(detail)}
        self.checks.append(item)
        self.waiting.append({"rule": str(rule), "detail": str(detail)})


def sha256_file(path: Path) -> str:
    """逐字节计算 SHA-256 / Compute a byte-exact SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    """生成 M2 payload 兼容严格 JSON / Render M2-compatible strict JSON."""

    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def compact_snapshot_bytes(value: Any) -> bytes:
    """生成测试 runner 的紧凑快照 / Render the test-runner compact snapshot."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reject_constant(token: str) -> None:
    """拒绝 NaN/Infinity JSON token / Reject a non-standard JSON constant."""

    raise StrictJsonError(f"non-standard JSON constant: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """拒绝重复对象键 / Reject duplicate object keys."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_load_json(path: Path) -> Any:
    """严格解析 UTF-8 JSON / Parse UTF-8 JSON with strict extensions disabled."""

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )


def checked_target(path: Path) -> Path:
    """拒绝 M3 包外目标 / Reject a write target outside the M3 package."""

    target = path.resolve(strict=False)
    try:
        target.relative_to(PACKAGE_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Refusing write outside M3 package: {target}") from exc
    return target


def atomic_write(path: Path, payload: bytes) -> None:
    """原子写包内生成物 / Atomically write a package-local artifact."""

    target = checked_target(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)


def _get_mapping(value: Any) -> dict[str, Any]:
    """把非对象安全降为空对象 / Safely normalize a non-object to an empty map."""

    return value if isinstance(value, dict) else {}


def _get_sequence(value: Any) -> list[Any]:
    """把非数组安全降为空数组 / Safely normalize a non-array to an empty list."""

    return value if isinstance(value, list) else []


def _is_finite_number(value: Any) -> bool:
    """判断非布尔有限数 / Test whether a value is a finite non-Boolean number."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _load_json(
    audit: Audit,
    path: Path,
    *,
    rule: str,
    wait_if_missing: bool = False,
) -> Any | None:
    """读取必需 JSON 并保留缺失语义 / Load required JSON without crashing."""

    if not path.is_file():
        detail = f"missing: {path}"
        if wait_if_missing:
            audit.wait(rule, detail)
        else:
            audit.check(False, rule, detail)
        return None
    try:
        value = strict_load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, StrictJsonError) as exc:
        audit.check(False, rule, f"strict JSON error: {exc}")
        return None
    audit.check(True, rule, f"strict JSON: {path.name}")
    return value


def validate_required_files_and_json(audit: Audit) -> dict[str, Any]:
    """验证必需文件及全部包内 JSON / Validate required files and package JSON."""

    for relative in CORE_REQUIRED_FILES:
        path = PACKAGE_ROOT / relative
        audit.check(
            path.is_file(),
            f"required_file::{relative}",
            f"exists={path.is_file()}",
        )
    for relative in CONCURRENT_REQUIRED_FILES:
        path = PACKAGE_ROOT / relative
        if path.is_file():
            audit.check(True, f"concurrent_file::{relative}", "finalized")
        else:
            audit.wait(
                f"concurrent_file::{relative}",
                "awaiting parallel contract/schema task",
            )

    parsed: dict[str, Any] = {}
    json_paths = sorted(
        (
            path
            for path in PACKAGE_ROOT.rglob("*.json")
            if path != REPORT_PATH and path.is_file()
        ),
        key=lambda item: item.relative_to(PACKAGE_ROOT).as_posix(),
    )
    for path in json_paths:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        value = _load_json(audit, path, rule=f"strict_json::{relative}")
        if value is not None:
            parsed[relative] = value
    audit.check(
        len(parsed) == len(json_paths),
        "strict_json::all_package_json_parsed",
        f"parsed={len(parsed)}, discovered={len(json_paths)}",
    )
    return parsed


def _has_bilingual_text(text: str) -> bool:
    """检查中英文标记 / Check for both CJK and English text."""

    return bool(re.search(r"[\u4e00-\u9fff]", text)) and bool(
        re.search(r"[A-Za-z]{3}", text)
    )


def validate_python_ast_and_comments(audit: Audit) -> int:
    """验证 AST 与公共 API 双语注释 / Validate AST and bilingual public docs."""

    python_paths = sorted(
        (
            path
            for path in PACKAGE_ROOT.rglob("*.py")
            if "__pycache__" not in path.parts and path.is_file()
        ),
        key=lambda item: item.relative_to(PACKAGE_ROOT).as_posix(),
    )
    parsed: dict[Path, ast.AST] = {}
    for path in python_paths:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            audit.check(False, f"python_ast::{relative}", str(exc))
            continue
        parsed[path] = tree
        audit.check(True, f"python_ast::{relative}", "AST parse succeeded")
        module_doc = ast.get_docstring(tree) or ""
        audit.check(
            _has_bilingual_text(module_doc),
            f"bilingual_module_doc::{relative}",
            "module docstring must contain Chinese and English",
        )
        public_nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not node.name.startswith("_")
        ]
        missing = [
            f"{node.name}@{node.lineno}"
            for node in public_nodes
            if not _has_bilingual_text(ast.get_docstring(node) or "")
        ]
        audit.check(
            not missing,
            f"bilingual_public_api::{relative}",
            "missing=" + ",".join(missing),
        )

    test_count = 0
    for path, tree in parsed.items():
        if path.parent != PACKAGE_ROOT / "tests" or not path.name.startswith("test_"):
            continue
        test_count += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in ast.walk(tree)
        )
    audit.check(
        test_count >= MIN_FORMAL_TESTS,
        "formal_tests::ast_discovered_at_least_38",
        f"discovered={test_count}, minimum={MIN_FORMAL_TESTS}",
    )
    return test_count


def validate_schema_and_registry_ids(
    audit: Audit, parsed: Mapping[str, Any]
) -> None:
    """核对 schema 与 registry 身份 / Verify schema and registry identities."""

    for relative, expected_id in EXPECTED_SCHEMA_IDS.items():
        value = parsed.get(relative)
        if value is None:
            # 中文：文件级检查已记录 fail/wait；此处避免重复误导。
            # English: The file-level gate already recorded fail/wait.
            continue
        observed = _get_mapping(value).get("$id")
        audit.check(
            observed == expected_id,
            f"schema_id::{relative}",
            f"observed={observed!r}, expected={expected_id!r}",
        )
        audit.check(
            _get_mapping(value).get("$schema")
            == "https://json-schema.org/draft/2020-12/schema",
            f"schema_draft::{relative}",
            f"observed={_get_mapping(value).get('$schema')!r}",
        )

    for relative, (field_name, expected_id) in EXPECTED_REGISTRY_IDENTITIES.items():
        value = parsed.get(relative)
        if value is None:
            continue
        observed = _get_mapping(value).get(field_name)
        audit.check(
            observed == expected_id,
            f"registry_id::{relative}",
            f"observed={observed!r}, expected={expected_id!r}",
        )
    for relative, expected_version in EXPECTED_REGISTRY_SCHEMA_VERSIONS.items():
        value = parsed.get(relative)
        if value is None:
            continue
        observed = _get_mapping(value).get("schema_version")
        audit.check(
            observed == expected_version,
            f"registry_schema_version::{relative}",
            f"observed={observed!r}, expected={expected_version!r}",
        )


def _profile_map(registry: Any) -> dict[str, dict[str, Any]]:
    """按 ID 索引 profile / Index profiles by identifier."""

    profiles = _get_sequence(_get_mapping(registry).get("profiles"))
    return {
        str(item.get("profile_id")): item
        for item in profiles
        if isinstance(item, dict) and isinstance(item.get("profile_id"), str)
    }


def _check_profile_fields(
    audit: Audit,
    profiles: Mapping[str, Mapping[str, Any]],
    profile_id: str,
    expected: Mapping[str, Any],
) -> None:
    """核对单个 profile 冻结字段 / Check frozen fields for one profile."""

    profile = profiles.get(profile_id)
    audit.check(
        profile is not None,
        f"profile_exists::{profile_id}",
        f"available={sorted(profiles)}",
    )
    if profile is None:
        return
    for key, expected_value in expected.items():
        observed = profile.get(key)
        audit.check(
            observed == expected_value,
            f"profile_field::{profile_id}::{key}",
            f"observed={observed!r}, expected={expected_value!r}",
        )


def _all_string_values(value: Any) -> Iterable[str]:
    """递归迭代 JSON 字符串 / Recursively iterate JSON string values."""

    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_string_values(item)


def _collect_path_fields(value: Any) -> list[str]:
    """收集 binding 中的实现路径 / Collect implementation paths from bindings."""

    results: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                isinstance(item, str)
                and key
                in {
                    "implementation_path",
                    "implementation_module",
                    "source_path",
                    "module_path",
                }
            ):
                results.append(item)
            results.extend(_collect_path_fields(item))
    elif isinstance(value, list):
        for item in value:
            results.extend(_collect_path_fields(item))
    return results


def validate_profiles_and_bindings(
    audit: Audit, parsed: Mapping[str, Any]
) -> set[str]:
    """验证冻结 profile 与唯一实现边界 / Validate profiles and sole boundary."""

    registry = parsed.get("registries/preprocessing_profiles_v1.json")
    if registry is None:
        return set()
    registry_map = _get_mapping(registry)
    audit.check(
        registry_map.get("status") == "future_active",
        "profiles::registry_future_active",
        f"status={registry_map.get('status')!r}",
    )
    audit.check(
        registry_map.get("future_primary_sampling_rate_hz") == 400,
        "profiles::future_fs_400",
        f"value={registry_map.get('future_primary_sampling_rate_hz')!r}",
    )
    audit.check(
        registry_map.get("training_only_statistics") is True,
        "profiles::training_only_statistics",
        f"value={registry_map.get('training_only_statistics')!r}",
    )
    profiles = _profile_map(registry)
    _check_profile_fields(
        audit,
        profiles,
        "frailty3_static_ppg_400_offline_v1",
        {
            "status": "future_active",
            "modality": "ppg",
            "sampling_rate_hz": 400,
            "bandpass_hz": [0.2, 8.0],
            "butterworth_order": 3,
            "phase_mode": "offline_zero_phase",
            "detrend": "linear",
            "notch": "disabled",
            "channel_order": ["RED", "IR"],
            "preserve_raw_dc_ac": True,
        },
    )
    for profile_id in (
        "frailty3_motion_ppg_400_offline_v1",
        "frailty3_peak_ppg_400_offline_v1",
        "frailty3_denoiser_ppg_400_offline_v1",
    ):
        _check_profile_fields(
            audit,
            profiles,
            profile_id,
            {
                "status": "future_active",
                "modality": "ppg",
                "sampling_rate_hz": 400,
                "bandpass_hz": [0.4, 8.0],
                "butterworth_order": 3,
                "phase_mode": "offline_zero_phase",
                "detrend": "linear",
                "notch": "disabled",
                "channel_order": ["RED", "IR"],
                "preserve_raw_dc_ac": True,
            },
        )
    _check_profile_fields(
        audit,
        profiles,
        "mobile_ppg_400_causal_v1",
        {
            "status": "future_active",
            "modality": "ppg",
            "sampling_rate_hz": 400,
            "bandpass_hz": [0.4, 8.0],
            "butterworth_order": 3,
            "phase_mode": "causal_stateful",
            "detrend": "none",
            "notch": "disabled",
            "preserve_raw_dc_ac": True,
        },
    )
    _check_profile_fields(
        audit,
        profiles,
        "frailty3_raw8_classifier_400_v1",
        {
            "status": "future_active",
            "modality": "multimodal",
            "sampling_rate_hz": 400,
            "channel_order": [
                "RED",
                "IR",
                "AX_dyn",
                "AY_dyn",
                "AZ_dyn",
                "GX",
                "GY",
                "GZ",
            ],
            "ppg_scaling": (
                "per_window_median_iqr_div_1p349_no_clip_zero_iqr_no_estimate"
            ),
            "imu_scaling": "training_fold_robust_scaler_no_clip",
        },
    )
    _check_profile_fields(
        audit,
        profiles,
        "imu_ekf_si_400_causal_v1",
        {
            "status": "future_active_primary",
            "modality": "imu",
            "sampling_rate_hz": 400,
            "input_order": ["AX", "AY", "AZ", "GX", "GY", "GZ"],
            "acceleration_output_unit": "m/s^2",
            "gyroscope_output_unit": "rad/s",
            "acceleration_lowpass_hz": 20.0,
            "gyroscope_lowpass_hz": 40.0,
            "gravity_method": "quaternion_error_state_ekf_without_precalibration",
            "algorithm_key": "ekf",
            "sensor_filter_order": 3,
            "phase_mode": "causal_stateful",
            "silent_fallback": False,
            "jerk_definition": (
                "vector_backward_difference_then_l2_norm_for_scalar_features"
            ),
        },
    )
    _check_profile_fields(
        audit,
        profiles,
        "imu_lpf_si_400_causal_v1",
        {
            "status": "future_active_comparator",
            "modality": "imu",
            "sampling_rate_hz": 400,
            "input_order": ["AX", "AY", "AZ", "GX", "GY", "GZ"],
            "gravity_method": "second_order_lowpass_0p3_hz",
            "algorithm_key": "lpf_0p3",
            "gravity_filter_order": 2,
            "phase_mode": "causal_stateful",
            "silent_fallback": False,
        },
    )
    for legacy_id in (
        "heartbeat_legacy_256_bug_compatible_v1",
        "hybrid_legacy_500_bug_compatible_v1",
    ):
        _check_profile_fields(
            audit,
            profiles,
            legacy_id,
            {"status": "historical_reproduction_only"},
        )

    physiology = _get_mapping(
        parsed.get("registries/physiology_algorithms_v1.json")
    )
    audit.check(
        physiology.get("status") == "future_active"
        and physiology.get("input_profile")
        == "frailty3_peak_ppg_400_offline_v1"
        and physiology.get("sampling_rate_hz") == 400,
        "physiology::profile_binding",
        (
            f"status={physiology.get('status')!r}, "
            f"input={physiology.get('input_profile')!r}, "
            f"fs={physiology.get('sampling_rate_hz')!r}"
        ),
    )
    expected_phys = {
        ("peak", "minimum_observation_sec"): 8.0,
        ("peak", "minimum_distance_sec"): 0.30,
        ("peak", "overlap_merge_action"): "keep_highest_confidence_existing_peak",
        ("ppi", "minimum_sec"): 0.30,
        ("ppi", "maximum_sec"): 2.00,
        ("ppi", "delete_source_peak_on_invalid_ppi"): False,
        ("hr", "minimum_peaks"): 5,
        ("hr", "minimum_valid_ppi"): 4,
        ("prv", "time_domain_minimum_sec"): 60,
        ("prv", "time_domain_minimum_valid_ppi_coverage"): 0.8,
        ("prv", "frequency_exploratory_minimum_contiguous_sec"): 120,
        ("prv", "frequency_confirmatory_minimum_contiguous_sec"): 300,
        ("dual_channel", "detect_independently"): True,
        ("dual_channel", "selection"): "valid_status_first_then_higher_sqi",
        ("dual_channel", "tie_break"): "RED",
        ("dual_channel", "generate_consensus_peaks"): False,
    }
    for (section, key), expected_value in expected_phys.items():
        observed = _get_mapping(physiology.get(section)).get(key)
        audit.check(
            observed == expected_value,
            f"physiology::{section}::{key}",
            f"observed={observed!r}, expected={expected_value!r}",
        )

    # The registries/ copy is the sole machine authority. The evidence/ copy is
    # only the byte-hashed historical audit snapshot listed by M3_BUILD_REPORT.
    crosswalk_value = parsed.get(
        "registries/historical_preprocessing_crosswalk_v1.json"
    )
    if crosswalk_value is not None:
        crosswalk = _get_mapping(crosswalk_value)
        boundary = (
            "final_v0/M3_unified_preprocessing_and_signal_algorithms/"
            "src/m3_signal_core"
        )
        audit.check(
            crosswalk.get("future_active_boundary") == boundary,
            "future_active::crosswalk_boundary",
            f"observed={crosswalk.get('future_active_boundary')!r}",
        )
        entries = _get_sequence(crosswalk.get("entries"))
        entry_failures: list[str] = []
        for item in entries:
            if not isinstance(item, dict):
                entry_failures.append("non-object entry")
                continue
            source = REPO_ROOT / str(item.get("path", ""))
            if item.get("status") != "historical_reproduction_only":
                entry_failures.append(f"{item.get('path')}: status")
            if (
                item.get("future_active_replacement")
                != "m3_signal_core registry-bound facade"
            ):
                entry_failures.append(f"{item.get('path')}: replacement")
            if not source.is_file():
                entry_failures.append(f"{item.get('path')}: missing")
                continue
            if source.stat().st_size != item.get("bytes"):
                entry_failures.append(f"{item.get('path')}: bytes")
            if sha256_file(source) != item.get("sha256"):
                entry_failures.append(f"{item.get('path')}: sha256")
        audit.check(
            bool(entries)
            and not entry_failures
            and crosswalk.get("missing_source_count") == 0,
            "historical_crosswalk::authority_root_sources_and_hashes",
            "; ".join(entry_failures),
        )

    active_ids = {
        profile_id
        for profile_id, profile in profiles.items()
        if str(profile.get("status", "")).startswith("future_active")
    }
    root_bindings: list[str] = []
    for path in sorted(REPO_ROOT.glob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        hits = sorted(profile_id for profile_id in active_ids if profile_id in text)
        if hits or re.search(r"(?:from|import)\s+m3_signal_core\b", text):
            root_bindings.append(f"{path.name}:{','.join(hits) or 'module_import'}")
    audit.check(
        not root_bindings,
        "future_active::no_root_python_binding",
        "; ".join(root_bindings),
    )

    bindings = parsed.get("registries/module_bindings_v1.json")
    if bindings is not None:
        paths = _collect_path_fields(bindings)
        invalid_paths = [
            value
            for value in paths
            if not (
                value == "src/m3_signal_core"
                or value.startswith("src/m3_signal_core/")
                or value.startswith("m3_signal_core.")
            )
        ]
        audit.check(
            bool(paths) and not invalid_paths,
            "future_active::module_binding_paths_only_m3_src",
            f"paths={paths!r}, invalid={invalid_paths!r}",
        )
        referenced_profiles = set(_all_string_values(bindings)) & set(profiles)
        audit.check(
            referenced_profiles <= active_ids,
            "future_active::binding_profiles_registered",
            f"referenced={sorted(referenced_profiles)}, active={sorted(active_ids)}",
        )
    return active_ids


def _literal_strings(node: ast.AST | None, assignments: Mapping[str, set[str]]) -> set[str]:
    """提取参数中可证明的字符串 / Extract statically provable string options."""

    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return set(assignments.get(node.id, set()))
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _call_leaf_name(call: ast.Call) -> str:
    """返回调用目标末级名 / Return the leaf name of a call target."""

    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _keyword_argument(call: ast.Call, name: str) -> ast.AST | None:
    """按名取得调用参数 / Return a keyword argument by name."""

    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def collect_static_reason_codes() -> set[str]:
    """从源码 AST 提取实际 reason codes / Extract emitted reason codes from AST."""

    codes: set[str] = set(REQUIRED_RUNTIME_REASON_CODES)
    for path in sorted((PACKAGE_ROOT / "src" / "m3_signal_core").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assignments: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            target_name: str | None = None
            value: ast.AST | None = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                if isinstance(node.targets[0], ast.Name):
                    target_name = node.targets[0].id
                    value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_name = node.target.id
                value = node.value
            if target_name and (
                target_name == "code"
                or target_name.endswith("_code")
                or target_name in {"reasons", "reason_codes"}
            ):
                assignments.setdefault(target_name, set()).update(
                    _literal_strings(value, {})
                )

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                leaf = _call_leaf_name(node)
                if leaf == "QualityIssue":
                    argument = (
                        node.args[0]
                        if node.args
                        else _keyword_argument(node, "code")
                    )
                    codes.update(_literal_strings(argument, assignments))
                elif leaf == "PeakResult":
                    argument = (
                        node.args[9]
                        if len(node.args) > 9
                        else _keyword_argument(node, "reason_codes")
                    )
                    codes.update(_literal_strings(argument, assignments))
                elif leaf == "HrvResult":
                    argument = (
                        node.args[2]
                        if len(node.args) > 2
                        else _keyword_argument(node, "reason_codes")
                    )
                    codes.update(_literal_strings(argument, assignments))
                elif (
                    leaf == "append"
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {"reasons", "reason_codes"}
                    and node.args
                ):
                    codes.update(_literal_strings(node.args[0], assignments))
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value in {"reason_code", "reason_codes"}
                    ):
                        codes.update(_literal_strings(value, assignments))
    # 中文：空串与 profile provenance 均不能进入 reason registry。
    # English: Empty strings and profile provenance are not reason codes.
    return {
        code
        for code in codes
        if code
        and not code.endswith("_profile_v1")
        and not code.endswith("_offline_v1")
        and not code.endswith("_causal_v1")
    }


def validate_reason_code_subset(audit: Audit, parsed: Mapping[str, Any]) -> set[str]:
    """验证源码 reason code 是注册表子集 / Require source codes in registry."""

    registry = _get_mapping(parsed.get("registries/reason_codes_v1.json"))
    registered = set(_get_mapping(registry.get("codes")))
    emitted = collect_static_reason_codes()
    missing = sorted(emitted - registered)
    audit.check(
        not missing,
        "reason_codes::static_source_subset_of_registry",
        f"missing={missing}, emitted={sorted(emitted)}",
    )
    invalid = sorted(
        code for code in registered if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", code)
    )
    audit.check(
        not invalid,
        "reason_codes::identifier_format",
        f"invalid={invalid}",
    )
    return emitted


def validate_upstream_authorities(audit: Audit) -> dict[str, Any]:
    """验证 M1/M2 精确 ID 与哈希 / Verify exact M1/M2 IDs and digests."""

    paths = {
        "m1_verification": M1_ROOT / "M1_CONTRACT_VERIFICATION_V3_CURRENT.json",
        "m1_routing": (
            M1_ROOT / "registries_v3" / "quality_routing_registry_v3_active.json"
        ),
        "m2_build": M2_ROOT / "M2_BUILD_REPORT.json",
        "m2_verification": M2_ROOT / "M2_CONTRACT_VERIFICATION.json",
        "m2_fold": M2_ROOT / "splits" / "frailty3_future_corrected_sgkf5_v2.json",
        "m2_manifest": M2_ROOT / "manifests" / "frailty3_file_manifest.csv",
        "m2_subject_manifest": (
            M2_ROOT / "manifests" / "frailty3_subject_manifest.csv"
        ),
    }
    loaded: dict[str, Any] = {}
    for key in ("m1_verification", "m1_routing", "m2_build", "m2_verification", "m2_fold"):
        value = _load_json(audit, paths[key], rule=f"upstream_strict_json::{key}")
        if value is not None:
            loaded[key] = value
    for key, expected_name in (
        ("m1_verification", "M1_CONTRACT_VERIFICATION_V3_CURRENT.json"),
        ("m1_routing", "quality_routing_registry_v3_active.json"),
        ("m2_build", "M2_BUILD_REPORT.json"),
        ("m2_verification", "M2_CONTRACT_VERIFICATION.json"),
        ("m2_fold", "frailty3_future_corrected_sgkf5_v2.json"),
    ):
        path = paths[key]
        expected = UPSTREAM_FILE_SHA256[expected_name]
        observed = sha256_file(path) if path.is_file() else None
        audit.check(
            observed == expected,
            f"upstream_sha256::{key}",
            f"observed={observed}, expected={expected}",
        )
    manifest_hash = sha256_file(paths["m2_manifest"]) if paths["m2_manifest"].is_file() else None
    audit.check(
        manifest_hash == M2_MANIFEST_SHA256,
        "upstream_sha256::m2_file_manifest",
        f"observed={manifest_hash}, expected={M2_MANIFEST_SHA256}",
    )

    m1_verify = _get_mapping(loaded.get("m1_verification"))
    m1_routing = _get_mapping(loaded.get("m1_routing"))
    audit.check(
        m1_verify.get("status") == "pass"
        and m1_verify.get("preflight_status") == "pass"
        and m1_verify.get("contract_version") == "m1.architecture.v3"
        and m1_verify.get("routing_registry")
        == "registries_v3/quality_routing_registry_v3_active.json",
        "m1_authority::current_v3_pass",
        (
            f"status={m1_verify.get('status')!r}, "
            f"contract={m1_verify.get('contract_version')!r}, "
            f"routing={m1_verify.get('routing_registry')!r}"
        ),
    )
    audit.check(
        m1_routing.get("registry_version") == "m1.quality_routing.v3"
        and m1_routing.get("architecture_version") == "m1.architecture.v3"
        and m1_routing.get("sqi_required") is True
        and m1_routing.get("motion_detector_optional") is True,
        "m1_authority::sequential_sqi_motion_router",
        (
            f"registry={m1_routing.get('registry_version')!r}, "
            f"architecture={m1_routing.get('architecture_version')!r}"
        ),
    )

    m2_build = _get_mapping(loaded.get("m2_build"))
    m2_verify = _get_mapping(loaded.get("m2_verification"))
    frailty = _get_mapping(m2_build.get("frailty3"))
    folds = _get_mapping(m2_build.get("fold_registries"))
    audit.check(
        m2_build.get("status") == "pass"
        and m2_build.get("schema_version") == "m2.build_report.v1"
        and frailty.get("dataset_version_id") == DATASET_VERSION_ID
        and frailty.get("file_count") == 261
        and frailty.get("subject_count") == 29
        and frailty.get("total_data_rows") == 18152248,
        "m2_authority::build_report_dataset",
        f"frailty3={frailty!r}",
    )
    audit.check(
        folds.get("future_registry_id") == FOLD_REGISTRY_ID
        and folds.get("future_payload_sha256") == FOLD_PAYLOAD_SHA256
        and folds.get("future_class_missing_folds") == 0
        and folds.get("future_invariants_pass") is True
        and folds.get("seeds") == SEEDS,
        "m2_authority::future_fold_binding",
        f"fold_registries={folds!r}",
    )
    audit.check(
        m2_verify.get("status") == "pass"
        and m2_verify.get("failure_count") == 0
        and m2_verify.get("dataset_version_id") == DATASET_VERSION_ID
        and m2_verify.get("active_fold_registry_id") == FOLD_REGISTRY_ID
        and m2_verify.get("active_fold_registry_sha256") == FOLD_PAYLOAD_SHA256,
        "m2_authority::verification_pass",
        (
            f"status={m2_verify.get('status')!r}, "
            f"dataset={m2_verify.get('dataset_version_id')!r}, "
            f"fold={m2_verify.get('active_fold_registry_id')!r}"
        ),
    )
    loaded["paths"] = paths
    loaded["m2_manifest_sha256"] = manifest_hash
    return loaded


def validate_m2_fold_payload(audit: Audit, authority: Mapping[str, Any]) -> None:
    """复算并验证 M2 5×5 fold payload / Recompute and validate M2 fold payload."""

    registry = _get_mapping(authority.get("m2_fold"))
    if not registry:
        return
    payload = {key: value for key, value in registry.items() if key != "payload_sha256"}
    computed = hashlib.sha256(stable_json_bytes(payload)).hexdigest()
    audit.check(
        computed == registry.get("payload_sha256") == FOLD_PAYLOAD_SHA256,
        "m2_fold::canonical_payload_sha256",
        (
            f"computed={computed}, field={registry.get('payload_sha256')}, "
            f"expected={FOLD_PAYLOAD_SHA256}"
        ),
    )
    audit.check(
        registry.get("registry_id") == FOLD_REGISTRY_ID
        and registry.get("dataset_version_id") == DATASET_VERSION_ID
        and registry.get("semantic_parent")
        == "sklearn.model_selection.StratifiedGroupKFold"
        and registry.get("algorithm")
        == "corrected_sgkf_joint_group_count_permutation_then_greedy_balance"
        and registry.get("runtime_split_recomputation_allowed") is False
        and registry.get("all_candidates_must_rerun") is True
        and registry.get("invariants_pass") is True
        and registry.get("class_missing_fold_count") == 0,
        "m2_fold::frozen_protocol_semantics",
        "future SGKF-derived subject membership must remain materialized",
    )
    audit.check(
        registry.get("n_splits") == 5
        and registry.get("n_repeats") == 5
        and registry.get("n_subjects") == 29
        and registry.get("seeds") == SEEDS,
        "m2_fold::five_by_five_seed_contract",
        (
            f"splits={registry.get('n_splits')}, repeats={registry.get('n_repeats')}, "
            f"subjects={registry.get('n_subjects')}, seeds={registry.get('seeds')}"
        ),
    )
    errors: list[str] = []
    repeats = _get_sequence(registry.get("repeats"))
    all_subjects: set[str] | None = None
    for repeat_index, repeat_value in enumerate(repeats):
        repeat = _get_mapping(repeat_value)
        seed = SEEDS[repeat_index] if repeat_index < len(SEEDS) else None
        if repeat.get("repeat_index") != repeat_index or repeat.get("split_seed") != seed:
            errors.append(f"repeat{repeat_index}: index/seed")
        fold_values = _get_sequence(repeat.get("folds"))
        if len(fold_values) != 5:
            errors.append(f"repeat{repeat_index}: fold_count={len(fold_values)}")
        oof_counter: Counter[str] = Counter()
        for fold_index, fold_value in enumerate(fold_values):
            fold = _get_mapping(fold_value)
            train = set(map(str, _get_sequence(fold.get("train_subject_ids"))))
            oof = set(
                map(str, _get_sequence(fold.get("oof_validation_subject_ids")))
            )
            if train & oof:
                errors.append(f"repeat{repeat_index}/fold{fold_index}: overlap")
            union = train | oof
            if len(union) != 29:
                errors.append(
                    f"repeat{repeat_index}/fold{fold_index}: union={len(union)}"
                )
            if all_subjects is None:
                all_subjects = union
            elif union != all_subjects:
                errors.append(f"repeat{repeat_index}/fold{fold_index}: roster drift")
            if fold.get("all_three_classes_present") is not True:
                errors.append(f"repeat{repeat_index}/fold{fold_index}: missing class")
            oof_counts = set(_get_mapping(fold.get("oof_validation_class_counts")))
            train_counts = set(_get_mapping(fold.get("train_class_counts")))
            if oof_counts != CLASS_NAMES or train_counts != CLASS_NAMES:
                errors.append(f"repeat{repeat_index}/fold{fold_index}: class keys")
            if fold.get("training_seed") != int(seed or 0) + fold_index:
                errors.append(f"repeat{repeat_index}/fold{fold_index}: training seed")
            oof_counter.update(oof)
        if all_subjects is not None and oof_counter != Counter(
            {subject: 1 for subject in all_subjects}
        ):
            errors.append(f"repeat{repeat_index}: OOF not exact partition")
    audit.check(
        len(repeats) == 5 and not errors,
        "m2_fold::materialized_membership_invariants",
        "; ".join(errors),
    )

    fold_source = PACKAGE_ROOT / "src" / "m3_signal_core" / "fold_contract.py"
    fold_text = fold_source.read_text(encoding="utf-8") if fold_source.is_file() else ""
    required_fragments = (
        "frailty3_future_corrected_sgkf5_v2.json",
        'registry["registry_id"] != "frailty3_future_corrected_sgkf5_v2"',
        '"fold_registry_payload_sha256": registry["payload_sha256"]',
    )
    audit.check(
        all(fragment in fold_text for fragment in required_fragments),
        "m2_fold::m3_source_exact_binding",
        (
            "fold_contract.py must load the fixed registry, reject another ID, "
            "and propagate its verified payload digest"
        ),
    )


def validate_fixture_and_build_hashes(
    audit: Audit, parsed: Mapping[str, Any]
) -> None:
    """验证 fixture/evidence 内容哈希 / Validate fixture and evidence hashes."""

    manifest = _get_mapping(parsed.get("fixtures/reference_fixture_manifest.json"))
    audit.check(
        manifest.get("fixture_manifest_id") == "m3_reference_fixtures_v1"
        and manifest.get("status") == "deterministic_synthetic_truth"
        and manifest.get("sampling_rate_hz") == 400.0
        and manifest.get("seed") == 20260815,
        "fixtures::manifest_identity",
        f"manifest={manifest!r}",
    )
    fixture_errors: list[str] = []
    for item_value in _get_sequence(manifest.get("files")):
        item = _get_mapping(item_value)
        path = PACKAGE_ROOT / "fixtures" / str(item.get("file", ""))
        if not path.is_file():
            fixture_errors.append(f"{item.get('file')}: missing")
            continue
        if sha256_file(path) != item.get("sha256"):
            fixture_errors.append(f"{item.get('file')}: sha256")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            fixture_errors.append(f"{item.get('file')}: load={exc}")
            continue
        if list(array.shape) != item.get("shape"):
            fixture_errors.append(f"{item.get('file')}: shape={array.shape}")
        if str(array.dtype) != item.get("dtype"):
            fixture_errors.append(f"{item.get('file')}: dtype={array.dtype}")
    audit.check(
        len(_get_sequence(manifest.get("files"))) == 3 and not fixture_errors,
        "fixtures::file_hash_shape_dtype",
        "; ".join(fixture_errors),
    )

    build = _get_mapping(parsed.get("M3_BUILD_REPORT.json"))
    audit.check(
        build.get("schema_version") == "m3.build_report.v1"
        and build.get("report_id") == "m3_complete_evidence_build_v1"
        and build.get("status") == "pass",
        "evidence::build_report_identity",
        f"build={build!r}",
    )
    output_errors: list[str] = []
    output_paths: set[str] = set()
    for item_value in _get_sequence(build.get("outputs")):
        item = _get_mapping(item_value)
        relative = str(item.get("path", ""))
        output_paths.add(relative)
        path = PACKAGE_ROOT / relative
        if not path.is_file():
            output_errors.append(f"{relative}: missing")
            continue
        if path.stat().st_size != item.get("bytes"):
            output_errors.append(f"{relative}: bytes")
        if sha256_file(path) != item.get("sha256"):
            output_errors.append(f"{relative}: sha256")
    expected_outputs = {
        "evidence/ekf_lpf_frailty3_role_proxy.json",
        "evidence/ekf_lpf_synthetic_comparison.json",
        "evidence/filter_response_comparison.json",
        "evidence/frailty3_signal_integrity_summary.json",
        "evidence/historical_preprocessing_crosswalk_v1.json",
        "evidence/legacy_peak_parity.json",
    }
    audit.check(
        output_paths == expected_outputs and not output_errors,
        "evidence::build_output_hashes",
        f"paths={sorted(output_paths)}, errors={output_errors}",
    )

    legacy = _get_mapping(parsed.get("evidence/legacy_peak_parity.json"))
    classifier = _get_mapping(legacy.get("classifier_alias_parity"))
    duplicate = _get_mapping(legacy.get("funcs_ppg_duplicate_parity"))
    cross = _get_mapping(legacy.get("cross_implementation_comparison"))
    source_hashes = _get_mapping(legacy.get("source_sha256"))
    source_paths = {
        "classifier": REPO_ROOT / "frailty_3class_classifier.py",
        "funcs": REPO_ROOT / "funcs.py",
        "ppg": REPO_ROOT / "ppg.py",
    }
    actual_source_hashes = {
        key: sha256_file(path) if path.is_file() else None
        for key, path in source_paths.items()
    }
    ppg_fixture = PACKAGE_ROOT / "fixtures" / "ppg_reference_v1.npy"
    audit.check(
        legacy.get("evidence_id") == "m3_legacy_peak_same_input_parity_v1"
        and legacy.get("status")
        == "pass_with_expected_cross_implementation_difference"
        and legacy.get("sampling_rate_hz") == 400
        and legacy.get("fixture_sha256")
        == (sha256_file(ppg_fixture) if ppg_fixture.is_file() else None),
        "legacy_parity::identity_fixture_status",
        f"legacy={legacy!r}",
    )
    audit.check(
        classifier.get("exact") is True
        and classifier.get("peak_count") == 35
        and duplicate.get("exact") is True
        and duplicate.get("peak_count") == 36
        and cross.get("exact") is False
        and cross.get("classifier_only_indices") == []
        and cross.get("funcs_ppg_only_indices") == [8318]
        and cross.get("matched_exact_index_count") == 35,
        "legacy_parity::expected_same_and_cross_implementation_results",
        (
            f"classifier={classifier!r}, duplicate={duplicate!r}, "
            f"cross={cross!r}"
        ),
    )
    audit.check(
        source_hashes == actual_source_hashes,
        "legacy_parity::historical_source_hashes",
        f"reported={source_hashes!r}, actual={actual_source_hashes!r}",
    )


def _build_reference_input_snapshot() -> tuple[dict[str, str], str]:
    """复算正式测试输入快照 / Recompute the formal test input snapshot."""

    patterns = (
        "src/**/*.py",
        "tests/test_*.py",
        "registries/*.json",
        "schemas/*.json",
        "fixtures/*",
    )
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in PACKAGE_ROOT.glob(pattern)
            if path.is_file()
        },
        key=lambda path: path.relative_to(PACKAGE_ROOT).as_posix(),
    )
    digests = {
        path.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(path) for path in paths
    }
    snapshot = hashlib.sha256(compact_snapshot_bytes(digests)).hexdigest()
    return digests, snapshot


def validate_formal_test_report(
    audit: Audit, parsed: Mapping[str, Any], ast_test_count: int
) -> dict[str, Any]:
    """验证不少于 38 项的当前快照报告 / Validate the current >=38-test report."""

    report = _get_mapping(parsed.get("M3_REFERENCE_TEST_RESULTS.json"))
    input_hashes, snapshot_sha = _build_reference_input_snapshot()
    test_hashes = {
        path.name: sha256_file(path)
        for path in sorted((PACKAGE_ROOT / "tests").glob("test_*.py"))
    }
    count = report.get("tests_run")
    audit.check(
        report.get("schema_version") == "m3.reference_test_report.v2"
        and report.get("report_id") == "m3_reference_tests_v1",
        "formal_tests::report_identity_v2",
        (
            f"schema={report.get('schema_version')!r}, "
            f"id={report.get('report_id')!r}"
        ),
    )
    audit.check(
        isinstance(count, int)
        and count >= MIN_FORMAL_TESTS
        and count == ast_test_count,
        "formal_tests::report_count_matches_current_ast",
        f"report={count!r}, ast={ast_test_count}, minimum={MIN_FORMAL_TESTS}",
    )
    audit.check(
        report.get("status") == "pass"
        and report.get("failure_count") == 0
        and report.get("error_count") == 0
        and report.get("skipped_count") == 0
        and report.get("failures") == []
        and report.get("errors") == [],
        "formal_tests::zero_failure_error_skip",
        (
            f"status={report.get('status')!r}, failures={report.get('failure_count')!r}, "
            f"errors={report.get('error_count')!r}, skipped={report.get('skipped_count')!r}"
        ),
    )
    audit.check(
        report.get("test_source_sha256") == test_hashes,
        "formal_tests::test_source_hashes_current",
        f"reported={report.get('test_source_sha256')!r}, current={test_hashes!r}",
    )
    audit.check(
        report.get("input_file_sha256") == input_hashes
        and report.get("input_snapshot_sha256") == snapshot_sha,
        "formal_tests::full_input_snapshot_current",
        (
            f"reported_snapshot={report.get('input_snapshot_sha256')!r}, "
            f"current_snapshot={snapshot_sha}"
        ),
    )
    return {
        "minimum_tests": MIN_FORMAL_TESTS,
        "ast_test_count": ast_test_count,
        "reported_tests": count,
        "input_snapshot_sha256": snapshot_sha,
    }


def validate_synthetic_ekf_gate(audit: Audit, parsed: Mapping[str, Any]) -> None:
    """验证合成真值 EKF 主门和 LPF 对照 / Validate synthetic EKF/LPF gate."""

    evidence = _get_mapping(
        parsed.get("evidence/ekf_lpf_synthetic_comparison.json")
    )
    routes = _get_mapping(evidence.get("route_metrics"))
    ekf = _get_mapping(routes.get("ekf"))
    lpf = _get_mapping(routes.get("lpf_0p3"))
    fixture_path = PACKAGE_ROOT / "fixtures" / "imu_reference_v1.npy"
    fixture_sha = sha256_file(fixture_path) if fixture_path.is_file() else None
    audit.check(
        evidence.get("evidence_id") == "m3_ekf_lpf_synthetic_truth_v1"
        and evidence.get("engineering_gate_status") == "pass"
        and evidence.get("fixture_sha256") == fixture_sha,
        "ekf_gate::evidence_identity_and_fixture",
        (
            f"id={evidence.get('evidence_id')!r}, "
            f"status={evidence.get('engineering_gate_status')!r}, "
            f"fixture={evidence.get('fixture_sha256')!r}/{fixture_sha!r}"
        ),
    )
    ekf_numbers = [
        ekf.get("coverage_fraction"),
        ekf.get("dynamic_acceleration_vector_rmse_mps2"),
        ekf.get("gravity_angle_p95_deg"),
        ekf.get("gravity_vector_rmse_mps2"),
    ]
    lpf_numbers = [
        lpf.get("coverage_fraction"),
        lpf.get("dynamic_acceleration_vector_rmse_mps2"),
        lpf.get("gravity_angle_p95_deg"),
        lpf.get("gravity_vector_rmse_mps2"),
    ]
    finite = all(_is_finite_number(value) for value in ekf_numbers + lpf_numbers)
    audit.check(
        finite,
        "ekf_gate::finite_route_metrics",
        f"ekf={ekf_numbers!r}, lpf={lpf_numbers!r}",
    )
    if finite:
        audit.check(
            float(ekf["coverage_fraction"]) >= 0.95
            and float(ekf["dynamic_acceleration_vector_rmse_mps2"]) <= 0.25
            and float(ekf["gravity_angle_p95_deg"]) <= 5.0
            and float(ekf["gravity_vector_rmse_mps2"]) <= 0.25,
            "ekf_gate::primary_thresholds",
            f"ekf={ekf!r}",
        )
        audit.check(
            float(ekf["dynamic_acceleration_vector_rmse_mps2"])
            < float(lpf["dynamic_acceleration_vector_rmse_mps2"])
            and float(ekf["gravity_angle_p95_deg"])
            < float(lpf["gravity_angle_p95_deg"])
            and float(ekf["gravity_vector_rmse_mps2"])
            < float(lpf["gravity_vector_rmse_mps2"]),
            "ekf_gate::ekf_outperforms_lpf_on_synthetic_truth",
            f"ekf={ekf!r}, lpf={lpf!r}",
        )
    audit.check(
        ekf.get("profile_id") == "imu_ekf_si_400_causal_v1"
        and lpf.get("profile_id") == "imu_lpf_si_400_causal_v1"
        and ekf.get("silent_fallback") is False
        and lpf.get("silent_fallback") is False
        and ekf.get("terminal_state") == "tracking"
        and lpf.get("terminal_state") == "tracking",
        "ekf_gate::routes_separate_no_fallback",
        f"ekf={ekf!r}, lpf={lpf!r}",
    )


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 UTF-8 CSV / Read a UTF-8 CSV as dictionaries."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_frailty261_gate(
    audit: Audit, parsed: Mapping[str, Any], authority: Mapping[str, Any]
) -> None:
    """验证 Frailty3 261 条代理完整性 / Validate all 261 Frailty3 proxies."""

    integrity = _get_mapping(
        parsed.get("evidence/frailty3_signal_integrity_summary.json")
    )
    expected_role_counts = {role: 29 for role in ROLES}
    audit.check(
        integrity.get("evidence_id") == "m3_frailty3_signal_integrity_binding_v1"
        and integrity.get("dataset_version_id") == DATASET_VERSION_ID
        and integrity.get("m2_manifest_sha256") == M2_MANIFEST_SHA256
        and integrity.get("file_count") == 261
        and integrity.get("subject_count") == 29
        and integrity.get("numeric_row_count") == 18152248
        and integrity.get("all_files_passed_finite_8_columns") is True
        and integrity.get("role_counts") == expected_role_counts,
        "frailty261::integrity_binding",
        f"integrity={integrity!r}",
    )

    proxy = _get_mapping(parsed.get("evidence/ekf_lpf_frailty3_role_proxy.json"))
    records = _get_sequence(proxy.get("records"))
    audit.check(
        proxy.get("evidence_id") == "m3_ekf_lpf_frailty3_first6s_proxy_v1"
        and proxy.get("dataset_version_id") == DATASET_VERSION_ID
        and proxy.get("m2_manifest_sha256") == M2_MANIFEST_SHA256
        and proxy.get("record_count") == 261
        and proxy.get("subject_count") == 29
        and len(records) == 261
        and proxy.get("segment_definition") == "first_6_seconds_no_padding_each_record",
        "frailty261::proxy_identity_and_count",
        (
            f"id={proxy.get('evidence_id')!r}, records={len(records)}, "
            f"declared={proxy.get('record_count')!r}"
        ),
    )
    manifest_path = _get_mapping(authority.get("paths")).get("m2_manifest")
    manifest_rows = (
        _load_csv_rows(manifest_path)
        if isinstance(manifest_path, Path) and manifest_path.is_file()
        else []
    )
    manifest_by_id = {row["file_id"]: row for row in manifest_rows}
    record_errors: list[str] = []
    file_ids: list[str] = []
    subject_ids: set[str] = set()
    role_counter: Counter[str] = Counter()
    for item_value in records:
        item = _get_mapping(item_value)
        file_id = str(item.get("file_id", ""))
        file_ids.append(file_id)
        subject_ids.add(str(item.get("subject_id", "")))
        role_counter[str(item.get("role", ""))] += 1
        manifest_row = manifest_by_id.get(file_id)
        if manifest_row is None:
            record_errors.append(f"{file_id}: absent from M2")
        else:
            if item.get("subject_id") != manifest_row.get("subject_id"):
                record_errors.append(f"{file_id}: subject")
            if item.get("role") != manifest_row.get("role"):
                record_errors.append(f"{file_id}: role")
            if item.get("source_sha256_from_m2") != manifest_row.get("sha256"):
                record_errors.append(f"{file_id}: source sha")
        samples = item.get("samples_read")
        if not isinstance(samples, int) or not 0 < samples <= 2400:
            record_errors.append(f"{file_id}: samples={samples!r}")
        routes = _get_mapping(item.get("routes"))
        if set(routes) != {"ekf", "lpf_0p3"}:
            record_errors.append(f"{file_id}: routes={sorted(routes)}")
            continue
        for route_name in ("ekf", "lpf_0p3"):
            route = _get_mapping(routes.get(route_name))
            numeric = [
                route.get("coverage_fraction"),
                route.get("dynamic_acceleration_rms_mps2"),
                route.get("gravity_norm_abs_error_median_mps2"),
                route.get("gravity_norm_median_mps2"),
            ]
            if not all(_is_finite_number(value) for value in numeric):
                record_errors.append(f"{file_id}/{route_name}: nonfinite")
            elif not 0.0 <= float(route["coverage_fraction"]) <= 1.0:
                record_errors.append(f"{file_id}/{route_name}: coverage")
            if route.get("terminal_state") not in {"tracking", "prediction_only"}:
                record_errors.append(
                    f"{file_id}/{route_name}: terminal={route.get('terminal_state')!r}"
                )
    audit.check(
        len(set(file_ids)) == 261
        and len(subject_ids) == 29
        and role_counter == Counter(expected_role_counts)
        and not record_errors,
        "frailty261::record_manifest_route_completeness",
        (
            f"unique_files={len(set(file_ids))}, subjects={len(subject_ids)}, "
            f"roles={dict(role_counter)}, errors={record_errors[:20]}"
        ),
    )
    summaries = _get_mapping(proxy.get("summary_by_role_family_and_route"))
    expected_summary_counts = {
        "B:ekf": 29,
        "B:lpf_0p3": 29,
        "R:ekf": 116,
        "R:lpf_0p3": 116,
        "S:ekf": 58,
        "S:lpf_0p3": 58,
        "W:ekf": 58,
        "W:lpf_0p3": 58,
    }
    summary_errors: list[str] = []
    for key, expected_count in expected_summary_counts.items():
        item = _get_mapping(summaries.get(key))
        if item.get("record_count") != expected_count:
            summary_errors.append(f"{key}: count={item.get('record_count')!r}")
        if item.get("record_with_any_valid_count") != expected_count:
            summary_errors.append(f"{key}: valid={item.get('record_with_any_valid_count')!r}")
        if item.get("terminal_no_estimate_count") != 0:
            summary_errors.append(f"{key}: no_estimate")
    audit.check(
        set(summaries) == set(expected_summary_counts) and not summary_errors,
        "frailty261::summary_route_coverage",
        f"errors={summary_errors}, keys={sorted(summaries)}",
    )


def validate_examples(audit: Audit, parsed: Mapping[str, Any]) -> None:
    """验证 M1/M2 示例绑定关键 ID / Validate authority IDs in examples."""

    offline = parsed.get("examples/m1_pipeline_config_m3_offline.json")
    mobile = parsed.get("examples/m1_pipeline_config_m3_mobile.json")
    provenance = parsed.get("examples/m2_result_provenance_m3_bound.json")
    if offline is not None:
        strings = set(_all_string_values(offline))
        audit.check(
            "frailty3_static_ppg_400_offline_v1" in strings
            and "imu_ekf_si_400_causal_v1" in strings,
            "examples::offline_profiles_bound",
            f"strings={sorted(strings)}",
        )
    if mobile is not None:
        strings = set(_all_string_values(mobile))
        audit.check(
            "mobile_ppg_400_causal_v1" in strings
            and "imu_ekf_si_400_causal_v1" in strings,
            "examples::mobile_profiles_bound",
            f"strings={sorted(strings)}",
        )
    if provenance is not None:
        strings = set(_all_string_values(provenance))
        audit.check(
            DATASET_VERSION_ID in strings
            and FOLD_REGISTRY_ID in strings
            and FOLD_PAYLOAD_SHA256 in strings,
            "examples::m2_provenance_bound",
            f"required IDs absent from strings={sorted(strings)}",
        )


def _status(audit: Audit) -> str:
    """按 fail > wait > pass 计算状态 / Compute fail-over-wait-over-pass status."""

    if audit.failures:
        return "fail"
    if audit.waiting:
        return "waiting"
    return "pass"


def build_report() -> dict[str, Any]:
    """执行完整 M3 验证并返回报告 / Run the full M3 verification."""

    audit = Audit()
    parsed = validate_required_files_and_json(audit)
    ast_test_count = validate_python_ast_and_comments(audit)
    validate_schema_and_registry_ids(audit, parsed)
    active_profiles = validate_profiles_and_bindings(audit, parsed)
    emitted_codes = validate_reason_code_subset(audit, parsed)
    authority = validate_upstream_authorities(audit)
    validate_m2_fold_payload(audit, authority)
    validate_fixture_and_build_hashes(audit, parsed)
    test_summary = validate_formal_test_report(audit, parsed, ast_test_count)
    validate_synthetic_ekf_gate(audit, parsed)
    validate_frailty261_gate(audit, parsed, authority)
    validate_examples(audit, parsed)
    status = _status(audit)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": REPORT_ID,
        "validator_revision": VALIDATOR_REVISION,
        "status": status,
        "check_count": len(audit.checks),
        "pass_count": sum(item["status"] == "pass" for item in audit.checks),
        "failure_count": len(audit.failures),
        "waiting_count": len(audit.waiting),
        "failures": audit.failures,
        "waiting": audit.waiting,
        "authority_bindings": {
            "m1_contract_version": "m1.architecture.v3",
            "m1_verification_sha256": UPSTREAM_FILE_SHA256[
                "M1_CONTRACT_VERIFICATION_V3_CURRENT.json"
            ],
            "m1_routing_registry_sha256": UPSTREAM_FILE_SHA256[
                "quality_routing_registry_v3_active.json"
            ],
            "m2_dataset_version_id": DATASET_VERSION_ID,
            "m2_manifest_sha256": M2_MANIFEST_SHA256,
            "m2_fold_registry_id": FOLD_REGISTRY_ID,
            "m2_fold_payload_sha256": FOLD_PAYLOAD_SHA256,
        },
        "formal_test_gate": test_summary,
        "future_active_profile_ids": sorted(active_profiles),
        "static_emitted_reason_codes": sorted(emitted_codes),
        "validator_sha256": sha256_file(Path(__file__)),
        "checks": audit.checks,
    }


def _description(relative: str) -> str:
    """为包树生成文件职责描述 / Generate a package-tree file description."""

    exact = {
        "README.md": "M3 package entry point, scope, commands, and authority boundary / M3 包入口、范围、命令与权威边界。",
        "M3_BUILD_REPORT.json": "Machine build output manifest with byte-exact evidence hashes / 带逐字节 evidence 哈希的构建清单。",
        "M3_REFERENCE_TEST_RESULTS.json": "Formal unittest report bound to the complete implementation snapshot / 绑定完整实现快照的正式测试报告。",
        "M3_CONTRACT_VERIFICATION.json": "Strict machine-readable M3 acceptance result / 严格机器可读 M3 验收结果。",
        "M3_PACKAGE_TREE.md": "Auto-refreshed package tree and file responsibility index / 自动更新的包树与文件职责索引。",
        "registries/preprocessing_profiles_v1.json": "Frozen 400 Hz PPG/IMU preprocessing profiles and legacy reproduction profiles / 冻结 400 Hz PPG/IMU 预处理与历史复现 profile。",
        "registries/physiology_algorithms_v1.json": "Peak, PPI, HR, PRV, and dual-wavelength selection semantics / Peak、PPI、HR、PRV 与双波长选择语义。",
        "registries/reason_codes_v1.json": "Controlled runtime reason-code vocabulary / 受控运行时原因码词表。",
        "registries/module_bindings_v1.json": "Only future-active module/profile implementation bindings / 唯一未来活动模块/profile 实现绑定。",
        "registries/feature_schemas_v1.json": "Versioned feature and raw-context field contracts / 版本化 feature 与 raw-context 字段合同。",
        "registries/status_mapping_v1.json": "M3-to-M1 status and terminal semantics / M3 到 M1 状态及终止语义映射。",
        "evidence/ekf_lpf_synthetic_comparison.json": "Synthetic-truth EKF primary gate and LPF comparator / 合成真值 EKF 主门与 LPF 对照。",
        "evidence/ekf_lpf_frailty3_role_proxy.json": "Paired first-six-second EKF/LPF proxies for all 261 Frailty3 records / 261 条 Frailty3 记录的配对前六秒 EKF/LPF 代理。",
        "evidence/frailty3_signal_integrity_summary.json": "Binding to M2 full-byte/full-numeric integrity evidence / 对 M2 全字节/全数值完整性证据的绑定。",
        "evidence/filter_response_comparison.json": "Frozen SOS coefficients and causal/offline response anchors / 冻结 SOS 系数与因果/离线响应锚点。",
        "evidence/historical_preprocessing_crosswalk_v1.json": "Read-only historical script hashes and replacement boundary / 只读历史脚本哈希与替换边界。",
        "fixtures/reference_fixture_manifest.json": "Deterministic fixture provenance, shape, dtype, and hashes / 确定性 fixture 来源、形状、类型与哈希。",
        "tools/validate_m3_contracts.py": "Fail-closed M3 validator and package-tree generator / fail-closed M3 验证器与包树生成器。",
        "tools/run_m3_reference_tests.py": "Formal test runner and full-input snapshot reporter / 正式测试 runner 与全输入快照报告器。",
    }
    if relative in exact:
        return exact[relative]
    if relative.startswith("src/m3_signal_core/"):
        return "Future-active registry-bound signal implementation / 未来活动、注册表绑定的信号实现。"
    if relative.startswith("tests/"):
        return "Deterministic contract/reference regression tests / 确定性合同与参考回归测试。"
    if relative.startswith("schemas/"):
        return "Strict JSON Schema for an M3 interchange artifact / M3 交换产物的严格 JSON Schema。"
    if relative.startswith("examples/"):
        return "Authority-bound M1/M2 integration example / 绑定权威 ID 的 M1/M2 集成示例。"
    if relative.startswith("fixtures/"):
        return "Deterministic synthetic binary test fixture / 确定性合成二进制测试 fixture。"
    if relative.startswith("tools/"):
        return "Reproducible M3 build/evaluation utility / 可复现 M3 构建或评价工具。"
    if relative.startswith("docs/"):
        return "M3 algorithm, contract, validation, or handoff documentation / M3 算法、合同、验证或交接文档。"
    if relative.startswith("algorithm_diagrams/"):
        return "Source diagram or rendered algorithm-flow artifact / 算法流程源图或渲染产物。"
    return "M3 package artifact / M3 包产物。"


def _tree_lines(paths: Sequence[Path]) -> list[str]:
    """渲染紧凑目录树 / Render a compact directory tree."""

    root: dict[str, Any] = {}
    for path in paths:
        parts = path.relative_to(PACKAGE_ROOT).parts
        cursor = root
        for part in parts:
            cursor = cursor.setdefault(part, {})

    lines = [PACKAGE_ROOT.name + "/"]

    def visit(node: Mapping[str, Any], prefix: str) -> None:
        """递归渲染一层 / Recursively render one tree level."""

        items = sorted(node.items(), key=lambda item: (bool(item[1]) is False, item[0]))
        for index, (name, children) in enumerate(items):
            last = index == len(items) - 1
            branch = "└── " if last else "├── "
            suffix = "/" if children else ""
            lines.append(prefix + branch + name + suffix)
            if children:
                visit(children, prefix + ("    " if last else "│   "))

    visit(root, "")
    return lines


def render_package_tree() -> str:
    """生成完整包树和逐文件说明 / Render the full tree and file descriptions."""

    paths = sorted(
        (
            path
            for path in PACKAGE_ROOT.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".tmp"}
        ),
        key=lambda item: item.relative_to(PACKAGE_ROOT).as_posix(),
    )
    if TREE_PATH not in paths:
        paths.append(TREE_PATH)
        paths.sort(key=lambda item: item.relative_to(PACKAGE_ROOT).as_posix())
    lines = [
        "# M3 package tree / M3 包树",
        "",
        "> Auto-generated by `tools/validate_m3_contracts.py --write-report`. ",
        "> 由 `tools/validate_m3_contracts.py --write-report` 自动生成；请勿手工维护。",
        "",
        "## Tree / 树状结构",
        "",
        "```text",
        *_tree_lines(paths),
        "```",
        "",
        "## File responsibilities and integrity / 文件职责与完整性",
        "",
        "| Path / 路径 | Bytes / 字节 | SHA-256 | Detailed responsibility / 详细职责 |",
        "|---|---:|---|---|",
    ]
    for path in paths:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        if path == TREE_PATH:
            size = "generated"
            digest = "self-reference omitted"
        elif path.is_file():
            size = str(path.stat().st_size)
            digest = sha256_file(path)
        else:
            size = "pending"
            digest = "pending"
        description = _description(relative).replace("|", "\\|")
        lines.append(f"| `{relative}` | {size} | `{digest}` | {description} |")
    lines.extend(
        [
            "",
            "## Update rule / 更新规则",
            "",
            "中文：任何 M3 写入批次完成后，先运行正式测试（若输入快照发生变化），",
            "再运行本验证器的 `--write-report`，最后运行 final_v0 三项 tracking 同步。",
            "本文件省略自身哈希以避免递归自引用；其余文件使用逐字节 SHA-256。",
            "",
            "English: After each M3 write batch, rerun the formal tests whenever the",
            "input snapshot changed, then run this validator with `--write-report`, and",
            "finally run the three final_v0 tracking synchronizers.  This file omits its",
            "own digest to avoid recursive self-reference; all other digests are byte exact.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """运行验证并可写报告/包树 / Run validation and optionally write artifacts."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="write M3_CONTRACT_VERIFICATION.json and M3_PACKAGE_TREE.md",
    )
    args = parser.parse_args()
    report = build_report()
    if args.write_report:
        atomic_write(REPORT_PATH, stable_json_bytes(report))
        # 中文：报告先落盘，包树才能记录其最终哈希；包树自身哈希明确省略。
        # English: Write the report first so the tree records its final digest.
        atomic_write(TREE_PATH, render_package_tree().encode("utf-8"))
        # 中文：生成后立即用 strict loader 复验，避免写出不可解析报告。
        # English: Reparse immediately so a generated non-strict report cannot survive.
        strict_load_json(REPORT_PATH)
    summary = {
        "status": report["status"],
        "check_count": report["check_count"],
        "pass_count": report["pass_count"],
        "failure_count": report["failure_count"],
        "waiting_count": report["waiting_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

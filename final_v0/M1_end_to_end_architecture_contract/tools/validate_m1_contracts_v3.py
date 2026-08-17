#!/usr/bin/env python3
"""验证 M1 V3 顺序质量路由合同；validate M1 V3 sequential routing contracts.

中文
----
默认模式只读检查 V3 schema、quality-routing registry、三档配置，以及继续沿用的
V2 input/platform/feature/classifier 合同。`--write-report` 只在本 M1 包内
原子写入 V3 验证报告和完整性树。工具不导入根目录训练代码、不安装依赖、不运行
模型、不联网，也不写入 `_agent`。

English
-------
The default mode read-only checks the V3 schemas, routing registry, three platform
configs, and the V2 input/platform/feature/classifier contracts retained by V3.
`--write-report` atomically writes only the V3 report and integrity tree inside
this M1 package. It imports no root training code, installs nothing, runs no model,
accesses no network, and never writes to `_agent`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


# 中文：所有路径相对脚本定位，调用者 cwd 无法把输出重定向到 final_v0 外。
# English: Resolve paths from this file so the caller's cwd cannot redirect writes.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PACKAGE_ROOT / "M1_CONTRACT_VERIFICATION_V3.json"
TREE_PATH = PACKAGE_ROOT / "M1_PACKAGE_TREE_V3.md"
CONTRACT_VERSION = "m1.architecture.v3"
CANONICAL_CHANNELS = ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"]
STATE_PRECEDENCE = ["invalid", "unrecoverable_quality", "motion", "low_quality", "high_quality"]
ALLOWED_DEPENDENCIES = {"numpy", "scipy", "onnxruntime", "scikit-learn"}
ALLOWED_MANUAL_POLICIES = {"drop", "denoise_then_extract_features"}
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
FORBIDDEN_MACHINE_TERMS = {
    "action_owner",
    "SQI_WEIGHT",
    "COARSE_REPLACE",
    "diagnostic_candidates_may_run_in_parallel",
}

SCHEMA_FILES = (
    "schemas_v2/signal_input_v2.schema.json",
    "schemas_v3/pipeline_config_v3.schema.json",
    "schemas_v3/inference_output_v3.schema.json",
)
REGISTRY_FILES = (
    "registries_v2/platform_profiles_v2.json",
    "registries_v3/quality_routing_registry_v3.json",
    "registries_v2/feature_extractor_registry_v2.json",
    "registries_v2/classifier_registry_v2.json",
)
EXAMPLE_FILES = (
    "examples_v3/pipeline_high_performance_x86_v3.json",
    "examples_v3/pipeline_accelerated_arm64_v3.json",
    "examples_v3/pipeline_value_arm64_v3.json",
)
REQUIRED_DOCS = (
    "00_CURRENT_STATUS_V3.md",
    "06_SEQUENTIAL_SQI_MOTION_ROUTING_V3.md",
)
TOOL_FILES = (
    "tools/validate_m1_contracts_v3.py",
    "tools/validate_m1_v3_routing_invariants.py",
)
CURRENT_FILES = (*REQUIRED_DOCS, *SCHEMA_FILES, *REGISTRY_FILES, *EXAMPLE_FILES, *TOOL_FILES)


def checked_relative(path: Path) -> str:
    """验证路径位于 M1 包内；ensure a path remains inside the M1 package."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PACKAGE_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Path escapes M1 package: {resolved}") from exc


def load_json(relative_path: str) -> Any:
    """严格按 UTF-8 JSON 读取；load a strict UTF-8 JSON document."""

    path = PACKAGE_ROOT / relative_path
    checked_relative(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(payload: bytes) -> str:
    """计算稳定 SHA-256；compute a stable SHA-256 digest."""

    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    """仅在 M1 包内原子写入；atomically write only inside the M1 package."""

    checked_relative(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    """写确定性、禁止 NaN 的 JSON；write deterministic JSON with NaN forbidden."""

    rendered = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write(path, rendered.encode("utf-8"))


def atomic_write_text(path: Path, rendered: str) -> None:
    """写 UTF-8/LF 文本；write UTF-8 text with LF line endings."""

    atomic_write(path, rendered.replace("\r\n", "\n").encode("utf-8"))


def add_failure(
    failures: list[dict[str, str]],
    file: str,
    rule: str,
    detail: object,
) -> None:
    """追加结构化失败；append one structured failure."""

    failures.append({"file": file, "rule": rule, "detail": str(detail)})


def index_by_id(
    records: Iterable[dict[str, Any]],
    file: str,
    failures: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """按唯一 ID 建索引；build a unique registry index by ID."""

    index: dict[str, dict[str, Any]] = {}
    for record in records:
        item_id = record.get("id")
        if not isinstance(item_id, str) or not item_id:
            add_failure(failures, file, "missing_id", record)
        elif item_id in index:
            add_failure(failures, file, "duplicate_id", item_id)
        else:
            index[item_id] = record
    return index


def validate_bundle_relative_path(value: object) -> list[str]:
    """检查 bundle 内 POSIX 相对路径；validate a bundle-internal POSIX path."""

    if not isinstance(value, str) or not value:
        return ["path_empty_or_not_string"]
    # 中文：Linux Path 会把部分 Windows 路径当普通字符，因此显式拒绝 drive/backslash。
    # English: Linux Path may treat Windows syntax as text, so reject drives/backslashes.
    if value.startswith("/") or value.startswith("\\") or "\\" in value or DRIVE_PATTERN.match(value):
        return ["path_not_posix_relative"]
    if ".." in PurePosixPath(value).parts:
        return ["path_traversal"]
    return []


def validate_with_jsonschema(failures: list[dict[str, str]]) -> str:
    """若本地已有 jsonschema，则补做 Draft 2020-12；run optional full schema checks."""

    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        # 中文：不为开发期验证器自动下载依赖；独立结构和语义检查仍会执行。
        # English: Never download a dev-only validator; independent checks still run.
        return "not_installed_structural_checks_only"

    schemas = {path: load_json(path) for path in SCHEMA_FILES}
    for relative_path, schema in schemas.items():
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:
            add_failure(failures, relative_path, "invalid_json_schema", exc)

    validator = Draft202012Validator(schemas["schemas_v3/pipeline_config_v3.schema.json"])
    for relative_path in EXAMPLE_FILES:
        errors = sorted(validator.iter_errors(load_json(relative_path)), key=lambda item: list(item.path))
        for error in errors:
            location = "/".join(str(part) for part in error.path) or "<root>"
            add_failure(failures, relative_path, "jsonschema_example", f"{location}: {error.message}")

    schema_rules = {"invalid_json_schema", "jsonschema_example"}
    return "fail" if any(item["rule"] in schema_rules for item in failures) else "pass"


def validate_contracts(require_generated: bool = False) -> dict[str, Any]:
    """执行 V3 文件、schema、registry 与配置交叉检查；run all V3 checks."""

    failures: list[dict[str, str]] = []
    for relative_path in CURRENT_FILES:
        if not (PACKAGE_ROOT / relative_path).is_file():
            add_failure(failures, relative_path, "missing_file", "Required V3 file is absent")
    if require_generated:
        for path in (REPORT_PATH, TREE_PATH):
            if not path.is_file():
                add_failure(failures, path.name, "missing_generated_file", "Run --write-report")

    # 中文：先验证 JSON 可解析和 schema 最小结构；English: parse JSON and check schema basics.
    parsed: dict[str, Any] = {}
    for relative_path in (*SCHEMA_FILES, *REGISTRY_FILES, *EXAMPLE_FILES):
        try:
            parsed[relative_path] = load_json(relative_path)
        except (OSError, json.JSONDecodeError) as exc:
            add_failure(failures, relative_path, "json_parse", exc)

    for relative_path in SCHEMA_FILES:
        schema = parsed.get(relative_path, {})
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            add_failure(failures, relative_path, "schema_draft", schema.get("$schema"))
        if schema.get("type") != "object" or not schema.get("required"):
            add_failure(failures, relative_path, "object_contract", "Expected object with required fields")

    signal_schema = parsed.get(SCHEMA_FILES[0], {})
    if signal_schema.get("properties", {}).get("channel_order", {}).get("const") != CANONICAL_CHANNELS:
        add_failure(failures, SCHEMA_FILES[0], "canonical_channels", "Channel order drift")
    if "channel_present" not in signal_schema.get("required", []):
        add_failure(failures, SCHEMA_FILES[0], "channel_present", "Missing sensors must remain explicit")

    platform_data = parsed.get(REGISTRY_FILES[0], {})
    routing_data = parsed.get(REGISTRY_FILES[1], {})
    feature_data = parsed.get(REGISTRY_FILES[2], {})
    classifier_data = parsed.get(REGISTRY_FILES[3], {})
    profiles = index_by_id(platform_data.get("profiles", []), REGISTRY_FILES[0], failures)
    policies = index_by_id(routing_data.get("policies", []), REGISTRY_FILES[1], failures)
    frontends = index_by_id(routing_data.get("denoiser_frontends", []), REGISTRY_FILES[1], failures)
    extractors = index_by_id(feature_data.get("extractors", []), REGISTRY_FILES[2], failures)
    classifiers = index_by_id(classifier_data.get("classifiers", []), REGISTRY_FILES[3], failures)

    expected_profiles = {"high_performance_x86_64", "accelerated_arm64_edge", "value_arm64_sbc"}
    if set(profiles) != expected_profiles:
        add_failure(failures, REGISTRY_FILES[0], "profile_set", sorted(profiles))
    if set(platform_data.get("allowed_dependencies", [])) != ALLOWED_DEPENDENCIES:
        add_failure(failures, REGISTRY_FILES[0], "allowed_dependencies", platform_data.get("allowed_dependencies"))
    streaming = platform_data.get("streaming_contract", {})
    if streaming.get("bounded_buffer_required") is not True or streaming.get("whole_record_inference_allowed") is not False:
        add_failure(failures, REGISTRY_FILES[0], "bounded_streaming", streaming)

    if routing_data.get("sqi_required") is not True:
        add_failure(failures, REGISTRY_FILES[1], "sqi_required", routing_data.get("sqi_required"))
    if routing_data.get("state_axes", {}).get("state_precedence") != STATE_PRECEDENCE:
        add_failure(failures, REGISTRY_FILES[1], "state_precedence", routing_data.get("state_axes", {}))
    manual_contract = routing_data.get("manual_policy_contract", {})
    if set(manual_contract.get("allowed_values", [])) != ALLOWED_MANUAL_POLICIES:
        add_failure(failures, REGISTRY_FILES[1], "manual_policy_set", manual_contract)
    if manual_contract.get("window_override_allowed") is not False:
        add_failure(failures, REGISTRY_FILES[1], "window_override_forbidden", manual_contract)
    first_stage = routing_data.get("first_stage_parallelism", {})
    if first_stage.get("join_required_before_route") is not True or first_stage.get("denoiser_may_start_before_join") is not False:
        add_failure(failures, REGISTRY_FILES[1], "denoiser_must_be_post_route", first_stage)

    # 中文：V3 machine JSON 不得残留 V2 动作所有者与替换波形枚举。
    # English: V3 machine JSON must not retain V2 owner/replacement semantics.
    for relative_path in ("schemas_v3/pipeline_config_v3.schema.json", "schemas_v3/inference_output_v3.schema.json", REGISTRY_FILES[1], *EXAMPLE_FILES):
        payload = json.dumps(parsed.get(relative_path, {}), ensure_ascii=False)
        for token in FORBIDDEN_MACHINE_TERMS:
            if token in payload:
                add_failure(failures, relative_path, "legacy_v2_term_forbidden", token)

    output_schema = parsed.get("schemas_v3/inference_output_v3.schema.json", {})
    output_actions = (
        output_schema.get("properties", {})
        .get("routing_summary", {})
        .get("properties", {})
        .get("action_counts", {})
        .get("required", [])
    )
    if set(output_actions) != set(routing_data.get("allowed_terminal_action_codes", [])):
        add_failure(failures, "schemas_v3/inference_output_v3.schema.json", "action_registry_drift", output_actions)

    seen_config_ids: set[str] = set()
    seen_profiles: set[str] = set()
    seen_manual_policies: set[str] = set()
    seen_motion_modes: set[bool] = set()
    valid_examples = 0
    for relative_path in EXAMPLE_FILES:
        before = len(failures)
        config = parsed.get(relative_path, {})
        config_id = config.get("config_id")
        if not isinstance(config_id, str) or not config_id:
            add_failure(failures, relative_path, "config_id", config_id)
        elif config_id in seen_config_ids:
            add_failure(failures, relative_path, "duplicate_config_id", config_id)
        else:
            seen_config_ids.add(config_id)

        if config.get("schema_version") != "m1.pipeline_config.v3":
            add_failure(failures, relative_path, "schema_version", config.get("schema_version"))
        if config.get("input_contract_ref") != "m1.signal_input.v2" or config.get("output_contract_ref") != "m1.inference_output.v3":
            add_failure(failures, relative_path, "contract_refs", "Expected V2 input and V3 output")
        profile_id = config.get("platform_profile_id")
        seen_profiles.add(str(profile_id))
        if profile_id not in profiles:
            add_failure(failures, relative_path, "platform_profile", profile_id)

        runtime = config.get("runtime", {})
        engine = str(runtime.get("engine", ""))
        providers = runtime.get("execution_providers", [])
        dependencies = set(runtime.get("allowed_dependencies", []))
        if not dependencies.issubset(ALLOWED_DEPENDENCIES):
            add_failure(failures, relative_path, "runtime_dependencies", sorted(dependencies))
        if "onnxruntime" in engine and (
            "onnxruntime" not in dependencies or not providers or providers[-1] != "CPUExecutionProvider"
        ):
            add_failure(failures, relative_path, "onnx_cpu_reference", providers)
        if runtime.get("device") == "accelerator_with_cpu_fallback" and len(providers) < 2:
            add_failure(failures, relative_path, "accelerator_fallback_chain", providers)
        if runtime.get("buffer_duration_sec", 0) < 40:
            add_failure(failures, relative_path, "bounded_buffer", runtime.get("buffer_duration_sec"))
        if runtime.get("backlog_policy") != "explicit_no_result":
            add_failure(failures, relative_path, "backlog_policy", runtime.get("backlog_policy"))

        modules = config.get("modules", {})
        router = modules.get("quality_router", {})
        policy = policies.get(str(router.get("id")))
        if policy is None:
            add_failure(failures, relative_path, "routing_policy", router.get("id"))
        if router.get("state_precedence") != STATE_PRECEDENCE:
            add_failure(failures, relative_path, "state_precedence", router.get("state_precedence"))
        sqi = router.get("sqi", {})
        if sqi.get("required") is not True or sqi.get("failure_policy") != "explicit_no_result":
            add_failure(failures, relative_path, "sqi_first_required", sqi)
        high = router.get("high_quality_branch", {})
        if high != {
            "action": "return_unchanged_to_feature_extractor",
            "signal_source": "preprocessed_raw",
            "denoiser_allowed": False,
        }:
            add_failure(failures, relative_path, "high_quality_bypass", high)

        motion = router.get("motion_detector", {})
        enabled = motion.get("enabled")
        if isinstance(enabled, bool):
            seen_motion_modes.add(enabled)
        else:
            add_failure(failures, relative_path, "motion_enabled_boolean", enabled)
        thresholds = config.get("thresholds", {})
        if enabled is True:
            if not motion.get("id") or not motion.get("version") or motion.get("threshold_key") != "motion_probability":
                add_failure(failures, relative_path, "enabled_motion_contract", motion)
            if motion.get("failure_policy") != "explicit_no_result":
                add_failure(failures, relative_path, "enabled_motion_fail_closed", motion)
        elif enabled is False:
            nullable = (motion.get("id"), motion.get("version"), motion.get("threshold_key"))
            if nullable != (None, None, None) or motion.get("failure_policy") != "not_applicable":
                add_failure(failures, relative_path, "disabled_motion_contract", motion)
            if thresholds.get("motion_probability") is not None:
                add_failure(failures, relative_path, "disabled_motion_threshold_must_be_null", thresholds)

        degraded = router.get("degraded_branch", {})
        manual_policy = degraded.get("manual_policy")
        seen_manual_policies.add(str(manual_policy))
        common_manual = (
            degraded.get("selection_scope") == "run_or_session_start"
            and degraded.get("selection_provenance") == "configuration_before_run"
            and degraded.get("allow_window_override") is False
        )
        if not common_manual:
            add_failure(failures, relative_path, "manual_policy_must_be_run_locked", degraded)
        if manual_policy == "drop":
            if any(degraded.get(key) is not None for key in ("denoiser_frontend_id", "denoiser_version", "feature_adapter_id")):
                add_failure(failures, relative_path, "drop_forbids_denoiser", degraded)
            if degraded.get("failure_policy") != "expected_abstention":
                add_failure(failures, relative_path, "drop_status", degraded)
        elif manual_policy == "denoise_then_extract_features":
            frontend = frontends.get(str(degraded.get("denoiser_frontend_id")))
            if frontend is None:
                add_failure(failures, relative_path, "denoiser_frontend", degraded.get("denoiser_frontend_id"))
            elif degraded.get("feature_adapter_id") != frontend.get("feature_adapter_id"):
                add_failure(failures, relative_path, "denoiser_feature_adapter", degraded.get("feature_adapter_id"))
            if not degraded.get("denoiser_version"):
                add_failure(failures, relative_path, "denoiser_version", degraded.get("denoiser_version"))
            if degraded.get("failure_policy") != "explicit_no_result_no_raw_fallback":
                add_failure(failures, relative_path, "denoiser_fail_closed", degraded)
        else:
            add_failure(failures, relative_path, "manual_policy", manual_policy)

        feature_ref = modules.get("feature_extractor", {})
        extractor = extractors.get(str(feature_ref.get("id")))
        if extractor is None:
            add_failure(failures, relative_path, "feature_extractor", feature_ref.get("id"))
        elif feature_ref.get("feature_schema_id") != extractor.get("feature_schema_id"):
            add_failure(failures, relative_path, "feature_schema", feature_ref.get("feature_schema_id"))
        if manual_policy == "denoise_then_extract_features":
            frontend = frontends.get(str(degraded.get("denoiser_frontend_id")))
            if frontend is not None and frontend.get("target_feature_schema_id") != feature_ref.get("feature_schema_id"):
                add_failure(failures, relative_path, "raw_denoised_feature_schema_mismatch", frontend)

        classifier_ref = modules.get("classifier", {})
        classifier = classifiers.get(str(classifier_ref.get("id")))
        if classifier is None:
            add_failure(failures, relative_path, "classifier", classifier_ref.get("id"))
        elif extractor is not None and extractor.get("output_adapter") not in classifier.get("accepted_input_adapters", []):
            add_failure(failures, relative_path, "adapter_compatibility", classifier_ref.get("id"))

        high_threshold = thresholds.get("sqi_high_quality_min")
        low_threshold = thresholds.get("sqi_unrecoverable_below")
        if isinstance(high_threshold, (int, float)) and isinstance(low_threshold, (int, float)):
            if not (math.isfinite(high_threshold) and math.isfinite(low_threshold) and low_threshold <= high_threshold):
                add_failure(failures, relative_path, "sqi_threshold_order", thresholds)

        artifacts = config.get("artifacts", {})
        if config.get("deployment_state") == "contract_example":
            if artifacts.get("status") != "pending_export" or artifacts.get("files") != []:
                add_failure(failures, relative_path, "example_artifact_state", artifacts)
        elif config.get("deployment_state") == "deploy_locked":
            files = artifacts.get("files", [])
            if artifacts.get("status") != "locked" or not isinstance(files, list) or not files:
                add_failure(failures, relative_path, "locked_bundle_requires_files", artifacts)
            for artifact in files if isinstance(files, list) else []:
                for error in validate_bundle_relative_path(artifact.get("relative_path")):
                    add_failure(failures, relative_path, error, artifact.get("relative_path"))
                if HASH_PATTERN.fullmatch(str(artifact.get("sha256", ""))) is None:
                    add_failure(failures, relative_path, "artifact_sha256", artifact.get("sha256"))
            required_thresholds = ("sqi_high_quality_min", "sqi_unrecoverable_below")
            if enabled is True:
                required_thresholds += ("motion_probability",)
            for name in required_thresholds:
                value = thresholds.get(name)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    add_failure(failures, relative_path, f"locked_threshold_invalid:{name}", value)

        # 中文：部署配置禁止携带训练/测试状态；English: exclude training/test state.
        forbidden_training = {"fold", "folds", "seed", "labels", "optimizer", "early_stopping", "validation_metrics", "test_metrics"}
        serialized = json.dumps(config, ensure_ascii=False).lower()
        for token in forbidden_training:
            if f'"{token}"' in serialized:
                add_failure(failures, relative_path, "training_state_forbidden", token)

        if len(failures) == before:
            valid_examples += 1

    if seen_profiles != expected_profiles:
        add_failure(failures, "examples_v3", "example_profile_coverage", sorted(seen_profiles))
    if seen_manual_policies != ALLOWED_MANUAL_POLICIES:
        add_failure(failures, "examples_v3", "manual_policy_example_coverage", sorted(seen_manual_policies))
    if seen_motion_modes != {False, True}:
        add_failure(failures, "examples_v3", "motion_mode_example_coverage", sorted(seen_motion_modes))

    jsonschema_status = validate_with_jsonschema(failures)
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "pass" if not failures else "fail",
        "failure_count": len(failures),
        "failures": failures,
        "jsonschema_validation": jsonschema_status,
        "schema_file_count": len(SCHEMA_FILES),
        "registry_file_count": len(REGISTRY_FILES),
        "platform_profile_count": len(profiles),
        "routing_policy_count": len(policies),
        "denoiser_frontend_count": len(frontends),
        "feature_extractor_count": len(extractors),
        "classifier_count": len(classifiers),
        "example_config_count": len(EXAMPLE_FILES),
        "valid_example_config_count": valid_examples,
        "manual_policies_covered": sorted(seen_manual_policies),
        "motion_modes_covered": sorted(seen_motion_modes),
        "canonical_channel_order": CANONICAL_CHANNELS,
        "implementation_status": "contract_only_no_model_execution",
    }


def describe(relative_path: str, payload: bytes) -> str:
    """生成 V3 文件说明；describe one V3 file."""

    suffix = Path(relative_path).suffix.lower()
    if suffix == ".md":
        rendered = payload.decode("utf-8", errors="replace")
        title = next((line[2:] for line in rendered.splitlines() if line.startswith("# ")), Path(relative_path).stem)
        return f"Markdown《{title}》"
    if suffix == ".json":
        data = json.loads(payload.decode("utf-8"))
        return "Machine JSON: " + ",".join(list(data)[:6])
    if suffix == ".py":
        return "Bilingual V3 contract or semantic validator"
    return f"{suffix.lstrip('.').upper()} file"


def render_tree() -> str:
    """渲染 V3 权威文件清单与哈希；render the V3 authority tree and hashes."""

    lines = [
        "# M1 V3 权威文件树与完整性 / M1 V3 Integrity Tree",
        "",
        "> 由 `tools/validate_m1_contracts_v3.py --write-report` 生成；V1/V2 树保留为历史。",
        "",
        "| File | Bytes | SHA-256 | Content |",
        "|---|---:|---|---|",
    ]
    for relative_path in CURRENT_FILES:
        path = PACKAGE_ROOT / relative_path
        payload = path.read_bytes()
        lines.append(f"| `{relative_path}` | {len(payload)} | `{sha256_bytes(payload)}` | {describe(relative_path, payload)} |")
    lines.extend(
        [
            f"| `{REPORT_PATH.name}` | self | intentionally omitted | V3 machine verification |",
            f"| `{TREE_PATH.name}` | self | intentionally omitted | V3 integrity tree |",
            "",
            f"- V3 authoritative/reused files including generated indexes: **{len(CURRENT_FILES) + 2}**.",
            "- All writes remain under `final_v0/`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析命令行；parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true", help="Write V3 report and integrity tree inside M1.")
    return parser.parse_args()


def main() -> int:
    """运行验证并返回确定性退出码；run validation and return a deterministic status."""

    args = parse_args()
    checked_relative(PACKAGE_ROOT)
    if args.write_report:
        preflight = validate_contracts(require_generated=False)
        # 中文：先写合法占位，再生成树，最后执行含生成文件的完整检查。
        # English: Seed a valid placeholder, render the tree, then check generated files.
        atomic_write_json(
            REPORT_PATH,
            {
                "contract_version": CONTRACT_VERSION,
                "status": "generating",
                "preflight_status": preflight["status"],
            },
        )
        atomic_write_text(TREE_PATH, render_tree())
        report = validate_contracts(require_generated=True)
        report["preflight_status"] = preflight["status"]
        atomic_write_json(REPORT_PATH, report)
    else:
        report = validate_contracts(require_generated=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


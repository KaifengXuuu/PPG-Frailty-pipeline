#!/usr/bin/env python3
"""验证 M1 V2 架构合同并生成可追溯报告；validate M1 V2 contracts.

中文
----
默认模式只读验证 V2 schemas、registries 与三档配置。`--write-report` 仅在
M1 包内原子写入 V2 报告和 V2 文件树。工具不导入根目录训练代码、不安装依赖、
不运行模型，也不写入 `_agent`。

English
-------
The default mode performs read-only validation of V2 schemas, registries, and three
platform examples. `--write-report` atomically writes only the V2 report and V2
integrity tree inside this M1 package. It imports no root training code, installs no
dependency, runs no model, and never writes to `_agent`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


# 中文：所有路径从脚本本身解析，调用者的 cwd 无法改变写入目标。
# English: Resolve every path from this file so the caller's cwd cannot redirect writes.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PACKAGE_ROOT / "M1_CONTRACT_VERIFICATION_V2.json"
TREE_PATH = PACKAGE_ROOT / "M1_PACKAGE_TREE_V2.md"
CONTRACT_VERSION = "m1.architecture.v2"
CANONICAL_CHANNELS = ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"]
ALLOWED_DEPENDENCIES = {"numpy", "scipy", "onnxruntime", "scikit-learn"}

SCHEMA_FILES = (
    "schemas_v2/signal_input_v2.schema.json",
    "schemas_v2/pipeline_config_v2.schema.json",
    "schemas_v2/inference_output_v2.schema.json",
)
REGISTRY_FILES = (
    "registries_v2/platform_profiles_v2.json",
    "registries_v2/quality_policy_registry_v2.json",
    "registries_v2/feature_extractor_registry_v2.json",
    "registries_v2/classifier_registry_v2.json",
)
EXAMPLE_FILES = (
    "examples_v2/pipeline_high_performance_x86_v2.json",
    "examples_v2/pipeline_accelerated_arm64_v2.json",
    "examples_v2/pipeline_value_arm64_v2.json",
)
REQUIRED_DOCS = (
    "00_CURRENT_STATUS_V2.md",
    "04_EXISTING_CODE_AUDIT_AND_MOBILE_RISKS.md",
)
CURRENT_FILES = (*REQUIRED_DOCS, *SCHEMA_FILES, *REGISTRY_FILES, *EXAMPLE_FILES, "tools/validate_m1_contracts_v2.py")


def checked_relative(path: Path) -> str:
    """验证路径留在 M1 包内；ensure that a path remains inside the M1 package."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PACKAGE_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Path escapes M1 package: {resolved}") from exc


def load_json(relative_path: str) -> Any:
    """以严格 UTF-8 JSON 读取文件；load a strict UTF-8 JSON document."""

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

    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write(path, text.encode("utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    """写 UTF-8/LF 文本；write UTF-8 text with LF line endings."""

    atomic_write(path, text.replace("\r\n", "\n").encode("utf-8"))


def add_failure(failures: list[dict[str, str]], file: str, rule: str, detail: object) -> None:
    """追加结构化失败；append a structured validation failure."""

    failures.append({"file": file, "rule": rule, "detail": str(detail)})


def index_by_id(records: Iterable[dict[str, Any]], file: str, failures: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """按 ID 建唯一索引；build a unique registry index by ID."""

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


def validate_with_jsonschema(failures: list[dict[str, str]]) -> str:
    """若本地存在 jsonschema，则执行 Draft 2020-12 校验；run optional full schema checks."""

    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        # 中文：V2 validator 仍有独立交叉检查；缺包只降低验证深度，不触发下载。
        # English: Cross-checks still run; a missing dev-only package never triggers download.
        return "not_installed_structural_checks_only"

    schemas = {path: load_json(path) for path in SCHEMA_FILES}
    for relative_path, schema in schemas.items():
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:  # jsonschema exposes several schema-specific subclasses.
            add_failure(failures, relative_path, "invalid_json_schema", exc)

    config_schema = schemas[SCHEMA_FILES[1]]
    validator = Draft202012Validator(config_schema)
    for relative_path in EXAMPLE_FILES:
        for error in sorted(validator.iter_errors(load_json(relative_path)), key=lambda item: list(item.path)):
            location = "/".join(str(part) for part in error.path) or "<root>"
            add_failure(failures, relative_path, "jsonschema_example", f"{location}: {error.message}")
    return "pass" if not any(item["rule"].startswith("invalid_json_schema") or item["rule"] == "jsonschema_example" for item in failures) else "fail"


def validate_contracts(require_generated: bool = False) -> dict[str, Any]:
    """执行 V2 文件、schema 与跨 registry 合同校验；run all V2 contract checks."""

    failures: list[dict[str, str]] = []
    for relative_path in CURRENT_FILES:
        if not (PACKAGE_ROOT / relative_path).is_file():
            add_failure(failures, relative_path, "missing_file", "Required V2 file is absent")
    if require_generated:
        for path in (REPORT_PATH, TREE_PATH):
            if not path.is_file():
                add_failure(failures, path.name, "missing_generated_file", "Run --write-report")

    # 中文：先检查每份 schema 的最小结构，再核对最关键的通道/缺失语义。
    # English: Check minimal schema structure, then enforce channel and missing-sensor semantics.
    for relative_path in SCHEMA_FILES:
        try:
            schema = load_json(relative_path)
        except (OSError, json.JSONDecodeError) as exc:
            add_failure(failures, relative_path, "json_parse", exc)
            continue
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            add_failure(failures, relative_path, "schema_draft", schema.get("$schema"))
        if schema.get("type") != "object" or not schema.get("required"):
            add_failure(failures, relative_path, "object_contract", "Expected object with required fields")

    signal_schema = load_json(SCHEMA_FILES[0])
    if signal_schema.get("properties", {}).get("channel_order", {}).get("const") != CANONICAL_CHANNELS:
        add_failure(failures, SCHEMA_FILES[0], "canonical_channels", "Channel order drift")
    if "channel_present" not in signal_schema.get("required", []):
        add_failure(failures, SCHEMA_FILES[0], "channel_present", "Missing sensors must remain explicit")

    try:
        platform_data = load_json(REGISTRY_FILES[0])
        quality_data = load_json(REGISTRY_FILES[1])
        feature_data = load_json(REGISTRY_FILES[2])
        classifier_data = load_json(REGISTRY_FILES[3])
    except (OSError, json.JSONDecodeError) as exc:
        add_failure(failures, "registries_v2", "json_parse", exc)
        platform_data, quality_data, feature_data, classifier_data = {}, {}, {}, {}

    profiles = index_by_id(platform_data.get("profiles", []), REGISTRY_FILES[0], failures)
    policies = index_by_id(quality_data.get("policies", []), REGISTRY_FILES[1], failures)
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
    if platform_data.get("budgets_are_measured_results") is not False:
        add_failure(failures, REGISTRY_FILES[0], "provisional_budgets", "Budgets must not be labelled measured")

    if quality_data.get("action_mode_cardinality") != "exactly_one":
        add_failure(failures, REGISTRY_FILES[1], "quality_cardinality", quality_data.get("action_mode_cardinality"))
    if classifier_data.get("label_order") != ["pre_frail", "robust_non_frail", "young"]:
        add_failure(failures, REGISTRY_FILES[3], "label_order", classifier_data.get("label_order"))

    for classifier_id, classifier in classifiers.items():
        for field in ("family", "accepted_input_adapters", "artifact_format", "mobile_adapter", "status"):
            if not classifier.get(field):
                add_failure(failures, REGISTRY_FILES[3], "classifier_entry_contract", f"{classifier_id}.{field}")
    for extractor_id, extractor in extractors.items():
        if extractor.get("output_dtype") != "float32":
            add_failure(failures, REGISTRY_FILES[2], "feature_output_dtype", extractor_id)

    valid_examples = 0
    for relative_path in EXAMPLE_FILES:
        before = len(failures)
        try:
            config = load_json(relative_path)
        except (OSError, json.JSONDecodeError) as exc:
            add_failure(failures, relative_path, "json_parse", exc)
            continue

        if config.get("schema_version") != "m1.pipeline_config.v2":
            add_failure(failures, relative_path, "schema_version", config.get("schema_version"))
        if config.get("input_contract_ref") != "m1.signal_input.v2" or config.get("output_contract_ref") != "m1.inference_output.v2":
            add_failure(failures, relative_path, "contract_refs", "Expected V2 input/output refs")
        if config.get("platform_profile_id") not in profiles:
            add_failure(failures, relative_path, "platform_profile", config.get("platform_profile_id"))

        runtime = config.get("runtime", {})
        engine = str(runtime.get("engine", ""))
        providers = runtime.get("execution_providers", [])
        dependencies = set(runtime.get("allowed_dependencies", []))
        if not dependencies.issubset(ALLOWED_DEPENDENCIES):
            add_failure(failures, relative_path, "runtime_dependencies", sorted(dependencies))
        if "onnxruntime" in engine and ("onnxruntime" not in dependencies or not providers or providers[-1] != "CPUExecutionProvider"):
            add_failure(failures, relative_path, "onnx_cpu_reference", providers)
        if runtime.get("device") == "accelerator_with_cpu_fallback" and len(providers) < 2:
            add_failure(failures, relative_path, "accelerator_fallback_chain", providers)
        if runtime.get("buffer_duration_sec", 0) < 40:
            add_failure(failures, relative_path, "bounded_buffer", runtime.get("buffer_duration_sec"))
        if runtime.get("backlog_policy") != "explicit_no_result":
            add_failure(failures, relative_path, "backlog_policy", runtime.get("backlog_policy"))

        modules = config.get("modules", {})
        preprocessing = modules.get("preprocessing", {})
        if preprocessing.get("execution_mode") not in {"streaming_causal", "buffered_zero_phase"}:
            add_failure(failures, relative_path, "preprocessing_mode", preprocessing.get("execution_mode"))

        quality = modules.get("quality_strategy", {})
        policy = policies.get(str(quality.get("policy_id")))
        if policy is None:
            add_failure(failures, relative_path, "quality_policy", quality.get("policy_id"))
        else:
            if quality.get("action_mode") != policy.get("action_mode"):
                add_failure(failures, relative_path, "action_owner", quality.get("action_mode"))
            if quality.get("signal_frontend_id") not in policy.get("allowed_signal_frontends", []):
                add_failure(failures, relative_path, "signal_frontend", quality.get("signal_frontend_id"))
        if quality.get("sqi_monitor_enabled") is not True:
            add_failure(failures, relative_path, "sqi_monitor", "SQI diagnostic must remain enabled")

        feature_ref = modules.get("feature_extractor", {})
        extractor = extractors.get(str(feature_ref.get("id")))
        if extractor is None:
            add_failure(failures, relative_path, "feature_extractor", feature_ref.get("id"))
        elif feature_ref.get("feature_schema_id") != extractor.get("feature_schema_id"):
            add_failure(failures, relative_path, "feature_schema", feature_ref.get("feature_schema_id"))

        classifier_ref = modules.get("classifier", {})
        classifier = classifiers.get(str(classifier_ref.get("id")))
        if classifier is None:
            add_failure(failures, relative_path, "classifier", classifier_ref.get("id"))
        elif extractor is not None and extractor.get("output_adapter") not in classifier.get("accepted_input_adapters", []):
            add_failure(failures, relative_path, "adapter_compatibility", classifier_ref.get("id"))

        artifacts = config.get("artifacts", {})
        if config.get("deployment_state") != "contract_example" or artifacts.get("status") != "pending_export":
            add_failure(failures, relative_path, "example_artifact_state", artifacts)

        # 中文：部署配置不得包含训练/验证状态；English: deployment configs must exclude training state.
        forbidden = {"fold", "folds", "seed", "labels", "optimizer", "early_stopping", "validation_metrics", "test_metrics"}
        serialized = json.dumps(config, ensure_ascii=False).lower()
        for token in forbidden:
            if f'"{token}"' in serialized:
                add_failure(failures, relative_path, "training_state_forbidden", token)

        if len(failures) == before:
            valid_examples += 1

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
        "quality_policy_count": len(policies),
        "feature_extractor_count": len(extractors),
        "classifier_count": len(classifiers),
        "example_config_count": len(EXAMPLE_FILES),
        "valid_example_config_count": valid_examples,
        "canonical_channel_order": CANONICAL_CHANNELS,
        "allowed_dependencies": sorted(ALLOWED_DEPENDENCIES),
        "streaming_contract": "bounded_40s_examples_no_whole_record",
        "implementation_status": "contract_only_no_model_execution",
    }


def describe(relative_path: str, payload: bytes) -> str:
    """生成 V2 文件说明；describe one V2 file."""

    suffix = Path(relative_path).suffix.lower()
    if suffix == ".md":
        text = payload.decode("utf-8", errors="replace")
        title = next((line[2:] for line in text.splitlines() if line.startswith("# ")), Path(relative_path).stem)
        return f"Markdown《{title}》"
    if suffix == ".json":
        data = json.loads(payload.decode("utf-8"))
        return "Machine JSON: " + ",".join(list(data)[:6])
    if suffix == ".py":
        return "Bilingual V2 contract validator"
    return f"{suffix.lstrip('.').upper()} file"


def render_tree() -> str:
    """渲染权威 V2 文件清单和哈希；render the authoritative V2 file list and hashes."""

    lines = [
        "# M1 V2 权威文件树与完整性 / M1 V2 Integrity Tree",
        "",
        "> 由 `tools/validate_m1_contracts_v2.py --write-report` 生成；V1 历史层见 `M1_PACKAGE_TREE.md`。",
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
            f"| `{REPORT_PATH.name}` | self | intentionally omitted | V2 machine verification |",
            f"| `{TREE_PATH.name}` | self | intentionally omitted | V2 integrity tree |",
            "",
            f"- V2 authoritative files including generated indexes: **{len(CURRENT_FILES) + 2}**.",
            "- All V2 writes remain under `final_v0/`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析命令行；parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true", help="Write V2 report and integrity tree inside the M1 package.")
    return parser.parse_args()


def main() -> int:
    """运行验证并返回确定性退出码；run validation and return a deterministic exit code."""

    args = parse_args()
    checked_relative(PACKAGE_ROOT)
    if args.write_report:
        preflight = validate_contracts(require_generated=False)
        # 中文：首次运行先写合法占位，随后生成树并执行含生成文件检查的最终验证。
        # English: Seed a first-run placeholder, then render the tree and run final checks.
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


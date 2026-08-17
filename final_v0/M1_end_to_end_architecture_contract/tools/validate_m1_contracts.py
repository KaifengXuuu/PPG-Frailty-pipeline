#!/usr/bin/env python3
"""验证 M1 架构合同并生成可追溯索引；validate M1 contracts and build traceable indexes.

中文
----
默认模式只读验证 schemas、registries 和三档配置。`--write-report` 仅在 M1 包内
写入机器验证报告与包树。工具不导入项目根训练代码，也不安装依赖或运行模型。

English
-------
The default mode validates schemas, registries, and the three platform examples without
writing. ``--write-report`` writes only the machine report and package tree inside this
M1 package. The tool neither imports root training code nor installs or runs models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


# 中文：从脚本位置解析包根，避免调用目录改变写入位置。
# English: Resolve the package root from this script so cwd cannot redirect writes.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PACKAGE_ROOT / "M1_CONTRACT_VERIFICATION.json"
TREE_PATH = PACKAGE_ROOT / "M1_PACKAGE_TREE.md"
CONTRACT_VERSION = "m1.architecture.v1"
CANONICAL_CHANNELS = ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"]
ALLOWED_DEPENDENCIES = {"numpy", "scipy", "onnxruntime", "scikit-learn"}

SCHEMA_FILES = (
    "schemas/signal_input.schema.json",
    "schemas/pipeline_config.schema.json",
    "schemas/inference_output.schema.json",
)
REGISTRY_FILES = (
    "registries/platform_profiles.json",
    "registries/quality_policy_registry.json",
    "registries/feature_extractor_registry.json",
    "registries/classifier_registry.json",
)
EXAMPLE_FILES = (
    "examples/pipeline_high_performance_x86.json",
    "examples/pipeline_accelerated_arm64.json",
    "examples/pipeline_value_arm64.json",
)
REQUIRED_DOCS = (
    "README.md",
    "01_END_TO_END_ARCHITECTURE_AND_API.md",
    "02_MOBILE_PLATFORM_PROFILES.md",
    "03_TRAINING_VS_MOBILE_INFERENCE_BOUNDARY.md",
)


def checked_relative(path: Path) -> str:
    """验证路径位于 M1 包并返回相对路径；ensure containment and return a relative path."""

    resolved = path.resolve()
    root = PACKAGE_ROOT.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Path escapes M1 package: {resolved}") from exc


def load_json(relative_path: str) -> Any:
    """以 UTF-8 读取 JSON；load UTF-8 JSON."""

    path = PACKAGE_ROOT / relative_path
    checked_relative(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(payload: bytes) -> str:
    """返回稳定 SHA-256；return a stable SHA-256 digest."""

    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    """在 M1 包内原子写入；atomically write inside the M1 package."""

    checked_relative(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    """写确定性 UTF-8 JSON；write deterministic UTF-8 JSON."""

    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write(path, text.encode("utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    """写 UTF-8/LF 文本；write UTF-8 text with LF newlines."""

    atomic_write(path, text.replace("\r\n", "\n").encode("utf-8"))


def add_failure(failures: list[dict[str, str]], file: str, rule: str, detail: str) -> None:
    """追加结构化失败；append a structured validation failure."""

    failures.append({"file": file, "rule": rule, "detail": detail})


def index_by_id(records: Iterable[dict[str, Any]], label: str, failures: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """按 ID 建索引并拒绝重复；index records by ID and reject duplicates."""

    index: dict[str, dict[str, Any]] = {}
    for record in records:
        item_id = str(record.get("id", ""))
        if not item_id:
            add_failure(failures, label, "missing_id", "Registry entry has no id")
        elif item_id in index:
            add_failure(failures, label, "duplicate_id", item_id)
        else:
            index[item_id] = record
    return index


def validate_schema_documents(failures: list[dict[str, str]]) -> None:
    """检查 JSON Schema 的最小机器合同；check the minimal machine-readable schema contract."""

    for relative_path in SCHEMA_FILES:
        try:
            schema = load_json(relative_path)
        except (OSError, json.JSONDecodeError) as exc:
            add_failure(failures, relative_path, "json_parse", str(exc))
            continue
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            add_failure(failures, relative_path, "schema_draft", "Expected JSON Schema draft 2020-12")
        if schema.get("type") != "object":
            add_failure(failures, relative_path, "top_level_type", "Expected object")
        if not schema.get("required") or not schema.get("properties"):
            add_failure(failures, relative_path, "required_properties", "Missing required/properties")

    # 中文：通道顺序是跨模块最重要的不变量，直接核对 const。
    # English: Channel order is a core cross-module invariant, so check its const directly.
    signal_schema = load_json(SCHEMA_FILES[0])
    actual_channels = signal_schema.get("properties", {}).get("channel_order", {}).get("const")
    if actual_channels != CANONICAL_CHANNELS:
        add_failure(failures, SCHEMA_FILES[0], "canonical_channels", str(actual_channels))


def validate_platform_profiles(platform_data: dict[str, Any], failures: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """验证平台分档和 provisional 预算；validate platform tiers and provisional budgets."""

    profiles = index_by_id(platform_data.get("profiles", []), REGISTRY_FILES[0], failures)
    expected = {"high_performance_x86_64", "accelerated_arm64_edge", "value_arm64_sbc"}
    if set(profiles) != expected:
        add_failure(failures, REGISTRY_FILES[0], "profile_set", f"Expected {sorted(expected)}, got {sorted(profiles)}")
    dependencies = set(platform_data.get("allowed_dependencies", []))
    if dependencies != ALLOWED_DEPENDENCIES:
        add_failure(failures, REGISTRY_FILES[0], "allowed_dependencies", str(sorted(dependencies)))
    for profile_id, profile in profiles.items():
        budgets = profile.get("budgets", {})
        for key in ("max_pipeline_latency_ms_per_hop", "max_peak_process_ram_mb", "max_bundle_mb", "target_display_update_sec"):
            value = budgets.get(key)
            if not isinstance(value, (int, float)) or value <= 0:
                add_failure(failures, REGISTRY_FILES[0], "positive_budget", f"{profile_id}.{key}={value}")
        if profile.get("examples_are_procurement_recommendations") is not False:
            add_failure(failures, REGISTRY_FILES[0], "procurement_boundary", profile_id)
    return profiles


def validate_example(
    relative_path: str,
    config: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    extractors: dict[str, dict[str, Any]],
    classifiers: dict[str, dict[str, Any]],
    failures: list[dict[str, str]],
) -> None:
    """验证单个可替换配置；validate one swappable deployment configuration."""

    if config.get("schema_version") != "m1.pipeline_config.v1":
        add_failure(failures, relative_path, "schema_version", str(config.get("schema_version")))
    if config.get("input_contract_ref") != "m1.signal_input.v1" or config.get("output_contract_ref") != "m1.inference_output.v1":
        add_failure(failures, relative_path, "contract_refs", "Input/output refs must stay canonical")
    if config.get("platform_profile_id") not in profiles:
        add_failure(failures, relative_path, "platform_profile", str(config.get("platform_profile_id")))

    runtime = config.get("runtime", {})
    dependencies = set(runtime.get("allowed_dependencies", []))
    if not dependencies.issubset(ALLOWED_DEPENDENCIES):
        add_failure(failures, relative_path, "runtime_dependencies", str(sorted(dependencies)))
    if runtime.get("device") != "cpu":
        add_failure(failures, relative_path, "cpu_fallback", str(runtime.get("device")))

    modules = config.get("modules", {})
    quality = modules.get("quality_strategy", {})
    policy_id = quality.get("policy_id")
    policy = policies.get(str(policy_id))
    if policy is None:
        add_failure(failures, relative_path, "quality_policy", str(policy_id))
    else:
        if quality.get("action_mode") != policy.get("action_mode"):
            add_failure(failures, relative_path, "quality_action_mode", f"{quality.get('action_mode')} != {policy.get('action_mode')}")
        if quality.get("signal_frontend_id") not in policy.get("allowed_signal_frontends", []):
            add_failure(failures, relative_path, "quality_frontend", str(quality.get("signal_frontend_id")))
    if quality.get("sqi_monitor_enabled") is not True:
        add_failure(failures, relative_path, "sqi_monitor", "SQI diagnostics must remain enabled")

    feature_ref = modules.get("feature_extractor", {})
    extractor = extractors.get(str(feature_ref.get("id")))
    if extractor is None:
        add_failure(failures, relative_path, "feature_extractor", str(feature_ref.get("id")))
    elif feature_ref.get("feature_schema_id") != extractor.get("feature_schema_id"):
        add_failure(failures, relative_path, "feature_schema", str(feature_ref.get("feature_schema_id")))

    classifier_ref = modules.get("classifier", {})
    classifier = classifiers.get(str(classifier_ref.get("id")))
    if classifier is None:
        add_failure(failures, relative_path, "classifier", str(classifier_ref.get("id")))
    elif extractor is not None and extractor.get("output_adapter") not in classifier.get("accepted_input_adapters", []):
        add_failure(
            failures,
            relative_path,
            "adapter_compatibility",
            f"{extractor.get('output_adapter')} not accepted by {classifier_ref.get('id')}",
        )
    if classifier_ref.get("label_map_id") != "frailty3.v1":
        add_failure(failures, relative_path, "label_map", str(classifier_ref.get("label_map_id")))

    # 中文：部署配置不得携带训练/验证状态；English: deploy configs must not carry training state.
    forbidden = {"fold", "folds", "seed", "labels", "optimizer", "early_stopping", "validation_metrics", "test_metrics"}
    serialized = json.dumps(config, ensure_ascii=False).lower()
    for token in forbidden:
        if f'"{token}"' in serialized:
            add_failure(failures, relative_path, "training_state_forbidden", token)


def validate_contracts(require_generated: bool = False) -> dict[str, Any]:
    """运行完整合同验证；run the complete contract validation."""

    failures: list[dict[str, str]] = []
    for relative_path in (*REQUIRED_DOCS, *SCHEMA_FILES, *REGISTRY_FILES, *EXAMPLE_FILES, "tools/validate_m1_contracts.py"):
        if not (PACKAGE_ROOT / relative_path).is_file():
            add_failure(failures, relative_path, "missing_file", "Required M1 file is absent")
    if require_generated:
        for relative_path in (REPORT_PATH.name, TREE_PATH.name):
            if not (PACKAGE_ROOT / relative_path).is_file():
                add_failure(failures, relative_path, "missing_generated_file", "Run --write-report")

    validate_schema_documents(failures)
    try:
        platform_data = load_json(REGISTRY_FILES[0])
        quality_data = load_json(REGISTRY_FILES[1])
        feature_data = load_json(REGISTRY_FILES[2])
        classifier_data = load_json(REGISTRY_FILES[3])
    except (OSError, json.JSONDecodeError) as exc:
        add_failure(failures, "registries", "json_parse", str(exc))
        platform_data, quality_data, feature_data, classifier_data = {}, {}, {}, {}

    profiles = validate_platform_profiles(platform_data, failures)
    policies = index_by_id(quality_data.get("policies", []), REGISTRY_FILES[1], failures)
    extractors = index_by_id(feature_data.get("extractors", []), REGISTRY_FILES[2], failures)
    classifiers = index_by_id(classifier_data.get("classifiers", []), REGISTRY_FILES[3], failures)
    if quality_data.get("action_mode_cardinality") != "exactly_one":
        add_failure(failures, REGISTRY_FILES[1], "quality_cardinality", str(quality_data.get("action_mode_cardinality")))
    if classifier_data.get("label_order") != ["pre_frail", "robust_non_frail", "young"]:
        add_failure(failures, REGISTRY_FILES[3], "label_order", str(classifier_data.get("label_order")))

    valid_examples = 0
    for relative_path in EXAMPLE_FILES:
        before = len(failures)
        try:
            config = load_json(relative_path)
        except (OSError, json.JSONDecodeError) as exc:
            add_failure(failures, relative_path, "json_parse", str(exc))
            continue
        validate_example(relative_path, config, profiles, policies, extractors, classifiers, failures)
        if len(failures) == before:
            valid_examples += 1

    return {
        "contract_version": CONTRACT_VERSION,
        "status": "pass" if not failures else "fail",
        "failure_count": len(failures),
        "failures": failures,
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
        "implementation_status": "contract_only_no_model_execution",
    }


def build_tree(relative_files: Iterable[str]) -> list[str]:
    """构建稳定 Unicode 树；build a stable Unicode tree."""

    tree: dict[str, dict[str, Any]] = {}
    for relative_path in relative_files:
        node = tree
        for part in Path(relative_path).parts:
            node = node.setdefault(part, {})
    lines = [PACKAGE_ROOT.name + "/"]

    def visit(node: dict[str, dict[str, Any]], prefix: str) -> None:
        names = sorted(node, key=lambda name: (not bool(node[name]), name.lower()))
        for index, name in enumerate(names):
            is_last = index == len(names) - 1
            lines.append(prefix + ("└── " if is_last else "├── ") + name)
            if node[name]:
                visit(node[name], prefix + ("    " if is_last else "│   "))

    visit(tree, "")
    return lines


def describe(relative_path: str, payload: bytes) -> str:
    """生成简短内容说明；produce a concise content description."""

    suffix = Path(relative_path).suffix.lower()
    if suffix == ".md":
        text = payload.decode("utf-8", errors="replace")
        title = next((line[2:] for line in text.splitlines() if line.startswith("# ")), Path(relative_path).stem)
        return f"Markdown《{title}》；Mermaid={text.count('```mermaid')}"
    if suffix == ".json":
        data = json.loads(payload.decode("utf-8"))
        keys = ",".join(list(data)[:6]) if isinstance(data, dict) else "array"
        return f"机器JSON；顶层={keys}"
    if suffix == ".py":
        return "双语合同验证与索引工具"
    return f"{suffix.lstrip('.').upper() or 'FILE'} 文件"


def render_tree() -> str:
    """渲染包树与逐文件哈希；render the package tree and per-file hashes."""

    excluded = {TREE_PATH.resolve(), REPORT_PATH.resolve()}
    files = sorted(path for path in PACKAGE_ROOT.rglob("*") if path.is_file() and path.resolve() not in excluded)
    relative_files = [path.relative_to(PACKAGE_ROOT).as_posix() for path in files]
    tree_files = sorted(relative_files + [TREE_PATH.name, REPORT_PATH.name])
    lines = [
        "# M1 包文件树与逐文件说明 / M1 Package Tree",
        "",
        "> 由 `tools/validate_m1_contracts.py --write-report` 生成；验证报告与本树不自哈希。",
        "",
        "## Tree",
        "",
        "```text",
        *build_tree(tree_files),
        "```",
        "",
        "## Integrity",
        "",
        "| File | Bytes | SHA-256 | Content |",
        "|---|---:|---|---|",
    ]
    for path, relative_path in zip(files, relative_files):
        payload = path.read_bytes()
        lines.append(f"| `{relative_path}` | {len(payload)} | `{sha256_bytes(payload)}` | {describe(relative_path, payload)} |")
    lines.extend(
        [
            f"| `{REPORT_PATH.name}` | self | intentionally omitted | 自动生成验证报告 |",
            f"| `{TREE_PATH.name}` | self | intentionally omitted | 自动生成包树 |",
            "",
            f"- Permanent files including generated indexes: **{len(tree_files)}**.",
            "- 所有写入均位于 `final_v0/M1_end_to_end_architecture_contract/`。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析只读/写报告模式；parse read-only or report-writing mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write M1_CONTRACT_VERIFICATION.json and M1_PACKAGE_TREE.md inside the package.",
    )
    return parser.parse_args()


def main() -> int:
    """执行验证并返回 shell 状态；run validation and return a shell status."""

    args = parse_args()
    checked_relative(PACKAGE_ROOT)
    if args.write_report:
        preflight = validate_contracts(require_generated=False)
        atomic_write_text(TREE_PATH, render_tree())
        report = validate_contracts(require_generated=True)
        # 中文：保留 preflight 以证明生成前静态合同也通过。
        # English: Preserve preflight status to show static contracts passed before generation.
        report["preflight_status"] = preflight["status"]
        atomic_write_json(REPORT_PATH, report)
    else:
        report = validate_contracts(require_generated=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


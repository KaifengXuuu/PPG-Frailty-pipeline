#!/usr/bin/env python3
"""V2 身份、配置、决策与依赖门验证器 / V2 identity and profile validator."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@dataclass(frozen=True)
class Check:
    """一个关闭失败检查 / One fail-closed check."""

    check_id: str
    run: Callable[[], str]


def _required_paths() -> str:
    """核对 V2 活动层交付 / Check active V2-layer deliverables."""

    required = [
        "README.md",
        "STATUS.md",
        "RUNBOOK.md",
        "pyproject.toml",
        "manifests/internal_records_v2.csv",
        "reports/internal_manifest_v2_report.json",
        "tools/materialize_internal_manifest_v2.py",
        "configs/reference_static_role_aware_v2.yaml",
        "configs/reference_static_feature_vector_v2.yaml",
        "configs/reference_static_feature_matrix_v2.yaml",
        "configs/reference_static_fusion_v2.yaml",
        "configs/v2_decision_profile.yaml",
        "configs/formal_ablation_profiles_v2.yaml",
        "configs/motion_detector_contract_v2.yaml",
        "src/ppg_frailty/config.py",
        "src/ppg_frailty/module_registry.py",
        "src/ppg_frailty/cli.py",
        "src/ppg_frailty/experiment.py",
        "src/ppg_frailty/v2_contract.py",
        "src/ppg_frailty/data/external_folds.py",
        "src/ppg_frailty/features/prv_backend_compare.py",
        "src/ppg_frailty/representations/imu_transform.py",
        "src/ppg_frailty/quality/motion.py",
        "src/ppg_frailty/quality/motion_runner.py",
        "src/ppg_frailty/models/motion.py",
        "src/ppg_frailty/training/trainer.py",
        "src/ppg_frailty/training/statistics.py",
        "splits/ptt_formal_repeated_grouped_5x5_v2.csv",
    ]
    missing = [item for item in required if not (ROOT / item).is_file()]
    if missing:
        raise AssertionError(f"missing V2 paths: {missing}")
    return f"required_paths={len(required)}"


def _package_identity() -> str:
    """拒绝活动 packaging 回退到 V1 / Reject an active V1 package identity."""

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project["name"] != "ppg-frailty-final-pipeline-v2":
        raise AssertionError("pyproject project.name is not V2")
    if not str(project["version"]).startswith("2."):
        raise AssertionError("pyproject project.version is not V2")
    return f"package={project['name']}@{project['version']}"


def _formal_and_legacy_configs() -> str:
    """正式只收 V2，V1 只可显式 provenance / Separate formal and legacy configs."""

    from ppg_frailty.config import load_config

    formal_names = (
        "reference_static_role_aware_v2.yaml",
        "reference_static_feature_vector_v2.yaml",
        "reference_static_feature_matrix_v2.yaml",
        "reference_static_fusion_v2.yaml",
    )
    formal_paths = [ROOT / "configs" / name for name in formal_names]
    formal = [load_config(item) for item in formal_paths]
    if not formal or len({item.config_id for item in formal}) != len(formal):
        raise AssertionError("formal V2 config IDs are missing or duplicated")
    for item in formal:
        manifest = item.section("manifest")
        if manifest.get("path") != "manifests/internal_records_v2.csv":
            raise AssertionError("formal V2 config does not use internal_records_v2")
        if manifest.get("manifest_version") != "internal_records_v2":
            raise AssertionError("formal V2 manifest_version drift")
    observed_modes = {item.representation_mode for item in formal}
    expected_modes = {"raw", "feature_vector", "feature_matrix", "fusion"}
    if observed_modes != expected_modes:
        raise AssertionError(
            f"formal representation entrypoints drift: {sorted(observed_modes)}"
        )
    legacy_paths = sorted((ROOT / "configs").glob("*_v1.yaml"))
    for item in legacy_paths:
        try:
            load_config(item)
        except ValueError as error:
            if "provenance-only" not in str(error):
                raise
        else:
            raise AssertionError(f"legacy config entered formal loader: {item.name}")
        if not load_config(item, allow_legacy=True).is_legacy:
            raise AssertionError(f"legacy inspection identity failed: {item.name}")
    return f"formal_v2={len(formal)},legacy_provenance={len(legacy_paths)}"


def _project_profiles() -> str:
    """Validate project decisions, ablations and ordinary runtime imports."""

    from ppg_frailty.config import (
        dependency_availability_report,
        load_config,
        load_formal_ablation_profiles,
        load_v2_decision_profile,
    )

    decision = load_v2_decision_profile(ROOT / "configs/v2_decision_profile.yaml")
    dependencies = dependency_availability_report(
        load_config(ROOT / "configs/reference_static_feature_vector_v2.yaml")
    )
    ablations = load_formal_ablation_profiles(
        ROOT / "configs/formal_ablation_profiles_v2.yaml"
    )
    if decision["confirmed_defaults"]["quality"]["default_mode"] != "off":
        raise AssertionError("SQI must remain off in the V2 default profile")
    fixed_cases = ablations["families"]["fixed_kernel_samples"]["cases"]
    if len(fixed_cases) != 12 or ablations["execution_policy"]["auto_run"]:
        raise AssertionError("formal single-factor profile catalogue drifted")
    return (
        "decision_profile=1,ablation_profiles=19,"
        f"runtime_ready={dependencies['ready']}"
    )


def _registry() -> str:
    """核对 V2 PRV 与固定sample身份 / Check V2 registry identities."""

    from ppg_frailty.module_registry import list_modules

    modules = list_modules()
    identifiers = {row["module_id"] for row in modules}
    required = {
        "local",
        "aura_hrv_analysis",
        "rhenan_hrv",
        "ShapeFormerChannelSpecificOSD",
        "fixed_kernel_samples_resampling_ablation",
        "emd_sifting_rate_only",
        "ceemd_lite_nlms_legacy",
        "dwt_a2_legacy",
        "sqi_only",
        "sqi_plus_motion_override",
        "historical_light_cnn_backup",
    }
    if not required.issubset(identifiers):
        raise AssertionError(f"missing registry identities: {sorted(required-identifiers)}")
    pisd = next(
        row
        for row in modules
        if row["module_id"] == "ShapeFormerChannelSpecificOSD"
    )
    if (
        pisd["implementation"]
        != "ppg_frailty.models.shapeformer_literature.LiteratureShapeFormerChannelSpecificOSD"
    ):
        raise AssertionError(
            "ShapeFormerChannelSpecificOSD registry points to a non-runtime symbol"
        )
    return f"module_count={len(identifiers)}"


def _python_syntax() -> str:
    """递归解析所有活动 Python / Recursively parse all active Python."""

    paths = sorted(
        path
        for base in (SRC, ROOT / "tools", ROOT / "tests")
        for path in base.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    if not paths:
        raise AssertionError("no active Python files found")
    for item in paths:
        ast.parse(item.read_text(encoding="utf-8"), filename=str(item))
    return f"python_files={len(paths)}"


def _active_generation_boundary() -> str:
    """禁止 V1 run schema 和活动 V1 import / Reject active V1 generation coupling."""

    paths = sorted(
        path for path in SRC.rglob("*.py") if "__pycache__" not in path.parts
    )
    violations: list[str] = []
    for item in paths:
        source = item.read_text(encoding="utf-8")
        relative = item.relative_to(ROOT).as_posix()
        if "ppg_frailty.pipeline_run.v1" in source:
            violations.append(f"{relative}:forbidden_pipeline_run_v1")
        tree = ast.parse(source, filename=str(item))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                function = node.func
                if (
                    isinstance(function, ast.Name) and function.id == "__import__"
                ) or (
                    isinstance(function, ast.Attribute) and function.attr == "import_module"
                ):
                    imported.append(node.args[0].value)
            if any("final_pipeline_v1" in name for name in imported):
                violations.append(
                    f"{relative}:{getattr(node, 'lineno', 0)}:active_v1_import"
                )
    if violations:
        raise AssertionError(
            "active generation boundary violations: " + ", ".join(violations)
        )
    return f"source_files={len(paths)},forbidden_generation_refs=0"


def _write(path: Path, payload: dict[str, object]) -> None:
    """只允许在 V2 内原子写报告 / Atomically write a report below V2."""

    target = path.resolve()
    target.relative_to(ROOT.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def main(argv: list[str] | None = None) -> int:
    """执行 V2 门并导出唯一 V2 schema / Execute the V2 gate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", type=Path)
    arguments = parser.parse_args(argv)
    if ROOT.name != "final_pipeline_v2":
        raise SystemExit("validator root is not final_pipeline_v2")
    checks = [
        Check("required_paths", _required_paths),
        Check("package_identity", _package_identity),
        Check("formal_and_legacy_configs", _formal_and_legacy_configs),
        Check("project_profiles", _project_profiles),
        Check("registry", _registry),
        Check("active_generation_boundary", _active_generation_boundary),
        Check("python_syntax", _python_syntax),
    ]
    rows: list[dict[str, str]] = []
    for check in checks:
        try:
            rows.append({"check_id": check.check_id, "status": "passed", "detail": check.run()})
        except Exception as error:  # 每项均保留 / Retain every check result.
            rows.append({"check_id": check.check_id, "status": "failed", "detail": repr(error)})
    status = "passed" if all(row["status"] == "passed" for row in rows) else "failed"
    payload: dict[str, object] = {
        "schema_version": "ppg_frailty.v2_validation.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": status,
        "checks_run": len(rows),
        "checks": rows,
    }
    if arguments.write_report is not None:
        _write(arguments.write_report, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

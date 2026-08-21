"""V2 非交互命令行 / V2 non-interactive command line.

中文：每个子命令输出 strict JSON 并以非零状态表示失败。正式 ``run`` 只读取
冻结 manifest/folds；量化 ``compare`` 与 ``ablate`` 明确标为 synthetic contract
tests，不会冒充真实 frailty 或 PTT benchmark。

English: Every command emits strict JSON and returns non-zero on failure. Formal
``run`` reads frozen manifest/folds only. Quantitative synthetic comparisons are
clearly scoped and never presented as frailty or external-PTT evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

import yaml
from pathlib import Path
from typing import Any, Sequence

from .config import (
    dependency_availability_report,
    load_config,
    load_formal_ablation_profiles,
    load_formal_experiment_catalog,
    load_v2_decision_profile,
    materialize_formal_ablation_config,
)
from .contracts import to_strict_json_value
from .module_registry import list_modules, registry_sha256
from .pipeline import (
    PipelinePaths,
    run_ablation,
    run_artifact_comparison,
    run_imu_gravity_comparison,
    run_model_comparison,
    run_pipeline,
    validate_installation,
    write_quantitative_report,
)


def _write_new_artifact_json(
    payload: dict[str, Any],
    requested: str | Path,
) -> Path:
    """Write one non-overwriting CLI report below the pipeline root."""

    paths = PipelinePaths.discover()
    target = paths.new_artifact_path(requested)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{time.time_ns()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary artifact already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(
                to_strict_json_value(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"artifact target appeared: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _print(payload: Any, *, stream: Any = None) -> None:
    """打印单行 strict JSON / Print one-line strict JSON."""

    print(
        json.dumps(to_strict_json_value(payload), ensure_ascii=False, sort_keys=True, allow_nan=False),
        file=stream or sys.stdout,
    )


class _TerminalProgress:
    """Small refresh-style progress display kept separate from JSON stdout."""

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream or sys.stderr
        self.total = 1
        self.current = 0
        self.enabled = bool(getattr(self.stream, "isatty", lambda: False)())

    def __call__(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        stage = str(event.get("stage", "running"))
        self.total = max(int(event.get("total_cells", self.total)), 1)
        if stage == "cell_start":
            self.current = max(int(event.get("current_cell", self.current + 1)) - 1, 0)
        elif stage in {"cell_complete", "run_complete"}:
            self.current = int(event.get("current_cell", self.total if stage == "run_complete" else self.current + 1))
        fraction = min(max(self.current / self.total, 0.0), 1.0)
        width = 24
        filled = int(round(width * fraction))
        repeat = event.get("repeat_index")
        fold = event.get("fold_index")
        cell = "" if repeat is None else f" r{repeat}f{fold}"
        line = (
            f"\r[{'#' * filled}{'-' * (width - filled)}] "
            f"{self.current:>2}/{self.total:<2} {stage}{cell}"
        )
        print(line + "\x1b[K", end="", file=self.stream, flush=True)
        if stage in {"run_complete", "run_error", "cell_error"} or (
            stage == "cell_complete" and self.total == 1
        ):
            print(file=self.stream, flush=True)


V2_CONFIG_ALIASES = {
    "default": "configs/reference_static_feature_vector_v2.yaml",
    "raw": "configs/reference_static_role_aware_v2.yaml",
    "feature-vector": "configs/reference_static_feature_vector_v2.yaml",
    "feature-matrix": "configs/reference_static_feature_matrix_v2.yaml",
    "fusion": "configs/reference_static_fusion_v2.yaml",
}


def _registered_config(value: str) -> str:
    """解析正式 V2 config；V1 内容仍由 loader 拒绝 / Resolve formal V2 paths."""

    if value in V2_CONFIG_ALIASES:
        return V2_CONFIG_ALIASES[value]
    candidate = Path(value)
    if candidate.suffix:
        return value
    return f"configs/{value}.yaml"


def _registered_legacy_config(value: str) -> str:
    """只用于 provenance inspection / Resolve an explicit legacy config path."""

    candidate = Path(value)
    if len(candidate.parts) == 1:
        name = candidate.name if candidate.suffix else f"{candidate.name}.yaml"
        return f"historical/v1_transition/configs/{name}"
    if candidate.parts[:3] != ("historical", "v1_transition", "configs"):
        raise argparse.ArgumentTypeError(
            "legacy configs must resolve under historical/v1_transition/configs"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    """构建无隐藏行为的 parser / Build the explicit command parser."""

    parser = argparse.ArgumentParser(prog="ppg-frailty", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    modules = subcommands.add_parser("list-modules", help="list canonical modules")
    modules.add_argument(
        "--family",
        choices=sorted({"all", *(row["family"] for row in list_modules())}),
        default="all",
    )

    subcommands.add_parser(
        "motion-validate",
        help="validate the V2-010 motion contract and frozen folds without training or PTT evaluation",
    )
    motion_internal = subcommands.add_parser(
        "motion-train-internal",
        help="explicitly execute the source-bound Frailty29 5x5 motion reference",
    )
    motion_internal.add_argument("--repository-root", required=True)
    motion_internal.add_argument("--output-dir", required=True)
    motion_internal.add_argument(
        "--device",
        default="cuda",
        help="CUDA training device (cuda or cuda:N); CPU fallback is forbidden.",
    )

    motion_ptt = subcommands.add_parser(
        "motion-evaluate-ptt",
        help="explicitly evaluate a completed internal motion reference on hash-bound PTT inputs",
    )
    motion_ptt.add_argument("--repository-root", required=True)
    motion_ptt.add_argument("--internal-evidence", required=True)
    motion_ptt.add_argument("--internal-evidence-sha256", required=True)
    motion_ptt.add_argument("--unit-evidence")
    motion_ptt.add_argument("--unit-evidence-sha256")
    motion_ptt.add_argument("--output-dir", required=True)

    validate = subcommands.add_parser("validate", help="validate installation or one config")
    validate_group = validate.add_mutually_exclusive_group()
    validate_group.add_argument("--config", type=_registered_config)
    validate_group.add_argument("--all-configs", action="store_true")
    validate_group.add_argument("--profiles-only", action="store_true")
    validate_group.add_argument("--legacy-config", type=_registered_legacy_config)

    smoke = subcommands.add_parser(
        "smoke",
        help="run a non-training V2 contract/preflight smoke",
    )
    smoke.add_argument("--config", default="default", type=_registered_config)

    tests = subcommands.add_parser("test", help="run one registered unittest suite")
    tests.add_argument(
        "--suite",
        choices=["safe", "all", "data", "signal", "artifacts", "features", "models", "training", "integration", "cli", "contracts"],
        default="safe",
        help="safe is the non-comparison default; all is explicit opt-in and may include compare/ablate or copied-V1 acceptance paths",
    )
    tests.add_argument("--report")
    tests.add_argument("--verbosity", choices=[0, 1, 2], default=1, type=int)

    build_data = subcommands.add_parser("build-data", help="rebuild frozen data contracts from authorities")
    build_data.add_argument("--confirm-byte-rehash", action="store_true", help="required because all 261 raw files are re-hashed")

    final_policy = subcommands.add_parser(
        "final-policy",
        help="inspect the manual-final refit and ONNX policy without executing either",
    )
    final_policy.add_argument("--config", required=True, type=_registered_config)

    catalog = subcommands.add_parser(
        "catalog",
        help=(
            "inspect active candidates and declared matched ensemble comparisons "
            "without running them"
        ),
    )
    catalog.add_argument("--line", choices=["line_a", "line_b"], required=True)

    build_catalog = subcommands.add_parser(
        "build-catalog",
        help="materialize fully resolved Line A/B configs; never train or evaluate",
    )
    build_catalog.add_argument("--line", choices=["line_a", "line_b"], required=True)
    build_catalog.add_argument(
        "--output-dir",
        required=True,
        help="new directory below final_pipeline_v2; existing paths are rejected",
    )

    run = subcommands.add_parser("run", help="run formal smoke or full input/protocol audit")
    run.add_argument("--config", required=True, type=_registered_config)
    run.add_argument("--mode", required=True, choices=["smoke", "full"])
    run.add_argument("--output", required=True, help="new JSON path below final_pipeline_v2")

    experiment = subcommands.add_parser(
        "run-experiment",
        help="train/evaluate a real frozen outer-fold experiment",
        description=(
            "Execute the scientific pipeline rather than the input-only audit. "
            "The formal runner is wired for raw, feature_vector, feature_matrix, "
            "and fusion representations through explicit V2 configs. The "
            "raw and matrix five-member ensembles remain explicit comparison configs. "
            "The ShapeFormer reference uses fold-local channel-specific variable-length "
            "OSD/PISD; fixed 128/64 effect-size discovery is a named ablation and never "
            "a fallback. Reduced smoke keeps the complete participant roster but uses "
            "the public 60-second/one-record/one-epoch defaults. Full runs all 25 "
            "cells unless one repeat/fold pair is explicitly selected. 正式 runner "
            "已接线四种表征；两条五成员 ensemble 仅作显式 comparison；"
            "ShapeFormer 参考线不使用固定 shapelet 长度且失败时不会静默回退。"
        ),
    )
    experiment.add_argument("--config", required=True, type=_registered_config)
    experiment.add_argument("--budget", required=True, choices=["reduced-smoke", "full"])
    experiment.add_argument(
        "--repeat", type=int, choices=range(5), default=None,
        help="reduced cell (default 0) or selected full cell; pair with --fold",
    )
    experiment.add_argument(
        "--fold", type=int, choices=range(5), default=None,
        help="reduced cell (default 0) or selected full cell; pair with --repeat",
    )
    experiment.add_argument(
        "--output-dir", required=True,
        help="new, non-existing experiment directory below final_pipeline_v2",
    )
    experiment.add_argument(
        "--measure-operational-costs",
        action="store_true",
        help=(
            "full budget only: explicitly run fixed CPU batch-1 warmup10/repeats100 "
            "after each fitted cell; never enabled by validation or reduced smoke"
        ),
    )
    comparison_archive = subcommands.add_parser(
        "comparison-archive",
        help="build an explicit OOF statistics archive from complete indexed 5x5 runs",
    )
    comparison_archive.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="CONFIG_ID=RUN_DIRECTORY",
        help="repeat for each complete run; directories must be below final_pipeline_v2",
    )
    comparison_archive.add_argument("--reference-config-id", required=True)
    comparison_archive.add_argument("--comparison-family", required=True)
    comparison_archive.add_argument("--comparison-id", required=True)
    comparison_archive.add_argument("--run-id", required=True)
    comparison_archive.add_argument(
        "--output-root",
        required=True,
        help="new archive parent below final_pipeline_v2; final ID/run paths cannot exist",
    )
    comparison_archive.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=None,
        help=(
            "optional exact override; otherwise inherit the persisted evaluation "
            "policy (legacy artifacts fall back to 10000)"
        ),
    )
    comparison_archive.add_argument(
        "--permutation-resamples",
        type=int,
        default=None,
        help=(
            "optional exact override; otherwise inherit the persisted evaluation "
            "policy (legacy artifacts fall back to 100000)"
        ),
    )
    comparison_archive.add_argument(
        "--statistics-seed",
        type=int,
        default=None,
        help=(
            "optional exact override; otherwise inherit the persisted evaluation "
            "policy (legacy artifacts fall back to 42)"
        ),
    )
    comparison_archive.add_argument(
        "--allowed-authority-difference",
        action="append",
        default=[],
        metavar="FACTOR",
        help=(
            "repeat once for each explicitly intended comparison axis; "
            "immutable manifest/fold/source/code authority can never be allowed"
        ),
    )

    manual_selection = subcommands.add_parser(
        "record-selection",
        help="write one immutable purpose-specific manual selection outside an archive",
    )
    manual_selection.add_argument("--comparison-archive", required=True)
    manual_selection.add_argument("--config-id", required=True)
    manual_selection.add_argument("--purpose", required=True)
    manual_selection.add_argument("--rationale", required=True)
    manual_selection.add_argument("--output", required=True)

    final_refit = subcommands.add_parser(
        "final-refit",
        help=(
            "validate complete-run OOF and a human selection; providing "
            "--bundle-directory executes the full-29 refit"
        ),
    )
    final_refit.add_argument("--run-directory", required=True)
    final_refit.add_argument("--selection-record", required=True)
    final_refit.add_argument("--comparison-archive", required=True)
    final_refit.add_argument("--config", required=True, type=_registered_config)
    final_refit.add_argument(
        "--bundle-directory",
        help=(
            "new output directory for the executed all-29 bundle; omit for "
            "preflight only"
        ),
    )

    materialize_ablation = subcommands.add_parser(
        "materialize-ablation-config",
        help="write one registered single-factor formal config; never execute it",
    )
    materialize_ablation.add_argument("--base-config", required=True, type=_registered_config)
    materialize_ablation.add_argument(
        "--family",
        required=True,
        choices=(
            "deep_fixed_epoch", "direct_filter", "imu_gravity",
            "fixed_kernel_samples", "aggregation_balance", "peak_detector",
        ),
    )
    materialize_ablation.add_argument("--profile-id", required=True)
    materialize_ablation.add_argument("--output", required=True)

    compare = subcommands.add_parser("compare", help="run a quantitative synthetic comparison")
    compare_commands = compare.add_subparsers(dest="comparison", required=True)
    artifacts = compare_commands.add_parser("artifacts")
    artifacts.add_argument(
        "--reducers",
        nargs="+",
        default=[
            "identity", "nlms_imu_anc", "ssa_decomposition", "spectral_mask",
            "pca_bss", "fastica_bss", "nmf_bss", "emd_sifting_rate_only",
            "ceemd_lite_nlms_legacy", "dwt_a2_legacy",
        ],
        help="canonical artifact module IDs; legacy short aliases are explicitly labelled",
    )
    artifacts.add_argument("--duration-s", type=float, default=10.0)
    artifacts.add_argument("--min-observation-sec", type=float, default=8.0)
    artifacts.add_argument("--min-peaks", type=int, default=5)
    artifacts.add_argument("--seed", type=int, default=42)
    artifacts.add_argument("--output")
    models = compare_commands.add_parser("models")
    models.add_argument(
        "--models",
        nargs="+",
        default=[
            "CompactCNN1D", "InceptionTimeFull", "InceptionTimeSmall",
            "LogisticRegressionL2", "RBFSVM", "ExtraTrees",
            "ShapeFormerChannelSpecificOSD", "ShapeFormerEffectSizeFixedV1",
            "FileBagFusionCompact", "FileBagFusionInception",
        ],
        help=(
            "registered model identities; defaults are the active catalog smoke "
            "candidates, while optional and ensemble identities are explicit"
        ),
    )
    models.add_argument("--seed", type=int, default=42)
    models.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        help="synthetic member count when an ensemble identity is requested",
    )
    models.add_argument("--output")
    gravity = compare_commands.add_parser("imu-gravity")
    gravity.add_argument("--duration-s", type=float, default=12.0)
    gravity.add_argument("--seed", type=int, default=42)
    gravity.add_argument("--output")
    prv = compare_commands.add_parser(
        "prv-backends",
        help="compare functions on frozen PPI fixtures; never train a classifier",
    )
    prv.add_argument(
        "--backends",
        nargs="+",
        choices=["local", "aura_hrv_analysis", "rhenan_hrv"],
        default=["local", "aura_hrv_analysis", "rhenan_hrv"],
    )
    prv.add_argument("--fixtures", nargs="+")
    prv.add_argument("--output")

    ablate = subcommands.add_parser("ablate", help="run one-factor synthetic ablation")
    ablate.add_argument(
        "--factor",
        required=True,
        choices=["artifact", "model", "dl_fs", "raw_window_s", "fixed_kernel_samples"],
        help="fixed_kernel_samples keeps convolution kernel sample counts fixed; it is not physical-time matched",
    )
    ablate.add_argument("--seed", type=int, default=42)
    ablate.add_argument("--output")
    return parser


def _run_tests(arguments: argparse.Namespace) -> int:
    """调用同一标准测试 runner / Invoke the same standard test runner."""

    paths = PipelinePaths.discover()
    command = [
        sys.executable,
        str(paths.pipeline_root / "tools/run_test_suite.py"),
        "--suite", arguments.suite,
        "--pattern", "test_*.py",
        "--verbosity", str(arguments.verbosity),
    ]
    if arguments.report:
        target = paths.output_path(arguments.report)
        command.extend(("--report", str(target)))
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def _build_data(arguments: argparse.Namespace) -> int:
    """重新物化但绝不重算 folds / Re-materialize without regenerating folds."""

    if not arguments.confirm_byte_rehash:
        raise ValueError("build-data requires --confirm-byte-rehash; it reads all 261 source files")
    paths = PipelinePaths.discover()
    completed = subprocess.run(
        [sys.executable, str(paths.pipeline_root / "tools/materialize_data_contracts.py")],
        cwd=str(paths.repository_root),
        check=False,
    )
    return int(completed.returncode)


def _catalog_summary(line: str) -> dict[str, Any]:
    paths = PipelinePaths.discover()
    catalog = load_formal_experiment_catalog(
        paths.pipeline_root / "configs/formal_experiment_catalog_v2.yaml"
    )
    ablations = load_formal_ablation_profiles(
        paths.pipeline_root / "configs/formal_ablation_profiles_v2.yaml"
    )
    entries = [
        {
            "entry_id": entry["entry_id"],
            "config_id": f"{entry['config_stem']}_{line}_v2",
            "representation_mode": entry["representation_mode"],
            "catalog_role": entry["catalog_role"],
            "model_id": entry["model"]["model_id"],
            "ensemble_size": entry["model"]["ensemble_size"],
            "seed_policy": entry["model"]["seed_policy"],
        }
        for entry in catalog["entries"]
    ]
    return {
        "schema_version": "ppg_frailty.formal_catalog_inspection.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "registered_not_executed",
        "balance_line": line,
        "catalog_id": catalog["catalog_id"],
        "catalog_sha256": catalog["catalog_sha256"],
        "ablation_profile_catalog_id": ablations["catalog_id"],
        "ablation_profile_catalog_sha256": ablations["catalog_sha256"],
        "fixed_kernel_case_count": len(
            ablations["families"]["fixed_kernel_samples"]["cases"]
        ),
        "ablation_profile_auto_run": False,
        "candidate_count": sum(
            row["catalog_role"] in {"reference_candidate", "ablation_candidate"}
            for row in entries
        ),
        "matched_comparator_count": sum(
            row["catalog_role"] == "matched_comparator" for row in entries
        ),
        "ensemble_comparison_count": sum(
            row["catalog_role"] == "ensemble_comparison" for row in entries
        ),
        "auto_run": False,
        "training_executed": False,
        "entries": entries,
    }


def _build_catalog(arguments: argparse.Namespace) -> int:
    paths = PipelinePaths.discover()
    completed = subprocess.run(
        [
            sys.executable,
            str(paths.pipeline_root / "tools/materialize_reference_configs.py"),
            "--line",
            arguments.line,
            "--output-dir",
            arguments.output_dir,
        ],
        cwd=str(paths.pipeline_root),
        check=False,
    )
    return int(completed.returncode)


def _resolve_pipeline_path(value: str | Path) -> Path:
    """限制 CLI 输入位于 V2 root / Resolve an input below the V2 root."""

    paths = PipelinePaths.discover()
    candidate = Path(value)
    candidate = candidate.resolve() if candidate.is_absolute() else (paths.pipeline_root / candidate).resolve()
    candidate.relative_to(paths.pipeline_root)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _resolve_formal_config_path(value: str | Path) -> Path:
    """Resolve an ordinary YAML config, including an explicit external archive."""

    candidate = Path(value)
    resolved = (
        candidate.expanduser().resolve()
        if candidate.is_absolute()
        else _resolve_pipeline_path(candidate)
    )
    if resolved.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("config path must be YAML")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _validate_profiles() -> dict[str, Any]:
    """Validate project decisions and report ordinary import availability."""

    paths = PipelinePaths.discover()
    decision = load_v2_decision_profile(paths.pipeline_root / "configs/v2_decision_profile.yaml")
    reference = load_config(
        paths.pipeline_root / "configs/reference_static_feature_vector_v2.yaml"
    )
    return {
        "schema_version": "ppg_frailty.profile_validation.v2",
        "status": "passed",
        "pipeline_generation": decision["pipeline_generation"],
        "decision_profile_id": decision["profile_id"],
        "runtime_dependencies": dependency_availability_report(reference),
        "deferred_evidence_ids": sorted(decision["deferred_evidence"]),
    }


def _validate_motion_contract() -> dict[str, Any]:
    """Validate only motion contracts/folds; never train or evaluate PTT."""

    paths = PipelinePaths.discover()
    contract_path = paths.pipeline_root / "configs/motion_detector_contract_v2.yaml"
    source = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("motion detector contract must be a mapping")
    from .data.external_folds import load_formal_ptt_repeated_folds
    from .quality.motion import (
        load_motion_fold_jobs,
        motion_contract_payload,
    )

    runtime = motion_contract_payload()
    if source.get("schema_version") != runtime["schema_version"]:
        raise ValueError("motion contract schema does not match runtime")
    source_options = {
        str(item.get("option_id")) for item in source.get("selection", {}).get("options", [])
        if isinstance(item, dict)
    }
    runtime_options = {str(item["option_id"]) for item in runtime["options"]}
    if source_options != runtime_options:
        raise ValueError("motion option registry drift")
    internal_jobs = load_motion_fold_jobs(
        paths.pipeline_root / str(source["internal_training_and_oof"]["split_csv"])
    )
    readiness = source["ptt_external_readiness_audit"]
    ptt_rows = load_formal_ptt_repeated_folds(
        paths.pipeline_root / str(readiness["split_csv"])
    )
    return {
        "schema_version": "ppg_frailty.motion_contract_validation.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "passed",
        "contract_id": str(source["contract_id"]),
        "execution_status": str(source["execution_status"]),
        "default_option": str(source["selection"]["default_option"]),
        "option_count": len(runtime_options),
        "internal_fold_job_count": len(internal_jobs),
        "ptt_assignment_row_count": len(ptt_rows),
        "ptt_readiness_audit_status": str(readiness["status"]),
        "ptt_readiness_execution_authority": str(
            readiness["execution_authority"]
        ),
        "network_tensor_status": str(
            source["formal_model"]["network_tensor_schema"]["status"]
        ),
        "training_executed": False,
        "ptt_evaluation_executed": False,
    }


def _validate_v2_installation(config_path: str | Path | None = None) -> dict[str, Any]:
    """包装继承结构检查但导出唯一 V2 身份 / Emit only a V2 validator identity."""

    paths = PipelinePaths.discover()
    if paths.pipeline_root.name != "final_pipeline_v2":
        raise ValueError("active package root is not final_pipeline_v2")
    inherited = validate_installation()
    payload: dict[str, Any] = {
        "schema_version": "ppg_frailty.installation_validation.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "passed",
        "canonical_boundaries": inherited["canonical_boundaries"],
        "spec_sha256": inherited["spec_sha256"],
        "module_count": len(list_modules()),
        "module_registry_hash": registry_sha256(),
        "profiles": _validate_profiles(),
    }
    if config_path is not None:
        candidate = Path(config_path)
        resolved = (
            candidate.expanduser().resolve()
            if candidate.is_absolute()
            else _resolve_pipeline_path(candidate)
        )
        if resolved.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("config path must be YAML")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        config = load_config(resolved)
        checked = validate_installation(config_path=resolved)
        try:
            displayed_path = resolved.relative_to(paths.pipeline_root).as_posix()
            path_scope = "pipeline_root"
        except ValueError:
            displayed_path = str(resolved)
            path_scope = "external_archive"
        payload["config"] = {
            "path": displayed_path,
            "path_scope": path_scope,
            "schema_version": config.schema_version,
            "config_id": config.config_id,
            "sha256": config.sha256,
        }
        payload["runtime_dependencies"] = dependency_availability_report(config)
        payload["preflight"] = checked["preflight"]
    return payload


def _inspect_legacy_config(config_path: str | Path) -> dict[str, Any]:
    """显式读取 V1 来源，不给予 formal eligibility / Inspect V1 provenance only."""

    paths = PipelinePaths.discover()
    resolved = _resolve_pipeline_path(config_path)
    config = load_config(resolved, allow_legacy=True)
    if not config.is_legacy:
        raise ValueError("--legacy-config accepts only a V1 provenance config")
    return {
        "schema_version": "ppg_frailty.legacy_config_inspection.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "passed",
        "config_id": config.config_id,
        "config_sha256": config.sha256,
        "source_path": resolved.relative_to(paths.pipeline_root).as_posix(),
        "formal_v2_eligible": False,
        "scientific_scope": "copied_historical_provenance_only",
    }


def _validate_all_configs() -> dict[str, Any]:
    """仅验证正式alias目标 / Validate formal alias targets, never other V2 YAML."""

    paths = PipelinePaths.discover()
    results: list[dict[str, Any]] = []
    formal_paths = sorted(set(V2_CONFIG_ALIASES.values()))
    for relative_path in formal_paths:
        path = paths.pipeline_root / relative_path
        try:
            payload = _validate_v2_installation(path)
            results.append({"config": path.name, "status": "passed", "config_identity": payload["config"], "preflight": payload["preflight"]})
        except Exception as error:  # noqa: BLE001 - retain one row per formal config.
            results.append({"config": path.name, "status": "failed", "error_type": type(error).__name__, "error": str(error)})
    if not results:
        raise ValueError("no formal V2 alias targets found")
    return {
        "schema_version": "ppg_frailty.config_validation_matrix.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "passed" if all(item["status"] == "passed" for item in results) else "failed",
        "profiles": _validate_profiles(),
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """分派命令并统一错误合同 / Dispatch commands with a uniform error contract."""

    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "list-modules":
            modules = list_modules(arguments.family)
            _print({"schema_version": "ppg_frailty.module_registry.v2", "pipeline_generation": "final_pipeline_v2", "registry_sha256": registry_sha256(), "count": len(modules), "modules": modules})
            return 0
        if arguments.command == "motion-validate":
            _print(_validate_motion_contract())
            return 0
        if arguments.command == "validate":
            if arguments.all_configs:
                payload = _validate_all_configs()
            elif arguments.profiles_only:
                payload = _validate_profiles()
            elif arguments.legacy_config:
                payload = _inspect_legacy_config(arguments.legacy_config)
            else:
                payload = _validate_v2_installation(arguments.config)
            _print(payload)
            return 0 if payload["status"] == "passed" else 1
        if arguments.command == "smoke":
            validation = _validate_v2_installation(arguments.config)
            payload = {
                "schema_version": "ppg_frailty.contract_smoke.v2",
                "pipeline_generation": "final_pipeline_v2",
                "status": "passed",
                "config": validation["config"],
                "preflight": validation["preflight"],
                "ablations_executed": False,
                "training_executed": False,
                "scientific_metrics_emitted": False,
                "scope": "config_manifest_fold_module_preflight_only",
            }
            _print(payload)
            return 0
        if arguments.command in {"motion-train-internal", "motion-evaluate-ptt"}:
            paths = PipelinePaths.discover()
            repository_root = Path(arguments.repository_root).resolve()
            if repository_root != paths.repository_root.resolve():
                raise ValueError("--repository-root must equal the discovered project root")
            output_dir = paths.output_path(arguments.output_dir)
            if arguments.command == "motion-train-internal":
                from .quality.motion_reference import run_formal_internal_motion_reference

                payload = asdict(
                    run_formal_internal_motion_reference(
                        repository_root,
                        output_dir=output_dir,
                        training_device=arguments.device,
                    )
                )
            else:
                from .quality.motion_reference import run_formal_ptt_motion_reference

                if (arguments.unit_evidence is None) != (
                    arguments.unit_evidence_sha256 is None
                ):
                    raise ValueError(
                        "--unit-evidence and --unit-evidence-sha256 must be supplied together"
                    )
                payload = asdict(
                    run_formal_ptt_motion_reference(
                        repository_root,
                        internal_evidence_path=paths.output_path(
                            arguments.internal_evidence
                        ),
                        expected_internal_evidence_sha256=(
                            arguments.internal_evidence_sha256
                        ),
                        output_dir=output_dir,
                        unit_evidence_path=(
                            None
                            if arguments.unit_evidence is None
                            else paths.output_path(arguments.unit_evidence)
                        ),
                        expected_unit_evidence_sha256=(
                            arguments.unit_evidence_sha256
                        ),
                    )
                )
            _print(payload)
            return 0
        if arguments.command == "test":
            return _run_tests(arguments)
        if arguments.command == "build-data":
            return _build_data(arguments)
        if arguments.command == "final-policy":
            from .experiment import final_refit_policy

            config = load_config(_resolve_formal_config_path(arguments.config))
            _print(final_refit_policy(config))
            return 0
        if arguments.command == "catalog":
            _print(_catalog_summary(arguments.line))
            return 0
        if arguments.command == "build-catalog":
            return _build_catalog(arguments)
        if arguments.command == "run":
            result = run_pipeline(arguments.config, mode=arguments.mode, output=arguments.output)
            _print(asdict(result))
            return 0
        if arguments.command == "run-experiment":
            # English: Import the heavy runner only for this command. The CLI does
            # not expose record/epoch caps, so formal full execution cannot inherit
            # reduced-smoke overrides. 中文：仅在该命令延迟导入重型 runner；CLI
            # 不暴露记录/epoch 裁剪参数，正式 full 因而无法继承 smoke override。
            from .experiment import (
                run_full_experiment,
                run_outer_cell,
                run_reduced_fold_experiment,
            )

            if (arguments.repeat is None) != (arguments.fold is None):
                raise ValueError("--repeat and --fold must be supplied together")
            progress = _TerminalProgress()
            if arguments.budget == "reduced-smoke":
                if arguments.measure_operational_costs:
                    raise ValueError(
                        "--measure-operational-costs is accepted only for full formal runs"
                    )
                result = run_reduced_fold_experiment(
                    arguments.config,
                    repeat_index=0 if arguments.repeat is None else arguments.repeat,
                    fold_index=0 if arguments.fold is None else arguments.fold,
                    output_dir=arguments.output_dir,
                    progress_callback=progress,
                )
            elif arguments.repeat is not None:
                result = run_outer_cell(
                    arguments.config,
                    arguments.repeat,
                    arguments.fold,
                    arguments.output_dir,
                    progress_callback=progress,
                    measure_operational_costs=bool(
                        arguments.measure_operational_costs
                    ),
                )
            else:
                result = run_full_experiment(
                    arguments.config,
                    output_dir=arguments.output_dir,
                    measure_operational_costs=bool(
                        arguments.measure_operational_costs
                    ),
                    progress_callback=progress,
                )
            payload = result.to_dict()
            _print(payload)
            return 0 if payload["status"] == "passed" else 1
        if arguments.command == "comparison-archive":
            from .experiment import build_comparison_archive_from_run_directories

            paths = PipelinePaths.discover()
            run_directories: dict[str, Path] = {}
            for value in arguments.run:
                config_id, separator, raw_path = str(value).partition("=")
                if not separator or not config_id.strip() or not raw_path.strip():
                    raise ValueError("--run must use CONFIG_ID=RUN_DIRECTORY")
                if config_id in run_directories:
                    raise ValueError(f"duplicate --run config ID: {config_id}")
                run_directories[config_id] = paths.output_path(raw_path)
            payload = build_comparison_archive_from_run_directories(
                run_directories,
                reference_config_id=arguments.reference_config_id,
                comparison_family=arguments.comparison_family,
                comparison_id=arguments.comparison_id,
                run_id=arguments.run_id,
                output_root=paths.output_path(arguments.output_root),
                n_bootstrap_resamples=arguments.bootstrap_resamples,
                n_permutation_resamples=arguments.permutation_resamples,
                statistics_seed=arguments.statistics_seed,
                allowed_authority_differences=(
                    arguments.allowed_authority_difference
                ),
            )
            payload["output_directory"] = (
                Path(payload["output_directory"])
                .resolve()
                .relative_to(paths.pipeline_root.resolve())
                .as_posix()
            )
            _print(payload)
            return 0
        if arguments.command == "record-selection":
            from .experiment import write_manual_selection_record

            paths = PipelinePaths.discover()
            payload = write_manual_selection_record(
                paths.output_path(arguments.comparison_archive),
                config_id=arguments.config_id,
                purpose=arguments.purpose,
                human_rationale=arguments.rationale,
                output_path=paths.new_artifact_path(arguments.output),
            )
            payload["selection_record_file"] = str(arguments.output)
            _print(payload)
            return 0
        if arguments.command == "final-refit":
            from .experiment import (
                execute_final_refit_from_verified_artifacts,
                final_refit_preflight_from_verified_artifacts,
            )

            paths = PipelinePaths.discover()
            inputs = {
                "run_directory": paths.output_path(arguments.run_directory),
                "selection_record": paths.output_path(arguments.selection_record),
                "comparison_archive": paths.output_path(
                    arguments.comparison_archive
                ),
                "config_path": _resolve_formal_config_path(arguments.config),
            }
            if arguments.bundle_directory is None:
                payload = final_refit_preflight_from_verified_artifacts(**inputs)
            else:
                bundle = execute_final_refit_from_verified_artifacts(
                    **inputs,
                    bundle_directory=arguments.bundle_directory,
                )
                payload = {
                    "schema_version": "ppg_frailty.final_refit_cli.v2",
                    "pipeline_generation": "final_pipeline_v2",
                    "status": "final_refit_bundle_saved",
                    "bundle_directory": str(bundle),
                    "training_executed": True,
                }
            _print(payload)
            return 0
        if arguments.command == "materialize-ablation-config":
            paths = PipelinePaths.discover()
            materialized = materialize_formal_ablation_config(
                _resolve_formal_config_path(arguments.base_config),
                family=arguments.family,
                profile_id=arguments.profile_id,
                output_path=paths.new_artifact_path(arguments.output),
                profiles_path=paths.pipeline_root / "configs/formal_ablation_profiles_v2.yaml",
            )
            _print(
                {
                    "schema_version": "ppg_frailty.formal_ablation_config_materialization.v2",
                    "pipeline_generation": "final_pipeline_v2",
                    "status": "materialized_not_run",
                    "config_id": materialized.config_id,
                    "config_sha256": materialized.sha256,
                    "output": paths.output_path(arguments.output).relative_to(
                        paths.pipeline_root
                    ).as_posix(),
                    "training_executed": False,
                    "ablation_executed": False,
                }
            )
            return 0
        if arguments.command == "compare":
            if arguments.comparison == "artifacts":
                payload = run_artifact_comparison(
                    arguments.reducers,
                    duration_s=arguments.duration_s,
                    seed=arguments.seed,
                    min_observation_sec=arguments.min_observation_sec,
                    min_peaks=arguments.min_peaks,
                )
            elif arguments.comparison == "models":
                payload = run_model_comparison(
                    arguments.models,
                    seed=arguments.seed,
                    ensemble_size=arguments.ensemble_size,
                )
            elif arguments.comparison == "prv-backends":
                from importlib.metadata import PackageNotFoundError, version
                from .features.prv_backend_compare import run_prv_backend_comparison

                requested = tuple(arguments.backends)
                if "aura_hrv_analysis" in requested:
                    try:
                        observed_hrv_version = version("hrv-analysis")
                    except PackageNotFoundError as exc:
                        raise RuntimeError(
                            "aura_hrv_analysis requires hrv-analysis==1.0.2"
                        ) from exc
                    if observed_hrv_version != "1.0.2":
                        raise RuntimeError(
                            "aura_hrv_analysis requires hrv-analysis==1.0.2; "
                            f"found {observed_hrv_version}"
                        )
                payload = run_prv_backend_comparison(
                    backends=requested,
                    fixture_ids=(
                        None
                        if arguments.fixtures is None
                        else tuple(arguments.fixtures)
                    ),
                )
                payload["execution_environment"] = {
                    "status": "ordinary_runtime",
                    "hrv_analysis_required_version": (
                        "1.0.2" if "aura_hrv_analysis" in requested else None
                    ),
                    "dependency_check": (
                        "requested_backend_package_version_only"
                    ),
                }
            else:
                payload = run_imu_gravity_comparison(
                    duration_s=arguments.duration_s,
                    seed=arguments.seed,
                )
            if arguments.output:
                payload["output"] = _write_new_artifact_json(
                    payload,
                    arguments.output,
                ).relative_to(PipelinePaths.discover().pipeline_root).as_posix()
            _print(payload)
            return 0 if payload.get("status") != "failed" else 1

        if arguments.command == "ablate":
            payload = run_ablation(arguments.factor, seed=arguments.seed)
            if arguments.output:
                payload["output"] = _write_new_artifact_json(
                    payload,
                    arguments.output,
                ).relative_to(PipelinePaths.discover().pipeline_root).as_posix()
            _print(payload)
            return 0
        raise RuntimeError("unreachable command")
    except Exception as error:  # noqa: BLE001 - CLI emits a machine-readable failure.
        _print(
            {
                "schema_version": "ppg_frailty.cli_error.v2",
                "status": "failed",
                "command": arguments.command,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            stream=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

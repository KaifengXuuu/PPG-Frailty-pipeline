#!/usr/bin/env python3
"""V1 CPU-only continuous-integration gate / V1 纯 CPU 连续集成门禁。

中文：该入口在隔离临时工作目录中执行全套标准库测试、全包导入、公共 CLI、
真实冻结记录 smoke、真实 motion feature-vector r0/f0 reduced 实验、所有已注册
reducer/model 的 synthetic 对照、5/10 秒窗口消融，最后运行 strict acceptance。
所有 warning 在测试阶段升级为错误；任何阶段失败仍写完整机器报告。

English: this entry point runs the full standard-library suite, complete package
imports, public CLI, one real frozen-record smoke, one real motion feature-vector
reduced r0/f0 experiment, every registered reducer/model synthetic comparison, the
5/10-second window ablation, and strict acceptance from an isolated working directory.
Test warnings are errors. The real experiment remains smoke-only, not a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PIPELINE_ROOT / "src"
ACCEPTANCE_ROOT = PIPELINE_ROOT / "artifacts/acceptance"
RUN_ROOT = ACCEPTANCE_ROOT / "runs"

ALL_REDUCER_ALIASES = ("identity", "nlms", "ssa", "spectral", "pca", "ica", "nmf")
ALL_MODEL_IDS = (
    "CompactCNN1D",
    "InceptionTimeFull",
    "InceptionTimeSmall",
    "InceptionTimeMatrix",
    "InceptionTimeFiveMemberEnsemble",
    "ROCKET",
    "MiniROCKET",
    "LogisticRegressionL2",
    "RBFSVM",
    "ExtraTrees",
    "ShapeFormerEffectSize",
    "FileBagFusionCompact",
    "FileBagFusionInception",
)


@dataclass(frozen=True)
class StageResult:
    """一个子进程阶段的精简证据 / Compact evidence for one subprocess stage."""

    stage_id: str
    status: str
    returncode: int
    duration_s: float
    command_role: str
    stdout_summary: Mapping[str, Any]
    stderr_tail: str


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """原子写 strict JSON / Atomically write strict JSON without NaN."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_last_json(text: str) -> dict[str, Any]:
    """解析 stdout 最后一个 JSON 行 / Parse the last JSON object printed."""

    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _summarize_stdout(payload: Mapping[str, Any]) -> dict[str, Any]:
    """保存足够身份而不复制大矩阵 / Retain identity without duplicating large rows."""

    keys = (
        "schema_version",
        "status",
        "suite",
        "counts",
        "count",
        "factor",
        "case_count",
        "constructed_case_count",
        "forward_case_count",
        "scientific_scope",
        "scientific_metrics_emitted",
        "output",
        "output_dir",
        "config_id",
        "repeat_indices",
        "fold_indices",
    )
    summary = {key: payload[key] for key in keys if key in payload}
    if isinstance(payload.get("results"), list):
        summary["result_count"] = len(payload["results"])
        identities = []
        for row in payload["results"]:
            if not isinstance(row, Mapping):
                continue
            for key in ("reducer_id", "model_id", "config", "dl_fs_hz", "window_s"):
                if key in row:
                    identities.append(row[key])
                    break
        if identities:
            summary["result_identities"] = identities
    if isinstance(payload.get("preflight"), Mapping):
        preflight = payload["preflight"]
        summary["preflight"] = {
            key: preflight.get(key)
            for key in ("config_id", "record_count", "participant_count", "split_count", "split_seeds")
        }
    return summary


def _run_stage(
    stage_id: str,
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_s: int = 300,
) -> StageResult:
    """执行且完整捕获一个阶段 / Execute and fully capture one stage."""

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=dict(environment),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        payload = _parse_last_json(completed.stdout)
        status = "passed" if completed.returncode == 0 else "failed"
        stderr_tail = completed.stderr[-4000:]
        returncode = int(completed.returncode)
    except subprocess.TimeoutExpired as error:
        payload = {"error": f"timeout_after_{timeout_s}_seconds"}
        status = "failed"
        stderr = error.stderr.decode("utf-8", errors="replace") if isinstance(error.stderr, bytes) else str(error.stderr or "")
        stderr_tail = stderr[-4000:]
        returncode = 124
    return StageResult(
        stage_id=stage_id,
        status=status,
        returncode=returncode,
        duration_s=round(time.perf_counter() - started, 9),
        command_role=stage_id,
        stdout_summary=_summarize_stdout(payload),
        stderr_tail=stderr_tail,
    )


def _environment() -> dict[str, str]:
    """构造 deterministic CPU 环境 / Build a deterministic CPU-only environment."""

    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONWARNINGS": "error",
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join((str(SOURCE_ROOT), str(PIPELINE_ROOT), existing)).rstrip(os.pathsep),
        }
    )
    return environment


def _package_versions() -> dict[str, str | None]:
    """记录 CPU 运行依赖版本 / Record CPU runtime dependency versions."""

    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for module_name, key in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sklearn", "scikit_learn"),
        ("onnxruntime", "onnxruntime"),
        ("torch", "torch"),
    ):
        try:
            module = __import__(module_name)
            versions[key] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[key] = None
    return versions


def _active_source_snapshot() -> dict[str, Any]:
    """散列活动源码/测试/配置 / Hash active source, tests, and configs."""

    paths: list[Path] = []
    for relative in ("src", "tools", "tests"):
        base = PIPELINE_ROOT / relative
        if base.is_dir():
            paths.extend(base.rglob("*.py"))
    config_root = PIPELINE_ROOT / "configs"
    if config_root.is_dir():
        paths.extend(config_root.glob("*.yaml"))
    for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt"):
        candidate = PIPELINE_ROOT / name
        if candidate.is_file():
            paths.append(candidate)
    rows = []
    for path in sorted(set(paths), key=lambda value: value.relative_to(PIPELINE_ROOT).as_posix()):
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(PIPELINE_ROOT).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": "ppg_frailty.active_source_snapshot.v1",
        "algorithm": "sha256(canonical_json(file_path,bytes,sha256))",
        "file_count": len(rows),
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": rows,
    }


def run_cpu_ci(*, include_quantitative: bool = True) -> dict[str, Any]:
    """运行完整 CPU gate 并返回报告 / Run the complete CPU gate and return a report."""

    ACCEPTANCE_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_parent = ACCEPTANCE_ROOT / "tmp"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    python = sys.executable
    environment = _environment()
    stages: list[StageResult] = []
    evidence: dict[str, str] = {}
    source_snapshot_path = ACCEPTANCE_ROOT / "source_snapshot_current.json"
    source_snapshot = _active_source_snapshot()
    _atomic_json(source_snapshot_path, source_snapshot)
    evidence["source_snapshot"] = source_snapshot_path.relative_to(PIPELINE_ROOT).as_posix()

    with tempfile.TemporaryDirectory(prefix=f"cpu_ci_{run_id}_", dir=temporary_parent) as directory:
        clean_cwd = Path(directory)
        test_report = ACCEPTANCE_ROOT / "cpu_ci_tests_current.json"
        stages.append(
            _run_stage(
                "all_tests_warnings_as_errors",
                (
                    python,
                    "-B",
                    str(PIPELINE_ROOT / "tools/run_test_suite.py"),
                    "--suite",
                    "all",
                    "--verbosity",
                    "1",
                    "--report",
                    str(test_report),
                ),
                cwd=clean_cwd,
                environment=environment,
            )
        )
        evidence["test_report"] = test_report.relative_to(PIPELINE_ROOT).as_posix()

        # 中文：逐个 import 所有包模块，发现可选依赖在 import-time 的隐藏耦合。
        # English: import every package module to expose hidden import-time coupling.
        import_code = (
            "import importlib,json,pkgutil,ppg_frailty;"
            "names=sorted(x.name for x in pkgutil.walk_packages(ppg_frailty.__path__,ppg_frailty.__name__+'.'));"
            "[importlib.import_module(x) for x in names];"
            "print(json.dumps({'schema_version':'ppg_frailty.import_sweep.v1','status':'passed','count':len(names)}))"
        )
        stages.append(
            _run_stage(
                "all_package_imports",
                (python, "-B", "-c", import_code),
                cwd=clean_cwd,
                environment=environment,
            )
        )

        cli = (python, "-B", "-m", "ppg_frailty.cli")
        stages.append(
            _run_stage(
                "cli_list_modules",
                cli + ("list-modules", "--family", "all"),
                cwd=clean_cwd,
                environment=environment,
            )
        )
        stages.append(
            _run_stage(
                "cli_validate_four_configs",
                cli + ("validate", "--all-configs"),
                cwd=clean_cwd,
                environment=environment,
            )
        )

        smoke_relative = f"artifacts/acceptance/runs/cli_smoke_{run_id}.json"
        stages.append(
            _run_stage(
                "cli_clean_temp_real_fold_smoke",
                cli
                + (
                    "run",
                    "--config",
                    "reference_static_v1",
                    "--mode",
                    "smoke",
                    "--output",
                    smoke_relative,
                ),
                cwd=clean_cwd,
                environment=environment,
            )
        )
        evidence["cli_smoke"] = smoke_relative

        # English: This is a real frozen-roster execution gate, but its scope stays
        # reduced smoke. It must never be reported as a 5x5 frailty benchmark.
        # 中文：这是保留冻结名单的真实执行门禁，但范围仍是 reduced smoke；
        # 绝不能把它报告成 5×5 frailty benchmark。
        experiment_relative = (
            f"artifacts/acceptance/runs/experiment_reduced_r0_f0_{run_id}"
        )
        stages.append(
            _run_stage(
                "cli_real_motion_feature_vector_reduced_r0_f0",
                cli
                + (
                    "run-experiment",
                    "--config",
                    "motion_benchmark_v1",
                    "--budget",
                    "reduced-smoke",
                    "--repeat",
                    "0",
                    "--fold",
                    "0",
                    "--output-dir",
                    experiment_relative,
                ),
                cwd=clean_cwd,
                environment=environment,
                timeout_s=600,
            )
        )
        evidence["real_reduced_experiment"] = experiment_relative

        if include_quantitative:
            artifact_relative = f"artifacts/acceptance/runs/artifact_parallel_{run_id}.json"
            stages.append(
                _run_stage(
                    "cli_all_artifact_comparison",
                    cli
                    + (
                        "compare",
                        "artifacts",
                        "--reducers",
                        *ALL_REDUCER_ALIASES,
                        "--duration-s",
                        "10",
                        "--seed",
                        "42",
                        "--output",
                        artifact_relative,
                    ),
                    cwd=clean_cwd,
                    environment=environment,
                )
            )
            evidence["artifact_parallel"] = artifact_relative

            model_relative = f"artifacts/acceptance/runs/model_parallel_{run_id}.json"
            stages.append(
                _run_stage(
                    "cli_all_model_comparison",
                    cli
                    + (
                        "compare",
                        "models",
                        "--models",
                        *ALL_MODEL_IDS,
                        "--seed",
                        "42",
                        "--output",
                        model_relative,
                    ),
                    cwd=clean_cwd,
                    environment=environment,
                )
            )
            evidence["model_parallel"] = model_relative

            window_relative = f"artifacts/acceptance/runs/raw_window_ablation_{run_id}.json"
            stages.append(
                _run_stage(
                    "cli_raw_window_ablation",
                    cli
                    + (
                        "ablate",
                        "--factor",
                        "raw_window_s",
                        "--seed",
                        "42",
                        "--output",
                        window_relative,
                    ),
                    cwd=clean_cwd,
                    environment=environment,
                )
            )
            evidence["raw_window_ablation"] = window_relative

        strict_report = ACCEPTANCE_ROOT / "strict_acceptance_current.json"
        stages.append(
            _run_stage(
                "strict_acceptance",
                (
                    python,
                    "-B",
                    str(PIPELINE_ROOT / "tools/acceptance_gate.py"),
                    "--write-report",
                    str(strict_report),
                ),
                cwd=clean_cwd,
                environment=environment,
            )
        )
        evidence["strict_acceptance"] = strict_report.relative_to(PIPELINE_ROOT).as_posix()

    failed = [row.stage_id for row in stages if row.status != "passed"]
    return {
        "schema_version": "ppg_frailty.cpu_ci.v1",
        "status": "passed" if not failed else "failed",
        "run_id": run_id,
        "cpu_only": True,
        "warnings_policy": "error",
        "clean_temporary_working_directory": True,
        "quantitative_contracts_included": include_quantitative,
        "environment": _package_versions(),
        "active_source_snapshot": {
            "report": evidence["source_snapshot"],
            "file_count": source_snapshot["file_count"],
            "tree_sha256": source_snapshot["tree_sha256"],
        },
        "stages": [
            {
                "stage_id": row.stage_id,
                "status": row.status,
                "returncode": row.returncode,
                "duration_s": row.duration_s,
                "command_role": row.command_role,
                "stdout_summary": dict(row.stdout_summary),
                "stderr_tail": row.stderr_tail,
            }
            for row in stages
        ],
        "failed_stages": failed,
        "evidence": evidence,
        "scientific_claim": (
            "cpu_acceptance_plus_real_reduced_smoke_and_synthetic_contracts_"
            "no_5x5_frailty_or_external_ptt_performance_claim"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI：运行并原子保存 current 报告 / Run and atomically save the current report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-quantitative",
        action="store_true",
        help="diagnostic only; strict acceptance will remain pending without current evidence",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ACCEPTANCE_ROOT / "cpu_ci_current.json",
    )
    arguments = parser.parse_args(argv)
    target = arguments.report.resolve()
    target.relative_to(PIPELINE_ROOT.resolve())
    payload = run_cpu_ci(include_quantitative=not arguments.skip_quantitative)
    _atomic_json(target, payload)
    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "status": payload["status"],
                "failed_stages": payload["failed_stages"],
                "report": target.relative_to(PIPELINE_ROOT).as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""V1 非交互命令行 / V1 non-interactive command line.

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
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

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


def _print(payload: Any, *, stream: Any = None) -> None:
    """打印单行 strict JSON / Print one-line strict JSON."""

    print(
        json.dumps(to_strict_json_value(payload), ensure_ascii=False, sort_keys=True, allow_nan=False),
        file=stream or sys.stdout,
    )


def _registered_config(value: str) -> str:
    """允许相对 V1 path 或注册 config stem / Accept a path or registered stem."""

    candidate = Path(value)
    if candidate.suffix:
        return value
    return f"configs/{value}.yaml"


def build_parser() -> argparse.ArgumentParser:
    """构建无隐藏行为的 parser / Build the explicit command parser."""

    parser = argparse.ArgumentParser(prog="ppg-frailty", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    modules = subcommands.add_parser("list-modules", help="list canonical modules")
    modules.add_argument("--family", choices=["all", "representation", "artifact", "model"], default="all")

    validate = subcommands.add_parser("validate", help="validate installation or one config")
    validate_group = validate.add_mutually_exclusive_group()
    validate_group.add_argument("--config", type=_registered_config)
    validate_group.add_argument("--all-configs", action="store_true")

    tests = subcommands.add_parser("test", help="run one registered unittest suite")
    tests.add_argument("--suite", choices=["all", "data", "signal", "artifacts", "features", "models", "training", "integration", "cli", "contracts"], default="all")
    tests.add_argument("--report")
    tests.add_argument("--verbosity", choices=[0, 1, 2], default=1, type=int)

    build_data = subcommands.add_parser("build-data", help="rebuild frozen data contracts from authorities")
    build_data.add_argument("--confirm-byte-rehash", action="store_true", help="required because all 261 raw files are re-hashed")

    run = subcommands.add_parser("run", help="run formal smoke or full input/protocol audit")
    run.add_argument("--config", required=True, type=_registered_config)
    run.add_argument("--mode", required=True, choices=["smoke", "full"])
    run.add_argument("--output", required=True, help="new JSON path below final_pipeline_v1")

    experiment = subcommands.add_parser(
        "run-experiment",
        help="train/evaluate a real frozen outer-fold experiment",
        description=(
            "Execute the scientific pipeline rather than the input-only audit. "
            "Current formal experiment execution supports feature_vector only. "
            "Raw, feature_matrix, and fusion remain runnable through compare/test, "
            "but their formal experiment request fails closed. "
            "Reduced smoke keeps the complete participant roster but uses the public "
            "60-second/one-record/one-epoch defaults. Full runs all 25 cells unless "
            "one repeat/fold pair is explicitly selected. 当前正式实验仅支持 "
            "feature_vector；其余三种表示可运行 comparison/test，但正式 runner "
            "会关闭失败。"
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
        help="new, non-existing experiment directory below final_pipeline_v1",
    )

    compare = subcommands.add_parser("compare", help="run a quantitative synthetic comparison")
    compare_commands = compare.add_subparsers(dest="comparison", required=True)
    artifacts = compare_commands.add_parser("artifacts")
    artifacts.add_argument(
        "--reducers",
        nargs="+",
        default=[
            "identity", "nlms_imu_anc", "ssa_decomposition", "spectral_mask",
            "pca_bss", "fastica_bss", "nmf_bss",
        ],
        help="canonical artifact module IDs; legacy short aliases are explicitly labelled",
    )
    artifacts.add_argument("--duration-s", type=float, default=10.0)
    artifacts.add_argument("--seed", type=int, default=42)
    artifacts.add_argument("--output")
    models = compare_commands.add_parser("models")
    models.add_argument(
        "--models",
        nargs="+",
        default=[
            "CompactCNN1D", "InceptionTimeFull", "InceptionTimeSmall",
            "InceptionTimeMatrix", "InceptionTimeFiveMemberEnsemble",
            "ROCKET", "MiniROCKET", "LogisticRegressionL2", "RBFSVM",
            "ExtraTrees", "ShapeFormerEffectSize", "FileBagFusionCompact",
            "FileBagFusionInception",
        ],
        help="canonical model IDs; the default executes all 13 registered models",
    )
    models.add_argument("--seed", type=int, default=42)
    models.add_argument("--output")
    gravity = compare_commands.add_parser("imu-gravity")
    gravity.add_argument("--duration-s", type=float, default=12.0)
    gravity.add_argument("--seed", type=int, default=42)
    gravity.add_argument("--output")

    ablate = subcommands.add_parser("ablate", help="run one-factor synthetic ablation")
    ablate.add_argument("--factor", required=True, choices=["artifact", "model", "dl_fs", "raw_window_s", "physical_time"])
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


def _validate_all_configs() -> dict[str, Any]:
    """逐 config 报告且保留全部失败 / Validate all configs without hiding failures."""

    paths = PipelinePaths.discover()
    results: list[dict[str, Any]] = []
    for path in sorted((paths.pipeline_root / "configs").glob("*.yaml")):
        try:
            payload = validate_installation(config_path=path)
            results.append({"config": path.name, "status": "passed", "preflight": payload["preflight"]})
        except Exception as error:  # noqa: BLE001 - every config receives an audit row.
            results.append({"config": path.name, "status": "failed", "error_type": type(error).__name__, "error": str(error)})
    return {
        "schema_version": "ppg_frailty.config_validation_matrix.v1",
        "status": "passed" if all(item["status"] == "passed" for item in results) else "failed",
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """分派命令并统一错误合同 / Dispatch commands with a uniform error contract."""

    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "list-modules":
            modules = list_modules(arguments.family)
            _print({"schema_version": "ppg_frailty.module_registry.v1", "registry_sha256": registry_sha256(), "count": len(modules), "modules": modules})
            return 0
        if arguments.command == "validate":
            payload = _validate_all_configs() if arguments.all_configs else validate_installation(config_path=arguments.config)
            _print(payload)
            return 0 if payload["status"] == "passed" else 1
        if arguments.command == "test":
            return _run_tests(arguments)
        if arguments.command == "build-data":
            return _build_data(arguments)
        if arguments.command == "run":
            result = run_pipeline(arguments.config, mode=arguments.mode, output=arguments.output)
            _print(asdict(result))
            return 0
        if arguments.command == "run-experiment":
            # English: Import the heavy runner only for this command. The CLI does
            # not expose record/epoch caps, so formal full execution cannot inherit
            # reduced-smoke overrides. 中文：仅在该命令延迟导入重型 runner；CLI
            # 不暴露记录/epoch 裁剪参数，正式 full 因而无法继承 smoke override。
            from .experiment import run_full_experiment, run_reduced_fold_experiment

            if (arguments.repeat is None) != (arguments.fold is None):
                raise ValueError("--repeat and --fold must be supplied together")
            if arguments.budget == "reduced-smoke":
                result = run_reduced_fold_experiment(
                    arguments.config,
                    repeat_index=0 if arguments.repeat is None else arguments.repeat,
                    fold_index=0 if arguments.fold is None else arguments.fold,
                    output_dir=arguments.output_dir,
                )
            else:
                full_options: dict[str, Any] = {"output_dir": arguments.output_dir}
                if arguments.repeat is not None:
                    full_options["repeats"] = (arguments.repeat,)
                    full_options["folds"] = (arguments.fold,)
                result = run_full_experiment(arguments.config, **full_options)
            payload = result.to_dict()
            _print(payload)
            return 0 if payload["status"] == "passed" else 1
        if arguments.command == "compare":
            if arguments.comparison == "artifacts":
                payload = run_artifact_comparison(arguments.reducers, duration_s=arguments.duration_s, seed=arguments.seed)
            elif arguments.comparison == "models":
                payload = run_model_comparison(arguments.models, seed=arguments.seed)
            else:
                payload = run_imu_gravity_comparison(duration_s=arguments.duration_s, seed=arguments.seed)
            if arguments.output:
                payload["output"] = write_quantitative_report(payload, arguments.output).relative_to(PipelinePaths.discover().pipeline_root).as_posix()
            _print(payload)
            return 0
        if arguments.command == "ablate":
            payload = run_ablation(arguments.factor, seed=arguments.seed)
            if arguments.output:
                payload["output"] = write_quantitative_report(payload, arguments.output).relative_to(PipelinePaths.discover().pipeline_root).as_posix()
            _print(payload)
            return 0
        raise RuntimeError("unreachable command")
    except Exception as error:  # noqa: BLE001 - CLI emits a machine-readable failure.
        _print(
            {
                "schema_version": "ppg_frailty.cli_error.v1",
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

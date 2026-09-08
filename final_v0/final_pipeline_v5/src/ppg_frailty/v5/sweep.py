"""YAML study-plan CLI using the same executor as :mod:`ppg_frailty.v5.cli`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from ..study import StudyRunner, load_study_plan
from .environment import DEFAULT_LOCK, evaluate_environment
from .output_contract import PIPELINE_OUTPUT_ROOT, V5_ROOT, export_pipeline_excel
from .run import data_only_plan, run_study
from .service import RefitOptions

def _source(value: str | Path) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (V5_ROOT / path).resolve()
    if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
        raise FileNotFoundError(f"study plan YAML not found: {path}")
    return path

def _environment_check(args: argparse.Namespace, plan: Any) -> Mapping[str, Any]:
    """Compatibility hook used by tests and embedded callers."""

    return evaluate_environment(
        args.environment_policy,
        device=str(plan.execution.device or "cuda"),
        lock_path=args.environment_lock,
    ).to_dict()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a data-only V5 study plan (run/comparison/repeat/fold).")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True)
        command.add_argument("--environment-policy", choices=("exact", "record"), default="exact")
        command.add_argument("--environment-lock", type=Path, default=DEFAULT_LOCK)
    run = commands.choices["run"]
    run.add_argument("--run-name")
    run.add_argument("--resume")
    run.add_argument("--hash-predictions", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--refit", action="store_true")
    excel = commands.add_parser("export-excel")
    excel.add_argument("--pipeline-output", required=True)
    excel.add_argument("--replace", action="store_true")
    return parser

def run_prepared_study(
    args: argparse.Namespace,
    *,
    source: Path,
    plan: Any,
    prepared_expansion: Any | None = None,
    initial_request_schema: str = "ppg_frailty.v5_sweep_request.v1",
    request_metadata: Mapping[str, Any] | None = None,
    runner_executor: Any | None = None,
) -> int:
    """Run a parsed YAML/Dashboard plan through the one V5 execution service."""

    lock = Path(args.environment_lock)
    lock = lock.resolve() if lock.is_absolute() else (V5_ROOT / lock).resolve()
    resume = None if args.resume is None else Path(args.resume)
    if resume is not None and not resume.is_absolute():
        resume = (V5_ROOT / resume).resolve()
    result = run_study(
        plan,
        pipeline_root=V5_ROOT,
        source=source,
        output_root=PIPELINE_OUTPUT_ROOT,
        run_name=args.run_name,
        resume=resume,
        environment_policy=args.environment_policy,
        environment_lock=lock,
        hash_predictions=bool(args.hash_predictions),
        refit=RefitOptions(enabled=bool(args.refit)),
        dry_run=bool(args.dry_run),
        request_schema=initial_request_schema,
        request_metadata=request_metadata,
        prepared_expansion=prepared_expansion,
        runner_executor=runner_executor,
        environment_hook=lambda candidate: _environment_check(args, candidate),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return int(result.get("exit_code", 0))

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "export-excel":
            result = export_pipeline_excel(args.pipeline_output, replace=args.replace)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        source = _source(args.plan)
        plan = data_only_plan(load_study_plan(source), PIPELINE_OUTPUT_ROOT)
        if args.command == "validate":
            runner = StudyRunner(pipeline_root=V5_ROOT, output_layout="v5")
            expansion = runner.expand(plan)
            result = {
                "status": "valid",
                "source_yaml": str(source),
                "environment_check": _environment_check(args, plan),
                "study": plan.to_dict(),
                "cases": [case.to_dict() for case in expansion.cases],
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        return run_prepared_study(args, source=source, plan=plan)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"status": "error", "error": f"{type(error).__name__}: {error}"}))
        return 2


__all__ = ["build_parser", "main", "run_prepared_study"]

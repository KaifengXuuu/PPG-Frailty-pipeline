"""Adapters for specialized V2 studies.

Computation writes data below pipeline_output. Presentation is delegated to the
specialized reporting module and writes below report_output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import yaml

from ..evaluate.decision_bias_oracle import (
    SCHEMA_VERSION as DECISION_ORACLE_SCHEMA,
    load_decision_bias_oracle_plan,
    run_decision_bias_oracle,
)
from ..evaluate.role_scope_decomposition import (
    SCHEMA_VERSION as ROLE_SCOPE_SCHEMA,
    load_role_scope_plan,
    run_role_scope_decomposition,
)
from ..quality.stage5_pre import (
    PEAK_ABLATION_SCHEMA,
    RESULT_SCHEMA as MOTION_PEAK_RESULT_SCHEMA,
    STAGE5_SCHEMA,
    load_motion_peak_plan,
    run_motion_peak_study,
)
from ..study import StudyRunner, TerminalProgressSink
from ..study.hyperparameter import (
    complete_successive_halving_study,
    inspect_successive_halving_completion,
    load_hyperparameter_plan,
    run_hyperparameter_study,
)
from .output_contract import (
    PIPELINE_OUTPUT_ROOT,
    REPORT_OUTPUT_ROOT,
    V5_ROOT,
    automatic_run_name,
    safe_output_name,
    try_export_pipeline_excel,
)
from .service import post_run_finalize
from .specialized_outputs import export_motion_model_config, export_specialized_data_excel


HYPERPARAMETER_SCHEMA = "ppg_frailty.hyperparameter_study_plan.v1"
COMPUTATION_SCHEMAS = frozenset({STAGE5_SCHEMA, PEAK_ABLATION_SCHEMA, HYPERPARAMETER_SCHEMA})
ANALYSIS_SCHEMAS = frozenset({DECISION_ORACLE_SCHEMA, ROLE_SCOPE_SCHEMA})
SUPPORTED_SCHEMAS = COMPUTATION_SCHEMAS | ANALYSIS_SCHEMAS

def _mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a JSON object: {path}")
    return dict(value)

def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

def _schema(path: str | Path) -> tuple[Path, str]:
    raw = Path(path).expanduser()
    source = (V5_ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"specialized YAML must contain a mapping: {source}")
    schema = str(payload.get("schema_version", ""))
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported specialized schema: {schema!r}")
    return source, schema

def _output_path(plan: Path, name: str | None, resume: str | Path | None) -> Path:
    if resume is not None:
        return Path(resume).expanduser().resolve()
    PIPELINE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    target = PIPELINE_OUTPUT_ROOT / (safe_output_name(name, label="run name") if name else automatic_run_name(plan))
    target.mkdir()
    return target

def validate_specialized_plan(path: str | Path, *, source_root: str | Path = V5_ROOT) -> Mapping[str, Any]:
    """Validate a plan with its native loader."""

    source, schema = _schema(path)
    root = Path(source_root).expanduser().resolve()
    if schema == DECISION_ORACLE_SCHEMA:
        plan = load_decision_bias_oracle_plan(source, pipeline_root=root)
        details = {"study_id": plan.study_id, "case_id": plan.case_id}
    elif schema == ROLE_SCOPE_SCHEMA:
        plan = load_role_scope_plan(source, pipeline_root=root)
        details = {"study_id": plan.study_id, "metrics": list(plan.metrics)}
    elif schema in {STAGE5_SCHEMA, PEAK_ABLATION_SCHEMA}:
        plan = load_motion_peak_plan(source)
        details = {"study_id": plan.study_id, "study_type": plan.study_type}
    else:
        plan = load_hyperparameter_plan(source)
        details = {
            "study_id": plan["study"]["study_id"],
            "study_type": plan["study"]["study_type"],
            "candidate_count": len(plan["candidates"]),
        }
    return {
        "status": "valid",
        "schema_version": schema,
        "workflow_kind": "analysis_only" if schema in ANALYSIS_SCHEMAS else "computation",
        **details,
    }

def _specialized_phase_runner_factory(*, source_yaml: Path, sink: TerminalProgressSink, **_: Any) -> Any:
    """Use the ordinary V5 layout for every hyperparameter phase."""

    del source_yaml

    def factory(plan: Any, phase_id: str, resume_directory: Path | None) -> StudyRunner:
        del plan, phase_id, resume_directory
        return StudyRunner(
            pipeline_root=V5_ROOT,
            progress_sink=sink,
            output_layout="v5",
        )

    return factory

def _hyperparameter_phases(output: Path) -> list[tuple[str, Path]]:
    manifest = _mapping(output / "study_manifest.json")
    phases = manifest.get("phase_directories", {})
    if not isinstance(phases, Mapping):
        raise TypeError("hyperparameter manifest lacks phase directories")
    return [(str(name), output / str(relative)) for name, relative in phases.items()]

def _publish_specialized_artifact_contract(output: Path, schema: str, **_: Any) -> Mapping[str, Any]:
    """Create the common data/Excel/model handoff for a specialized run."""

    excel = export_specialized_data_excel(output, replace=(output / "tables/pipeline_data.xlsx").exists())
    payload: dict[str, Any] = {
        "schema_version": "ppg_frailty.v5_specialized_artifacts.v1",
        "status": "complete",
        "source_schema": schema,
        "pipeline_excel": excel.get("workbook"),
        "public_phase_runs": [],
    }
    if schema == STAGE5_SCHEMA:
        exported = export_motion_model_config(output)
        payload.update(
            model_trained=True,
            model_kind="formal_motion_cnn",
            per_fold_learned_weights="complete",
            model_config_export=exported.get("output_directory"),
        )
    elif schema == PEAK_ABLATION_SCHEMA:
        payload.update(model_trained=False, model_kind="not_applicable")
    else:
        phases = []
        for name, phase in _hyperparameter_phases(output):
            finalized = post_run_finalize(
                phase,
                pipeline_root=V5_ROOT,
                hash_prediction_files=False,
                export_configuration=True,
            )
            phase_excel = try_export_pipeline_excel(
                phase,
                allow_legacy_location=True,
                replace=(phase / "tables/pipeline_data.xlsx").exists(),
            )
            phases.append(
                {
                    "phase": name,
                    "pipeline_output": phase.relative_to(V5_ROOT).as_posix(),
                    "pipeline_excel": phase_excel.get("workbook"),
                    "model_config_export": (finalized.get("model_config_export") or {}).get("output_directory"),
                }
            )
        payload.update(
            model_trained=True,
            model_kind="frailty_classifier",
            public_phase_runs=phases,
        )
    _write_json(output / "v5_specialized_artifacts.json", payload)
    return payload

def run_specialized_computation(
    plan_path: str | Path,
    *,
    run_name: str | None = None,
    resume: str | Path | None = None,
    source_root: str | Path = V5_ROOT,
    upstream_study: str | Path | None = None,
    device: str | None = None,
    jobs: int | None = None,
    include_denoiser: bool = True,
    **_: Any,
) -> Path:
    """Run one specialized computation without creating presentation files."""

    source, schema = _schema(plan_path)
    if schema not in COMPUTATION_SCHEMAS:
        raise ValueError("analysis-only plans belong to analyse_report.py")
    if run_name and resume:
        raise ValueError("run_name cannot be combined with resume")
    sink = TerminalProgressSink()
    try:
        if schema in {STAGE5_SCHEMA, PEAK_ABLATION_SCHEMA}:
            if upstream_study is not None or jobs is not None:
                raise ValueError("upstream-study/jobs apply only to hyperparameter plans")
            if schema == PEAK_ABLATION_SCHEMA and (device or not include_denoiser):
                raise ValueError("device/no-denoiser apply only to Stage5-pre")
            output = _output_path(source, run_name, resume)
            output = run_motion_peak_study(
                source,
                pipeline_root=Path(source_root).resolve(),
                output_root=PIPELINE_OUTPUT_ROOT,
                resume=output,
                progress_sink=sink,
                device=device,
                include_denoiser=include_denoiser,
            )
        else:
            if not include_denoiser:
                raise ValueError("no-denoiser applies only to Stage5")
            factory = _specialized_phase_runner_factory(source_yaml=source, sink=sink)
            output = run_hyperparameter_study(
                source,
                pipeline_root=V5_ROOT,
                upstream_study=upstream_study,
                output_root=PIPELINE_OUTPUT_ROOT,
                device=device,
                jobs=jobs,
                progress_sink=sink,
                run_name=None if resume else run_name or automatic_run_name(source),
                resume=resume,
                phase_runner_factory=factory,
            )
    finally:
        sink.close()
    _publish_specialized_artifact_contract(Path(output), schema)
    return Path(output)

def complete_specialized_halving(
    study_dir: str | Path, *, device: str | None = None, jobs: int | None = None, **_: Any
) -> Path:
    """Train only the declared unpromoted full-CV candidates."""

    output = Path(study_dir).expanduser().resolve()
    if inspect_successive_halving_completion(output)["status"] != "already_complete":
        sink = TerminalProgressSink()
        try:
            output = complete_successive_halving_study(
                output,
                pipeline_root=V5_ROOT,
                device=device,
                jobs=jobs,
                progress_sink=sink,
                phase_runner_factory=_specialized_phase_runner_factory(
                    source_yaml=output / "study_plan.yaml", sink=sink
                ),
            )
        finally:
            sink.close()
    _publish_specialized_artifact_contract(output, HYPERPARAMETER_SCHEMA)
    return output

def _report_target(name: str) -> Path:
    REPORT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    target = REPORT_OUTPUT_ROOT / safe_output_name(name, label="report output name")
    if target.exists():
        raise FileExistsError(target)
    return target

def run_specialized_analysis(
    plan_path: str | Path,
    *,
    source_root: str | Path = V5_ROOT,
    output_name: str | None = None,
    source_study_dir: str | Path | None = None,
    case_id: str | None = None,
    prediction_file: str | Path | None = None,
    step: float | None = None,
) -> Path:
    """Run a decision oracle or role-scope analysis under report_output."""

    source, schema = _schema(plan_path)
    root = Path(source_root).expanduser().resolve()
    if schema == DECISION_ORACLE_SCHEMA:
        plan = load_decision_bias_oracle_plan(source, pipeline_root=root)
        runner = run_decision_bias_oracle
        overrides = dict(
            source_study_dir=source_study_dir,
            case_id=case_id,
            prediction_file=prediction_file,
            step=step,
        )
    elif schema == ROLE_SCOPE_SCHEMA:
        plan = load_role_scope_plan(source, pipeline_root=root)
        runner = run_role_scope_decomposition
        overrides = {}
    else:
        raise ValueError("computation plans belong to specialized_pipeline.py")
    target = _report_target(output_name or plan.study_id)
    with tempfile.TemporaryDirectory(dir=REPORT_OUTPUT_ROOT) as temporary:
        generated = runner(source, pipeline_root=root, output_root=temporary, **overrides)
        os.replace(generated, target)
    return target

def rebuild_specialized_report(study_dir: str | Path, *, output_name: str | None = None) -> Path:
    """Build a report directly from immutable pipeline artifacts."""

    from ..reporting.specialized import (
        generate_motion_peak_report,
        rebuild_hyperparameter_report,
    )

    source = Path(study_dir).expanduser().resolve()
    manifest = _mapping(source / "study_manifest.json")
    target = _report_target(output_name or source.name)
    if manifest.get("schema_version") == MOTION_PEAK_RESULT_SCHEMA:
        generate_motion_peak_report(source, output_dir=target)
    elif manifest.get("schema_version") == "ppg_frailty.hyperparameter_study_manifest.v1":
        rebuild_hyperparameter_report(source, output_dir=target)
    else:
        raise ValueError("unsupported specialized result")
    return target

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="specialized_pipeline.py")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--source-root", default=str(V5_ROOT))
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--run-name")
    run.add_argument("--resume")
    run.add_argument("--source-root", default=str(V5_ROOT))
    run.add_argument("--upstream-study")
    run.add_argument("--device")
    run.add_argument("--jobs", type=int)
    run.add_argument("--no-denoiser", action="store_true")
    complete = commands.add_parser("complete")
    complete.add_argument("--study-dir", required=True)
    complete.add_argument("--device")
    complete.add_argument("--jobs", type=int)
    complete.add_argument("--dry-run", action="store_true")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            result: Any = validate_specialized_plan(args.plan, source_root=args.source_root)
        elif args.command == "complete" and args.dry_run:
            result = inspect_successive_halving_completion(args.study_dir)
        elif args.command == "complete":
            result = {
                "status": "complete",
                "pipeline_output": str(
                    complete_specialized_halving(args.study_dir, device=args.device, jobs=args.jobs)
                ),
            }
        else:
            result = {
                "status": "complete",
                "pipeline_output": str(
                    run_specialized_computation(
                        args.plan,
                        run_name=args.run_name,
                        resume=args.resume,
                        source_root=args.source_root,
                        upstream_study=args.upstream_study,
                        device=args.device,
                        jobs=args.jobs,
                        include_denoiser=not args.no_denoiser,
                    )
                ),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2


__all__ = [
    "ANALYSIS_SCHEMAS",
    "COMPUTATION_SCHEMAS",
    "SUPPORTED_SCHEMAS",
    "build_parser",
    "complete_specialized_halving",
    "main",
    "rebuild_specialized_report",
    "run_specialized_analysis",
    "run_specialized_computation",
    "validate_specialized_plan",
]

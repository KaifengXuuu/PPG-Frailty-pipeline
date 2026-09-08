"""Command-line interface for V5's data-only training and inference workflow."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shlex
from typing import Any, Mapping

import yaml

from ..config import canonical_json_bytes
from ..module_registry import list_modules
from ..pipeline import PipelinePaths, preflight_pipeline, validate_installation
from ..study import (
    AxisSpec,
    ExecutionSpec,
    OutputSpec,
    PreprocessingCacheSpec,
    ReportSpec,
    StudyInfo,
    StudyPlan,
    load_study_plan,
)
from .configuration import (
    PRESETS,
    manual_cli_command,
    parameter_rows,
    parse_assignment,
    parse_yaml_value,
    preset_rows,
    resolve_configuration,
)
from .environment import DEFAULT_LOCK, evaluate_environment
from .model_config_export import export_model_config
from .output_contract import (
    PIPELINE_OUTPUT_ROOT,
    automatic_run_name,
    export_pipeline_excel,
    safe_output_name,
)
from .results import build_study_data_index
from .run import data_only_plan, run_study
from .service import RefitOptions


PIPELINE_ROOT = Path(__file__).resolve().parents[3]

class _CatalogHelp(argparse.Action):
    """Render large live catalogs only when their subcommand asks for help."""

    def __init__(self, *args: Any, catalog: str, **kwargs: Any) -> None:
        self.catalog = catalog
        super().__init__(*args, nargs=0, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        del namespace, values, option_string
        parser.print_help()
        if self.catalog == "modules":
            print("\nFAMILY\tMODULE_ID\tSTATUS")
            for row in list_modules():
                print(f"{row['family']}\t{row['module_id']}\t{row.get('scientific_status', '')}")
        else:
            print(
                "\nFinalcase leaves are shown below. For the cross-preset/study union run "
                "`python pipeline.py parameters --source-preset all`."
            )
            print("\nPATH\tTYPE\tRANGE\tCLI INPUT")
            for row in parameter_rows(PIPELINE_ROOT, source_preset="finalcase"):
                print(f"{row['path']}\t{row['type']}\t{row['range']}\t{row['input']}")
        parser.exit()

def _path(value: str | Path, *, must_exist: bool = False) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (PIPELINE_ROOT / path).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path

def _indices(value: str) -> tuple[int, ...]:
    if value.lower() == "all":
        return tuple(range(5))
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use all or comma-separated indices 0..4") from error
    if not result or len(set(result)) != len(result) or not set(result) <= set(range(5)):
        raise argparse.ArgumentTypeError("indices must be a unique subset of 0..4")
    return result

def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values

def _vary(value: str) -> tuple[str, tuple[Any, ...]]:
    path, values = parse_assignment(value)
    if not isinstance(values, list) or len(values) < 2:
        raise argparse.ArgumentTypeError("--vary requires PATH='[value1,value2,...]'")
    return path, tuple(values)

def _config_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=tuple(PRESETS))
    source.add_argument("--config", help="Complete pipeline YAML.")
    source.add_argument("--manual", action="store_true", help="Preset-free --set/--module mode.")
    parser.add_argument("--module", action="append", default=[], metavar="FAMILY=MODULE_ID")
    parser.add_argument("--set", dest="assignments", action="append", default=[], metavar="PATH=YAML")
    parser.add_argument("--unset", action="append", default=[], metavar="PATH")
    parser.add_argument("--config-id")

def _execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repeats", type=_indices)
    parser.add_argument("--folds", type=_indices)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--measure-operational-costs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--preprocessing-cache-mode", choices=("off", "read_only", "read_write"))
    parser.add_argument("--preprocessing-cache-root")
    parser.add_argument("--preprocessing-cache-namespaces", type=_csv)
    parser.add_argument("--output-root", default=str(PIPELINE_OUTPUT_ROOT))
    parser.add_argument("--run-name")
    parser.add_argument("--resume")
    parser.add_argument("--hash-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--environment-policy", choices=("exact", "record"), default="exact")
    parser.add_argument("--environment-lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--refit", action="store_true")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V5 data-only PPG frailty pipeline; figures belong to analyse_report.py.",
        epilog="Use 'parameters --source-preset all' and 'modules' for the complete CLI surface.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "Run one configuration."),
        ("ablation", "Run one single-factor comparison."),
        ("grid", "Run a Cartesian comparison."),
    ):
        command = commands.add_parser(name, help=help_text)
        _config_arguments(command)
        _execution_arguments(command)
        command.add_argument("--study-id")
        command.add_argument("--purpose")
    commands.choices["run"].add_argument("--case-id", type=safe_output_name)
    ablation = commands.choices["ablation"]
    ablation.add_argument("--factor", required=True)
    ablation.add_argument("--values", nargs="+", required=True, type=parse_yaml_value)
    ablation.add_argument("--reference-value", required=True, type=parse_yaml_value)
    grid = commands.choices["grid"]
    grid.add_argument("--vary", action="append", required=True, type=_vary)
    grid.add_argument("--reference", action="append", default=[], type=parse_assignment)

    plan = commands.add_parser("run-plan", help="Run a reusable study-plan YAML.")
    plan.add_argument("--plan", required=True)
    _execution_arguments(plan)
    validate = commands.add_parser("validate", help="Validate a resolved configuration and installation.")
    _config_arguments(validate)
    validate.add_argument("--mode", choices=("config", "smoke", "full"), default="smoke")
    validate.add_argument("--environment-policy", choices=("exact", "record"), default="exact")
    validate.add_argument("--environment-lock", default=str(DEFAULT_LOCK))
    show = commands.add_parser("show-config", help="Print the fully resolved configuration.")
    _config_arguments(show)
    modules = commands.add_parser(
        "modules",
        help="List selectable module implementations.",
        add_help=False,
        description="List every selectable module. --help appends the live module catalog.",
    )
    modules.add_argument("-h", "--help", action=_CatalogHelp, catalog="modules")
    modules.add_argument("--family", default="all")
    commands.add_parser("presets", help="List named configuration presets.")
    parameters = commands.add_parser(
        "parameters",
        help="List leaf parameters, types, ranges, defaults and CLI forms.",
        add_help=False,
        description="List configurable leaves. --help appends names, types, ranges and input forms.",
    )
    parameters.add_argument("-h", "--help", action=_CatalogHelp, catalog="parameters")
    parameters.add_argument("--source-preset", choices=(*tuple(PRESETS), "all"), default="baseline")
    parameters.add_argument("--format", choices=("json", "yaml", "markdown"), default="json")
    manual = commands.add_parser("manual-cli", help="Expand a preset into a preset-free CLI command.")
    manual.add_argument("--source-preset", choices=tuple(PRESETS), default="finalcase")
    manual.add_argument("--run-name")
    infer = commands.add_parser("infer", help="Run no-fit inference with an exported learned bundle.")
    infer.add_argument("--model-config", required=True)
    infer.add_argument("--case-id")
    infer.add_argument("--input-manifest", required=True)
    index = commands.add_parser("index", help="Rebuild the economical run data index.")
    index.add_argument("--study-dir", required=True)
    index.add_argument("--hash-predictions", action="store_true")
    export = commands.add_parser("export-model-config", help="Export reusable configs and selected weights.")
    export.add_argument("--pipeline-output", required=True)
    export.add_argument("--replace", action="store_true")
    excel = commands.add_parser("export-excel", help="Rebuild the pipeline data workbook.")
    excel.add_argument("--pipeline-output", required=True)
    excel.add_argument("--replace", action="store_true")

    return parser

def _execution(args: argparse.Namespace, base: ExecutionSpec | None = None) -> ExecutionSpec:
    current = base or ExecutionSpec()
    cache = current.preprocessing_cache
    jobs = current.jobs if args.jobs is None else args.jobs
    if jobs < 1:
        raise ValueError("--jobs must be positive")
    return replace(
        current,
        repeats=current.repeats if args.repeats is None else args.repeats,
        folds=current.folds if args.folds is None else args.folds,
        jobs=jobs,
        device=current.device if args.device is None else args.device,
        continue_on_error=current.continue_on_error if args.continue_on_error is None else args.continue_on_error,
        measure_operational_costs=(
            current.measure_operational_costs
            if args.measure_operational_costs is None
            else args.measure_operational_costs
        ),
        preprocessing_cache=PreprocessingCacheSpec(
            mode=args.preprocessing_cache_mode or cache.mode,
            root=args.preprocessing_cache_root or cache.root,
            namespaces=args.preprocessing_cache_namespaces or cache.namespaces,
            verify_source_sha256=True,
        ),
    )

def _resolved(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    assignments = list(args.assignments)
    if getattr(args, "device", None) is not None:
        explicit = [value for path, value in map(parse_assignment, assignments) if path == "training.device"]
        if explicit and any(str(value) != args.device for value in explicit):
            raise ValueError("--device conflicts with --set training.device")
    options = dict(
        pipeline_root=PIPELINE_ROOT,
        preset=args.preset or "baseline",
        config_path=None if args.config is None else _path(args.config, must_exist=True),
        assignments=assignments,
        unsets=args.unset,
        modules=args.module,
        config_id=args.config_id,
        manual=args.manual,
    )
    resolved = resolve_configuration(**options)
    if getattr(args, "device", None) is None or explicit:
        return resolved
    if str(resolved[0]["training"]["device"]) == args.device:
        return resolved
    return resolve_configuration(**{**options, "assignments": [*assignments, f"training.device={args.device}"]})


_resolved_config = _resolved

def _resolved_file(config: Mapping[str, Any], provenance: Mapping[str, Any]) -> Path:
    target = PIPELINE_ROOT / "cache/resolved_cli_configs" / f"{provenance['resolved_config_sha256']}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)
    if not target.exists():
        target.write_text(encoded, encoding="utf-8")
    elif canonical_json_bytes(
        yaml.safe_load(target.read_text(encoding="utf-8"))
    ) != canonical_json_bytes(config):
        raise RuntimeError(f"resolved config digest collision: {target}")
    return target

def _plan(args: argparse.Namespace, config: Mapping[str, Any], source: Path) -> StudyPlan:
    if args.command == "ablation":
        axes = (AxisSpec(path=args.factor, values=tuple(args.values), reference=args.reference_value),)
        kind, role = "ablation", "ablation"
    elif args.command == "grid":
        references = dict(args.reference)
        varied = dict(args.vary)
        if set(references) - set(varied):
            raise ValueError("--reference path is absent from --vary")
        axes = tuple(AxisSpec(path=path, values=values, reference=references.get(path)) for path, values in args.vary)
        kind, role = "grid", "screening"
    else:
        axes, kind, role = (), "single", "single_run"
    name = "manual" if args.manual else "config" if args.config else args.preset
    return StudyPlan(
        schema_version="ppg_frailty.study_plan.v2",
        study=StudyInfo(
            study_id=args.study_id or f"v5_{name}",
            kind=kind,
            purpose=args.purpose or f"V5 data-only {name} run.",
            flow_position="V5 configurable workflow",
            decision_role=role,
            reference_case_id=getattr(args, "case_id", None),
            thesis_sections=("Methods 3.1-3.6", "Appendix E"),
        ),
        base_config=str(source),
        axes=axes,
        execution=_execution(args, ExecutionSpec(device=str(config["training"]["device"]))),
        output=OutputSpec(root=str(_path(args.output_root))),
        report=ReportSpec(write_html=False, write_static_figures=False, write_excel_workbook=False),
    )

def _plan_from_config_command(args: argparse.Namespace, config_path: Path, config: Mapping[str, Any]) -> StudyPlan:
    return _plan(args, config, config_path)

def _run_config(args: argparse.Namespace) -> int:
    if args.resume is not None and args.run_name is not None:
        raise ValueError("--run-name cannot be combined with --resume")
    config, provenance = _resolved(args)
    source = _resolved_file(config, provenance)
    naming_source = "manual.yaml" if args.manual else args.config or PRESETS[args.preset].relative_path
    resume = None if args.resume is None else _path(args.resume, must_exist=True)
    result = run_study(
        _plan(args, config, source),
        pipeline_root=PIPELINE_ROOT,
        source=source,
        output_root=_path(args.output_root),
        run_name=None if resume is not None else args.run_name or automatic_run_name(naming_source),
        resume=resume,
        environment_policy=args.environment_policy,
        environment_lock=_path(args.environment_lock, must_exist=True),
        hash_predictions=args.hash_predictions,
        dry_run=args.dry_run,
        refit=RefitOptions(enabled=args.refit),
        request_metadata={"command": args.command, "configuration_resolution": provenance},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return int(result.get("exit_code", 0))

def _print_parameters(args: argparse.Namespace) -> None:
    rows = parameter_rows(PIPELINE_ROOT, source_preset=args.source_preset)
    if args.format == "yaml":
        print(yaml.safe_dump(rows, sort_keys=False, allow_unicode=True), end="")
    elif args.format == "markdown":
        print("| Parameter | Type | Input | Range | Default |\n|---|---|---|---|---|")
        for row in rows:
            values = (
                row["path"],
                row["type"],
                row["input"],
                row["range"],
                json.dumps(row["default"], ensure_ascii=False),
            )
            print("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    else:
        print(json.dumps(rows, ensure_ascii=False, indent=2))

def _dispatch(args: argparse.Namespace) -> int:
    if args.command in {"run", "ablation", "grid"}:
        return _run_config(args)
    if args.command == "run-plan":
        source = _path(args.plan, must_exist=True)
        result = run_study(
            data_only_plan(replace(load_study_plan(source), execution=_execution(args)), _path(args.output_root)),
            pipeline_root=PIPELINE_ROOT,
            source=source,
            output_root=_path(args.output_root),
            run_name=args.run_name,
            resume=None if args.resume is None else _path(args.resume, must_exist=True),
            environment_policy=args.environment_policy,
            environment_lock=_path(args.environment_lock, must_exist=True),
            hash_predictions=args.hash_predictions,
            dry_run=args.dry_run,
            refit=RefitOptions(enabled=args.refit),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return int(result.get("exit_code", 0))
    if args.command == "modules":
        print(json.dumps(list_modules(args.family), ensure_ascii=False, indent=2))
        return 0
    if args.command == "presets":
        print(json.dumps(preset_rows(PIPELINE_ROOT), ensure_ascii=False, indent=2))
        return 0
    if args.command == "parameters":
        _print_parameters(args)
        return 0
    if args.command == "manual-cli":
        preset = PRESETS[args.source_preset]
        identity = (
            "" if preset.case_id is None
            else f" --study-id {preset.name} --case-id {preset.case_id}"
        )
        suffix = (
            ""
            if args.run_name is None
            else " --run-name " + shlex.quote(safe_output_name(args.run_name, label="run name"))
        )
        print(
            manual_cli_command(PIPELINE_ROOT, source_preset=args.source_preset)
            + " --repeats all --folds all --jobs 1 --device cuda --no-continue-on-error"
            + " --no-measure-operational-costs --preprocessing-cache-mode read_write"
            + " --preprocessing-cache-root cache/preprocessing"
            + " --preprocessing-cache-namespaces imu_calibration,canonical_signal_views,raw_windows"
            + " --output-root pipeline_output --environment-policy exact"
            + " --environment-lock requirements/environment-finalcase-lock.yaml"
            + identity
            + suffix
        )
        return 0
    if args.command in {"show-config", "validate"}:
        config, provenance = _resolved(args)
        result: dict[str, Any] = {"status": "passed", "configuration_resolution": provenance, "config": config}
        if args.command == "validate":
            environment = evaluate_environment(
                args.environment_policy,
                device=str(config["training"]["device"]),
                lock_path=_path(args.environment_lock, must_exist=True),
            )
            result["environment_check"] = environment.to_dict()
            if args.mode != "config":
                source = _resolved_file(config, provenance)
                report, loaded, rows, folds = preflight_pipeline(source, mode=args.mode, paths=PipelinePaths.discover())
                result.update(
                    preflight=report.__dict__,
                    config_sha256=loaded.sha256,
                    manifest_rows=len(rows),
                    fold_rows=len(folds.assignments),
                    installation=validate_installation(),
                )
        print(
            yaml.safe_dump(result, sort_keys=False, allow_unicode=True)
            if args.command == "show-config"
            else json.dumps(result, ensure_ascii=False, indent=2, default=str),
            end="" if args.command == "show-config" else "\n",
        )
        return 0
    if args.command == "infer":
        from .inference_service import infer_from_manifest

        result = infer_from_manifest(
            model_config_directory=_path(args.model_config, must_exist=True),
            case_id=args.case_id,
            input_manifest=_path(args.input_manifest, must_exist=True),
            pipeline_root=PIPELINE_ROOT,
        )
    elif args.command == "index":
        result = build_study_data_index(
            _path(args.study_dir, must_exist=True), hash_prediction_files=args.hash_predictions
        )
    elif args.command == "export-model-config":
        result = export_model_config(
            _path(args.pipeline_output, must_exist=True), pipeline_root=PIPELINE_ROOT, replace_existing=args.replace
        )
    elif args.command == "export-excel":
        result = export_pipeline_excel(_path(args.pipeline_output, must_exist=True), replace=args.replace)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        return _dispatch(parser.parse_args(argv))
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(f"{type(error).__name__}: {error}")
    return 2


__all__ = ["PIPELINE_ROOT", "_execution", "build_parser", "main"]

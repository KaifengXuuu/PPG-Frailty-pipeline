"""Read immutable V2/V5 prediction artifacts into the existing report model."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import yaml

from ppg_frailty.reporting.collect import (
    CollectedStudy,
    _cell_rows,
    _config_metrics,
    _history_rows,
    _quality_rows,
    collect_study,
)
from ppg_frailty.reporting.cache_audit import collect_preprocessing_cache_rows
from ppg_frailty.training import read_oof_parquet

from .contracts import (
    ArtifactRecord,
    LoadedReportData,
    PREDICTION_LAYERS,
    ReportContractError,
    ReportRequest,
)

_CELL = re.compile(r"^repeat_(\d+)_fold_(\d+)$")
_REPEAT = re.compile(r"^repeat_(\d+)$")
_FOLD = re.compile(r"^fold_(\d+)$")
_V2_ROOT = Path(__file__).resolve().parents[4] / "final_pipeline_v2"
_FILES: Mapping[str, tuple[str, ...]] = {
    "window": ("oof_window_predictions.parquet", ),
    "file": ("oof_file_predictions.parquet", ),
    "role": ("oof_role_predictions.parquet", ),
    "participant": (
        "oof_subject_predictions.parquet",
        "oof_participant_predictions.parquet",
    ),
    "member": ("oof_member_predictions.parquet", ),
}
_CASE_FIELDS = (
    "case_records",
    "cell_rows",
    "history_rows",
    "file_oof_rows",
    "subject_oof_rows",
    "role_oof_rows",
    "quality_rows",
    "trusted_config_metrics",
    "oof_read_failures",
    "window_oof_rows",
    "resolved_aggregation_configs",
    "resolved_config_failures",
    "preprocessing_cache_rows",
)


def _read_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ReportContractError(f"mapping root required: {path}")
    return dict(value)


def _artifact_root(path: Path) -> tuple[Path, Mapping[str, Any]]:
    """Resolve a case directory or an already-resolved experiment directory."""

    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    case_result = root / "case_result.json"
    if not case_result.is_file():
        return root, {}
    record = _read_mapping(case_result)
    raw = record.get("artifact_root")
    if not isinstance(raw, str) or not raw.strip():
        raise ReportContractError(f"case_result.json lacks artifact_root: {case_result}")
    target = (root / raw).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ReportContractError("case artifact_root escapes its case directory") from error
    if not target.is_dir():
        raise FileNotFoundError(target)
    return target, record


def _coordinates(path: Path) -> tuple[int | None, int | None]:
    match = _CELL.fullmatch(path.parent.name)
    if match is not None:
        return tuple(int(value) for value in match.groups())  # type: ignore[return-value]
    fold = _FOLD.fullmatch(path.parent.name)
    repeat = _REPEAT.fullmatch(path.parent.parent.name)
    if fold is not None and repeat is not None:
        return int(repeat.group(1)), int(fold.group(1))
    return None, None


def _paths_for_layer(root: Path, layer: str) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for filename in _FILES[layer]:
        candidates.extend(root.rglob(filename))
    candidates = sorted({path.resolve() for path in candidates}, key=str)
    cell_paths = [path for path in candidates if _coordinates(path) != (None, None)]
    if cell_paths:
        # Per-fold artifacts are authoritative.  Aggregate/copy artifacts are
        # intentionally ignored so a report cannot silently duplicate rows.
        if layer == "participant":
            aliases: dict[Path, set[str]] = {}
            for path in cell_paths:
                aliases.setdefault(path.parent, set()).add(path.name)
            ambiguous = [directory for directory, names in aliases.items() if len(names) > 1]
            if ambiguous:
                raise ReportContractError("both participant artifact aliases exist in fold directory: "
                                          f"{ambiguous[0]}")
        return tuple(cell_paths)
    direct = [path for path in candidates if path.parent == root]
    if layer == "participant" and len(direct) > 1:
        raise ReportContractError(f"both participant artifact aliases exist in {root}")
    return tuple(direct)


def _parquet_footer(path: Path, *, expected_coordinate: tuple[int, int] | None = None) -> tuple[int, dict[str, str]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - dependency error is explicit.
        raise RuntimeError("OOF Parquet input requires pyarrow") from error
    value = parquet.ParquetFile(path)
    # Reuse the exact V2 schema/metadata contract without materializing row
    # groups.  The normal reader separately validates values where rows are read.
    from ppg_frailty.training.oof import _schema_for_readback

    class _TableView:
        schema = value.schema_arrow
        num_rows = int(value.metadata.num_rows)

    expected = _schema_for_readback(pa, _TableView())
    if not value.schema_arrow.equals(expected, check_metadata=True):
        raise ValueError(f"OOF Parquet does not match the exact V2 schema: {path}")
    if expected_coordinate is not None and _TableView.num_rows:
        coordinates = parquet.read_table(path, columns=["repeat", "fold"])
        observed = set(
            zip(
                coordinates.column("repeat").to_pylist(),
                coordinates.column("fold").to_pylist(),
                strict=True,
            ))
        if observed != {expected_coordinate}:
            raise ReportContractError(f"OOF coordinate disagrees with fold directory: {path}; "
                                      f"expected={expected_coordinate}, observed={sorted(observed)}")
    metadata = {key.decode("ascii"): item.decode("utf-8") for key, item in (value.schema_arrow.metadata or {}).items()}
    return int(value.metadata.num_rows), metadata


def _artifact_record(case_id: str, layer: str, path: Path, *, expected_rows: int | None = None) -> ArtifactRecord:
    repeat, fold = _coordinates(path)
    row_count, metadata = _parquet_footer(
        path,
        expected_coordinate=(repeat, fold) if repeat is not None and fold is not None else None,
    )
    if expected_rows is not None and row_count != expected_rows:
        raise ReportContractError(f"Parquet footer row count changed while reading: {path}")
    return ArtifactRecord(
        case_id=case_id,
        layer=layer,
        path=path,
        repeat=repeat,
        fold=fold,
        row_count=row_count,
        byte_count=path.stat().st_size,
        artifact_state=str(metadata.get("artifact_state", "unknown")),
        empty_reason=str(metadata.get("empty_reason", "")),
        sha256="not_computed",
    )


def _load_layers(
    case_id: str,
    root: Path,
) -> tuple[dict[str, tuple[Mapping[str, Any], ...]], tuple[ArtifactRecord, ...]]:
    layers: dict[str, tuple[Mapping[str, Any], ...]] = {}
    inventory: list[ArtifactRecord] = []
    for layer in PREDICTION_LAYERS:
        rows: list[Mapping[str, Any]] = []
        for path in _paths_for_layer(root, layer):
            repeat, fold = _coordinates(path)
            try:
                values = read_oof_parquet(path)
            except Exception as error:  # noqa: BLE001 - exact input failure is useful.
                raise ReportContractError(f"cannot read {layer} prediction artifact {path}: "
                                          f"{type(error).__name__}: {error}") from error
            for value in values:
                row = {"case_id": case_id, **asdict(value)}
                if repeat is not None and (int(row["repeat"]) != repeat or int(row["fold"]) != fold):
                    raise ReportContractError(f"OOF coordinate disagrees with directory: {path}")
                rows.append(row)
            # Footer-only metadata avoids a second materialization of large
            # window tables after the strict schema reader above.
            inventory.append(_artifact_record(case_id, layer, path, expected_rows=len(values)))
        layers[layer] = tuple(rows)
    return layers, tuple(inventory)


def _find_config(case_directory: Path) -> tuple[dict[str, Any], str | None]:
    # A caller may point either at the case directory or at
    # ``attempts/attempt_NNN/experiment``.  Search only this bounded ancestry;
    # never perform an unbounded upward lookup that could bind another run.
    bases = (case_directory, *tuple(case_directory.parents)[:3])
    for base in bases:
        for name in ("resolved_config.yaml", "config.resolved.yaml"):
            path = base / name
            if path.is_file():
                return _read_mapping(path), path.as_posix()
    return {}, None


def _evaluation_manifest(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    for name in (
            "evaluation_manifest.json",
            "analysis_input_manifest.json",
            "run_manifest.json",
    ):
        path = root / name
        if path.is_file():
            return _read_mapping(path)
    manifests = sorted(root.rglob("run_manifest.json"), key=str)
    if manifests:
        return _read_mapping(manifests[0])
    return dict(record)


def _evaluation_scope(manifest: Mapping[str, Any]) -> str:
    raw = manifest.get("evaluation_scope")
    evaluation = manifest.get("evaluation")
    if raw is None and isinstance(evaluation, Mapping):
        raw = evaluation.get("scope")
    if raw is not None:
        value = str(raw).strip()
        if value not in {"outer_oof", "independent_test"}:
            raise ReportContractError(f"unknown evaluation_scope: {value!r}")
        declared = manifest.get("independent_test")
        if isinstance(declared, bool) and declared != (value == "independent_test"):
            raise ReportContractError("evaluation_scope contradicts the independent_test flag")
        return value
    # V2 run manifests use a boolean and explicitly declare outer-CV runs.
    return "independent_test" if manifest.get("independent_test") is True else "outer_oof"


def _cell_payloads(root: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result.get("cell_results"), list):
        return dict(result)
    cells: list[Mapping[str, Any]] = []
    for path in sorted(root.rglob("experiment_result.json"), key=str):
        payload = _read_mapping(path)
        cell = payload.get("cell")
        if isinstance(cell, Mapping):
            cells.append(dict(cell))
    return {"cell_results": cells}


def _direct_case(case_id: str, path: Path) -> LoadedReportData:
    artifact_root, record = _artifact_root(path)
    result = record.get("result") if isinstance(record.get("result"), Mapping) else {}
    result = _cell_payloads(artifact_root, result)
    layers, inventory = _load_layers(case_id, artifact_root)
    config, config_path = _find_config(path)
    manifest = _evaluation_manifest(artifact_root, record)
    cell_rows = tuple(_cell_rows(case_id, result, artifact_root))
    history_rows = tuple(_history_rows(case_id, result, artifact_root))
    quality_rows = tuple(_quality_rows(case_id, artifact_root))
    trusted = _config_metrics(case_id, artifact_root)
    cache_rows, cache_notes = collect_preprocessing_cache_rows(case_id, artifact_root)
    repeats = sorted({int(row["repeat"]) for row in layers["participant"]})
    folds = sorted({int(row["fold"]) for row in layers["participant"]})
    statistics = (config.get("evaluation", {}).get("statistics", {})
                  if isinstance(config.get("evaluation"), Mapping) else {})
    aggregation = config.get("aggregation", {})
    resolved = ()
    if isinstance(aggregation, Mapping) and isinstance(statistics, Mapping):
        resolved = ({
            "case_id": case_id,
            "resolved_config_path": config_path,
            "aggregation": dict(aggregation),
            "evaluation_statistics": dict(statistics),
        }, )
    status = str(record.get("status", "passed" if layers["participant"] else "incomplete"))
    collected = CollectedStudy(
        root=artifact_root,
        plan={
            "execution": {
                "repeats": repeats,
                "folds": folds
            },
            "report": {},
        },
        manifest={
            "schema_version": "ppg_frailty.v5_report_direct.v1",
            "cases": ({
                "case_id": case_id
            }, ),
        },
        case_records=({
            "case_id": case_id,
            "status": status,
            "artifact_root": str(artifact_root),
            "resolved_config_path": config_path,
        }, ),
        varied_parameters=(),
        controlled_parameters=(),
        cell_rows=cell_rows,
        history_rows=history_rows,
        file_oof_rows=layers["file"],
        subject_oof_rows=layers["participant"],
        role_oof_rows=layers["role"],
        quality_rows=quality_rows,
        trusted_config_metrics=(() if trusted is None else (trusted, )),
        limitations=tuple(cache_notes),
        window_oof_rows=layers["window"],
        resolved_aggregation_configs=resolved,
        preprocessing_cache_rows=tuple(cache_rows),
    )
    try:
        path.resolve().relative_to(_V2_ROOT.resolve())
        legacy_v2 = True
    except ValueError:
        legacy_v2 = False
    return LoadedReportData(
        collected=collected,
        layer_rows=layers,
        artifact_records=inventory,
        evaluation_scope_by_case={case_id: _evaluation_scope(manifest)},
        source_root_by_case={case_id: artifact_root},
        config_by_case={case_id: config},
        manifest_by_case={case_id: manifest},
        legacy_v2_cases=frozenset({case_id}) if legacy_v2 else frozenset(),
        source_kind="v2_run" if legacy_v2 else "run",
    )


def _merge_direct(values: Iterable[LoadedReportData]) -> LoadedReportData:
    items = tuple(values)
    first = items[0]
    fields: dict[str, Any] = {}
    for name in _CASE_FIELDS:
        fields[name] = tuple(row for item in items for row in getattr(item.collected, name))
    fields["varied_parameters"] = ()
    fields["controlled_parameters"] = ()
    fields["limitations"] = tuple(dict.fromkeys(note for item in items for note in item.collected.limitations))
    repeats = sorted({int(row["repeat"]) for item in items for row in item.collected.subject_oof_rows})
    folds = sorted({int(row["fold"]) for item in items for row in item.collected.subject_oof_rows})
    collected = replace(
        first.collected,
        root=first.collected.root,
        plan={
            "execution": {
                "repeats": repeats,
                "folds": folds
            },
            "report": {}
        },
        manifest={
            "schema_version": "ppg_frailty.v5_report_direct.v1",
            "cases": tuple({"case_id": item.case_ids[0]} for item in items),
        },
        **fields,
    )
    return LoadedReportData(
        collected=collected,
        layer_rows={
            layer: tuple(row for item in items for row in item.layer_rows[layer])
            for layer in PREDICTION_LAYERS
        },
        artifact_records=tuple(row for item in items for row in item.artifact_records),
        evaluation_scope_by_case={key: value
                                  for item in items for key, value in item.evaluation_scope_by_case.items()},
        source_root_by_case={key: value
                             for item in items for key, value in item.source_root_by_case.items()},
        config_by_case={key: value
                        for item in items for key, value in item.config_by_case.items()},
        manifest_by_case={key: value
                          for item in items for key, value in item.manifest_by_case.items()},
        legacy_v2_cases=frozenset(case_id for item in items for case_id in item.legacy_v2_cases),
        source_kind="multi_run" if len(items) > 1 else first.source_kind,
    )


def _study_case_root(root: Path, case: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    relative = Path(str(case.get("case_directory", Path("cases") / str(case["case_id"]))))
    if relative.is_absolute():
        raise ReportContractError("study case_directory must be relative")
    directory = (root / relative).resolve()
    try:
        directory.relative_to(root)
    except ValueError as error:
        raise ReportContractError("study case_directory escapes study") from error
    return _artifact_root(directory)


def _load_study(root: Path) -> LoadedReportData:
    collected = collect_study(root)
    cases = tuple(case for case in collected.manifest.get("cases", ()) if isinstance(case, Mapping))
    layers: dict[str, list[Mapping[str, Any]]] = {
        "window": list(collected.window_oof_rows),
        "file": list(collected.file_oof_rows),
        "role": [],
        "participant": list(collected.subject_oof_rows),
        "member": [],
    }
    inventory: list[ArtifactRecord] = []
    scopes: dict[str, str] = {}
    roots: dict[str, Path] = {}
    configs: dict[str, Mapping[str, Any]] = {}
    manifests: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        case_id = str(case["case_id"])
        try:
            artifact_root, record = _study_case_root(root, case)
        except FileNotFoundError:
            # ``collect_study`` represents not-run cases explicitly.  Preserve
            # that compatibility and let selected-case validation explain the
            # absence of participant predictions.
            raw_directory = Path(str(case.get("case_directory", Path("cases") / case_id)))
            artifact_root = (root / raw_directory).resolve()
            record = {}
            scopes[case_id] = "outer_oof"
            roots[case_id] = artifact_root
            configs[case_id] = {}
            manifests[case_id] = {}
            continue
        for layer in PREDICTION_LAYERS:
            for path in _paths_for_layer(artifact_root, layer):
                if layer in {"role", "member"}:
                    values = read_oof_parquet(path)
                    layers[layer].extend({"case_id": case_id, **asdict(value)} for value in values)
                    inventory.append(_artifact_record(case_id, layer, path, expected_rows=len(values)))
                else:
                    # V2 ``collect_study`` already materialized and strictly
                    # validated these four layers.  Only the cheap footer is
                    # needed here to complete the five-layer input index.
                    inventory.append(_artifact_record(case_id, layer, path))
        raw_config = case.get("resolved_config_path")
        config: Mapping[str, Any] = {}
        if isinstance(raw_config, str) and raw_config.strip():
            config_path = (root / raw_config).resolve()
            try:
                config_path.relative_to(root)
            except ValueError as error:
                raise ReportContractError("resolved_config_path escapes study") from error
            if config_path.is_file():
                config = _read_mapping(config_path)
        manifest = _evaluation_manifest(artifact_root, record)
        scopes[case_id] = _evaluation_scope(manifest)
        roots[case_id] = artifact_root
        configs[case_id] = config
        manifests[case_id] = manifest
    try:
        root.resolve().relative_to(_V2_ROOT.resolve())
        legacy_cases = frozenset(scopes)
    except ValueError:
        legacy_cases = frozenset()
    return LoadedReportData(
        collected=collected,
        layer_rows={name: tuple(rows)
                    for name, rows in layers.items()},
        artifact_records=tuple(inventory),
        evaluation_scope_by_case=scopes,
        source_root_by_case=roots,
        config_by_case=configs,
        manifest_by_case=manifests,
        legacy_v2_cases=legacy_cases,
        source_kind="v2_study" if legacy_cases else "v5_study",
    )


def _filter_cases(data: LoadedReportData, request: ReportRequest) -> LoadedReportData:
    available = set(data.case_ids)
    unknown_include = set(request.include_cases) - available
    unknown_exclude = set(request.exclude_cases) - available
    if unknown_include or unknown_exclude:
        raise ReportContractError("case selection names not present in input: "
                                  f"include={sorted(unknown_include)}, exclude={sorted(unknown_exclude)}")
    selected = (set(request.include_cases) if request.include_cases else set(available)) - set(request.exclude_cases)
    if not selected:
        raise ReportContractError("case selection is empty")
    if request.reference_case is not None and request.reference_case not in selected:
        raise ReportContractError("reference case must remain in the selected cases")

    def rows(name: str) -> tuple[Mapping[str, Any], ...]:
        values = getattr(data.collected, name)
        return tuple(row for row in values if row.get("case_id") is None or str(row.get("case_id")) in selected)

    manifest = dict(data.collected.manifest)
    manifest["cases"] = tuple(case for case in manifest.get("cases", ())
                              if isinstance(case, Mapping) and str(case.get("case_id")) in selected)
    if request.reference_case is not None:
        manifest["reference_case"] = request.reference_case
    replacements = {name: rows(name) for name in _CASE_FIELDS}
    replacements["varied_parameters"] = tuple(row for row in data.collected.varied_parameters
                                              if row.get("case_id") is None or str(row.get("case_id")) in selected)
    replacements["controlled_parameters"] = tuple(row for row in data.collected.controlled_parameters
                                                  if row.get("case_id") is None or str(row.get("case_id")) in selected)
    limitations = tuple(note for note in data.collected.limitations
                        if not any(note.startswith(f"{case_id}:") for case_id in available - selected))
    collected = replace(
        data.collected,
        manifest=manifest,
        limitations=limitations,
        **replacements,
    )
    return replace(
        data,
        collected=collected,
        layer_rows={
            layer: tuple(row for row in values if str(row.get("case_id")) in selected)
            for layer, values in data.layer_rows.items()
        },
        artifact_records=tuple(row for row in data.artifact_records if row.case_id in selected),
        evaluation_scope_by_case={
            key: value
            for key, value in data.evaluation_scope_by_case.items() if key in selected
        },
        source_root_by_case={key: value
                             for key, value in data.source_root_by_case.items() if key in selected},
        config_by_case={key: value
                        for key, value in data.config_by_case.items() if key in selected},
        manifest_by_case={key: value
                          for key, value in data.manifest_by_case.items() if key in selected},
        legacy_v2_cases=frozenset(data.legacy_v2_cases & selected),
    )


def load_report_data(request: ReportRequest) -> LoadedReportData:
    """Load then apply exact include/exclude/reference semantics."""

    studies = [
        run for run in request.runs
        if (run.path / "study_plan.yaml").is_file() and (run.path / "study_manifest.json").is_file()
    ]
    if studies:
        if len(request.runs) != 1:
            raise ReportContractError("a study input cannot be mixed with other --run inputs")
        data = _load_study(studies[0].path)
        if data.legacy_v2_cases and not request.allow_v2_compatibility:
            raise ReportContractError("V2 study compatibility was disabled")
    else:
        data = _merge_direct(_direct_case(run.case_id, run.path) for run in request.runs)
    return _filter_cases(data, request)


__all__ = ["load_report_data"]

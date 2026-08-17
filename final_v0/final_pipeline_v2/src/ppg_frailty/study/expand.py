"""Load, validate, and deterministically expand study configurations."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .schema import (
    AxisSpec,
    OutputSpec,
    ReportSpec,
    StudyInfo,
    StudyPlan,
    catalog_cases_from_mapping,
    catalog_spec_from_mapping,
    execution_from_mapping,
    sparse_search_spec_from_mapping,
)


_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")
_IDENTITY_ONLY_PATHS = frozenset({"config_id"})
_CATALOG_SEARCH_OVERRIDE_PATHS = frozenset(
    {
        "training.learning_rate",
        "training.weight_decay",
        "model.logistic_c",
        "model.architecture_parameters.C",
        "model.svm_c",
        "model.svm_gamma",
        "model.architecture_parameters.gamma",
        "model.extra_trees_max_features",
        "model.extra_trees_min_samples_leaf",
        "model.architecture_parameters.max_features",
        "model.architecture_parameters.min_samples_leaf",
        "model.n_kernels",
        "model.alpha",
        "model.architecture_parameters.n_kernels",
        "model.architecture_parameters.ridge_alpha",
    }
)


def _derived_axis_values(path: str, value: Any) -> dict[str, Any]:
    """Return schema fields that describe the same declared study factor."""

    if path != "training.fixed_epochs":
        return {}
    if isinstance(value, bool) or value not in {7, 10, 15}:
        return {}
    profile_by_epoch = {
        7: "ablation_7",
        10: "default_10",
        15: "ablation_15",
    }
    return {"training.epoch_profile": profile_by_epoch[int(value)]}


def _epoch_materialization_identity(
    *,
    base_path: Path,
    pipeline_root: Path,
    base_config_sha256: str,
    fixed_epochs: int,
) -> dict[str, Any]:
    """Describe the registered 7/10/15 profile without creating a side file."""

    from ppg_frailty.config import load_formal_ablation_profiles

    catalog = load_formal_ablation_profiles(
        pipeline_root / "configs" / "formal_ablation_profiles_v2.yaml"
    )
    entries = catalog["families"]["deep_fixed_epoch"]["entries"]
    matches = [
        dict(row)
        for row in entries
        if int(row["fixed_epochs"]) == int(fixed_epochs)
    ]
    if len(matches) != 1:
        raise ValueError(f"unregistered fixed epoch profile: {fixed_epochs}")
    selected = matches[0]
    try:
        base_name = base_path.relative_to(pipeline_root).as_posix()
    except ValueError:
        base_name = str(base_path)
    return {
        "schema_version": "ppg_frailty.formal_ablation_materialization.v2",
        "family": "deep_fixed_epoch",
        "profile_id": str(selected["profile_id"]),
        "catalog_role": str(selected["catalog_role"]),
        "base_config_path": base_name,
        "base_config_sha256": base_config_sha256,
        "profile_catalog_sha256": catalog["catalog_sha256"],
        "single_factor_only": True,
        "automatic_execution": False,
        "scientific_execution_completed": False,
    }


def _strict_value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _slug(value: Any, *, limit: int = 60) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = "null"
    elif isinstance(value, (dict, list, tuple)):
        text = _strict_value_key(value)
    else:
        text = str(value)
    cleaned = _SLUG_RE.sub("-", text).strip("-_").lower() or "value"
    if len(cleaned) <= limit:
        return cleaned
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:limit - 10]}--{digest}"


def flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings while treating sequences as atomic values."""

    flattened: dict[str, Any] = {}
    for raw_key in sorted(value):
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        item = value[raw_key]
        if isinstance(item, Mapping):
            flattened.update(flatten_mapping(item, path))
        else:
            flattened[path] = copy.deepcopy(item)
    return flattened


def _set_dotted(payload: dict[str, Any], path: str, value: Any) -> None:
    fields = path.split(".")
    cursor: dict[str, Any] = payload
    for field in fields[:-1]:
        existing = cursor.get(field)
        if not isinstance(existing, dict):
            raise KeyError(f"unknown nested configuration path: {path}")
        cursor = existing
    leaf = fields[-1]
    if leaf not in cursor:
        raise KeyError(f"unknown configuration field: {path}")
    cursor[leaf] = copy.deepcopy(value)


def _config_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ResolvedCase:
    """One fully resolved configuration produced by a study plan."""

    case_id: str
    config: Mapping[str, Any]
    changed_values: Mapping[str, Any]
    config_sha256: str
    is_reference: bool
    output_group: str | None = None
    catalog_entry: str | None = None
    screen_profile_id: str | None = None
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "case_id": self.case_id,
            "changed_values": dict(self.changed_values),
            "config_sha256": self.config_sha256,
            "is_reference": self.is_reference,
            "config_id": self.config.get("config_id"),
        }
        if self.output_group is not None:
            payload.update(
                {
                    "output_group": self.output_group,
                    "catalog_entry": self.catalog_entry,
                    "screen_profile_id": self.screen_profile_id,
                    "rationale": self.rationale,
                }
            )
        return payload


@dataclass(frozen=True)
class StudyExpansion:
    """Resolved cases plus machine-generated variable/control tables."""

    plan: StudyPlan
    base_config_path: Path
    base_config: Mapping[str, Any]
    cases: tuple[ResolvedCase, ...]
    varied_parameters: tuple[Mapping[str, Any], ...]
    controlled_parameters: tuple[Mapping[str, Any], ...]
    reference_case_id: str | None


def validate_canonical_expansion(expansion: StudyExpansion) -> StudyExpansion:
    """Validate every resolved case with the canonical pipeline contract.

    Expansion deliberately accepts arbitrary YAML scalar values so it can remain
    a generic Cartesian-product utility.  Human-facing pipeline commands call
    this boundary before dry-run output or execution, so invalid overrides are
    rejected without starting a study.
    """

    from ppg_frailty.config import validate_config_payload

    for case in expansion.cases:
        try:
            validate_config_payload(case.config)
        except Exception as error:
            raise ValueError(
                f"resolved study case {case.case_id!r} is not a valid canonical "
                f"pipeline configuration: {error}"
            ) from error
    return expansion


def parse_study_plan(
    payload: Mapping[str, Any], *, plan_path: str | Path | None = None
) -> StudyPlan:
    raw = dict(payload)
    study_raw = dict(raw.get("study") or {})
    kind = str(study_raw.get("kind", "")).strip()
    if kind == "catalog_sweep":
        expected_top = {
            "schema_version",
            "study",
            "catalog",
            "search",
            "cases",
            "execution",
            "output",
            "report",
        }
        if set(raw) != expected_top:
            raise ValueError(
                "catalog_sweep plan key mismatch: "
                f"missing={sorted(expected_top-set(raw))}, "
                f"unknown={sorted(set(raw)-expected_top)}"
            )
        expected_study = {
            "study_id",
            "kind",
            "purpose",
            "flow_position",
            "decision_role",
            "reference_case_id",
            "thesis_sections",
        }
        if set(study_raw) != expected_study:
            raise ValueError(
                "catalog_sweep study key mismatch: "
                f"missing={sorted(expected_study-set(study_raw))}, "
                f"unknown={sorted(set(study_raw)-expected_study)}"
            )
    default_role = {
        "single": "single_run",
        "ablation": "ablation",
        "grid": "screening",
    }.get(kind, "screening")
    study = StudyInfo(
        study_id=str(study_raw.get("study_id", "")).strip(),
        kind=kind,
        purpose=str(study_raw.get("purpose", "")).strip(),
        flow_position=str(study_raw.get("flow_position", "")).strip(),
        decision_role=str(study_raw.get("decision_role", default_role)).strip(),
        reference_case_id=(
            None
            if study_raw.get("reference_case_id") in (None, "")
            else str(study_raw["reference_case_id"])
        ),
        thesis_sections=tuple(str(value) for value in study_raw.get("thesis_sections", ())),
    )
    axes = tuple(
        AxisSpec(
            path=str(item["path"]),
            values=tuple(item.get("values", ())),
            reference=item.get("reference"),
        )
        for item in raw.get("axes", ())
    )
    output_raw = dict(raw.get("output") or {})
    report_raw = dict(raw.get("report") or {})
    return StudyPlan(
        schema_version=str(raw.get("schema_version", "")),
        study=study,
        base_config=(
            None
            if kind == "catalog_sweep"
            else str(raw.get("base_config", ""))
        ),
        axes=axes,
        catalog=(
            catalog_spec_from_mapping(raw["catalog"])
            if kind == "catalog_sweep"
            else None
        ),
        search=(
            sparse_search_spec_from_mapping(raw["search"])
            if kind == "catalog_sweep"
            else None
        ),
        cases=(
            catalog_cases_from_mapping(raw["cases"])
            if kind == "catalog_sweep"
            else ()
        ),
        execution=execution_from_mapping(raw.get("execution")),
        output=OutputSpec(root=str(output_raw.get("root", "artifacts/studies"))),
        report=ReportSpec(
            top_k=int(report_raw.get("top_k", 10)),
            write_html=bool(report_raw.get("write_html", True)),
            write_static_figures=bool(
                report_raw.get("write_static_figures", True)
            ),
            calibration_bins=int(report_raw.get("calibration_bins", 10)),
        ),
        plan_path=None if plan_path is None else Path(plan_path).resolve(),
    )


def load_study_plan(path: str | Path) -> StudyPlan:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("study plan root must be a mapping")
    return parse_study_plan(payload, plan_path=source)


def _resolve_base_config(plan: StudyPlan, pipeline_root: Path) -> Path:
    if plan.base_config is None:
        raise ValueError("non-catalog study requires base_config")
    raw = Path(plan.base_config)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(pipeline_root / raw)
        if plan.plan_path is not None:
            candidates.append(plan.plan_path.parent / raw)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"base config not found: {plan.base_config}")


def _resolve_catalog_path(plan: StudyPlan, pipeline_root: Path) -> Path:
    if plan.catalog is None:
        raise ValueError("catalog_sweep requires catalog")
    raw = Path(plan.catalog.path)
    candidates = [raw] if raw.is_absolute() else [pipeline_root / raw]
    if not raw.is_absolute() and plan.plan_path is not None:
        candidates.append(plan.plan_path.parent / raw)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"catalog not found: {plan.catalog.path}")


def _apply_fixed_kernel_profile(
    payload: dict[str, Any],
    *,
    profile_id: str,
    pipeline_root: Path,
    catalog_path: Path,
    catalog_entry: str,
) -> None:
    """Apply one registered raw-DL sampling profile without writing side files."""

    from ppg_frailty.config import (
        canonical_json_bytes,
        load_formal_ablation_profiles,
    )
    from ppg_frailty.models import normalize_model_id
    from ppg_frailty.models.time_scale import fixed_kernel_case

    base_payload = copy.deepcopy(payload)
    selected = fixed_kernel_case(profile_id)
    _canonical, machine_id = normalize_model_id(str(payload["model"]["model_id"]))
    expected_machine = (
        "compact_cnn"
        if selected.model_name == "CompactCNN1D"
        else "inception_full"
    )
    if payload["representation_mode"] != "raw" or machine_id != expected_machine:
        raise ValueError(
            f"formal profile {profile_id} is incompatible with {catalog_entry}"
        )
    payload["windows"]["raw_dl"]["length_s"] = float(
        selected.raw_window_seconds
    )
    resampling = payload["signal"]["dl_resampling"]
    resampling["case_id"] = selected.case_id
    resampling["enabled"] = float(selected.dl_fs_hz) != 400.0
    resampling["target_fs_hz"] = float(selected.dl_fs_hz)
    if machine_id == "compact_cnn":
        dilations = [int(selected.dilation)] * 3
        payload["model"]["dilations"] = dilations
        payload["model"]["architecture_parameters"]["dilations"] = dilations
    else:
        payload["model"]["dilation"] = int(selected.dilation)
        payload["model"]["architecture_parameters"]["dilation"] = int(
            selected.dilation
        )
    profiles = load_formal_ablation_profiles(
        pipeline_root / "configs" / "formal_ablation_profiles_v2.yaml"
    )
    payload["output"]["formal_ablation_materialization"] = {
        "schema_version": "ppg_frailty.formal_ablation_materialization.v2",
        "family": "fixed_kernel_samples",
        "profile_id": selected.case_id,
        "catalog_role": (
            "reference"
            if selected.case_id.endswith("__reference")
            else "ablation"
        ),
        "base_config_path": (
            f"{catalog_path.relative_to(pipeline_root).as_posix()}"
            f"#{catalog_entry}"
        ),
        "base_config_sha256": hashlib.sha256(
            canonical_json_bytes(base_payload)
        ).hexdigest(),
        "profile_catalog_sha256": profiles["catalog_sha256"],
        "single_factor_only": True,
        "automatic_execution": False,
        "scientific_execution_completed": False,
    }


def _catalog_varied_rows(
    cases: Iterable[ResolvedCase],
) -> tuple[Mapping[str, Any], ...]:
    values = tuple(cases)
    paths = sorted(
        {
            path
            for case in values
            for path in case.changed_values
        }
    )
    return tuple(
        {
            "parameter_path": path,
            "case_values": {
                case.case_id: copy.deepcopy(
                    case.changed_values.get(path, "not_applicable")
                )
                for case in values
            },
        }
        for path in paths
    )


def _expand_catalog_sweep(
    plan: StudyPlan,
    *,
    pipeline_root: Path,
) -> StudyExpansion:
    from ppg_frailty.catalog import resolved_catalog_payloads
    from ppg_frailty.config import (
        load_formal_experiment_catalog,
        validate_config_payload,
    )

    if plan.catalog is None or plan.search is None:
        raise ValueError("catalog_sweep lacks catalog/search metadata")
    catalog_path = _resolve_catalog_path(plan, pipeline_root)
    catalog = load_formal_experiment_catalog(catalog_path)
    entry_by_id = {str(row["entry_id"]): dict(row) for row in catalog["entries"]}
    ordinary_ids = {
        entry_id
        for entry_id, row in entry_by_id.items()
        if row["catalog_role"] in {"reference_candidate", "ablation_candidate"}
    }
    if len(ordinary_ids) != 13:
        raise RuntimeError("formal catalog no longer contains exactly 13 ordinary candidates")
    requested_ids = {case.catalog_entry for case in plan.cases}
    if requested_ids != ordinary_ids:
        raise ValueError(
            "ordinary_13 cases must cover the exact registered candidate set: "
            f"missing={sorted(ordinary_ids-requested_ids)}, "
            f"unknown={sorted(requested_ids-ordinary_ids)}"
        )
    payloads = resolved_catalog_payloads(
        pipeline_root=pipeline_root,
        line=plan.catalog.balance_line,
        catalog_path=catalog_path,
    )
    payload_by_entry = {
        entry_id: payload
        for entry_id, payload in (
            (
                str(row["entry_id"]),
                next(
                    value
                    for value in payloads
                    if value["config_id"]
                    == f"{row['config_stem']}_{plan.catalog.balance_line}_v2"
                ),
            )
            for row in catalog["entries"]
        )
    }
    resolved: list[ResolvedCase] = []
    for case in plan.cases:
        entry = entry_by_id[case.catalog_entry]
        if entry["catalog_role"] not in {
            "reference_candidate",
            "ablation_candidate",
        }:
            raise ValueError(
                f"catalog case cannot use role {entry['catalog_role']}"
            )
        expected_group = str(entry["representation_mode"])
        if case.output_group != expected_group:
            raise ValueError(
                f"{case.case_id} output_group={case.output_group!r} differs "
                f"from representation_mode={expected_group!r}"
            )
        unknown_overrides = set(case.overrides) - _CATALOG_SEARCH_OVERRIDE_PATHS
        if unknown_overrides:
            raise ValueError(
                f"{case.case_id} has non-screening overrides: "
                f"{sorted(unknown_overrides)}"
            )
        payload = copy.deepcopy(payload_by_entry[case.catalog_entry])
        if case.formal_profile is not None:
            if case.formal_profile.family != "fixed_kernel_samples":
                raise ValueError("catalog sweep supports only fixed_kernel_samples profiles")
            if case.overrides:
                raise ValueError(
                    "fixed-kernel single-factor profiles cannot add search overrides"
                )
            _apply_fixed_kernel_profile(
                payload,
                profile_id=case.formal_profile.profile_id,
                pipeline_root=pipeline_root,
                catalog_path=catalog_path,
                catalog_entry=case.catalog_entry,
            )
        for path, value in case.overrides.items():
            _set_dotted(payload, path, value)
        payload["config_id"] = (
            f"{payload['config_id']}__{case.screen_profile_id}"
        )
        checked = validate_config_payload(payload)
        changed: dict[str, Any] = {
            "study.catalog_entry": case.catalog_entry,
            "study.screen_profile_id": case.screen_profile_id,
        }
        changed.update(copy.deepcopy(dict(case.overrides)))
        if case.formal_profile is not None:
            changed["study.formal_profile"] = (
                f"{case.formal_profile.family}/"
                f"{case.formal_profile.profile_id}"
            )
            changed["signal.dl_resampling.target_fs_hz"] = checked[
                "signal"
            ]["dl_resampling"]["target_fs_hz"]
            changed["signal.dl_resampling.enabled"] = checked[
                "signal"
            ]["dl_resampling"]["enabled"]
            if "dilations" in checked["model"]:
                changed["model.dilations"] = copy.deepcopy(
                    checked["model"]["dilations"]
                )
            if "dilation" in checked["model"]:
                changed["model.dilation"] = checked["model"]["dilation"]
        resolved.append(
            ResolvedCase(
                case_id=case.case_id,
                config=checked,
                changed_values=changed,
                config_sha256=_config_sha256(checked),
                is_reference=case.case_id == plan.study.reference_case_id,
                output_group=case.output_group,
                catalog_entry=case.catalog_entry,
                screen_profile_id=case.screen_profile_id,
                rationale=case.rationale,
            )
        )
    reference_ids = [case.case_id for case in resolved if case.is_reference]
    if len(reference_ids) > 1:
        raise ValueError("catalog_sweep has more than one global reference case")
    if plan.study.reference_case_id is not None and not reference_ids:
        raise ValueError("catalog_sweep reference_case_id is not a declared case")
    return StudyExpansion(
        plan=plan,
        base_config_path=catalog_path,
        base_config=catalog,
        cases=tuple(resolved),
        varied_parameters=_catalog_varied_rows(resolved),
        controlled_parameters=_rows_for_parameters(resolved, varied=False),
        reference_case_id=reference_ids[0] if reference_ids else None,
    )


def _case_id(changes: Mapping[str, Any], index: int) -> str:
    if not changes:
        return "reference"
    parts = [
        f"{_slug(path.replace('.', '-'), limit=34)}--{_slug(value, limit=34)}"
        for path, value in sorted(changes.items())
    ]
    descriptive = "__".join(parts)
    if len(descriptive) <= 120:
        return descriptive
    digest = hashlib.sha256(descriptive.encode("utf-8")).hexdigest()[:10]
    return f"case-{index + 1:03d}--{digest}"


def _rows_for_parameters(
    cases: Iterable[ResolvedCase], *, varied: bool
) -> tuple[Mapping[str, Any], ...]:
    case_values = {
        case.case_id: {
            path: value
            for path, value in flatten_mapping(case.config).items()
            if path not in _IDENTITY_ONLY_PATHS
        }
        for case in cases
    }
    if not case_values:
        return ()
    all_paths = sorted(set.intersection(*(set(values) for values in case_values.values())))
    rows: list[Mapping[str, Any]] = []
    for path in all_paths:
        values = {case_id: mapping[path] for case_id, mapping in case_values.items()}
        keys = {_strict_value_key(value) for value in values.values()}
        if (len(keys) > 1) != varied:
            continue
        if varied:
            rows.append({"parameter_path": path, "case_values": values})
        else:
            rows.append(
                {
                    "parameter_path": path,
                    "value": next(iter(values.values())),
                }
            )
    return tuple(rows)


def expand_study(plan: StudyPlan, *, pipeline_root: str | Path) -> StudyExpansion:
    root = Path(pipeline_root).resolve()
    if plan.study.kind == "catalog_sweep":
        return _expand_catalog_sweep(plan, pipeline_root=root)
    base_path = _resolve_base_config(plan, root)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, Mapping):
        raise TypeError("base pipeline config must be a mapping")
    base_payload = copy.deepcopy(dict(base))
    combinations = (
        [()]
        if not plan.axes
        else itertools.product(*(axis.values for axis in plan.axes))
    )
    cases: list[ResolvedCase] = []
    base_flat = flatten_mapping(base_payload)
    for index, values in enumerate(combinations):
        changes = {
            axis.path: copy.deepcopy(value)
            for axis, value in zip(plan.axes, values)
        }
        config = copy.deepcopy(base_payload)
        derived_changes: dict[str, Any] = {}
        for path, value in changes.items():
            _set_dotted(config, path, value)
            for derived_path, derived_value in _derived_axis_values(
                path, value
            ).items():
                _set_dotted(config, derived_path, derived_value)
                derived_changes[derived_path] = derived_value
        if "training.fixed_epochs" in changes and _derived_axis_values(
            "training.fixed_epochs", changes["training.fixed_epochs"]
        ):
            identity = _epoch_materialization_identity(
                base_path=base_path,
                pipeline_root=root,
                base_config_sha256=_config_sha256(base_payload),
                fixed_epochs=int(changes["training.fixed_epochs"]),
            )
            output = config.get("output")
            if not isinstance(output, dict):
                raise TypeError("base pipeline output section must be a mapping")
            output["formal_ablation_materialization"] = identity
            derived_changes.update(
                flatten_mapping(
                    {"output": {"formal_ablation_materialization": identity}}
                )
            )
        case_id = _case_id(changes, index)
        original_id = str(base_payload.get("config_id", "study_config"))
        config["config_id"] = f"{original_id}__{case_id}"
        changed_paths = {
            path
            for path, value in flatten_mapping(config).items()
            if path not in _IDENTITY_ONLY_PATHS
            and _strict_value_key(value) != _strict_value_key(base_flat.get(path))
        }
        allowed_changed_paths = set(changes) | set(derived_changes)
        if not changed_paths <= allowed_changed_paths:
            raise ValueError(
                f"case {case_id} changed undeclared fields: "
                f"expected={sorted(allowed_changed_paths)}, "
                f"observed={sorted(changed_paths)}"
            )
        resolved_flat = flatten_mapping(config)
        mismatches = {
            path: {"requested": value, "resolved": resolved_flat.get(path)}
            for path, value in changes.items()
            if _strict_value_key(resolved_flat.get(path)) != _strict_value_key(value)
        }
        if mismatches:
            raise ValueError(
                f"case {case_id} did not preserve requested axis values: {mismatches}"
            )
        derived_mismatches = {
            path: {"derived": value, "resolved": resolved_flat.get(path)}
            for path, value in derived_changes.items()
            if _strict_value_key(resolved_flat.get(path))
            != _strict_value_key(value)
        }
        if derived_mismatches:
            raise ValueError(
                f"case {case_id} did not preserve derived axis values: "
                f"{derived_mismatches}"
            )
        inferred_reference = bool(plan.axes) and all(
            axis.reference is not None and value == axis.reference
            for axis, value in zip(plan.axes, values)
        )
        is_reference = case_id == plan.study.reference_case_id or inferred_reference
        cases.append(
            ResolvedCase(
                case_id=case_id,
                config=config,
                changed_values=changes,
                config_sha256=_config_sha256(config),
                is_reference=is_reference,
            )
        )
    if plan.study.kind == "single":
        cases[0] = ResolvedCase(
            case_id=cases[0].case_id,
            config=cases[0].config,
            changed_values=cases[0].changed_values,
            config_sha256=cases[0].config_sha256,
            is_reference=True,
        )
    case_ids = tuple(case.case_id for case in cases)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(
            "study values produced colliding case identifiers; use values with "
            "distinct filesystem-safe representations"
        )
    explicit_reference = plan.study.reference_case_id
    if explicit_reference is not None and explicit_reference not in {
        case.case_id for case in cases
    }:
        raise ValueError(f"reference_case_id is not a resolved case: {explicit_reference}")
    references = [case.case_id for case in cases if case.is_reference]
    if len(references) > 1:
        raise ValueError(f"study has more than one reference case: {references}")
    reference_case_id = references[0] if references else None
    derived_axis_paths = {
        derived_path
        for axis in plan.axes
        for value in axis.values
        for derived_path in _derived_axis_values(axis.path, value)
    }
    if any(axis.path == "training.fixed_epochs" for axis in plan.axes):
        derived_axis_paths.update(
            row["parameter_path"]
            for row in _rows_for_parameters(cases, varied=True)
            if str(row["parameter_path"]).startswith(
                "output.formal_ablation_materialization."
            )
        )
    varied = tuple(
        row
        for row in _rows_for_parameters(cases, varied=True)
        if row["parameter_path"] not in derived_axis_paths
    )
    declared = set(axis.path for axis in plan.axes)
    observed = {str(row["parameter_path"]) for row in varied}
    if observed != declared:
        raise ValueError(
            f"resolved varied parameters differ from declared axes: "
            f"declared={sorted(declared)}, observed={sorted(observed)}"
        )
    return StudyExpansion(
        plan=plan,
        base_config_path=base_path,
        base_config=base_payload,
        cases=tuple(cases),
        varied_parameters=varied,
        controlled_parameters=_rows_for_parameters(cases, varied=False),
        reference_case_id=reference_case_id,
    )

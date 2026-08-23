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
    legacy_bridge_spec_from_mapping,
    sparse_search_spec_from_mapping,
)


_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")
_IDENTITY_ONLY_PATHS = frozenset({"config_id"})
_NAMED_EPOCH_PROFILES = {
    7: "ablation_7",
    10: "default_10",
    15: "ablation_15",
}


def _derived_axis_values(path: str, value: Any) -> dict[str, Any]:
    """Return schema fields that describe the same declared study factor."""

    if path == "training.optimizer":
        from ppg_frailty.training.trainer import resolve_optimizer_parameters

        # Switching the strategy also switches to that strategy's own defaults.
        # Individual optimizer parameters remain ordinary independent axes.
        return {
            "training.optimizer_parameters": resolve_optimizer_parameters(
                str(value), {}
            )
        }
    if path == "aggregation.balance_line":
        derived = {
            "line_a_equal_files": {
                "aggregation.hierarchy": ["window", "file", "participant"],
                "aggregation.file_to_role": "not_applicable",
                "aggregation.role_to_participant": "not_applicable",
                "aggregation.missing_role_policy": "not_applicable",
            },
            "line_b_equal_role_families": {
                "aggregation.hierarchy": [
                    "window", "file", "role", "participant",
                ],
                "aggregation.file_to_role": "ordinary_mean",
                "aggregation.role_to_participant": "ordinary_mean",
                "aggregation.missing_role_policy": "mean_available_roles",
            },
        }
        return copy.deepcopy(derived.get(str(value), {}))
    if path != "training.fixed_epochs":
        return {}
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return {}
    from ppg_frailty.training.trainer import derived_epoch_profile

    return {
        "training.epoch_profile": derived_epoch_profile("fixed_epoch", int(value))
    }


def _normalize_axis_value(path: str, value: Any) -> Any:
    """Normalize semantic aliases before case identity/mismatch accounting."""

    if path == "features.enabled_groups":
        from ppg_frailty.features.registry import canonicalize_feature_groups

        return list(canonicalize_feature_groups(value))
    return copy.deepcopy(value)


def _is_reportable_parameter_path(path: str) -> bool:
    """Exclude automatically derived model provenance from study axes."""

    return (
        path not in _IDENTITY_ONLY_PATHS
        and path != "model.ensemble_size"
        and not path.startswith("model.architecture_parameters")
    )


def _drop_model_derived_provenance(payload: dict[str, Any]) -> None:
    """Remove generated fields before applying/validating user study inputs."""

    from ppg_frailty.module_registry import model_factory_contract

    model = payload["model"]
    contract = model_factory_contract(str(model["model_id"]))
    for field in contract["derived_provenance_fields"]:
        model.pop(str(field), None)


def _apply_execution_device(payload: dict[str, Any], requested: str | None) -> None:
    """Apply a plan device only to backends that can actually execute it.

    CUDA is an operational control for Torch models. Estimator backends do not
    consume a Torch device, so their effective configuration must remain the
    truthful neutral CPU value in mixed-model studies.
    """

    if requested is None:
        return
    from ppg_frailty.module_registry import model_factory_contract

    model = payload.get("model")
    if not isinstance(model, Mapping) or not str(model.get("model_id", "")).strip():
        raise ValueError("execution device requires a resolved model.model_id")
    backend = str(model_factory_contract(str(model["model_id"]))["execution_backend"])
    if backend not in {"torch", "estimator"}:
        raise ValueError(f"unknown model execution backend: {backend!r}")
    _set_dotted(
        payload,
        "training.device",
        str(requested).strip() if backend == "torch" else "cpu",
    )


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


def _set_dotted(
    payload: dict[str, Any],
    path: str,
    value: Any,
    *,
    allow_new_leaf: bool = False,
) -> None:
    fields = path.split(".")
    cursor: dict[str, Any] = payload
    for field in fields[:-1]:
        existing = cursor.get(field)
        if not isinstance(existing, dict):
            raise KeyError(f"unknown nested configuration path: {path}")
        cursor = existing
    leaf = fields[-1]
    if leaf not in cursor and not allow_new_leaf:
        raise KeyError(f"unknown configuration field: {path}")
    cursor[leaf] = copy.deepcopy(value)


def _get_dotted(payload: Mapping[str, Any], path: str) -> Any:
    cursor: Any = payload
    for field in path.split("."):
        if not isinstance(cursor, Mapping) or field not in cursor:
            return None
        cursor = cursor[field]
    return copy.deepcopy(cursor)


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
        if "legacy_bridge" in raw:
            expected_top.add("legacy_bridge")
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
    raw_figure_modules = report_raw.get("figure_modules", ("all",))
    if isinstance(raw_figure_modules, str):
        raw_figure_modules = (raw_figure_modules,)
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
        legacy_bridge=(
            legacy_bridge_spec_from_mapping(raw["legacy_bridge"])
            if "legacy_bridge" in raw
            else None
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
            figure_modules=tuple(
                str(value)
                for value in raw_figure_modules
            ),
            compact_mean_sd=bool(report_raw.get("compact_mean_sd", True)),
            write_excel_workbook=bool(
                report_raw.get("write_excel_workbook", True)
            ),
            classification_tsne_random_state=int(
                report_raw.get("classification_tsne_random_state", 42)
            ),
            classification_tsne_perplexity=float(
                report_raw.get("classification_tsne_perplexity", 30.0)
            ),
            classification_tsne_max_samples=int(
                report_raw.get("classification_tsne_max_samples", 5000)
            ),
            classification_roc_macro_grid_points=int(
                report_raw.get("classification_roc_macro_grid_points", 201)
            ),
            classification_score_histogram_bins=int(
                report_raw.get("classification_score_histogram_bins", 40)
            ),
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
    else:
        payload["model"]["dilation"] = int(selected.dilation)
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
    if not ordinary_ids:
        raise RuntimeError("formal catalog contains no active ordinary candidates")
    requested_ids = {case.catalog_entry for case in plan.cases}
    unknown_ids = requested_ids - set(entry_by_id)
    if unknown_ids:
        raise ValueError(
            "catalog cases contain unknown entries: "
            f"{sorted(unknown_ids)}"
        )
    if plan.catalog.scope in {"ordinary_active", "ordinary_13"} and requested_ids != ordinary_ids:
        raise ValueError(
            "ordinary cases must cover the exact active registered candidate set: "
            f"missing={sorted(ordinary_ids-requested_ids)}, "
            f"unknown={sorted(requested_ids-ordinary_ids)}"
        )
    if (
        plan.catalog.scope == "selected_ordinary"
        and not requested_ids <= ordinary_ids
    ):
        raise ValueError(
            "selected_ordinary cases must use only registered ordinary entries: "
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
    selected_roles: list[str] = []
    selected_groups: list[str] = []
    for case in plan.cases:
        entry = entry_by_id[case.catalog_entry]
        allowed_roles = (
            {"matched_comparator", "ensemble_comparison"}
            if plan.catalog.scope == "matched_ensemble_pair"
            else {"reference_candidate", "ablation_candidate"}
        )
        if entry["catalog_role"] not in allowed_roles:
            raise ValueError(
                f"catalog case cannot use role {entry['catalog_role']}"
            )
        selected_roles.append(str(entry["catalog_role"]))
        selected_groups.append(str(entry["representation_mode"]))
        expected_group = str(entry["representation_mode"])
        if case.output_group != expected_group:
            raise ValueError(
                f"{case.case_id} output_group={case.output_group!r} differs "
                f"from representation_mode={expected_group!r}"
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
            # Inactive module parameters are intentionally absent from resolved
            # base configs. The strict pipeline validator below remains the
            # authority for whether a newly supplied leaf is registered.
            _set_dotted(payload, path, value, allow_new_leaf=True)
        _apply_execution_device(payload, plan.execution.device)
        payload["config_id"] = (
            f"{payload['config_id']}__{case.screen_profile_id}"
        )
        _drop_model_derived_provenance(payload)
        checked = validate_config_payload(payload)
        changed: dict[str, Any] = {
            "study.catalog_entry": case.catalog_entry,
            "study.screen_profile_id": case.screen_profile_id,
        }
        if plan.legacy_bridge is not None:
            profile_matches = tuple(
                profile
                for profile in plan.legacy_bridge.profiles
                if str(profile["catalog_case_id"]) == case.case_id
            )
            if len(profile_matches) != 1:
                raise ValueError(
                    "legacy bridge case must bind exactly one frozen profile: "
                    f"{case.case_id}"
                )
            changed["study.legacy_bridge_profile"] = str(
                profile_matches[0]["profile_id"]
            )
            changed["study.legacy_bridge_runtime"] = (
                plan.legacy_bridge.runtime_status
            )
            changed["study.legacy_bridge_design"] = plan.legacy_bridge.design
            if plan.legacy_bridge.uses_inline_profiles:
                changed["study.legacy_bridge_profile_definition_sha256"] = (
                    plan.legacy_bridge.controls_sha256(profile_matches[0])
                )
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
    if plan.catalog.scope == "matched_ensemble_pair":
        if sorted(selected_roles) != ["ensemble_comparison", "matched_comparator"]:
            raise ValueError(
                "matched_ensemble_pair requires one matched comparator and "
                "one ensemble comparison"
            )
        if len(set(selected_groups)) != 1:
            raise ValueError(
                "matched_ensemble_pair entries must use one representation route"
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
            if _is_reportable_parameter_path(path)
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
    # Canonical V2 axes operate on the complete effective configuration, not
    # only fields that happened to be written in the source YAML.  This makes
    # newly configurable defaults (for example samples_per_epoch or optimizer
    # parameters) valid study axes while keeping this expansion utility generic
    # for the lightweight synthetic schemas used by product-layer tests.
    if base_payload.get("schema_version") == "ppg_frailty.pipeline_config.v2":
        from ppg_frailty.config import validate_config_payload

        base_payload = validate_config_payload(base_payload)
        _apply_execution_device(base_payload, plan.execution.device)
        if plan.execution.device is not None:
            base_payload = validate_config_payload(base_payload)
    combinations = (
        [()]
        if not plan.axes
        else itertools.product(*(axis.values for axis in plan.axes))
    )
    cases: list[ResolvedCase] = []
    base_flat = flatten_mapping(base_payload)
    for index, values in enumerate(combinations):
        changes = {
            axis.path: _normalize_axis_value(axis.path, value)
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
        changed_epoch = changes.get("training.fixed_epochs")
        if (
            not isinstance(changed_epoch, bool)
            and isinstance(changed_epoch, int)
            and changed_epoch in _NAMED_EPOCH_PROFILES
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
        if config.get("schema_version") == "ppg_frailty.pipeline_config.v2":
            from ppg_frailty.config import validate_config_payload

            _drop_model_derived_provenance(config)
            config = validate_config_payload(config)
            if set(changes) & {
                "features.enabled_groups",
                "features.matrix_k",
            }:
                for field in (
                    "registry_id",
                    "file_vector_schema",
                    "matrix_schema",
                ):
                    path = f"features.{field}"
                    derived_changes[path] = config["features"][field]
            derived_changes.update(
                flatten_mapping(
                    {
                        "model": {
                            "architecture_parameters": config["model"][
                                "architecture_parameters"
                            ],
                            "ensemble_size": config["model"]["ensemble_size"],
                        }
                    }
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
        allowed_changed_prefixes = tuple(
            f"{path}."
            for path, value in derived_changes.items()
            if isinstance(value, Mapping)
        )
        undeclared_changed_paths = {
            path
            for path in changed_paths
            if path not in allowed_changed_paths
            and not path.startswith(allowed_changed_prefixes)
        }
        if undeclared_changed_paths:
            raise ValueError(
                f"case {case_id} changed undeclared fields: "
                f"expected={sorted(allowed_changed_paths)}, "
                f"observed={sorted(undeclared_changed_paths)}"
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
            path: {"derived": value, "resolved": _get_dotted(config, path)}
            for path, value in derived_changes.items()
            if _strict_value_key(_get_dotted(config, path))
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
    derived_axis_paths: set[str] = set()
    for axis in plan.axes:
        for value in axis.values:
            for derived_path, derived_value in _derived_axis_values(
                axis.path, value
            ).items():
                derived_axis_paths.add(derived_path)
                if isinstance(derived_value, Mapping):
                    derived_axis_paths.update(
                        flatten_mapping(derived_value, derived_path)
                    )
        if axis.path in {"features.enabled_groups", "features.matrix_k"}:
            derived_axis_paths.update(
                {
                    "features.registry_id",
                    "features.file_vector_schema",
                    "features.matrix_schema",
                }
            )
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

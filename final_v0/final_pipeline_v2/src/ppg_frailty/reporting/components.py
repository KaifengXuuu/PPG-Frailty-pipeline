"""Shared test-component contract tables for every V2 report family.

The table is deliberately built from persisted resolved configurations or the
persisted resolved motion/peak plan.  Hashes are provenance, not input-data
descriptions, and are therefore removed from this human audit projection.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..motion_ids import FORMAL_MOTION_MODEL_ID
from .profiles import annotate_component_row, annotate_component_rows
from .tabular import markdown_column_definitions_block


TEST_COMPONENT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("module_id", "Model / module"),
    ("component_role", "Component role"),
    ("participating_cases", "Cases / phases"),
    ("execution_state", "State"),
    ("input_data", "Input data (values and paths; no hashes)"),
    ("fixed_parameters", "Detailed fixed parameters"),
    ("algorithm_kernel_description", "Algorithm and kernel (≤300 chars)"),
    ("reporter_profile_id", "Reporter profile"),
    ("model_reporter_extension_id", "Model reporter extension"),
    ("algorithm_references", "Algorithm / literature source"),
)

TOP_MODEL_CONFIGURATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("predictive_rank", "Rank"),
    ("case_id", "Case"),
    ("model_id", "Model"),
    ("representation_mode", "Representation"),
    ("resolved_config_path", "Resolved config"),
    ("config_section", "Section"),
    ("parameter_path", "Parameter"),
    ("resolved_value", "Resolved value"),
)

# Human-facing tables are deliberately narrower than the lossless CSV/JSON row.
# The three views together cover every field in ``TEST_COMPONENT_COLUMNS`` while
# keeping unrelated audit concepts out of one horizontally scrolling table.
TEST_COMPONENT_VIEW_SCHEMAS: tuple[
    tuple[str, tuple[tuple[str, str], ...]], ...
] = (
    (
        "Participation and reporter binding",
        (
            ("module_id", "Model / module"),
            ("component_role", "Component role"),
            ("participating_cases", "Cases / phases"),
            ("execution_state", "State"),
            ("reporter_profile_id", "Reporter profile"),
            ("model_reporter_extension_id", "Model reporter extension"),
        ),
    ),
    (
        "Input data and fixed parameters",
        (
            ("module_id", "Model / module"),
            ("component_role", "Component role"),
            ("input_data", "Input data (values and paths; no hashes)"),
            ("fixed_parameters", "Detailed fixed parameters"),
        ),
    ),
    (
        "Algorithm kernel and literature",
        (
            ("module_id", "Model / module"),
            ("component_role", "Component role"),
            ("algorithm_kernel_description", "Algorithm and kernel (≤300 chars)"),
            ("algorithm_references", "Algorithm / literature source"),
        ),
    ),
)

_MAX_HUMAN_FACING_TABLE_COLUMNS = 8

_HASH_KEY = re.compile(r"(?:^|_)(?:sha(?:1|224|256|384|512)?|hash|checksum)(?:_|$)", re.I)


def without_hashes(value: Any) -> Any:
    """Recursively remove provenance hash fields from a report projection."""

    if isinstance(value, Mapping):
        return {
            str(key): without_hashes(item)
            for key, item in value.items()
            if not _HASH_KEY.search(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [without_hashes(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _cell(value: Any) -> str:
    return json.dumps(
        without_hashes(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _row(
    case: str,
    role: str,
    module_id: Any,
    state: str,
    input_data: Mapping[str, Any],
    parameters: Any,
    *,
    description: str = "",
    reporter_profile_id: str | None = None,
) -> dict[str, str]:
    text = str(description).strip()
    if len(text) > 300:
        raise ValueError(f"{module_id} algorithm/kernel description exceeds 300 characters")
    return annotate_component_row({
        "participating_cases": str(case),
        "component_role": str(role),
        "module_id": str(module_id),
        "execution_state": str(state),
        "input_data": _cell(input_data),
        "fixed_parameters": _cell(parameters),
        "algorithm_kernel_description": text,
        **(
            {"reporter_profile_id": reporter_profile_id}
            if reporter_profile_id is not None
            else {}
        ),
    })


def _declared_component_identity(
    module_id: Any,
    *,
    declared_state: str,
) -> tuple[str, str]:
    """Return an explicit N/A identity for an absent persisted declaration.

    Historical and synthetic report fixtures can predate a component section.
    Absence is not an unknown active module: the row remains visible, but is
    marked not executed so the active-module registry still fails closed.
    """

    if module_id is None or not str(module_id).strip() or str(module_id) == "unavailable":
        return "not_declared", "not_executed_not_declared"
    return str(module_id), str(declared_state)


def _group_identical_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[str]] = {}
    fields = tuple(name for name, _ in TEST_COMPONENT_COLUMNS if name != "participating_cases")
    for source in rows:
        key = tuple(str(source.get(field, "")) for field in fields)
        grouped.setdefault(key, []).append(str(source.get("participating_cases", "")))
    output: list[dict[str, str]] = []
    for key, cases in grouped.items():
        row = dict(zip(fields, key))
        assembled = {
            "participating_cases": "; ".join(sorted(dict.fromkeys(cases))),
            **row,
        }
        output.append(
            {
                field: assembled.get(field, "")
                for field, _label in TEST_COMPONENT_COLUMNS
            }
        )
    return sorted(
        output,
        key=lambda item: (
            item["component_role"], item["module_id"], item["participating_cases"]
        ),
    )


def markdown_test_component_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render canonical narrow Markdown views reused verbatim in two files.

    The lossless table serialization remains one row with
    :data:`TEST_COMPONENT_COLUMNS`; this function changes presentation only.
    """

    if not rows:
        return "N/A — no persisted component configuration was available."
    annotated = annotate_component_rows(rows)
    lines: list[str] = []
    for title, schema in TEST_COMPONENT_VIEW_SCHEMAS:
        if len(schema) > _MAX_HUMAN_FACING_TABLE_COLUMNS:
            raise ValueError(
                f"human-facing test-component table {title!r} has "
                f"{len(schema)} columns; maximum is {_MAX_HUMAN_FACING_TABLE_COLUMNS}"
            )
        headings = [label for _field, label in schema]
        lines.extend(
            (
                f"### {title}",
                "",
                "| " + " | ".join(headings) + " |",
                "|" + "|".join("---" for _ in headings) + "|",
            )
        )
        for row in annotated:
            rendered = []
            for field, _label in schema:
                value = str(row.get(field, "")).replace("|", r"\|").replace("\n", " ")
                rendered.append(value)
            lines.append("| " + " | ".join(rendered) + " |")
        lines.extend(
            (
                "",
                markdown_column_definitions_block(
                    [field for field, _label in schema],
                    display_labels=[label for _field, label in schema],
                ),
                "",
            )
        )
    return "\n".join(lines).rstrip()


def write_test_component_markdown(root: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write the required standalone file containing the exact report table."""

    target = Path(root) / "TEST_COMPONENTS.md"
    target.write_text(
        "# Test models, modules, inputs, and fixed parameters\n\n"
        "This table is generated from the persisted resolved execution contract. "
        "Input data are named by dataset/path, signal view, channels, units, sampling "
        "rate, and window—not by a hash. Provenance hashes remain in the lossless "
        "manifests and resolved configs.\n\n"
        + markdown_test_component_table(rows)
        + "\n",
        encoding="utf-8",
    )
    return target


def _resolved_case_configs(
    root: Path,
    manifest: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    output: list[tuple[str, Mapping[str, Any]]] = []
    for case in manifest.get("cases", ()):
        if not isinstance(case, Mapping):
            continue
        case_id = str(case.get("case_id", "unknown_case"))
        raw = case.get("resolved_config_path")
        if not isinstance(raw, str) or not raw.strip():
            continue
        relative = Path(raw)
        if relative.is_absolute():
            continue
        target = (root / relative).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            continue
        if not target.is_file():
            continue
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            output.append((case_id, payload))
    return output


def _flatten_resolved_config(
    value: Any,
    *,
    prefix: str = "",
) -> list[tuple[str, Any]]:
    """Flatten one hash-free resolved config without losing empty containers."""

    cleaned = without_hashes(value)
    if isinstance(cleaned, Mapping):
        if not cleaned:
            return [(prefix, {})]
        rows: list[tuple[str, Any]] = []
        for key in sorted(cleaned, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_resolved_config(cleaned[key], prefix=path))
        return rows
    # Lists are kept as a complete ordered value. Expanding list indices would
    # make channel/role orders harder to audit and would not add information.
    return [(prefix, cleaned)]


def build_top_model_configuration_rows(
    root: str | Path,
    manifest: Mapping[str, Any],
    predictive_leaderboard: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return complete long-form resolved configs for the ranked top models.

    The leaderboard supplies ranking only. Configuration values are reread from
    the immutable per-case resolved YAML recorded by the manifest. Provenance
    hashes are deliberately excluded because named input data and executable
    values—not hashes—belong in this human configuration table.
    """

    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if top_k == 0:
        return []
    root_path = Path(root).resolve()
    config_by_case = dict(_resolved_case_configs(root_path, manifest))
    manifest_by_case = {
        str(row.get("case_id")): row
        for row in manifest.get("cases", ())
        if isinstance(row, Mapping) and row.get("case_id") not in (None, "")
    }
    ranked = sorted(
        (
            row
            for row in predictive_leaderboard
            if str(row.get("case_id", "")) in config_by_case
        ),
        key=lambda row: (
            int(row.get("predictive_rank", 10**9)),
            str(row.get("case_id", "")),
        ),
    )[:top_k]
    output: list[dict[str, Any]] = []
    for fallback_rank, ranked_row in enumerate(ranked, start=1):
        case_id = str(ranked_row["case_id"])
        config = config_by_case[case_id]
        model = config.get("model", {})
        model = model if isinstance(model, Mapping) else {}
        manifest_row = manifest_by_case.get(case_id, {})
        resolved_path = str(manifest_row.get("resolved_config_path", "N/A"))
        rank = int(ranked_row.get("predictive_rank", fallback_rank))
        for parameter_path, resolved_value in _flatten_resolved_config(config):
            output.append(
                {
                    "predictive_rank": rank,
                    "case_id": case_id,
                    "model_id": str(model.get("model_id", "N/A")),
                    "representation_mode": str(
                        config.get("representation_mode", "N/A")
                    ),
                    "resolved_config_path": resolved_path,
                    "config_section": parameter_path.split(".", 1)[0],
                    "parameter_path": parameter_path,
                    "resolved_value": _cell(resolved_value),
                }
            )
    return output


def _legacy_bridge_profiles(root: Path) -> dict[str, Mapping[str, Any]]:
    """Return hash-bound effective controls keyed by persisted catalog case id."""

    source = root / "study_plan.yaml"
    if not source.is_file():
        return {}
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    bridge = payload.get("legacy_bridge")
    if not isinstance(bridge, Mapping):
        return {}
    output: dict[str, Mapping[str, Any]] = {}
    for profile in bridge.get("profiles", ()):
        if not isinstance(profile, Mapping):
            continue
        case_id = str(profile.get("catalog_case_id", ""))
        controls = profile.get("controls")
        if case_id and isinstance(controls, Mapping):
            output[case_id] = profile
    return output


def _legacy_bridge_component_rows(
    case_id: str,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Describe actual field-driven bridge execution, not its catalog carrier."""

    common = _pipeline_input_context(config)
    controls = dict(profile["controls"])
    legacy_imu = controls["imu_preprocessing"] == "legacy_filtered_axes"
    channels = (
        ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"]
        if legacy_imu
        else ["RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"]
    )
    window = {
        "length_s": controls["window_seconds"],
        "hop_s": controls["hop_seconds"],
        "historical_retained_fraction": controls["historical_retained_fraction"],
        "cap_per_file": controls["max_windows_per_file"],
        "allow_short_record_padding": controls["allow_short_record_padding"],
    }
    model = {
        **dict(config.get("model", {})),
        "input_channels": len(channels),
        "input_channel_order": channels,
    }
    model_input = {
        **common,
        "representation_mode": "raw",
        "signal_view": "legacy_bridge_effective_DL_tensor",
        "channels": channels,
        "sampling_rate_hz": controls["target_fs_hz"],
        "window": window,
        "ppg_preprocessing": controls["ppg_preprocessing"],
        "imu_preprocessing": controls["imu_preprocessing"],
        "normalization": controls["normalization"],
    }
    trainer = {
        "optimizer": controls["optimizer"],
        "batch_size": controls["batch_size"],
        "fixed_epochs": controls["fixed_epochs"],
        "learning_rate": controls["learning_rate"],
        "weight_decay": controls["weight_decay"],
        "sampler": controls["sampler"],
        "class_weighting": controls["class_weighting"],
        "training_metric_aggregation_rule": controls[
            "training_metric_aggregation_rule"
        ],
        "training_seed": 42,
        "device": config.get("training", {}).get("device"),
        "outer_labels_visible_to_trainer": False,
    }
    return [
        _row(
            case_id, "dataset_adapter", common.get("dataset_id"), "enabled",
            common, config.get("manifest", {}),
        ),
        _row(
            case_id, "split_registry",
            config.get("splits", {}).get("registry_id"), "enabled",
            {**common, "groups": "participant_id", "labels": "frailty_class"},
            config.get("splits", {}),
        ),
        _row(
            case_id, "legacy_bridge_effective_profile",
            profile.get("profile_id"), "executed_hash_bound_controls",
            model_input,
            {
                "factor_id": profile.get("factor_id"),
                "changed_control_paths": profile.get("changed_control_paths"),
                "controls": controls,
                "interpretation": profile.get("interpretation"),
            },
        ),
        _row(
            case_id, "ppg_preprocessing", controls["ppg_preprocessing"],
            "enabled", {**common, "input_channels": ["RED", "IR"]},
            {"method": controls["ppg_preprocessing"]},
        ),
        _row(
            case_id, "imu_preprocessing", controls["imu_preprocessing"],
            "enabled",
            {
                **common,
                "input_channels": ["AX", "AY", "AZ", "GX", "GY", "GZ"],
                "output_channels": channels[2:],
            },
            {"method": controls["imu_preprocessing"]},
        ),
        _row(
            case_id, "signal_views_and_scaling", controls["normalization"],
            "enabled", model_input,
            {
                "normalization": controls["normalization"],
                "sampling_rate_hz": controls["target_fs_hz"],
            },
        ),
        _row(
            case_id, "window_planner", "legacy_bridge_reviewed_window_plan_v1",
            "enabled", model_input, window,
        ),
        _row(
            case_id, "representation", "raw", "enabled", model_input,
            {"representation_mode": "raw", "input_contract": model_input},
        ),
        _row(
            case_id, "classifier", model.get("model_id"), "enabled",
            model_input, model,
        ),
        _row(
            case_id, "trainer", controls["optimizer"], "enabled",
            {**common, "model_input": model_input, "labels": "participant frailty class"},
            trainer,
        ),
        _row(
            case_id, "aggregation",
            controls["primary_report_aggregation_view"], "enabled",
            {"input_data": "held-out window/file probabilities", "roles": common.get("roles")},
            {
                "primary_report_aggregation_view": controls[
                    "primary_report_aggregation_view"
                ],
                "training_metric_aggregation_rule": controls[
                    "training_metric_aggregation_rule"
                ],
            },
        ),
        _row(
            case_id, "evaluation",
            config.get("evaluation", {}).get("primary_metric"), "enabled",
            {
                "input_data": "held-out participant predictions and frailty labels",
                "class_order": config.get("manifest", {}).get("class_name_order"),
            },
            config.get("evaluation", {}),
        ),
    ]


def _pipeline_input_context(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = config.get("manifest", {})
    signal = config.get("signal", {})
    return {
        "dataset_id": manifest.get("source_dataset_id"),
        "manifest_path": manifest.get("path"),
        "participants": manifest.get("expected_participant_count"),
        "records": manifest.get("expected_record_count"),
        "roles": config.get("roles"),
        "source_channels": manifest.get("channel_order", signal.get("channel_order")),
        "source_units": {
            "PPG": signal.get("ppg_native_unit"),
            "ACC": signal.get("accelerometer_input_unit"),
            "GYRO": signal.get("gyroscope_input_unit"),
        },
        "pipeline_fs_hz": signal.get("internal_fs_hz"),
    }


def _representation_input(config: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(config.get("representation_mode", "unknown"))
    model = config.get("model", {})
    features = config.get("features", {})
    windows = config.get("windows", {})
    if mode == "raw":
        view = "x_dl_all8_window_norm"
        details = {
            "channels": model.get("input_channel_order"),
            "window": windows.get("raw_dl"),
        }
    elif mode == "fusion":
        view = "x_dl_all8_window_norm + engineered feature vector"
        details = {
            "raw_channels": model.get("raw_input_channel_order"),
            "raw_window": windows.get("raw_dl"),
            "feature_schema": features.get("file_vector_schema"),
        }
    elif mode == "feature_matrix":
        view = "x_analysis/x_native + processed_imu_physical engineering sequence"
        details = {
            "window": windows.get("engineering"),
            "feature_schema": features.get("engineering_sequence_schema"),
            "matrix_schema": features.get("matrix_schema"),
            "matrix_k": features.get("matrix_k"),
        }
    else:
        view = "x_analysis/x_native + processed_imu_physical engineered file vector"
        details = {
            "window": windows.get("engineering"),
            "feature_schema": features.get("file_vector_schema"),
        }
    return {"representation_mode": mode, "signal_view": view, **details}


def _trainer_component(config: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    """Report executable trainer semantics without assigning DL epochs to sklearn."""

    model = dict(config.get("model", {}))
    training = dict(config.get("training", {}))
    if str(config.get("representation_mode")) == "feature_vector":
        architecture = model.get("architecture_parameters")
        architecture = (
            dict(architecture) if isinstance(architecture, Mapping) else {}
        )
        estimator = str(
            architecture.get("estimator", model.get("model_id", "sklearn_estimator"))
        )
        return (
            estimator,
            {
                "fit_semantics": "one estimator fit on each outer-training fold",
                "epoch_rule": "not_applicable_classical_estimator",
                "estimator_parameters": model,
            },
        )
    return str(training.get("optimizer", "unavailable")), training


def build_pipeline_test_component_rows(
    root: str | Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Build module rows from every persisted per-case pipeline config."""

    rows: list[dict[str, str]] = []
    bridge_profiles = _legacy_bridge_profiles(Path(root))
    for case_id, config in _resolved_case_configs(Path(root), manifest):
        if case_id in bridge_profiles:
            rows.extend(
                _legacy_bridge_component_rows(
                    case_id, config, bridge_profiles[case_id]
                )
            )
            continue
        common = _pipeline_input_context(config)
        signal = config.get("signal", {})
        artifact = config.get("artifact", {})
        quality = config.get("quality", {})
        model = config.get("model", {})
        motion_detector = (
            artifact.get("motion_detector", {})
            if isinstance(artifact.get("motion_detector"), Mapping)
            else {}
        )
        reducer_id = str(artifact.get("reducer", "identity"))
        reducer_parameters = (
            artifact.get("parameters", {})
            if isinstance(artifact.get("parameters"), Mapping)
            else {}
        )
        reducer_description = ""
        runtime_reducer_version = "unavailable"
        resolved_reducer_parameters: Mapping[str, Any] = reducer_parameters
        try:
            from ..artifact import reducer_audit_metadata

            reducer_metadata = reducer_audit_metadata(reducer_id, reducer_parameters)
            reducer_description = str(
                reducer_metadata["algorithm_kernel_description"]
            )
            runtime_reducer_version = str(reducer_metadata["reducer_version"])
            resolved_reducer_parameters = reducer_metadata["resolved_parameters"]
        except (KeyError, ValueError):
            # A separately registered experimental reducer still appears with
            # its full resolved config; only missing source-local prose is blank.
            pass
        representation = _representation_input(config)
        trainer_id, trainer_parameters = _trainer_component(config)
        dataset_id, dataset_state = _declared_component_identity(
            common.get("dataset_id"), declared_state="enabled"
        )
        split_id, split_state = _declared_component_identity(
            config.get("splits", {}).get("registry_id"), declared_state="enabled"
        )
        ppg_filter_id, ppg_filter_state = _declared_component_identity(
            signal.get("ppg_filter", {}).get("family"), declared_state="enabled"
        )
        imu_id, imu_state = _declared_component_identity(
            signal.get("imu", {}).get("gravity_method"), declared_state="enabled"
        )
        peak_id, peak_state = _declared_component_identity(
            signal.get("peak_detector", {}).get("detector_id"),
            declared_state="enabled",
        )
        signal_views_id, signal_views_state = _declared_component_identity(
            "parallel_physical_analysis_and_dl_views" if signal else None,
            declared_state="enabled",
        )
        window_id, window_state = _declared_component_identity(
            config.get("windows", {}).get("shared_planner_version"),
            declared_state="enabled",
        )
        quality_mode = quality.get("mode")
        sqi_id, sqi_state = _declared_component_identity(
            f"quality_{quality_mode}" if quality_mode is not None else None,
            declared_state=(
                "enabled" if quality_mode != "off" else "disabled_control"
            ),
        )
        motion_declared = (
            "motion_detector_enabled" in artifact or bool(motion_detector)
        )
        motion_id, motion_state = _declared_component_identity(
            (
                motion_detector.get("model_id", FORMAL_MOTION_MODEL_ID)
                if motion_declared
                else None
            ),
            declared_state=(
                "enabled"
                if artifact.get("motion_detector_enabled")
                else "disabled_control"
            ),
        )
        denoiser_declared = (
            "denoiser_enabled" in artifact or "reducer" in artifact
        )
        denoiser_id, denoiser_state = _declared_component_identity(
            reducer_id if denoiser_declared else None,
            declared_state=(
                "enabled"
                if artifact.get("denoiser_enabled")
                else "identity_or_disabled_control"
            ),
        )
        feature_id, feature_state = _declared_component_identity(
            config.get("features", {}).get("registry_id"),
            declared_state=(
                "enabled"
                if config.get("representation_mode")
                in {"feature_vector", "feature_matrix", "fusion"}
                else "auxiliary_not_classifier_input"
            ),
        )
        representation_id, representation_state = _declared_component_identity(
            config.get("representation_mode"), declared_state="enabled"
        )
        classifier_id, classifier_state = _declared_component_identity(
            model.get("model_id"), declared_state="enabled"
        )
        resolved_trainer_id, trainer_state = _declared_component_identity(
            trainer_id, declared_state="enabled"
        )
        aggregation_id, aggregation_state = _declared_component_identity(
            config.get("aggregation", {}).get("balance_line"),
            declared_state="enabled",
        )
        evaluation_id, evaluation_state = _declared_component_identity(
            config.get("evaluation", {}).get("primary_metric"),
            declared_state="enabled",
        )
        sqi_profile = (
            "sqi_route_coverage_v1"
            if quality_mode is not None and quality_mode != "off"
            else "audit_provenance_v1"
        )
        motion_profile = (
            "motion_route_component_v1"
            if motion_state == "enabled"
            else "audit_provenance_v1"
        )
        denoiser_profile = (
            "frailty_denoiser_route_v1"
            if denoiser_state == "enabled"
            else "audit_provenance_v1"
        )
        rows.extend(
            (
                _row(case_id, "dataset_adapter", dataset_id, dataset_state, common, config.get("manifest", {})),
                _row(case_id, "split_registry", split_id, split_state, {**common, "groups": "participant_id", "labels": "frailty_class"}, config.get("splits", {})),
                _row(case_id, "ppg_preprocessing", ppg_filter_id, ppg_filter_state, {**common, "input_channels": ["RED", "IR"], "input_view": "repaired native PPG"}, {"ppg_filter": signal.get("ppg_filter"), "gap_repair": signal.get("gap_repair"), "analysis_view": signal.get("analysis_view")}),
                _row(case_id, "imu_preprocessing", imu_id, imu_state, {**common, "input_channels": ["AX", "AY", "AZ", "GX", "GY", "GZ"], "output_view": "processed_imu_physical"}, signal.get("imu", {})),
                _row(case_id, "peak_detector", peak_id, peak_state, {**common, "input_view": "x_analysis/x_native", "channels": ["RED", "IR"]}, signal.get("peak_detector", {})),
                _row(case_id, "signal_views_and_scaling", signal_views_id, signal_views_state, {**common, "views": ["processed_imu_physical", "x_dl_all8_window_norm", "x_analysis/x_native"]}, {"normalization": signal.get("normalization"), "dl_resampling": signal.get("dl_resampling")}),
                _row(case_id, "window_planner", window_id, window_state, {**common, "input_views": ["x_dl_all8_window_norm", "x_analysis/x_native", "processed_imu_physical"]}, config.get("windows", {})),
                _row(case_id, "sqi", sqi_id, sqi_state, {**common, "input_views": ["x_analysis", "pulse train", "processed_imu_physical"]}, quality, reporter_profile_id=sqi_profile),
                _row(case_id, "motion_detector", motion_id, motion_state, {**common, "input_view": "RED/IR + processed physical A_dyn/GX/GY/GZ"}, {"enabled": artifact.get("motion_detector_enabled"), **dict(motion_detector)}, reporter_profile_id=motion_profile),
                _row(case_id, "denoiser", denoiser_id, denoiser_state, {**common, "input_views": ["filtered RED/IR", "processed_imu_physical"]}, {"denoiser_enabled": artifact.get("denoiser_enabled"), "reducer": reducer_id if denoiser_declared else "not_declared", "declared_reducer_version": artifact.get("reducer_version"), "runtime_reducer_version": runtime_reducer_version if denoiser_declared else "not_applicable", "resolved_parameters": resolved_reducer_parameters if denoiser_declared else {}, "degraded_policy": artifact.get("degraded_policy"), "failure_action": artifact.get("failure_action")}, description=reducer_description if denoiser_declared else "", reporter_profile_id=denoiser_profile),
                _row(case_id, "feature_extractor", feature_id, feature_state, {**common, "input_views": ["x_analysis/x_native", "processed_imu_physical"], "engineering_window": config.get("windows", {}).get("engineering")}, config.get("features", {})),
                _row(case_id, "representation", representation_id, representation_state, {**common, **representation}, {"representation_mode": config.get("representation_mode"), "input_contract": representation}),
                _row(case_id, "classifier", classifier_id, classifier_state, {**common, **representation}, model),
                _row(case_id, "trainer", resolved_trainer_id, trainer_state, {**common, "model_input": representation, "labels": "participant frailty class"}, trainer_parameters),
                _row(case_id, "aggregation", aggregation_id, aggregation_state, {"input_data": "held-out window/file probabilities", "roles": common.get("roles")}, config.get("aggregation", {})),
                _row(case_id, "evaluation", evaluation_id, evaluation_state, {"input_data": "held-out participant predictions and frailty labels", "class_order": config.get("manifest", {}).get("class_name_order")}, config.get("evaluation", {})),
            )
        )
    return _group_identical_rows(rows)


def _motion_architecture_and_training(device: str) -> dict[str, Any]:
    from ..models.motion import LightCnnArchitecture
    from ..quality.motion_adapters import FormalMotionTrainerConfig
    from ..representations.motion import (
        MOTION_NETWORK_CHANNEL_SCHEMA,
        motion_network_schema_payload,
    )

    return {
        "architecture": asdict(LightCnnArchitecture(tuple(MOTION_NETWORK_CHANNEL_SCHEMA))),
        "training": asdict(FormalMotionTrainerConfig(device=device)),
        "tensor": motion_network_schema_payload(),
    }


def _peak_detector_parameters(algorithm_id: str, declared: Mapping[str, Any]) -> dict[str, Any]:
    if algorithm_id == "aboy_project_v1":
        return {
            **dict(declared),
            "input_preprocessing": "shared repaired PPG then shared 0.2-8 Hz analysis filter",
            "block_s": 10.0,
            "adaptive_bandpass_hz": "0.5 to min(8,max(1.5,3*(1+HRI)))",
            "bandpass_order": 2,
            "initial_hri": 0.0,
            "pulse_rate_bpm": [35.0, 210.0],
            "prominence_fraction": 0.25,
            "mad_scale": 1.4826,
            "mad_limit": 4.0,
        }
    if algorithm_id == "aboy_project_v2":
        return {
            **dict(declared),
            "owned_highpass_hz": 0.2,
            "owned_highpass_order": 2,
            "block_s": 10.0,
            "adaptive_bandpass_hz": "0.5 to min(8,max(1.5,3*(1+HRI)))",
            "bandpass_order": 2,
            "initial_hri": 0.0,
            "pulse_rate_bpm": [35.0, 210.0],
            "prominence_fraction": 0.25,
            "interval_merge_limits": [0.5, 1.8],
            "mad_scale": 1.4826,
            "mad_limit": 4.0,
        }
    if algorithm_id == "msptdfast_v2_3_python_port":
        from ..peaks.msptdfast_v2 import resolve_parameters

        return {**dict(declared), "parameters": resolve_parameters(declared.get("parameters"))}
    return dict(declared)


def build_motion_peak_test_component_rows(
    resolved_plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    study_root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Build the same contract rows for Stage5-pre and peak ablation reports."""

    rows: list[dict[str, str]] = []
    ptt = dict(resolved_plan.get("ptt_dataset", {}))
    ptt_input = {
        "dataset_id": ptt.get("dataset_id"),
        "dataset_root": ptt.get("root"),
        "participants": ptt.get("participant_count"),
        "records": ptt.get("record_count"),
        "activities": ptt.get("activities", resolved_plan.get("activities")),
        "channels": ptt.get("distal_channels"),
        "ecg_peak_annotation_column": ptt.get("ecg_peak_annotation_column"),
        "source_fs_hz": ptt.get("source_fs_hz", 500.0),
        "pipeline_fs_hz": ptt.get("pipeline_fs_hz", 400.0),
    }
    study_type = str(manifest.get("study_type", resolved_plan.get("study_type", "")))
    rows.append(_row("all", "dataset_adapter", ptt.get("dataset_id"), "enabled", ptt_input, ptt))
    if study_type == "stage5_pre_motion_ptt":
        from ..artifact import reducer_audit_metadata
        from ..data.manifest import M2_DATASET_VERSION_ID, M2_FILE_MANIFEST
        from ..motion_ids import FORMAL_MOTION_MODEL_ID
        from ..signal.motion_imu import RollPitchEkfConfig

        detector = dict(resolved_plan.get("motion_detector", {}))
        trainer = _motion_architecture_and_training(str(detector.get("training_device", "cuda")))

        def persisted_threshold(
            stage_id: str,
            filename: str,
            field: str,
        ) -> dict[str, Any]:
            if study_root is None:
                return {
                    "provenance_status": "not_available_without_study_root",
                    "fit_scope": "not_inferred_from_current_defaults",
                }
            stage = manifest.get("stages", {}).get(stage_id, {})
            if not isinstance(stage, Mapping) or not stage.get("artifact_dir"):
                return {
                    "provenance_status": "stage_not_executed",
                    "fit_scope": "not_applicable",
                }
            source = Path(study_root) / str(stage["artifact_dir"]) / filename
            if not source.is_file():
                return {
                    "provenance_status": "persisted_threshold_evidence_missing",
                    "fit_scope": "not_inferred_from_current_defaults",
                }
            payload = json.loads(source.read_text(encoding="utf-8"))
            threshold = payload.get(field, {})
            if not isinstance(threshold, Mapping):
                return {
                    "provenance_status": "persisted_threshold_field_missing",
                    "fit_scope": "not_inferred_from_current_defaults",
                }
            projected_fields = (
                "schema_version",
                "threshold_rule_id",
                "fit_scope",
                "score_origin",
                "score_space",
                "center_statistic",
                "participant_weighting",
                "observed_row_count",
                "static_center",
                "motion_center",
                "threshold",
            )
            return {
                "provenance_status": "read_from_persisted_execution_evidence",
                **{
                    key: threshold.get(key)
                    for key in projected_fields
                    if key in threshold
                },
            }

        internal_threshold = persisted_threshold(
            "internal_motion_oof",
            "motion_internal_evidence.json",
            "final_threshold",
        )
        reverse_threshold = persisted_threshold(
            "ptt_motion_training_ablation",
            "motion_ptt_training_evidence.json",
            "deployment_threshold",
        )
        internal_input = {
            "dataset_id": M2_DATASET_VERSION_ID,
            "manifest_path": M2_FILE_MANIFEST.as_posix(),
            "participants": detector.get("internal_participant_count"),
            "roles": ["B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2"],
            "labels": {"static": ["B", "R1", "R2", "R3", "R4"], "motion": ["S1", "S2", "W1", "W2"]},
            "channels": trainer["tensor"]["channel_schema"],
            "units": trainer["tensor"]["channel_units"],
            "fs_hz": trainer["tensor"]["fs_hz"],
            "window_s": trainer["tensor"]["window_s"],
            "hop_s": trainer["tensor"]["hop_s"],
        }
        rows.extend(
            (
                _row("Frailty29 OOF + all-29 final → PTT22", "motion_detector", FORMAL_MOTION_MODEL_ID, "executed", {"training": internal_input, "frozen_evaluation": ptt_input}, {**trainer, "split": detector.get("split"), "threshold": internal_threshold, "external_fit_or_recalibration": detector.get("external_fit_or_recalibration")}),
                _row("PTT22 OOF + all-22 final → Frailty29", "motion_detector_reverse_ablation", FORMAL_MOTION_MODEL_ID, "executed" if detector.get("reverse_ablation", {}).get("enabled") else "disabled", {"training": ptt_input, "frozen_evaluation": internal_input}, {**trainer, "reverse_ablation": detector.get("reverse_ablation"), "threshold": reverse_threshold}),
                _row("all motion-detector phases", "imu_preprocessing", "calibrated_roll_pitch_ekf", "executed", {"input_channels": ["AX", "AY", "AZ", "GX", "GY", "GZ"], "internal_units": {"ACC": "g", "GYRO": "deg/s"}, "ptt_units": {"ACC": "m/s² identity conversion", "GYRO": "deg/s → rad/s"}, "output_view": "processed_imu_physical"}, asdict(RollPitchEkfConfig())),
                _row("Frailty29-trained deployment route", "motion_threshold", "participant_balanced_midpoint", "executed", {"input_data": internal_threshold.get("score_origin", "persisted evidence unavailable")}, {**internal_threshold, "held_out_or_cross_dataset_tuning": False, "deployment_application": "frozen once"}),
            )
        )
        benchmark = dict(resolved_plan.get("denoiser_benchmark", {}))
        denoiser_stage = manifest.get("stages", {}).get("ptt_denoiser_benchmark", {})
        denoiser_executed = isinstance(denoiser_stage, Mapping) and denoiser_stage.get("status") == "passed"
        for reducer_id in benchmark.get("reducers", ()):
            metadata = reducer_audit_metadata(str(reducer_id))
            rows.append(
                _row(
                    "PTT denoiser benchmark",
                    "denoiser",
                    metadata["reducer_id"],
                    "executed" if denoiser_executed else "configured_not_executed",
                    {**ptt_input, "activities": benchmark.get("activities"), "segments_s": benchmark.get("segment_s"), "input_ppg": "shared 0.2-8 Hz filtered RED/IR", "input_imu": "processed_imu_physical six axes", "scoring_peak_detector": benchmark.get("scoring_peak_detector")},
                    {"reducer_version": metadata["reducer_version"], "resolved_parameters": metadata["resolved_parameters"], "fit_scope": benchmark.get("reducer_fit_scope"), "validation": benchmark.get("validation")},
                    description=metadata["algorithm_kernel_description"],
                )
            )
    else:
        validation = dict(resolved_plan.get("validation", {}))
        beat_reporter_profile = (
            "beat_detector_recording_v1"
            if float(validation.get("lag_window_s", 0.0)) == 300.0
            and float(validation.get("beat_tolerance_s", 0.0)) == 0.15
            else "beat_detector_legacy_persisted_v1"
        )
        for declared in resolved_plan.get("algorithms", ()):
            if not isinstance(declared, Mapping):
                continue
            algorithm_id = str(declared.get("algorithm_id"))
            module_id = str(declared.get("module_id", algorithm_id))
            rows.append(
                _row(
                    "PTT sit static peak ablation",
                    "peak_detector",
                    algorithm_id,
                    "executed",
                    {**ptt_input, "selected_activity": "sit", "input_view": resolved_plan.get("detector_input"), "channels": ["RED", "IR"], "scoring_windows_s": validation.get("lag_window_s")},
                    {
                        "registered_module_id": module_id,
                        "display_name": declared.get("display_name", algorithm_id),
                        **_peak_detector_parameters(module_id, declared),
                    },
                    reporter_profile_id=beat_reporter_profile,
                )
            )
        rows.append(
            _row(
                "PTT sit static peak ablation",
                "peak_validation",
                validation.get("alignment"),
                "executed",
                {"reference": validation.get("reference"), "annotation_column": ptt.get("ecg_peak_annotation_column"), "predictions": "detected PPG pulse times"},
                validation,
                reporter_profile_id=beat_reporter_profile,
            )
        )
    return _group_identical_rows(rows)


__all__ = [
    "TEST_COMPONENT_COLUMNS",
    "TEST_COMPONENT_VIEW_SCHEMAS",
    "TOP_MODEL_CONFIGURATION_COLUMNS",
    "build_motion_peak_test_component_rows",
    "build_pipeline_test_component_rows",
    "build_top_model_configuration_rows",
    "markdown_test_component_table",
    "without_hashes",
    "write_test_component_markdown",
]

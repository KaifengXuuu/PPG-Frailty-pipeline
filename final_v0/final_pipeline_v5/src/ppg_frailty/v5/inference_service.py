"""Pretrained participant inference through the frozen V2 raw workflow.

This module is deliberately an application adapter, not a second signal
pipeline.  It constructs audited live-input records and calls the same V2
preprocessing, windowing, resampling, model-input binding, prediction and
hierarchical aggregation functions used by outer CV.

Only workflows whose model-ready representation has no missing fold-fitted
transform are accepted.  The first supported contract is the user-selected
``finalcase`` raw route (quality/artifact modules off and ``raw_imu=none``).
Unsupported configurations fail closed instead of approximating training-time
semantics.
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from ..contracts import ManifestRow
from ..models import ModelInputSpec, normalize_model_id
from ..pipeline import PipelinePaths, _load_record, preflight_pipeline
from ..provenance import stable_payload_sha256
from ..training.bundle import FrozenRepresentationTransformArchive, load_bundle
from .environment import prepare_deterministic_runtime, require_environment

_SCHEMA = "ppg_frailty.v5_participant_inference.v1"
_MODEL_EXPORT_SCHEMA = "ppg_frailty.v5_model_config_export.v1"
_TRUSTED_REFIT_ROLE = "all29_full_cohort_refit"
_CHANNELS = ("RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ")
_MODEL_CHANNELS = (
    "RED",
    "IR",
    "A_dyn_x",
    "A_dyn_y",
    "A_dyn_z",
    "GX",
    "GY",
    "GZ",
)
_UNITS = {
    "RED": "raw_device_counts_adc_scale_unknown",
    "IR": "raw_device_counts_adc_scale_unknown",
    "AX": "g_source_declared",
    "AY": "g_source_declared",
    "AZ": "g_source_declared",
    "GX": "degree_per_second_source_declared",
    "GY": "degree_per_second_source_declared",
    "GZ": "degree_per_second_source_declared",
}
_DYNAMIC_FAMILIES = frozenset({"R", "S", "W"})
_FIXED_GRID_SYNCHRONY = "row_aligned_eight_channel_fixed_grid_no_timestamp"
_SOURCE_CONTRACT = {
    "provenance": "user_declared",
    "sampling_rate_hz": 400.0,
    "channel_order": _CHANNELS,
    "accelerometer_unit": "g",
    "gyroscope_unit": "deg/s",
    "synchrony": _FIXED_GRID_SYNCHRONY,
}


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _inside(path: str | Path, root: Path, *, label: str) -> Path:
    raw = Path(path)
    target = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside {root}") from error
    if not target.exists():
        raise FileNotFoundError(target)
    return target


def _repository_input(
    path: str | Path,
    *,
    pipeline_root: Path,
    repository_root: Path,
    label: str,
) -> Path:
    raw = Path(path)
    target = raw.resolve() if raw.is_absolute() else (pipeline_root / raw).resolve()
    try:
        target.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside {repository_root}") from error
    if not target.exists():
        raise FileNotFoundError(target)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _role_family(role: str) -> str:
    text = str(role).strip().upper()
    if text == "B":
        return text
    if text[:1] in _DYNAMIC_FAMILIES and (len(text) == 1 or text[1:].isdigit()):
        return text[0]
    raise ValueError(f"unsupported role {role!r}; expected B/R*/S*/W*")


def _concrete_role(role: str) -> str:
    """Return the normalized concrete role without collapsing R1/S2/W1."""

    concrete = str(role).strip().upper()
    _role_family(concrete)
    return concrete


def _assert_source_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Check CSV assumptions that values and a header cannot reveal."""
    contract = _mapping(payload.get("source_contract"), label="source_contract")
    missing, extra = set(_SOURCE_CONTRACT) - set(contract), set(contract) - set(_SOURCE_CONTRACT)
    if missing:
        raise ValueError(f"source_contract is missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"source_contract contains unsupported fields: {sorted(extra)}")
    for field, expected in _SOURCE_CONTRACT.items():
        observed = tuple(contract[field]) if field == "channel_order" else contract[field]
        if isinstance(observed, bool) or observed != expected:
            raise ValueError(f"source_contract.{field} must be exactly {expected}")
    return contract


def _load_input_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(text)
    else:
        raise ValueError("inference manifest must be YAML or JSON")
    return (
        _mapping(value, label="inference manifest"),
        hashlib.sha256(raw).hexdigest(),
    )


def _label_id(value: Any, class_names: tuple[str, ...]) -> int | None:
    if value is None or str(value).strip().lower() in {
            "",
            "none",
            "unknown",
            "unlabelled",
    }:
        return None
    if isinstance(value, bool):
        raise ValueError("participant label cannot be boolean")
    try:
        number = int(value)
    except (TypeError, ValueError):
        normalized = str(value).strip().casefold()
        matches = [index for index, name in enumerate(class_names) if name.casefold() == normalized]
        if len(matches) != 1:
            raise ValueError("label must be one of 0/1/2 or the configured class names: " + ", ".join(class_names))
        return matches[0]
    if str(number) != str(value).strip() and not isinstance(value, int):
        raise ValueError("numeric participant labels must be exact integers")
    if number not in range(len(class_names)):
        raise ValueError(f"participant label must be in 0..{len(class_names) - 1}")
    return number


def _csv_identity(path: Path) -> tuple[str, int]:
    """Read only CSV structure here; the canonical V2 loader audits values/QC."""

    content_hash = _sha256(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if tuple(header or ()) != _CHANNELS:
            raise ValueError(f"live CSV channel order must be {_CHANNELS}: {path}")
        count = 0
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(_CHANNELS):
                raise ValueError(f"live CSV column-count drift at {path}:{line_number}")
            count += 1
    if count <= 0:
        raise ValueError(f"live CSV has no samples: {path}")
    return content_hash, count


def _resolve_export(
    export_directory: Path,
    case_id: str | None,
) -> tuple[Path, Path, Mapping[str, Any]]:
    manifest_path = export_directory / "export_manifest.json"
    manifest = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        label=str(manifest_path),
    )
    if manifest.get("schema_version") != _MODEL_EXPORT_SCHEMA:
        raise ValueError("unsupported model_config export schema")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("model_config export contains no cases")
    cases = [_mapping(row, label="model_config case") for row in raw_cases]
    selected = next(
        (row for row in cases if str(row.get("case_id")) == str(case_id)),
        cases[0] if case_id is None else None,
    )
    if selected is None:
        raise ValueError(f"unknown model_config case: {case_id}")
    model_role = selected.get("model_role")
    if not isinstance(model_role, str) or not model_role:
        raise RuntimeError("selected model_config case lacks a model role")
    config = _inside(
        export_directory / str(selected.get("resolved_config", "")),
        export_directory,
        label="resolved inference configuration",
    )
    raw_bundle = selected.get("bundle_path")
    if not isinstance(raw_bundle, str) or not raw_bundle:
        raise RuntimeError("selected model_config case has no learned-weight bundle")
    bundle = _inside(
        export_directory / raw_bundle,
        export_directory,
        label="learned model bundle",
    )
    if not bundle.is_dir() or not (bundle / "manifest.json").is_file():
        raise FileNotFoundError(bundle / "manifest.json")
    return config, bundle, selected


def _assert_supported_raw_contract(config: Mapping[str, Any]) -> None:
    """Reject routes that require fitted state not present in the fold bundle."""
    signal = _mapping(config.get("signal"), label="signal")
    normalization = _mapping(signal.get("normalization"), label="signal.normalization")
    quality = _mapping(config.get("quality"), label="quality")
    window_selection = _mapping(quality.get("window_selection"), label="quality.window_selection")
    artifact = _mapping(config.get("artifact"), label="artifact")
    aggregation = _mapping(config.get("aggregation"), label="aggregation")
    manifest = _mapping(config.get("manifest"), label="manifest")
    checks = (
        (config.get("representation_mode") == "raw", "representation_mode must be raw"),
        (float(signal.get("internal_fs_hz", 0.0)) == 400.0, "signal.internal_fs_hz must be 400"),
        (tuple(signal.get("channel_order", ())) == _CHANNELS, "signal.channel_order must be canonical"),
        (tuple(manifest.get("channel_order", ())) == _CHANNELS, "manifest.channel_order must be canonical"),
        (signal.get("accelerometer_input_unit") == "g", "accelerometer input must be g"),
        (signal.get("gyroscope_input_unit") == "deg/s", "gyroscope input must be deg/s"),
        (normalization.get("raw_imu") == "none", "fold-fitted raw IMU transform is not bundled"),
        (quality.get("mode") == "off", "quality routing fitted state is not bundled"),
        (window_selection.get("policy") == "none", "training-only window selection is unsupported"),
        (artifact.get("motion_detector_enabled") is False, "motion detector bundle is unavailable"),
        (
            artifact.get("denoiser_enabled") is False and artifact.get("reducer") == "identity",
            "artifact reducer state is unavailable",
        ),
        (aggregation.get("quality_weighting") is False, "quality-weighted aggregation is unsupported"),
    )
    unsupported = [message for passed, message in checks if not passed]
    if unsupported:
        raise RuntimeError("unsupported live-inference contract: " + "; ".join(unsupported))


def _validated_role_scope(config: Mapping[str, Any],
                          pipeline_adapter: Any | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Bind concrete input roles to the classifier-family training contract."""
    raw_roles = config.get("roles")
    if not isinstance(raw_roles, list) or not raw_roles:
        raise RuntimeError("resolved config roles must be a non-empty list")
    concrete_roles = tuple(_concrete_role(str(value)) for value in raw_roles)
    if len(concrete_roles) != len(set(concrete_roles)):
        raise RuntimeError("resolved config roles must be unique")
    raw_families = _mapping(config.get("training"), label="training").get("classifier_role_families")
    if not isinstance(raw_families, list) or not raw_families:
        raise RuntimeError("training.classifier_role_families must be a non-empty list")
    classifier_families = tuple(_role_family(str(value)) for value in raw_families)
    if len(classifier_families) != len(set(classifier_families)):
        raise RuntimeError("training.classifier_role_families must be unique")
    missing = sorted(set(classifier_families) - {_role_family(role) for role in concrete_roles})
    if missing:
        raise RuntimeError(f"classifier role families lack concrete config.roles selectors: {missing}")
    if pipeline_adapter is not None:
        raw_adapter_families = getattr(pipeline_adapter, "allowed_role_families", None)
        if raw_adapter_families is None:
            raise RuntimeError("bundle adapter does not declare allowed_role_families")
        adapter_families = tuple(_role_family(str(value)) for value in raw_adapter_families)
        if adapter_families != classifier_families:
            raise RuntimeError("bundle adapter allowed_role_families differ from the resolved config")
    return concrete_roles, classifier_families


def _assert_loaded_bundle_contract(
    loaded: Any,
    config: Mapping[str, Any],
    *,
    config_hash: str,
) -> tuple[ModelInputSpec, tuple[str, ...], tuple[str, ...]]:
    """Bind a verified bundle to the resolved raw model-input boundary."""
    _assert_supported_raw_contract(config)
    if loaded.manifest.get("config_hash") != config_hash:
        raise RuntimeError("learned bundle and resolved inference config hashes differ")
    spec = ModelInputSpec.from_value(loaded.manifest.get("input_spec", {}))
    if spec.mode.value != "raw":
        raise RuntimeError("selected learned bundle is not a raw representation model")
    class_order = tuple(
        int(value) for value in _mapping(config.get("manifest"), label="manifest").get("class_id_order", ()))
    metadata = _mapping(loaded.manifest.get("metadata"), label="bundle metadata")
    if class_order != tuple(range(spec.n_classes)) or tuple(metadata.get("class_order", ())) != class_order:
        raise RuntimeError("learned bundle class schema differs from the resolved config")
    model = _mapping(config.get("model"), label="model")
    declared_channels = tuple(str(value) for value in model.get("input_channel_order", ()))
    if (not declared_channels or any(value not in _MODEL_CHANNELS for value in declared_channels)
            or tuple(value for value in _MODEL_CHANNELS if value in declared_channels) != declared_channels
            or spec.n_channels != len(declared_channels) or tuple(spec.channel_schema) != declared_channels):
        raise RuntimeError("learned bundle raw channel schema differs from the resolved config")
    _, configured_model_id = normalize_model_id(str(model.get("model_id", "")))
    if loaded.manifest.get("machine_model_id") != configured_model_id:
        raise RuntimeError("learned bundle model identity differs from the resolved config")
    transforms = loaded.transforms
    if transforms is not None:
        if not isinstance(transforms, FrozenRepresentationTransformArchive):
            raise RuntimeError("live raw inference accepts only the V2 identity transform archive")
        if (transforms.representation_mode != "raw"
                or transforms.boundary != "already_preprocessed_and_fitted_transforms_applied_model_input"
                or transforms.input_schema_hash != loaded.manifest.get("input_spec_hash")):
            raise RuntimeError("final-refit transform archive differs from the raw model-input boundary")
    concrete_roles, classifier_families = _validated_role_scope(config, loaded.pipeline_adapter)
    return spec, concrete_roles, classifier_families


def _manifest_rows(
    payload: Mapping[str, Any],
    *,
    repository_root: Path,
    input_root: Path | None = None,
    class_names: tuple[str, ...],
    configured_roles: tuple[str, ...],
    classifier_role_families: tuple[str, ...],
) -> tuple[str, tuple[ManifestRow, ...], tuple[ManifestRow, ...], int | None, ]:
    source_contract = _assert_source_contract(payload)
    source_contract_hash = stable_payload_sha256(source_contract)
    participant_id = str(payload.get("participant_id", "")).strip()
    if not participant_id:
        raise ValueError("inference manifest requires participant_id")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("inference manifest requires a non-empty files list")
    allowed_roles = set(configured_roles)
    classifier_families = set(classifier_role_families)
    parsed: list[tuple[Mapping[str, Any], Path, str, str, int, int | None]] = []
    ids: set[str] = set()
    roles: set[str] = set()
    observed_labels: set[int] = set()
    for index, raw in enumerate(raw_files):
        item = _mapping(raw, label=f"inference file {index + 1}")
        file_id = str(item.get("file_id", "")).strip()
        if not file_id or file_id in ids:
            raise ValueError("inference file_id values must be non-empty and unique")
        ids.add(file_id)
        role = _concrete_role(str(item.get("role", "")))
        family = _role_family(role)
        if role not in allowed_roles:
            raise ValueError(f"inference role {role!r} is not an exact member of config.roles")
        if family not in classifier_families and role != "B":
            raise ValueError(f"inference role {role!r} is outside classifier_role_families; "
                             "only B may be calibration-only")
        roles.add(role)
        raw_source = Path(str(item.get("path", "")))
        relative_base = repository_root if input_root is None else input_root
        source = raw_source.resolve() if raw_source.is_absolute() else (relative_base / raw_source).resolve()
        try:
            source.relative_to(repository_root)
        except ValueError as error:
            raise ValueError(f"inference file {file_id} must remain inside {repository_root}") from error
        if not source.exists():
            raise FileNotFoundError(source)
        if source.suffix.lower() != ".csv" or not source.is_file():
            raise ValueError(f"inference source must be a CSV file: {source}")
        label = _label_id(item.get("label"), class_names)
        if label is not None:
            observed_labels.add(label)
        source_hash, sample_count = _csv_identity(source)
        parsed.append((item, source, role, source_hash, sample_count, label))
    classifier_parsed = [row for row in parsed if _role_family(row[2]) in classifier_families]
    if not classifier_parsed:
        raise ValueError("inference manifest contains no classifier-scope recordings")
    classifier_families_present = {_role_family(row[2]) for row in classifier_parsed}
    if classifier_families_present & _DYNAMIC_FAMILIES and "B" not in roles:
        raise ValueError("missing_static_b_calibration: dynamic R/S/W input requires the same "
                         "participant's B recording; missing-B silent calibration is a V5 TODO only")
    if len(observed_labels) > 1:
        raise ValueError("all labelled files for one participant must share one label")
    participant_label = next(iter(observed_labels), None)
    rows: list[ManifestRow] = []
    for item, source, role, source_hash, sample_count, _ in parsed:
        effective_label = -1 if participant_label is None else participant_label
        rows.append(
            ManifestRow(
                record_id=str(item["file_id"]),
                participant_id=participant_id,
                class_id=effective_label,
                class_name=("unlabelled" if participant_label is None else class_names[participant_label]),
                class_name_provenance_alias="live_input_user_declared_source_contract",
                class_source=("not_supplied" if participant_label is None else "user_supplied"),
                label_record_id="",
                role=role,
                source_path=source.relative_to(repository_root).as_posix(),
                source_hash=source_hash,
                source_version=(f"v5_user_declared_{source_contract_hash[:12]}_{source_hash[:16]}"),
                fs=400.0,
                n_samples=sample_count,
                duration_s=sample_count / 400.0,
                channel_schema=_CHANNELS,
                channel_units=dict(_UNITS),
                synchrony_status=f"user_declared_{_FIXED_GRID_SYNCHRONY}",
                reference_available=False,
                qc_status="pass",
                qc_reasons=(),
                manifest_version="v5_live_inference_user_declared_v1",
            ))
    classifier_rows = tuple(row for row in rows if _role_family(row.role) in classifier_families)
    return participant_id, tuple(rows), classifier_rows, participant_label


def _row_dicts(rows: Iterable[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        value = asdict(row)
        value["probabilities"] = list(value.get("probabilities", ()))
        value["class_order"] = list(value.get("class_order", ()))
        output.append(value)
    return output


def _preprocess_live_scope(
    core: Any,
    *,
    classifier_rows: tuple[ManifestRow, ...],
    calibration_rows: tuple[ManifestRow, ...],
    config: Any,
    loader: Any,
) -> list[Any]:
    """Run V2 preprocessing on classifier rows with all B calibration rows."""

    states = [core._RuntimeRecord(row=row) for row in classifier_rows]
    core._preprocess_records(
        states,
        config,
        None,
        loader,
        calibration_rows=calibration_rows,
        cache_session=None,
    )
    return states


def infer_from_manifest(
    *,
    model_config_directory: str | Path | None = None,
    model_export: str | Path | None = None,
    case_id: str | None = None,
    input_manifest: str | Path,
    validated_input: Any | None = None,
    pipeline_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Classify one participant without fitting or modifying model state."""

    del validated_input  # The service independently revalidates the same contract.
    root = Path(pipeline_root or Path(__file__).resolve().parents[3]).resolve()
    repository_root = root.parents[1]
    export_raw = model_config_directory or model_export
    if export_raw is None:
        raise ValueError("model_config_directory is required")
    export = _inside(export_raw, root, label="model_config export")
    if not export.is_dir():
        raise NotADirectoryError(export)
    manifest_path = _repository_input(
        input_manifest,
        pipeline_root=root,
        repository_root=repository_root,
        label="inference manifest",
    )
    config_path, bundle_path, selected_case = _resolve_export(export, case_id)
    config_payload = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        label=str(config_path),
    )
    _assert_supported_raw_contract(config_payload)

    # This verifies the copied resolved config against the same strict schema,
    # data manifest, split registry and module contracts as a training run.
    paths = PipelinePaths.discover()
    report, config, _, _ = preflight_pipeline(config_path, mode="smoke", paths=paths)
    loaded = load_bundle(bundle_path)
    trusted_refit = loaded.manifest.get("bundle_kind") == "trusted_final_refit_v2"
    if trusted_refit != (selected_case["model_role"] == _TRUSTED_REFIT_ROLE):
        raise RuntimeError("selected model role and learned bundle kind differ")
    resolved_config_payload = config.to_dict()
    spec, configured_roles, classifier_families = _assert_loaded_bundle_contract(
        loaded,
        resolved_config_payload,
        config_hash=config.sha256,
    )
    device = str(config.section("training").get("device", "cuda"))
    prepare_deterministic_runtime()
    environment = require_environment(
        device=device,
        require_determinism_env=True,
    ).to_dict()

    payload, input_hash = _load_input_manifest(manifest_path)
    class_names = tuple(str(value) for value in config.section("manifest")["class_name_order"])
    participant_id, calibration_rows, rows, participant_label = _manifest_rows(
        payload,
        repository_root=repository_root,
        input_root=root,
        class_names=class_names,
        configured_roles=configured_roles,
        classifier_role_families=classifier_families,
    )

    # Import the frozen orchestration primitives only after all external
    # contracts have passed.  None of these calls fits the classifier.
    from .. import experiment as core

    loader = lambda row, maximum: _load_record(row, paths, max_samples=maximum)
    states = _preprocess_live_scope(
        core,
        classifier_rows=rows,
        calibration_rows=calibration_rows,
        config=config,
        loader=loader,
    )
    routing = core._apply_quality_motion_routing(
        states,
        config,
        report,
        paths,
        train_ids=(),
        oof_ids=(participant_id, ),
        cache_session=None,
    )
    for state in states:
        core._extract_raw(state, report, config.section("signal"), cache_session=None)
    window_selection = core._apply_window_quality_selection(
        states,
        config,
        train_ids=(),
        oof_ids=(participant_id, ),
    )
    # finalcase has raw_imu=none, so this call only validates/records the
    # identity transform.  A fold-fitted strategy was rejected above.
    representation = core._fit_representation_artifacts(
        states,
        "raw",
        (participant_id, ),
        (),
    )
    dataset = core._materialize_representation_dataset(
        states,
        (participant_id, ),
        "raw",
        quality_weight_source=str(config.section("aggregation").get("quality_weight_source", "none")),
    )
    if dataset is None or len(dataset) == 0:
        failures = [{"file_id": state.row.record_id, "reason": state.reason} for state in states]
        raise RuntimeError(f"no model-ready live windows: {failures}")
    dataset, resampling_key, resampling = core._prepare_dl_input_dataset(
        dataset,
        "raw",
        config.section("signal")["dl_resampling"],
    )
    model_id = str(loaded.manifest.get("machine_model_id", ""))
    dataset, input_binding = core._bind_raw_dataset_for_model(
        dataset,
        model_id,
        declared_channel_order=config.section("model").get("input_channel_order"),
    )
    if dataset.values.shape[1] != spec.n_channels or (spec.channel_schema
                                                      and tuple(dataset.channel_schema) != tuple(spec.channel_schema)):
        raise RuntimeError("live model-ready input differs from the learned bundle schema")
    model_inputs = {"x": dataset.values, "mask": dataset.sample_mask}
    if loaded.transforms is not None:
        transformed = loaded.transforms.transform_inputs(model_inputs)
        if set(transformed) != set(model_inputs) or any(
                not np.array_equal(np.asarray(transformed[name]), np.asarray(value))
                for name, value in model_inputs.items()):
            raise RuntimeError("V2 final-refit transform archive is not identity at inference")

    # Use the same V2 prediction entry points and configured batch size as the
    # outer-CV path.  This avoids a second whole-array inference implementation
    # and preserves its row-alignment and probability validation contracts.
    from ..training.trainer import TrainingConfig, UnifiedTrainer

    prediction = UnifiedTrainer(TrainingConfig.from_mapping(dict(config.section("training"))))
    if loaded.manifest.get("kind") == "estimator":
        probabilities, _, prediction_identities = prediction.predict_estimator_probabilities(loaded.model, dataset)
        classes = tuple(int(value) for value in loaded.model.classes_)
        if set(classes) != {0, 1, 2}:
            raise RuntimeError(f"trained model is missing a class: {classes}")
        if classes != (0, 1, 2):
            probabilities = probabilities[:, [classes.index(value) for value in (0, 1, 2)]]
    elif hasattr(loaded.model, "member_probabilities"):
        _, probabilities, _, prediction_identities = prediction.predict_ensemble_members(loaded.model, dataset)
    else:
        probabilities, _, prediction_identities = prediction.predict_probabilities(loaded.model, dataset)

    metadata = _mapping(loaded.manifest.get("metadata"), label="bundle metadata")
    fitted_objects = metadata.get("fitted_objects")
    fitted = (dict(fitted_objects[0])
              if isinstance(fitted_objects, list) and fitted_objects and isinstance(fitted_objects[0], Mapping) else {})
    source_identity = (metadata.get("golden_case", {}).get("source_identity", {}) if isinstance(
        metadata.get("golden_case"), Mapping) else {})
    preprocessing_hash = stable_payload_sha256({
        "config_hash": config.sha256,
        "input_manifest_sha256": input_hash,
        "source_hashes": [row.source_hash for row in calibration_rows],
        "routing": routing,
        "window_selection": window_selection,
        "representation": representation,
        "resampling": resampling,
        "input_binding": input_binding,
    })
    model_hash = str(fitted.get(
        "state_hash",
        loaded.manifest["file_hashes"][loaded.manifest["state_file"]],
    ))
    common = {
        "repeat": int(source_identity.get("repeat", 0)),
        "fold": int(source_identity.get("fold", 0)),
        "split_seed": 0,
        "training_seed": int(fitted.get("training_seed", 42)),
        "config_hash": config.sha256,
        "manifest_hash": input_hash,
        "fold_hash": str(metadata.get("fold_hash", fitted.get("fold_hash", "live_inference"))),
        "preprocessing_hash": preprocessing_hash,
        "feature_hash": stable_payload_sha256(representation),
        "model_hash": model_hash,
        "representation_mode": "raw",
        # V2 OOF rows require a labelled target to occur in class_order.  Live
        # unlabelled rows use the existing empty-order escape hatch; labelled
        # inputs retain the model's explicit class-order provenance.
        "class_order": (() if participant_label is None else tuple(range(len(class_names)))),
        "code_commit": str(metadata.get("code_version", "not_git_bound")),
        "data_schema_id": "v5_live_inference_user_declared_v1",
        "feature_schema_id": "raw_red_ir_imu_axes_8ch_live_v1",
        "model_version": str(config.section("model").get("variant", "")),
        "aggregation_rule": str(config.section("aggregation")["balance_line"]),
        "environment_hash": stable_payload_sha256(environment["observed"]),
        "manifest_version": "v5_live_inference_user_declared_v1",
        "fold_registry_version": "not_applicable_live_inference",
        "source_snapshot_hash": str(metadata.get("source_snapshot_hash", config.sha256)),
    }
    window_rows, file_rows, role_rows, participant_rows = core._make_oof(
        states,
        (participant_id, ),
        prediction_identities,
        probabilities,
        common,
        balance_line=str(config.section("aggregation")["balance_line"]),
        quality_weighting=False,
        quality_weight_source="none",
    )
    retained_participant = [row for row in participant_rows if row.retained]
    if len(retained_participant) != 1:
        raise RuntimeError("live inference did not produce exactly one participant prediction")
    participant_probability = np.asarray(retained_participant[0].probabilities, dtype=np.float64)
    predicted_id = int(np.argmax(participant_probability))
    return {
        "schema_version": _SCHEMA,
        "status": "complete",
        "training_performed": False,
        "participant_id": participant_id,
        "observed_label": participant_label,
        "observed_label_name": (None if participant_label is None else class_names[participant_label]),
        "predicted_class_id": predicted_id,
        "predicted_class_name": class_names[predicted_id],
        "class_order": list(range(len(class_names))),
        "class_names": list(class_names),
        "probabilities": participant_probability.tolist(),
        "case_id": str(selected_case.get("case_id", "")),
        "model_role": selected_case.get("model_role"),
        "bundle": {
            "path": bundle_path.relative_to(root).as_posix(),
            "bundle_kind": loaded.manifest.get("bundle_kind"),
            "config_hash": loaded.manifest.get("config_hash"),
            "model_id": model_id,
            "model_state_sha256": loaded.manifest["file_hashes"][loaded.manifest["state_file"]],
        },
        "environment_check": environment,
        "stage_preview": {
            "input": {
                "file_count": len(calibration_rows),
                "classifier_file_count": len(rows),
                "calibration_only_file_count": len(calibration_rows) - len(rows),
                "source_hashes": {row.record_id: row.source_hash
                                  for row in calibration_rows},
                "concrete_roles": [row.role for row in calibration_rows],
                "classifier_role_families": list(classifier_families),
                "static_b_present": any(row.role == "B" for row in calibration_rows),
            },
            "preprocessing": {
                "retained_files":
                sum(state.retained for state in states),
                "dropped": [{
                    "file_id": state.row.record_id,
                    "reason": state.reason
                } for state in states if not state.retained],
                "routing":
                routing,
            },
            "windows": {
                "model_ready_window_count": len(dataset),
                "window_selection": window_selection,
                "resampling_key": resampling_key,
                "resampling": resampling,
            },
            "model_input": {
                "shape": list(dataset.values.shape),
                "channel_schema": list(dataset.channel_schema),
                "binding": input_binding,
            },
            "prediction_counts": {
                "window": len(window_rows),
                "file": len(file_rows),
                "role": len(role_rows),
                "participant": len(participant_rows),
            },
        },
        "predictions": {
            "window": _row_dicts(window_rows),
            "file": _row_dicts(file_rows),
            "role": _row_dicts(role_rows),
            "participant": _row_dicts(participant_rows),
        },
        "limitations": {
            "single_participant_statistics": ("ROC/AUC, confusion matrices and significance tests require a labelled "
                                              "multi-participant cohort with adequate class coverage"),
            "median_fold_model": ("A median outer-fold bundle is a research replay model trained on an "
                                  "outer-training subset; only an approved all-29 refit is a deployment refit"),
            "missing_b_ablation":
            "V5 TODO; not implemented",
        },
    }


__all__ = ["infer_from_manifest"]

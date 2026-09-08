"""V2 主流水线、预运行与量化比较 / V2 pipeline, preflight, and comparisons.

中文：正式路径只读已经物化的 manifest/fold CSV，不调用 splitter。``smoke``
会读取一条真实 held-out recording，执行信号 view、唯一窗口计划与 representation
入口；``full`` 会逐文件重算 source hash，并审计完整 5×5 split roster。两者都不把
未训练 logits 冒充科学分类结果。

English: Formal paths read materialized manifest/fold CSVs and never call a
splitter. ``smoke`` traverses one real held-out record through signal views, the sole
window planner, and a representation entry. ``full`` re-hashes all sources and audits
the complete 5-by-5 roster. Neither reports untrained logits as scientific metrics.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import PipelineConfig, load_config
from .contracts import SignalRoute, to_strict_json_value
from .data.folds import (
    FrozenFoldRegistry,
    M2_SEEDS,
    M2_SPLIT_FILE_SHA256,
    M2_SPLIT_PAYLOAD_SHA256,
)
from .data.manifest import (
    M2_FILE_MANIFEST_SHA256,
    audit_manifest,
    load_internal_manifest,
)
from .module_registry import (
    list_modules,
    registry_sha256,
    resolve_artifact_config,
    resolve_peak_detector_config,
    resolve_window_config,
    validate_model_config,
)
from .provenance import runtime_environment, sha256_file
from .train.selection import validate_epoch_selection

@dataclass(frozen=True)
class PipelinePaths:
    """发现并冻结 V2/repository 路径 / Discover and freeze V2/repository paths."""

    pipeline_root: Path
    repository_root: Path

    @classmethod
    def discover(cls) -> "PipelinePaths":
        """从已安装源码定位 / Locate from installed source."""

        root = Path(__file__).resolve().parents[2]
        return cls(root, root.parents[1])

    def input_path(self, relative: str | Path) -> Path:
        """限制配置输入位于 V2 root / Restrict configured inputs to the V2 root."""

        candidate = (self.pipeline_root / Path(relative)).resolve()
        candidate.relative_to(self.pipeline_root)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    def output_path(self, path: str | Path) -> Path:
        """限制所有输出位于 V2 root / Restrict every output to the V2 root."""

        candidate = Path(path)
        candidate = candidate.resolve() if candidate.is_absolute() else (self.pipeline_root / candidate).resolve()
        candidate.relative_to(self.pipeline_root)
        return candidate

    def new_artifact_path(self, path: str | Path) -> Path:
        """Resolve one immutable output below artifacts and reject symlink parents."""

        candidate = self.output_path(path)
        artifacts = (self.pipeline_root / "artifacts").resolve()
        try:
            candidate.relative_to(artifacts)
        except ValueError as exc:
            raise ValueError("new V2 artifacts must remain below the artifacts directory") from exc
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError(f"artifact overwrite forbidden: {candidate}")
        cursor = candidate.parent
        while cursor != artifacts.parent:
            if cursor.is_symlink():
                raise ValueError(f"artifact parent symlink forbidden: {cursor}")
            if cursor == artifacts:
                break
            cursor = cursor.parent
        return candidate

@dataclass(frozen=True)
class PreflightReport:
    """正式运行前机器验收 / Machine-verifiable formal preflight."""

    status: str
    config_id: str
    config_hash: str
    representation_mode: str
    model: dict[str, str]
    artifact: dict[str, Any]
    peak_detector: dict[str, Any]
    manifest_path: str
    manifest_hash: str
    fold_path: str
    fold_hash: str
    manifest_authority_hash: str
    fold_authority_file_hash: str
    fold_authority_payload_hash: str
    manifest_materialization_report_hash: str
    source_unit_schema_hash: str
    record_count: int
    selected_record_count: int
    participant_count: int
    split_count: int
    split_seeds: tuple[int, ...]
    module_registry_hash: str
    window_profiles: dict[str, dict[str, Any]]

@dataclass(frozen=True)
class PipelineRunResult:
    """预处理/输入级执行结果 / Preprocessing and input-level execution result."""

    status: str
    mode: str
    preflight: PreflightReport
    audited_source_count: int
    smoke_record: dict[str, Any] | None
    output_path: str

def _atomic_json(path: Path, payload: Mapping[str, Any], *, root: Path) -> None:
    """在 V2 内原子写 strict JSON / Atomically write strict JSON inside V2."""

    target = path.resolve()
    target.relative_to(root.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                to_strict_json_value(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

def _config_path(path: str | Path, paths: PipelinePaths) -> Path:
    """Resolve a config path.

    Explicit absolute YAMLs may live in an external study archive. Relative
    configs remain rooted in V2, while every manifest/fold/data reference inside
    the config is still resolved through :meth:`PipelinePaths.input_path`.
    """

    candidate = Path(path)
    if candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        candidate = (paths.pipeline_root / candidate).resolve()
        candidate.relative_to(paths.pipeline_root)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("pipeline config must be a YAML file")
    return candidate

def preflight_pipeline(
    config_path: str | Path,
    *,
    mode: str = "smoke",
    paths: PipelinePaths | None = None,
) -> tuple[PreflightReport, PipelineConfig, list[Any], FrozenFoldRegistry]:
    """验证 config、冻结数据和所有跨模块约束 / Run fail-closed preflight."""

    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    resolved_paths = paths or PipelinePaths.discover()
    config = load_config(_config_path(config_path, resolved_paths))
    payload = config.to_dict()
    artifact = resolve_artifact_config(config.section("artifact"))
    peak_detector = resolve_peak_detector_config(config.section("signal"))
    windows = resolve_window_config(config.section("windows"))
    model = validate_model_config(config.section("model"), config.representation_mode)
    validate_epoch_selection(config.section("training"))

    manifest_section = config.section("manifest")
    split_section = config.section("splits")
    if split_section.get("runtime_recompute") is not False:
        raise ValueError("formal pipeline forbids runtime fold recomputation")
    if (
        int(split_section.get("n_splits", 0)) != 5
        or int(split_section.get("n_repeats", 0)) != 5
        or tuple(int(value) for value in split_section.get("split_seeds", ())) != M2_SEEDS
    ):
        raise ValueError("formal pipeline requires the frozen corrected 5x5 registry")
    manifest_path = resolved_paths.input_path(str(manifest_section["path"]))
    fold_path = resolved_paths.input_path(str(split_section["path"]))
    if manifest_section.get("source_manifest_sha256") != M2_FILE_MANIFEST_SHA256:
        raise ValueError("config internal-manifest authority SHA drift")
    if split_section.get("source_registry_file_sha256") != M2_SPLIT_FILE_SHA256:
        raise ValueError("config fold authority file SHA drift")
    if split_section.get("source_registry_payload_sha256") != M2_SPLIT_PAYLOAD_SHA256:
        raise ValueError("config fold authority payload SHA drift")
    materialization_report_path = resolved_paths.input_path("reports/internal_manifest_v2_report.json")
    materialization = json.loads(materialization_report_path.read_text(encoding="utf-8"))
    expected_materialized = materialization.get("generated_artifact", {})
    if (
        materialization.get("schema_version") != "ppg_frailty.internal_manifest_materialization.v2"
        or materialization.get("pipeline_generation") != "final_pipeline_v2"
        or materialization.get("status") != "passed"
        or materialization.get("all_261_source_hashes_verified") is not True
        or expected_materialized.get("path") != manifest_path.relative_to(resolved_paths.pipeline_root).as_posix()
        or expected_materialized.get("sha256") != sha256_file(manifest_path)
        or int(expected_materialized.get("bytes", -1)) != manifest_path.stat().st_size
    ):
        raise ValueError("internal V2 manifest is not bound to a passed byte-rehash report")
    rows = load_internal_manifest(manifest_path)
    summary = audit_manifest(rows)
    if int(manifest_section.get("expected_record_count", -1)) != int(summary["record_count"]):
        raise ValueError("manifest record count differs from frozen config")
    if int(manifest_section.get("expected_participant_count", -1)) != int(summary["participant_count"]):
        raise ValueError("manifest participant count differs from frozen config")
    expected_channels = tuple(str(value) for value in manifest_section["channel_order"])
    if any(tuple(row.channel_schema) != expected_channels for row in rows):
        raise ValueError("manifest channel order differs from frozen config")
    unit_schemas = {json.dumps(row.channel_units, sort_keys=True, separators=(",", ":")) for row in rows}
    if len(unit_schemas) != 1:
        raise ValueError("internal records do not share one frozen source-unit schema")
    source_unit_schema_hash = hashlib.sha256(next(iter(unit_schemas)).encode("utf-8")).hexdigest()
    fold_registry = FrozenFoldRegistry.from_csv(fold_path)
    if any(
        row.source_registry_file_sha256 != M2_SPLIT_FILE_SHA256
        or row.source_registry_payload_sha256 != M2_SPLIT_PAYLOAD_SHA256
        for row in fold_registry.assignments
    ):
        raise ValueError("materialized fold CSV is not bound to the frozen authority")
    participants = {row.participant_id for row in rows}
    if set(fold_registry.participant_ids) != participants:
        raise ValueError("manifest and fold participant rosters differ")
    roles = set(payload["roles"])
    selected = [row for row in rows if row.role in roles and row.qc_status in {"pass", "pass_with_warnings"}]
    if not selected:
        raise ValueError("configuration selected no eligible records")
    # 中文：逐 split 解析能够发现缺 repeat/fold；English: resolving every split
    # detects missing repeat/fold memberships without ever regenerating them.
    splits = [fold_registry.get_split(repeat, fold) for repeat in range(5) for fold in range(5)]
    if mode == "smoke":
        splits = splits[:1]
    report = PreflightReport(
        status="passed",
        config_id=config.config_id,
        config_hash=config.sha256,
        representation_mode=config.representation_mode,
        model=model,
        artifact=artifact,
        peak_detector=peak_detector,
        manifest_path=manifest_path.relative_to(resolved_paths.pipeline_root).as_posix(),
        manifest_hash=sha256_file(manifest_path),
        fold_path=fold_path.relative_to(resolved_paths.pipeline_root).as_posix(),
        fold_hash=sha256_file(fold_path),
        manifest_authority_hash=M2_FILE_MANIFEST_SHA256,
        fold_authority_file_hash=M2_SPLIT_FILE_SHA256,
        fold_authority_payload_hash=M2_SPLIT_PAYLOAD_SHA256,
        manifest_materialization_report_hash=sha256_file(materialization_report_path),
        source_unit_schema_hash=source_unit_schema_hash,
        record_count=int(summary["record_count"]),
        selected_record_count=len(selected),
        participant_count=int(summary["participant_count"]),
        split_count=len(splits),
        split_seeds=tuple(sorted({int(item["split_seed"]) for item in splits})),
        module_registry_hash=registry_sha256(),
        window_profiles=windows,
    )
    return report, config, rows, fold_registry

def _audit_source_identity(
    row: Any,
    paths: PipelinePaths,
) -> tuple[Path, bytes, str]:
    """Read once, then verify source bytes and row/header/unit identity."""

    source = (paths.repository_root / row.source_path).resolve()
    source.relative_to(paths.repository_root)
    if not source.is_file():
        raise FileNotFoundError(source)
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != row.source_hash:
        raise ValueError(f"source hash drift: {row.record_id}")
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"source UTF-8 decode failed: {row.record_id}") from exc
    with io.StringIO(source_text, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        row_count = 0
        for line_number, values in enumerate(reader, start=2):
            if len(values) != 8:
                raise ValueError(f"source column-count drift: {row.record_id}:line={line_number}")
            row_count += 1
    if tuple(header or ()) != tuple(row.channel_schema):
        raise ValueError(f"source channel order drift: {row.record_id}")
    if row_count != int(row.n_samples):
        raise ValueError(f"source sample-count drift: {row.record_id}:{row_count}!={row.n_samples}")
    units = dict(row.channel_units)
    expected_units = {
        "RED": "raw_device_counts_adc_scale_unknown",
        "IR": "raw_device_counts_adc_scale_unknown",
        "AX": "g_source_declared",
        "AY": "g_source_declared",
        "AZ": "g_source_declared",
        "GX": "degree_per_second_source_declared",
        "GY": "degree_per_second_source_declared",
        "GZ": "degree_per_second_source_declared",
    }
    if units != expected_units:
        raise ValueError(f"source unit schema drift: {row.record_id}")
    if not np.isclose(float(row.duration_s), row_count / float(row.fs), atol=1e-9):
        raise ValueError(f"source duration/sample-rate drift: {row.record_id}")
    return source, source_bytes, source_sha256

def _load_record(row: Any, paths: PipelinePaths, *, max_samples: int | None) -> dict[str, Any]:
    """Full-load, audit physical QC, then optionally return a leading slice."""

    source, source_bytes, source_sha256 = _audit_source_identity(row, paths)
    source_text = source_bytes.decode("utf-8")
    full_values = np.loadtxt(
        io.StringIO(source_text, newline=""),
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
    )
    full_values = np.atleast_2d(full_values)
    if full_values.ndim != 2 or full_values.shape[1] != 8:
        raise ValueError(f"source numeric structure failed: {row.record_id}")
    from .data.qc import assess_manifest_record, require_recording_qc_pass

    thresholds = physical_recording_qc_thresholds_v2()
    admission = assess_manifest_record(
        row,
        full_values,
        observed_channel_names=tuple(row.channel_schema),
        observed_fs=float(row.fs),
        thresholds=thresholds,
        timestamps_s=None,
    )
    require_recording_qc_pass(admission)
    if max_samples is None:
        returned_samples = int(full_values.shape[0])
    else:
        returned_samples = int(max_samples)
        if returned_samples <= 0 or returned_samples > int(full_values.shape[0]):
            raise ValueError(f"invalid max_samples for {row.record_id}: {max_samples}")
    values = full_values[:returned_samples]
    return {
        "record_id": row.record_id,
        "fs_hz": float(row.fs),
        "ppg": values[:, :2],
        "acc": values[:, 2:5],
        "gyro": values[:, 5:8],
        "acc_unit": "g",
        "gyro_unit": "deg/s",
        "source_path": source,
        "recording_qc": to_strict_json_value(
            {
                **dict(admission.evidence),
                "source_byte_identity": {
                    "expected_sha256": row.source_hash,
                    "observed_buffer_sha256": source_sha256,
                    "read_operation_count": 1,
                    "header_parsed_from_same_buffer": True,
                    "numeric_values_parsed_from_same_buffer": True,
                },
            }
        ),
        "recording_qc_profile": physical_recording_qc_profile_v2(),
        "full_record_n_samples_before_slice": int(full_values.shape[0]),
        "returned_n_samples_after_slice": returned_samples,
    }

def physical_recording_qc_thresholds_v2() -> Any:
    """Return the named non-device physical admission profile.

    Zero standard-deviation/span floors reject only exact constants. Any
    non-finite run is rejected. CSV files have no source timestamp column, so
    sampling-grid identity remains manifest-bound. Device rails, absolute scale,
    clipping and saturation remain deferred and are not executed.
    """

    from .data.qc import physical_recording_qc_thresholds_v2 as canonical

    return canonical()

def physical_recording_qc_profile_v2() -> dict[str, Any]:
    """Serialize the exact applied thresholds without invented device limits."""

    from .data.qc import physical_recording_qc_profile_v2 as canonical

    return canonical()

def _run_real_smoke(
    report: PreflightReport,
    config: PipelineConfig,
    rows: Sequence[Any],
    registry: FrozenFoldRegistry,
    paths: PipelinePaths,
) -> dict[str, Any]:
    """一条真实 OOF recording 穿过实际 signal/representation / Traverse one record."""

    split = registry.get_split(0, 0)
    roles = set(config.to_dict()["roles"])
    eligible = sorted(
        (row for row in rows if row.participant_id in split["oof_participant_ids"] and row.role in roles),
        key=lambda row: row.record_id,
    )
    if not eligible:
        raise ValueError("smoke fold contains no configured held-out record")
    row = eligible[0]
    # 12 s covers the 10 s engineering and 8 s heartbeat gates without processing
    # the entire five-minute file. / 12 秒同时覆盖工程窗与 heartbeat 最短门槛。
    record = _load_record(row, paths, max_samples=min(row.n_samples, 4_800))
    from .artifact import run_artifact_route
    from .features.engineering import (
        engineering_feature_names,
        extract_engineering_features,
    )
    from .features.registry import default_registry, summarize_engineering
    from .peaks import (
        ABLATION_DETECTOR_ID,
        detect_pulses_per_wavelength,
        select_reference_wavelength,
    )
    from .peaks.aboy_project import DETECTOR_VERSION as CANONICAL_DETECTOR_VERSION
    from .representations.raw import build_raw_windows
    from .signal.optical import extract_dual_optical
    from .signal.ppg_preprocess import build_signal_views
    from .signal.window_plan import WindowPlan

    signal_config = config.section("signal")
    imu_config = signal_config["imu"]
    payload = {
        **record,
        "participant_id": str(row.participant_id),
    }
    if str(imu_config["gravity_method"]) in {
        "calibrated_roll_pitch_ekf",
        "profile_a_lowpass_0p3hz",
        "sensor_filter_only_no_gravity_removal",
    }:
        from .signal import (
            fit_motion_imu_calibration,
            roll_pitch_ekf_config_from_resolved,
        )

        calibration_candidates = sorted(
            (
                item
                for item in rows
                if str(item.participant_id) == str(row.participant_id)
                and str(item.role) == "B"
                and str(item.qc_status) in {"pass", "pass_with_warnings"}
            ),
            key=lambda item: (-float(item.duration_s), str(item.record_id)),
        )
        if not calibration_candidates:
            raise ValueError(
                "canonical smoke requires a same-participant role-B " f"calibration record: {row.participant_id}"
            )
        calibration_row = calibration_candidates[0]
        calibration_record = _load_record(
            calibration_row,
            paths,
            max_samples=None,
        )
        payload["imu_calibration"] = fit_motion_imu_calibration(
            np.asarray(calibration_record["acc"], dtype=np.float64),
            np.asarray(calibration_record["gyro"], dtype=np.float64),
            participant_id=str(row.participant_id),
            file_id=str(calibration_row.record_id),
            source_role="B",
            fs_hz=float(calibration_row.fs),
            acceleration_unit=str(calibration_record["acc_unit"]),
            gyroscope_unit=str(calibration_record["gyro_unit"]),
            config=roll_pitch_ekf_config_from_resolved(imu_config),
        )
    views = build_signal_views(payload, config.to_dict())
    route = run_artifact_route(
        views,
        report.artifact["runtime_reducer"],
        parameters=report.artifact["parameters"],
    )
    if route.result.status != "success" or route.views is None:
        raise RuntimeError(f"configured artifact route failed without fallback: {route.result.reasons}")
    resolved_views = route.views
    profile_name = "raw_dl" if config.representation_mode in {"raw", "fusion"} else "engineering"
    profile = report.window_profiles[profile_name]
    plan = WindowPlan(source_record_id=row.record_id, **profile)
    planned_windows = plan.plan(resolved_views.x_filter.shape[0], 400.0)
    raw_plan = WindowPlan(
        source_record_id=row.record_id,
        **report.window_profiles["raw_dl"],
    )
    engineering_plan = WindowPlan(
        source_record_id=row.record_id,
        **report.window_profiles["engineering"],
    )
    raw_smoke = build_raw_windows(
        resolved_views,
        raw_plan,
        normalization=config.section("signal")["normalization"],
    )
    engineering_smoke = extract_engineering_features(
        resolved_views,
        plan=engineering_plan,
    )
    engineering_values, engineering_validity = summarize_engineering(engineering_smoke)
    pulses_per_wavelength = detect_pulses_per_wavelength(
        resolved_views,
        detector_id=report.peak_detector["detector_id"],
        min_observation_sec=report.peak_detector["min_observation_sec"],
        min_peaks=report.peak_detector["min_peaks"],
        detector_parameters=report.peak_detector.get("parameters"),
    )
    pulse = pulses_per_wavelength[select_reference_wavelength(pulses_per_wavelength)]
    details: dict[str, Any] = {
        "record_id": row.record_id,
        "participant_id": row.participant_id,
        "role": row.role,
        "class_id": row.class_id,
        "outer_repeat": 0,
        "outer_fold": 0,
        "samples_read": int(resolved_views.x_filter.shape[0]),
        "signal_route": route.route.value,
        "artifact_status": route.result.status,
        "imu_calibration": (
            {
                "participant_id": payload["imu_calibration"].participant_id,
                "file_id": payload["imu_calibration"].file_id,
                "source_role": payload["imu_calibration"].source_role,
                "artifact_sha256": payload["imu_calibration"].artifact_sha256,
                "fallback_used": False,
            }
            if "imu_calibration" in payload
            else {"status": "not_required"}
        ),
        "q_morph_semantics": "available" if route.result.is_identity else "not_applicable",
        "window_profile": profile_name,
        "window_count": len(planned_windows),
        "detected_peak_count": int(pulse.peaks.size),
        "median_ppi_s": float(np.median(pulse.ppi_s[pulse.valid_interval_mask])),
        "pulse_detector": {
            "detector_id": report.peak_detector["detector_id"],
            "min_observation_sec": report.peak_detector["min_observation_sec"],
            "min_peaks": report.peak_detector["min_peaks"],
            "parameters": dict(report.peak_detector.get("parameters", {})),
            "reference_wavelength": pulse.wavelength,
            "per_wavelength": {
                wavelength: {
                    "detector_version": str(result.detector_version),
                    "detection_run_id": str(result.detection_run_id),
                    "selected_polarity": int(result.selected_polarity),
                    "block_hri_provenance_hash": str(result.block_hri_provenance_hash),
                    "detector_score": float(result.detector_score),
                    "detector_coverage": float(result.detector_coverage),
                    "detected_peak_count": int(result.peaks.size),
                }
                for wavelength, result in pulses_per_wavelength.items()
            },
        },
        "canonical_parity_smoke": {
            "detector_id": report.peak_detector["detector_id"],
            "min_observation_sec": report.peak_detector["min_observation_sec"],
            "min_peaks": report.peak_detector["min_peaks"],
            "old_detector_invoked": any(
                str(result.detector_id) == ABLATION_DETECTOR_ID
                or str(result.detector_version) != CANONICAL_DETECTOR_VERSION
                for result in pulses_per_wavelength.values()
            ),
            "imu_sensor_filters": dict(resolved_views.metadata["imu_diagnostics"]["sensor_filters"]),
            "imu_outer_train_scaler_identity": str(signal_config["normalization"]["raw_imu"]),
            "engineering_window_column_count": len(engineering_feature_names()),
            "engineering_observed_shape": list(engineering_smoke.sequence.values.shape),
            "engineering_file_summary_field_count": len(engineering_values),
            "engineering_file_summary_validity_field_count": len(engineering_validity),
            "raw_frailty_channel_count": int(raw_smoke.values.shape[1]),
            "raw_frailty_shape": list(raw_smoke.values.shape),
            "dual_optical_lag_limit_s": None,
            "coherence_in_formal_registry": ("optical.red_ir_cardiac_coherence" in default_registry().names),
            "manifest_sha256": report.manifest_hash,
            "fold_registry_sha256": report.fold_hash,
            "manifest_authority_sha256": report.manifest_authority_hash,
            "fold_authority_payload_sha256": (report.fold_authority_payload_hash),
        },
    }
    if resolved_views.route in {SignalRoute.DIRECT, SignalRoute.IDENTITY}:
        optical = extract_dual_optical(
            resolved_views.x_native,
            resolved_views.x_filter,
            pulses_per_wavelength,
            route=resolved_views.route,
        )
        details["dual_optical"] = to_strict_json_value(
            {
                "schema_version": optical.schema_version,
                "pairing": asdict(optical.pairing),
                "beat_audit": [asdict(row) for row in optical.beat_audit],
                "aggregate_values": optical.aggregate_values,
                "aggregate_validity": optical.aggregate_validity,
                "diagnostics": optical.diagnostics,
            }
        )
        waveform_diagnostics = optical.diagnostics["waveform_agreement"]
        details["canonical_parity_smoke"]["dual_optical_lag_limit_s"] = float(waveform_diagnostics["max_lag_seconds"])
    else:
        details["dual_optical"] = {
            "status": "not_applicable",
            "reason": "non_identity_rate_only_route",
        }
    if config.representation_mode in {"raw", "fusion"}:
        raw = raw_smoke
        details.update(
            {
                "representation_shape": list(raw.values.shape),
                "finite": bool(np.isfinite(raw.values).all()),
                "candidate_window_count": raw.candidate_count,
                "dropped_invalid_window_count": raw.dropped_invalid_count,
                "window_coverage": raw.values.shape[0] / raw.candidate_count,
            }
        )
    else:
        extraction = engineering_smoke
        details.update(
            {
                "representation_shape": list(extraction.sequence.values.shape),
                "valid_feature_fraction": float(np.mean(extraction.value_validity)),
            }
        )
    return details

def run_pipeline(
    config_path: str | Path,
    *,
    mode: str,
    output: str | Path,
    paths: PipelinePaths | None = None,
) -> PipelineRunResult:
    """执行 smoke 或 full input audit 并写 run manifest / Execute an auditable run."""

    resolved_paths = paths or PipelinePaths.discover()
    report, config, rows, registry = preflight_pipeline(config_path, mode=mode, paths=resolved_paths)
    target = resolved_paths.output_path(output)
    if target.exists():
        raise FileExistsError(f"run output already exists and overwrite is forbidden: {target}")
    audited = 0
    smoke: dict[str, Any] | None = None
    if mode == "smoke":
        smoke = _run_real_smoke(report, config, rows, registry, resolved_paths)
        audited = 1
        status = "smoke_passed"
    else:
        # 中文：完整模式逐字节重算 261 个来源 hash；不以旧 manifest 的声明代替读取。
        # English: Full mode byte-hashes all 261 sources rather than trusting declarations.
        for row in rows:
            _audit_source_identity(row, resolved_paths)
            audited += 1
        status = "full_input_and_protocol_audit_passed"
    payload = {
        "schema_version": "ppg_frailty.pipeline_run.v2",
        "status": status,
        "mode": mode,
        "preflight": asdict(report),
        "audited_source_count": audited,
        "smoke_record": smoke,
        "runtime_environment": runtime_environment(),
        "scientific_metrics_emitted": False,
        "reason": "input/protocol execution does not report untrained predictions",
    }
    _atomic_json(target, payload, root=resolved_paths.pipeline_root)
    return PipelineRunResult(
        status, mode, report, audited, smoke, target.relative_to(resolved_paths.pipeline_root).as_posix()
    )

def validate_installation(*, config_path: str | Path | None = None) -> dict[str, Any]:
    """验证规范目录、spec lock 与可选 config / Validate installation and config."""

    paths = PipelinePaths.discover()
    required = [
        "quality",
        "artifact",
        "peaks",
        "representations",
        "train",
        "evaluate",
        "bundle",
    ]
    missing = [
        name for name in required if not (paths.pipeline_root / "src/ppg_frailty" / name / "__init__.py").is_file()
    ]
    if missing:
        raise ValueError(f"canonical package boundaries missing: {missing}")
    lock_path = paths.pipeline_root / "docs/spec/SPEC_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source = paths.repository_root / lock["source_path"]
    if sha256_file(source) != lock["source_sha256"]:
        raise ValueError("implementation specification hash drift")
    result: dict[str, Any] = {
        "schema_version": "ppg_frailty.installation_validation.v2",
        "status": "passed",
        "canonical_boundaries": required,
        "spec_sha256": lock["source_sha256"],
        "module_count": len(list_modules()),
        "module_registry_hash": registry_sha256(),
    }
    if config_path is not None:
        report, _, _, _ = preflight_pipeline(config_path, mode="smoke", paths=paths)
        result["preflight"] = asdict(report)
    return result


# Synthetic model/artifact fixtures lived here in V2. They were test harnesses,
# not production workflow modules; V5 keeps executable comparisons in study
# plans and report modules, leaving this file as the data/training facade.


__all__ = [
    "PipelinePaths",
    "PipelineRunResult",
    "PreflightReport",
    "preflight_pipeline",
    "physical_recording_qc_profile_v2",
    "physical_recording_qc_thresholds_v2",
    "run_pipeline",
    "validate_installation",
]

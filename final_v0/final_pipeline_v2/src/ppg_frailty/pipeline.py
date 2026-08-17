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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
            raise ValueError(
                "new V2 artifacts must remain below the artifacts directory"
            ) from exc
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
            json.dumps(to_strict_json_value(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
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
    if candidate.suffix.lower() not in {'.yaml', '.yml'}:
        raise ValueError('pipeline config must be a YAML file')
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
    materialization_report_path = resolved_paths.input_path(
        "reports/internal_manifest_v2_report.json"
    )
    materialization = json.loads(
        materialization_report_path.read_text(encoding="utf-8")
    )
    expected_materialized = materialization.get("generated_artifact", {})
    if (
        materialization.get("schema_version")
        != "ppg_frailty.internal_manifest_materialization.v2"
        or materialization.get("pipeline_generation") != "final_pipeline_v2"
        or materialization.get("status") != "passed"
        or materialization.get("all_261_source_hashes_verified") is not True
        or expected_materialized.get("path")
        != manifest_path.relative_to(resolved_paths.pipeline_root).as_posix()
        or expected_materialized.get("sha256") != sha256_file(manifest_path)
        or int(expected_materialized.get("bytes", -1)) != manifest_path.stat().st_size
    ):
        raise ValueError("internal V2 manifest is not bound to a passed byte-rehash report")
    rows = load_internal_manifest(manifest_path)
    summary = audit_manifest(rows)
    if int(manifest_section.get("expected_record_count", -1)) != int(
        summary["record_count"]
    ):
        raise ValueError("manifest record count differs from frozen config")
    if int(manifest_section.get("expected_participant_count", -1)) != int(
        summary["participant_count"]
    ):
        raise ValueError("manifest participant count differs from frozen config")
    expected_channels = tuple(str(value) for value in manifest_section["channel_order"])
    if any(tuple(row.channel_schema) != expected_channels for row in rows):
        raise ValueError("manifest channel order differs from frozen config")
    unit_schemas = {
        json.dumps(row.channel_units, sort_keys=True, separators=(",", ":"))
        for row in rows
    }
    if len(unit_schemas) != 1:
        raise ValueError("internal records do not share one frozen source-unit schema")
    source_unit_schema_hash = hashlib.sha256(
        next(iter(unit_schemas)).encode("utf-8")
    ).hexdigest()
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
        manifest_materialization_report_hash=sha256_file(
            materialization_report_path
        ),
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
                raise ValueError(
                    f"source column-count drift: {row.record_id}:line={line_number}"
                )
            row_count += 1
    if tuple(header or ()) != tuple(row.channel_schema):
        raise ValueError(f"source channel order drift: {row.record_id}")
    if row_count != int(row.n_samples):
        raise ValueError(
            f"source sample-count drift: {row.record_id}:{row_count}!={row.n_samples}"
        )
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
                "canonical smoke requires a same-participant role-B "
                f"calibration record: {row.participant_id}"
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
    profile_name = (
        "raw_dl"
        if config.representation_mode in {"raw", "fusion"}
        else "engineering"
    )
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
    raw_smoke = build_raw_windows(resolved_views, raw_plan)
    engineering_smoke = extract_engineering_features(
        resolved_views,
        plan=engineering_plan,
    )
    engineering_values, engineering_validity = summarize_engineering(
        engineering_smoke
    )
    pulses_per_wavelength = detect_pulses_per_wavelength(
        resolved_views,
        detector_id=report.peak_detector["detector_id"],
    )
    pulse = pulses_per_wavelength[
        select_reference_wavelength(pulses_per_wavelength)
    ]
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
                "artifact_sha256":
                    payload["imu_calibration"].artifact_sha256,
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
            "reference_wavelength": pulse.wavelength,
            "per_wavelength": {
                wavelength: {
                    "detector_version": str(result.detector_version),
                    "detection_run_id": str(result.detection_run_id),
                    "selected_polarity": int(result.selected_polarity),
                    "block_hri_provenance_hash": str(
                        result.block_hri_provenance_hash
                    ),
                    "detector_score": float(result.detector_score),
                    "detector_coverage": float(result.detector_coverage),
                    "detected_peak_count": int(result.peaks.size),
                }
                for wavelength, result in pulses_per_wavelength.items()
            },
        },
        "canonical_parity_smoke": {
            "detector_id": report.peak_detector["detector_id"],
            "old_detector_invoked": any(
                str(result.detector_id) == ABLATION_DETECTOR_ID
                or str(result.detector_version) != CANONICAL_DETECTOR_VERSION
                for result in pulses_per_wavelength.values()
            ),
            "imu_sensor_filters": dict(
                resolved_views.metadata["imu_diagnostics"]["sensor_filters"]
            ),
            "imu_outer_train_scaler_identity": str(
                signal_config["normalization"]["raw_imu"]
            ),
            "engineering_window_column_count": len(engineering_feature_names()),
            "engineering_observed_shape": list(
                engineering_smoke.sequence.values.shape
            ),
            "engineering_file_summary_field_count": len(engineering_values),
            "engineering_file_summary_validity_field_count": len(
                engineering_validity
            ),
            "raw_frailty_channel_count": int(raw_smoke.values.shape[1]),
            "raw_frailty_shape": list(raw_smoke.values.shape),
            "dual_optical_lag_limit_s": None,
            "coherence_in_formal_registry": (
                "optical.red_ir_cardiac_coherence"
                in default_registry().names
            ),
            "manifest_sha256": report.manifest_hash,
            "fold_registry_sha256": report.fold_hash,
            "manifest_authority_sha256": report.manifest_authority_hash,
            "fold_authority_payload_sha256": (
                report.fold_authority_payload_hash
            ),
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
        details["canonical_parity_smoke"]["dual_optical_lag_limit_s"] = float(
            waveform_diagnostics["max_lag_seconds"]
        )
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
    return PipelineRunResult(status, mode, report, audited, smoke, target.relative_to(resolved_paths.pipeline_root).as_posix())


def validate_installation(*, config_path: str | Path | None = None) -> dict[str, Any]:
    """验证规范目录、spec lock 与可选 config / Validate installation and config."""

    paths = PipelinePaths.discover()
    required = [
        "quality", "artifact", "peaks", "representations", "train", "evaluate", "bundle",
    ]
    missing = [name for name in required if not (paths.pipeline_root / "src/ppg_frailty" / name / "__init__.py").is_file()]
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


def _synthetic_motion_fixture(duration_s: float, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, float]:
    """生成带真实事件的双波长+IMU fixture / Generate a dual-PPG/IMU event fixture."""

    fs_hz = 400.0
    sample_count = int(round(duration_s * fs_hz))
    if sample_count < 3_200:
        raise ValueError("synthetic artifact benchmark requires at least eight seconds")
    rng = np.random.default_rng(seed)
    time_axis = np.arange(sample_count) / fs_hz
    period_s = 0.80
    peak_times = np.arange(0.8, duration_s - 0.4, period_s)
    clean = np.zeros((sample_count, 2), dtype=np.float64)
    for peak in peak_times:
        pulse = np.exp(-0.5 * np.square((time_axis - peak) / 0.035))
        clean[:, 0] += pulse
        clean[:, 1] += 0.85 * np.exp(-0.5 * np.square((time_axis - peak - 0.008) / 0.040))
    motion = np.sin(2.0 * np.pi * 1.7 * time_axis) * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.12 * time_axis))
    dynamic_acc = np.column_stack((motion, 0.6 * np.roll(motion, 3), 0.3 * np.sin(2.0 * np.pi * 3.4 * time_axis)))
    gyro = np.column_stack((0.4 * motion, 0.2 * np.roll(motion, 7), 0.15 * motion))
    observed = clean + np.column_stack((0.70 * motion, 0.50 * np.roll(motion, 5))) + rng.normal(0.0, 0.035, clean.shape)
    imu = {
        "dynamic_acc_mps2": dynamic_acc,
        "gyro_rads": gyro,
        "dynamic_magnitude": np.linalg.norm(dynamic_acc, axis=1),
        "gyro_magnitude": np.linalg.norm(gyro, axis=1),
        "imu_valid_mask": np.ones(sample_count, dtype=bool),
    }
    return observed, imu, peak_times, period_s


def run_artifact_comparison(
    reducers: Sequence[str] = (
        "identity",
        "nlms_imu_anc",
        "ssa_decomposition",
        "spectral_mask",
        "pca_bss",
        "fastica_bss",
        "nmf_bss",
    ),
    *,
    duration_s: float = 10.0,
    seed: int = 42,
) -> dict[str, Any]:
    """同一 synthetic fixture 量化各 reducer / Quantify reducers on one fixture."""

    from .artifact import get_reducer
    from .module_registry import resolve_artifact_module_id
    from .peaks import CANONICAL_DETECTOR_ID, detect_pulses
    from .peaks.pairing import match_events

    observed, imu, reference_times, reference_period = _synthetic_motion_fixture(duration_s, seed)
    def quantify_rate(values: np.ndarray) -> dict[str, Any]:
        """共享事件/HR 量化 / Shared event and HR quantitation."""

        try:
            pulse = detect_pulses(
                values,
                detector_id=CANONICAL_DETECTOR_ID,
            )
            match = match_events(reference_times, pulse.peak_timestamps_s, tolerance_s=0.15)
            valid_ppi = pulse.ppi_s[pulse.valid_interval_mask]
            if valid_ppi.size == 0:
                raise ValueError("rate extraction produced no valid PPI")
            estimated_hr = 60.0 / float(np.median(valid_ppi))
            return {
                "event_precision": match.precision,
                "event_recall": match.recall,
                "event_f1": match.f1,
                "timing_mae_s": match.timing_mae_s,
                "hr_mae_bpm": abs(estimated_hr - 60.0 / reference_period),
                "coverage": float(np.mean(pulse.valid_interval_mask)),
            }
        except ValueError as error:
            return {
                "rate_extraction_status": "failed",
                "rate_extraction_error": str(error),
                "coverage": 0.0,
            }

    # English: These are separate policy controls. Raw bypasses both SQI and reducer;
    # quality-only gates the unchanged waveform and must not be called identity reducer.
    # 中文：两者是独立策略对照。raw 同时绕过 SQI/reducer；quality-only 仅门控原
    # 波形，不能与 identity reducer 混称。
    rows: list[dict[str, Any]] = [
        {
            "canonical_module_id": "raw_no_denoise",
            "control_type": "raw_waveform_without_quality_or_artifact_gate",
            "status": "success",
            "waveform_modified": False,
            "runtime_s": 0.0,
            **quantify_rate(observed),
        }
    ]
    from .quality.endpoint_sqi import SqiConfig, evaluate_quality

    quality_started = time.perf_counter()
    quality = evaluate_quality(
        observed,
        route=SignalRoute.IDENTITY,
        imu_processed=imu,
        fs_hz=400.0,
        config=SqiConfig(),
        detector_id=CANONICAL_DETECTOR_ID,
    )
    quality_pass = quality.q_rate.state.value == "pass"
    quality_row: dict[str, Any] = {
        "canonical_module_id": "quality_only",
        "control_type": "direct_q_rate_gate_without_waveform_modification",
        "status": "success" if quality_pass else "quality_rejected",
        "waveform_modified": False,
        "q_rate_state": quality.q_rate.state.value,
        "q_rate_score": quality.q_rate.score,
        "q_rate_threshold": quality.q_rate.threshold,
        "q_morph_state": quality.q_morph.state.value,
        "signal_coverage": quality.coverage,
        "runtime_s": time.perf_counter() - quality_started,
    }
    quality_row.update(quantify_rate(observed) if quality_pass else {"coverage": 0.0})
    rows.append(quality_row)
    for name in reducers:
        module_identity = resolve_artifact_module_id(name)
        reducer = get_reducer(module_identity["runtime_reducer"])
        started = time.perf_counter()
        result = reducer.reduce(observed, imu, fs_hz=400.0)
        elapsed = time.perf_counter() - started
        row: dict[str, Any] = {
            **module_identity,
            "runtime_result_reducer_id": result.reducer_id,
            "status": result.status,
            "confidence": result.confidence,
            "runtime_s": elapsed,
            "q_morph_state": "available" if result.is_identity else "not_applicable",
            "morphology_features_emitted": False,
            "failure_reasons": list(result.reasons),
        }
        if result.status == "success" and result.x_ar is not None:
            metrics = quantify_rate(result.x_ar)
            row.update(metrics)
            if metrics.get("rate_extraction_status") == "failed":
                row["status"] = "rate_extraction_failed"
                row["failure_reasons"] = [str(metrics["rate_extraction_error"])]
        else:
            row["coverage"] = 0.0
        rows.append(row)
    return {
        "schema_version": "ppg_frailty.synthetic_artifact_comparison.v2",
        "status": "passed",
        "seed": seed,
        "duration_s": duration_s,
        "reference_hr_bpm": 60.0 / reference_period,
        "control_count": 2,
        "reducer_count": len(reducers),
        "results": rows,
        "scientific_scope": "synthetic_contract_test_not_external_ptt_benchmark",
    }


def _synthetic_imu_gravity_fixture(
    duration_s: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成已知 gravity/dynamic 真值 / Generate known gravity and dynamic truth."""

    if duration_s < 8.0:
        raise ValueError("IMU gravity comparison requires at least eight seconds")
    fs_hz = 400.0
    time_axis = np.arange(int(round(duration_s * fs_hz)), dtype=np.float64) / fs_hz
    roll = 0.32 * np.sin(2.0 * np.pi * 0.22 * time_axis)
    roll_rate = 0.32 * 2.0 * np.pi * 0.22 * np.cos(2.0 * np.pi * 0.22 * time_axis)
    gravity = np.column_stack(
        (
            np.zeros_like(time_axis),
            9.80665 * np.sin(roll),
            9.80665 * np.cos(roll),
        )
    )
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.07 * time_axis) ** 2
    dynamic = envelope[:, None] * np.column_stack(
        (
            0.90 * np.sin(2.0 * np.pi * 1.25 * time_axis),
            0.55 * np.sin(2.0 * np.pi * 1.75 * time_axis + 0.4),
            0.35 * np.sin(2.0 * np.pi * 2.10 * time_axis + 0.9),
        )
    )
    gyro = np.column_stack((roll_rate, np.zeros_like(time_axis), np.zeros_like(time_axis)))
    rng = np.random.default_rng(seed)
    observed_acc = gravity + dynamic + rng.normal(0.0, 0.025, gravity.shape)
    observed_gyro = gyro + rng.normal(0.0, 0.002, gyro.shape)
    return observed_acc, observed_gyro, gravity, dynamic


def run_imu_gravity_comparison(
    *, duration_s: float = 12.0, seed: int = 42
) -> dict[str, Any]:
    """量化无预校准 EKF 与 LPF / Quantify no-precalibration EKF versus LPF."""

    from .signal.imu_preprocess import preprocess_imu

    acc, gyro, gravity_truth, dynamic_truth = _synthetic_imu_gravity_fixture(duration_s, seed)
    rows: list[dict[str, Any]] = []
    for method, module_id in (
        ("no_precalibration_ekf", "ekf_no_precalibration"),
        ("lpf_0p3", "lowpass_gravity_0p3hz"),
    ):
        started = time.perf_counter()
        result = preprocess_imu(
            acc,
            gyro,
            fs_hz=400.0,
            acc_unit="m/s2",
            gyro_unit="rad/s",
            gravity_method=method,
        )
        elapsed = time.perf_counter() - started
        gravity = np.asarray(result.processed["gravity_mps2"], dtype=np.float64)
        dynamic = np.asarray(result.processed["dynamic_acc_mps2"], dtype=np.float64)
        valid = (
            np.asarray(result.valid_mask, dtype=bool)
            & np.all(np.isfinite(gravity), axis=1)
            & np.all(np.isfinite(dynamic), axis=1)
        )
        if not np.any(valid):
            raise RuntimeError(f"{method} produced zero valid synthetic samples")
        gravity_error = gravity[valid] - gravity_truth[valid]
        dynamic_error = dynamic[valid] - dynamic_truth[valid]
        rows.append(
            {
                "canonical_module_id": module_id,
                "gravity_method": method,
                "status": result.status,
                "reasons": list(result.reasons),
                "coverage": float(np.mean(valid)),
                "runtime_s": elapsed,
                "gravity_rmse_mps2": float(np.sqrt(np.mean(np.square(gravity_error)))),
                "gravity_mae_mps2": float(np.mean(np.abs(gravity_error))),
                "dynamic_rmse_mps2": float(np.sqrt(np.mean(np.square(dynamic_error)))),
                "dynamic_mae_mps2": float(np.mean(np.abs(dynamic_error))),
                "valid_samples": int(np.count_nonzero(valid)),
            }
        )
    return {
        "schema_version": "ppg_frailty.synthetic_imu_gravity_comparison.v2",
        "status": "passed",
        "duration_s": duration_s,
        "seed": seed,
        "sampling_rate_hz": 400.0,
        "results": rows,
        "scientific_scope": "synthetic_known_truth_contract_test_not_human_motion_benchmark",
    }


def run_model_comparison(
    models: Sequence[str] = (
        "CompactCNN1D",
        "InceptionTimeFull",
        "InceptionTimeSmall",
        "InceptionTimeMatrix",
        "ROCKET",
        "MiniROCKET",
        "LogisticRegressionL2",
        "RBFSVM",
        "ExtraTrees",
        "ShapeFormerChannelSpecificOSD",
        "ShapeFormerEffectSizeFixedV1",
        "FileBagFusionCompact",
        "FileBagFusionInception",
    ),
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """实际构造/拟合 reduced synthetic 模型 / Exercise real models quantitatively."""

    from sklearn.metrics import balanced_accuracy_score

    from .models.factory import (
        ModelInputSpec,
        create_model,
        materialize_architecture_parameters,
        normalize_model_id,
    )

    def explicit(
        values: Mapping[str, Any],
        input_spec: ModelInputSpec,
    ) -> dict[str, Any]:
        payload = dict(values)
        payload["architecture_parameters"] = materialize_architecture_parameters(
            payload,
            input_spec,
        )
        return payload

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    feature_names = tuple(f"feature_{index}" for index in range(8))
    x_vector = rng.normal(size=(36, len(feature_names)))
    y = np.tile(np.arange(3), 12)
    x_vector += y[:, None] * np.linspace(0.15, 0.55, len(feature_names))[None, :]
    participants = tuple(f"synthetic_{index:03d}" for index in range(36))
    for name in models:
        canonical_model_id, machine_model_id = normalize_model_id(name)
        started = time.perf_counter()
        if machine_model_id in {"logistic_regression", "rbf_svm", "extra_trees"}:
            representation_mode = "feature_vector"
        elif machine_model_id in {
            "rocket_numpy",
            "minirocket_ablation",
            "inception_matrix",
            "inception_matrix_five_member_ensemble",
        }:
            representation_mode = "feature_matrix"
        elif machine_model_id in {"fusion_compact", "fusion_inception"}:
            representation_mode = "fusion"
        else:
            representation_mode = "raw"
        # English: Variant metadata makes reduced synthetic execution explicit. In
        # particular, the matrix/ensemble routes use the reviewed small Inception
        # encoder and ROCKET routes use 64 kernels only for this contract test.
        # 中文：variant 元数据明确 synthetic reduced 执行条件；matrix/ensemble
        # 使用已审核 small Inception，ROCKET 的 64 kernels 仅用于合同测试。
        variant_by_model = {
            "CompactCNN1D": "reviewed_compact",
            "InceptionTimeFull": "full",
            "InceptionTimeSmall": "small",
            "InceptionTimeMatrix": "small",
            "InceptionTimeFullFiveMemberEnsemble": "full_comparison_only",
            "InceptionTimeMatrixFiveMemberEnsemble": "full_comparison_only",
            "ROCKET": "numpy_reference_reduced_contract",
            "MiniROCKET": "engineering_ablation_reduced_contract",
            "LogisticRegressionL2": "l2",
            "RBFSVM": "rbf",
            "ExtraTrees": "500_tree",
            "ShapeFormerChannelSpecificOSD": "channel_specific_osd_reference",
            "ShapeFormerEffectSizeFixedV1": "effect_size_fixed_v1_ablation",
            "FileBagFusionCompact": "compact_raw_encoder",
            "FileBagFusionInception": "inception_raw_encoder",
        }
        if machine_model_id in {"logistic_regression", "rbf_svm", "extra_trees"}:
            spec = ModelInputSpec("feature_vector", n_classes=3, n_file_features=8, feature_names=feature_names)
            baseline_options: dict[str, Any] = {
                "logistic_regression": {
                    "class_weight": None,
                    "logistic_c": 1.0,
                    "logistic_max_iter": 5000,
                    "logistic_solver": "lbfgs",
                },
                "rbf_svm": {
                    "class_weight": None,
                    "svm_kernel": "rbf",
                    "svm_probability": True,
                    "svm_c": 1.0,
                    "svm_gamma": "scale",
                },
                "extra_trees": {
                    "class_weight": None,
                    "extra_trees_n_estimators": 500,
                    "extra_trees_n_jobs": 1,
                    "extra_trees_max_features": "sqrt",
                    "extra_trees_min_samples_leaf": 1,
                },
            }[machine_model_id]
            model = create_model(
                explicit(
                    {
                        "model_id": machine_model_id,
                        "seed": seed,
                        **baseline_options,
                    },
                    spec,
                ),
                spec,
            )
            model.fit(x_vector[:27], y[:27], participant_ids=participants[:27])
            probability = model.predict_proba(x_vector[27:])
            metric = float(balanced_accuracy_score(y[27:], probability.argmax(axis=1)))
            kind = "reduced_synthetic_fit"
            parameters = None
        elif machine_model_id in {"rocket_numpy", "minirocket_ablation"}:
            x_matrix = rng.normal(size=(36, 6, 32)).astype(np.float32)
            x_matrix += y[:, None, None] * 0.12
            mask = np.ones((36, 32), dtype=bool)
            spec = ModelInputSpec("feature_matrix", n_channels=6, n_classes=3, channel_schema=tuple(f"channel_{index}" for index in range(6)))
            model = create_model(
                explicit(
                    {
                        "model_id": machine_model_id,
                        "seed": seed,
                        "n_kernels": 64,
                        "alpha": 1.0,
                    },
                    spec,
                ),
                spec,
            )
            model.fit(x_matrix[:27], y[:27], mask=mask[:27], participant_ids=participants[:27])
            probability = model.predict_proba(x_matrix[27:], mask[27:])
            metric = float(balanced_accuracy_score(y[27:], probability.argmax(axis=1)))
            kind = "reduced_synthetic_fit"
            parameters = None
        else:
            import torch

            torch.manual_seed(seed)
            if machine_model_id == "inception_matrix":
                spec = ModelInputSpec("feature_matrix", n_channels=6, n_classes=3)
                model_config: dict[str, Any] = {
                    "model_id": machine_model_id,
                    "seed": seed,
                    "variant": "full",
                    "dropout": 0.2,
                    "kernel_sizes": [39, 19, 9],
                    "dilation": 1,
                }
                inputs = torch.from_numpy(rng.normal(size=(2, 6, 32)).astype(np.float32))
                mask_tensor = torch.ones((2, 32), dtype=torch.bool)
                model = create_model(explicit(model_config, spec), spec)
                with torch.no_grad():
                    logits = model(inputs, mask_tensor)
            elif machine_model_id in {
                "inception_full_five_member_ensemble",
                "inception_matrix_five_member_ensemble",
            }:
                ensemble_mode = (
                    "raw"
                    if machine_model_id == "inception_full_five_member_ensemble"
                    else "feature_matrix"
                )
                ensemble_channels = 8 if ensemble_mode == "raw" else 6
                spec = ModelInputSpec(
                    ensemble_mode,
                    n_channels=ensemble_channels,
                    n_classes=3,
                )
                model = create_model(
                    explicit(
                        {
                            "model_id": machine_model_id,
                            "member_seeds": [
                                50042,
                                60042,
                                70042,
                                80042,
                                90042,
                            ],
                            "comparison_only": True,
                            "dropout": 0.2,
                            "kernel_sizes": [39, 19, 9],
                            "dilation": 1,
                        },
                        spec,
                    ),
                    spec,
                )
                inputs = torch.from_numpy(
                    rng.normal(
                        size=(2, ensemble_channels, 64),
                    ).astype(np.float32)
                )
                with torch.no_grad():
                    logits = model(inputs, torch.ones((2, 64), dtype=torch.bool))
            elif machine_model_id in {
                "shapeformer_channel_specific_osd",
                "shapeformer_effect_size_fixed_v1",
            }:
                channel_schema = (
                    "RED", "IR", "A_dyn_x", "A_dyn_y",
                    "A_dyn_z", "GX", "GY", "GZ",
                )
                spec = ModelInputSpec(
                    "raw",
                    n_channels=8,
                    n_classes=3,
                    channel_schema=channel_schema,
                )
                # English: Both discovery implementations fit the same explicit
                # synthetic outer-train roster. The tiny bank is a callable contract
                # check only; it is never reported as a scientific comparison.
                # 中文：两种 discovery 都只拟合同一份显式 synthetic outer-train
                # 名单；小型 bank 只验证可调用合同，不作为科学比较结果。
                input_fs_hz = 64.0
                outer_repeat_index = 0
                outer_fold_index = 0
                shapelet_y = np.asarray((0, 0, 1, 1, 2, 2), dtype=np.int64)
                discovery_length = (
                    64
                    if machine_model_id == "shapeformer_channel_specific_osd"
                    else 256
                )
                shapelet_x = rng.normal(
                    size=(6, 8, discovery_length)
                ).astype(np.float32)
                shapelet_x += shapelet_y[:, None, None] * 0.10
                shapelet_participants = participants[:6]
                shapelet_files = tuple(
                    f"synthetic_file_{index:03d}" for index in range(6)
                )
                shapelet_windows = tuple(
                    f"synthetic_window_{index:03d}" for index in range(6)
                )
                if machine_model_id == "shapeformer_channel_specific_osd":
                    from .models.pisd_port import (
                        DISCOVERY_BALANCE,
                        DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE,
                        INFORMATION_GAIN_SPLIT_RULE,
                        MAX_DISCOVERY_WINDOWS,
                        NUM_PIP_RATIO,
                        PISD_DISCOVERY_METHOD,
                        PIP_ROUNDING_RULE,
                        PIP_SELECTION_RULE,
                        POSITION_SEARCH_NEIGHBOURHOOD_SAMPLES,
                        SHAPELETS_PER_CLASS,
                        CANDIDATE_GENERATION_RULE,
                        CANDIDATE_ENUMERATION_RULE,
                        CANDIDATE_RANKING_RULE,
                        SELECTED_BANK_ORDER_RULE,
                        discover_pisd_shapelets,
                    )

                    discovery_method = PISD_DISCOVERY_METHOD
                    shapelets = discover_pisd_shapelets(
                        shapelet_x,
                        shapelet_y,
                        shapelet_participants,
                        shapelet_files,
                        shapelet_windows,
                        channel_schema,
                        discovery_method=discovery_method,
                        input_fs_hz=input_fs_hz,
                        outer_repeat_index=outer_repeat_index,
                        outer_fold_index=outer_fold_index,
                        num_pip_ratio=NUM_PIP_RATIO,
                        shapelets_per_class=SHAPELETS_PER_CLASS,
                        max_discovery_windows=MAX_DISCOVERY_WINDOWS,
                        discovery_balance=DISCOVERY_BALANCE,
                        position_search_neighbourhood_samples=(
                            POSITION_SEARCH_NEIGHBOURHOOD_SAMPLES
                        ),
                        distance_position_chunk_size=16,
                        seed=seed,
                    )
                else:
                    from .models.shapeformer import discover_effect_size_shapelets

                    discovery_method = "effect_size_fixed_v1"
                    shapelets = discover_effect_size_shapelets(
                        shapelet_x,
                        shapelet_y,
                        shapelet_participants,
                        discovery_method=discovery_method,
                        input_fs_hz=input_fs_hz,
                        shapelet_length=128,
                        outer_repeat_index=outer_repeat_index,
                        outer_fold_index=outer_fold_index,
                        shapelets_per_class=3,
                        stride=64,
                        max_candidates_per_class=6,
                        seed=seed,
                    )
                shared_shapeformer = {
                    "model_id": machine_model_id,
                    "seed": seed,
                    "shapelets": shapelets,
                    "discovery_method": discovery_method,
                    "input_fs_hz": input_fs_hz,
                    "outer_repeat_index": outer_repeat_index,
                    "outer_fold_index": outer_fold_index,
                    "outer_train_participant_hash": (
                        shapelets.outer_train_participant_hash
                    ),
                }
                if machine_model_id == "shapeformer_channel_specific_osd":
                    shapeformer_config = {
                        **shared_shapeformer,
                        "information_gain_split_rule":
                            INFORMATION_GAIN_SPLIT_RULE,
                        "pip_rounding_rule": PIP_ROUNDING_RULE,
                        "pip_selection_rule": PIP_SELECTION_RULE,
                        "candidate_generation_rule":
                            CANDIDATE_GENERATION_RULE,
                        "candidate_enumeration_rule": CANDIDATE_ENUMERATION_RULE,
                        "candidate_ranking_rule": CANDIDATE_RANKING_RULE,
                        "selected_bank_order_rule": SELECTED_BANK_ORDER_RULE,
                        "discovery_position_search_boundary_rule":
                            DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE,
                        "sequence_length_samples": discovery_length,
                        "local_kernel_width_samples": 8,
                        "local_embedding_channels": 16,
                        "shape_embedding_channels": 16,
                        "attention_feedforward_channels": 32,
                        "attention_heads": 4,
                        "attention_query_chunk_size": 16,
                        "distance_position_chunk_size": 16,
                        "dropout": 0.0,
                        "complexity_norm": 1000.0,
                        "max_complexity_ratio": 3.0,
                        "position_search_neighbourhood_samples":
                            POSITION_SEARCH_NEIGHBOURHOOD_SAMPLES,
                    }
                else:
                    shapeformer_config = {
                        **shared_shapeformer,
                        "hidden_channels": 16,
                        "patch_size_samples": 16,
                        "attention_heads": 4,
                        "attention_layers": 1,
                        "dropout": 0.0,
                        "distance_position_chunk_size": 16,
                    }
                model = create_model(
                    explicit(shapeformer_config, spec),
                    spec,
                )
                inputs = torch.from_numpy(
                    rng.normal(
                        size=(2, 8, discovery_length)
                    ).astype(np.float32)
                )
                with torch.no_grad():
                    logits = model(
                        inputs,
                        torch.ones((2, discovery_length), dtype=torch.bool),
                    )
            elif machine_model_id in {"fusion_compact", "fusion_inception"}:
                spec = ModelInputSpec("fusion", n_channels=8, n_classes=3, n_file_features=8)
                if machine_model_id == "fusion_compact":
                    fusion_options = {
                        "signal_dropout": 0.0,
                        "signal_kernel_sizes": [9, 9, 7],
                        "signal_dilations": [1, 1, 1],
                        "signal_pool_sizes": [4, 4],
                    }
                else:
                    fusion_options = {
                        "signal_variant": "small",
                        "signal_dropout": 0.0,
                        "signal_kernel_sizes": [39, 19, 9],
                        "signal_dilation": 1,
                    }
                model = create_model(
                    explicit(
                        {
                            "model_id": machine_model_id,
                            "seed": seed,
                            **fusion_options,
                            "feature_hidden_dim": 32,
                            "fusion_hidden_dim": 64,
                            "pooling": "mean",
                            "dropout": 0.2,
                        },
                        spec,
                    ),
                    spec,
                )
                bag = torch.from_numpy(rng.normal(size=(2, 3, 8, 256)).astype(np.float32))
                window_mask = torch.tensor([[True, True, False], [True, True, True]])
                file_features = torch.from_numpy(rng.normal(size=(2, 8)).astype(np.float32))
                with torch.no_grad():
                    logits = model(bag, window_mask, file_features)
            else:
                spec = ModelInputSpec("raw", n_channels=8, n_classes=3)
                raw_options: dict[str, Any]
                if machine_model_id == "compact_cnn":
                    raw_options = {
                        "dropout": 0.2,
                        "kernel_sizes": [9, 9, 7],
                        "dilations": [1, 1, 1],
                        "pool_sizes": [4, 4],
                    }
                elif machine_model_id in {"inception_full", "inception_small"}:
                    raw_options = {
                        "dropout": 0.2,
                        "kernel_sizes": [39, 19, 9],
                        "dilation": 1,
                    }
                else:
                    raise ValueError(
                        f"unsupported synthetic raw model: {machine_model_id}"
                    )
                model = create_model(
                    explicit(
                        {
                            "model_id": machine_model_id,
                            "seed": seed,
                            **raw_options,
                        },
                        spec,
                    ),
                    spec,
                )
                inputs = torch.from_numpy(rng.normal(size=(2, 8, 256)).astype(np.float32))
                with torch.no_grad():
                    logits = model(inputs, torch.ones((2, 256), dtype=torch.bool))
            probability = torch.softmax(logits, dim=1).cpu().numpy()
            metric = float(np.max(np.abs(probability.sum(axis=1) - 1.0)))
            kind = "forward_contract_probability_sum_error"
            parameters = int(sum(item.numel() for item in model.parameters() if item.requires_grad))
        rows.append(
            {
                "model_id": canonical_model_id,
                "canonical_model_id": canonical_model_id,
                "machine_model_id": machine_model_id,
                "representation_mode": representation_mode,
                "variant": variant_by_model[canonical_model_id],
                "ensemble_size": (
                    5
                    if machine_model_id in {
                        "inception_full_five_member_ensemble",
                        "inception_matrix_five_member_ensemble",
                    }
                    else 1
                ),
                "comparison_only": machine_model_id in {
                    "inception_full_five_member_ensemble",
                    "inception_matrix_five_member_ensemble",
                },
                "n_kernels": (
                    64
                    if machine_model_id in {
                        "rocket_numpy",
                        "minirocket_ablation",
                    }
                    else None
                ),
                "discovery_method": (
                    "channel_specific_osd"
                    if machine_model_id == "shapeformer_channel_specific_osd"
                    else (
                        "effect_size_fixed_v1"
                        if machine_model_id == "shapeformer_effect_size_fixed_v1"
                        else "not_applicable"
                    )
                ),
                "shapelet_length_samples": (
                    128
                    if machine_model_id == "shapeformer_effect_size_fixed_v1"
                    else None
                ),
                "candidate_stride_samples": (
                    64
                    if machine_model_id == "shapeformer_effect_size_fixed_v1"
                    else None
                ),
                "model_status": (
                    "ablation"
                    if machine_model_id in {
                        "minirocket_ablation",
                        "shapeformer_effect_size_fixed_v1",
                    }
                    else (
                        "comparison"
                        if machine_model_id in {
                            "inception_full_five_member_ensemble",
                            "inception_matrix_five_member_ensemble",
                        }
                        else "reference"
                    )
                ),
                "status": "passed",
                "quantitation_kind": kind,
                "quantitation_value": metric,
                "trainable_parameters": parameters,
                "probability_shape": list(probability.shape),
                "finite_probabilities": bool(np.isfinite(probability).all()),
                "runtime_s": time.perf_counter() - started,
            }
        )
    return {
        "schema_version": "ppg_frailty.synthetic_model_comparison.v2",
        "status": "passed",
        "seed": seed,
        "results": rows,
        "scientific_scope": "reduced_synthetic_contract_test_not_frailty_benchmark",
    }


def run_ablation(factor: str, *, seed: int = 42) -> dict[str, Any]:
    """执行单因素 synthetic ablation / Run a one-factor synthetic ablation."""

    if factor == "artifact":
        return {"factor": factor, "one_factor_only": True, "comparison": run_artifact_comparison(seed=seed)}
    if factor == "model":
        return {"factor": factor, "one_factor_only": True, "comparison": run_model_comparison(seed=seed)}
    if factor == "dl_fs":
        cases = [
            {"dl_fs_hz": fs, "five_second_samples": int(fs * 5), "ten_second_samples": int(fs * 10), "nyquist_hz": fs / 2.0}
            for fs in (100, 160, 200, 400)
        ]
        return {"factor": factor, "one_factor_only": True, "fixed_fields": ["folds", "seed", "roles", "model"], "results": cases}
    if factor == "raw_window_s":
        cases = [
            {"window_s": seconds, "samples_at_400hz": int(seconds * 400), "nominal_cycles_at_75bpm": seconds / 0.8}
            for seconds in (5.0, 10.0)
        ]
        return {"factor": factor, "one_factor_only": True, "fixed_fields": ["folds", "seed", "dl_fs", "model"], "results": cases}
    if factor == "fixed_kernel_samples":
        import torch

        from .models.time_scale import (
            build_fixed_kernel_resampling_cases,
            create_fixed_kernel_resampling_model,
        )

        cases: list[dict[str, Any]] = []
        for item in build_fixed_kernel_resampling_cases():
            started = time.perf_counter()
            model = create_fixed_kernel_resampling_model(
                item.model_name,
                n_channels=8,
                n_classes=3,
                dl_fs_hz=item.dl_fs_hz,
                raw_window_seconds=item.raw_window_seconds,
                dilation=item.dilation,
                seed=seed,
            )
            model.eval()
            inputs = torch.zeros(
                (1, 8, item.sequence_length_samples),
                dtype=torch.float32,
            )
            mask = torch.ones((1, item.sequence_length_samples), dtype=torch.bool)
            with torch.inference_mode():
                probabilities = torch.softmax(model(inputs, mask), dim=-1)
            row = asdict(item)
            row["execution"] = {
                "status": "passed",
                "forward_executed": True,
                "trainable_parameters": int(
                    sum(value.numel() for value in model.parameters() if value.requires_grad)
                ),
                "probability_shape": list(probabilities.shape),
                "probability_sum_error": float(
                    torch.max(torch.abs(probabilities.sum(dim=-1) - 1.0)).item()
                ),
                "finite_probabilities": bool(torch.isfinite(probabilities).all().item()),
                "runtime_s": time.perf_counter() - started,
                "provenance": dict(model.fixed_kernel_resampling_provenance),
            }
            cases.append(row)
        return {
            "schema_version": "ppg_frailty.fixed_kernel_samples_ablation.v2",
            "factor": factor,
            "one_factor_only": True,
            "design": "two_models_x_reference_plus_five_one_factor_conditions",
            "case_count": len(cases),
            "fixed_fields": [
                "folds", "seed", "roles", "model_channels", "kernel_sample_counts"
            ],
            "formal_training_status": "not_run_no_outer_fold_predictions_or_metrics",
            "scientific_scope": "synthetic_forward_contract_not_frailty_benchmark",
            "results": cases,
        }
    raise ValueError(
        "factor must be artifact, model, dl_fs, raw_window_s, or fixed_kernel_samples"
    )


def write_quantitative_report(payload: Mapping[str, Any], output: str | Path) -> Path:
    """将 comparison/ablation 写入 V2 / Write a comparison or ablation report."""

    paths = PipelinePaths.discover()
    target = paths.output_path(output)
    _atomic_json(target, payload, root=paths.pipeline_root)
    return target


__all__ = [
    "PipelinePaths", "PipelineRunResult", "PreflightReport", "preflight_pipeline",
    "physical_recording_qc_profile_v2", "physical_recording_qc_thresholds_v2",
    "run_ablation", "run_artifact_comparison", "run_imu_gravity_comparison",
    "run_model_comparison", "run_pipeline",
    "validate_installation", "write_quantitative_report",
]

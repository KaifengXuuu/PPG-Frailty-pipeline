"""V1 主流水线、预运行与量化比较 / V1 pipeline, preflight, and comparisons.

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
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import PipelineConfig, load_config
from .contracts import SignalRoute, to_strict_json_value
from .data.folds import FrozenFoldRegistry, M2_SEEDS
from .data.manifest import audit_manifest, load_internal_manifest
from .module_registry import (
    list_modules,
    registry_sha256,
    resolve_artifact_config,
    resolve_window_config,
    validate_model_config,
)
from .provenance import runtime_environment, sha256_file
from .train.selection import validate_epoch_selection


@dataclass(frozen=True)
class PipelinePaths:
    """发现并冻结 V1/repository 路径 / Discover and freeze V1/repository paths."""

    pipeline_root: Path
    repository_root: Path

    @classmethod
    def discover(cls) -> "PipelinePaths":
        """从已安装源码定位 / Locate from installed source."""

        root = Path(__file__).resolve().parents[2]
        return cls(root, root.parents[1])

    def input_path(self, relative: str | Path) -> Path:
        """限制配置输入位于 V1 root / Restrict configured inputs to the V1 root."""

        candidate = (self.pipeline_root / Path(relative)).resolve()
        candidate.relative_to(self.pipeline_root)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    def output_path(self, path: str | Path) -> Path:
        """限制所有输出位于 V1 root / Restrict every output to the V1 root."""

        candidate = Path(path)
        candidate = candidate.resolve() if candidate.is_absolute() else (self.pipeline_root / candidate).resolve()
        candidate.relative_to(self.pipeline_root)
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
    manifest_path: str
    manifest_hash: str
    fold_path: str
    fold_hash: str
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
    """在 V1 内原子写 strict JSON / Atomically write strict JSON inside V1."""

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
    """解析并限制 config / Resolve and confine a configuration path."""

    candidate = Path(path)
    candidate = candidate.resolve() if candidate.is_absolute() else (paths.pipeline_root / candidate).resolve()
    candidate.relative_to(paths.pipeline_root)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
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
    rows = load_internal_manifest(manifest_path)
    summary = audit_manifest(rows)
    fold_registry = FrozenFoldRegistry.from_csv(fold_path)
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
        manifest_path=manifest_path.relative_to(resolved_paths.pipeline_root).as_posix(),
        manifest_hash=sha256_file(manifest_path),
        fold_path=fold_path.relative_to(resolved_paths.pipeline_root).as_posix(),
        fold_hash=sha256_file(fold_path),
        record_count=int(summary["record_count"]),
        selected_record_count=len(selected),
        participant_count=int(summary["participant_count"]),
        split_count=len(splits),
        split_seeds=tuple(sorted({int(item["split_seed"]) for item in splits})),
        module_registry_hash=registry_sha256(),
        window_profiles=windows,
    )
    return report, config, rows, fold_registry


def _load_record(row: Any, paths: PipelinePaths, *, max_samples: int | None) -> dict[str, Any]:
    """加载一条 8-channel CSV 并核对 header / Load and validate one raw CSV."""

    source = (paths.repository_root / row.source_path).resolve()
    source.relative_to(paths.repository_root)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    if tuple(header or ()) != tuple(row.channel_schema):
        raise ValueError(f"source channel order drift: {row.record_id}")
    values = np.loadtxt(source, delimiter=",", skiprows=1, max_rows=max_samples, dtype=np.float64)
    values = np.atleast_2d(values)
    if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
        raise ValueError(f"source numeric structure failed: {row.record_id}")
    return {
        "record_id": row.record_id,
        "fs_hz": float(row.fs),
        "ppg": values[:, :2],
        "acc": values[:, 2:5],
        "gyro": values[:, 5:8],
        "acc_unit": "g",
        "gyro_unit": "deg/s",
        "source_path": source,
    }


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
    from .features.engineering import extract_engineering_features
    from .peaks.aboy_project import detect_pulses
    from .representations.raw import build_raw_windows
    from .signal.ppg_preprocess import build_signal_views
    from .signal.window_plan import WindowPlan

    views = build_signal_views(record, config.to_dict())
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
    pulse = detect_pulses(resolved_views)
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
        "q_morph_semantics": "available" if route.result.is_identity else "not_applicable",
        "window_profile": profile_name,
        "window_count": len(planned_windows),
        "detected_peak_count": int(pulse.peaks.size),
        "median_ppi_s": float(np.median(pulse.ppi_s[pulse.valid_interval_mask])),
    }
    if config.representation_mode in {"raw", "fusion"}:
        raw = build_raw_windows(resolved_views, plan)
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
        extraction = extract_engineering_features(resolved_views, plan=plan)
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
            source = (resolved_paths.repository_root / row.source_path).resolve()
            source.relative_to(resolved_paths.repository_root)
            if sha256_file(source) != row.source_hash:
                raise ValueError(f"source hash drift: {row.record_id}")
            audited += 1
        status = "full_input_and_protocol_audit_passed"
    payload = {
        "schema_version": "ppg_frailty.pipeline_run.v1",
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
        "schema_version": "ppg_frailty.installation_validation.v1",
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
    from .peaks.aboy_project import detect_pulses
    from .peaks.pairing import match_events

    observed, imu, reference_times, reference_period = _synthetic_motion_fixture(duration_s, seed)
    def quantify_rate(values: np.ndarray) -> dict[str, Any]:
        """共享事件/HR 量化 / Shared event and HR quantitation."""

        try:
            pulse = detect_pulses(values)
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
        "schema_version": "ppg_frailty.synthetic_artifact_comparison.v1",
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
        "schema_version": "ppg_frailty.synthetic_imu_gravity_comparison.v1",
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
        "InceptionTimeFiveMemberEnsemble",
        "ROCKET",
        "MiniROCKET",
        "LogisticRegressionL2",
        "RBFSVM",
        "ExtraTrees",
        "ShapeFormerEffectSize",
        "FileBagFusionCompact",
        "FileBagFusionInception",
    ),
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """实际构造/拟合 reduced synthetic 模型 / Exercise real models quantitatively."""

    from sklearn.metrics import balanced_accuracy_score

    from .models.factory import ModelInputSpec, create_model, normalize_model_id

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
        if canonical_model_id in {"LogisticRegressionL2", "RBFSVM", "ExtraTrees"}:
            representation_mode = "feature_vector"
        elif canonical_model_id in {"ROCKET", "MiniROCKET", "InceptionTimeMatrix"}:
            representation_mode = "feature_matrix"
        elif canonical_model_id in {"FileBagFusionCompact", "FileBagFusionInception"}:
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
            "InceptionTimeFiveMemberEnsemble": "small",
            "ROCKET": "numpy_reference_reduced_contract",
            "MiniROCKET": "engineering_ablation_reduced_contract",
            "LogisticRegressionL2": "l2",
            "RBFSVM": "rbf",
            "ExtraTrees": "500_tree",
            "ShapeFormerEffectSize": "effect_size_experimental",
            "FileBagFusionCompact": "compact_raw_encoder",
            "FileBagFusionInception": "inception_raw_encoder",
        }
        if canonical_model_id in {"LogisticRegressionL2", "RBFSVM", "ExtraTrees"}:
            spec = ModelInputSpec("feature_vector", n_classes=3, n_file_features=8, feature_names=feature_names)
            model = create_model({"model_id": canonical_model_id, "seed": seed}, spec)
            model.fit(x_vector[:27], y[:27], participant_ids=participants[:27])
            probability = model.predict_proba(x_vector[27:])
            metric = float(balanced_accuracy_score(y[27:], probability.argmax(axis=1)))
            kind = "reduced_synthetic_fit"
            parameters = None
        elif canonical_model_id in {"ROCKET", "MiniROCKET"}:
            x_matrix = rng.normal(size=(36, 6, 32)).astype(np.float32)
            x_matrix += y[:, None, None] * 0.12
            mask = np.ones((36, 32), dtype=bool)
            spec = ModelInputSpec("feature_matrix", n_channels=6, n_classes=3, channel_schema=tuple(f"channel_{index}" for index in range(6)))
            model = create_model({"model_id": canonical_model_id, "seed": seed, "n_kernels": 64, "alpha": 1.0}, spec)
            model.fit(x_matrix[:27], y[:27], mask=mask[:27], participant_ids=participants[:27])
            probability = model.predict_proba(x_matrix[27:], mask[27:])
            metric = float(balanced_accuracy_score(y[27:], probability.argmax(axis=1)))
            kind = "reduced_synthetic_fit"
            parameters = None
        else:
            import torch

            torch.manual_seed(seed)
            if canonical_model_id == "InceptionTimeMatrix":
                spec = ModelInputSpec("feature_matrix", n_channels=6, n_classes=3)
                model_config: dict[str, Any] = {"model_id": canonical_model_id, "seed": seed, "variant": "small"}
                inputs = torch.from_numpy(rng.normal(size=(2, 6, 32)).astype(np.float32))
                mask_tensor = torch.ones((2, 32), dtype=torch.bool)
                model = create_model(model_config, spec)
                with torch.no_grad():
                    logits = model(inputs, mask_tensor)
            elif canonical_model_id == "InceptionTimeFiveMemberEnsemble":
                spec = ModelInputSpec("raw", n_channels=8, n_classes=3)
                model = create_model({"model_id": canonical_model_id, "seed": seed, "variant": "small", "member_seeds": [seed + offset for offset in range(5)]}, spec)
                inputs = torch.from_numpy(rng.normal(size=(2, 8, 256)).astype(np.float32))
                with torch.no_grad():
                    logits = model(inputs, torch.ones((2, 256), dtype=torch.bool))
            elif canonical_model_id == "ShapeFormerEffectSize":
                from .models.shapeformer import discover_effect_size_shapelets

                spec = ModelInputSpec("raw", n_channels=8, n_classes=3)
                # English: Fit the synthetic bank on an explicit outer-train roster;
                # never invent partial provenance or silently emulate PISD. The small
                # candidate budget keeps this a contract test, not a scientific fit.
                # 中文：在显式 synthetic outer-train 名单上真实拟合 shapelet bank；
                # 不伪造残缺 provenance，也不静默模拟 PISD。小候选预算仅用于合同测试。
                discovery_method = "effect_size_shapelets_v1"
                input_fs_hz = 100.0
                outer_repeat_index = 0
                outer_fold_index = 0
                shapelet_y = np.asarray((0, 0, 1, 1, 2, 2), dtype=np.int64)
                shapelet_x = rng.normal(size=(6, 8, 64)).astype(np.float32)
                shapelet_x += shapelet_y[:, None, None] * 0.10
                shapelet_participants = participants[:6]
                shapelets = discover_effect_size_shapelets(
                    shapelet_x,
                    shapelet_y,
                    shapelet_participants,
                    discovery_method=discovery_method,
                    input_fs_hz=input_fs_hz,
                    shapelet_length=16,
                    outer_repeat_index=outer_repeat_index,
                    outer_fold_index=outer_fold_index,
                    per_class=1,
                    stride=16,
                    max_candidates_per_class=4,
                    seed=seed,
                )
                model = create_model(
                    {
                        "model_id": canonical_model_id,
                        "seed": seed,
                        "shapelets": shapelets,
                        "discovery_method": discovery_method,
                        "input_fs_hz": input_fs_hz,
                        "outer_repeat_index": outer_repeat_index,
                        "outer_fold_index": outer_fold_index,
                        "outer_train_participant_hash": shapelets.outer_train_participant_hash,
                        "hidden_channels": 16,
                        "patch_size_samples": 16,
                        "attention_heads": 4,
                        "attention_layers": 1,
                    },
                    spec,
                )
                inputs = torch.from_numpy(rng.normal(size=(2, 8, 256)).astype(np.float32))
                with torch.no_grad():
                    logits = model(inputs, torch.ones((2, 256), dtype=torch.bool))
            elif canonical_model_id in {"FileBagFusionCompact", "FileBagFusionInception"}:
                spec = ModelInputSpec("fusion", n_channels=8, n_classes=3, n_file_features=8)
                model = create_model({"model_id": canonical_model_id, "seed": seed}, spec)
                bag = torch.from_numpy(rng.normal(size=(2, 3, 8, 256)).astype(np.float32))
                window_mask = torch.tensor([[True, True, False], [True, True, True]])
                file_features = torch.from_numpy(rng.normal(size=(2, 8)).astype(np.float32))
                with torch.no_grad():
                    logits = model(bag, window_mask, file_features)
            else:
                spec = ModelInputSpec("raw", n_channels=8, n_classes=3)
                model = create_model({"model_id": canonical_model_id, "seed": seed}, spec)
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
                "ensemble_size": 5 if canonical_model_id == "InceptionTimeFiveMemberEnsemble" else 1,
                "n_kernels": 64 if canonical_model_id in {"ROCKET", "MiniROCKET"} else None,
                "discovery_method": (
                    "effect_size_shapelets_v1"
                    if canonical_model_id == "ShapeFormerEffectSize"
                    else "not_applicable"
                ),
                "model_status": (
                    "experimental"
                    if canonical_model_id == "ShapeFormerEffectSize"
                    else "reviewed_candidate"
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
        "schema_version": "ppg_frailty.synthetic_model_comparison.v1",
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
    if factor == "physical_time":
        import gc
        import tracemalloc

        import torch

        from .models.time_scale import (
            build_physical_time_cases,
            create_time_scaled_model,
            inception_local_receptive_field,
        )

        cases = []
        constructed_count = 0
        forward_count = 0
        not_applicable_count = 0
        for item in build_physical_time_cases():
            row = asdict(item)
            row["inception_receptive_field_samples"] = inception_local_receptive_field(
                item.inception_kernels.sample_counts, depth=6, dilation=item.dilation
            )
            row["inception_receptive_field_s"] = row["inception_receptive_field_samples"] / item.dl_fs_hz

            # English: A feature vector has no temporal convolution kernel. It remains
            # in the matched four-representation grid as an explicit negative control.
            # 中文：特征向量不存在时域卷积核；仍把它保留在四表示配对网格中，作为
            # 显式负对照，而不是偷偷套用不适用的 CNN。
            if item.representation_mode == "feature_vector":
                not_applicable_count += 1
                row["execution"] = {
                    "status": "not_applicable",
                    "reason": "feature_vector_has_no_temporal_convolution_kernel",
                    "forward_executed": False,
                    "coverage": {"status": "not_applicable_without_predictions"},
                    "weakest_class": {"status": "not_applicable_without_predictions"},
                }
                cases.append(row)
                continue

            # English: Matrix uses its canonical Inception route; fusion audits the
            # time-scaled raw branch. Five- and ten-second raw cases are both forwarded.
            # Matrix/fusion ten-second rows reuse the same variable-length architecture
            # after construction, while their five-second counterparts exercise forward.
            # 中文：矩阵使用规范 Inception 路线；fusion 审计其时间缩放 raw 分支。
            # raw 的 5/10 秒都实际前向；矩阵与 fusion 的 10 秒行构造同一可变长架构，
            # 由对应 5 秒行完成前向合约验证，以控制合约测试耗时。
            model_name, n_channels = {
                "raw": ("CompactCNN1D", 8),
                "feature_matrix": ("InceptionTimeMatrix", 6),
                "fusion": ("InceptionTimeSmall", 8),
            }[item.representation_mode]
            forward_executed = item.raw_window_s == 5.0 or item.representation_mode == "raw"
            started = time.perf_counter()
            tracemalloc.start()
            try:
                model = create_time_scaled_model(
                    model_name,
                    n_channels=n_channels,
                    n_classes=3,
                    dl_fs_hz=item.dl_fs_hz,
                    dilation=item.dilation,
                    seed=seed,
                )
                model.eval()
                parameters = int(sum(value.numel() for value in model.parameters() if value.requires_grad))
                parameter_memory = int(sum(value.numel() * value.element_size() for value in model.parameters()))
                probability_shape: list[int] | None = None
                probability_sum_error: float | None = None
                finite_probabilities: bool | None = None
                if forward_executed:
                    samples = int(round(item.dl_fs_hz * item.raw_window_s))
                    inputs = torch.zeros((1, n_channels, samples), dtype=torch.float32)
                    mask = torch.ones((1, samples), dtype=torch.bool)
                    with torch.inference_mode():
                        probabilities = torch.softmax(model(inputs, mask), dim=-1)
                    probability_shape = list(probabilities.shape)
                    probability_sum_error = float(torch.max(torch.abs(probabilities.sum(dim=-1) - 1.0)).item())
                    finite_probabilities = bool(torch.isfinite(probabilities).all().item())
                    forward_count += 1
                _, peak_python_memory = tracemalloc.get_traced_memory()
                constructed_count += 1
                row["execution"] = {
                    "status": "passed",
                    "model_name": model_name,
                    "route_scope": "full_representation" if item.representation_mode != "fusion" else "fusion_raw_branch",
                    "forward_executed": forward_executed,
                    "forward_input_samples": int(round(item.dl_fs_hz * item.raw_window_s)) if forward_executed else None,
                    "trainable_parameters": parameters,
                    "parameter_memory_bytes": parameter_memory,
                    "peak_python_memory_bytes": int(peak_python_memory),
                    "runtime_s": time.perf_counter() - started,
                    "probability_shape": probability_shape,
                    "probability_sum_error": probability_sum_error,
                    "finite_probabilities": finite_probabilities,
                    "physical_time_provenance": dict(model.physical_time_provenance),
                    "coverage": {"status": "not_applicable_without_predictions"},
                    "weakest_class": {"status": "not_applicable_without_predictions"},
                }
            finally:
                tracemalloc.stop()
                if "model" in locals():
                    del model
                gc.collect()
            cases.append(row)
        return {
            "factor": factor,
            "one_factor_only": False,
            "design": "4_dl_fs_x_2_context_x_2_dilation_x_4_representation",
            "case_count": len(cases),
            "constructed_case_count": constructed_count,
            "forward_case_count": forward_count,
            "not_applicable_case_count": not_applicable_count,
            "frozen_fold_seed_requirement": "identical_5x5_registry_and_training_seeds",
            "required_formal_result_fields": [
                "participant_balanced_accuracy", "participant_macro_f1", "worst_class_recall",
                "worst_class_f1", "brier_score", "ece", "coverage", "runtime_s",
                "peak_memory_bytes", "train_heldout_gap",
            ],
            "execution_gate": "passed_synthetic_construction_and_representative_forward_contract",
            "formal_training_status": "not_run_no_outer_fold_logits_or_scientific_metrics_emitted",
            "scientific_scope": "physical_time_model_contract_audit_not_frozen_5x5_benchmark",
            "results": cases,
        }
    raise ValueError("factor must be artifact, model, dl_fs, raw_window_s, or physical_time")


def write_quantitative_report(payload: Mapping[str, Any], output: str | Path) -> Path:
    """将 comparison/ablation 写入 V1 / Write a comparison or ablation report."""

    paths = PipelinePaths.discover()
    target = paths.output_path(output)
    _atomic_json(target, payload, root=paths.pipeline_root)
    return target


__all__ = [
    "PipelinePaths", "PipelineRunResult", "PreflightReport", "preflight_pipeline",
    "run_ablation", "run_artifact_comparison", "run_imu_gravity_comparison",
    "run_model_comparison", "run_pipeline",
    "validate_installation", "write_quantitative_report",
]

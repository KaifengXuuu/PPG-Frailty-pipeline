#!/usr/bin/env python3
"""构建 M3 核心机器证据 / Build core M3 machine evidence.

中文：只读根目录源码、M2 manifest 与确定性 fixture；所有输出仅写 M3 evidence。
English: Read root sources, M2 manifests, and fixed fixtures; write only M3 evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
M2_MANIFEST = (
    REPOSITORY_ROOT
    / "final_v0/M2_data_manifest_and_evaluation_protocol/manifests/frailty3_file_manifest.csv"
)
EVIDENCE_ROOT = PACKAGE_ROOT / "evidence"
FIXTURE = PACKAGE_ROOT / "fixtures/imu_reference_v1.npy"
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
from m3_signal_core import preprocess_imu  # noqa: E402


def sha256_file(path: Path) -> str:
    """逐字节 hash / Hash every byte."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """原子 strict JSON / Atomic strict JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def historical_crosswalk() -> dict[str, Any]:
    """登记只读历史实现 / Register read-only historical implementations."""

    # 中文：规则与手工审计所用关键词一致，覆盖根目录、Arc 与 archive 版本。
    # English: The deterministic rule covers root, Arc, and archive implementations.
    discovery_expression = (
        r"0\.2.?8|0\.5.?5|0\.5.?8|median/IQR|sosfilt|bandpass_filter|"
        r"StandardScaler|RobustScaler|resample_poly|gravity|aboypp"
    )
    discovery = re.compile(discovery_expression, flags=re.IGNORECASE)
    excluded_roots = {"final_v0", ".git", "_agent", "AA_TODO"}
    sources = []
    for source in REPOSITORY_ROOT.rglob("*.py"):
        relative = source.relative_to(REPOSITORY_ROOT)
        if relative.parts and relative.parts[0] in excluded_roots:
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        if discovery.search(text):
            sources.append((relative.as_posix(), source, text))
    sources.sort(key=lambda item: item[0].encode("utf-8"))

    detailed = {
        "frailty_3class_classifier.py": (
            ["dual-polarity peaks", "35-210 bpm PPI", "record-wide scaling"],
            ["peak contract differs", "record-wide MAD may suppress recovery transitions"],
        ),
        "funcs.py": (
            ["legacy Aboy-like peaks", "Euler roll/pitch EKF", "gravity LPF"],
            ["artifact-rejection positional defect", "Euler singularity", "computed initializer unused"],
        ),
        "ppg.py": (
            ["legacy Aboy-like peaks", "PPI/HRV helpers", "Euler roll/pitch EKF"],
            ["same artifact defect", "numeric-value PPI deduplication", "SDNN inconsistency"],
        ),
        "ppg_peak_hr_gating_train.py": (
            ["PPG/IMU motion gate", "ECG-supervised peak/HR gate", "resampling"],
            ["legacy 256 Hz grid", "dataset-specific unit heuristics"],
        ),
    }

    def generic_algorithms(text: str) -> list[str]:
        """从命中词标记算法族 / Tag algorithm families from matched source text."""

        lowered = text.lower()
        tags = []
        if "bandpass" in lowered or "sosfilt" in lowered:
            tags.append("PPG filtering")
        if "aboy" in lowered or "peak" in lowered:
            tags.append("peak/PPI")
        if "gravity" in lowered or "ekf" in lowered:
            tags.append("IMU gravity/motion")
        if "scaler" in lowered or "median/iqr" in lowered:
            tags.append("scaling")
        if "resample_poly" in lowered:
            tags.append("resampling")
        return tags or ["preprocessing-related implementation"]

    entries = []
    for relative_path, source, text in sources:
        algorithms, risks = detailed.get(
            relative_path,
            (
                generic_algorithms(text),
                [
                    "parameters or semantics differ from the M3 corrected registry",
                    "must not become a second future-active implementation",
                ],
            ),
        )
        path_parts_lower = {part.lower() for part in Path(relative_path).parts}
        is_archive = bool(
            {"arc", "archiv", "archive"} & path_parts_lower
            or any("archive" in part for part in path_parts_lower)
        )
        entries.append(
            {
                "path": relative_path,
                "exists": True,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "algorithms": algorithms,
                "known_risks": risks,
                "status": (
                    "historical_archive_reproduction_only"
                    if is_archive
                    else "historical_reproduction_only"
                ),
                "future_active_replacement": "m3_signal_core registry-bound facade",
            }
        )
    return {
        "evidence_id": "m3_historical_preprocessing_crosswalk_snapshot_v1",
        "crosswalk_target_registry_id": "m3_historical_preprocessing_crosswalk_v1",
        "authority_role": "audit_snapshot_non_authoritative",
        "status": "root_read_only_historical_reproduction_only",
        "discovery_rule": {
            "file_glob": "**/*.py",
            "regex": discovery_expression,
            "excluded_top_level": sorted(excluded_roots),
            "stable_order": "relative_path_utf8_bytewise",
        },
        "future_active_boundary": "final_v0/M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core",
        "entries": entries,
        "discovered_source_count": len(entries),
        "missing_source_count": sum(not entry["exists"] for entry in entries),
    }


def filter_response() -> dict[str, Any]:
    """计算冻结 PPG 响应 / Compute frozen PPG responses."""

    frequencies = np.array([0.10, 0.20, 0.40, 35.0 / 60.0, 1.0, 4.0, 8.0, 12.0])
    profiles = {}
    for name, low_hz in (("static", 0.2), ("motion_peak_denoiser", 0.4)):
        sos = signal.butter(
            3, [low_hz, 8.0], btype="bandpass", fs=400.0, output="sos"
        )
        _, response = signal.sosfreqz(sos, worN=frequencies, fs=400.0)
        profiles[name] = {
            "bandpass_hz": [low_hz, 8.0],
            "order": 3,
            "sos_coefficients": sos.tolist(),
            "causal_single_pass_amplitude": np.abs(response).tolist(),
            "offline_zero_phase_effective_amplitude": (np.abs(response) ** 2).tolist(),
        }
    return {
        "evidence_id": "m3_ppg_filter_response_v1",
        "sampling_rate_hz": 400,
        "frequencies_hz": frequencies.tolist(),
        "notch": "disabled",
        "profiles": profiles,
        "interpretation": "sosfiltfilt squares magnitude; causal and offline outputs are not numeric parity.",
    }


def synthetic_comparison() -> dict[str, Any]:
    """在已知真值上比较 EKF/LPF / Compare EKF and LPF against known truth."""

    fixture = np.load(FIXTURE, allow_pickle=False)
    acceleration, gyroscope = fixture[:, :3], fixture[:, 3:6]
    truth_gravity, truth_dynamic = fixture[:, 6:9], fixture[:, 9:12]
    routes = {}
    for key, profile_id in (
        ("ekf", "imu_ekf_si_400_causal_v1"),
        ("lpf_0p3", "imu_lpf_si_400_causal_v1"),
    ):
        result = preprocess_imu(
            acceleration,
            gyroscope,
            400.0,
            acceleration_unit="m/s^2",
            gyroscope_unit="rad/s",
            profile_id=profile_id,
        )
        mask = result.sample_valid_mask
        gravity_error = np.linalg.norm(
            result.gravity_mps2[mask] - truth_gravity[mask], axis=1
        )
        dynamic_error = np.linalg.norm(
            result.dynamic_acc_mps2[mask] - truth_dynamic[mask], axis=1
        )
        cosine = np.sum(
            result.gravity_mps2[mask] * truth_gravity[mask], axis=1
        ) / (
            np.linalg.norm(result.gravity_mps2[mask], axis=1)
            * np.linalg.norm(truth_gravity[mask], axis=1)
        )
        angle = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
        routes[key] = {
            "profile_id": profile_id,
            "status": result.status.value,
            "valid_sample_count": int(mask.sum()),
            "coverage_fraction": float(mask.mean()),
            "gravity_vector_rmse_mps2": float(np.sqrt(np.mean(gravity_error**2))),
            "gravity_angle_p95_deg": float(np.percentile(angle, 95.0)),
            "dynamic_acceleration_vector_rmse_mps2": float(
                np.sqrt(np.mean(dynamic_error**2))
            ),
            "terminal_state": result.diagnostics["terminal_state"],
            "silent_fallback": result.diagnostics["silent_fallback"],
        }
    passed = (
        routes["ekf"]["gravity_angle_p95_deg"] <= 2.0
        and routes["ekf"]["dynamic_acceleration_vector_rmse_mps2"] <= 0.35
    )
    return {
        "evidence_id": "m3_ekf_lpf_synthetic_truth_v1",
        "fixture": str(FIXTURE.relative_to(REPOSITORY_ROOT)),
        "fixture_sha256": sha256_file(FIXTURE),
        "fixture_columns": [
            "AX", "AY", "AZ", "GX", "GY", "GZ",
            "gravity_truth_x", "gravity_truth_y", "gravity_truth_z",
            "dynamic_truth_x", "dynamic_truth_y", "dynamic_truth_z",
        ],
        "route_metrics": routes,
        "engineering_gate_status": "pass" if passed else "fail",
        "limitations": [
            "Synthetic truth is an engineering fixture, not clinical validation.",
            "EKF coverage excludes explicit online initialization.",
        ],
    }


def m2_integrity_binding() -> dict[str, Any]:
    """绑定 M2 全量扫描，不伪造重复权威 / Bind the authoritative M2 scan."""

    with M2_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    role_counts = Counter(row["role"] for row in rows)
    activity_counts = Counter(row["activity_state"] for row in rows)
    return {
        "evidence_id": "m3_frailty3_signal_integrity_binding_v1",
        "dataset_version_id": rows[0]["dataset_version_id"],
        "m2_manifest": str(M2_MANIFEST.relative_to(REPOSITORY_ROOT)),
        "m2_manifest_sha256": sha256_file(M2_MANIFEST),
        "file_count": len(rows),
        "subject_count": len({row["subject_id"] for row in rows}),
        "numeric_row_count": sum(int(row["n_samples"]) for row in rows),
        "all_files_passed_finite_8_columns": all(
            row["numeric_full_scan"] == "passed_finite_8_columns" for row in rows
        ),
        "role_counts": dict(sorted(role_counts.items())),
        "activity_counts": dict(sorted(activity_counts.items())),
        "authority_boundary": "M2 performed full-byte/full-numeric scans; M3 binds that evidence.",
    }


def main() -> None:
    """写出确定性证据 / Write deterministic evidence."""

    outputs = {
        "historical_preprocessing_crosswalk_v1.json": historical_crosswalk(),
        "filter_response_comparison.json": filter_response(),
        "ekf_lpf_synthetic_comparison.json": synthetic_comparison(),
        "frailty3_signal_integrity_summary.json": m2_integrity_binding(),
    }
    for name, payload in outputs.items():
        write_json(EVIDENCE_ROOT / name, payload)
    # 中文：报告不含时间戳，保持相同输入下字节稳定。
    # English: Omit timestamps so identical inputs produce stable bytes.
    core_paths = {f"evidence/{name}" for name in outputs}
    # 中文：重建核心证据时保留已独立验证的全数据 proxy/parity 条目。
    # English: Preserve separately validated full-data proxy/parity entries on rebuild.
    preserved: list[dict[str, Any]] = []
    existing_report = PACKAGE_ROOT / "M3_BUILD_REPORT.json"
    if existing_report.is_file():
        previous = json.loads(existing_report.read_text(encoding="utf-8"))
        for item in previous.get("outputs", []):
            candidate = PACKAGE_ROOT / item.get("path", "")
            if item.get("path") not in core_paths and candidate.is_file():
                preserved.append(
                    {
                        "path": item["path"],
                        "sha256": sha256_file(candidate),
                        "bytes": candidate.stat().st_size,
                    }
                )
    report = {
        "schema_version": "m3.build_report.v1",
        "report_id": (
            "m3_complete_evidence_build_v1"
            if preserved
            else "m3_core_evidence_build_v1"
        ),
        "status": "pass",
        "producer_sha256": sha256_file(Path(__file__)),
        "outputs": sorted([
            {
                "path": f"evidence/{name}",
                "sha256": sha256_file(EVIDENCE_ROOT / name),
                "bytes": (EVIDENCE_ROOT / name).stat().st_size,
            }
            for name in sorted(outputs)
        ] + preserved, key=lambda item: item["path"]),
    }
    write_json(PACKAGE_ROOT / "M3_BUILD_REPORT.json", report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""构建 Frailty3 EKF/LPF 配对任务代理 / Build paired Frailty3 IMU proxies.

中文：读取 M2 冻结的 261 条记录，每条只取未填充的起始六秒。Frailty3 没有
姿态/重力真值，因此输出只能评价 coverage、残差强度与角色差异，不能证明物理
重力估计精度。所有结果只写 M3 evidence。

English: Read the first unpadded six seconds of all 261 M2-frozen records. Frailty3
has no orientation/gravity truth, so outputs are task proxies rather than physical
accuracy claims. Results are written only to M3 evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
MANIFEST = (
    REPOSITORY_ROOT
    / "final_v0/M2_data_manifest_and_evaluation_protocol/manifests/frailty3_file_manifest.csv"
)
OUTPUT = PACKAGE_ROOT / "evidence/ekf_lpf_frailty3_role_proxy.json"
BUILD_REPORT = PACKAGE_ROOT / "M3_BUILD_REPORT.json"
FS_HZ = 400.0
SAMPLES = int(6.0 * FS_HZ)
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


def read_prefix(path: Path) -> np.ndarray:
    """读取头与固定前缀 / Read a header and fixed prefix."""

    values = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        max_rows=SAMPLES,
        dtype=np.float64,
    )
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"expected_8_columns:{path}")
    return values


def route_metrics(result: Any) -> dict[str, Any]:
    """提取无真值代理 / Extract no-truth proxies."""

    mask = np.asarray(result.sample_valid_mask, dtype=bool)
    if (
        result.dynamic_acc_mps2 is None
        or result.gravity_mps2 is None
        or not np.any(mask)
    ):
        return {
            "status": result.status.value,
            "terminal_state": result.diagnostics.get("terminal_state"),
            "coverage_fraction": 0.0,
            "dynamic_acceleration_rms_mps2": None,
            "gravity_norm_median_mps2": None,
            "gravity_norm_abs_error_median_mps2": None,
        }
    dynamic_squared = np.sum(result.dynamic_acc_mps2[mask] ** 2, axis=1)
    gravity_norm = np.linalg.norm(result.gravity_mps2[mask], axis=1)
    return {
        "status": result.status.value,
        "terminal_state": result.diagnostics.get("terminal_state"),
        "coverage_fraction": float(mask.mean()),
        "dynamic_acceleration_rms_mps2": float(np.sqrt(np.mean(dynamic_squared))),
        "gravity_norm_median_mps2": float(np.median(gravity_norm)),
        "gravity_norm_abs_error_median_mps2": float(
            np.median(np.abs(gravity_norm - 9.80665))
        ),
    }


def median_or_none(values: list[float | None]) -> float | None:
    """有限中位数 / Finite median."""

    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def main() -> None:
    """逐记录执行并汇总 / Execute every record and aggregate."""

    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = sorted(
            csv.DictReader(handle),
            key=lambda row: row["file_id"].encode("utf-8"),
        )
    if len(rows) != 261:
        raise ValueError(f"unexpected_manifest_count:{len(rows)}")
    records = []
    aggregates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        values = read_prefix(REPOSITORY_ROOT / row["relative_path"])
        routes = {}
        for route, profile_id in (
            ("ekf", "imu_ekf_si_400_causal_v1"),
            ("lpf_0p3", "imu_lpf_si_400_causal_v1"),
        ):
            result = preprocess_imu(
                values[:, 2:5],
                values[:, 5:8],
                FS_HZ,
                acceleration_unit="g",
                gyroscope_unit="deg/s",
                profile_id=profile_id,
            )
            metrics = route_metrics(result)
            routes[route] = metrics
            aggregates[(row["role_family"], route)].append(metrics)
        records.append(
            {
                "file_id": row["file_id"],
                "subject_id": row["subject_id"],
                "role": row["role"],
                "role_family": row["role_family"],
                "activity_state": row["activity_state"],
                "source_sha256_from_m2": row["sha256"],
                "samples_read": int(values.shape[0]),
                "routes": routes,
            }
        )
    summary = {}
    for (role_family, route), items in sorted(aggregates.items()):
        summary[f"{role_family}:{route}"] = {
            "record_count": len(items),
            "record_with_any_valid_count": sum(
                item["coverage_fraction"] > 0.0 for item in items
            ),
            "terminal_no_estimate_count": sum(
                item["terminal_state"] == "no_estimate" for item in items
            ),
            "coverage_fraction_median": median_or_none(
                [item["coverage_fraction"] for item in items]
            ),
            "dynamic_acceleration_rms_mps2_median": median_or_none(
                [item["dynamic_acceleration_rms_mps2"] for item in items]
            ),
            "gravity_norm_abs_error_median_mps2": median_or_none(
                [item["gravity_norm_abs_error_median_mps2"] for item in items]
            ),
        }
    payload = {
        "evidence_id": "m3_ekf_lpf_frailty3_first6s_proxy_v1",
        "dataset_version_id": rows[0]["dataset_version_id"],
        "m2_manifest_sha256": sha256_file(MANIFEST),
        "record_count": len(records),
        "subject_count": len({row["subject_id"] for row in rows}),
        "segment_definition": "first_6_seconds_no_padding_each_record",
        "paired_upstream": "same validation, explicit g/deg-s conversion, causal 20/40 Hz sensor filters",
        "summary_by_role_family_and_route": summary,
        "records": records,
        "limitations": [
            "Frailty3 has no gravity truth; proxies are not physical-accuracy metrics.",
            "The first segment may begin during motion, challenging no-precal initialization.",
            "A fixed-magnitude EKF retains radial scale/DC bias that LPF may absorb.",
            "Formal motion BA and corrected-SGKF downstream comparison belong to M5/M8.",
        ],
    }
    write_json(OUTPUT, payload)
    report = json.loads(BUILD_REPORT.read_text(encoding="utf-8"))
    entry = {
        "path": "evidence/ekf_lpf_frailty3_role_proxy.json",
        "sha256": sha256_file(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
    }
    report["outputs"] = [
        item for item in report["outputs"] if item["path"] != entry["path"]
    ] + [entry]
    report["outputs"] = sorted(report["outputs"], key=lambda item: item["path"])
    report["report_id"] = "m3_core_and_frailty_proxy_build_v1"
    write_json(BUILD_REPORT, report)


if __name__ == "__main__":
    # 中文：工具固定读取 M2 roster；无随机抽样。
    # English: The tool always reads the frozen M2 roster without random sampling.
    main()

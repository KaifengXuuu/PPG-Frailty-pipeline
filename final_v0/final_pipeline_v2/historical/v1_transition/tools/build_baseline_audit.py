#!/usr/bin/env python3
"""冻结 dev0 基线库存 / Freeze the dev0 baseline inventory.

中文：只读取工作区历史文件并只向 V1 artifacts 写结果。历史性能被明确标为
characterization，不会被当作新协议 benchmark。

English: The tool reads historical workspace files and writes only into the V1
artifact root. Historical metrics are explicitly characterization-only.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "audit"


def _sha(path: Path) -> str:
    """流式计算文件 hash / Hash one file as a byte stream."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    """读取包版本；未安装时返回 null / Return an installed package version or null."""

    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:  # 包缺失是需要记录的环境事实 / Missing packages are audit facts.
        return None


def _git_text(*arguments: str) -> str:
    """读取 git 元数据而不改变仓库 / Read Git metadata without mutation."""

    result = subprocess.run(
        ["git", *arguments], cwd=REPO, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _strict_write(path: Path, payload: Any) -> None:
    """原子写 strict JSON / Atomically write strict JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    """生成基线和历史表征报告 / Generate baseline and characterization reports."""

    tracked_sources = [
        "frailty_3class_classifier.py",
        "frailty_3class_overfitting_sweep.py",
        "frailty_3class_holdout_eval.py",
        "frailty_3class_cnn_fusion.py",
        "shapeformer_port.py",
        "funcs.py",
        "ppg.py",
        "ppg_peak_hr_gating_train.py",
        "pttppg_denoiser_hybrid_core.py",
    ]
    source_fingerprints = {
        relative: {"sha256": _sha(REPO / relative), "bytes": (REPO / relative).stat().st_size}
        for relative in tracked_sources
        if (REPO / relative).is_file()
    }
    inventory = {
        "schema_version": "ppg_frailty.baseline_inventory.v1",
        "status": "frozen_characterization_only",
        "audited_commit": "2eca0ecf0e17a4deaa1d3cc8e821098e5848e421",
        "observed_head": _git_text("rev-parse", "HEAD"),
        "observed_branch": _git_text("branch", "--show-current"),
        "spec_sha256": "cd7c4907a8beccd301048dea07ae6fdc4e9d2dc839759cb8dddd4a461e3c5000",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "scikit_learn": _version("scikit-learn"),
            "torch": _version("torch"),
            "pandas": _version("pandas"),
            "pyarrow": _version("pyarrow"),
            "onnxruntime": _version("onnxruntime"),
        },
        "internal_dataset": {
            "dataset_id": "frailty3_m2_20260815_a054800abda272f6",
            "participants": 29,
            "recordings": 261,
            "class_participants": {"Pre-Frail": 9, "Robust/Non-Frail": 12, "Young": 8},
            "roles": {role: 29 for role in ["B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2"]},
            "manifest_sha256": "bd429ae9c56974ba9ffcb924dfbad0ed930f7d2d47418365754a1929ada06e90",
            "fold_registry_id": "frailty3_future_corrected_sgkf5_v2",
            "fold_registry_payload_sha256": "0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46",
        },
        "source_fingerprints": source_fingerprints,
        "historical_result_roots": [
            "results_frailty3/20260527_1320_cnn_inceptionTime",
            "results_frailty3/20260528_1045_shapeformer_0extra",
            "results_frailty3/_overfitting_sweep/20260630_0630_overfitting_sweep_generalization_rank2",
        ],
        "cuda_available": False,
        "cudnn_version": None,
    }
    characterization = {
        "schema_version": "ppg_frailty.legacy_characterization.v1",
        "status": "historical_non_strict_not_eligible_for_v1_ranking",
        "network_snapshots": {
            "CompactCNN1D": {"parameters": 79139, "origin_name": "Cnn1DClassifier"},
            "InceptionTimeFull_single_network": {"parameters": 456579},
            "InceptionTimeSmall_single_network": {"parameters": 57027},
        },
        "known_metrics": [
            {"route": "historical_inception_outer_selected_epoch", "mean_balanced_accuracy_approx": 0.737, "eligible": False, "reason": "outer_fold_epoch_leakage"},
            {"route": "historical_shapeformer_outer_selected_epoch", "mean_balanced_accuracy_approx": 0.641, "eligible": False, "reason": "outer_fold_epoch_leakage"},
            {"route": "historical_fixed_epoch_inception_full", "mean_balanced_accuracy_approx": 0.623, "eligible": False, "reason": "not_bound_to_complete_frozen_registry_and_requires_uniform_rerun"},
            {"route": "historical_fixed_epoch_inception_small", "mean_balanced_accuracy_approx": 0.581, "eligible": False, "reason": "not_bound_to_complete_frozen_registry_and_requires_uniform_rerun"},
        ],
        "m3_live_test_observation": {
            "observed_tests": 46,
            "observed_passed": 46,
            "saved_report_tests": 22,
            "saved_report_status": "stale_not_authoritative_for_v1",
        },
        "rules": [
            "No historical score is a V1 benchmark result.",
            "All candidates must be rerun on identical frozen membership.",
            "Only source-hashed pure behavior may be migrated with parity tests.",
        ],
    }
    _strict_write(AUDIT / "baseline_inventory.json", inventory)
    _strict_write(AUDIT / "legacy_characterization.json", characterization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

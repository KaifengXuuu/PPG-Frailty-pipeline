#!/usr/bin/env python3
"""V1 严格验收门禁 / Strict V1 acceptance gate.

中文：本工具把实现规范第 4、8、10 节转换为可重复、关闭失败的机器检查，
并验证真实 reduced r0/f0 smoke 的完整追溯工件。它自身不训练模型，不生成论文
性能数字，也不会把 reduced/synthetic 证据包装成 frailty 或 PTT 基准。缺少证据记为 ``pending``；结构漂移、
错误声明或不可解析工件记为 ``failed``。

English: This tool converts specification sections 4, 8, and 10 into repeatable,
fail-closed checks and validates complete provenance artifacts from a real reduced
r0/f0 smoke. It does not itself train or promote reduced/synthetic evidence to a
frailty/PTT benchmark. Evidence that has not run is ``pending``; structural
drift, false claims, and malformed artifacts are ``failed``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]
SOURCE_ROOT = PIPELINE_ROOT / "src"

# 中文：导入必须绑定当前 V1，而不是根目录历史脚本。
# English: imports must resolve to this V1 package, never to historical root scripts.
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


SPEC_SHA256 = "cd7c4907a8beccd301048dea07ae6fdc4e9d2dc839759cb8dddd4a461e3c5000"
SPEC_BYTES = 41_122
SPEC_LINES = 766
EXPECTED_SEEDS = (42, 10_042, 20_042, 30_042, 40_042)
EXPECTED_ROLES = ("B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2")
EXPECTED_CHANNELS = ("RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ")
EXPECTED_LOCAL_HASHES = {
    "manifests/internal_records_v1.csv": "5b5788fff09910e6c224e2548869f4085fd2bbb480adcc92e0f11b09ee0387ee",
    "splits/sgkf5_repeats_v1.csv": "1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702",
    "splits/sgkf5_v1.csv": "130b2887eb29a5a534397b4ce4dc7032f9de30ae46533fa0b2c41559ff4a1284",
}
EXPECTED_AUTHORITY_HASHES = {
    "registry_file": "c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c",
    "registry_payload": "0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46",
    "source_manifest": "bd429ae9c56974ba9ffcb924dfbad0ed930f7d2d47418365754a1929ada06e90",
}


# 中文：列表逐字来自规范第 4 节；额外兼容文件可以存在，但这些规范路径不可缺。
# English: this list is transcribed from section 4; compatibility files may coexist,
# but every canonical path below is mandatory.
REQUIRED_PACKAGE_FILES = (
    "src/ppg_frailty/config.py",
    "src/ppg_frailty/provenance.py",
    "src/ppg_frailty/contracts.py",
    "src/ppg_frailty/pipeline.py",
    "src/ppg_frailty/data/schema.py",
    "src/ppg_frailty/data/manifest.py",
    "src/ppg_frailty/data/external_manifest.py",
    "src/ppg_frailty/data/qc.py",
    "src/ppg_frailty/data/folds.py",
    "src/ppg_frailty/signal/views.py",
    "src/ppg_frailty/signal/ppg_preprocess.py",
    "src/ppg_frailty/signal/imu_preprocess.py",
    "src/ppg_frailty/signal/window_plan.py",
    "src/ppg_frailty/signal/resample.py",
    "src/ppg_frailty/quality/components.py",
    "src/ppg_frailty/quality/endpoint_sqi.py",
    "src/ppg_frailty/quality/routing.py",
    "src/ppg_frailty/artifact/base.py",
    "src/ppg_frailty/artifact/identity.py",
    "src/ppg_frailty/artifact/nlms.py",
    "src/ppg_frailty/artifact/decomposition.py",
    "src/ppg_frailty/artifact/spectral.py",
    "src/ppg_frailty/artifact/bss.py",
    "src/ppg_frailty/peaks/aboy_project.py",
    "src/ppg_frailty/peaks/intervals.py",
    "src/ppg_frailty/peaks/pairing.py",
    "src/ppg_frailty/features/registry.py",
    "src/ppg_frailty/features/prv.py",
    "src/ppg_frailty/features/morphology.py",
    "src/ppg_frailty/features/dual_wavelength.py",
    "src/ppg_frailty/features/spectral.py",
    "src/ppg_frailty/features/engineering.py",
    "src/ppg_frailty/features/file_vector.py",
    "src/ppg_frailty/features/ordered_matrix.py",
    "src/ppg_frailty/representations/modes.py",
    "src/ppg_frailty/representations/raw.py",
    "src/ppg_frailty/representations/feature_vector.py",
    "src/ppg_frailty/representations/feature_matrix.py",
    "src/ppg_frailty/representations/fusion.py",
    "src/ppg_frailty/models/compact_cnn.py",
    "src/ppg_frailty/models/inception_time_port.py",
    "src/ppg_frailty/models/inception_ensemble.py",
    "src/ppg_frailty/models/shapeformer_port.py",
    "src/ppg_frailty/models/feature_models.py",
    "src/ppg_frailty/models/rocket_ridge.py",
    "src/ppg_frailty/models/file_fusion.py",
    "src/ppg_frailty/train/datasets.py",
    "src/ppg_frailty/train/sampling.py",
    "src/ppg_frailty/train/losses.py",
    "src/ppg_frailty/train/trainer.py",
    "src/ppg_frailty/train/selection.py",
    "src/ppg_frailty/evaluate/aggregate.py",
    "src/ppg_frailty/evaluate/metrics.py",
    "src/ppg_frailty/evaluate/oof.py",
    "src/ppg_frailty/evaluate/calibration.py",
    "src/ppg_frailty/evaluate/benchmark.py",
    "src/ppg_frailty/bundle/schema.py",
    "src/ppg_frailty/bundle/save.py",
    "src/ppg_frailty/bundle/load.py",
    "src/ppg_frailty/bundle/infer.py",
    "src/ppg_frailty/cli.py",
)

REQUIRED_TOP_LEVEL_FILES = (
    "configs/reference_static_v1.yaml",
    "configs/reference_all_roles_v1.yaml",
    "configs/motion_benchmark_v1.yaml",
    "configs/feature_matrix_v1.yaml",
    "manifests/internal_records_v1.csv",
    "splits/sgkf5_repeats_v1.csv",
    "tests/__init__.py",
)

CANONICAL_FACADE_MODULES = (
    "ppg_frailty.signal.ppg_preprocess",
    "ppg_frailty.signal.imu_preprocess",
    "ppg_frailty.signal.window_plan",
    "ppg_frailty.signal.resample",
    "ppg_frailty.features.prv",
    "ppg_frailty.features.morphology",
    "ppg_frailty.features.dual_wavelength",
    "ppg_frailty.features.spectral",
    "ppg_frailty.features.file_vector",
    "ppg_frailty.features.ordered_matrix",
    "ppg_frailty.models.inception_time_port",
    "ppg_frailty.models.inception_ensemble",
    "ppg_frailty.models.feature_models",
    "ppg_frailty.models.rocket_ridge",
    "ppg_frailty.models.file_fusion",
)

EXPECTED_REPRESENTATIONS = {"raw", "feature_vector", "feature_matrix", "fusion"}
EXPECTED_ARTIFACTS = {
    "identity",
    "nlms_imu_anc",
    "ssa_decomposition",
    "spectral_mask",
    "pca_bss",
    "fastica_bss",
    "nmf_bss",
}
EXPECTED_MODELS = {
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
}

# English: The feature-vector runner emits file/participant rows with this exact
# trace schema. Empty window/member tables intentionally use a different two-column
# scientific-empty schema and are checked separately.
# 中文：feature-vector runner 的 file/participant 行必须具备以下完整追溯 schema；
# window/member 科学空表使用独立双列 schema，后文单独校验。
EXPECTED_OOF_TRACE_FIELDS = (
    "participant_id", "file_id", "role", "label", "probabilities", "repeat",
    "fold", "seed", "config_hash", "manifest_hash", "fold_hash",
    "preprocessing_hash", "feature_hash", "model_hash", "representation_mode",
    "signal_route", "quality_score", "retained", "level", "window_id",
    "member_index", "class_order", "code_commit", "data_schema_id",
    "feature_schema_id", "model_version", "aggregation_rule", "environment_hash",
    "manifest_version", "fold_registry_version", "artifact_reducer_name",
    "artifact_reducer_version", "route_status", "rejection_reason",
)
REAL_REDUCED_ARTIFACTS = {
    "experiment_result.json",
    "run_manifest.json",
    "metrics_per_fold_seed.json",
    "confusion_matrices.json",
    "oof_window_predictions.parquet",
    "oof_file_predictions.parquet",
    "oof_subject_predictions.parquet",
    "oof_member_predictions.parquet",
}
FORMAL_EXPERIMENT_REPRESENTATION = "feature_vector"
NONFORMAL_REPRESENTATIONS = {"raw", "feature_matrix", "fusion"}

TYPED_CONTAINER_FIELDS = {
    "ManifestRow": {"record_id", "participant_id", "class_id", "role", "source_path"},
    "SignalViews": {"x_native", "x_filter", "x_analysis", "imu_processed", "metadata"},
    "QualityResult": {"q_rate", "q_morph", "state", "components", "reasons", "coverage"},
    "PulseResult": {
        "peaks",
        "accepted_peak_mask",
        "ppi_s",
        "valid_interval_mask",
        "adjacency_mask",
        "wavelength",
        "detector_version",
    },
    "FeatureVectorV1": {"values", "validity", "schema_version", "provenance"},
    "EngineeringFeatureSequence": {"values", "start_samples", "valid_row_mask", "schema_version"},
    "OrderedFeatureMatrixV1": {
        "values",
        "row_mask",
        "channel_schema",
        "context_schema",
        "schema_version",
    },
    "PredictionBundle": {
        "file_probabilities",
        "participant_probabilities",
        "coverage",
        "route",
        "model_version",
    },
}

FORBIDDEN_LEGACY_MODULES = {
    "funcs",
    "ppg",
    "frailty_3class_classifier",
    "frailty_3class_overfitting_sweep",
    "frailty_3class_holdout_eval",
    "frailty_3class_cnn_fusion",
    "shapeformer_port",
    "ppg_peak_hr_gating_train",
    "pttppg_detector_v8_scores_audit_fix9",
    "pttppg_denoiser_hybrid_core",
    "pttppg_denoiser_hybrid_train",
}

MODEL_CARD_REQUIRED_TEXT = (
    "Scientific status / 科学状态",
    "Representation mode / 表征",
    "Eligible signal routes / 可用信号路线",
    "Independent test / 独立测试",
    "Identity and deviation / 身份与偏离",
    "Limitations / 限制",
    "Required provenance / 必需追溯字段",
)

# 中文：只把真正的结局指标键视为性能声明；runtime/shape/parameter 数量不是论文结果。
# English: only outcome metric keys trigger the scientific-claim gate; runtime, shape,
# and parameter counts are implementation measurements rather than clinical claims.
SCIENTIFIC_METRIC_TOKENS = (
    "balanced_accuracy",
    "macro_f1",
    "worst_class_recall",
    "worst_class_f1",
    "event_precision",
    "event_recall",
    "event_f1",
    "timing_mae",
    "hr_mae",
    "hr_rmse",
    "ppi_mae",
    "brier_score",
    "expected_calibration_error",
)


class AcceptanceFailure(AssertionError):
    """验收不变量被违反 / An acceptance invariant was violated."""


@dataclass(frozen=True)
class CheckResult:
    """一个 strict-JSON 检查结果 / One strict-JSON-compatible result."""

    check_id: str
    status: str
    detail: str
    evidence: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    """逐块计算文件身份 / Hash a file in bounded-memory chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """生成 deterministic strict JSON / Produce deterministic strict JSON bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def python_tree_snapshot(test_root: Path) -> dict[str, Any]:
    """绑定测试源码路径、字节和 hash / Bind test paths, sizes, and hashes."""

    rows = []
    for path in sorted(test_root.rglob("*.py")):
        rows.append(
            {
                "path": path.relative_to(test_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "algorithm": "sha256(canonical_json(file_path,bytes,sha256))",
        "file_count": len(rows),
        "tree_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
        "files": rows,
    }


def active_source_snapshot(root: Path) -> dict[str, Any]:
    """绑定活动源码、测试与配置 / Bind active source, tests, and configs."""

    paths: list[Path] = []
    for relative in ("src", "tools", "tests"):
        base = root / relative
        if base.is_dir():
            paths.extend(base.rglob("*.py"))
    config_root = root / "configs"
    if config_root.is_dir():
        paths.extend(config_root.glob("*.yaml"))
    for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt"):
        candidate = root / name
        if candidate.is_file():
            paths.append(candidate)
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(paths), key=lambda value: value.relative_to(root).as_posix())
    ]
    return {
        "schema_version": "ppg_frailty.active_source_snapshot.v1",
        "algorithm": "sha256(canonical_json(file_path,bytes,sha256))",
        "file_count": len(rows),
        "tree_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
        "files": rows,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """原子写 strict JSON，拒绝 NaN / Atomically write strict JSON and reject NaN."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_strict_json(path: Path) -> Any:
    """解析并拒绝非有限 JSON 常量 / Parse JSON while rejecting non-finite constants."""

    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON token: {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _result(check_id: str, detail: str, **evidence: Any) -> CheckResult:
    """构造通过结果 / Construct a passed result."""

    return CheckResult(check_id, "passed", detail, evidence)


def _pending(check_id: str, detail: str, **evidence: Any) -> CheckResult:
    """构造未运行证据结果 / Construct a not-yet-run evidence result."""

    return CheckResult(check_id, "pending", detail, evidence)


def check_target_package(root: Path, _: Path) -> CheckResult:
    """核对第 4 节逐路径边界 / Verify every section-4 canonical path."""

    required = REQUIRED_PACKAGE_FILES + REQUIRED_TOP_LEVEL_FILES
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise AcceptanceFailure(f"missing canonical files: {missing}")
    return _result("target_package_boundary", f"canonical_files={len(required)}", files=len(required))


def check_spec_lock(root: Path, repository: Path) -> CheckResult:
    """逐字节复算规范锁 / Recompute the byte-level specification lock."""

    lock_path = root / "docs/spec/SPEC_LOCK.json"
    lock = load_strict_json(lock_path)
    source = repository / str(lock["source_path"])
    payload = source.read_bytes()
    observed_lines = len(payload.splitlines())
    observed_hash = hashlib.sha256(payload).hexdigest()
    expected = (SPEC_SHA256, SPEC_BYTES, SPEC_LINES)
    observed = (observed_hash, len(payload), observed_lines)
    if observed != expected:
        raise AcceptanceFailure(f"spec identity drift: expected={expected}, observed={observed}")
    if (
        lock.get("source_sha256") != SPEC_SHA256
        or int(lock.get("source_bytes", -1)) != SPEC_BYTES
        or int(lock.get("source_lines", -1)) != SPEC_LINES
    ):
        raise AcceptanceFailure("SPEC_LOCK fields do not match the byte-verified source")
    return _result("spec_lock", f"sha256={observed_hash}", bytes=len(payload), lines=observed_lines)


def check_manifest_and_folds(root: Path, _: Path) -> CheckResult:
    """锁定 261/29/9 roles 与 5×5 分组折叠 / Lock manifest and 5x5 groups."""

    manifest_path = root / "manifests/internal_records_v1.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 261:
        raise AcceptanceFailure(f"manifest rows={len(manifest)}, expected=261")
    record_ids = [row["record_id"] for row in manifest]
    if len(record_ids) != len(set(record_ids)):
        raise AcceptanceFailure("manifest record_id is not unique")
    participants = {row["participant_id"] for row in manifest}
    roles = Counter(row["role"] for row in manifest)
    if len(participants) != 29 or roles != Counter({role: 29 for role in EXPECTED_ROLES}):
        raise AcceptanceFailure(f"manifest roster drift: participants={len(participants)}, roles={dict(roles)}")
    for row in manifest:
        if tuple(json.loads(row["channel_schema"])) != EXPECTED_CHANNELS:
            raise AcceptanceFailure(f"channel order drift: {row['record_id']}")
    participant_classes: dict[str, int] = {}
    for row in manifest:
        class_id = int(row["class_id"])
        previous = participant_classes.setdefault(row["participant_id"], class_id)
        if previous != class_id:
            raise AcceptanceFailure(f"participant label changed across roles: {row['participant_id']}")
    if Counter(participant_classes.values()) != Counter({0: 9, 1: 12, 2: 8}):
        raise AcceptanceFailure(f"participant class counts drift: {Counter(participant_classes.values())}")

    repeat_path = root / "splits/sgkf5_repeats_v1.csv"
    with repeat_path.open("r", encoding="utf-8", newline="") as handle:
        folds = list(csv.DictReader(handle))
    if len(folds) != 145:
        raise AcceptanceFailure(f"repeated fold rows={len(folds)}, expected=145")
    by_repeat: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in folds:
        by_repeat[int(row["repeat_index"])].append(row)
        if row["source_registry_file_sha256"] != EXPECTED_AUTHORITY_HASHES["registry_file"]:
            raise AcceptanceFailure("source registry file hash drift inside fold rows")
        if row["source_registry_payload_sha256"] != EXPECTED_AUTHORITY_HASHES["registry_payload"]:
            raise AcceptanceFailure("source registry payload hash drift inside fold rows")
    if set(by_repeat) != set(range(5)):
        raise AcceptanceFailure(f"repeat indices drift: {sorted(by_repeat)}")
    for repeat, expected_seed in enumerate(EXPECTED_SEEDS):
        rows = by_repeat[repeat]
        if len(rows) != 29 or {int(row["split_seed"]) for row in rows} != {expected_seed}:
            raise AcceptanceFailure(f"repeat {repeat} roster/seed mismatch")
        if {int(row["fold_index"]) for row in rows} != set(range(5)):
            raise AcceptanceFailure(f"repeat {repeat} does not contain five folds")
        ids = [row["participant_id"] for row in rows]
        if set(ids) != participants or len(ids) != len(set(ids)):
            raise AcceptanceFailure(f"repeat {repeat} is not an exact OOF participant partition")
        for fold in range(5):
            fold_rows = [row for row in rows if int(row["fold_index"]) == fold]
            if {int(row["class_id"]) for row in fold_rows} != {0, 1, 2}:
                raise AcceptanceFailure(f"repeat={repeat}, fold={fold} lacks a class")

    observed_hashes = {relative: sha256_file(root / relative) for relative in EXPECTED_LOCAL_HASHES}
    if observed_hashes != EXPECTED_LOCAL_HASHES:
        raise AcceptanceFailure(f"materialized manifest/fold hash drift: {observed_hashes}")
    report = load_strict_json(root / "reports/data_contract_report.json")
    audit = report["frozen_fold_authority"]["audit"]
    required_audit = {
        "all_classes_present": True,
        "oof_partition_exact": True,
        "train_oof_disjoint": True,
        "class_balance_spread_at_most_one": True,
        "participant_count": 29,
        "repeat_count": 5,
        "fold_count_per_repeat": 5,
    }
    for key, expected_value in required_audit.items():
        if audit.get(key) != expected_value:
            raise AcceptanceFailure(f"data contract audit mismatch: {key}={audit.get(key)!r}")
    return _result(
        "manifest_frozen_5x5",
        "records=261 participants=29 roles=9 repeats=5 folds=5",
        record_count=261,
        participant_count=29,
        role_counts=dict(sorted(roles.items())),
        seeds=list(EXPECTED_SEEDS),
        local_hashes=observed_hashes,
    )


def check_formal_config_preflights(root: Path, _: Path) -> CheckResult:
    """用真实 loader/full preflight 验证 4/4 配置 / Run all four real preflights."""

    from ppg_frailty.pipeline import PipelinePaths, preflight_pipeline

    paths = PipelinePaths.discover()
    if paths.pipeline_root.resolve() != root.resolve():
        raise AcceptanceFailure("imported PipelinePaths does not bind the requested V1 root")
    names = (
        "reference_static_v1.yaml",
        "reference_all_roles_v1.yaml",
        "motion_benchmark_v1.yaml",
        "feature_matrix_v1.yaml",
    )
    rows = []
    for name in names:
        report, config, _manifest, _registry = preflight_pipeline(
            root / "configs" / name,
            mode="full",
            paths=paths,
        )
        if report.status != "passed" or report.split_count != 25:
            raise AcceptanceFailure(f"{name} did not resolve all 25 held-out splits")
        if tuple(report.split_seeds) != EXPECTED_SEEDS:
            raise AcceptanceFailure(f"{name} split seed drift: {report.split_seeds}")
        rows.append(
            {
                "config": name,
                "config_id": config.config_id,
                "config_sha256": config.sha256,
                "representation_mode": config.representation_mode,
                "split_count": report.split_count,
            }
        )
    return _result("formal_config_preflights", "passed=4/4 full_preflight", configs=rows)


def _import_implementation(path: str) -> str:
    """导入 module 或 module.attribute / Import a module or dotted attribute."""

    try:
        module = importlib.import_module(path)
        return module.__name__
    except ModuleNotFoundError as direct_error:
        module_name, separator, attribute = path.rpartition(".")
        if not separator:
            raise direct_error
        module = importlib.import_module(module_name)
        if not hasattr(module, attribute):
            raise AcceptanceFailure(f"registered implementation missing: {path}")
        return f"{module.__name__}.{attribute}"


def check_module_registry_and_facades(root: Path, _: Path) -> CheckResult:
    """核对唯一注册表、实现导入和 facade / Verify registry, imports, and facades."""

    from ppg_frailty.module_registry import list_modules, registry_sha256

    rows = list_modules()
    ids = [str(row["module_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise AcceptanceFailure("module registry has duplicate module_id values")
    families = defaultdict(set)
    imported = []
    for row in rows:
        families[str(row["family"])].add(str(row["module_id"]))
        imported.append(_import_implementation(str(row["implementation"])))
        if not str(row.get("quantitative_suite", "")).strip():
            raise AcceptanceFailure(f"module lacks quantitative suite: {row['module_id']}")
    expected = {
        "representation": EXPECTED_REPRESENTATIONS,
        "artifact": EXPECTED_ARTIFACTS,
        "model": EXPECTED_MODELS,
    }
    if dict(families) != expected:
        raise AcceptanceFailure(f"registry coverage drift: {dict(families)}")
    facade_exports = {}
    for module_name in CANONICAL_FACADE_MODULES:
        module = importlib.import_module(module_name)
        exports = tuple(getattr(module, "__all__", ()))
        if not exports:
            raise AcceptanceFailure(f"canonical facade has no explicit __all__: {module_name}")
        facade_exports[module_name] = len(exports)
    digest = registry_sha256()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise AcceptanceFailure("module registry hash is not SHA-256")
    return _result(
        "module_registry_public_facades",
        f"modules={len(rows)} facades={len(facade_exports)}",
        registry_sha256=digest,
        family_counts={key: len(value) for key, value in sorted(families.items())},
        imported_implementations=len(imported),
        facade_export_counts=facade_exports,
    )


def check_typed_containers(_: Path, __: Path) -> CheckResult:
    """核对第 4 节 typed containers / Verify typed section-4 containers."""

    from ppg_frailty import contracts

    observed = {}
    for name, required_fields in TYPED_CONTAINER_FIELDS.items():
        candidate = getattr(contracts, name, None)
        if candidate is None or not is_dataclass(candidate):
            raise AcceptanceFailure(f"{name} is missing or is not a dataclass")
        names = {item.name for item in fields(candidate)}
        missing = required_fields - names
        if missing:
            raise AcceptanceFailure(f"{name} missing fields: {sorted(missing)}")
        observed[name] = sorted(names)
    if contracts.QualityState.NOT_APPLICABLE.value != "not_applicable":
        raise AcceptanceFailure("QualityResult.q_morph cannot express not_applicable")
    return _result("typed_containers", f"typed_containers={len(observed)}", containers=observed)


def _python_paths(root: Path) -> list[Path]:
    """返回活动代码和测试，排除空 init / Return active code and tests."""

    paths = []
    for relative in ("src", "tools", "tests"):
        base = root / relative
        if base.is_dir():
            paths.extend(sorted(base.rglob("*.py")))
    return paths


def check_python_ast_bilingual(root: Path, _: Path) -> CheckResult:
    """解析 AST 并验证文件级中英文说明 / Parse AST and require bilingual files."""

    failures = []
    checked = 0
    for path in _python_paths(root):
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as error:
            failures.append(f"{path.relative_to(root)}: syntax={error}")
            continue
        if path.name == "__init__.py" and not source.strip():
            continue
        checked += 1
        if re.search(r"[\u4e00-\u9fff]", source) is None:
            failures.append(f"{path.relative_to(root)}: missing Chinese explanation")
        if re.search(r"[A-Za-z]{4,}", source) is None:
            failures.append(f"{path.relative_to(root)}: missing English explanation")
    if failures:
        raise AcceptanceFailure("; ".join(failures))
    return _result("python_ast_bilingual", f"python_files={checked}", checked_files=checked)


def check_no_legacy_imports_or_unfinished(root: Path, _: Path) -> CheckResult:
    """拒绝历史运行时依赖和空实现 / Reject legacy imports and unfinished code."""

    violations = []
    scanned = 0
    for path in sorted((root / "src/ppg_frailty").rglob("*.py")) + sorted((root / "tools").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        scanned += 1
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module.split(".")[0]]
            overlap = FORBIDDEN_LEGACY_MODULES.intersection(imported)
            if overlap:
                violations.append(f"{path.relative_to(root)}: legacy imports={sorted(overlap)}")
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                if isinstance(node.exc.func, ast.Name) and node.exc.func.id == "NotImplementedError":
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: NotImplementedError")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [item for item in node.body if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str))]
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: pass-only function {node.name}")
                if (
                    len(body) == 1
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and body[0].value.value is Ellipsis
                ):
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: ellipsis-only function {node.name}")
    if violations:
        raise AcceptanceFailure("; ".join(violations))
    return _result("no_legacy_or_unfinished", f"python_files={scanned}", violations=0)


def check_strict_json_tree(root: Path, _: Path) -> CheckResult:
    """解析所有 V1 JSON 并拒绝 NaN/Infinity / Parse every V1 JSON strictly."""

    count = 0
    for path in sorted(root.rglob("*.json")):
        load_strict_json(path)
        count += 1
    return _result("strict_json_tree", f"json_files={count}", json_files=count)


def check_model_cards(root: Path, _: Path) -> CheckResult:
    """确保每个注册模型有完整模型卡 / Require a complete card per model."""

    card_root = root / "model_cards"
    cards = [path for path in sorted(card_root.glob("*.md")) if path.name != "README.md"]
    if len(cards) != len(EXPECTED_MODELS):
        raise AcceptanceFailure(f"model cards={len(cards)}, registered models={len(EXPECTED_MODELS)}")
    headings = set()
    for path in cards:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("# "):
            raise AcceptanceFailure(f"model card lacks title: {path.name}")
        headings.add(text.splitlines()[0][2:].strip())
        missing = [marker for marker in MODEL_CARD_REQUIRED_TEXT if marker not in text]
        if missing:
            raise AcceptanceFailure(f"{path.name} missing card fields: {missing}")
        if "independent_test=false" not in text:
            raise AcceptanceFailure(f"{path.name} does not explicitly deny independent-test evidence")
    if headings != EXPECTED_MODELS:
        raise AcceptanceFailure(f"model-card/model-registry identity mismatch: {sorted(headings ^ EXPECTED_MODELS)}")
    return _result("model_cards", f"cards={len(cards)}", model_ids=sorted(headings))


def check_required_test_sources(root: Path, _: Path) -> CheckResult:
    """核对第 8 节测试类别均有源码 / Require source for every test category."""

    categories = (
        "audit",
        "contracts",
        "data",
        "signal",
        "artifacts",
        "features",
        "models",
        "training",
        "integration",
        "acceptance",
    )
    counts = {}
    missing = []
    # 中文：目录存在不足以证明关键回归守卫；逐项锁定可审计测试名称/fixture。
    # English: directory presence alone is insufficient; bind critical regression
    # guards to explicit test names or frozen fixture markers.
    required_markers = {
        'external_ecg_one_to_one_and_symmetric_schema': (
            'test_raw_quality_reducer_have_symmetric_quantitative_schema',
        ),
        'label_shuffle_sanity': (
            'test_label_shuffle_destroys_separable_feature_signal',
        ),
        'technical_metadata_exclusion': (
            'participant_id',
            'build_feature_vector',
            'assertRaises(ValueError)',
        ),
        'heldout_training_perturbation': (
            'test_heldout_roster_mutation_cannot_change_fitted_state',
        ),
        'heldout_sqi_perturbation': (
            'test_empirical_calibrator_ignores_heldout_mutation',
        ),
        'four_formal_config_preflights': (
            'test_all_four_materialized_configs_preflight',
        ),
        'rocket_10000_full_serialization': (
            'test_rocket_10000_full_config_serialization_parity',
            'n_kernels=10_000',
        ),
    }
    corpus = chr(10).join(
        path.read_text(encoding='utf-8')
        for path in sorted((root / 'tests').rglob('test_*.py'))
    )
    absent_markers = [
        name
        for name, markers in required_markers.items()
        if not all(marker in corpus for marker in markers)
    ]
    if absent_markers:
        raise AcceptanceFailure(f'missing required regression guards: {absent_markers}')
    for category in categories:
        count = len(list((root / "tests" / category).glob("test_*.py")))
        counts[category] = count
        if count == 0:
            missing.append(category)
    if missing:
        raise AcceptanceFailure(f"required test categories have no source: {missing}")
    return _result("required_test_sources", f"categories={len(categories)}", category_file_counts=counts)


def check_current_test_report(root: Path, _: Path) -> CheckResult:
    """证明测试报告对应当前源码 / Prove the report binds the current test tree."""

    path = root / "artifacts/acceptance/cpu_ci_tests_current.json"
    if not path.is_file():
        return _pending("current_test_report", "CPU test snapshot has not run", expected_path=path.relative_to(root).as_posix())
    report = load_strict_json(path)
    current = python_tree_snapshot(root / "tests")
    reported = report.get("test_source_snapshot")
    if reported is None:
        raise AcceptanceFailure("current test report has no source snapshot")
    if reported.get("tree_sha256") != current["tree_sha256"] or int(reported.get("file_count", -1)) != current["file_count"]:
        raise AcceptanceFailure("test report is stale relative to current test source")
    counts = report.get("counts", {})
    if report.get('warnings_policy') != 'error':
        raise AcceptanceFailure('current CPU test report did not run with warnings=error')
    if report.get("suite") != "all" or report.get("status") != "passed":
        raise AcceptanceFailure("current all-suite report is not passed")
    if int(counts.get("failed", -1)) != 0 or int(counts.get("errors", -1)) != 0:
        raise AcceptanceFailure(f"current all-suite has failures: {counts}")
    if int(counts.get("skipped", -1)) != 0:
        raise AcceptanceFailure(f"current all-suite contains skipped tests: {counts}")
    if int(counts.get("run", 0)) < 1:
        raise AcceptanceFailure("current all-suite ran zero tests")
    return _result(
        "current_test_report",
        f"tests={counts.get('run')} source_sha256={current['tree_sha256']}",
        report_path=path.relative_to(root).as_posix(),
        counts=counts,
        test_source_snapshot=current,
    )


def check_current_source_snapshot(root: Path, _: Path) -> CheckResult:
    """证明 CPU CI 绑定当前活动源码 / Prove CPU CI binds current active source."""

    path = root / "artifacts/acceptance/source_snapshot_current.json"
    if not path.is_file():
        return _pending(
            "current_source_snapshot",
            "active source snapshot has not run",
            expected_path=path.relative_to(root).as_posix(),
        )
    reported = load_strict_json(path)
    current = active_source_snapshot(root)
    if (
        reported.get("tree_sha256") != current["tree_sha256"]
        or int(reported.get("file_count", -1)) != current["file_count"]
        or reported.get("files") != current["files"]
    ):
        raise AcceptanceFailure("active source snapshot is stale relative to src/tools/tests/configs")
    return _result(
        "current_source_snapshot",
        f"files={current['file_count']} source_sha256={current['tree_sha256']}",
        report_path=path.relative_to(root).as_posix(),
        file_count=current["file_count"],
        tree_sha256=current["tree_sha256"],
    )


def _latest_artifact(root: Path, patterns: Sequence[str]) -> Path | None:
    """选择最新机器工件，仅用于 current gate / Select newest current evidence."""

    candidates = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    files = [path for path in candidates if path.is_file()]
    return max(files, key=lambda path: (path.stat().st_mtime_ns, path.as_posix())) if files else None


def _require_synthetic_scope(payload: Mapping[str, Any], path: Path) -> None:
    """要求结果明确否认真实 benchmark / Require an explicit non-benchmark scope."""

    scope = str(payload.get("scientific_scope", "")).lower()
    if "synthetic" not in scope or ("not" not in scope and "contract" not in scope):
        raise AcceptanceFailure(f"synthetic metric artifact lacks non-benchmark scope: {path}")


def check_quantitative_artifacts(root: Path, _: Path) -> CheckResult:
    """验证并行模块和时间消融的机器工件 / Verify quantitative module evidence."""

    patterns = {
        "artifact": (
            "artifacts/acceptance/runs/artifact_parallel_*.json",
            "artifacts/test_reports/artifact_comparison_canonical_manual.json",
            "artifacts/test_reports/artifact_comparison_manual.json",
        ),
        "model": ("artifacts/acceptance/runs/model_parallel_*.json", "artifacts/test_reports/model_comparison_manual.json"),
        "dl_fs": ("artifacts/acceptance/runs/dl_fs_ablation_*.json", "artifacts/test_reports/dl_fs_ablation_manual.json"),
        "raw_window": ("artifacts/acceptance/runs/raw_window_ablation_*.json",),
        "physical_time": (
            "artifacts/acceptance/runs/physical_time_ablation_*.json",
            "artifacts/test_reports/physical_time_contract_manual.json",
            "artifacts/test_reports/physical_time_ablation_manual.json",
        ),
        "cli_smoke": ("artifacts/acceptance/runs/cli_smoke_*.json", "artifacts/test_reports/integration_smoke_manual.json"),
    }
    selected = {name: _latest_artifact(root, value) for name, value in patterns.items()}
    absent = [name for name, path in selected.items() if path is None]
    if absent:
        return _pending("quantitative_artifacts", f"evidence not yet run: {absent}", missing=absent)

    artifact = load_strict_json(selected["artifact"])
    _require_synthetic_scope(artifact, selected["artifact"])
    artifact_ids = {
        str(row.get('canonical_module_id'))
        for row in artifact.get('results', [])
    }
    controls = {'raw_no_denoise', 'quality_only'}
    if artifact_ids - controls != EXPECTED_ARTIFACTS or not controls.issubset(artifact_ids):
        raise AcceptanceFailure(f'artifact comparison coverage mismatch: {artifact_ids}')
    for row in artifact["results"]:
        module_id = row.get('canonical_module_id')
        if (
            module_id in EXPECTED_ARTIFACTS - {'identity'}
            and row.get('q_morph_state') != 'not_applicable'
        ):
            raise AcceptanceFailure(
                f'non-identity reducer emitted applicable morphology: {module_id}'
            )

    model = load_strict_json(selected["model"])
    _require_synthetic_scope(model, selected["model"])
    model_ids = {str(row.get("model_id")) for row in model.get("results", [])}
    if model_ids != EXPECTED_MODELS:
        raise AcceptanceFailure(f"model comparison coverage mismatch: {model_ids}")
    if any(not bool(row.get("finite_probabilities")) for row in model["results"]):
        raise AcceptanceFailure("model comparison contains non-finite probabilities")

    dl_fs = load_strict_json(selected["dl_fs"])
    if {int(row["dl_fs_hz"]) for row in dl_fs.get("results", [])} != {100, 160, 200, 400}:
        raise AcceptanceFailure("dl_fs ablation does not contain the four specified rates")
    raw_window = load_strict_json(selected["raw_window"])
    if {float(row["window_s"]) for row in raw_window.get("results", [])} != {5.0, 10.0}:
        raise AcceptanceFailure("raw-window ablation does not contain 5 s and 10 s")
    physical = load_strict_json(selected["physical_time"])
    if int(physical.get("case_count", -1)) != 64:
        raise AcceptanceFailure("physical-time ablation is not the 4x2x2x4 grid")
    scope = str(physical.get("scientific_scope", ""))
    if "not_frozen_5x5_benchmark" not in scope:
        raise AcceptanceFailure("physical-time contract audit is not clearly separated from benchmark evidence")
    smoke = load_strict_json(selected["cli_smoke"])
    if smoke.get("scientific_metrics_emitted") is not False or smoke.get("status") != "smoke_passed":
        raise AcceptanceFailure("CLI smoke must pass without emitting scientific metrics")
    return _result(
        "quantitative_artifacts",
        "artifact=2_controls_plus_7_reducers model=13 dl_fs=4 window=2 physical_time=64 cli_smoke=passed",
        paths={name: path.relative_to(root).as_posix() for name, path in selected.items()},
    )


def _walk_json_keys(value: Any) -> Iterable[tuple[str, Any]]:
    """递归遍历 JSON 键值 / Recursively walk JSON key-value pairs."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_json_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json_keys(item)


def check_no_fabricated_scientific_results(root: Path, _: Path) -> CheckResult:
    """阻止 synthetic/历史分数冒充锁定结果 / Reject unsupported metric claims."""

    violations = []
    metric_artifacts = 0
    # English: Metrics/confusion companion JSON files inherit the explicit scope
    # from their colocated experiment_result. The latest current run is separately
    # subjected to the complete OOF validator; this set only prevents its companion
    # files (which intentionally avoid duplicating scope fields) being misclassified.
    # 中文：metrics/confusion 配套 JSON 从同目录 experiment_result 继承明确 scope；
    # 最新 current run 另由完整 OOF 门禁校验。此集合仅避免未重复 scope 字段的
    # 配套文件被误判为伪造结果。
    real_smoke_json: set[Path] = set()
    for result_path in sorted((root / "artifacts").rglob("experiment_result.json")):
        experiment = load_strict_json(result_path)
        if (
            experiment.get("status") == "passed"
            and experiment.get("scientific_scope") == "smoke_not_scientific_benchmark"
            and experiment.get("config_id") == "motion_benchmark_spectral_rate_only_v1"
            and experiment.get("repeat_indices") == [0]
            and experiment.get("fold_indices") == [0]
        ):
            real_smoke_json.update(result_path.parent.glob("*.json"))
    for base in (root / "artifacts", root / "reports"):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.json")):
            payload = load_strict_json(path)
            pairs = list(_walk_json_keys(payload))
            if any(key in {"independent_test", "independent_test_available"} and value is True for key, value in pairs):
                violations.append(f"{path.relative_to(root)}: unsupported independent-test claim")
            metric_keys = [key for key, _value in pairs if any(token in key.lower() for token in SCIENTIFIC_METRIC_TOKENS)]
            if not metric_keys:
                continue
            metric_artifacts += 1
            scope = str(payload.get("scientific_scope", "")).lower() if isinstance(payload, Mapping) else ""
            schema = str(payload.get("schema_version", "")).lower() if isinstance(payload, Mapping) else ""
            status = str(payload.get("status", "")).lower() if isinstance(payload, Mapping) else ""
            if "synthetic" in scope and ("not" in scope or "contract" in scope):
                continue
            if path in real_smoke_json:
                continue
            if "audit" in schema or "historical" in status or "not_eligible" in status:
                eligible_values = [value for key, value in pairs if key == "eligible"]
                if eligible_values and all(value is False for value in eligible_values):
                    continue
            # 中文：真实结果必须具备完整身份和 OOF 明示；目前 V1 不会伪造这些字段。
            # English: real results require explicit OOF identity and complete hashes.
            keys = {key for key, _value in pairs}
            required_identity = {"config_hash", "manifest_hash", "fold_hash", "participant_id", "repeat_index", "fold_index"}
            if required_identity.issubset(keys) and "oof" in schema:
                continue
            violations.append(
                f"{path.relative_to(root)}: metric keys without synthetic/audit/full-OOF scope {sorted(set(metric_keys))[:8]}"
            )
    if violations:
        raise AcceptanceFailure("; ".join(violations))
    return _result("no_fabricated_scientific_results", f"metric_artifacts_audited={metric_artifacts}", audited=metric_artifacts)


def check_comparison_documents(root: Path, _: Path) -> CheckResult:
    """核对用户要求的五份对照评估 / Verify the five requested comparison reports."""

    expected = [
        "docs/comparisons/01_SPEC_VS_TODO_OVERLAP_AND_DIFFERENCES.md",
        "docs/comparisons/02_SPEC_VS_COMPLETED_TODO.md",
        "docs/comparisons/03_SPEC_VS_LOCAL_FROZEN_WORKFLOW.md",
        "docs/comparisons/04_ALGORITHM_REASONABLENESS_AND_TRADEOFFS.md",
        "docs/comparisons/05_V1_TO_V2_CONFIRMATION_SUMMARY.md",
    ]
    missing = [relative for relative in expected if not (root / relative).is_file()]
    if missing:
        raise AcceptanceFailure(f"missing comparison reports: {missing}")
    return _result("comparison_documents", "reports=5", paths=expected)


def check_real_fold_cli_smoke(root: Path, _: Path) -> CheckResult:
    '''核对真实源文件/真实折叠冒烟 / Verify a real-source, real-fold CLI smoke.

    中文：该工件必须读取冻结 manifest 中的一条真实 8-channel 记录，并绑定
    repeat/fold；它仍不产生未训练分类指标。

    English: the artifact must read one real 8-channel record from the frozen
    manifest and bind repeat/fold, while emitting no untrained class metrics.
    '''

    path = _latest_artifact(
        root,
        (
            'artifacts/acceptance/runs/cli_smoke_*.json',
            'artifacts/test_reports/integration_smoke_manual.json',
        ),
    )
    if path is None:
        return _pending('real_fold_cli_smoke', 'real-fold CLI smoke has not run')
    payload = load_strict_json(path)
    preflight = payload.get('preflight', {})
    record = payload.get('smoke_record', {})
    if payload.get('status') != 'smoke_passed':
        raise AcceptanceFailure('real-fold CLI smoke status is not smoke_passed')
    if payload.get('scientific_metrics_emitted') is not False:
        raise AcceptanceFailure('real-fold smoke emitted unsupported scientific metrics')
    required_preflight = {
        'record_count': 261,
        'participant_count': 29,
        'split_count': 1,
    }
    for key, expected in required_preflight.items():
        if preflight.get(key) != expected:
            raise AcceptanceFailure(f'real-fold smoke preflight {key} drift')
    if preflight.get('split_seeds') != [42]:
        raise AcceptanceFailure('real-fold smoke did not bind seed 42')
    if (
        record.get('outer_repeat') != 0
        or record.get('outer_fold') != 0
        or int(record.get('samples_read', 0)) <= 0
        or not str(record.get('record_id', '')).startswith('frailty3:')
        or not str(record.get('participant_id', '')).strip()
    ):
        raise AcceptanceFailure('real-fold smoke record identity is incomplete')
    return _result(
        'real_fold_cli_smoke',
        f'record={record.get("record_id")} repeat=0 fold=0 samples={record.get("samples_read")}',
        path=path.relative_to(root).as_posix(),
        record_id=record.get('record_id'),
        participant_id=record.get('participant_id'),
        samples_read=record.get('samples_read'),
    )


def _real_reduced_candidates(root: Path) -> list[Path]:
    """列出 CI 结果及冻结通过参考 / List CI outputs and frozen passing reference."""

    candidates = [
        path
        for path in (root / "artifacts/acceptance/runs").glob(
            "experiment_reduced_r0_f0_*"
        )
        if path.is_dir() and (path / "experiment_result.json").is_file()
    ]
    registry_path = root / "artifacts/experiments/reference_registry.json"
    if registry_path.is_file():
        registry = load_strict_json(registry_path)
        relative = registry.get("current_passing_reference", {}).get("path")
        if isinstance(relative, str):
            reference = (registry_path.parent / relative).resolve()
            reference.relative_to((root / "artifacts/experiments").resolve())
            if reference.is_dir() and (reference / "experiment_result.json").is_file():
                candidates.append(reference)
    return sorted(
        set(candidates),
        key=lambda path: (
            (path / "experiment_result.json").stat().st_mtime_ns,
            path.as_posix(),
        ),
    )


def _frozen_r0_f0_rosters(root: Path) -> tuple[set[str], set[str], dict[str, int]]:
    """读取冻结 r0/f0 train/OOF 名单 / Read frozen r0/f0 train/OOF rosters."""

    with (root / "splits/sgkf5_repeats_v1.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        repeat_rows = [
            row for row in csv.DictReader(handle) if int(row["repeat_index"]) == 0
        ]
    oof_rows = [row for row in repeat_rows if int(row["fold_index"]) == 0]
    train_rows = [row for row in repeat_rows if int(row["fold_index"]) != 0]
    oof = {row["participant_id"] for row in oof_rows}
    train = {row["participant_id"] for row in train_rows}
    labels = {row["participant_id"]: int(row["class_id"]) for row in oof_rows}
    if len(oof) != 6 or len(train) != 23 or oof & train or len(oof | train) != 29:
        raise AcceptanceFailure("frozen repeat-0/fold-0 roster is not exact 6 OOF + 23 train")
    return train, oof, labels


def _require_finite_metric(metrics: Mapping[str, Any], key: str) -> float:
    """要求指标存在且有限但不锁数值 / Require a finite metric without locking it."""

    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceFailure(f"real reduced metric is missing/non-numeric: {key}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise AcceptanceFailure(f"real reduced metric is non-finite: {key}")
    return numeric


def _validate_real_reduced_experiment(directory: Path, root: Path) -> dict[str, Any]:
    """验证真实 feature-vector r0/f0 产物 / Validate real feature-vector r0/f0 artifacts."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - the runner itself requires it.
        raise AcceptanceFailure("pyarrow is required to validate real OOF artifacts") from error

    observed_files = {path.name for path in directory.iterdir() if path.is_file()}
    if observed_files != REAL_REDUCED_ARTIFACTS:
        raise AcceptanceFailure(
            f"real reduced artifact set drift: expected={sorted(REAL_REDUCED_ARTIFACTS)}, "
            f"observed={sorted(observed_files)}"
        )
    result = load_strict_json(directory / "experiment_result.json")
    manifest = load_strict_json(directory / "run_manifest.json")
    metrics_document = load_strict_json(directory / "metrics_per_fold_seed.json")
    confusion_document = load_strict_json(directory / "confusion_matrices.json")

    from ppg_frailty.config import load_config

    config = load_config(root / "configs/motion_benchmark_v1.yaml")
    if config.representation_mode != FORMAL_EXPERIMENT_REPRESENTATION:
        raise AcceptanceFailure("motion benchmark is no longer the feature_vector formal route")
    if result.get("status") != "passed":
        raise AcceptanceFailure("real reduced experiment status is not passed")
    if result.get("scientific_scope") != "smoke_not_scientific_benchmark":
        raise AcceptanceFailure("real reduced experiment is not explicitly smoke-only")
    if result.get("config_id") != config.config_id or result.get("config_hash") != config.sha256:
        raise AcceptanceFailure("real reduced experiment does not bind the current motion config")
    if result.get("repeat_indices") != [0] or result.get("fold_indices") != [0]:
        raise AcceptanceFailure("real reduced experiment is not repeat-0/fold-0")
    if result.get("failure_reasons") != [] or len(result.get("cell_results", [])) != 1:
        raise AcceptanceFailure("real reduced experiment does not contain one successful cell")

    provenance = result.get("provenance", {})
    expected_provenance = {
        "preflight_status": "passed",
        "manifest_hash": EXPECTED_LOCAL_HASHES["manifests/internal_records_v1.csv"],
        "fold_hash": EXPECTED_LOCAL_HASHES["splits/sgkf5_repeats_v1.csv"],
        "frozen_outer_split": True,
        "outer_train_only_calibrator": True,
        "record_seconds_cap": 60.0,
        "record_cap_per_participant": 1,
        "fixed_epochs_override": 1,
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise AcceptanceFailure(f"real reduced provenance drift: {key}={provenance.get(key)!r}")

    cell = result["cell_results"][0]
    if (
        cell.get("status") != "passed"
        or cell.get("scientific_scope") != "smoke_not_scientific_benchmark"
        or cell.get("representation_mode") != FORMAL_EXPERIMENT_REPRESENTATION
        or cell.get("repeat_index") != 0
        or cell.get("fold_index") != 0
        or cell.get("split_seed") != 42
        or cell.get("training_seed") != 42
        or cell.get("selected_record_count") != 29
    ):
        raise AcceptanceFailure("real reduced cell identity/protocol drift")

    train_roster, oof_roster, oof_labels = _frozen_r0_f0_rosters(root)
    fitted = set(cell.get("fitted_provenance", {}).get("fitted_participant_ids", []))
    sqi = cell.get("sqi_calibrator_provenance", {})
    sqi_fitted = set(sqi.get("fitted_on_participant_ids", []))
    if fitted != train_roster or sqi_fitted != train_roster:
        raise AcceptanceFailure("model/SQI fitted participant IDs are not the exact outer-train roster")
    if fitted & oof_roster or sqi_fitted & oof_roster or sqi.get("outer_oof_ids_absent") is not True:
        raise AcceptanceFailure("held-out participant leaked into fitted model/SQI provenance")

    if manifest.get("status") != "passed" or manifest.get("scientific_scope") != result["scientific_scope"]:
        raise AcceptanceFailure("run manifest status/scope does not match experiment result")
    if manifest.get("cell") != cell:
        raise AcceptanceFailure("run manifest cell differs from experiment result")
    expected_manifest_artifacts = REAL_REDUCED_ARTIFACTS - {"experiment_result.json"}
    if set(manifest.get("mandatory_artifacts", [])) != expected_manifest_artifacts:
        raise AcceptanceFailure("run manifest mandatory artifact list is incomplete")
    if metrics_document.get("cells") != [cell]:
        raise AcceptanceFailure("metrics_per_fold_seed does not contain the exact experiment cell")

    metrics = result.get("metrics", {})
    if metrics != cell.get("metrics"):
        raise AcceptanceFailure("experiment and cell metrics differ")
    for key in (
        "balanced_accuracy", "macro_f1", "worst_class_precision",
        "worst_class_recall", "worst_class_f1", "multiclass_brier",
        "expected_calibration_error", "multiclass_log_loss", "coverage_rate",
    ):
        _require_finite_metric(metrics, key)
    if metrics.get("class_order") != [0, 1, 2]:
        raise AcceptanceFailure("real reduced metrics class order is not [0,1,2]")
    per_class = metrics.get("per_class", [])
    if {row.get("label") for row in per_class} != {0, 1, 2}:
        raise AcceptanceFailure("real reduced per-class metrics are incomplete")
    for row in per_class:
        for key in ("precision", "recall", "f1"):
            _require_finite_metric(row, key)

    file_table = parquet.read_table(directory / "oof_file_predictions.parquet")
    subject_table = parquet.read_table(directory / "oof_subject_predictions.parquet")
    if tuple(file_table.column_names) != EXPECTED_OOF_TRACE_FIELDS:
        raise AcceptanceFailure("file OOF trace schema drift")
    if tuple(subject_table.column_names) != EXPECTED_OOF_TRACE_FIELDS:
        raise AcceptanceFailure("subject OOF trace schema drift")
    file_rows = file_table.to_pylist()
    subject_rows = subject_table.to_pylist()
    if len(file_rows) != len(oof_roster) or len(subject_rows) != len(oof_roster):
        raise AcceptanceFailure("file/subject OOF does not contain one row per held-out participant")
    for level, rows in (("file", file_rows), ("participant", subject_rows)):
        identities = [str(row["participant_id"]) for row in rows]
        if set(identities) != oof_roster or any(count != 1 for count in Counter(identities).values()):
            raise AcceptanceFailure(f"{level} OOF held-out roster is not exact-once")
        if len({str(row["file_id"]) for row in rows}) != len(rows):
            raise AcceptanceFailure(f"{level} OOF IDs are not unique")
        for row in rows:
            participant = str(row["participant_id"])
            if (
                row["level"] != level
                or row["label"] != oof_labels[participant]
                or row["repeat"] != 0
                or row["fold"] != 0
                or row["seed"] != 42
                or row["config_hash"] != config.sha256
                or row["manifest_hash"] != expected_provenance["manifest_hash"]
                or row["fold_hash"] != expected_provenance["fold_hash"]
                or row["representation_mode"] != FORMAL_EXPERIMENT_REPRESENTATION
            ):
                raise AcceptanceFailure(f"{level} OOF trace identity drift for {participant}")
            probabilities = list(row["probabilities"] or [])
            if row["retained"]:
                if row["class_order"] != [0, 1, 2] or len(probabilities) != 3:
                    raise AcceptanceFailure(f"retained {level} OOF probability schema is incomplete")
                if not all(math.isfinite(float(value)) for value in probabilities):
                    raise AcceptanceFailure(f"retained {level} OOF probabilities are non-finite")
                if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8):
                    raise AcceptanceFailure(f"retained {level} OOF probabilities do not sum to one")
            elif probabilities or row["class_order"] or not row["rejection_reason"]:
                raise AcceptanceFailure(f"dropped {level} OOF trace is not explicit")

    retained_subjects = sum(bool(row["retained"]) for row in subject_rows)
    if retained_subjects <= 0:
        raise AcceptanceFailure("real reduced OOF contains no retained participant")
    if {
        str(row["participant_id"]): bool(row["retained"]) for row in file_rows
    } != {
        str(row["participant_id"]): bool(row["retained"]) for row in subject_rows
    }:
        raise AcceptanceFailure("file and participant retained/drop coverage disagree")
    if (
        metrics.get("n_total") != len(oof_roster)
        or metrics.get("n_retained") != retained_subjects
        or metrics.get("n_dropped") != len(oof_roster) - retained_subjects
        or metrics.get("n_rows") != retained_subjects
        or not math.isclose(
            float(metrics.get("coverage_rate")),
            retained_subjects / len(oof_roster),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise AcceptanceFailure("real reduced metrics coverage/counts disagree with subject OOF")
    confusion = metrics.get("confusion_matrix")
    if (
        not isinstance(confusion, list)
        or len(confusion) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in confusion)
        or sum(int(value) for row in confusion for value in row) != retained_subjects
    ):
        raise AcceptanceFailure("real reduced confusion matrix is incomplete")
    if confusion_document.get("cells") != [
        {
            "repeat_index": 0,
            "fold_index": 0,
            "class_order": [0, 1, 2],
            "confusion_matrix": confusion,
        }
    ]:
        raise AcceptanceFailure("confusion artifact differs from evaluated subject OOF")

    empty_contracts = {
        "oof_window_predictions.parquet": "feature_vector_predictions_begin_at_file_level",
        "oof_member_predictions.parquet": "not_an_ensemble_model",
    }
    for name, reason in empty_contracts.items():
        table = parquet.read_table(directory / name)
        metadata = table.schema.metadata or {}
        if (
            table.num_rows != 0
            or table.column_names != ["record_id", "empty_reason"]
            or metadata.get(b"scientific_empty_reason", b"").decode("utf-8") != reason
        ):
            raise AcceptanceFailure(f"scientific-empty OOF contract drift: {name}")

    return {
        "path": directory.relative_to(root).as_posix(),
        "artifact_sha256": {
            name: sha256_file(directory / name) for name in sorted(REAL_REDUCED_ARTIFACTS)
        },
        "train_participants": len(train_roster),
        "oof_participants": len(oof_roster),
        "retained_oof_participants": retained_subjects,
        "formal_runner_supported_representation": FORMAL_EXPERIMENT_REPRESENTATION,
        "comparison_test_only_representations": sorted(NONFORMAL_REPRESENTATIONS),
        "outcome_metric_values_locked": False,
    }


def check_real_reduced_experiment(root: Path, _: Path) -> CheckResult:
    """要求真实 motion r0/f0 成为门禁 / Require the real motion r0/f0 gate."""

    candidates = _real_reduced_candidates(root)
    if not candidates:
        return _pending(
            "real_reduced_feature_vector_experiment",
            "real motion_benchmark r0/f0 experiment has not run",
        )
    evidence = _validate_real_reduced_experiment(candidates[-1], root)
    return _result(
        "real_reduced_feature_vector_experiment",
        (
            f"train={evidence['train_participants']} oof={evidence['oof_participants']} "
            f"retained={evidence['retained_oof_participants']} scope=smoke_not_benchmark"
        ),
        **evidence,
    )


CheckCallable = Callable[[Path, Path], CheckResult]

CHECKS: tuple[tuple[str, CheckCallable], ...] = (
    ('real_fold_cli_smoke', check_real_fold_cli_smoke),
    ('real_reduced_feature_vector_experiment', check_real_reduced_experiment),
    ("target_package_boundary", check_target_package),
    ("spec_lock", check_spec_lock),
    ("manifest_frozen_5x5", check_manifest_and_folds),
    ("formal_config_preflights", check_formal_config_preflights),
    ("module_registry_public_facades", check_module_registry_and_facades),
    ("typed_containers", check_typed_containers),
    ("python_ast_bilingual", check_python_ast_bilingual),
    ("no_legacy_or_unfinished", check_no_legacy_imports_or_unfinished),
    ("strict_json_tree", check_strict_json_tree),
    ("model_cards", check_model_cards),
    ("required_test_sources", check_required_test_sources),
    ("current_test_report", check_current_test_report),
    ("current_source_snapshot", check_current_source_snapshot),
    ("quantitative_artifacts", check_quantitative_artifacts),
    ("no_fabricated_scientific_results", check_no_fabricated_scientific_results),
    ("comparison_documents", check_comparison_documents),
)


def run_acceptance(
    *,
    root: Path = PIPELINE_ROOT,
    repository: Path = REPOSITORY_ROOT,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """运行全部检查并保留每个失败 / Run all checks without hiding later failures."""

    rows: list[CheckResult] = []
    for check_id, function in CHECKS:
        try:
            row = function(root, repository)
            if row.check_id != check_id:
                raise AcceptanceFailure(f"check returned wrong id: {row.check_id}")
            rows.append(row)
        except Exception as error:  # 每项独立记录 / Record every check independently.
            rows.append(
                CheckResult(
                    check_id=check_id,
                    status="failed",
                    detail=f"{type(error).__name__}: {error}",
                    evidence={},
                )
            )
    counts = Counter(row.status for row in rows)
    if counts["failed"]:
        status = "failed"
    elif counts["pending"]:
        status = "passed_with_pending" if allow_pending else "pending"
    else:
        status = "passed"
    return {
        "schema_version": "ppg_frailty.strict_acceptance.v1",
        "status": status,
        "mode": "allow_pending" if allow_pending else "strict",
        "pipeline_root": root.name,
        "spec_sha256": SPEC_SHA256,
        "counts": {
            "checks": len(rows),
            "passed": counts["passed"],
            "pending": counts["pending"],
            "failed": counts["failed"],
        },
        "checks": [asdict(row) for row in rows],
        "scientific_claim": (
            "acceptance_plus_real_reduced_smoke_and_synthetic_contracts_only_"
            "no_frailty_benchmark_or_external_ptt_performance_claim"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI：输出/可选保存 strict JSON / Emit and optionally save strict JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", type=Path)
    parser.add_argument("--allow-pending", action="store_true")
    arguments = parser.parse_args(argv)
    payload = run_acceptance(allow_pending=arguments.allow_pending)
    if arguments.write_report is not None:
        target = arguments.write_report.resolve()
        # 中文：本工具只允许写入 V1；English: keep every acceptance write in V1.
        target.relative_to(PIPELINE_ROOT.resolve())
        atomic_write_json(target, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    accepted = payload["status"] in ({"passed", "passed_with_pending"} if arguments.allow_pending else {"passed"})
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

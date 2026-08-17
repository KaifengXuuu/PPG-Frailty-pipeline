"""哈希、原子写入与 fold-local 拟合证明 / Provenance and leakage guards."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy
import scipy
import sklearn


def sha256_file(path: str | Path) -> str:
    """逐字节计算 SHA-256 / Compute byte-exact SHA-256."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_payload_sha256(value: Any) -> str:
    """哈希规范 JSON payload / Hash a canonical JSON payload."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: str | Path, value: Any, *, root: str | Path) -> None:
    """仅在授权 root 内原子写 strict JSON / Atomically write inside a root."""

    root_path = Path(root).resolve()
    target = Path(path).resolve(strict=False)
    target.relative_to(root_path)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(target)


def assert_training_only(
    fitted_on_participant_ids: Iterable[str],
    outer_train_participant_ids: Iterable[str],
    outer_oof_participant_ids: Iterable[str],
) -> tuple[str, ...]:
    """证明拟合集仅为 outer-train 子集 / Prove fit membership is train-only."""

    fitted = tuple(sorted(set(fitted_on_participant_ids)))
    training = set(outer_train_participant_ids)
    held_out = set(outer_oof_participant_ids)
    if training & held_out:
        raise ValueError("outer train and OOF participant sets overlap")
    if not set(fitted) <= training:
        raise ValueError("fitted object contains a non-training participant")
    if set(fitted) & held_out:
        raise ValueError("held-out participant contaminated a fitted object")
    return fitted


@dataclass(frozen=True)
class FittedArtifactProvenance:
    """任一 fitted object 的身份 / Identity of any fold-fitted object."""

    artifact_id: str
    artifact_type: str
    repeat_index: int
    fold_index: int
    split_seed: int
    fitted_on_participant_ids: tuple[str, ...]
    manifest_sha256: str
    split_file_sha256: str
    config_sha256: str
    producer_sha256: str
    parameters_sha256: str


def runtime_environment() -> dict[str, Any]:
    """记录可复现实验环境 / Capture the reproducible runtime environment."""

    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import torch

        environment.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    except ImportError:
        environment["torch"] = None
    return environment


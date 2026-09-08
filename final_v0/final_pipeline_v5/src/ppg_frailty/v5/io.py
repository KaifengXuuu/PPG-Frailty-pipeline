"""Small shared file helpers for the V5 application layer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write one JSON object atomically and return its resolved path."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{time.time_ns()}")
    try:
        temporary.write_text(
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def resolve_path(
    value: str | Path,
    *,
    base: str | Path,
    within: str | Path | None = None,
    must_exist: bool = False,
    label: str = "path",
) -> Path:
    """Resolve a relative path once, optionally confining it to one tree."""

    raw = Path(value)
    root = Path(base).resolve()
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if within is not None:
        try:
            path.relative_to(Path(within).resolve())
        except ValueError as error:
            raise ValueError(f"{label} must remain inside {Path(within).resolve()}") from error
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


__all__ = ["atomic_json", "file_sha256", "payload_sha256", "resolve_path"]

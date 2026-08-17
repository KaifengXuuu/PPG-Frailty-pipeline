"""M3 版本化 profile 注册表加载器 / M3 versioned profile-registry loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = PACKAGE_ROOT / "registries" / "preprocessing_profiles_v1.json"


def load_registry(path: Path | None = None) -> dict[str, Any]:
    """读取严格 JSON 注册表 / Load the strict JSON profile registry."""

    source = path or DEFAULT_REGISTRY
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("registry_id") != "m3_preprocessing_profiles_corrected_v1":
        raise ValueError("unexpected M3 registry_id")
    if data.get("future_primary_sampling_rate_hz") != 400:
        raise ValueError("future primary sampling rate must be 400 Hz")
    return data


def registry_sha256(path: Path | None = None) -> str:
    """返回注册表文件哈希 / Return the registry file digest."""

    source = path or DEFAULT_REGISTRY
    return hashlib.sha256(source.read_bytes()).hexdigest()


def get_profile(profile_id: str, path: Path | None = None) -> dict[str, Any]:
    """按 ID 获取唯一 profile / Get one profile by exact ID."""

    registry = load_registry(path)
    matches = [
        profile for profile in registry["profiles"] if profile["profile_id"] == profile_id
    ]
    if len(matches) != 1:
        raise KeyError(f"profile_id must resolve exactly once: {profile_id}")
    return matches[0]


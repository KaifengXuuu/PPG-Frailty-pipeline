"""内容寻址、来源绑定的安全缓存 / Provenance-bound content-addressed cache.

中文：cache key 覆盖源文件、配置、schema、producer 和 fold 文件 hash。缓存
payload 另有字节 hash；metadata 或 payload 任一被修改都会 fail closed。这里不
提供 pickle 接口，NumPy 读取固定 allow_pickle=False。

English: Cache keys cover source, configuration, schema, producer, and fold hashes.
Payload bytes have a separate digest and any metadata/payload change fails closed.
No pickle interface is exposed; NumPy loading always disables pickle.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ppg_frailty.provenance import stable_payload_sha256


_SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CacheMissError(FileNotFoundError):
    """请求的完整 provenance identity 尚无缓存 / Exact identity is absent."""


class StaleCacheError(ValueError):
    """缓存 metadata 或 payload 不匹配 / Cache provenance or bytes mismatch."""


@dataclass(frozen=True)
class CacheIdentity:
    """会改变结果的全部 cache identity / Complete result-changing identity."""

    namespace: str
    source_sha256: tuple[str, ...]
    config_sha256: str
    schema_sha256: tuple[str, ...]
    producer_sha256: str
    fold_file_sha256: str | None
    extra: Mapping[str, str] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """校验并输出规范 identity / Validate and serialize identity."""

        if not self.namespace or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in self.namespace
        ):
            raise ValueError("cache namespace contains unsafe characters")
        hashes = (
            tuple(self.source_sha256)
            + (self.config_sha256,)
            + tuple(self.schema_sha256)
            + (self.producer_sha256,)
            + (() if self.fold_file_sha256 is None else (self.fold_file_sha256,))
        )
        if not hashes or any(not _SHA_PATTERN.fullmatch(value) for value in hashes):
            raise ValueError("cache identity requires lowercase SHA-256 values")
        payload = asdict(self)
        payload["source_sha256"] = sorted(set(self.source_sha256))
        payload["schema_sha256"] = sorted(set(self.schema_sha256))
        payload["extra"] = dict(sorted((str(k), str(v)) for k, v in self.extra.items()))
        return payload

    @property
    def key(self) -> str:
        """返回 canonical content key / Return the canonical content key."""

        return stable_payload_sha256(self.to_payload())


class ContentAddressedCache:
    """只接受完整 identity 的 cache store / Cache store requiring full identity."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(strict=False)

    def _paths(self, identity: CacheIdentity) -> tuple[Path, Path]:
        """解析安全 payload/metadata 路径 / Resolve safe cache paths."""

        key = identity.key
        directory = self.root / identity.namespace / key[:2]
        payload = (directory / f"{key}.bin").resolve(strict=False)
        metadata = (directory / f"{key}.json").resolve(strict=False)
        payload.relative_to(self.root)
        metadata.relative_to(self.root)
        return payload, metadata

    def put_bytes(self, identity: CacheIdentity, payload: bytes) -> Path:
        """原子保存 payload 和 provenance / Atomically store bytes and metadata."""

        payload_path, metadata_path = self._paths(identity)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        # 中文：payload_sha256 是原始 bytes 的标准 SHA-256，不混入 JSON 包装。
        # English: payload_sha256 is the standard digest of the raw bytes.
        payload_digest = hashlib.sha256(bytes(payload)).hexdigest()
        metadata = {
            "schema_version": "ppg_frailty.content_cache.v1",
            "cache_key": identity.key,
            "identity": identity.to_payload(),
            "payload_sha256": payload_digest,
            "payload_bytes": len(payload),
        }
        payload_tmp = payload_path.with_suffix(".bin.tmp")
        metadata_tmp = metadata_path.with_suffix(".json.tmp")
        payload_tmp.write_bytes(bytes(payload))
        metadata_tmp.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        payload_tmp.replace(payload_path)
        # 中文：metadata 最后提交；半写入 payload 不会被视为有效 cache。
        # English: Commit metadata last so a partial payload is never considered valid.
        metadata_tmp.replace(metadata_path)
        return payload_path

    def get_bytes(self, identity: CacheIdentity) -> bytes:
        """加载并完整校验 cache / Load and fully validate one cache entry."""

        payload_path, metadata_path = self._paths(identity)
        if not payload_path.is_file() or not metadata_path.is_file():
            raise CacheMissError(identity.key)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StaleCacheError("cache metadata is unreadable") from exc
        if metadata.get("cache_key") != identity.key:
            raise StaleCacheError("cache key mismatch")
        if metadata.get("identity") != identity.to_payload():
            raise StaleCacheError("cache provenance mismatch")
        payload = payload_path.read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if metadata.get("payload_sha256") != observed:
            raise StaleCacheError("cache payload hash mismatch")
        if metadata.get("payload_bytes") != len(payload):
            raise StaleCacheError("cache payload length mismatch")
        return payload

    def put_npz(self, identity: CacheIdentity, arrays: Mapping[str, np.ndarray]) -> Path:
        """安全保存无 pickle NPZ / Store a pickle-free NPZ payload."""

        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            **{str(key): np.asarray(value) for key, value in arrays.items()},
        )
        return self.put_bytes(identity, buffer.getvalue())

    def get_npz(self, identity: CacheIdentity) -> dict[str, np.ndarray]:
        """安全加载 NPZ / Load NPZ with pickle disabled."""

        with np.load(io.BytesIO(self.get_bytes(identity)), allow_pickle=False) as data:
            return {key: np.array(data[key], copy=True) for key in data.files}


__all__ = [
    "CacheIdentity",
    "CacheMissError",
    "ContentAddressedCache",
    "StaleCacheError",
]

"""Recording-local, label-free preprocessing cache.

Entries are content addressed by source recordings and ordered preprocessing
modules. Arrays use NPY with pickle disabled and load as read-only memory maps.
A per-key file lock makes get_or_compute safe across workers. Supervision labels
and fold-fitted objects have no serialization path here.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from ppg_frailty.provenance import stable_payload_sha256

from .cache import CacheMissError, StaleCacheError


RECORDING_CACHE_SCHEMA_VERSION = "ppg_frailty.recording_preprocess_cache.v1"
RECORDING_CACHE_COMMIT_VERSION = "ppg_frailty.recording_preprocess_commit.v1"
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METADATA_FILENAME, _COMMIT_FILENAME, _ARRAY_DIRECTORY = "metadata.json", "COMMITTED.json", "arrays"
_METADATA_KEYS = {"schema_version", "cache_key", "identity", "arrays", "attributes"}
_COMMIT_KEYS = {"schema_version", "cache_key", "metadata_filename", "metadata_sha256", "array_count"}
_ARRAY_KEYS = {"filename", "dtype", "shape", "logical_nbytes", "file_bytes", "file_sha256", "content_sha256"}

class RecordingCacheError(StaleCacheError):
    """Base error for a recording-cache failure."""

class RecordingCacheAccessError(RecordingCacheError):
    """The cache filesystem or lock could not be used."""

class RecordingCacheSourceError(RecordingCacheError):
    """A recording no longer matches its source identity."""

class RecordingCacheCorruptionError(RecordingCacheError):
    """A published entry failed validation."""

class ImmutableCacheConflictError(RecordingCacheCorruptionError):
    """An existing identity contains different data."""

def _require_pattern(value: str, pattern: re.Pattern[str], field_name: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        description = "a lowercase SHA-256 digest" if pattern is _SHA256 else "free of unsafe characters"
        raise ValueError(f"{field_name} must be {description}")
    return value

def _normalise_json(value: Any, *, field_name: str) -> Any:
    """Return a deterministic strict-JSON value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{field_name} requires string mapping keys")
        return {key: _normalise_json(value[key], field_name=f"{field_name}.{key}") for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{field_name} contains non-JSON value of type {type(value).__name__}")

def _normalise_object(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a JSON object")
    return _normalise_json(value, field_name=field_name)

def _strict_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result

def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant {token}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RecordingCacheCorruptionError(f"cache JSON is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise RecordingCacheCorruptionError(f"cache JSON root is not an object: {path.name}")
    return value

def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise RecordingCacheCorruptionError(
            f"{context} schema mismatch; missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}"
        )

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _file_identity(path: Path) -> tuple[str, int, int, int, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

@lru_cache(maxsize=32768)
def _file_digest(path: str, device: int, inode: int, size: int, mtime: int, ctime: int) -> str:
    del device, inode, size, mtime, ctime
    return _sha256_file(Path(path))

@lru_cache(maxsize=32768)
def _array_digest(path: str, device: int, inode: int, size: int, mtime: int, ctime: int, name: str) -> str:
    del device, inode, size, mtime, ctime
    return _prepare_array(name, np.load(Path(path), allow_pickle=False, mmap_mode="r")).content_sha256

@dataclass(frozen=True)
class NamedSourceDependency:
    name: str
    sha256: str
    properties: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": _require_pattern(self.name, _SAFE_TOKEN, "source dependency name"),
            "sha256": _require_pattern(self.sha256, _SHA256, f"source dependency {self.name!r} sha256"),
            "properties": _normalise_object(self.properties, field_name=f"source dependency {self.name!r} properties"),
        }

@dataclass(frozen=True)
class OrderedModuleSpec:
    module_id: str
    module_version: str
    implementation_sha256: str
    enabled: bool
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self, *, position: int) -> dict[str, Any]:
        if not isinstance(self.enabled, bool):
            raise TypeError("module enabled must be bool")
        return {
            "position": int(position),
            "module_id": _require_pattern(self.module_id, _SAFE_TOKEN, "module_id"),
            "module_version": _require_pattern(self.module_version, _SAFE_TOKEN, f"module {self.module_id!r} version"),
            "implementation_sha256": _require_pattern(
                self.implementation_sha256, _SHA256, f"module {self.module_id!r} implementation_sha256"
            ),
            "enabled": self.enabled,
            "parameters": _normalise_object(self.parameters, field_name=f"module {self.module_id!r} parameters"),
        }

@dataclass(frozen=True)
class RecordingCacheIdentity:
    namespace: str
    layer: str
    recording_id: str
    source_dependencies: tuple[NamedSourceDependency, ...]
    module_chain: tuple[OrderedModuleSpec, ...]
    producer_sha256: str
    output_schema: Mapping[str, Any]
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        if not isinstance(self.recording_id, str) or not self.recording_id.strip() or len(self.recording_id) > 1024:
            raise ValueError("recording_id must be a non-empty reasonable-length string")
        if not self.source_dependencies:
            raise ValueError("recording cache requires at least one source dependency")
        if any(not isinstance(item, NamedSourceDependency) for item in self.source_dependencies):
            raise TypeError("source_dependencies must contain NamedSourceDependency")
        if any(not isinstance(item, OrderedModuleSpec) for item in self.module_chain):
            raise TypeError("module_chain must contain OrderedModuleSpec")
        dependencies = sorted((item.to_payload() for item in self.source_dependencies), key=lambda item: item["name"])
        if len({item["name"] for item in dependencies}) != len(dependencies):
            raise ValueError("source dependency names must be unique")
        return {
            "identity_schema_version": RECORDING_CACHE_SCHEMA_VERSION,
            "namespace": _require_pattern(self.namespace, _SAFE_TOKEN, "cache namespace"),
            "layer": _require_pattern(self.layer, _SAFE_TOKEN, "cache layer"),
            "recording_id": self.recording_id,
            "source_dependencies": dependencies,
            "module_chain": [module.to_payload(position=i) for i, module in enumerate(self.module_chain)],
            "producer_sha256": _require_pattern(self.producer_sha256, _SHA256, "producer_sha256"),
            "output_schema": _normalise_object(self.output_schema, field_name="output_schema"),
            "extra": _normalise_object(self.extra, field_name="identity extra"),
        }

    @property
    def key(self) -> str:
        return stable_payload_sha256(self.to_payload())

@dataclass(frozen=True)
class RecordingCacheBuild:
    arrays: Mapping[str, np.ndarray]
    attributes: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RecordingCacheEntry:
    key: str
    path: Path
    arrays: Mapping[str, np.ndarray]
    attributes: Mapping[str, Any]
    metadata: Mapping[str, Any]

@dataclass(frozen=True)
class RecordingCacheResult:
    disposition: str
    entry: RecordingCacheEntry

@dataclass(frozen=True)
class _PreparedArray:
    value: np.ndarray
    dtype: str
    shape: tuple[int, ...]
    logical_nbytes: int
    content_sha256: str

def _prepare_array(name: str, value: np.ndarray) -> _PreparedArray:
    _require_pattern(name, _SAFE_TOKEN, "array name")
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"cache array {name!r} has an object dtype")
    if array.dtype.fields is not None:
        raise TypeError(f"cache array {name!r} has a structured dtype")
    canonical = np.asarray(array, order="C")
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(canonical).cast("B"))
    return _PreparedArray(
        canonical, canonical.dtype.str, tuple(map(int, canonical.shape)), int(canonical.nbytes), digest.hexdigest()
    )

def _prepare_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, _PreparedArray]:
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("recording cache requires at least one named array")
    if any(not isinstance(name, str) for name in arrays):
        raise TypeError("recording cache array names must be strings")
    return {name: _prepare_array(name, arrays[name]) for name in sorted(arrays)}

@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    descriptor: int | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    except OSError as exc:
        raise RecordingCacheAccessError(f"cannot use recording-cache lock: {path}") from exc
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            except OSError as exc:
                raise RecordingCacheAccessError(f"cannot release recording-cache lock: {path}") from exc

def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | (getattr(os, "O_DIRECTORY", 0) if path.is_dir() else 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

class RecordingPreprocessingCache:
    """Content-addressed store for recording-level NumPy arrays."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(strict=False)

    def _beneath_root(self, path: Path, purpose: str) -> Path:
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(self.root)
            return resolved
        except (OSError, RuntimeError, ValueError) as exc:
            raise RecordingCacheAccessError(f"unsafe recording-cache {purpose} path") from exc

    def _entry_path(self, identity: RecordingCacheIdentity) -> Path:
        return self._beneath_root(
            self.root / "v1" / identity.namespace / identity.layer / identity.key[:2] / identity.key, "entry"
        )

    def _lock_path(self, identity: RecordingCacheIdentity) -> Path:
        return self._beneath_root(
            self.root / "locks" / identity.namespace / identity.layer / f"{identity.key}.lock", "lock"
        )

    def _staging_path(self, identity: RecordingCacheIdentity) -> Path:
        name = f"{identity.namespace}.{identity.layer}.{identity.key}.{os.getpid()}.{uuid.uuid4().hex}"
        return self._beneath_root(self.root / "staging" / name, "staging")

    def load(self, identity: RecordingCacheIdentity) -> RecordingCacheEntry:
        try:
            return self._load_verified(identity)
        except (CacheMissError, RecordingCacheError):
            raise
        except OSError as exc:
            raise RecordingCacheAccessError(f"cannot read recording-cache entry: {identity.key}") from exc

    def _load_verified(self, identity: RecordingCacheIdentity) -> RecordingCacheEntry:
        entry = self._entry_path(identity)
        if not entry.exists():
            raise CacheMissError(identity.key)
        metadata_path, commit_path, arrays_dir = (
            entry / _METADATA_FILENAME,
            entry / _COMMIT_FILENAME,
            entry / _ARRAY_DIRECTORY,
        )
        if (
            not entry.is_dir()
            or entry.is_symlink()
            or not metadata_path.is_file()
            or metadata_path.is_symlink()
            or not commit_path.is_file()
            or commit_path.is_symlink()
            or not arrays_dir.is_dir()
            or arrays_dir.is_symlink()
        ):
            raise RecordingCacheCorruptionError("cache entry is incomplete or contains unsafe links")

        metadata_bytes = metadata_path.read_bytes()
        metadata, commit = _load_json(metadata_path), _load_json(commit_path)
        _exact_keys(metadata, _METADATA_KEYS, "recording cache metadata")
        _exact_keys(commit, _COMMIT_KEYS, "recording cache commit marker")
        checks = (
            (metadata["schema_version"] == RECORDING_CACHE_SCHEMA_VERSION, "recording cache schema mismatch"),
            (commit["schema_version"] == RECORDING_CACHE_COMMIT_VERSION, "recording cache commit schema mismatch"),
            (metadata["cache_key"] == identity.key == commit["cache_key"], "recording cache key mismatch"),
            (metadata["identity"] == identity.to_payload(), "recording cache identity mismatch"),
            (commit["metadata_filename"] == _METADATA_FILENAME, "recording cache metadata filename mismatch"),
            (
                commit["metadata_sha256"] == hashlib.sha256(metadata_bytes).hexdigest(),
                "recording cache metadata hash mismatch",
            ),
        )
        for valid, message in checks:
            if not valid:
                raise RecordingCacheCorruptionError(message)

        specifications = metadata["arrays"]
        if not isinstance(specifications, dict) or not specifications:
            raise RecordingCacheCorruptionError("recording cache arrays must be non-empty")
        if type(commit["array_count"]) is not int or commit["array_count"] != len(specifications):
            raise RecordingCacheCorruptionError("recording cache array count mismatch")
        try:
            attributes = _normalise_object(metadata["attributes"], field_name="cache attributes")
        except (TypeError, ValueError) as exc:
            raise RecordingCacheCorruptionError("recording cache attributes violate strict JSON") from exc

        expected_paths = {Path(_METADATA_FILENAME), Path(_COMMIT_FILENAME), Path(_ARRAY_DIRECTORY)}
        arrays: dict[str, np.ndarray] = {}
        for name, specification in sorted(specifications.items()):
            if not isinstance(specification, dict):
                raise RecordingCacheCorruptionError("cached array spec is not an object")
            _exact_keys(specification, _ARRAY_KEYS, f"cached array {name!r}")
            try:
                _require_pattern(name, _SAFE_TOKEN, "cached array name")
            except ValueError as exc:
                raise RecordingCacheCorruptionError("unsafe cached array name") from exc
            relative = Path(_ARRAY_DIRECTORY) / f"{name}.npy"
            if specification["filename"] != relative.as_posix():
                raise RecordingCacheCorruptionError("cached array filename mismatch")
            expected_paths.add(relative)
            path = entry / relative
            if not path.is_file() or path.is_symlink():
                raise RecordingCacheCorruptionError("cached array file is missing or unsafe")
            if type(specification["file_bytes"]) is not int or specification["file_bytes"] != path.stat().st_size:
                raise RecordingCacheCorruptionError("cached array file length mismatch")
            file_identity = _file_identity(path)
            if not isinstance(specification["file_sha256"], str) or specification["file_sha256"] != _file_digest(
                *file_identity
            ):
                raise RecordingCacheCorruptionError("cached array file hash mismatch")
            try:
                array = np.load(path, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError, TypeError) as exc:
                raise RecordingCacheCorruptionError("cached array cannot be loaded without pickle") from exc
            shape = specification["shape"]
            if (
                not isinstance(shape, list)
                or any(type(x) is not int or x < 0 for x in shape)
                or list(array.shape) != shape
            ):
                raise RecordingCacheCorruptionError("cached array shape mismatch")
            if specification["dtype"] != array.dtype.str:
                raise RecordingCacheCorruptionError("cached array dtype mismatch")
            if type(specification["logical_nbytes"]) is not int or specification["logical_nbytes"] != int(array.nbytes):
                raise RecordingCacheCorruptionError("cached array logical size mismatch")
            if not isinstance(specification["content_sha256"], str) or specification["content_sha256"] != _array_digest(
                *file_identity, name
            ):
                raise RecordingCacheCorruptionError("cached array content hash mismatch")
            array.flags.writeable = False
            arrays[name] = array

        if {path.relative_to(entry) for path in entry.rglob("*")} != expected_paths:
            raise RecordingCacheCorruptionError("cache entry contains unexpected or missing filesystem objects")
        return RecordingCacheEntry(
            identity.key,
            entry,
            MappingProxyType(arrays),
            MappingProxyType(attributes),
            MappingProxyType(metadata),
        )

    def put_arrays(
        self,
        identity: RecordingCacheIdentity,
        arrays: Mapping[str, np.ndarray],
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> RecordingCacheResult:
        prepared = _prepare_arrays(arrays)
        strict_attributes = _normalise_object(attributes or {}, field_name="cache attributes")
        with _exclusive_file_lock(self._lock_path(identity)):
            entry_path = self._entry_path(identity)
            if entry_path.exists():
                entry = self.load(identity)
                self._assert_same(entry, prepared, strict_attributes)
                return RecordingCacheResult("existing", entry)
            self._publish(identity, prepared, strict_attributes)
            return RecordingCacheResult("written", self.load(identity))

    def get_or_compute(
        self, identity: RecordingCacheIdentity, builder: Callable[[], RecordingCacheBuild]
    ) -> RecordingCacheResult:
        try:
            return RecordingCacheResult("hit", self.load(identity))
        except CacheMissError:
            pass
        with _exclusive_file_lock(self._lock_path(identity)):
            if self._entry_path(identity).exists():
                return RecordingCacheResult("hit", self.load(identity))
            built = builder()
            if not isinstance(built, RecordingCacheBuild):
                raise TypeError("recording cache builder must return RecordingCacheBuild")
            self._publish(
                identity,
                _prepare_arrays(built.arrays),
                _normalise_object(built.attributes, field_name="cache attributes"),
            )
            return RecordingCacheResult("written", self.load(identity))

    @staticmethod
    def _assert_same(
        entry: RecordingCacheEntry, prepared: Mapping[str, _PreparedArray], attributes: Mapping[str, Any]
    ) -> None:
        if dict(entry.attributes) != dict(attributes):
            raise ImmutableCacheConflictError("cache identity already exists with different attributes")
        if set(entry.arrays) != set(prepared):
            raise ImmutableCacheConflictError("cache identity already exists with different array names")
        for name, candidate in prepared.items():
            existing = entry.metadata["arrays"][name]
            if (
                existing["dtype"],
                tuple(existing["shape"]),
                existing["logical_nbytes"],
                existing["content_sha256"],
            ) != (candidate.dtype, candidate.shape, candidate.logical_nbytes, candidate.content_sha256):
                raise ImmutableCacheConflictError(f"cache identity already exists with different array {name!r}")

    def _publish(
        self, identity: RecordingCacheIdentity, arrays: Mapping[str, _PreparedArray], attributes: Mapping[str, Any]
    ) -> None:
        entry, staging = self._entry_path(identity), self._staging_path(identity)
        try:
            if entry.exists():
                raise ImmutableCacheConflictError("cache entry appeared during publication")
            entry.parent.mkdir(parents=True, exist_ok=True)
            staging.parent.mkdir(parents=True, exist_ok=True)
            array_dir = staging / _ARRAY_DIRECTORY
            array_dir.mkdir(parents=True)
            manifest: dict[str, dict[str, Any]] = {}
            for name, prepared in arrays.items():
                path = array_dir / f"{name}.npy"
                with path.open("wb") as handle:
                    np.save(handle, prepared.value, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                manifest[name] = {
                    "filename": f"{_ARRAY_DIRECTORY}/{name}.npy",
                    "dtype": prepared.dtype,
                    "shape": list(prepared.shape),
                    "logical_nbytes": prepared.logical_nbytes,
                    "file_bytes": path.stat().st_size,
                    "file_sha256": _sha256_file(path),
                    "content_sha256": prepared.content_sha256,
                }
            _fsync(array_dir)
            metadata = {
                "schema_version": RECORDING_CACHE_SCHEMA_VERSION,
                "cache_key": identity.key,
                "identity": identity.to_payload(),
                "arrays": manifest,
                "attributes": attributes,
            }
            metadata_bytes = _strict_json_bytes(metadata)
            (staging / _METADATA_FILENAME).write_bytes(metadata_bytes)
            _fsync(staging / _METADATA_FILENAME)
            commit = {
                "schema_version": RECORDING_CACHE_COMMIT_VERSION,
                "cache_key": identity.key,
                "metadata_filename": _METADATA_FILENAME,
                "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "array_count": len(manifest),
            }
            (staging / _COMMIT_FILENAME).write_bytes(_strict_json_bytes(commit))
            _fsync(staging / _COMMIT_FILENAME)
            _fsync(staging)
            staging.rename(entry)
            _fsync(entry.parent)
        except BaseException as exc:
            try:
                if staging.exists():
                    shutil.rmtree(staging)
            except OSError as cleanup_exc:
                raise RecordingCacheAccessError(
                    f"cannot clean failed recording-cache staging path: {staging}"
                ) from cleanup_exc
            if isinstance(exc, (RecordingCacheError, TypeError, ValueError)):
                raise
            if isinstance(exc, OSError):
                raise RecordingCacheAccessError(f"cannot publish recording-cache entry: {identity.key}") from exc
            raise


__all__ = [
    "ImmutableCacheConflictError",
    "NamedSourceDependency",
    "OrderedModuleSpec",
    "RECORDING_CACHE_COMMIT_VERSION",
    "RECORDING_CACHE_SCHEMA_VERSION",
    "RecordingCacheAccessError",
    "RecordingCacheBuild",
    "RecordingCacheCorruptionError",
    "RecordingCacheEntry",
    "RecordingCacheError",
    "RecordingCacheIdentity",
    "RecordingCacheResult",
    "RecordingCacheSourceError",
    "RecordingPreprocessingCache",
]

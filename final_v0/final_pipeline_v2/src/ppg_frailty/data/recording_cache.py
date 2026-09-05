"""Immutable recording-level preprocessing cache.

The cache in this module is deliberately narrower than a generic object cache:
it stores only named NumPy arrays produced by a fully materialised, ordered
preprocessing chain.  Fold-fitted objects, labels, and Python pickles are not
part of this API.

An entry becomes visible only after every array and the strict metadata document
have been written, hashed, fsynced, and a commit marker has been created.  The
staging directory is then atomically renamed into its content-addressed path.
Existing entries are immutable.  A producer that attempts to publish different
bytes under an existing identity receives an explicit conflict instead of
silently replacing evidence used by an earlier study.
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
_METADATA_FILENAME = "metadata.json"
_COMMIT_FILENAME = "COMMITTED.json"
_ARRAY_DIRECTORY = "arrays"


class RecordingCacheError(StaleCacheError):
    """Base class for cache failures that must abort the scientific cell."""


class RecordingCacheAccessError(RecordingCacheError):
    """Cache filesystem/locking infrastructure could not be used safely."""


class RecordingCacheSourceError(RecordingCacheError):
    """A recording source no longer matches its frozen manifest identity."""


class RecordingCacheCorruptionError(RecordingCacheError):
    """A published recording cache entry failed strict validation."""


class ImmutableCacheConflictError(RecordingCacheCorruptionError):
    """An existing identity was produced with different arrays or attributes."""


def _require_sha256(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_token(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _SAFE_TOKEN.fullmatch(value) is None:
        raise ValueError(f"{field_name} contains unsafe characters")
    return value


def _normalise_json(value: Any, *, field_name: str) -> Any:
    """Return a strict JSON value with deterministic container semantics."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalised: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} requires string mapping keys")
            if key in normalised:
                raise ValueError(f"{field_name} contains duplicate key {key!r}")
            normalised[key] = _normalise_json(
                item,
                field_name=f"{field_name}.{key}",
            )
        return dict(sorted(normalised.items()))
    if isinstance(value, (list, tuple)):
        return [
            _normalise_json(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{field_name} contains non-JSON value of type {type(value).__name__}"
    )


def _normalise_json_object(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a JSON object")
    normalised = _normalise_json(value, field_name=field_name)
    if not isinstance(normalised, dict):  # defensive: Mapping always maps to dict above
        raise TypeError(f"{field_name} must be a JSON object")
    return normalised


def _strict_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RecordingCacheCorruptionError(
            f"cache JSON is unreadable: {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise RecordingCacheCorruptionError(
            f"cache JSON root is not an object: {path.name}"
        )
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RecordingCacheCorruptionError(
            f"{context} schema mismatch; missing={missing}, extra={extra}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[str, int, int, int, int, int]:
    stat = path.stat()
    return (
        str(path.resolve()),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


@lru_cache(maxsize=32768)
def _verified_file_sha256_once(
    path_text: str,
    device: int,
    inode: int,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    """Verify large immutable arrays once per stable process/file identity."""

    del device, inode, size, mtime_ns, ctime_ns
    return _sha256_file(Path(path_text))


@lru_cache(maxsize=32768)
def _verified_array_content_once(
    path_text: str,
    device: int,
    inode: int,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
    name: str,
) -> str:
    del device, inode, size, mtime_ns, ctime_ns
    array = np.load(Path(path_text), allow_pickle=False, mmap_mode="r")
    return _prepare_array(name, array).content_sha256


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cache_path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        raise RecordingCacheAccessError(
            f"cannot inspect recording-cache path: {path}"
        ) from exc


@dataclass(frozen=True)
class NamedSourceDependency:
    """One named byte source used to produce a recording cache entry."""

    name: str
    sha256: str
    properties: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": _require_token(self.name, field_name="source dependency name"),
            "sha256": _require_sha256(
                self.sha256,
                field_name=f"source dependency {self.name!r} sha256",
            ),
            "properties": _normalise_json_object(
                self.properties,
                field_name=f"source dependency {self.name!r} properties",
            ),
        }


@dataclass(frozen=True)
class OrderedModuleSpec:
    """A versioned preprocessing module at one exact chain position."""

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
            "module_id": _require_token(self.module_id, field_name="module_id"),
            "module_version": _require_token(
                self.module_version,
                field_name=f"module {self.module_id!r} version",
            ),
            "implementation_sha256": _require_sha256(
                self.implementation_sha256,
                field_name=f"module {self.module_id!r} implementation_sha256",
            ),
            "enabled": self.enabled,
            "parameters": _normalise_json_object(
                self.parameters,
                field_name=f"module {self.module_id!r} parameters",
            ),
        }


@dataclass(frozen=True)
class RecordingCacheIdentity:
    """Complete identity for one recording-level preprocessing payload.

    Source dependencies are named and therefore canonicalised by name.  The
    module chain is intentionally *not* sorted: position is part of the payload,
    so swapping otherwise identical modules always changes the cache key.
    """

    namespace: str
    layer: str
    recording_id: str
    source_dependencies: tuple[NamedSourceDependency, ...]
    module_chain: tuple[OrderedModuleSpec, ...]
    producer_sha256: str
    output_schema: Mapping[str, Any]
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        namespace = _require_token(self.namespace, field_name="cache namespace")
        layer = _require_token(self.layer, field_name="cache layer")
        if not isinstance(self.recording_id, str) or not self.recording_id.strip():
            raise ValueError("recording_id must be a non-empty string")
        if len(self.recording_id) > 1024:
            raise ValueError("recording_id is unreasonably long")

        if not self.source_dependencies:
            raise ValueError("recording cache requires at least one source dependency")
        if any(
            not isinstance(item, NamedSourceDependency)
            for item in self.source_dependencies
        ):
            raise TypeError("source_dependencies must contain NamedSourceDependency")
        if any(not isinstance(item, OrderedModuleSpec) for item in self.module_chain):
            raise TypeError("module_chain must contain OrderedModuleSpec")
        dependencies = [item.to_payload() for item in self.source_dependencies]
        names = [item["name"] for item in dependencies]
        if len(names) != len(set(names)):
            raise ValueError("source dependency names must be unique")
        dependencies.sort(key=lambda item: item["name"])

        modules = [
            module.to_payload(position=position)
            for position, module in enumerate(self.module_chain)
        ]
        return {
            "identity_schema_version": RECORDING_CACHE_SCHEMA_VERSION,
            "namespace": namespace,
            "layer": layer,
            "recording_id": self.recording_id,
            "source_dependencies": dependencies,
            "module_chain": modules,
            "producer_sha256": _require_sha256(
                self.producer_sha256,
                field_name="producer_sha256",
            ),
            "output_schema": _normalise_json_object(
                self.output_schema,
                field_name="output_schema",
            ),
            "extra": _normalise_json_object(
                self.extra,
                field_name="identity extra",
            ),
        }

    @property
    def key(self) -> str:
        return stable_payload_sha256(self.to_payload())


@dataclass(frozen=True)
class RecordingCacheBuild:
    """Arrays and audit attributes returned by a cache builder."""

    arrays: Mapping[str, np.ndarray]
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordingCacheEntry:
    """A fully verified immutable cache entry backed by read-only memmaps."""

    key: str
    path: Path
    arrays: Mapping[str, np.ndarray]
    attributes: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RecordingCacheResult:
    """Result of a cache write or get-or-compute operation."""

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
    _require_token(name, field_name="array name")
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"cache array {name!r} has an object dtype")
    if array.dtype.fields is not None:
        raise TypeError(f"cache array {name!r} has a structured dtype")
    # np.ascontiguousarray promotes a scalar from shape () to (1,).  np.asarray
    # with order="C" preserves rank while still copying non-contiguous inputs.
    canonical = np.asarray(array, order="C")
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(memoryview(canonical).cast("B"))
    return _PreparedArray(
        value=canonical,
        dtype=canonical.dtype.str,
        shape=tuple(int(item) for item in canonical.shape),
        logical_nbytes=int(canonical.nbytes),
        content_sha256=digest.hexdigest(),
    )


def _prepare_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, _PreparedArray]:
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("recording cache requires at least one named array")
    prepared: dict[str, _PreparedArray] = {}
    for raw_name, value in arrays.items():
        if not isinstance(raw_name, str):
            raise TypeError("recording cache array names must be strings")
        if raw_name in prepared:
            raise ValueError(f"duplicate recording cache array {raw_name!r}")
        prepared[raw_name] = _prepare_array(raw_name, value)
    return dict(sorted(prepared.items()))


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    descriptor: int | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise RecordingCacheAccessError(
            f"cannot acquire recording-cache lock: {path}"
        ) from exc
    assert descriptor is not None
    try:
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        except OSError as exc:
            raise RecordingCacheAccessError(
                f"cannot release recording-cache lock: {path}"
            ) from exc


class RecordingPreprocessingCache:
    """Content-addressed store for recording-level NumPy preprocessing arrays."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(strict=False)

    def _resolve_beneath_root(self, path: Path, *, purpose: str) -> Path:
        """Resolve a cache path and fail closed if links escape the store root."""

        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(self.root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RecordingCacheAccessError(
                f"unsafe recording-cache {purpose} path"
            ) from exc
        return resolved

    def _entry_path(self, identity: RecordingCacheIdentity) -> Path:
        path = self._resolve_beneath_root(
            self.root
            / "v1"
            / identity.namespace
            / identity.layer
            / identity.key[:2]
            / identity.key,
            purpose="entry",
        )
        return path

    def _lock_path(self, identity: RecordingCacheIdentity) -> Path:
        path = self._resolve_beneath_root(
            self.root
            / "locks"
            / identity.namespace
            / identity.layer
            / f"{identity.key}.lock",
            purpose="lock",
        )
        return path

    def _new_staging_path(self, identity: RecordingCacheIdentity) -> Path:
        name = (
            f"{identity.namespace}.{identity.layer}.{identity.key}."
            f"{os.getpid()}.{uuid.uuid4().hex}"
        )
        path = self._resolve_beneath_root(
            self.root / "staging" / name,
            purpose="staging",
        )
        return path

    def load(self, identity: RecordingCacheIdentity) -> RecordingCacheEntry:
        """Load one entry and normalize infrastructure errors as cache-fatal."""

        try:
            return self._load_verified(identity)
        except (CacheMissError, RecordingCacheError):
            raise
        except OSError as exc:
            raise RecordingCacheAccessError(
                f"cannot read recording-cache entry: {identity.key}"
            ) from exc

    def _load_verified(self, identity: RecordingCacheIdentity) -> RecordingCacheEntry:
        """Load and validate one entry; all arrays are returned as read-only mmap."""

        entry_path = self._entry_path(identity)
        if not entry_path.exists():
            raise CacheMissError(identity.key)
        if not entry_path.is_dir() or entry_path.is_symlink():
            raise RecordingCacheCorruptionError("cache entry is not a real directory")

        metadata_path = entry_path / _METADATA_FILENAME
        commit_path = entry_path / _COMMIT_FILENAME
        arrays_path = entry_path / _ARRAY_DIRECTORY
        if (
            not metadata_path.is_file()
            or metadata_path.is_symlink()
            or not commit_path.is_file()
            or commit_path.is_symlink()
            or not arrays_path.is_dir()
            or arrays_path.is_symlink()
        ):
            raise RecordingCacheCorruptionError(
                "cache entry is incomplete or contains unsafe links"
            )

        try:
            metadata_bytes = metadata_path.read_bytes()
        except OSError as exc:
            raise RecordingCacheCorruptionError(
                "cache metadata bytes are unreadable"
            ) from exc
        metadata = _load_strict_json(metadata_path)
        commit = _load_strict_json(commit_path)
        _require_exact_keys(
            metadata,
            {
                "schema_version",
                "cache_key",
                "identity",
                "arrays",
                "attributes",
            },
            context="recording cache metadata",
        )
        _require_exact_keys(
            commit,
            {
                "schema_version",
                "cache_key",
                "metadata_filename",
                "metadata_sha256",
                "array_count",
            },
            context="recording cache commit marker",
        )
        if metadata["schema_version"] != RECORDING_CACHE_SCHEMA_VERSION:
            raise RecordingCacheCorruptionError("recording cache schema mismatch")
        if commit["schema_version"] != RECORDING_CACHE_COMMIT_VERSION:
            raise RecordingCacheCorruptionError("recording cache commit schema mismatch")
        if metadata["cache_key"] != identity.key or commit["cache_key"] != identity.key:
            raise RecordingCacheCorruptionError("recording cache key mismatch")
        if metadata["identity"] != identity.to_payload():
            raise RecordingCacheCorruptionError("recording cache identity mismatch")
        if commit["metadata_filename"] != _METADATA_FILENAME:
            raise RecordingCacheCorruptionError("recording cache metadata filename mismatch")
        if commit["metadata_sha256"] != hashlib.sha256(metadata_bytes).hexdigest():
            raise RecordingCacheCorruptionError("recording cache metadata hash mismatch")

        array_metadata = metadata["arrays"]
        if not isinstance(array_metadata, dict) or not array_metadata:
            raise RecordingCacheCorruptionError("recording cache arrays must be non-empty")
        if type(commit["array_count"]) is not int or commit["array_count"] != len(
            array_metadata
        ):
            raise RecordingCacheCorruptionError("recording cache array count mismatch")
        try:
            attributes = _normalise_json_object(
                metadata["attributes"],
                field_name="cache attributes",
            )
        except (TypeError, ValueError) as exc:
            raise RecordingCacheCorruptionError(
                "recording cache attributes violate strict JSON"
            ) from exc

        expected_paths = {
            Path(_METADATA_FILENAME),
            Path(_COMMIT_FILENAME),
            Path(_ARRAY_DIRECTORY),
        }
        arrays: dict[str, np.ndarray] = {}
        for name, specification in sorted(array_metadata.items()):
            try:
                _require_token(name, field_name="cached array name")
            except ValueError as exc:
                raise RecordingCacheCorruptionError("unsafe cached array name") from exc
            if not isinstance(specification, dict):
                raise RecordingCacheCorruptionError("cached array spec is not an object")
            _require_exact_keys(
                specification,
                {
                    "filename",
                    "dtype",
                    "shape",
                    "logical_nbytes",
                    "file_bytes",
                    "file_sha256",
                    "content_sha256",
                },
                context=f"cached array {name!r}",
            )
            expected_filename = f"{_ARRAY_DIRECTORY}/{name}.npy"
            if specification["filename"] != expected_filename:
                raise RecordingCacheCorruptionError("cached array filename mismatch")
            relative_path = Path(expected_filename)
            expected_paths.add(relative_path)
            array_path = entry_path / relative_path
            if not array_path.is_file() or array_path.is_symlink():
                raise RecordingCacheCorruptionError("cached array file is missing or unsafe")
            if type(specification["file_bytes"]) is not int or specification[
                "file_bytes"
            ] != array_path.stat().st_size:
                raise RecordingCacheCorruptionError("cached array file length mismatch")
            file_identity = _file_identity(array_path)
            if (
                not isinstance(specification["file_sha256"], str)
                or _SHA256.fullmatch(specification["file_sha256"]) is None
                or specification["file_sha256"]
                != _verified_file_sha256_once(*file_identity)
            ):
                raise RecordingCacheCorruptionError("cached array file hash mismatch")
            try:
                array = np.load(array_path, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError, TypeError) as exc:
                raise RecordingCacheCorruptionError(
                    "cached array cannot be loaded without pickle"
                ) from exc
            if not isinstance(array, np.ndarray) or array.dtype.hasobject:
                raise RecordingCacheCorruptionError("cached payload is not a safe NumPy array")
            shape = specification["shape"]
            if (
                not isinstance(shape, list)
                or any(type(item) is not int or item < 0 for item in shape)
                or list(array.shape) != shape
            ):
                raise RecordingCacheCorruptionError("cached array shape mismatch")
            if specification["dtype"] != array.dtype.str:
                raise RecordingCacheCorruptionError("cached array dtype mismatch")
            if (
                type(specification["logical_nbytes"]) is not int
                or specification["logical_nbytes"] != int(array.nbytes)
            ):
                raise RecordingCacheCorruptionError("cached array logical size mismatch")
            observed_content = _verified_array_content_once(
                *file_identity,
                name,
            )
            if (
                not isinstance(specification["content_sha256"], str)
                or _SHA256.fullmatch(specification["content_sha256"]) is None
                or specification["content_sha256"] != observed_content
            ):
                raise RecordingCacheCorruptionError("cached array content hash mismatch")
            array.flags.writeable = False
            arrays[name] = array

        observed_paths = {
            path.relative_to(entry_path)
            for path in entry_path.rglob("*")
        }
        if observed_paths != expected_paths:
            raise RecordingCacheCorruptionError(
                "cache entry contains unexpected or missing filesystem objects"
            )

        return RecordingCacheEntry(
            key=identity.key,
            path=entry_path,
            arrays=MappingProxyType(arrays),
            attributes=MappingProxyType(attributes),
            metadata=MappingProxyType(metadata),
        )

    def put_arrays(
        self,
        identity: RecordingCacheIdentity,
        arrays: Mapping[str, np.ndarray],
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> RecordingCacheResult:
        """Publish arrays or verify an identical immutable entry already exists."""

        prepared = _prepare_arrays(arrays)
        strict_attributes = _normalise_json_object(
            {} if attributes is None else attributes,
            field_name="cache attributes",
        )
        lock_path = self._lock_path(identity)
        with _exclusive_file_lock(lock_path):
            entry_path = self._entry_path(identity)
            if _cache_path_exists(entry_path):
                entry = self.load(identity)
                self._assert_same_payload(entry, prepared, strict_attributes)
                return RecordingCacheResult(disposition="existing", entry=entry)
            self._publish_locked(identity, prepared, strict_attributes)
            return RecordingCacheResult(disposition="written", entry=self.load(identity))

    def get_or_compute(
        self,
        identity: RecordingCacheIdentity,
        builder: Callable[[], RecordingCacheBuild],
    ) -> RecordingCacheResult:
        """Load a hit or run exactly one builder while holding the per-key lock."""

        try:
            return RecordingCacheResult(disposition="hit", entry=self.load(identity))
        except CacheMissError:
            pass

        with _exclusive_file_lock(self._lock_path(identity)):
            entry_path = self._entry_path(identity)
            if _cache_path_exists(entry_path):
                return RecordingCacheResult(disposition="hit", entry=self.load(identity))
            built = builder()
            if not isinstance(built, RecordingCacheBuild):
                raise TypeError("recording cache builder must return RecordingCacheBuild")
            prepared = _prepare_arrays(built.arrays)
            strict_attributes = _normalise_json_object(
                built.attributes,
                field_name="cache attributes",
            )
            self._publish_locked(identity, prepared, strict_attributes)
            return RecordingCacheResult(disposition="written", entry=self.load(identity))

    def _assert_same_payload(
        self,
        entry: RecordingCacheEntry,
        prepared: Mapping[str, _PreparedArray],
        attributes: Mapping[str, Any],
    ) -> None:
        if dict(entry.attributes) != dict(attributes):
            raise ImmutableCacheConflictError(
                "cache identity already exists with different attributes"
            )
        if set(entry.arrays) != set(prepared):
            raise ImmutableCacheConflictError(
                "cache identity already exists with different array names"
            )
        metadata_arrays = entry.metadata["arrays"]
        for name, candidate in prepared.items():
            existing = metadata_arrays[name]
            if (
                existing["dtype"] != candidate.dtype
                or tuple(existing["shape"]) != candidate.shape
                or existing["logical_nbytes"] != candidate.logical_nbytes
                or existing["content_sha256"] != candidate.content_sha256
            ):
                raise ImmutableCacheConflictError(
                    f"cache identity already exists with different array {name!r}"
                )

    def _publish_locked(
        self,
        identity: RecordingCacheIdentity,
        arrays: Mapping[str, _PreparedArray],
        attributes: Mapping[str, Any],
    ) -> Path:
        entry_path = self._entry_path(identity)
        staging_path = self._new_staging_path(identity)
        try:
            if _cache_path_exists(entry_path):
                raise ImmutableCacheConflictError(
                    "cache entry appeared during publication"
                )
            entry_path.parent.mkdir(parents=True, exist_ok=True)
            staging_path.parent.mkdir(parents=True, exist_ok=True)
            staging_path.mkdir()
            array_directory = staging_path / _ARRAY_DIRECTORY
            array_directory.mkdir()
            array_manifest: dict[str, dict[str, Any]] = {}
            for name, prepared in arrays.items():
                array_path = array_directory / f"{name}.npy"
                with array_path.open("wb") as handle:
                    np.save(handle, prepared.value, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                array_manifest[name] = {
                    "filename": f"{_ARRAY_DIRECTORY}/{name}.npy",
                    "dtype": prepared.dtype,
                    "shape": list(prepared.shape),
                    "logical_nbytes": prepared.logical_nbytes,
                    "file_bytes": array_path.stat().st_size,
                    "file_sha256": _sha256_file(array_path),
                    "content_sha256": prepared.content_sha256,
                }
            _fsync_directory(array_directory)

            metadata = {
                "schema_version": RECORDING_CACHE_SCHEMA_VERSION,
                "cache_key": identity.key,
                "identity": identity.to_payload(),
                "arrays": array_manifest,
                "attributes": attributes,
            }
            metadata_bytes = _strict_json_bytes(metadata)
            metadata_path = staging_path / _METADATA_FILENAME
            metadata_path.write_bytes(metadata_bytes)
            _fsync_file(metadata_path)

            # The commit marker is intentionally the final file written.
            commit = {
                "schema_version": RECORDING_CACHE_COMMIT_VERSION,
                "cache_key": identity.key,
                "metadata_filename": _METADATA_FILENAME,
                "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "array_count": len(array_manifest),
            }
            commit_path = staging_path / _COMMIT_FILENAME
            commit_path.write_bytes(_strict_json_bytes(commit))
            _fsync_file(commit_path)
            _fsync_directory(staging_path)

            staging_path.rename(entry_path)
            _fsync_directory(entry_path.parent)
            return entry_path
        except BaseException as exc:
            try:
                if staging_path.exists():
                    shutil.rmtree(staging_path)
            except OSError as cleanup_exc:
                raise RecordingCacheAccessError(
                    f"cannot clean failed recording-cache staging path: {staging_path}"
                ) from cleanup_exc
            if isinstance(exc, (RecordingCacheError, TypeError, ValueError)):
                raise
            if isinstance(exc, OSError):
                raise RecordingCacheAccessError(
                    f"cannot publish recording-cache entry: {identity.key}"
                ) from exc
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

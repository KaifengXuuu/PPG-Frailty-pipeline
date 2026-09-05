"""Collect operational preprocessing-cache evidence for study reports.

The cache audit is deliberately report-only.  It summarizes the immutable
per-cell ``preprocessing_cache.json`` artifacts without reopening cache entries
or treating cache hits as scientific evidence.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


PREPROCESSING_CACHE_AUDIT_SCHEMA = "ppg_frailty.preprocessing_cache_audit.v1"

_CELL_DIRECTORY = re.compile(r"^repeat_(\d+)_fold_(\d+)$")
_HIT_DISPOSITIONS = frozenset({"hit", "existing"})
_WRITE_DISPOSITIONS = frozenset({"written"})
_BYPASS_DISPOSITIONS = frozenset(
    {"cache_off", "namespace_bypassed", "read_only_miss_computed"}
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_coordinates(
    path: Path,
    payload: Mapping[str, Any],
) -> tuple[int | None, int | None]:
    repeat = payload.get("repeat_index")
    fold = payload.get("fold_index")
    if isinstance(repeat, int) and not isinstance(repeat, bool):
        repeat_value: int | None = repeat
    else:
        repeat_value = None
    if isinstance(fold, int) and not isinstance(fold, bool):
        fold_value: int | None = fold
    else:
        fold_value = None
    if repeat_value is not None and fold_value is not None:
        return repeat_value, fold_value
    match = _CELL_DIRECTORY.fullmatch(path.parent.name)
    if match is None:
        return repeat_value, fold_value
    return int(match.group(1)), int(match.group(2))


def _nonnegative_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _empty_group(
    *,
    case_id: str,
    repeat: int | None,
    fold: int | None,
    payload: Mapping[str, Any],
    namespace: str,
    layer: str,
    artifact_path: Path,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "repeat": repeat,
        "fold": fold,
        "mode": payload.get("mode"),
        "cache_root": payload.get("root"),
        "namespace": namespace,
        "layer": layer,
        "audit_status": "available",
        "event_count": 0,
        "hit_count": 0,
        "hit_logical_array_bytes": 0,
        "hit_elapsed_seconds": 0.0,
        "write_count": 0,
        "write_logical_array_bytes": 0,
        "write_elapsed_seconds": 0.0,
        "bypass_count": 0,
        "bypass_logical_array_bytes": 0,
        "bypass_elapsed_seconds": 0.0,
        "other_count": 0,
        "other_logical_array_bytes": 0,
        "other_elapsed_seconds": 0.0,
        "logical_array_bytes": 0,
        "elapsed_seconds": 0.0,
        "unique_cache_key_count": 0,
        "cache_keys": set(),
        "module_chains": set(),
        "disposition_counts": {},
        "affects_predictions": payload.get("affects_predictions"),
        "labels_cached": payload.get("labels_cached"),
        "fold_local_artifacts_cached": payload.get(
            "fold_local_artifacts_cached"
        ),
        "route_masks_cached": payload.get("route_masks_cached"),
        "audit_artifact_within_case_artifact_root": artifact_path.as_posix(),
    }


def collect_preprocessing_cache_rows(
    case_id: str,
    artifact_root: str | Path,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    """Return per-cell/per-layer cache counts, logical bytes, and wall time.

    ``logical_array_bytes`` is the size of arrays materialized for an operation;
    it is intentionally not labelled as physical disk I/O.  ``existing`` is a
    cache hit, while an uncached computation in read-only mode is reported as a
    bypass and also retained in the exact ``disposition_counts`` mapping.
    """

    root = Path(artifact_root).resolve()
    groups: dict[tuple[int | None, int | None, str, str, str], dict[str, Any]] = {}
    limitations: list[str] = []
    for path in sorted(root.rglob("preprocessing_cache.json"), key=str):
        relative = path.relative_to(root)
        try:
            payload = _read_json(path)
            if not isinstance(payload, Mapping):
                raise TypeError("cache audit root must be a mapping")
            if payload.get("schema_version") != PREPROCESSING_CACHE_AUDIT_SCHEMA:
                raise ValueError(
                    "unsupported cache audit schema "
                    f"{payload.get('schema_version')!r}"
                )
            events = payload.get("events")
            if not isinstance(events, list):
                raise TypeError("cache audit events must be a list")
            repeat, fold = _cell_coordinates(path, payload)
            validated_events: list[
                tuple[str, str, str, float, int, str | None, tuple[str, ...] | None]
            ] = []
            for event_index, event in enumerate(events):
                if not isinstance(event, Mapping):
                    raise TypeError(f"event {event_index} must be a mapping")
                namespace = event.get("namespace")
                layer = event.get("layer")
                disposition = event.get("disposition")
                if not isinstance(namespace, str) or not namespace:
                    raise ValueError(f"event {event_index} lacks namespace")
                if not isinstance(layer, str) or not layer:
                    raise ValueError(f"event {event_index} lacks layer")
                if not isinstance(disposition, str) or not disposition:
                    raise ValueError(f"event {event_index} lacks disposition")
                elapsed = _nonnegative_number(
                    event.get("elapsed_seconds"),
                    field=f"event {event_index} elapsed_seconds",
                )
                logical_bytes_float = _nonnegative_number(
                    event.get("logical_array_bytes"),
                    field=f"event {event_index} logical_array_bytes",
                )
                if not logical_bytes_float.is_integer():
                    raise ValueError(
                        f"event {event_index} logical_array_bytes must be integral"
                    )
                logical_bytes = int(logical_bytes_float)
                cache_key = event.get("cache_key")
                cache_key = (
                    cache_key
                    if isinstance(cache_key, str) and cache_key
                    else None
                )
                module_chain = event.get("module_chain")
                module_chain_value = (
                    tuple(module_chain)
                    if isinstance(module_chain, list)
                    and all(isinstance(item, str) for item in module_chain)
                    else None
                )
                validated_events.append(
                    (
                        namespace,
                        layer,
                        disposition,
                        elapsed,
                        logical_bytes,
                        cache_key,
                        module_chain_value,
                    )
                )
            for (
                namespace,
                layer,
                disposition,
                elapsed,
                logical_bytes,
                cache_key,
                module_chain,
            ) in validated_events:
                key = (repeat, fold, str(payload.get("mode")), namespace, layer)
                row = groups.setdefault(
                    key,
                    _empty_group(
                        case_id=case_id,
                        repeat=repeat,
                        fold=fold,
                        payload=payload,
                        namespace=namespace,
                        layer=layer,
                        artifact_path=relative,
                    ),
                )
                row["event_count"] += 1
                row["logical_array_bytes"] += logical_bytes
                row["elapsed_seconds"] += elapsed
                row["disposition_counts"][disposition] = (
                    row["disposition_counts"].get(disposition, 0) + 1
                )
                if disposition in _HIT_DISPOSITIONS:
                    category = "hit"
                elif disposition in _WRITE_DISPOSITIONS:
                    category = "write"
                elif disposition in _BYPASS_DISPOSITIONS:
                    category = "bypass"
                else:
                    category = "other"
                row[f"{category}_count"] += 1
                row[f"{category}_logical_array_bytes"] += logical_bytes
                row[f"{category}_elapsed_seconds"] += elapsed
                if cache_key is not None:
                    row["cache_keys"].add(cache_key)
                if module_chain is not None:
                    row["module_chains"].add(module_chain)
        except Exception as error:  # noqa: BLE001 - keep the rest reportable.
            limitations.append(
                f"{case_id}: cannot summarize {relative.as_posix()}: "
                f"{type(error).__name__}: {error}"
            )

    rows: list[Mapping[str, Any]] = []
    for row in groups.values():
        keys = sorted(row.pop("cache_keys"))
        chains = [list(value) for value in sorted(row.pop("module_chains"))]
        row["unique_cache_key_count"] = len(keys)
        row["module_chains"] = chains
        row["disposition_counts"] = dict(sorted(row["disposition_counts"].items()))
        event_count = int(row["event_count"])
        row["hit_rate"] = row["hit_count"] / event_count if event_count else None
        row["write_rate"] = row["write_count"] / event_count if event_count else None
        row["bypass_rate"] = row["bypass_count"] / event_count if event_count else None
        rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row["case_id"]),
            -1 if row["repeat"] is None else int(row["repeat"]),
            -1 if row["fold"] is None else int(row["fold"]),
            str(row["namespace"]),
            str(row["layer"]),
        )
    )
    return tuple(rows), tuple(dict.fromkeys(limitations))


__all__ = [
    "PREPROCESSING_CACHE_AUDIT_SCHEMA",
    "collect_preprocessing_cache_rows",
]

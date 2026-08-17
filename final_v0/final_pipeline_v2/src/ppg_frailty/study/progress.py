"""Structured progress events with JSONL and dependency-free terminal sinks."""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TextIO


@dataclass(frozen=True)
class ProgressEvent:
    """One transport-neutral study progress update."""

    event: str
    current: int = 0
    total: int = 0
    case_id: str | None = None
    repeat: int | None = None
    fold: int | None = None
    epoch: int | None = None
    message: str = ""
    timestamp_utc: str = ""

    def __post_init__(self) -> None:
        if not self.event.strip():
            raise ValueError("progress event name must be non-empty")
        if self.current < 0 or self.total < 0:
            raise ValueError("progress counters cannot be negative")
        if self.total and self.current > self.total:
            raise ValueError("progress current cannot exceed total")
        if not self.timestamp_utc:
            object.__setattr__(
                self,
                "timestamp_utc",
                datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: "ProgressEvent | Mapping[str, Any] | str") -> "ProgressEvent":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(event="executor_progress", message=value)
        payload = dict(value)
        return cls(
            event=str(
                payload.pop(
                    "event",
                    payload.pop("type", payload.pop("stage", "executor_progress")),
                )
            ),
            current=int(payload.pop("current", payload.pop("current_cell", 0)) or 0),
            total=int(payload.pop("total", payload.pop("total_cells", 0)) or 0),
            case_id=payload.pop("case_id", None),
            repeat=payload.pop("repeat", payload.pop("repeat_index", None)),
            fold=payload.pop("fold", payload.pop("fold_index", None)),
            epoch=payload.pop("epoch", None),
            message=str(payload.pop("message", payload.pop("error", ""))),
            timestamp_utc=str(payload.pop("timestamp_utc", "")),
        )


ProgressSink = Callable[[ProgressEvent], None]


class NullProgressSink:
    def __call__(self, event: ProgressEvent) -> None:
        del event

    def close(self) -> None:
        return None


class CompositeProgressSink:
    """Fan out one event to terminal, JSONL, Dash, or test sinks."""

    def __init__(self, sinks: Iterable[ProgressSink]) -> None:
        self._sinks = tuple(sinks)

    def __call__(self, event: ProgressEvent) -> None:
        for sink in self._sinks:
            sink(event)

    def close(self) -> None:
        for sink in self._sinks:
            close = getattr(sink, "close", None)
            if callable(close):
                close()


class JsonlProgressSink:
    """Append structured events for resume diagnostics and Dash polling."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __call__(self, event: ProgressEvent) -> None:
        encoded = json.dumps(
            event.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")

    def close(self) -> None:
        return None


class TerminalProgressSink:
    """One ANSI-refreshed progress bar; it never prints one line per event."""

    def __init__(self, *, stream: TextIO | None = None, width: int = 28) -> None:
        self.stream = stream or sys.stderr
        self.width = max(10, int(width))
        self._lock = threading.Lock()
        self._last_width = 0
        self._closed = False

    def __call__(self, event: ProgressEvent) -> None:
        if self._closed:
            return
        total = max(0, int(event.total))
        current = max(0, int(event.current))
        ratio = 0.0 if total == 0 else min(1.0, current / total)
        filled = int(round(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        location = ""
        if event.case_id:
            location += f" {event.case_id}"
        if event.repeat is not None and event.fold is not None:
            location += f" r{event.repeat}f{event.fold}"
        if event.epoch is not None:
            location += f" ep{event.epoch}"
        count = f"{current}/{total}" if total else "working"
        message = f"[{bar}] {count} {event.event}{location} {event.message}".strip()
        with self._lock:
            padding = " " * max(0, self._last_width - len(message))
            self.stream.write("\r" + message + padding)
            self.stream.flush()
            self._last_width = len(message)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self.stream.write("\n")
                self.stream.flush()
                self._closed = True

"""Structured progress events with JSONL and dependency-free terminal sinks."""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
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
    unit_current: int | None = None
    unit_total: int | None = None
    detail_current: int | None = None
    detail_total: int | None = None
    detail_label: str = ""
    message: str = ""
    timestamp_utc: str = ""

    def __post_init__(self) -> None:
        if not self.event.strip():
            raise ValueError("progress event name must be non-empty")
        if self.current < 0 or self.total < 0:
            raise ValueError("progress counters cannot be negative")
        if self.total and self.current > self.total:
            raise ValueError("progress current cannot exceed total")
        for current_name, total_name in (
            ("unit_current", "unit_total"),
            ("detail_current", "detail_total"),
        ):
            current_value = getattr(self, current_name)
            total_value = getattr(self, total_name)
            if (current_value is None) != (total_value is None):
                raise ValueError(f"{current_name} and {total_name} must be paired")
            if current_value is not None:
                if current_value < 0 or total_value < 0:
                    raise ValueError("progress counters cannot be negative")
                if total_value and current_value > total_value:
                    raise ValueError(f"{current_name} cannot exceed {total_name}")
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
            unit_current=(
                None
                if payload.get("unit_current") is None
                else int(payload.pop("unit_current"))
            ),
            unit_total=(
                None
                if payload.get("unit_total") is None
                else int(payload.pop("unit_total"))
            ),
            detail_current=(
                None
                if payload.get("detail_current") is None
                else int(payload.pop("detail_current"))
            ),
            detail_total=(
                None
                if payload.get("detail_total") is None
                else int(payload.pop("detail_total"))
            ),
            detail_label=str(payload.pop("detail_label", "")),
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

    def __init__(
        self,
        sinks: Iterable[ProgressSink],
        *,
        close_sinks: Iterable[ProgressSink] | None = None,
    ) -> None:
        self._sinks = tuple(sinks)
        self._close_sinks = (
            self._sinks if close_sinks is None else tuple(close_sinks)
        )

    def __call__(self, event: ProgressEvent) -> None:
        for sink in self._sinks:
            sink(event)

    def close(self) -> None:
        for sink in self._close_sinks:
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


@dataclass
class _ActiveDetail:
    label: str
    current: int
    total: int
    started_at: float
    sequence: int


def _duration(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TerminalProgressSink:
    """Compact two-level progress display with elapsed time and approximate ETA."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        width: int = 20,
        refresh_interval: float = 1.0,
        ansi: bool | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.stream = stream or sys.stderr
        self.width = max(10, int(width))
        self._clock = clock or time.perf_counter
        self._started_at = self._clock()
        self._refresh_interval = max(0.0, float(refresh_interval))
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._ansi = is_tty if ansi is None else bool(ansi)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._rendered_lines = 0
        self._last_plain_width = 0
        self._sequence = 0
        self._study_mode = False
        self._overall_current = 0
        self._overall_total = 0
        self._overall_label = "starting"
        self._units_by_case: dict[str, int] = {}
        self._outcomes_by_case: dict[str, str] = {}
        self._study_status: str | None = None
        self._active: dict[str, _ActiveDetail] = {}

    def __call__(self, event: ProgressEvent) -> None:
        with self._lock:
            if self._closed:
                return
            self._apply(event)
            self._render_locked()
            if (
                self._ansi
                and self._refresh_interval > 0.0
                and self._thread is None
            ):
                self._thread = threading.Thread(
                    target=self._refresh_loop,
                    args=(self._stop,),
                    name="ppg-progress-refresh",
                    daemon=True,
                )
                self._thread.start()

    def suspend_refresh(self) -> None:
        """Pause only the timer thread while a process pool is being forked."""

        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop.set()
        if thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self._refresh_interval * 2.0))
        with self._lock:
            if self._thread is thread and not self._closed:
                self._thread = None
                self._stop = threading.Event()

    def _apply(self, event: ProgressEvent) -> None:
        self._sequence += 1
        now = self._clock()
        name = event.event
        if name == "study_started":
            units_per_case = int(event.unit_total or 1)
            self._study_mode = True
            self._overall_current = 0
            self._overall_total = int(event.total) * units_per_case
            self._overall_label = event.message or "study"
            self._units_by_case.clear()
            self._outcomes_by_case.clear()
            self._study_status = None
            self._active.clear()
            return
        if name == "study_running":
            self._overall_label = event.message or "running"
            return
        if name in {"program_started", "study_preparing", "plan_loaded"}:
            self._overall_label = event.message or name.replace("_", " ")
            self._set_detail(
                "__startup__",
                self._overall_label,
                event.detail_current or 0,
                event.detail_total or 0,
                now,
            )
            return
        if name == "case_queued":
            return
        if event.case_id and event.unit_current is not None:
            previous = self._units_by_case.get(event.case_id, 0)
            self._units_by_case[event.case_id] = max(
                previous, int(event.unit_current)
            )
            if self._study_mode:
                self._overall_current = min(
                    self._overall_total, sum(self._units_by_case.values())
                )
        if name == "case_resumed" and event.case_id:
            completed = int(event.unit_total or event.unit_current or 0)
            self._units_by_case[event.case_id] = completed
            self._outcomes_by_case[event.case_id] = "resumed"
            self._overall_current = min(
                self._overall_total, sum(self._units_by_case.values())
            )
            self._active.pop(event.case_id, None)
            return
        if name == "case_finished" and event.case_id:
            completed = int(event.unit_total or event.unit_current or 0)
            self._units_by_case[event.case_id] = completed
            self._outcomes_by_case[event.case_id] = event.message or "unknown"
            self._overall_current = min(
                self._overall_total, sum(self._units_by_case.values())
            )
            self._active.pop(event.case_id, None)
            self._overall_label = (
                f"{event.current}/{event.total} cases · last {event.message}"
                if event.total
                else event.message or self._overall_label
            )
            return
        if name == "study_finished":
            if event.total:
                units_per_case = max(1, self._overall_total // int(event.total))
                self._overall_current = min(
                    self._overall_total,
                    int(event.current) * units_per_case,
                )
            self._study_status = event.message or "complete"
            counts: dict[str, int] = {}
            for outcome in self._outcomes_by_case.values():
                counts[outcome] = counts.get(outcome, 0) + 1
            summary = " · ".join(
                f"{count} {outcome}"
                for outcome, count in sorted(counts.items())
            )
            self._overall_label = " · ".join(
                part for part in (self._study_status, summary) if part
            )
            self._active.clear()
            return
        if name == "report_started":
            self._set_detail("__report__", "report", 0, 1, now)
            return
        if name == "report_finished":
            self._active.pop("__report__", None)
            self._overall_label = event.message or (
                "report complete"
                if self._study_status is None
                else f"report complete · study {self._study_status}"
            )
            return
        if not self._study_mode and event.total:
            self._overall_current = int(event.current)
            self._overall_total = int(event.total)
            self._overall_label = event.message or name.replace("_", " ")
        detail_key = event.case_id or "__task__"
        if event.detail_current is not None:
            detail_current = int(event.detail_current)
            detail_total = int(event.detail_total or 0)
        else:
            detail_current = int(event.current)
            detail_total = int(event.total)
        label_parts = [part for part in (event.case_id, event.detail_label) if part]
        if not event.detail_label:
            if event.repeat is not None:
                label_parts.append(f"repeat {event.repeat + 1}")
            if event.fold is not None:
                label_parts.append(f"fold {event.fold + 1}")
            if not label_parts:
                label_parts.append(name.replace("_", " "))
        self._set_detail(
            detail_key,
            " · ".join(label_parts),
            detail_current,
            detail_total,
            now,
        )
        self._active.pop("__startup__", None)

    def _set_detail(
        self,
        key: str,
        label: str,
        current: int,
        total: int,
        now: float,
    ) -> None:
        previous = self._active.get(key)
        self._active[key] = _ActiveDetail(
            label=label,
            current=max(0, int(current)),
            total=max(0, int(total)),
            started_at=previous.started_at if previous is not None else now,
            sequence=self._sequence,
        )

    def _bar(self, current: int, total: int) -> str:
        ratio = 0.0 if total <= 0 else min(1.0, current / total)
        filled = int(round(self.width * ratio))
        return "#" * filled + "-" * (self.width - filled)

    def _eta(self, elapsed: float, current: int, total: int) -> str:
        if total <= 0 or current <= 0:
            return "--:--:--"
        if current >= total:
            return "00:00:00"
        return _duration(elapsed * (total - current) / current)

    def _lines(self) -> tuple[str, str | None]:
        now = self._clock()
        elapsed = now - self._started_at
        count = (
            f"{self._overall_current}/{self._overall_total} case-repeats"
            if self._study_mode and self._overall_total
            else f"{self._overall_current}/{self._overall_total}"
            if self._overall_total
            else "working"
        )
        total_line = (
            f"TOTAL [{self._bar(self._overall_current, self._overall_total)}] "
            f"{count}  elapsed {_duration(elapsed)}  "
            f"ETA~ {self._eta(elapsed, self._overall_current, self._overall_total)} "
            f"{self._overall_label}"
        )
        if not self._active:
            return total_line, None
        detail = max(self._active.values(), key=lambda value: value.sequence)
        detail_count = (
            f"{detail.current}/{detail.total}"
            if detail.total
            else "working"
        )
        detail_line = (
            f" SUB  [{self._bar(detail.current, detail.total)}] {detail_count}  "
            f"elapsed {_duration(now - detail.started_at)}  {detail.label}"
        )
        return total_line, detail_line

    @staticmethod
    def _fit(line: str) -> str:
        columns = max(40, shutil.get_terminal_size(fallback=(120, 24)).columns)
        return line if len(line) <= columns else line[: columns - 3] + "..."

    def _render_locked(self) -> None:
        total_line, detail_line = self._lines()
        total_line = self._fit(total_line)
        detail_line = None if detail_line is None else self._fit(detail_line)
        if not self._ansi:
            combined = total_line if detail_line is None else f"{total_line} | {detail_line}"
            padding = " " * max(0, self._last_plain_width - len(combined))
            self.stream.write("\r" + combined + padding)
            self.stream.flush()
            self._last_plain_width = len(combined)
            return
        if self._rendered_lines == 2:
            self.stream.write("\r\x1b[2K\x1b[1A\r\x1b[2K")
        elif self._rendered_lines == 1:
            self.stream.write("\r\x1b[2K")
        self.stream.write(total_line)
        if detail_line is not None:
            self.stream.write("\n" + detail_line)
            self._rendered_lines = 2
        else:
            self._rendered_lines = 1
        self.stream.flush()

    def _refresh_loop(self, stop: threading.Event) -> None:
        while not stop.wait(self._refresh_interval):
            with self._lock:
                if self._closed:
                    return
                self._render_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self._refresh_interval * 2.0))
        with self._lock:
            self._active.clear()
            self._render_locked()
            self.stream.write("\n")
            self.stream.flush()

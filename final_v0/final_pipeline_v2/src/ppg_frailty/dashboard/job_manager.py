"""Subprocess study jobs for Dash; long training never runs inside a callback."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class StudyJob:
    job_id: str
    command: tuple[str, ...]
    started_at: float
    process: subprocess.Popen[str]
    log_path: Path
    output_root: Path


class StudyJobManager:
    """Launch and observe the dedicated study CLI without shell evaluation."""

    def __init__(self, pipeline_root: str | Path | None = None) -> None:
        inferred = Path(__file__).resolve().parents[3]
        self.pipeline_root = Path(pipeline_root or inferred).resolve()
        self._jobs: dict[str, StudyJob] = {}
        self._lock = threading.Lock()

    def start(self, arguments: Sequence[str]) -> str:
        if not arguments:
            raise ValueError("study CLI arguments must be non-empty")
        script = (self.pipeline_root / "frailty_3class_sweep_v2.py").resolve()
        script.relative_to(self.pipeline_root)
        if not script.is_file():
            raise FileNotFoundError(script)
        log_root = self.pipeline_root / "artifacts" / "dashboard_jobs"
        log_root.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        log_path = log_root / f"{job_id}.log"
        supplied = [str(value) for value in arguments]
        resume = self._argument_value(supplied, "--resume")
        report_dir = self._argument_value(supplied, "--study-dir")
        output_value = self._argument_value(supplied, "--output-root")
        if resume is not None:
            output_root = self._resolve_output_path(resume)
        elif report_dir is not None:
            output_root = self._resolve_output_path(report_dir)
        elif output_value is not None:
            output_root = self._resolve_output_path(output_value)
        else:
            output_root = (
                self.pipeline_root
                / "artifacts"
                / "studies"
                / f"dashboard_{job_id}"
            ).resolve()
            supplied.extend(["--output-root", str(output_root)])
        command = (sys.executable, str(script), *tuple(supplied))
        handle = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=self.pipeline_root,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
                start_new_session=(os.name != "nt"),
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if os.name == "nt"
                    else 0
                ),
            )
        finally:
            handle.close()
        job = StudyJob(
            job_id,
            command,
            time.time(),
            process,
            log_path,
            output_root,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job_id

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(str(job_id))
        if job is None:
            raise KeyError(f"unknown study job: {job_id}")
        return_code = job.process.poll()
        progress = self._latest_progress(job.output_root)
        return {
            "job_id": job.job_id,
            "state": "running" if return_code is None else ("passed" if return_code == 0 else "failed"),
            "return_code": return_code,
            "started_at_unix": job.started_at,
            "elapsed_s": max(0.0, time.time() - job.started_at),
            "command": list(job.command),
            "log_path": str(job.log_path),
            "output_root": str(job.output_root),
            "log_tail": self._tail(job.log_path, 30),
            "progress": progress,
        }

    def terminate(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(str(job_id))
        if job is None:
            raise KeyError(f"unknown study job: {job_id}")
        if job.process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    [
                        "taskkill",
                        "/PID",
                        str(job.process.pid),
                        "/T",
                        "/F",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            else:
                try:
                    os.killpg(os.getpgid(job.process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    return
            try:
                job.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    job.process.kill()
                else:
                    try:
                        os.killpg(os.getpgid(job.process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def _latest_progress(
        self,
        output_root: Path | None = None,
    ) -> dict[str, Any] | None:
        studies = (
            output_root.resolve()
            if output_root is not None
            else (self.pipeline_root / "artifacts" / "studies").resolve()
        )
        if not studies.exists():
            return None
        candidates = sorted(
            studies.rglob("progress_events.jsonl"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            try:
                lines = candidates[0].read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                payload = json.loads(lines[-1]) if lines else {}
            except (OSError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, dict) and payload:
                return payload
        candidates = sorted(
            studies.rglob("progress.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        try:
            payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _resolve_output_path(self, value: str) -> Path:
        raw = Path(value)
        return (
            raw.resolve()
            if raw.is_absolute()
            else (self.pipeline_root / raw).resolve()
        )

    @staticmethod
    def _argument_value(arguments: Sequence[str], name: str) -> str | None:
        try:
            index = list(arguments).index(name)
        except ValueError:
            return None
        if index + 1 >= len(arguments):
            raise ValueError(f"{name} requires a path value")
        return str(arguments[index + 1])

    @staticmethod
    def _tail(path: Path, lines: int) -> list[str]:
        try:
            values = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return values[-int(lines):]

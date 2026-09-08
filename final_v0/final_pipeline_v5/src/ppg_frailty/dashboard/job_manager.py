"""Shell-free background jobs for the V5 Dash control panel."""
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
from .control_service import CommandRequest, PIPELINE_OUTPUT, REPORT_OUTPUT

_ALLOWED_SCRIPTS = frozenset({'pipeline.py', 'sweep.py', 'analyse_report.py', 'specialized_pipeline.py', 'comparison_sequence.py'})


@dataclass(frozen=True)
class DashboardJob:
    job_id: str
    kind: str
    command: tuple[str, ...]
    started_at: float
    process: subprocess.Popen[str]
    log_path: Path
    watched_root: Path


class DashboardJobManager:
    """Launch V5 CLI requests outside callbacks and observe their progress.

    Arguments are always passed as an argv sequence with ``shell=False``. The
    UI cannot choose an executable or escape the V5 project root.
    """
    def __init__(self, pipeline_root: str | Path | None = None) -> None:
        inferred = Path(__file__).resolve().parents[3]
        self.pipeline_root = Path(pipeline_root or inferred).resolve()
        self._jobs: dict[str, DashboardJob] = {}
        self._lock = threading.Lock()

    def start_request(self, request: CommandRequest, *, kind: str) -> str:
        if request.script not in _ALLOWED_SCRIPTS:
            raise ValueError(f'dashboard cannot launch script: {request.script}')
        return self.start(request.script, request.arguments, kind=kind)

    def start(self, script: str | Sequence[str], arguments: Sequence[str] | None = None, *, kind: str = 'pipeline') -> str:
        """Start an approved root script; legacy ``start(argv)`` maps to pipeline."""
        if arguments is None and (not isinstance(script, str)):
            arguments = script
            script = 'pipeline.py'
        if not isinstance(script, str) or script not in _ALLOWED_SCRIPTS:
            raise ValueError(
                'dashboard script must be pipeline.py, sweep.py, analyse_report.py, specialized_pipeline.py, or comparison_sequence.py'
            )
        supplied = tuple((str(value) for value in arguments or ()))
        if not supplied or any(('\x00' in value for value in supplied)):
            raise ValueError('dashboard CLI arguments must be non-empty safe strings')
        script_path = (self.pipeline_root / script).resolve()
        script_path.relative_to(self.pipeline_root)
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
        log_root = self.pipeline_root / PIPELINE_OUTPUT / '.dashboard_jobs'
        log_root.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        log_path = log_root / f'{job_id}.log'
        watched_root = self._watched_root(supplied, script=script)
        command = (sys.executable, str(script_path), *supplied)
        handle = log_path.open('w', encoding='utf-8')
        try:
            process = subprocess.Popen(command,
                                       cwd=self.pipeline_root,
                                       stdin=subprocess.DEVNULL,
                                       stdout=handle,
                                       stderr=subprocess.STDOUT,
                                       text=True,
                                       shell=False,
                                       start_new_session=os.name != 'nt',
                                       creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        finally:
            handle.close()
        job = DashboardJob(job_id=job_id,
                           kind=str(kind),
                           command=command,
                           started_at=time.time(),
                           process=process,
                           log_path=log_path,
                           watched_root=watched_root)
        with self._lock:
            self._jobs[job_id] = job
        return job_id

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(str(job_id))
        if job is None:
            raise KeyError(f'unknown dashboard job: {job_id}')
        return_code = job.process.poll()
        return {
            'job_id': job.job_id,
            'kind': job.kind,
            'state': 'running' if return_code is None else 'passed' if return_code == 0 else 'failed',
            'return_code': return_code,
            'started_at_unix': job.started_at,
            'elapsed_s': max(0.0,
                             time.time() - job.started_at),
            'command': list(job.command),
            'log_path': str(job.log_path),
            'watched_root': str(job.watched_root),
            'log_tail': self._tail(job.log_path, 60),
            'progress': self._latest_progress(job.watched_root)
        }

    def terminate(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(str(job_id))
        if job is None:
            raise KeyError(f'unknown dashboard job: {job_id}')
        if job.process.poll() is not None:
            return
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(job.process.pid), '/T', '/F'], check=False, capture_output=True, text=True)
        else:
            try:
                os.killpg(os.getpgid(job.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                return
        try:
            job.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                job.process.kill()
            else:
                try:
                    os.killpg(os.getpgid(job.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def _watched_root(self, arguments: Sequence[str], *, script: str) -> Path:
        for name in ('--output-dir', '--output-root', '--resume'):
            value = self._argument_value(arguments, name)
            if value is not None:
                raw = Path(value)
                target = raw.resolve() if raw.is_absolute() else (self.pipeline_root / raw).resolve()
                target.relative_to(self.pipeline_root)
                return target
        if script == 'analyse_report.py':
            output_name = self._argument_value(arguments, '--output-name')
            root = (self.pipeline_root / REPORT_OUTPUT).resolve()
            if output_name is None:
                return root
            target = (root / output_name).resolve()
            target.relative_to(root)
            return target
        return (self.pipeline_root / PIPELINE_OUTPUT).resolve()

    @staticmethod
    def _argument_value(arguments: Sequence[str], name: str) -> str | None:
        try:
            index = list(arguments).index(name)
        except ValueError:
            return None
        if index + 1 >= len(arguments):
            raise ValueError(f'{name} requires a path value')
        return str(arguments[index + 1])

    @staticmethod
    def _latest_progress(root: Path) -> dict[str, Any] | None:
        if not root.exists():
            return None
        candidates = sorted(
            (path for pattern in ('progress_events.jsonl', 'progress.json') for path in root.rglob(pattern) if path.is_file()),
            key=lambda item: item.stat().st_mtime,
            reverse=True)
        for path in candidates:
            try:
                if path.suffix == '.jsonl':
                    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
                    payload = json.loads(lines[-1]) if lines else None
                else:
                    payload = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _tail(path: Path, lines: int) -> list[str]:
        try:
            values = path.read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError:
            return []
        return values[-int(lines):]


StudyJobManager = DashboardJobManager
__all__ = ['DashboardJobManager', 'StudyJobManager']

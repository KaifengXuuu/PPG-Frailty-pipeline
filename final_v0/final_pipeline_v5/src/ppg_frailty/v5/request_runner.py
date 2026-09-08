"""Minimal request recording around the unchanged study runner."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from filelock import FileLock, Timeout

from ..study import StudyRunner
from .io import atomic_json, file_sha256, payload_sha256


REQUEST_ANCHOR_SCHEMA = "ppg_frailty.v5_request_anchor.v1"
REQUEST_BINDING_ENV = "PPG_FRAILTY_V5_TRAINING_REQUEST_BINDING"

def execution_binding(plan: Any, expansion: Any) -> dict[str, Any]:
    core = {
        "repeats": list(plan.execution.repeats),
        "folds": list(plan.execution.folds),
        "device": plan.execution.device,
        "reference_case_id": expansion.reference_case_id,
        "cases": [
            {
                "case_id": case.case_id,
                "config_id": case.config.get("config_id"),
                "config_sha256": case.config_sha256,
            }
            for case in expansion.cases
        ],
    }
    return {"schema_version": "ppg_frailty.v5_execution_binding.v1", **core, "binding_sha256": payload_sha256(core)}

def read_anchored_request(
    output: str | Path,
    relative_path: str | Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Read a request and digest; old anchors remain readable."""

    root, relative = Path(output).resolve(), Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("request path must be safe and study-relative")
    path = (root / relative).resolve()
    path.relative_to(root)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise TypeError(f"request must contain a mapping: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    anchor_path = path.with_suffix(".anchor.json")
    if not anchor_path.exists():
        return dict(payload), digest, {}
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    if not isinstance(anchor, Mapping) or anchor.get("request_sha256") != digest:
        raise ValueError(f"request anchor mismatch: {path}")
    return dict(payload), digest, dict(anchor)

def write_request_status(
    output: str | Path,
    relative_path: str | Path,
    *,
    status: str,
    error: BaseException | None = None,
) -> None:
    root = Path(output).resolve()
    payload, digest, _ = read_anchored_request(root, relative_path)
    target = root / ("v5_resume_request.json" if payload.get("resumed") else "v5_run_status.json")
    record: dict[str, Any] = {
        **payload,
        "latest_request": Path(relative_path).as_posix(),
        "latest_request_sha256": digest,
        "latest_immutable_request": Path(relative_path).as_posix(),
        "latest_immutable_request_sha256": digest,
        "attempt_status": status,
    }
    if error is not None:
        record["attempt_error"] = {"type": type(error).__name__, "message": str(error)}
    atomic_json(target, record)

@contextmanager
def exclusive_resume_lock(output: str | Path | None) -> Iterator[None]:
    """Prevent two processes from publishing the same run simultaneously."""

    if output is None:
        yield
        return
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(target.parent / f".{target.name}.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout as error:
        raise RuntimeError(f"another V5 process is using run target: {target}") from error
    try:
        yield
    finally:
        lock.release()

def validate_resume_environment(
    output: str | Path,
    current_request: Mapping[str, Any],
) -> None:
    """Deprecated compatibility hook; the entry service checks environment once."""

    del output, current_request

class RequestRecordingStudyRunner(StudyRunner):
    """Publish a plain request before delegating to ``StudyRunner``."""

    def __init__(
        self,
        *,
        pre_run_artifacts: Mapping[str, Mapping[str, Any]],
        precomputed_expansion: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if len(pre_run_artifacts) != 1:
            raise ValueError("one V5 execution must have exactly one training request")
        self._requests = {str(name): dict(value) for name, value in pre_run_artifacts.items()}
        self._expansion = precomputed_expansion

    def expand(self, plan: Any) -> Any:
        if self._expansion is None:
            return super().expand(plan)
        if self._expansion.plan.to_dict() != plan.to_dict():
            raise ValueError("precomputed expansion and executing study plan differ")
        return self._expansion

    def _publish_pre_run_artifacts(self, output: Path) -> None:
        root = output.resolve()
        for name, payload in self._requests.items():
            relative = Path(name)
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ValueError("pre-run artifact must be a safe relative path")
            target = (root / relative).resolve()
            target.relative_to(root)
            if target.exists():
                raise FileExistsError(f"pre-run request already exists: {target}")
            atomic_json(target, payload)

    def _new_output(self, plan: Any, output_root: str | Path | None, *, run_name: str | None = None) -> Path:
        output = super()._new_output(plan, output_root, run_name=run_name)
        self._publish_pre_run_artifacts(output)
        return output

    def run(
        self,
        plan: Any,
        *,
        output_root: str | Path | None = None,
        resume_directory: str | Path | None = None,
        run_name: str | None = None,
    ) -> Any:
        if resume_directory is not None:
            self._publish_pre_run_artifacts(Path(resume_directory))
        try:
            return super().run(
                plan,
                output_root=output_root,
                resume_directory=resume_directory,
                run_name=run_name,
            )
        except BaseException as error:
            output = (
                Path(resume_directory) if resume_directory else Path(output_root or self.pipeline_root) / str(run_name)
            )
            if output.is_dir():
                write_request_status(output, next(iter(self._requests)), status="runner_failed", error=error)
            raise


sha256_file = file_sha256

__all__ = [
    "RequestRecordingStudyRunner",
    "REQUEST_ANCHOR_SCHEMA",
    "REQUEST_BINDING_ENV",
    "exclusive_resume_lock",
    "execution_binding",
    "read_anchored_request",
    "sha256_file",
    "validate_resume_environment",
    "write_request_status",
]

"""Shared fail-closed source and dependency gate for V2 scientific execution."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .config import dependency_gate_report_for_profiles
from .provenance import stable_payload_sha256


SCIENTIFIC_SOURCE_GATE_SCHEMA = "ppg_frailty.scientific_source_gate.v1"
SCIENTIFIC_GATE_CAPTURE_SCHEMA = "ppg_frailty.scientific_gate_capture.v1"
SCIENTIFIC_GATE_EVIDENCE_SCHEMA = "ppg_frailty.scientific_gate_evidence.v1"
MOTION_SCIENTIFIC_GATE_CONFIG_SCHEMA = (
    "ppg_frailty.motion_scientific_execution_gate.v1"
)
MOTION_GATE_CONFIG_RELATIVE_PATH = Path("configs/motion_detector_contract_v2.yaml")
MOTION_PIPELINE_RELATIVE_PATH = Path("final_v0/final_pipeline_v2")
SCIENTIFIC_SOURCE_SNAPSHOT_ROOTS = (
    "src",
    "configs",
    "tools",
    "requirements",
    "locks",
)
MOTION_REQUIRED_DEPENDENCY_PROFILES = ("core", "deep")
MOTION_INTERNAL_GATE_PHASES = (
    "entry_preflight",
    "fold_0_0_fit_pre",
    "fold_0_0_fit_post",
    "fold_0_1_fit_pre",
    "fold_0_1_fit_post",
    "fold_0_2_fit_pre",
    "fold_0_2_fit_post",
    "fold_0_3_fit_pre",
    "fold_0_3_fit_post",
    "fold_0_4_fit_pre",
    "fold_0_4_fit_post",
    "final_fit_pre",
    "final_fit_post",
)
MOTION_PTT_GATE_PHASES = (
    "entry_preflight",
    "ptt_predict_pre",
    "ptt_predict_post",
)
_CONTROLLED_OUTPUT_RELATIVE_ROOT = "artifacts"


class ScientificGateError(RuntimeError):
    """Raised before scientific work when source or dependency identity drifts."""


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def git_source_state(
    repository_root: str | Path,
    pipeline_root: str | Path,
) -> dict[str, Any]:
    """Describe whether the exact V2 tree is tracked, clean, and commit-bound."""

    repository = Path(repository_root).resolve()
    pipeline = Path(pipeline_root).resolve()
    try:
        relative = pipeline.relative_to(repository).as_posix()
    except ValueError as exc:
        raise ScientificGateError(
            "formal_scientific_pipeline_root_outside_repository"
        ) from exc
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            relative,
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "--",
            f"{relative}/pyproject.toml",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = head.stdout.strip()
    if (
        status.returncode != 0
        or head.returncode != 0
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ScientificGateError("formal_scientific_git_source_status_unavailable")
    raw_rows = tuple(line for line in status.stdout.splitlines() if line)
    controlled_prefix = (
        f"{relative}/{_CONTROLLED_OUTPUT_RELATIVE_ROOT}/"
    )

    def is_controlled_output(row: str) -> bool:
        # Porcelain-v1 prefixes every path with two status columns and a space.
        # Renames contain old -> new; both endpoints must remain inside the
        # one reviewed output namespace before the row may be ignored.
        payload = row[3:] if len(row) >= 4 else row
        names = tuple(part.strip().strip('"') for part in payload.split(" -> "))
        return bool(names) and all(
            name == controlled_prefix[:-1] or name.startswith(controlled_prefix)
            for name in names
        )

    rows = tuple(row for row in raw_rows if not is_controlled_output(row))
    return {
        "schema_version": SCIENTIFIC_SOURCE_GATE_SCHEMA,
        "v2_path": relative,
        "git_head_commit": commit,
        "v2_pyproject_tracked": tracked.returncode == 0,
        "status_entry_count": len(rows),
        "status_sha256": hashlib.sha256(
            "\n".join(rows).encode("utf-8")
        ).hexdigest(),
        "status_preview": list(rows[:20]),
        "tracked_and_clean": bool(tracked.returncode == 0 and not rows),
    }


def require_clean_tracked_source_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject dirty or untracked V2 state without an override."""

    payload = dict(state)
    if (
        payload.get("schema_version") != SCIENTIFIC_SOURCE_GATE_SCHEMA
        or payload.get("tracked_and_clean") is not True
        or payload.get("v2_pyproject_tracked") is not True
        or payload.get("status_entry_count") != 0
    ):
        raise ScientificGateError(
            "formal_scientific_run_requires_clean_tracked_final_pipeline_v2"
        )
    return payload


def source_snapshot_sha256(pipeline_root: str | Path) -> str:
    """Hash all active V2 code/config/dependency inputs in stable path order."""

    root = Path(pipeline_root).resolve()
    candidates: list[Path] = []
    for relative_root in SCIENTIFIC_SOURCE_SNAPSHOT_ROOTS:
        directory = root / relative_root
        if directory.is_dir():
            candidates.extend(
                item
                for item in directory.rglob("*")
                if item.is_file()
                and "__pycache__" not in item.parts
                and item.suffix != ".pyc"
            )
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        candidates.append(pyproject)
    unique = sorted(
        set(candidates),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    if not unique:
        raise ScientificGateError("formal_scientific_source_snapshot_empty")
    digest = hashlib.sha256()
    for item in unique:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _load_motion_gate_config(pipeline_root: Path) -> dict[str, Any]:
    """Load the exact core/deep gate roster from the authoritative YAML."""

    source = (pipeline_root / MOTION_GATE_CONFIG_RELATIVE_PATH).resolve()
    try:
        source.relative_to(pipeline_root)
        source_bytes = source.read_bytes()
        payload = yaml.safe_load(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        raise ScientificGateError(
            "formal_motion_scientific_gate_config_unavailable"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ScientificGateError("formal_motion_scientific_gate_config_invalid")
    section = payload.get("scientific_execution_gate")
    required_keys = {
        "schema_version",
        "operation",
        "source_tree_policy",
        "source_snapshot_roots",
        "required_dependency_profile_ids",
        "require_exact_locks",
        "internal_checkpoints",
        "ptt_checkpoints",
    }
    if not isinstance(section, Mapping) or set(section) != required_keys:
        raise ScientificGateError("formal_motion_scientific_gate_config_invalid")
    try:
        normalized = {
            "schema_version": str(section["schema_version"]),
            "operation": str(section["operation"]),
            "source_tree_policy": str(section["source_tree_policy"]),
            "source_snapshot_roots": [
                str(value) for value in section["source_snapshot_roots"]
            ],
            "required_dependency_profile_ids": [
                str(value) for value in section["required_dependency_profile_ids"]
            ],
            "require_exact_locks": section["require_exact_locks"],
            "internal_checkpoints": [
                str(value) for value in section["internal_checkpoints"]
            ],
            "ptt_checkpoints": [
                str(value) for value in section["ptt_checkpoints"]
            ],
        }
    except (KeyError, TypeError) as exc:
        raise ScientificGateError(
            "formal_motion_scientific_gate_config_invalid"
        ) from exc
    if (
        normalized["schema_version"] != MOTION_SCIENTIFIC_GATE_CONFIG_SCHEMA
        or normalized["operation"] != "motion_formal_reference"
        or normalized["source_tree_policy"]
        != "tracked_clean_final_pipeline_v2_no_override"
        or tuple(normalized["source_snapshot_roots"])
        != SCIENTIFIC_SOURCE_SNAPSHOT_ROOTS
        or tuple(normalized["required_dependency_profile_ids"])
        != MOTION_REQUIRED_DEPENDENCY_PROFILES
        or normalized["require_exact_locks"] is not True
        or tuple(normalized["internal_checkpoints"])
        != ("entry_preflight", "every_fit_pre", "every_fit_post")
        or tuple(normalized["ptt_checkpoints"])
        != ("entry_preflight", "predict_pre", "predict_post")
    ):
        raise ScientificGateError("formal_motion_scientific_gate_config_drift")
    formal_model = payload.get("formal_model")
    if not isinstance(formal_model, Mapping):
        raise ScientificGateError("formal_motion_scientific_gate_config_invalid")
    return {
        "contract_id": str(payload.get("contract_id", "")),
        "formal_model_id": str(formal_model.get("model_id", "")),
        "config_relative_path": MOTION_GATE_CONFIG_RELATIVE_PATH.as_posix(),
        "config_sha256": hashlib.sha256(source_bytes).hexdigest(),
        **normalized,
    }


def capture_motion_scientific_gate(
    repository_root: str | Path,
    pipeline_root: str | Path | None = None,
) -> dict[str, Any]:
    """Capture one exact source/code/dependency state before scientific work."""

    repository = Path(repository_root).resolve()
    pipeline = (
        Path(pipeline_root).resolve()
        if pipeline_root is not None
        else (repository / MOTION_PIPELINE_RELATIVE_PATH).resolve()
    )
    source_state = require_clean_tracked_source_state(
        git_source_state(repository, pipeline)
    )
    gate_config = _load_motion_gate_config(pipeline)
    try:
        dependency = dependency_gate_report_for_profiles(
            config_id=(
                f"{gate_config['contract_id']}:{gate_config['formal_model_id']}"
            ),
            required_profile_ids=gate_config["required_dependency_profile_ids"],
            operation=gate_config["operation"],
            profiles_path=pipeline / "requirements/profiles.json",
            lock_path=pipeline / "locks/profiles.lock.json",
            require_exact_lock=True,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ScientificGateError(
            "formal_motion_exact_dependency_gate_closed"
        ) from exc
    dependency_sha256 = stable_payload_sha256(dependency)
    source_snapshot = source_snapshot_sha256(pipeline)
    source_state_after = require_clean_tracked_source_state(
        git_source_state(repository, pipeline)
    )
    source_snapshot_after = source_snapshot_sha256(pipeline)
    if (
        stable_payload_sha256(source_state_after)
        != stable_payload_sha256(source_state)
        or source_snapshot_after != source_snapshot
    ):
        raise ScientificGateError(
            "formal_motion_source_changed_during_gate_capture"
        )
    return {
        "schema_version": SCIENTIFIC_GATE_CAPTURE_SCHEMA,
        "pipeline_root": str(pipeline),
        "motion_gate_config": gate_config,
        "source_tree_state": source_state,
        "source_snapshot_sha256": source_snapshot,
        "dependency_gate": dependency,
        "dependency_gate_sha256": dependency_sha256,
    }


def _checkpoint_record(phase: str, capture: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        "phase": str(phase),
        "gate_capture_sha256": stable_payload_sha256(capture),
        "source_snapshot_sha256": str(capture["source_snapshot_sha256"]),
        "dependency_gate_sha256": str(capture["dependency_gate_sha256"]),
    }
    return {**record, "checkpoint_sha256": stable_payload_sha256(record)}


@dataclass
class ScientificGateSession:
    """TOCTOU-resistant source/dependency baseline with named checkpoints."""

    repository_root: Path
    pipeline_root: Path
    baseline: dict[str, Any]
    baseline_sha256: str
    checkpoints: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def establish(
        cls,
        repository_root: str | Path,
        pipeline_root: str | Path | None = None,
    ) -> "ScientificGateSession":
        repository = Path(repository_root).resolve()
        pipeline = (
            Path(pipeline_root).resolve()
            if pipeline_root is not None
            else (repository / MOTION_PIPELINE_RELATIVE_PATH).resolve()
        )
        baseline = capture_motion_scientific_gate(repository, pipeline)
        session = cls(
            repository_root=repository,
            pipeline_root=pipeline,
            baseline=baseline,
            baseline_sha256=stable_payload_sha256(baseline),
        )
        session.checkpoints.append(_checkpoint_record("entry_preflight", baseline))
        return session

    def checkpoint(self, phase: str) -> dict[str, Any]:
        if not str(phase).strip() or any(
            item["phase"] == str(phase) for item in self.checkpoints
        ):
            raise ScientificGateError(
                "formal_motion_scientific_gate_checkpoint_invalid_or_duplicate"
            )
        current = capture_motion_scientific_gate(
            self.repository_root,
            self.pipeline_root,
        )
        if stable_payload_sha256(current) != self.baseline_sha256:
            raise ScientificGateError(
                f"formal_motion_scientific_gate_changed_at_{phase}"
            )
        record = _checkpoint_record(str(phase), current)
        self.checkpoints.append(record)
        return dict(record)

    def evidence(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCIENTIFIC_GATE_EVIDENCE_SCHEMA,
            "baseline": dict(self.baseline),
            "baseline_sha256": self.baseline_sha256,
            "checkpoints": [dict(item) for item in self.checkpoints],
        }
        return {
            **payload,
            "scientific_gate_evidence_sha256": stable_payload_sha256(payload),
        }


def establish_motion_scientific_gate(
    repository_root: str | Path,
    pipeline_root: str | Path | None = None,
) -> ScientificGateSession:
    """Establish the mandatory motion formal-entry baseline."""

    return ScientificGateSession.establish(repository_root, pipeline_root)


def verify_scientific_gate_evidence(
    evidence: object,
    *,
    expected_phases: Sequence[str],
) -> tuple[str, ...]:
    """Verify archived gate/checkpoint hashes without claiming a live recheck."""

    if not isinstance(evidence, Mapping):
        return ("scientific_gate_evidence_missing",)
    payload = dict(evidence)
    expected_keys = {
        "schema_version",
        "baseline",
        "baseline_sha256",
        "checkpoints",
        "scientific_gate_evidence_sha256",
    }
    if set(payload) != expected_keys:
        return ("scientific_gate_evidence_field_schema_drift",)
    reasons: list[str] = []
    seal = payload.pop("scientific_gate_evidence_sha256", "")
    if stable_payload_sha256(payload) != seal:
        reasons.append("scientific_gate_evidence_hash_drift")
    baseline = payload.get("baseline")
    if not isinstance(baseline, Mapping):
        return tuple(dict.fromkeys([*reasons, "scientific_gate_baseline_missing"]))
    baseline_hash = stable_payload_sha256(baseline)
    source_state = baseline.get("source_tree_state")
    gate_config = baseline.get("motion_gate_config")
    dependency = baseline.get("dependency_gate")
    dependency_profiles = (
        dependency.get("profiles")
        if isinstance(dependency, Mapping)
        else None
    )
    profile_rows_valid = (
        isinstance(dependency_profiles, list)
        and all(
            isinstance(row, Mapping)
            and row.get("lock_status") == "validated_exact_lock"
            and row.get("exact_lock_ready") is True
            and isinstance(row.get("live_runtime"), Mapping)
            and row["live_runtime"].get("live_exact_match") is True
            for row in dependency_profiles
        )
        and [str(row["profile_id"]) for row in dependency_profiles]
        == list(MOTION_REQUIRED_DEPENDENCY_PROFILES)
    )
    head_commit = (
        str(source_state.get("git_head_commit", ""))
        if isinstance(source_state, Mapping)
        else ""
    )
    if (
        payload.get("schema_version") != SCIENTIFIC_GATE_EVIDENCE_SCHEMA
        or payload.get("baseline_sha256") != baseline_hash
        or baseline.get("schema_version") != SCIENTIFIC_GATE_CAPTURE_SCHEMA
        or not str(baseline.get("pipeline_root", "")).strip()
        or not _is_sha256(baseline.get("source_snapshot_sha256"))
        or not isinstance(source_state, Mapping)
        or source_state.get("schema_version") != SCIENTIFIC_SOURCE_GATE_SCHEMA
        or source_state.get("v2_path") != MOTION_PIPELINE_RELATIVE_PATH.as_posix()
        or source_state.get("tracked_and_clean") is not True
        or source_state.get("v2_pyproject_tracked") is not True
        or source_state.get("status_entry_count") != 0
        or source_state.get("status_preview") != []
        or source_state.get("status_sha256")
        != hashlib.sha256(b"").hexdigest()
        or len(head_commit) != 40
        or any(character not in "0123456789abcdef" for character in head_commit)
        or not isinstance(gate_config, Mapping)
        or gate_config.get("schema_version")
        != MOTION_SCIENTIFIC_GATE_CONFIG_SCHEMA
        or gate_config.get("operation") != "motion_formal_reference"
        or gate_config.get("source_tree_policy")
        != "tracked_clean_final_pipeline_v2_no_override"
        or gate_config.get("source_snapshot_roots")
        != list(SCIENTIFIC_SOURCE_SNAPSHOT_ROOTS)
        or gate_config.get("required_dependency_profile_ids")
        != list(MOTION_REQUIRED_DEPENDENCY_PROFILES)
        or gate_config.get("require_exact_locks") is not True
        or gate_config.get("internal_checkpoints")
        != ["entry_preflight", "every_fit_pre", "every_fit_post"]
        or gate_config.get("ptt_checkpoints")
        != ["entry_preflight", "predict_pre", "predict_post"]
        or not _is_sha256(gate_config.get("config_sha256"))
        or not isinstance(dependency, Mapping)
        or dependency.get("schema_version") != "ppg_frailty.dependency_gate.v2"
        or dependency.get("operation") != "motion_formal_reference"
        or not str(dependency.get("config_id", "")).strip()
        or dependency.get("require_exact_lock") is not True
        or dependency.get("all_required_exact_locks_ready") is not True
        or dependency.get("required_profile_ids")
        != list(MOTION_REQUIRED_DEPENDENCY_PROFILES)
        or not profile_rows_valid
        or baseline.get("dependency_gate_sha256")
        != stable_payload_sha256(dependency)
    ):
        reasons.append("scientific_gate_baseline_semantic_drift")
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or [
        item.get("phase") if isinstance(item, Mapping) else None
        for item in checkpoints
    ] != list(expected_phases):
        reasons.append("scientific_gate_checkpoint_phase_drift")
    elif checkpoints:
        for item in checkpoints:
            record = dict(item)
            checkpoint_sha = record.pop("checkpoint_sha256", "")
            if (
                set(record)
                != {
                    "phase",
                    "gate_capture_sha256",
                    "source_snapshot_sha256",
                    "dependency_gate_sha256",
                }
                or stable_payload_sha256(record) != checkpoint_sha
                or record["gate_capture_sha256"] != baseline_hash
                or record["source_snapshot_sha256"]
                != baseline.get("source_snapshot_sha256")
                or record["dependency_gate_sha256"]
                != baseline.get("dependency_gate_sha256")
            ):
                reasons.append("scientific_gate_checkpoint_hash_drift")
                break
    return tuple(dict.fromkeys(reasons))


def require_archived_gate_matches_session(
    evidence: object,
    session: ScientificGateSession,
    *,
    expected_phases: Sequence[str],
) -> None:
    """Require archived training code/dependencies to equal the live PTT gate."""

    reasons = verify_scientific_gate_evidence(
        evidence,
        expected_phases=expected_phases,
    )
    if reasons:
        raise ScientificGateError(
            "archived_scientific_gate_evidence_rejected:" + ";".join(reasons)
        )
    archived = dict(evidence)
    if archived.get("baseline_sha256") != session.baseline_sha256:
        raise ScientificGateError(
            "internal_training_scientific_gate_differs_from_live_ptt_gate"
        )


__all__ = [
    "MOTION_INTERNAL_GATE_PHASES",
    "MOTION_PTT_GATE_PHASES",
    "MOTION_REQUIRED_DEPENDENCY_PROFILES",
    "MOTION_SCIENTIFIC_GATE_CONFIG_SCHEMA",
    "SCIENTIFIC_GATE_EVIDENCE_SCHEMA",
    "SCIENTIFIC_SOURCE_SNAPSHOT_ROOTS",
    "ScientificGateError",
    "ScientificGateSession",
    "capture_motion_scientific_gate",
    "establish_motion_scientific_gate",
    "git_source_state",
    "require_archived_gate_matches_session",
    "require_clean_tracked_source_state",
    "source_snapshot_sha256",
    "verify_scientific_gate_evidence",
]

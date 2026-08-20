"""Report-only seed and frozen-split reproducibility audit.

The audit follows the study manifest and each selected ``case_result`` artifact
root.  Superseded attempts are deliberately invisible.  It never gates
training or report generation: incomplete historical provenance is represented
as ``NOT_VERIFIABLE`` and contradictory evidence as ``FAIL``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PASS = "PASS"
FAIL = "FAIL"
NOT_VERIFIABLE = "NOT_VERIFIABLE"
_SHA256 = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class ReproducibilityAudit:
    """Five reusable report tables plus their aggregate status."""

    schema_version: str
    status: str
    summary: Mapping[str, Any]
    case_rows: tuple[Mapping[str, Any], ...]
    cell_rows: tuple[Mapping[str, Any], ...]
    split_rows: tuple[Mapping[str, Any], ...]
    issues: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON root is not a mapping: {path}")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ints(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    result = tuple(_int(item) for item in value)
    if any(item is None for item in result):
        return None
    return tuple(int(item) for item in result if item is not None)


def _sha(value: Any) -> str | None:
    text = str(value) if value is not None else ""
    return text if len(text) == 64 and set(text) <= _SHA256 else None


def _safe_child(root: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    relative = Path(raw)
    if relative.is_absolute():
        return None
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    return target


def _resolve_input(study_root: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    declared = Path(raw)
    if declared.is_absolute():
        return declared.resolve() if declared.is_file() else None
    for base in (study_root, *study_root.parents):
        candidate = (base / declared).resolve()
        if candidate.is_file():
            return candidate
    return None


def _status(issues: Sequence[Mapping[str, Any]]) -> str:
    if any(row.get("severity") == "error" for row in issues):
        return FAIL
    if any(row.get("severity") == "not_verifiable" for row in issues):
        return NOT_VERIFIABLE
    return PASS


def _load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = tuple(dict(row) for row in csv.DictReader(stream))
    required = {"repeat_index", "split_seed", "fold_index", "participant_id"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("split registry lacks repeat/fold/seed/participant columns")
    assignments: dict[tuple[int, int], set[str]] = {}
    participants: dict[int, set[str]] = {}
    seeds: dict[int, set[int]] = {}
    row_keys: set[tuple[int, int, str]] = set()
    for row in rows:
        repeat = int(row["repeat_index"])
        fold = int(row["fold_index"])
        seed = int(row["split_seed"])
        participant = str(row["participant_id"])
        key = (repeat, fold, participant)
        if not participant or key in row_keys:
            raise ValueError("split registry has an empty or duplicate assignment")
        row_keys.add(key)
        assignments.setdefault((repeat, fold), set()).add(participant)
        participants.setdefault(repeat, set()).add(participant)
        seeds.setdefault(repeat, set()).add(seed)
    if any(len(values) != 1 for values in seeds.values()):
        raise ValueError("split seed varies within a repeat")
    for repeat, roster in participants.items():
        observed = [
            participant
            for (row_repeat, _), values in assignments.items()
            if row_repeat == repeat
            for participant in values
        ]
        if len(observed) != len(roster):
            raise ValueError("participant is assigned to multiple folds in one repeat")
    metadata: dict[str, tuple[str, ...]] = {}
    for name in (
        "source_registry_id",
        "source_registry_file_sha256",
        "source_registry_payload_sha256",
        "dataset_version_id",
    ):
        metadata[name] = tuple(sorted({str(row.get(name, "")) for row in rows}))
    return {
        "path": path,
        "sha256": _file_hash(path),
        "assignments": assignments,
        "participants": participants,
        "seeds": {key: next(iter(value)) for key, value in seeds.items()},
        "metadata": metadata,
    }


def audit_study_reproducibility(collected: Any) -> ReproducibilityAudit:
    """Extract and compare current study seed/split evidence without gating.

    ``collected`` is intentionally duck-typed to ``CollectedStudy`` so this
    module remains independent of report collection and is easy to unit test.
    """

    root = Path(collected.root).resolve()
    manifest = _mapping(collected.manifest)
    plan = _mapping(collected.plan)
    execution = _mapping(manifest.get("execution")) or _mapping(plan.get("execution"))
    planned_repeats = _ints(execution.get("repeats"))
    planned_folds = _ints(execution.get("folds"))
    issues: list[dict[str, Any]] = []

    def issue(
        severity: str,
        code: str,
        message: str,
        *,
        case_id: str | None = None,
        repeat: int | None = None,
        fold: int | None = None,
    ) -> None:
        issues.append(
            {
                "severity": severity,
                "code": code,
                "case_id": case_id,
                "repeat": repeat,
                "fold": fold,
                "message": message,
            }
        )

    if planned_repeats is None or planned_folds is None:
        issue(
            "not_verifiable",
            "planned_cell_roster_unavailable",
            "study execution repeats/folds are not explicit integer lists",
        )
        planned_keys: set[tuple[int, int]] = set()
    else:
        planned_keys = {
            (repeat, fold)
            for repeat in planned_repeats
            for fold in planned_folds
        }

    manifest_cases = [
        row for row in manifest.get("cases", ()) if isinstance(row, Mapping)
    ]
    case_ids = [str(row.get("case_id", "")) for row in manifest_cases]
    if not case_ids or any(not value for value in case_ids):
        issue("error", "manifest_case_roster_invalid", "manifest has no valid cases")
    if len(case_ids) != len(set(case_ids)):
        issue("error", "manifest_case_duplicate", "manifest case IDs are not unique")

    records: dict[str, list[Mapping[str, Any]]] = {}
    for row in getattr(collected, "case_records", ()):
        if isinstance(row, Mapping):
            records.setdefault(str(row.get("case_id", "")), []).append(row)
    compact_cells: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for row in getattr(collected, "cell_rows", ()):
        if not isinstance(row, Mapping):
            continue
        repeat, fold = _int(row.get("repeat")), _int(row.get("fold"))
        if repeat is not None and fold is not None:
            compact_cells.setdefault(
                (str(row.get("case_id", "")), repeat, fold), []
            ).append(row)

    registry_cache: dict[Path, dict[str, Any] | Exception] = {}
    cell_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    split_by_identity: dict[str, dict[str, Any]] = {}

    for case in manifest_cases:
        case_id = str(case.get("case_id", ""))
        case_issue_start = len(issues)
        matching_records = records.get(case_id, [])
        if len(matching_records) != 1:
            issue(
                "error",
                "selected_case_record_count_invalid",
                f"expected one selected/current case record, observed {len(matching_records)}",
                case_id=case_id,
            )
            record: Mapping[str, Any] = matching_records[0] if matching_records else {}
        else:
            record = matching_records[0]

        case_dir = _safe_child(root, case.get("case_directory"))
        config_path = _safe_child(root, case.get("resolved_config_path"))
        config: Mapping[str, Any] = {}
        if config_path is None or not config_path.is_file():
            issue(
                "not_verifiable",
                "resolved_config_unavailable",
                "selected case resolved config is unavailable",
                case_id=case_id,
            )
        else:
            try:
                loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, Mapping):
                    raise TypeError("root is not a mapping")
                config = loaded
            except Exception as error:  # noqa: BLE001 - report evidence gap.
                issue(
                    "not_verifiable",
                    "resolved_config_unreadable",
                    f"{type(error).__name__}: {error}",
                    case_id=case_id,
                )

        split_config = _mapping(config.get("splits"))
        model_config = _mapping(config.get("model"))
        training_config = _mapping(config.get("training"))
        evaluation_config = _mapping(config.get("evaluation"))
        statistics_config = _mapping(evaluation_config.get("statistics"))
        registry_path = _resolve_input(root, split_config.get("path"))
        registry: Mapping[str, Any] = {}
        if registry_path is None:
            issue(
                "not_verifiable",
                "split_registry_unavailable",
                f"cannot resolve declared split registry {split_config.get('path')!r}",
                case_id=case_id,
            )
        else:
            if registry_path not in registry_cache:
                try:
                    registry_cache[registry_path] = _load_registry(registry_path)
                except Exception as error:  # noqa: BLE001
                    registry_cache[registry_path] = error
            cached = registry_cache[registry_path]
            if isinstance(cached, Exception):
                issue(
                    "not_verifiable",
                    "split_registry_unreadable",
                    f"{type(cached).__name__}: {cached}",
                    case_id=case_id,
                )
            else:
                registry = cached

        manifest_config = _mapping(config.get("manifest"))
        data_manifest_path = _resolve_input(root, manifest_config.get("path"))
        data_manifest_sha = _file_hash(data_manifest_path) if data_manifest_path else None
        if data_manifest_path is None:
            issue(
                "not_verifiable",
                "data_manifest_unavailable",
                f"cannot resolve declared data manifest {manifest_config.get('path')!r}",
                case_id=case_id,
            )

        if registry:
            declared_registry = str(split_config.get("registry_id", ""))
            observed_registry = registry["metadata"]["source_registry_id"]
            if declared_registry and observed_registry != (declared_registry,):
                issue(
                    "error",
                    "split_registry_id_drift",
                    f"declared {declared_registry!r}, observed {observed_registry!r}",
                    case_id=case_id,
                )
            for config_name, metadata_name in (
                ("source_registry_file_sha256", "source_registry_file_sha256"),
                ("source_registry_payload_sha256", "source_registry_payload_sha256"),
            ):
                declared = str(split_config.get(config_name, ""))
                observed = registry["metadata"][metadata_name]
                if declared and observed != (declared,):
                    issue(
                        "error",
                        "split_registry_authority_hash_drift",
                        f"{config_name}: declared {declared!r}, observed {observed!r}",
                        case_id=case_id,
                    )
            declared_seeds = _ints(split_config.get("split_seeds"))
            if declared_seeds is not None:
                for repeat, observed in registry["seeds"].items():
                    if repeat >= len(declared_seeds) or declared_seeds[repeat] != observed:
                        issue(
                            "error",
                            "declared_split_seed_drift",
                            f"repeat {repeat}: declared sequence disagrees with registry seed {observed}",
                            case_id=case_id,
                            repeat=repeat,
                        )

        artifact_root = (
            _safe_child(case_dir, record.get("artifact_root"))
            if case_dir is not None
            else None
        )
        run_cells: dict[tuple[int, int], list[tuple[Mapping[str, Any], Path]]] = {}
        if artifact_root is None or not artifact_root.is_dir():
            issue(
                "not_verifiable",
                "selected_artifact_root_unavailable",
                "selected/current artifact_root is unavailable",
                case_id=case_id,
            )
        else:
            for path in sorted(artifact_root.rglob("run_manifest.json")):
                try:
                    cell = _mapping(_json(path).get("cell"))
                except Exception as error:  # noqa: BLE001
                    issue(
                        "not_verifiable",
                        "cell_manifest_unreadable",
                        f"{path.relative_to(artifact_root)}: {type(error).__name__}: {error}",
                        case_id=case_id,
                    )
                    continue
                repeat, fold = _int(cell.get("repeat_index")), _int(cell.get("fold_index"))
                if repeat is not None and fold is not None:
                    run_cells.setdefault((repeat, fold), []).append((cell, path))

        observed_keys = set(run_cells)
        observed_keys.update(
            (repeat, fold)
            for cell_case, repeat, fold in compact_cells
            if cell_case == case_id
        )
        if planned_keys:
            for repeat, fold in sorted(planned_keys - observed_keys):
                issue(
                    "error",
                    "planned_cell_missing",
                    "planned repeat/fold cell is missing from selected artifacts",
                    case_id=case_id,
                    repeat=repeat,
                    fold=fold,
                )
            for repeat, fold in sorted(observed_keys - planned_keys):
                issue(
                    "error",
                    "unplanned_cell_present",
                    "selected artifacts contain an unplanned repeat/fold cell",
                    case_id=case_id,
                    repeat=repeat,
                    fold=fold,
                )

        for repeat, fold in sorted(observed_keys | planned_keys):
            cell_issue_start = len(issues)
            compact_matches = compact_cells.get((case_id, repeat, fold), [])
            manifest_matches = run_cells.get((repeat, fold), [])
            if len(compact_matches) > 1 or len(manifest_matches) > 1:
                issue(
                    "error",
                    "duplicate_cell_evidence",
                    "selected artifact root has duplicate evidence for one repeat/fold",
                    case_id=case_id,
                    repeat=repeat,
                    fold=fold,
                )
            compact = compact_matches[0] if compact_matches else {}
            cell = manifest_matches[0][0] if manifest_matches else {}
            cell_path = manifest_matches[0][1].parent if manifest_matches else None
            if not cell:
                issue(
                    "not_verifiable",
                    "cell_run_manifest_missing",
                    "cell has no detailed run_manifest evidence",
                    case_id=case_id,
                    repeat=repeat,
                    fold=fold,
                )

            def agree(name: str, *values: Any) -> Any:
                present = [value for value in values if value is not None]
                identities = {json.dumps(value, sort_keys=True, default=str) for value in present}
                if len(identities) > 1:
                    issue(
                        "error",
                        "cell_evidence_drift",
                        f"{name} disagrees across selected cell artifacts: {present!r}",
                        case_id=case_id,
                        repeat=repeat,
                        fold=fold,
                    )
                return present[0] if present else None

            split_seed = _int(agree("split_seed", cell.get("split_seed"), compact.get("split_seed")))
            training_seed = agree("training_seed", cell.get("training_seed"), compact.get("training_seed"))
            training_seed = _int(training_seed) if training_seed is not None else None
            orchestration_seed = _int(cell.get("training_orchestration_seed"))
            seed_policy = str(cell.get("seed_policy", ""))
            member_seeds = _ints(cell.get("member_training_seeds"))
            member_seeds = member_seeds if member_seeds is not None else ()
            fitted = _mapping(cell.get("fitted_provenance"))
            frozen = _mapping(cell.get("frozen_model_run_provenance"))
            random_seeds = _ints(frozen.get("random_seeds"))
            random_seeds = random_seeds if random_seeds is not None else ()
            frozen_policy = str(frozen.get("seed_policy", ""))
            agree("seed_policy", seed_policy or None, frozen_policy or None)
            agree("member_training_seeds", list(member_seeds), fitted.get("member_training_seeds"))

            experiment_provenance: Mapping[str, Any] = {}
            if cell_path is not None and (cell_path / "experiment_result.json").is_file():
                try:
                    experiment_provenance = _mapping(
                        _json(cell_path / "experiment_result.json").get("provenance")
                    )
                except Exception as error:  # noqa: BLE001
                    issue(
                        "not_verifiable",
                        "experiment_provenance_unreadable",
                        f"{type(error).__name__}: {error}",
                        case_id=case_id,
                        repeat=repeat,
                        fold=fold,
                    )
            manifest_hash = agree(
                "manifest_hash",
                _sha(experiment_provenance.get("manifest_hash")),
                _sha(experiment_provenance.get("manifest_sha256")),
                _sha(_mapping(cell.get("legacy_bridge")).get("manifest_sha256")),
            )
            fold_hash = agree(
                "fold_hash",
                _sha(experiment_provenance.get("fold_hash")),
                _sha(experiment_provenance.get("split_sha256")),
                _sha(fitted.get("fold_hash")),
                _sha(frozen.get("fold_hash")),
                _sha(frozen.get("split_sha256")),
            )
            registry_hash = agree(
                "registry_hash",
                _sha(fitted.get("registry_hash")),
                _sha(split_config.get("source_registry_payload_sha256")),
            )
            if data_manifest_sha and manifest_hash and data_manifest_sha != manifest_hash:
                issue(
                    "error",
                    "data_manifest_hash_drift",
                    f"cell {manifest_hash} != current declared file {data_manifest_sha}",
                    case_id=case_id,
                    repeat=repeat,
                    fold=fold,
                )
            if registry and fold_hash and registry["sha256"] != fold_hash:
                issue(
                    "error",
                    "split_file_hash_drift",
                    f"cell {fold_hash} != current registry file {registry['sha256']}",
                    case_id=case_id,
                    repeat=repeat,
                    fold=fold,
                )

            expected_train: tuple[str, ...] = ()
            expected_oof: tuple[str, ...] = ()
            registry_seed: int | None = None
            if registry and repeat in registry["participants"]:
                expected_oof = tuple(sorted(registry["assignments"].get((repeat, fold), ())))
                expected_train = tuple(
                    sorted(registry["participants"][repeat] - set(expected_oof))
                )
                registry_seed = int(registry["seeds"][repeat])
                if not expected_train or not expected_oof:
                    issue(
                        "error",
                        "split_roster_empty",
                        "registry does not define non-empty train and OOF rosters",
                        case_id=case_id,
                        repeat=repeat,
                        fold=fold,
                    )
                if split_seed is not None and split_seed != registry_seed:
                    issue(
                        "error",
                        "cell_split_seed_drift",
                        f"cell {split_seed} != registry {registry_seed}",
                        case_id=case_id,
                        repeat=repeat,
                        fold=fold,
                    )
            else:
                issue(
                    "not_verifiable",
                    "cell_split_roster_unavailable",
                    "cannot derive cell membership from the declared registry",
                    case_id=case_id,
                    repeat=repeat,
                    fold=fold,
                )

            fitted_ids = tuple(sorted(str(value) for value in fitted.get("fitted_participant_ids", ())))
            if expected_train and fitted_ids and fitted_ids != expected_train:
                issue(
                    "error",
                    "outer_train_roster_drift",
                    "fitted participant roster differs from frozen registry train roster",
                    case_id=case_id,
                    repeat=repeat,
                    fold=fold,
                )
            elif expected_train and not fitted_ids:
                issue(
                    "not_verifiable",
                    "fitted_train_roster_unavailable",
                    "cell does not persist fitted participant IDs",
                    case_id=case_id,
                    repeat=repeat,
                    fold=fold,
                )
            overlap = len(set(fitted_ids) & set(expected_oof))
            if overlap:
                issue(
                    "error",
                    "outer_train_oof_overlap",
                    f"{overlap} fitted participants occur in outer OOF",
                    case_id=case_id,
                    repeat=repeat,
                    fold=fold,
                )

            outer_membership_hash = _sha(fitted.get("outer_membership_hash"))
            dataset_hash = _sha(fitted.get("dataset_binding_hash"))
            if (
                outer_membership_hash
                and expected_train
                and expected_oof
                and split_seed is not None
                and registry_hash
                and fold_hash
                and dataset_hash
            ):
                computed_membership_hash = _stable_hash(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "seed": split_seed,
                        "train": expected_train,
                        "oof": expected_oof,
                        "registry_hash": registry_hash,
                        "fold_hash": fold_hash,
                        "train_dataset_hash": dataset_hash,
                    }
                )
                if computed_membership_hash != outer_membership_hash:
                    issue(
                        "error",
                        "outer_membership_hash_drift",
                        "persisted outer membership hash cannot be reproduced",
                        case_id=case_id,
                        repeat=repeat,
                        fold=fold,
                    )

            statistics_seed = _int(_mapping(cell.get("evaluation_policy")).get("statistics", {}).get("seed"))
            configured_statistics_seed = _int(statistics_config.get("seed"))
            statistics_seed = _int(
                agree("evaluation_statistics_seed", statistics_seed, configured_statistics_seed)
            )

            if not seed_policy:
                issue(
                    "not_verifiable",
                    "seed_policy_unavailable",
                    "cell does not persist model seed policy",
                    case_id=case_id,
                    repeat=repeat,
                    fold=fold,
                )
            elif seed_policy in {"outer_repeat", "outer_cv_repeat_seed_equals_split_seed"}:
                if split_seed is not None and (
                    training_seed != split_seed
                    or (orchestration_seed is not None and orchestration_seed != split_seed)
                    or random_seeds != (split_seed,)
                ):
                    issue(
                        "error",
                        "repeat_seed_policy_drift",
                        "repeat-local model/training seeds do not equal this repeat's split seed",
                        case_id=case_id,
                        repeat=repeat,
                        fold=fold,
                    )
            elif seed_policy == "legacy_bridge_fixed_training_seed_42":
                if training_seed != 42 or orchestration_seed != 42 or random_seeds != (42,):
                    issue(
                        "error",
                        "legacy_fixed_seed_policy_drift",
                        "legacy bridge requires training/orchestration/model seed 42",
                        case_id=case_id,
                        repeat=repeat,
                        fold=fold,
                    )
            elif seed_policy == "cv_fixed_member0_seed_50042_comparator":
                if training_seed != 50042 or random_seeds != (50042,):
                    issue(
                        "error",
                        "member0_comparator_seed_policy_drift",
                        "member-0 comparator requires model seed 50042",
                        case_id=case_id,
                        repeat=repeat,
                        fold=fold,
                    )
            elif seed_policy in {"member_roster", "cv_fixed_five_member_seed_roster"}:
                configured_members = _ints(model_config.get("member_seeds")) or ()
                if (
                    training_seed is not None
                    or not member_seeds
                    or random_seeds != member_seeds
                    or (configured_members and configured_members != member_seeds)
                ):
                    issue(
                        "error",
                        "ensemble_member_seed_policy_drift",
                        "ensemble average/member/frozen seed rosters disagree",
                        case_id=case_id,
                        repeat=repeat,
                        fold=fold,
                    )
            elif seed_policy in {"fixed", "fixed_explicit", "final_refit_single_seed_42"}:
                configured_seed = _int(model_config.get("seed"))
                if configured_seed is None:
                    configured_seed = _int(training_config.get("seed"))
                if (
                    configured_seed is None
                    or training_seed != configured_seed
                    or random_seeds != (configured_seed,)
                ):
                    issue(
                        "error",
                        "fixed_seed_policy_drift",
                        "fixed model seed disagrees with resolved config or frozen provenance",
                        case_id=case_id,
                        repeat=repeat,
                        fold=fold,
                    )
            else:
                issue(
                    "not_verifiable",
                    "seed_policy_unknown",
                    f"unregistered seed policy {seed_policy!r}",
                    case_id=case_id,
                    repeat=repeat,
                    fold=fold,
                )
            if orchestration_seed is None:
                issue(
                    "not_verifiable",
                    "training_orchestration_seed_unavailable",
                    "historical cell lacks explicit orchestration seed",
                    case_id=case_id,
                    repeat=repeat,
                    fold=fold,
                )

            epoch_records: list[dict[str, int]] = []
            selected_epoch = _int(fitted.get("selected_epoch"))
            if cell_path is not None and (cell_path / "training_history.json").is_file():
                try:
                    history = _json(cell_path / "training_history.json")
                    rows = history.get("rows", ())
                    for raw in rows if isinstance(rows, list) else ():
                        if not isinstance(raw, Mapping) or raw.get("epoch_rng_seed") is None:
                            continue
                        epoch = _int(raw.get("epoch"))
                        member = _int(raw.get("member"))
                        base_seed = _int(raw.get("training_seed"))
                        epoch_seed = _int(raw.get("epoch_rng_seed"))
                        numpy_seed = _int(raw.get("numpy_epoch_rng_seed"))
                        if None in (epoch, member, base_seed, epoch_seed, numpy_seed):
                            continue
                        epoch_records.append(
                            {
                                "member": int(member),
                                "epoch": int(epoch),
                                "training_seed": int(base_seed),
                                "epoch_rng_seed": int(epoch_seed),
                                "numpy_epoch_rng_seed": int(numpy_seed),
                            }
                        )
                except Exception as error:  # noqa: BLE001
                    issue(
                        "not_verifiable",
                        "epoch_seed_history_unreadable",
                        f"{type(error).__name__}: {error}",
                        case_id=case_id,
                        repeat=repeat,
                        fold=fold,
                    )
            expected_model_seeds = member_seeds or ((training_seed,) if training_seed is not None else ())
            if selected_epoch is not None and selected_epoch > 0:
                if not epoch_records:
                    issue(
                        "not_verifiable",
                        "epoch_seed_roster_unavailable",
                        "iterative model has no persisted epoch RNG roster",
                        case_id=case_id,
                        repeat=repeat,
                        fold=fold,
                    )
                else:
                    observed_epoch_keys: set[tuple[int, int]] = set()
                    for row in epoch_records:
                        key = (row["member"], row["epoch"])
                        expected_seed = (
                            expected_model_seeds[row["member"]]
                            if 0 <= row["member"] < len(expected_model_seeds)
                            else None
                        )
                        if (
                            key in observed_epoch_keys
                            or expected_seed is None
                            or row["training_seed"] != expected_seed
                            or row["epoch_rng_seed"] != expected_seed + row["epoch"] * 1_000_000
                            or row["numpy_epoch_rng_seed"] != row["epoch_rng_seed"] % (1 << 32)
                        ):
                            issue(
                                "error",
                                "epoch_seed_derivation_drift",
                                "epoch/member RNG evidence is duplicate or violates the declared derivation",
                                case_id=case_id,
                                repeat=repeat,
                                fold=fold,
                            )
                            break
                        observed_epoch_keys.add(key)
                    expected_epoch_keys = {
                        (member, epoch)
                        for member in range(len(expected_model_seeds))
                        for epoch in range(1, selected_epoch + 1)
                    }
                    if observed_epoch_keys != expected_epoch_keys:
                        issue(
                            "error",
                            "epoch_seed_roster_incomplete",
                            "epoch RNG roster does not cover every selected member/epoch",
                            case_id=case_id,
                            repeat=repeat,
                            fold=fold,
                        )

            train_hash = _stable_hash(expected_train) if expected_train else None
            oof_hash = _stable_hash(expected_oof) if expected_oof else None
            split_identity = (
                _stable_hash(
                    {
                        "registry_file_sha256": registry.get("sha256"),
                        "repeat": repeat,
                        "fold": fold,
                        "split_seed": registry_seed,
                        "train_participant_ids": expected_train,
                        "oof_participant_ids": expected_oof,
                    }
                )
                if expected_train and expected_oof
                else None
            )
            if split_identity is not None:
                split = split_by_identity.setdefault(
                    split_identity,
                    {
                        "split_identity_sha256": split_identity,
                        "split_registry_path": split_config.get("path"),
                        "split_registry_resolved_path": str(registry_path),
                        "split_registry_id": split_config.get("registry_id"),
                        "split_registry_file_sha256": registry.get("sha256"),
                        "source_registry_file_sha256": split_config.get("source_registry_file_sha256"),
                        "source_registry_payload_sha256": split_config.get("source_registry_payload_sha256"),
                        "repeat": repeat,
                        "fold": fold,
                        "split_seed": registry_seed,
                        "train_participant_count": len(expected_train),
                        "oof_participant_count": len(expected_oof),
                        "train_oof_overlap_count": len(set(expected_train) & set(expected_oof)),
                        "train_participant_roster_sha256": train_hash,
                        "oof_participant_roster_sha256": oof_hash,
                        "train_participant_ids": list(expected_train),
                        "oof_participant_ids": list(expected_oof),
                        "case_ids": [],
                    },
                )
                split["case_ids"].append(case_id)

            cell_specific_issues = issues[cell_issue_start:]
            cell_rows.append(
                {
                    "case_id": case_id,
                    "repeat": repeat,
                    "fold": fold,
                    "selected_attempt": record.get("attempt"),
                    "status": cell.get("status", compact.get("status")),
                    "model_id": cell.get("model_machine_id", compact.get("model_id")),
                    "seed_policy": seed_policy or None,
                    "split_seed": split_seed,
                    "training_orchestration_seed": orchestration_seed,
                    "training_seed": training_seed,
                    "member_training_seeds": list(member_seeds),
                    "frozen_random_seeds": list(random_seeds),
                    "evaluation_statistics_seed": statistics_seed,
                    "epoch_rng_seed_count": len(epoch_records),
                    "epoch_rng_seed_roster": sorted({row["epoch_rng_seed"] for row in epoch_records}),
                    "epoch_rng_roster_sha256": _stable_hash(epoch_records) if epoch_records else None,
                    "data_manifest_path": manifest_config.get("path"),
                    "data_manifest_sha256": manifest_hash,
                    "split_registry_path": split_config.get("path"),
                    "split_registry_id": split_config.get("registry_id"),
                    "split_registry_file_sha256": fold_hash,
                    "source_registry_payload_sha256": registry_hash,
                    "split_identity_sha256": split_identity,
                    "train_participant_count": len(expected_train) or None,
                    "oof_participant_count": len(expected_oof) or None,
                    "train_oof_overlap_count": overlap if expected_oof else None,
                    "train_participant_roster_sha256": train_hash,
                    "oof_participant_roster_sha256": oof_hash,
                    "outer_membership_sha256": outer_membership_hash,
                    "evidence_path": (
                        str(manifest_matches[0][1].relative_to(root))
                        if manifest_matches
                        else None
                    ),
                    "audit_status": _status(cell_specific_issues),
                }
            )

        current_case_cells = [row for row in cell_rows if row["case_id"] == case_id]
        case_specific_issues = issues[case_issue_start:]
        case_rows.append(
            {
                "case_id": case_id,
                "selected_case_status": record.get("status"),
                "selected_attempt": record.get("attempt"),
                "selected_artifact_root": record.get("artifact_root"),
                "planned_cell_count": len(planned_keys) if planned_keys else None,
                "observed_cell_count": len(observed_keys),
                "seed_policies": sorted({str(row["seed_policy"]) for row in current_case_cells if row["seed_policy"]}),
                "split_seeds": sorted({int(row["split_seed"]) for row in current_case_cells if row["split_seed"] is not None}),
                "training_orchestration_seeds": sorted({int(row["training_orchestration_seed"]) for row in current_case_cells if row["training_orchestration_seed"] is not None}),
                "training_seeds": sorted({int(row["training_seed"]) for row in current_case_cells if row["training_seed"] is not None}),
                "member_training_seeds": sorted({seed for row in current_case_cells for seed in row["member_training_seeds"]}),
                "evaluation_statistics_seeds": sorted({int(row["evaluation_statistics_seed"]) for row in current_case_cells if row["evaluation_statistics_seed"] is not None}),
                "data_manifest_sha256": sorted({str(row["data_manifest_sha256"]) for row in current_case_cells if row["data_manifest_sha256"]}),
                "split_registry_file_sha256": sorted({str(row["split_registry_file_sha256"]) for row in current_case_cells if row["split_registry_file_sha256"]}),
                "split_identity_sha256": sorted({str(row["split_identity_sha256"]) for row in current_case_cells if row["split_identity_sha256"]}),
                "audit_status": _status(case_specific_issues),
            }
        )

    for repeat, fold in sorted({(row["repeat"], row["fold"]) for row in cell_rows}):
        peers = [row for row in cell_rows if row["repeat"] == repeat and row["fold"] == fold]
        for field in (
            "split_seed",
            "data_manifest_sha256",
            "split_registry_file_sha256",
            "source_registry_payload_sha256",
            "split_identity_sha256",
        ):
            values = {json.dumps(row[field], sort_keys=True) for row in peers if row[field] is not None}
            if len(values) > 1:
                issue(
                    "error",
                    "cross_case_split_drift",
                    f"{field} differs across cases for the same repeat/fold",
                    repeat=repeat,
                    fold=fold,
                )

    final_status = _status(issues)
    for row in split_by_identity.values():
        row["case_ids"] = sorted(set(row["case_ids"]))
        row["audit_status"] = PASS if row["train_oof_overlap_count"] == 0 else FAIL
    summary = {
        "audit_status": final_status,
        "planned_case_count": len(manifest_cases),
        "selected_case_record_count": sum(len(records.get(case_id, ())) == 1 for case_id in case_ids),
        "planned_repeats": list(planned_repeats) if planned_repeats is not None else None,
        "planned_folds": list(planned_folds) if planned_folds is not None else None,
        "planned_cell_count": len(manifest_cases) * len(planned_keys) if planned_keys else None,
        "observed_cell_count": len(cell_rows),
        "split_seed_by_repeat": {
            str(repeat): sorted({row["split_seed"] for row in cell_rows if row["repeat"] == repeat and row["split_seed"] is not None})
            for repeat in sorted({row["repeat"] for row in cell_rows})
        },
        "seed_policies": sorted({str(row["seed_policy"]) for row in cell_rows if row["seed_policy"]}),
        "evaluation_statistics_seeds": sorted({int(row["evaluation_statistics_seed"]) for row in cell_rows if row["evaluation_statistics_seed"] is not None}),
        "unique_data_manifest_hashes": sorted({str(row["data_manifest_sha256"]) for row in cell_rows if row["data_manifest_sha256"]}),
        "unique_split_registry_file_hashes": sorted({str(row["split_registry_file_sha256"]) for row in cell_rows if row["split_registry_file_sha256"]}),
        "unique_split_assignment_count": len(split_by_identity),
        "error_count": sum(row["severity"] == "error" for row in issues),
        "not_verifiable_count": sum(row["severity"] == "not_verifiable" for row in issues),
        "scope": "study_manifest_cases_and_selected_case_result_artifact_roots_only",
        "training_or_report_gate": False,
    }
    return ReproducibilityAudit(
        schema_version="ppg_frailty.reporting.reproducibility_audit.v1",
        status=final_status,
        summary=summary,
        case_rows=tuple(case_rows),
        cell_rows=tuple(cell_rows),
        split_rows=tuple(
            sorted(
                split_by_identity.values(),
                key=lambda row: (row["repeat"], row["fold"], row["split_identity_sha256"]),
            )
        ),
        issues=tuple(issues),
    )


__all__ = [
    "FAIL",
    "NOT_VERIFIABLE",
    "PASS",
    "ReproducibilityAudit",
    "audit_study_reproducibility",
]

"""Report-only audit of selected study seeds and frozen data splits."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

PASS, FAIL, NOT_VERIFIABLE = "PASS", "FAIL", "NOT_VERIFIABLE"

@dataclass(frozen=True)
class ReproducibilityAudit:
    """Reusable summary/case/cell/split/issues report payload."""

    schema_version: str
    status: str
    summary: Mapping[str, Any]
    case_rows: tuple[Mapping[str, Any], ...]
    cell_rows: tuple[Mapping[str, Any], ...]
    split_rows: tuple[Mapping[str, Any], ...]
    issues: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(raw).hexdigest()

def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}

def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _integers(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    result = tuple(_integer(item) for item in value)
    return None if any(item is None for item in result) else tuple(result)  # type: ignore[arg-type]

def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("JSON root is not a mapping")
    return value

def _child(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path

def _input(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    if raw.is_absolute():
        return raw if raw.is_file() else None
    for base in (root, *root.parents):
        path = (base / raw).resolve()
        if path.is_file():
            return path
    return None

def _registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"repeat_index", "fold_index", "split_seed", "participant_id"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("registry columns/rows unavailable")
    by_fold: dict[tuple[int, int], set[str]] = {}
    by_repeat: dict[int, set[str]] = {}
    seeds: dict[int, set[int]] = {}
    for row in rows:
        repeat, fold, seed = map(
            int, (row["repeat_index"], row["fold_index"], row["split_seed"])
        )
        participant = row["participant_id"]
        if not participant or participant in by_fold.setdefault((repeat, fold), set()):
            raise ValueError("empty/duplicate participant assignment")
        by_fold[(repeat, fold)].add(participant)
        by_repeat.setdefault(repeat, set()).add(participant)
        seeds.setdefault(repeat, set()).add(seed)
    if any(len(values) != 1 for values in seeds.values()):
        raise ValueError("split seed varies within repeat")
    if any(
        sum(participant in roster for (r, _), roster in by_fold.items() if r == repeat) != 1
        for repeat, cohort in by_repeat.items()
        for participant in cohort
    ):
        raise ValueError("participant assigned to multiple folds")
    return {
        "sha256": _file_hash(path),
        "by_fold": by_fold,
        "by_repeat": by_repeat,
        "seeds": {key: next(iter(values)) for key, values in seeds.items()},
        "metadata": {
            name: tuple(sorted({str(row.get(name, "")) for row in rows}))
            for name in ("source_registry_id", "source_registry_file_sha256",
                         "source_registry_payload_sha256")
            if name in rows[0]
        },
    }

def _overall(issues: Sequence[Mapping[str, Any]]) -> str:
    if any(row["severity"] == "error" for row in issues):
        return FAIL
    return NOT_VERIFIABLE if issues else PASS

def audit_study_reproducibility(collected: Any) -> ReproducibilityAudit:
    """Audit manifest cases and current artifact roots without becoming a gate."""

    root, issues = Path(collected.root).resolve(), []
    def add(
        code: str,
        message: str,
        *,
        severity: str = "error",
        case_id: str | None = None,
        repeat: int | None = None,
        fold: int | None = None,
    ) -> None:
        issues.append(
            dict(
                severity=severity, code=code, case_id=case_id,
                repeat=repeat, fold=fold, message=message,
            )
        )
    manifest, plan = _map(collected.manifest), _map(collected.plan)
    execution = _map(manifest.get("execution")) or _map(plan.get("execution"))
    repeats = _integers(execution.get("repeats"))
    folds = _integers(execution.get("folds"))
    planned = (
        {(repeat, fold) for repeat in repeats for fold in folds}
        if repeats is not None and folds is not None else set()
    )
    if not planned:
        add("plan_roster_unavailable", "execution repeats/folds are not explicit",
            severity="not_verifiable")
    cases = [row for row in manifest.get("cases", ()) if isinstance(row, Mapping)]
    case_ids = [str(row.get("case_id", "")) for row in cases]
    if not cases or any(not case_id for case_id in case_ids):
        add("manifest_case_roster_invalid", "manifest has no valid case roster")
    elif len(case_ids) != len(set(case_ids)):
        add("manifest_case_roster_invalid", "manifest case IDs are not unique")
    records: dict[str, list[Mapping[str, Any]]] = {}
    for row in collected.case_records:
        records.setdefault(str(row.get("case_id")), []).append(row)
    compact: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for row in collected.cell_rows:
        repeat, fold = _integer(row.get("repeat")), _integer(row.get("fold"))
        if repeat is not None and fold is not None:
            compact.setdefault((str(row.get("case_id")), repeat, fold), []).append(row)
    oof: dict[tuple[str, int, int], set[str]] = {}
    for row in collected.subject_oof_rows:
        repeat, fold = _integer(row.get("repeat")), _integer(row.get("fold"))
        participant = row.get("participant_id")
        if repeat is not None and fold is not None and participant is not None:
            oof.setdefault((str(row.get("case_id")), repeat, fold), set()).add(str(participant))
    history: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for row in collected.history_rows:
        repeat, fold = _integer(row.get("repeat")), _integer(row.get("fold"))
        if repeat is not None and fold is not None:
            history.setdefault((str(row.get("case_id")), repeat, fold), []).append(row)
    registry_cache: dict[Path, Mapping[str, Any] | Exception] = {}
    cells: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    splits: dict[str, dict[str, Any]] = {}
    observed_cell_count = 0
    for case in cases:
        case_id = str(case.get("case_id"))
        selected = records.get(case_id, [])
        if len(selected) != 1:
            add("selected_case_record_invalid", f"expected one current record, found {len(selected)}",
                case_id=case_id)
        record = selected[0] if selected else {}
        if record and record.get("status") != "passed":
            add("selected_case_not_passed", f"selected status={record.get('status')!r}",
                severity="not_verifiable", case_id=case_id)
        case_dir = _child(root, case.get("case_directory"))
        config_path = _child(root, case.get("resolved_config_path"))
        config: Mapping[str, Any] = {}
        try:
            value = yaml.safe_load(config_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
            config = value if isinstance(value, Mapping) else {}
        except Exception as error:  # noqa: BLE001
            add("resolved_config_unavailable", str(error), severity="not_verifiable",
                case_id=case_id)
        split_cfg = _map(config.get("splits"))
        manifest_cfg = _map(config.get("manifest"))
        model_cfg = _map(config.get("model"))
        training_cfg = _map(config.get("training"))
        statistics_cfg = _map(_map(config.get("evaluation")).get("statistics"))
        registry_path, registry = _input(root, split_cfg.get("path")), {}
        if registry_path is not None:
            if registry_path not in registry_cache:
                try:
                    registry_cache[registry_path] = _registry(registry_path)
                except Exception as error:  # noqa: BLE001
                    registry_cache[registry_path] = error
            cached = registry_cache[registry_path]
            if isinstance(cached, Exception):
                add("split_registry_unreadable", str(cached), severity="not_verifiable",
                    case_id=case_id)
            else:
                registry = cached
                for repeat, fold in sorted(planned - set(registry.get("by_fold", {}))):
                    add("split_registry_planned_roster_missing",
                        "materialized split CSV lacks a planned repeat/fold roster",
                        case_id=case_id, repeat=repeat, fold=fold)
                metadata = _map(registry.get("metadata"))
                for config_key, csv_key in (
                    ("registry_id", "source_registry_id"),
                    ("source_registry_file_sha256", "source_registry_file_sha256"),
                    ("source_registry_payload_sha256", "source_registry_payload_sha256"),
                ):
                    declared, observed = split_cfg.get(config_key), metadata.get(csv_key)
                    if declared and observed and observed != (str(declared),):
                        add("split_registry_authority_drift",
                            f"{config_key} disagrees with materialized split CSV",
                            case_id=case_id)
        else:
            add("split_registry_unavailable", str(split_cfg.get("path")),
                severity="not_verifiable", case_id=case_id)
        manifest_path = _input(root, manifest_cfg.get("path"))
        current_manifest_hash = _file_hash(manifest_path) if manifest_path else None
        artifact = _child(case_dir, record.get("artifact_root")) if case_dir else None
        runtime: dict[tuple[int, int], list[tuple[Mapping[str, Any], Path]]] = {}
        if artifact and artifact.is_dir():
            for path in artifact.rglob("run_manifest.json"):
                try:
                    cell = _map(_json(path).get("cell"))
                    key = (_integer(cell.get("repeat_index")), _integer(cell.get("fold_index")))
                    if None not in key:
                        runtime.setdefault(key, []).append((cell, path))  # type: ignore[arg-type]
                except Exception as error:  # noqa: BLE001
                    add("run_manifest_unreadable", str(error), severity="not_verifiable",
                        case_id=case_id)
        else:
            add("selected_artifact_unavailable", str(record.get("artifact_root")),
                severity="not_verifiable", case_id=case_id)
        observed = set(runtime) | {
            (repeat, fold) for cid, repeat, fold in compact if cid == case_id
        }
        observed_cell_count += len(observed)
        for repeat, fold in sorted(planned - observed):
            add("planned_cell_missing", "selected cell missing", case_id=case_id,
                repeat=repeat, fold=fold)
        for repeat, fold in sorted(observed - planned) if planned else ():
            add("unplanned_cell", "unexpected selected cell", case_id=case_id,
                repeat=repeat, fold=fold)
        for repeat, fold in sorted(observed | planned):
            scope = dict(case_id=case_id, repeat=repeat, fold=fold)
            run = runtime.get((repeat, fold), [])
            brief = compact.get((case_id, repeat, fold), [])
            if len(run) > 1 or len(brief) > 1:
                add("duplicate_cell", "duplicate selected cell evidence", **scope)
            cell, path = run[0] if run else ({}, None)
            short = brief[0] if brief else {}
            if not cell:
                add("run_manifest_missing", "detailed cell evidence missing",
                    severity="not_verifiable", **scope)
            def agree(field: str, *values: Any) -> Any:
                present = [value for value in values if value is not None]
                if len({_hash(value) for value in present}) > 1:
                    add("cell_evidence_drift", f"{field} disagrees", **scope)
                return present[0] if present else None
            fitted = _map(cell.get("fitted_provenance"))
            frozen = _map(cell.get("frozen_model_run_provenance"))
            split_seed = _integer(agree("split_seed", cell.get("split_seed"), short.get("split_seed")))
            training_seed = _integer(agree(
                "training_seed", cell.get("training_seed"), short.get("training_seed"),
                fitted.get("training_seed"),
            ))
            orchestration = _integer(cell.get("training_orchestration_seed"))
            policy = str(cell.get("seed_policy", ""))
            declared_policy = str(model_cfg.get("seed_policy", ""))
            members = _integers(agree(
                "member_training_seeds", cell.get("member_training_seeds"),
                fitted.get("member_training_seeds"),
            )) or ()
            random_seeds = _integers(frozen.get("random_seeds")) or ()
            agree("runtime_seed_policy", policy or None, frozen.get("seed_policy"))
            model_seeds = members or ((training_seed,) if training_seed is not None else ())
            if random_seeds and random_seeds != model_seeds:
                add("model_seed_roster_drift", "frozen and runtime model seeds differ", **scope)
            if policy in {"outer_repeat", "outer_cv_repeat_seed_equals_split_seed"} and (
                training_seed != split_seed
                or (orchestration is not None and orchestration != split_seed)
                or (random_seeds and random_seeds != (split_seed,))
            ):
                add("repeat_seed_policy_drift", "repeat-local seeds differ from split seed", **scope)
            elif policy == "legacy_bridge_fixed_training_seed_42" and (
                training_seed != 42
                or (orchestration is not None and orchestration != 42)
                or (random_seeds and random_seeds != (42,))
            ):
                add("legacy_seed_policy_drift", "legacy bridge seeds are not 42", **scope)
            elif policy in {"member_roster", "cv_fixed_five_member_seed_roster",
                            "final_refit_five_member_seeds"} and (
                not members or (_integers(model_cfg.get("member_seeds")) or members) != members
                or (policy != "member_roster" and len(members) != 5)
            ):
                add("ensemble_seed_policy_drift", "ensemble member roster is invalid", **scope)
            elif policy == "cv_fixed_member0_seed_50042_comparator":
                if model_seeds != (50042,):
                    add("fixed_seed_policy_drift", "comparator model seed is not 50042", **scope)
            elif policy == "final_refit_single_seed_42":
                if model_seeds != (42,):
                    add("fixed_seed_policy_drift", "final-refit model seed is not 42", **scope)
            elif policy in {"fixed", "fixed_explicit"}:
                configured = _integer(model_cfg.get("seed"))
                if configured is None:
                    configured = _integer(training_cfg.get("seed"))
                if configured is not None and model_seeds != (configured,):
                    add("fixed_seed_policy_drift", "fixed seed disagrees with config", **scope)
            elif policy not in {
                "outer_repeat", "outer_cv_repeat_seed_equals_split_seed",
                "legacy_bridge_fixed_training_seed_42", "member_roster",
                "cv_fixed_five_member_seed_roster", "final_refit_five_member_seeds",
                "cv_fixed_member0_seed_50042_comparator", "fixed", "fixed_explicit",
                "final_refit_single_seed_42",
            }:
                add("seed_policy_unknown", f"unregistered seed policy {policy!r}",
                    severity="not_verifiable", **scope)
            if not policy or orchestration is None:
                add("seed_semantics_incomplete", "policy/orchestration seed unavailable",
                    severity="not_verifiable", **scope)

            fold_hash = agree("fold_hash", fitted.get("fold_hash"), frozen.get("fold_hash"),
                              frozen.get("split_sha256"))
            registry_hash = agree("registry_payload_hash", fitted.get("registry_hash"),
                                  split_cfg.get("source_registry_payload_sha256"))
            manifest_hash = frozen.get("manifest_sha256")
            if path and (path.parent / "experiment_result.json").is_file():
                provenance = _map(_json(path.parent / "experiment_result.json").get("provenance"))
                manifest_hash = agree("manifest_hash", manifest_hash, provenance.get("manifest_hash"),
                                      provenance.get("manifest_sha256"))
                fold_hash = agree("fold_hash", fold_hash, provenance.get("fold_hash"),
                                  provenance.get("split_sha256"))
            if registry and fold_hash and fold_hash != registry["sha256"]:
                add("split_file_hash_drift", "runtime fold hash differs from split CSV", **scope)
            if current_manifest_hash and manifest_hash and current_manifest_hash != manifest_hash:
                add("data_manifest_hash_drift", "runtime manifest hash differs from current file", **scope)

            registry_oof = set(registry.get("by_fold", {}).get((repeat, fold), ()))
            if registry and not registry_oof:
                add("split_registry_cell_roster_missing",
                    "materialized split CSV has no roster for this cell", **scope)
            observed_oof = oof.get((case_id, repeat, fold))
            if observed_oof is None:
                add("subject_oof_roster_unavailable", "subject OOF roster unavailable",
                    severity="not_verifiable", **scope)
            test_ids = observed_oof or registry_oof
            cohort = set(registry.get("by_repeat", {}).get(repeat, ()))
            registry_seed = _integer(registry.get("seeds", {}).get(repeat))
            if registry_seed is not None and split_seed != registry_seed:
                add("cell_split_seed_drift", "cell split seed differs from registry", **scope)
            declared_seeds = _integers(split_cfg.get("split_seeds"))
            if (declared_seeds is not None and registry_seed is not None
                    and (repeat >= len(declared_seeds) or declared_seeds[repeat] != registry_seed)):
                add("declared_split_seed_drift", "declared split seed differs from registry", **scope)
            train_ids = set(map(str, fitted.get("fitted_participant_ids", ())))
            expected_train = cohort - registry_oof if cohort and registry_oof else set()
            if expected_train and train_ids and train_ids != expected_train:
                add("train_roster_drift", "fitted train roster differs from registry", **scope)
            if registry_oof and test_ids != registry_oof:
                add("oof_roster_drift", "subject OOF roster differs from registry", **scope)
            overlap = len(train_ids & test_ids)
            if overlap:
                add("train_oof_overlap", f"overlap={overlap}", **scope)
            if not train_ids or not test_ids:
                add("participant_roster_incomplete", "train/OOF roster unavailable",
                    severity="not_verifiable", **scope)

            epoch_rows = [row for row in history.get((case_id, repeat, fold), ())
                          if row.get("epoch_rng_seed") is not None]
            selected_epoch = _integer(fitted.get("selected_epoch"))
            epoch_roster = []
            epoch_keys: set[tuple[int, int]] = set()
            for row in epoch_rows:
                values = tuple(map(_integer, (row.get("epoch"), row.get("member"),
                                              row.get("training_seed"), row.get("epoch_rng_seed"))))
                if None in values:
                    continue
                epoch, member, base, epoch_seed = values
                epoch_roster.append(values)
                expected = model_seeds[member] if member < len(model_seeds) else None
                numpy_seed = _integer(row.get("numpy_epoch_rng_seed"))
                if ((member, epoch) in epoch_keys or expected is None or base != expected
                        or epoch_seed != expected + epoch * 1_000_000
                        or (numpy_seed is not None and numpy_seed != epoch_seed % (1 << 32))):
                    add("epoch_seed_drift", "epoch RNG derivation disagrees", **scope)
                epoch_keys.add((member, epoch))
            if selected_epoch and not epoch_roster:
                add("epoch_seed_unavailable", "iterative cell lacks epoch RNG evidence",
                    severity="not_verifiable", **scope)
            elif selected_epoch and epoch_keys != {
                (member, epoch) for member in range(len(model_seeds))
                for epoch in range(1, selected_epoch + 1)
            }:
                add("epoch_seed_roster_incomplete", "epoch RNG roster is incomplete", **scope)

            train_tuple, test_tuple = tuple(sorted(train_ids)), tuple(sorted(test_ids))
            split_identity = (
                _hash((registry.get("sha256"), repeat, fold, split_seed, train_tuple, test_tuple))
                if train_tuple and test_tuple else None
            )
            if split_identity:
                row = splits.setdefault(split_identity, dict(
                    split_identity_sha256=split_identity, repeat=repeat, fold=fold,
                    split_seed=split_seed, split_registry_path=split_cfg.get("path"),
                    materialized_split_csv_sha256=registry.get("sha256"),
                    declared_source_registry_json_file_sha256=split_cfg.get("source_registry_file_sha256"),
                    declared_source_registry_payload_sha256=split_cfg.get("source_registry_payload_sha256"),
                    train_participant_count=len(train_tuple), oof_participant_count=len(test_tuple),
                    train_oof_overlap_count=overlap,
                    train_participant_roster_sha256=_hash(train_tuple),
                    oof_participant_roster_sha256=_hash(test_tuple),
                    train_participant_ids=list(train_tuple), oof_participant_ids=list(test_tuple),
                    case_ids=[]))
                row["case_ids"].append(case_id)
            relation = "same" if declared_policy == policy else (
                "numeric_equivalent_but_policy_different"
                if split_seed == training_seed and model_seeds == (split_seed,)
                else "effective_policy_differs"
            )
            statistics = _map(_map(cell.get("evaluation_policy")).get("statistics"))
            statistics_seed = _integer(agree(
                "evaluation_statistics_seed", statistics.get("seed"), statistics_cfg.get("seed")
            ))
            cells.append(dict(
                case_id=case_id, repeat=repeat, fold=fold,
                status=cell.get("status", short.get("status")),
                selected_attempt=record.get("attempt"),
                model_id=cell.get("model_machine_id", short.get("model_id")),
                declared_seed_policy=declared_policy or None,
                runtime_seed_policy=policy or None, seed_policy_relation=relation,
                split_seed=split_seed, training_orchestration_seed=orchestration,
                training_seed=training_seed, model_seed_roster=list(model_seeds),
                member_training_seeds=list(members) if members else None,
                member_seed_semantics=("independent_ensemble_roster" if members
                                       else "N/A_single_model_training_seed_alias"),
                evaluation_statistics_seed=statistics_seed,
                epoch_rng_seed_count=len(epoch_roster),
                epoch_rng_seed_roster_sha256=_hash(epoch_roster) if epoch_roster else None,
                data_manifest_sha256=manifest_hash,
                materialized_split_csv_sha256=fold_hash,
                declared_source_registry_json_file_sha256=split_cfg.get("source_registry_file_sha256"),
                runtime_registry_payload_sha256=registry_hash,
                split_identity_sha256=split_identity,
                train_participant_count=len(train_tuple) or None,
                oof_participant_count=len(test_tuple) or None,
                train_oof_overlap_count=overlap if test_tuple else None,
                train_participant_roster_sha256=_hash(train_tuple) if train_tuple else None,
                oof_participant_roster_sha256=_hash(test_tuple) if test_tuple else None))

        attempts = (sorted(path.name for path in (case_dir / "attempts").glob("attempt_*")
                           if path.is_dir()) if case_dir else [])
        current = [row for row in cells if row["case_id"] == case_id]
        selected_number = _integer(record.get("attempt"))
        selected_name = f"attempt_{selected_number:03d}" if selected_number is not None else None
        case_rows.append(dict(
            case_id=case_id, selected_attempt=record.get("attempt"),
            selected_case_status=record.get("status"),
            selected_artifact_root=record.get("artifact_root"),
            observed_attempt_count=len(attempts),
            excluded_attempts=[name for name in attempts if name != selected_name],
            planned_cell_count=len(planned) or None, observed_cell_count=len(observed),
            declared_seed_policies=sorted({row["declared_seed_policy"] for row in current
                                           if row["declared_seed_policy"]}),
            runtime_seed_policies=sorted({row["runtime_seed_policy"] for row in current
                                          if row["runtime_seed_policy"]}),
            split_seeds=sorted({row["split_seed"] for row in current
                                if row["split_seed"] is not None}),
            model_seeds=sorted({seed for row in current for seed in row["model_seed_roster"]}),
            training_orchestration_seeds=sorted({row["training_orchestration_seed"]
                                                   for row in current if row["training_orchestration_seed"] is not None}),
            evaluation_statistics_seeds=sorted({row["evaluation_statistics_seed"]
                                                for row in current if row["evaluation_statistics_seed"] is not None}),
            split_identity_sha256=sorted({row["split_identity_sha256"] for row in current
                                          if row["split_identity_sha256"]})))

    for repeat, fold in sorted({(row["repeat"], row["fold"]) for row in cells}):
        peers = [row for row in cells if (row["repeat"], row["fold"]) == (repeat, fold)]
        for field in ("split_seed", "data_manifest_sha256", "materialized_split_csv_sha256",
                      "runtime_registry_payload_sha256", "split_identity_sha256"):
            if len({_hash(row[field]) for row in peers if row[field] is not None}) > 1:
                add("cross_case_split_drift", f"{field} differs across cases",
                    repeat=repeat, fold=fold)

    def scoped(case_id: str | None = None, repeat: int | None = None,
               fold: int | None = None) -> str:
        def matches(value: Any, target: Any) -> bool:
            return target is None or value in (None, target)
        relevant = [row for row in issues
                    if matches(row["case_id"], case_id)
                    and matches(row["repeat"], repeat)
                    and matches(row["fold"], fold)]
        return _overall(relevant)

    for row in cells:
        row["audit_status"] = scoped(row["case_id"], row["repeat"], row["fold"])
    for row in case_rows:
        row["audit_status"] = scoped(row["case_id"])
    for row in splits.values():
        row["case_ids"] = sorted(set(row["case_ids"]))
        row["audit_status"] = scoped(None, row["repeat"], row["fold"])
    status = _overall(issues)
    summary = dict(
        audit_status=status, planned_case_count=len(cases),
        planned_repeats=list(repeats) if repeats is not None else None,
        planned_folds=list(folds) if folds is not None else None,
        planned_cell_count=len(cases) * len(planned) if planned else None,
        observed_cell_count=observed_cell_count,
        split_seed_by_repeat={
            str(repeat): sorted({row["split_seed"] for row in cells
                                 if row["repeat"] == repeat and row["split_seed"] is not None})
            for repeat in sorted({row["repeat"] for row in cells})},
        error_count=sum(row["severity"] == "error" for row in issues),
        not_verifiable_count=sum(row["severity"] != "error" for row in issues),
        scope="manifest_cases_and_selected_artifact_roots_only",
        training_or_report_gate=False)
    return ReproducibilityAudit(
        "ppg_frailty.reporting.reproducibility_audit.v1", status, summary,
        tuple(case_rows), tuple(cells),
        tuple(sorted(splits.values(), key=lambda row: (row["repeat"], row["fold"]))),
        tuple(issues))


__all__ = ["PASS", "FAIL", "NOT_VERIFIABLE", "ReproducibilityAudit",
           "audit_study_reproducibility"]

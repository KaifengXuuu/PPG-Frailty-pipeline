"""Post-hoc decomposition of training-role and aggregation-role scope.

The analysis is intentionally prediction-locked: it reads completed OOF file
and participant probabilities, replays the registered hierarchy on a selected
role-family subset, and never refits a classifier.  With the three observable
cells A=(static train, static aggregate), C=(all-role train, static aggregate),
and D=(all-role train, all-role aggregate), it reports the path-specific
identity D-A=(C-A)+(D-C).  The unobserved fourth cell is never imputed.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime
from html import escape as html_escape
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from scipy.stats import t as student_t
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..data.schema import CANONICAL_CLASS_NAMES
from ..reporting.tabular import (
    write_excel_workbook_from_csv_directory,
    write_table_column_definitions,
)
from ..training import read_oof_parquet, write_oof_parquet
from ..training.aggregation import (
    LINE_B_EQUAL_ROLE_FAMILIES,
    aggregate_hierarchy,
    canonical_role_family,
)
from ..training.oof import OofPredictionRow
from ..training.statistics import (
    ParticipantPrediction,
    holm_adjust_by_family_metric,
    paired_participant_cluster_bootstrap,
    paired_participant_permutation,
    participant_cluster_bootstrap,
)


SCHEMA_VERSION = "ppg_frailty.role_scope_decomposition.v1"
SCIENTIFIC_ROLE = "posthoc_prediction_locked_path_decomposition"
SUPPORTED_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "macro_roc_auc_ovr",
)
METRIC_LABELS = {
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro-F1",
    "macro_roc_auc_ovr": "Macro ROC-AUC (OvR)",
}
_CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    study_dir: Path
    case_id: str
    prediction_root: Path | None
    training_role_families: tuple[str, ...]
    expected_files_per_participant: int


@dataclass(frozen=True)
class RankingEvidenceSpec:
    leaderboard_path: Path
    static_case_id: str
    static_expected_rank: int
    all_role_case_id: str
    all_role_expected_rank: int


@dataclass(frozen=True)
class RoleScopePlan:
    plan_path: Path
    study_id: str
    purpose: str
    static_source: SourceSpec
    all_role_source: SourceSpec
    ranking_evidence: RankingEvidenceSpec | None
    static_aggregation_roles: tuple[str, ...]
    all_aggregation_roles: tuple[str, ...]
    balance_line: str
    expected_participants: int
    expected_repeats: tuple[int, ...]
    expected_class_order: tuple[int, ...]
    metrics: tuple[str, ...]
    bootstrap_resamples: int
    permutation_resamples: int
    inference_seed: int
    alpha: float
    multiplicity_family: str
    require_complete_retention: bool
    probability_tolerance: float
    output_slug: str
    write_static_figures: bool
    write_excel_workbook: bool
    write_result_backup: bool


@dataclass(frozen=True)
class LoadedSource:
    spec: SourceSpec
    case_root: Path
    experiment_root: Path
    resolved_config_path: Path
    file_prediction_path: Path
    subject_prediction_path: Path
    file_rows: tuple[OofPredictionRow, ...]
    subject_rows: tuple[OofPredictionRow, ...]
    file_sha256: str
    subject_sha256: str
    config_sha256: str
    checkpoint_paths: tuple[Path, ...]


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _resolve_path(value: Any, *, base: Path, field: str, optional: bool = False) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _canonical_roles(values: Iterable[Any], *, field: str) -> tuple[str, ...]:
    roles = tuple(canonical_role_family(str(value)) for value in values)
    if not roles or len(roles) != len(set(roles)):
        raise ValueError(f"{field} must contain unique canonical role families")
    return roles


def _source_spec(value: Any, *, base: Path, field: str) -> SourceSpec:
    source = _mapping(value, field)
    name = str(source.get("name", "")).strip()
    case_id = str(source.get("case_id", "")).strip()
    if not name or not case_id:
        raise ValueError(f"{field}.name and {field}.case_id are required")
    expected_files = int(source.get("expected_files_per_participant", 0))
    if expected_files <= 0:
        raise ValueError(f"{field}.expected_files_per_participant must be positive")
    return SourceSpec(
        name=name,
        study_dir=_resolve_path(source.get("study_dir"), base=base, field=f"{field}.study_dir"),
        case_id=case_id,
        prediction_root=_resolve_path(
            source.get("prediction_root"),
            base=base,
            field=f"{field}.prediction_root",
            optional=True,
        ),
        training_role_families=_canonical_roles(
            source.get("training_role_families", ()),
            field=f"{field}.training_role_families",
        ),
        expected_files_per_participant=expected_files,
    )


def _ranking_evidence_spec(value: Any, *, base: Path) -> RankingEvidenceSpec | None:
    if value is None:
        return None
    source = _mapping(value, "ranking_evidence")
    static_case = str(source.get("static_case_id", "")).strip()
    all_role_case = str(source.get("all_role_case_id", "")).strip()
    static_rank = int(source.get("static_expected_rank", 0))
    all_role_rank = int(source.get("all_role_expected_rank", 0))
    if not static_case or not all_role_case or static_rank <= 0 or all_role_rank <= 0:
        raise ValueError("ranking_evidence requires case ids and positive expected ranks")
    return RankingEvidenceSpec(
        leaderboard_path=_resolve_path(
            source.get("leaderboard_path"),
            base=base,
            field="ranking_evidence.leaderboard_path",
        ),
        static_case_id=static_case,
        static_expected_rank=static_rank,
        all_role_case_id=all_role_case,
        all_role_expected_rank=all_role_rank,
    )


def load_role_scope_plan(
    path: str | Path,
    *,
    pipeline_root: str | Path | None = None,
) -> RoleScopePlan:
    """Load and fail-closed validate one role-scope decomposition plan."""

    plan_path = Path(path).resolve()
    payload = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    root = _mapping(payload, "plan")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema {root.get('schema_version')!r}; expected {SCHEMA_VERSION!r}"
        )
    base = Path(pipeline_root).resolve() if pipeline_root is not None else plan_path.parent
    study = _mapping(root.get("study"), "study")
    sources = _mapping(root.get("sources"), "sources")
    scopes = _mapping(root.get("aggregation_scopes"), "aggregation_scopes")
    inference = _mapping(root.get("inference"), "inference")
    output = _mapping(root.get("output"), "output")
    study_id = str(study.get("study_id", "")).strip()
    purpose = str(study.get("purpose", "")).strip()
    if not study_id or not purpose:
        raise ValueError("study.study_id and study.purpose are required")
    static_roles = _canonical_roles(scopes.get("static", ()), field="aggregation_scopes.static")
    all_roles = _canonical_roles(scopes.get("all", ()), field="aggregation_scopes.all")
    if not set(static_roles) < set(all_roles):
        raise ValueError("static aggregation roles must be a strict subset of all roles")
    balance_line = str(scopes.get("balance_line", ""))
    if balance_line != LINE_B_EQUAL_ROLE_FAMILIES:
        raise ValueError("this decomposition currently requires line_b_equal_role_families")
    expected_repeats = tuple(int(value) for value in inference.get("expected_repeats", ()))
    expected_order = tuple(int(value) for value in inference.get("expected_class_order", ()))
    metrics = tuple(str(value) for value in inference.get("metrics", SUPPORTED_METRICS))
    if not expected_repeats or len(expected_repeats) != len(set(expected_repeats)):
        raise ValueError("inference.expected_repeats must be non-empty and unique")
    if len(expected_order) < 2 or len(expected_order) != len(set(expected_order)):
        raise ValueError("inference.expected_class_order must be unique")
    if not metrics or len(metrics) != len(set(metrics)) or not set(metrics) <= set(SUPPORTED_METRICS):
        raise ValueError(f"metrics must be a unique subset of {SUPPORTED_METRICS}")
    if "balanced_accuracy" not in metrics:
        raise ValueError("balanced_accuracy is required for the primary decomposition")
    bootstrap = int(inference.get("bootstrap_resamples", 10_000))
    permutation = int(inference.get("permutation_resamples", 100_000))
    seed = int(inference.get("seed", 42))
    alpha = float(inference.get("alpha", 0.05))
    tolerance = float(inference.get("probability_tolerance", 1e-12))
    if bootstrap <= 0 or permutation <= 0 or seed < 0:
        raise ValueError("resample counts must be positive and seed non-negative")
    if not 0.0 < alpha < 1.0 or tolerance <= 0.0:
        raise ValueError("alpha must be in (0,1) and probability_tolerance positive")
    expected_participants = int(inference.get("expected_participants", 0))
    if expected_participants < len(expected_order):
        raise ValueError("expected_participants is too small")
    slug = str(output.get("slug", "")).strip()
    if not slug:
        raise ValueError("output.slug is required")
    family = str(inference.get("multiplicity_family", "")).strip()
    if not family:
        raise ValueError("inference.multiplicity_family is required")
    return RoleScopePlan(
        plan_path=plan_path,
        study_id=study_id,
        purpose=purpose,
        static_source=_source_spec(
            sources.get("static_training"), base=base, field="sources.static_training"
        ),
        all_role_source=_source_spec(
            sources.get("all_role_training"), base=base, field="sources.all_role_training"
        ),
        ranking_evidence=_ranking_evidence_spec(root.get("ranking_evidence"), base=base),
        static_aggregation_roles=static_roles,
        all_aggregation_roles=all_roles,
        balance_line=balance_line,
        expected_participants=expected_participants,
        expected_repeats=expected_repeats,
        expected_class_order=expected_order,
        metrics=metrics,
        bootstrap_resamples=bootstrap,
        permutation_resamples=permutation,
        inference_seed=seed,
        alpha=alpha,
        multiplicity_family=family,
        require_complete_retention=bool(inference.get("require_complete_retention", True)),
        probability_tolerance=tolerance,
        output_slug=slug,
        write_static_figures=bool(output.get("write_static_figures", True)),
        write_excel_workbook=bool(output.get("write_excel_workbook", True)),
        write_result_backup=bool(output.get("write_result_backup", True)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_experiment_root(spec: SourceSpec) -> tuple[Path, Path]:
    case_root = spec.study_dir / "raw" / spec.case_id
    if spec.prediction_root is not None:
        experiment = spec.prediction_root
    else:
        candidates = sorted(case_root.glob("attempts/attempt_*/experiment"))
        if len(candidates) != 1:
            raise ValueError(
                f"source {spec.name!r} must resolve to exactly one completed experiment; "
                f"found {len(candidates)}"
            )
        experiment = candidates[0]
    if not experiment.is_dir():
        raise FileNotFoundError(f"prediction root not found: {experiment}")
    return case_root, experiment


def _row_key(row: OofPredictionRow) -> tuple[str, int]:
    return str(row.participant_id), int(row.repeat)


def _validate_subject_rows(
    rows: Sequence[OofPredictionRow],
    *,
    plan: RoleScopePlan,
    source_name: str,
) -> None:
    if not rows or any(row.level != "participant" for row in rows):
        raise ValueError(f"{source_name} must contain only participant-level OOF rows")
    selected = tuple(row for row in rows if row.level == "participant")
    if len(selected) != plan.expected_participants * len(plan.expected_repeats):
        raise ValueError(f"{source_name} participant OOF roster is incomplete")
    keys = [_row_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{source_name} has duplicate participant-repeat predictions")
    if tuple(sorted({row.repeat for row in selected})) != tuple(sorted(plan.expected_repeats)):
        raise ValueError(f"{source_name} repeat roster differs from the plan")
    if len({row.participant_id for row in selected}) != plan.expected_participants:
        raise ValueError(f"{source_name} participant count differs from the plan")
    if {tuple(row.class_order) for row in selected} != {plan.expected_class_order}:
        raise ValueError(f"{source_name} class order differs from the plan")
    if plan.require_complete_retention and not all(row.retained for row in selected):
        raise ValueError(f"{source_name} contains participant abstentions")


def _validate_file_rows(
    rows: Sequence[OofPredictionRow],
    *,
    spec: SourceSpec,
    plan: RoleScopePlan,
) -> None:
    if not rows or any(row.level != "file" for row in rows):
        raise ValueError(f"{spec.name} must contain only file-level OOF rows")
    if plan.require_complete_retention and not all(row.retained for row in rows):
        raise ValueError(f"{spec.name} contains dropped file predictions")
    groups: dict[tuple[str, int], list[OofPredictionRow]] = {}
    for row in rows:
        groups.setdefault(_row_key(row), []).append(row)
    expected_keys = plan.expected_participants * len(plan.expected_repeats)
    if len(groups) != expected_keys:
        raise ValueError(f"{spec.name} file OOF roster is incomplete")
    expected_roles = set(spec.training_role_families)
    for key, values in groups.items():
        if len(values) != spec.expected_files_per_participant:
            raise ValueError(f"{spec.name} has unexpected file count for {key}")
        if {canonical_role_family(row.role) for row in values} != expected_roles:
            raise ValueError(f"{spec.name} has unexpected role-family coverage for {key}")
        if {tuple(row.class_order) for row in values} != {plan.expected_class_order}:
            raise ValueError(f"{spec.name} has an unexpected file-level class order for {key}")
        if len({row.label for row in values}) != 1:
            raise ValueError(f"{spec.name} has inconsistent file-level labels for {key}")


def _load_source(spec: SourceSpec, plan: RoleScopePlan) -> LoadedSource:
    case_root, experiment = _resolve_experiment_root(spec)
    file_path = experiment / "oof_file_predictions.parquet"
    subject_path = experiment / "oof_subject_predictions.parquet"
    config_path = case_root / "resolved_config.yaml"
    for path in (file_path, subject_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"required source artifact not found: {path}")
    file_rows = read_oof_parquet(file_path)
    subject_rows = read_oof_parquet(subject_path)
    _validate_file_rows(file_rows, spec=spec, plan=plan)
    _validate_subject_rows(subject_rows, plan=plan, source_name=spec.name)
    checkpoints = tuple(
        sorted(
            path
            for path in case_root.rglob("*")
            if path.is_file() and path.suffix.lower() in _CHECKPOINT_SUFFIXES
        )
    )
    return LoadedSource(
        spec=spec,
        case_root=case_root,
        experiment_root=experiment,
        resolved_config_path=config_path,
        file_prediction_path=file_path,
        subject_prediction_path=subject_path,
        file_rows=file_rows,
        subject_rows=subject_rows,
        file_sha256=_sha256(file_path),
        subject_sha256=_sha256(subject_path),
        config_sha256=_sha256(config_path),
        checkpoint_paths=checkpoints,
    )


def _config_without_role_scope(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"resolved config is not a mapping: {path}")
    normalized = json.loads(json.dumps(payload))
    normalized.pop("config_id", None)
    normalized.pop("roles", None)
    training = normalized.get("training")
    if isinstance(training, dict):
        training.pop("classifier_role_families", None)
    return normalized


def _assert_source_pair(static: LoadedSource, all_role: LoadedSource, plan: RoleScopePlan) -> None:
    if _config_without_role_scope(static.resolved_config_path) != _config_without_role_scope(
        all_role.resolved_config_path
    ):
        raise ValueError("source configs differ outside config_id/roles/classifier_role_families")
    static_by_key = {_row_key(row): row for row in static.subject_rows}
    all_by_key = {_row_key(row): row for row in all_role.subject_rows}
    if set(static_by_key) != set(all_by_key):
        raise ValueError("static/all-role sources have different participant-repeat rosters")
    for key in sorted(static_by_key):
        left, right = static_by_key[key], all_by_key[key]
        if (
            left.label != right.label
            or left.fold != right.fold
            or left.split_seed != right.split_seed
            or left.training_seed != right.training_seed
            or tuple(left.class_order) != tuple(right.class_order)
            or left.manifest_hash != right.manifest_hash
            or left.fold_hash != right.fold_hash
            or left.source_snapshot_hash != right.source_snapshot_hash
            or left.manifest_version != right.manifest_version
            or left.fold_registry_version != right.fold_registry_version
        ):
            raise ValueError(f"source pairing metadata differs for {key}")
    if tuple(static.spec.training_role_families) != tuple(plan.static_aggregation_roles):
        raise ValueError("static source training roles must match the static aggregation scope")
    if tuple(all_role.spec.training_role_families) != tuple(plan.all_aggregation_roles):
        raise ValueError("all-role source training roles must match the all aggregation scope")
    static_file_by_key = {
        (row.participant_id, row.repeat, row.file_id, row.role): row
        for row in static.file_rows
    }
    all_static_file_by_key = {
        (row.participant_id, row.repeat, row.file_id, row.role): row
        for row in all_role.file_rows
        if canonical_role_family(row.role) in set(plan.static_aggregation_roles)
    }
    if set(static_file_by_key) != set(all_static_file_by_key):
        raise ValueError("static/all-role sources have different shared B/R file rosters")
    for key in sorted(static_file_by_key):
        left, right = static_file_by_key[key], all_static_file_by_key[key]
        if (
            left.label != right.label
            or left.fold != right.fold
            or left.split_seed != right.split_seed
            or left.training_seed != right.training_seed
            or tuple(left.class_order) != tuple(right.class_order)
            or left.manifest_hash != right.manifest_hash
            or left.fold_hash != right.fold_hash
            or left.source_snapshot_hash != right.source_snapshot_hash
            or left.feature_hash != right.feature_hash
            or left.quality_score != right.quality_score
            or left.retained != right.retained
        ):
            raise ValueError(f"shared B/R file pairing metadata differs for {key}")


def _aggregate_roles(
    rows: Sequence[OofPredictionRow],
    roles: Sequence[str],
    *,
    balance_line: str,
) -> tuple[OofPredictionRow, ...]:
    allowed = set(roles)
    selected = tuple(row for row in rows if canonical_role_family(row.role) in allowed)
    if not selected:
        raise ValueError("role filter produced no file OOF predictions")
    return aggregate_hierarchy(
        selected,
        balance_line=balance_line,
        quality_weighted=False,
        quality_weight_source="none",
    ).participant_rows


def _assert_probability_replay(
    observed: Sequence[OofPredictionRow],
    replayed: Sequence[OofPredictionRow],
    *,
    tolerance: float,
    source_name: str,
) -> float:
    left = {_row_key(row): row for row in observed}
    right = {_row_key(row): row for row in replayed}
    if set(left) != set(right):
        raise ValueError(f"{source_name} replay roster differs from persisted participant OOF")
    maximum = 0.0
    for key in sorted(left):
        if left[key].label != right[key].label or left[key].fold != right[key].fold:
            raise ValueError(f"{source_name} replay identity mismatch for {key}")
        delta = float(
            np.max(
                np.abs(
                    np.asarray(left[key].probabilities, dtype=np.float64)
                    - np.asarray(right[key].probabilities, dtype=np.float64)
                )
            )
        )
        maximum = max(maximum, delta)
    if maximum > tolerance:
        raise ValueError(
            f"{source_name} replay differs from persisted probabilities: max={maximum:g}"
        )
    return maximum


def _participant_predictions(rows: Sequence[OofPredictionRow]) -> tuple[ParticipantPrediction, ...]:
    return tuple(
        ParticipantPrediction(
            participant_id=row.participant_id,
            label=row.label,
            repeat=row.repeat,
            probabilities=tuple(row.probabilities),
        )
        for row in sorted(rows, key=lambda item: (item.participant_id, item.repeat))
    )


def _metric_value(rows: Sequence[OofPredictionRow], metric: str) -> float:
    ordered = sorted(rows, key=lambda item: item.participant_id)
    labels = np.asarray([row.label for row in ordered], dtype=np.int64)
    probability = np.asarray([row.probabilities for row in ordered], dtype=np.float64)
    prediction = np.asarray([row.class_order[int(np.argmax(row.probabilities))] for row in ordered])
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(labels, prediction))
    if metric == "macro_f1":
        return float(f1_score(labels, prediction, labels=ordered[0].class_order, average="macro"))
    if metric == "macro_roc_auc_ovr":
        return float(
            roc_auc_score(
                labels,
                probability,
                labels=ordered[0].class_order,
                average="macro",
                multi_class="ovr",
            )
        )
    raise ValueError(f"unsupported metric: {metric}")


def _repeat_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return float("nan"), float("nan")
    half = float(student_t.ppf(0.975, df=array.size - 1) * array.std(ddof=1) / math.sqrt(array.size))
    return float(array.mean() - half), float(array.mean() + half)


def _cell_metric_rows(
    cells: Mapping[str, Sequence[OofPredictionRow]],
    plan: RoleScopePlan,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    labels = {
        "A_static_train_static_aggregate": (
            plan.static_source.name,
            "static",
            "static",
            f"{plan.static_source.name} (persisted)",
        ),
        "B_static_train_all_aggregate": (
            f"{plan.static_source.name} — unavailable all-role counterfactual",
            "static",
            "all",
            "N/A",
        ),
        "C_all_train_static_aggregate": (
            f"{plan.all_role_source.name} — static-role reaggregation",
            "all",
            "static",
            f"derived from {plan.all_role_source.name} file OOF",
        ),
        "D_all_train_all_aggregate": (
            plan.all_role_source.name,
            "all",
            "all",
            f"{plan.all_role_source.name} (persisted)",
        ),
    }
    for cell_id, (model_name, train_scope, aggregation_scope, provenance) in labels.items():
        if cell_id not in cells:
            for metric in plan.metrics:
                summaries.append(
                    {
                        "model_name": model_name,
                        "cell_id": cell_id,
                        "training_scope": train_scope,
                        "aggregation_scope": aggregation_scope,
                        "metric": metric,
                        "mean_percent": None,
                        "sd_percent": None,
                        "mean_sd": "N/A",
                        "repeat_ci95_percent": "N/A",
                        "participant_cluster_ci95_percent": "N/A",
                        "n_participants": plan.expected_participants,
                        "n_repeats": len(plan.expected_repeats),
                        "availability": "unavailable_static_model_S_W_predictions_absent",
                        "provenance": provenance,
                    }
                )
            continue
        rows = tuple(cells[cell_id])
        predictions = _participant_predictions(rows)
        for metric in plan.metrics:
            values = []
            for repeat in plan.expected_repeats:
                selected = tuple(row for row in rows if row.repeat == repeat)
                value = _metric_value(selected, metric)
                values.append(value)
                repeat_rows.append(
                    {
                        "cell_id": cell_id,
                        "training_scope": train_scope,
                        "aggregation_scope": aggregation_scope,
                        "repeat": repeat,
                        "metric": metric,
                        "value": value,
                        "value_percent": 100.0 * value,
                    }
                )
            bootstrap = participant_cluster_bootstrap(
                predictions,
                class_order=plan.expected_class_order,
                metric=metric,
                n_resamples=plan.bootstrap_resamples,
                seed=plan.inference_seed,
            )
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            ci_low, ci_high = _repeat_ci(values)
            summaries.append(
                {
                    "model_name": model_name,
                    "cell_id": cell_id,
                    "training_scope": train_scope,
                    "aggregation_scope": aggregation_scope,
                    "metric": metric,
                    "mean_percent": 100.0 * mean,
                    "sd_percent": 100.0 * sd,
                    "mean_sd": f"{100.0 * mean:.1f} ± {100.0 * sd:.1f}",
                    "repeat_ci95_percent": f"[{100.0 * ci_low:.1f}, {100.0 * ci_high:.1f}]",
                    "participant_cluster_ci95_percent": (
                        f"[{100.0 * bootstrap.ci95_lower:.1f}, "
                        f"{100.0 * bootstrap.ci95_upper:.1f}]"
                    ),
                    "n_participants": bootstrap.n_participants,
                    "n_repeats": bootstrap.n_repeats,
                    "availability": "available",
                    "provenance": provenance,
                }
            )
    return summaries, repeat_rows


def _contrast_rows(
    cells: Mapping[str, Sequence[OofPredictionRow]],
    plan: RoleScopePlan,
) -> list[dict[str, Any]]:
    definitions = (
        (
            "training_side_at_static_aggregation",
            "A_static_train_static_aggregate",
            "C_all_train_static_aggregate",
            "C − A",
            "training-scope bundle: samples, fitted state, class weights, and optimization",
        ),
        (
            "aggregation_side_with_all_role_training",
            "C_all_train_static_aggregate",
            "D_all_train_all_aggregate",
            "D − C",
            "same fitted all-role models; only participant aggregation adds S/W",
        ),
        (
            "total_all_role_minus_static",
            "A_static_train_static_aggregate",
            "D_all_train_all_aggregate",
            "D − A",
            "observed all-role/all-role minus static/static difference",
        ),
    )
    raw: dict[tuple[str, str, str], float] = {}
    rows: list[dict[str, Any]] = []
    total_by_metric: dict[str, float] = {}
    for contrast_id, reference_id, candidate_id, formula, interpretation in definitions:
        reference = _participant_predictions(cells[reference_id])
        candidate = _participant_predictions(cells[candidate_id])
        for metric in plan.metrics:
            bootstrap = paired_participant_cluster_bootstrap(
                reference,
                candidate,
                class_order=plan.expected_class_order,
                metric=metric,
                n_resamples=plan.bootstrap_resamples,
                seed=plan.inference_seed,
            )
            permutation = None
            if metric in {"balanced_accuracy", "macro_f1"}:
                permutation = paired_participant_permutation(
                    reference,
                    candidate,
                    class_order=plan.expected_class_order,
                    metric=metric,
                    n_resamples=plan.permutation_resamples,
                    seed=plan.inference_seed,
                )
                raw[(plan.multiplicity_family, metric, contrast_id)] = (
                    permutation.two_sided_p_value
                )
            if contrast_id == "total_all_role_minus_static":
                total_by_metric[metric] = bootstrap.observed_candidate_minus_reference
            rows.append(
                {
                    "contrast_id": contrast_id,
                    "reference_cell": reference_id,
                    "candidate_cell": candidate_id,
                    "formula": formula,
                    "metric": metric,
                    "delta_percentage_points": 100.0
                    * bootstrap.observed_candidate_minus_reference,
                    "participant_cluster_ci95_percentage_points": (
                        f"[{100.0 * bootstrap.ci95_lower:.2f}, "
                        f"{100.0 * bootstrap.ci95_upper:.2f}]"
                    ),
                    "raw_p_value": (
                        None if permutation is None else permutation.two_sided_p_value
                    ),
                    "holm_adjusted_p_value": None,
                    "holm_reject_0_05": None,
                    "share_of_total_delta_percent": None,
                    "interpretation": interpretation,
                    "inference_unit": "participant_with_all_repeats",
                }
            )
    holm = {
        (value.metric, value.comparison_id): value
        for value in holm_adjust_by_family_metric(raw, alpha=plan.alpha)
    }
    for row in rows:
        key = (str(row["metric"]), str(row["contrast_id"]))
        if key in holm:
            row["holm_adjusted_p_value"] = holm[key].adjusted_p_value
            row["holm_reject_0_05"] = holm[key].reject_null
        total = total_by_metric.get(str(row["metric"]))
        if row["contrast_id"] == "total_all_role_minus_static":
            row["share_of_total_delta_percent"] = 100.0
        elif total is not None and not math.isclose(total, 0.0, abs_tol=1e-15):
            row["share_of_total_delta_percent"] = (
                float(row["delta_percentage_points"]) / (100.0 * total) * 100.0
            )
    return rows


def _per_class_rows(
    cells: Mapping[str, Sequence[OofPredictionRow]], plan: RoleScopePlan
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    confusions: list[dict[str, Any]] = []
    class_names = {
        index: CANONICAL_CLASS_NAMES.get(index, f"class_{index}")
        for index in plan.expected_class_order
    }
    for cell_id, rows in cells.items():
        for repeat in plan.expected_repeats:
            selected = sorted(
                (row for row in rows if row.repeat == repeat),
                key=lambda item: item.participant_id,
            )
            labels = np.asarray([row.label for row in selected], dtype=np.int64)
            predicted = np.asarray(
                [row.class_order[int(np.argmax(row.probabilities))] for row in selected],
                dtype=np.int64,
            )
            precision, recall, f1, support = precision_recall_fscore_support(
                labels,
                predicted,
                labels=plan.expected_class_order,
                zero_division=0,
            )
            matrix = confusion_matrix(labels, predicted, labels=plan.expected_class_order)
            for position, class_id in enumerate(plan.expected_class_order):
                true_negative = matrix.sum() - matrix[position, :].sum() - matrix[:, position].sum() + matrix[position, position]
                false_positive = matrix[:, position].sum() - matrix[position, position]
                specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
                output.append(
                    {
                        "cell_id": cell_id,
                        "repeat": repeat,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "support": int(support[position]),
                        "precision": float(precision[position]),
                        "sensitivity_recall": float(recall[position]),
                        "specificity": float(specificity),
                        "balanced_accuracy_ovr": float(
                            0.5 * (recall[position] + specificity)
                        ),
                        "f1": float(f1[position]),
                    }
                )
            for true_position, true_class in enumerate(plan.expected_class_order):
                for predicted_position, predicted_class in enumerate(plan.expected_class_order):
                    confusions.append(
                        {
                            "cell_id": cell_id,
                            "repeat": repeat,
                            "true_class": true_class,
                            "predicted_class": predicted_class,
                            "count": int(matrix[true_position, predicted_position]),
                        }
                    )
    return output, confusions


def _per_class_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    plan: RoleScopePlan,
) -> list[dict[str, Any]]:
    model_names = {
        "A_static_train_static_aggregate": plan.static_source.name,
        "C_all_train_static_aggregate": (
            f"{plan.all_role_source.name} — static-role reaggregation"
        ),
        "D_all_train_all_aggregate": plan.all_role_source.name,
    }
    metrics = (
        "balanced_accuracy_ovr",
        "f1",
        "sensitivity_recall",
        "specificity",
        "precision",
    )
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["cell_id"]), int(row["class_id"]), str(row["class_name"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (cell_id, class_id, class_name), selected in sorted(grouped.items()):
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
            output.append(
                {
                    "model_name": model_names[cell_id],
                    "cell_id": cell_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "metric": metric,
                    "mean_percent": 100.0 * mean,
                    "sd_percent": 100.0 * sd,
                    "mean_sd": f"{100.0 * mean:.1f} ± {100.0 * sd:.1f}",
                    "n_repeats": len(selected),
                }
            )
    return output


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ("status",)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
        else:
            writer.writerow({"status": "no_rows"})
    path.with_suffix(".json").write_text(
        json.dumps(list(rows), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _plots(
    output_dir: Path,
    cell_rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    repeat_rows: Sequence[Mapping[str, Any]],
    per_class_summary_rows: Sequence[Mapping[str, Any]],
    plan: RoleScopePlan,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ba = {
        str(row["cell_id"]): row["mean_percent"]
        for row in cell_rows
        if row["metric"] == "balanced_accuracy"
    }
    matrix = np.asarray(
        [
            [ba.get("A_static_train_static_aggregate", np.nan), np.nan],
            [ba.get("C_all_train_static_aggregate", np.nan), ba.get("D_all_train_all_aggregate", np.nan)],
        ],
        dtype=np.float64,
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    chance_percent = 100.0 / len(plan.expected_class_order)
    image = axis.imshow(
        matrix,
        cmap="Blues",
        vmin=chance_percent,
        vmax=max(80.0, np.nanmax(matrix)),
    )
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            text = "N/A\nno static-model S/W predictions" if np.isnan(value) else f"{value:.1f}%"
            axis.text(column, row, text, ha="center", va="center", fontsize=10)
    axis.set_xticks((0, 1), ("Static B/R", "All B/R/S/W"))
    axis.set_yticks((0, 1), ("Static-trained", "All-role-trained"))
    axis.set_xlabel("Participant aggregation scope")
    axis.set_ylabel("Model training scope")
    axis.set_title("Prediction-locked role-scope matrix — balanced accuracy")
    fig.colorbar(image, ax=axis, label="Balanced accuracy (%)")
    fig.tight_layout()
    fig.savefig(figures / "role_scope_matrix.png", dpi=180)
    plt.close(fig)

    components = [
        row
        for row in contrast_rows
        if row["contrast_id"]
        in {"training_side_at_static_aggregation", "aggregation_side_with_all_role_training"}
    ]
    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    x = np.arange(len(plan.metrics), dtype=np.float64)
    width = 0.36
    for offset, contrast_id, label, color in (
        (-width / 2, "training_side_at_static_aggregation", "Training-side C−A", "#4c78a8"),
        (width / 2, "aggregation_side_with_all_role_training", "Aggregation-side D−C", "#f58518"),
    ):
        values = [
            next(
                float(row["delta_percentage_points"])
                for row in components
                if row["contrast_id"] == contrast_id and row["metric"] == metric
            )
            for metric in plan.metrics
        ]
        axis.bar(x + offset, values, width, label=label, color=color)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(
        x,
        [METRIC_LABELS[value] for value in plan.metrics],
        rotation=12,
        ha="right",
    )
    axis.set_ylabel("Contribution along A→C→D path (percentage points)")
    axis.set_title("All-role/all-role minus static/static path decomposition")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "decomposition_contributions.png", dpi=180)
    plt.close(fig)

    repeat_lookup = {
        (str(row["cell_id"]), int(row["repeat"]), str(row["metric"])): float(row["value_percent"])
        for row in repeat_rows
    }
    fig, axes = plt.subplots(
        1,
        len(plan.metrics),
        figsize=(4.4 * len(plan.metrics), 4.6),
        squeeze=False,
    )
    for index, metric in enumerate(plan.metrics):
        axis = axes[0, index]
        training = []
        aggregation = []
        for repeat in sorted({key[1] for key in repeat_lookup}):
            a = repeat_lookup[("A_static_train_static_aggregate", repeat, metric)]
            c = repeat_lookup[("C_all_train_static_aggregate", repeat, metric)]
            d = repeat_lookup[("D_all_train_all_aggregate", repeat, metric)]
            training.append(c - a)
            aggregation.append(d - c)
        axis.scatter(np.zeros(len(training)), training, color="#4c78a8")
        axis.scatter(np.ones(len(aggregation)), aggregation, color="#f58518")
        for left, right in zip(training, aggregation, strict=True):
            axis.plot((0, 1), (left, right), color="#bbbbbb", linewidth=0.8)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks((0, 1), ("Training", "Aggregation"))
        axis.set_title(METRIC_LABELS[metric])
        axis.set_ylabel("Repeat delta (pp)")
    fig.suptitle("Repeat-wise path contributions")
    fig.tight_layout()
    fig.savefig(figures / "paired_repeat_contributions.png", dpi=180)
    plt.close(fig)

    f1_rows = [row for row in per_class_summary_rows if row["metric"] == "f1"]
    cell_order = (
        "A_static_train_static_aggregate",
        "C_all_train_static_aggregate",
        "D_all_train_all_aggregate",
    )
    class_order = tuple(plan.expected_class_order)
    x = np.arange(len(class_order), dtype=np.float64)
    width = 0.24
    fig, axis = plt.subplots(figsize=(9.0, 5.2))
    for cell_index, cell_id in enumerate(cell_order):
        selected = {
            int(row["class_id"]): row
            for row in f1_rows
            if row["cell_id"] == cell_id
        }
        means = [float(selected[class_id]["mean_percent"]) for class_id in class_order]
        errors = [float(selected[class_id]["sd_percent"]) for class_id in class_order]
        label = str(selected[class_order[0]]["model_name"])
        axis.bar(
            x + (cell_index - 1) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=label,
        )
    axis.set_xticks(
        x,
        [CANONICAL_CLASS_NAMES.get(value, f"class_{value}") for value in class_order],
    )
    axis.set_ylim(0.0, 100.0)
    axis.set_ylabel("Per-class F1 across repeats, mean ± SD (%)")
    axis.set_title("Per-class F1 under the three observable role-scope cells")
    axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(figures / "per_class_f1.png", dpi=180)
    plt.close(fig)


def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join((header, divider, *body))


def _html_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    header = "".join(f"<th>{html_escape(field)}</th>" for field in fields)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html_escape(str(row.get(field, '')))}</td>" for field in fields)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _write_reports(
    output_dir: Path,
    plan: RoleScopePlan,
    cell_rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    per_class_summary_rows: Sequence[Mapping[str, Any]],
    provenance_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    dynamic_weight: float,
    figures_written: bool,
) -> None:
    selected_cells = {
        metric: [row for row in cell_rows if row["metric"] == metric]
        for metric in plan.metrics
    }
    selected_contrasts = []
    for source in contrast_rows:
        row = dict(source)
        row["delta_percentage_points"] = f"{float(source['delta_percentage_points']):.2f}"
        for field in ("raw_p_value", "holm_adjusted_p_value"):
            value = source[field]
            row[field] = "N/A" if value is None else f"{float(value):.6f}"
        share = source["share_of_total_delta_percent"]
        row["share_of_total_delta_percent"] = (
            "N/A" if share is None else f"{float(share):.1f}"
        )
        selected_contrasts.append(row)
    ba = {str(row["contrast_id"]): row for row in contrast_rows if row["metric"] == "balanced_accuracy"}
    training = ba["training_side_at_static_aggregation"]
    aggregation = ba["aggregation_side_with_all_role_training"]
    total = ba["total_all_role_minus_static"]
    conclusion = (
        f"Along the defined post-hoc A→C→D path, {training['delta_percentage_points']:.2f} pp "
        f"({training['share_of_total_delta_percent']:.1f}%) of the {total['delta_percentage_points']:.2f} pp "
        f"BA difference is assigned to the training side and {aggregation['delta_percentage_points']:.2f} pp "
        f"({aggregation['share_of_total_delta_percent']:.1f}%) to adding S/W during aggregation. "
        "This is path-specific, not a complete factorial causal decomposition."
    )
    fields_cells = (
        "model_name",
        "cell_id",
        "training_scope",
        "aggregation_scope",
        "mean_sd",
        "participant_cluster_ci95_percent",
        "availability",
    )
    fields_contrasts = (
        "contrast_id",
        "metric",
        "delta_percentage_points",
        "participant_cluster_ci95_percentage_points",
        "raw_p_value",
        "holm_adjusted_p_value",
        "share_of_total_delta_percent",
    )
    figure_markdown = ""
    figure_html = ""
    if figures_written:
        figure_markdown = """
## Paired figures

![Role-scope matrix](figures/role_scope_matrix.png)

![Path contributions](figures/decomposition_contributions.png)

![Repeat contributions](figures/paired_repeat_contributions.png)

![Per-class F1](figures/per_class_f1.png)
"""
        figure_html = """<img src="figures/role_scope_matrix.png" alt="Role-scope matrix"><img src="figures/decomposition_contributions.png" alt="Decomposition contributions"><img src="figures/paired_repeat_contributions.png" alt="Repeat contributions"><img src="figures/per_class_f1.png" alt="Per-class F1">"""
    absolute_markdown = "\n\n".join(
        f"## Absolute {METRIC_LABELS[metric]}\n\n"
        + _markdown_table(selected_cells[metric], fields_cells)
        for metric in plan.metrics
    )
    absolute_html = "".join(
        f"<h2>Absolute {html_escape(METRIC_LABELS[metric])}</h2>"
        + _html_table(selected_cells[metric], fields_cells)
        for metric in plan.metrics
    )
    per_class_f1 = [
        row for row in per_class_summary_rows if row["metric"] == "f1"
    ]
    per_class_fields = ("model_name", "class_name", "mean_sd")
    markdown = f"""# Training-role / aggregation-role decomposition

## Status

- Scientific role: `{SCIENTIFIC_ROLE}`.
- No classifier was retrained; only persisted OOF probabilities were read.
- The fourth cell is unavailable and was not imputed.
- Line-B gives equal weight to B/R/S/W role families. Dynamic S+W weight is therefore **{100.0 * dynamic_weight:.1f}%**, not 44.4%.

{absolute_markdown}

## Per-class F1

{_markdown_table(per_class_f1, per_class_fields)}

## Paired decomposition

{_markdown_table(selected_contrasts, fields_contrasts)}

## Main interpretation

{conclusion}

`C−A` is a training-scope bundle: it includes the additional S/W training windows, row-count-derived class weights, any training-scope-dependent fold-fitted preprocessing/state, and the resulting optimization trajectory. It is not a pure representation-learning effect. `D−C` holds the fitted all-role models fixed and changes only participant aggregation, so it is the clean aggregation contrast.

The static-trained models have no persisted S/W predictions and no saved DL checkpoints. Consequently, the static-train/all-role-aggregate cell and the factorial interaction cannot be recovered without retraining the 25 static CV models or locating an external checkpoint archive.

The paired intervals use a stratified participant-cluster bootstrap: each sampled participant carries all {len(plan.expected_repeats)} repeat predictions. BA/F1 P values use a two-sided paired participant-cluster permutation test; Holm correction is applied separately within each metric over the three declared post-hoc contrasts. This post-hoc analysis remains conditional on an adaptively reused {plan.expected_participants}-participant development cohort.

## Provenance

{_markdown_table(provenance_rows, ('source_name', 'case_id', 'file_oof_rows', 'participant_oof_rows', 'file_oof_sha256', 'checkpoint_count'))}

## Rank-label evidence

{_markdown_table(ranking_rows, ('status', 'source_scope', 'merged_case_id', 'expected_rank', 'observed_rank', 'file_oof_sha256'))}

Complete resolved configurations are copied under `source_configs/`; their flattened comparison is `tables/source_parameters.csv`, and the test-component registry is `tables/test_components.csv`. The 25 paired fold-level class counts, class weights, training-roster checks, and preprocessing hashes are in `tables/training_runtime_comparison.csv`.

{figure_markdown}
"""
    (output_dir / "STUDY_SUMMARY.md").write_text(markdown, encoding="utf-8")
    interpretation = f"""# Result interpretation

{conclusion}

The observed point decomposition is numerically dominated by the aggregation-side component, but uncertainty is wide and no definitive component attribution is supported. It does not establish that dynamic data teach no useful representation: the training-side CI includes both losses and gains, and the missing fourth cell prevents estimation of a train×aggregation interaction or a path-order-independent/Shapley attribution.

The exact numerical result is descriptive post-hoc evidence. No contrast remains significant after the applied Holm correction across all three declared post-hoc contrasts within BA or Macro-F1.
"""
    (output_dir / "RESULT_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    methods = f"""# Methods

1. Read the completed root-level file and participant OOF parquet for the static-trained and all-role-trained Full InceptionTime cases.
2. Verify identical participant/repeat/fold/split/training-seed/label rosters, exact shared B/R file identities, and resolved configs differing only in `config_id`, concrete roles, and classifier role families.
3. When ranking evidence is declared, verify each expected leaderboard rank and require the merged file OOF, participant OOF, and resolved config to be byte-exact copies of the analysed source case.
4. Replay Line-B from file OOF. Line-B is file→canonical role family→participant, with ordinary probability means and equal available role-family weights.
5. Verify complete-role replay reproduces both persisted participant OOF probability arrays within absolute tolerance `{plan.probability_tolerance:g}`.
6. Filter the all-role-trained file OOF to `{list(plan.static_aggregation_roles)}` and reaggregate to obtain cell C.
7. Compute each metric within each repeat and average repeats equally. Repeat SD uses sample SD (`ddof=1`).
8. Compute {plan.bootstrap_resamples:,} stratified participant-cluster bootstrap draws and {plan.permutation_resamples:,} paired participant-cluster permutations with seed {plan.inference_seed}. A participant carries all repeats in every draw.
9. Apply Holm correction separately within BA and Macro-F1 over the three declared post-hoc contrasts.

The decomposition is the algebraic path identity `D−A=(C−A)+(D−C)` on each metric. It is not a unique causal decomposition because cell B is unavailable.

References: Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, DOI `10.1007/978-1-4899-4541-9`; Phipson & Smyth (2010), permutation P-value plus-one correction, DOI `10.2202/1544-6115.1585`; Holm (1979), sequentially rejective multiple testing, DOI `10.2307/4615733`.
"""
    (output_dir / "REPORT_METHODS.md").write_text(methods, encoding="utf-8")
    components = f"""# Test components

- Static-trained model: `{plan.static_source.case_id}`; training roles `{list(plan.static_source.training_role_families)}`.
- All-role-trained model: `{plan.all_role_source.case_id}`; training roles `{list(plan.all_role_source.training_role_families)}`.
- Static aggregation roles: `{list(plan.static_aggregation_roles)}`.
- All-role aggregation roles: `{list(plan.all_aggregation_roles)}`.
- Aggregator: `{plan.balance_line}`.
- Metrics: `{list(plan.metrics)}`.
- Classifier training: frozen completed OOF products; no fitting in this study.
- Missing cell policy: fail closed and report N/A; never impute predictions.
"""
    (output_dir / "TEST_COMPONENTS.md").write_text(components, encoding="utf-8")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Role-scope decomposition</title>
<style>body{{font-family:sans-serif;max-width:1200px;margin:2rem auto;line-height:1.45}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #aaa;padding:.4rem}}.warning{{border:2px solid #b36b00;background:#fff7e8;padding:1rem}}img{{max-width:100%;height:auto}}</style></head><body>
<h1>Training-role / aggregation-role decomposition</h1><div class="warning"><strong>Post-hoc, prediction-locked development analysis.</strong> No retraining; fourth cell unavailable; no causal main-effect claim.</div>
{absolute_html}
<h2>Per-class F1</h2>{_html_table(per_class_f1, per_class_fields)}
<h2>Paired decomposition</h2>{_html_table(selected_contrasts, fields_contrasts)}
<h2>Interpretation</h2><p>{html_escape(conclusion)}</p><p>Line-B dynamic role-family weight: {100.0 * dynamic_weight:.1f}%.</p>
{figure_html}
<h2>Provenance</h2>{_html_table(provenance_rows, ('source_name','case_id','file_oof_rows','participant_oof_rows','file_oof_sha256','checkpoint_count'))}
<h2>Rank-label evidence</h2>{_html_table(ranking_rows, ('status','source_scope','merged_case_id','expected_rank','observed_rank','file_oof_sha256'))}</body></html>"""
    (output_dir / "STUDY_SUMMARY.html").write_text(html, encoding="utf-8")


def _provenance_rows(sources: Sequence[LoadedSource]) -> list[dict[str, Any]]:
    return [
        {
            "source_name": source.spec.name,
            "case_id": source.spec.case_id,
            "study_dir": str(source.spec.study_dir),
            "experiment_root": str(source.experiment_root),
            "training_role_families": list(source.spec.training_role_families),
            "file_oof_rows": len(source.file_rows),
            "participant_oof_rows": len(source.subject_rows),
            "file_oof_sha256": source.file_sha256,
            "participant_oof_sha256": source.subject_sha256,
            "resolved_config_sha256": source.config_sha256,
            "checkpoint_count": len(source.checkpoint_paths),
            "checkpoint_paths": [str(path) for path in source.checkpoint_paths],
        }
        for source in sources
    ]


def _ranking_evidence_rows(
    plan: RoleScopePlan,
    static: LoadedSource,
    all_role: LoadedSource,
) -> list[dict[str, Any]]:
    evidence = plan.ranking_evidence
    if evidence is None:
        return [{"status": "ranking_evidence_not_declared"}]
    if not evidence.leaderboard_path.is_file():
        raise FileNotFoundError(f"ranking leaderboard not found: {evidence.leaderboard_path}")
    with evidence.leaderboard_path.open("r", encoding="utf-8", newline="") as stream:
        leaderboard = list(csv.DictReader(stream))
    by_case: dict[str, Mapping[str, str]] = {}
    for row in leaderboard:
        case_id = str(row.get("case_id", ""))
        if case_id in by_case:
            raise ValueError(f"duplicate leaderboard case id: {case_id}")
        by_case[case_id] = row
    merged_root = evidence.leaderboard_path.parent.parent
    definitions = (
        (
            "static_training",
            static,
            evidence.static_case_id,
            evidence.static_expected_rank,
        ),
        (
            "all_role_training",
            all_role,
            evidence.all_role_case_id,
            evidence.all_role_expected_rank,
        ),
    )
    output: list[dict[str, Any]] = []
    for scope, source, merged_case_id, expected_rank in definitions:
        row = by_case.get(merged_case_id)
        if row is None:
            raise ValueError(f"ranking case missing from leaderboard: {merged_case_id}")
        observed_rank = int(row["predictive_rank"])
        if observed_rank != expected_rank:
            raise ValueError(
                f"ranking evidence changed for {merged_case_id}: "
                f"expected {expected_rank}, observed {observed_rank}"
            )
        merged_case_root = merged_root / "cases" / merged_case_id
        merged_file = merged_case_root / "artifacts/oof_file_predictions.parquet"
        merged_subject = merged_case_root / "artifacts/oof_subject_predictions.parquet"
        merged_config = merged_case_root / "resolved_config.yaml"
        for path in (merged_file, merged_subject, merged_config):
            if not path.is_file():
                raise FileNotFoundError(f"ranking evidence artifact not found: {path}")
        merged_file_sha = _sha256(merged_file)
        merged_subject_sha = _sha256(merged_subject)
        merged_config_sha = _sha256(merged_config)
        if (
            merged_file_sha != source.file_sha256
            or merged_subject_sha != source.subject_sha256
            or merged_config_sha != source.config_sha256
        ):
            raise ValueError(f"merged ranking case is not an exact copy of source {scope}")
        output.append(
            {
                "status": "verified_exact_source_copy",
                "source_scope": scope,
                "source_case_id": source.spec.case_id,
                "merged_case_id": merged_case_id,
                "expected_rank": expected_rank,
                "observed_rank": observed_rank,
                "primary_ranking_metric": row.get("primary_ranking_metric", ""),
                "ranking_interpretation": row.get("ranking_interpretation", ""),
                "leaderboard_path": str(evidence.leaderboard_path),
                "leaderboard_sha256": _sha256(evidence.leaderboard_path),
                "file_oof_sha256": merged_file_sha,
                "participant_oof_sha256": merged_subject_sha,
                "resolved_config_sha256": merged_config_sha,
            }
        )
    return output


def _flatten_config(value: Any, *, prefix: str = "") -> dict[str, Any]:
    """Flatten mappings while preserving lists as one auditable parameter value."""

    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_config(value[key], prefix=child))
        return flattened
    return {prefix: value}


def _parameter_text(value: Any) -> str:
    if value is None:
        return "<absent>"
    if isinstance(value, (list, tuple, dict, bool, int, float)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _source_parameter_rows(
    static: LoadedSource,
    all_role: LoadedSource,
) -> list[dict[str, Any]]:
    left = _flatten_config(
        _mapping(
            yaml.safe_load(static.resolved_config_path.read_text(encoding="utf-8")),
            "static resolved config",
        )
    )
    right = _flatten_config(
        _mapping(
            yaml.safe_load(all_role.resolved_config_path.read_text(encoding="utf-8")),
            "all-role resolved config",
        )
    )
    expected_differences = {
        "config_id",
        "roles",
        "training.classifier_role_families",
    }
    output: list[dict[str, Any]] = []
    for parameter in sorted(set(left) | set(right)):
        left_value = left.get(parameter)
        right_value = right.get(parameter)
        if left_value == right_value:
            comparison = "identical"
        elif parameter in expected_differences:
            comparison = "declared_role_scope_difference"
        else:  # The fail-closed source-pair check should make this unreachable.
            comparison = "unexpected_difference"
        output.append(
            {
                "parameter_path": parameter,
                "static_training": _parameter_text(left_value),
                "all_role_training": _parameter_text(right_value),
                "comparison": comparison,
            }
        )
    output.append(
        {
            "parameter_path": "training.runtime_class_weight_effect",
            "static_training": "derived from static-role outer-training rows",
            "all_role_training": "derived from all-role outer-training rows",
            "comparison": "consequence_of_class_count_basis_row_and_role_scope",
        }
    )
    return output


def _fold_runtime_map(source: LoadedSource) -> dict[tuple[int, int], Mapping[str, Any]]:
    output: dict[tuple[int, int], Mapping[str, Any]] = {}
    for path in sorted(
        source.experiment_root.glob("repeat_*_fold_*/metrics_per_fold_seed.json")
    ):
        payload = _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
        cells = payload.get("cells")
        if not isinstance(cells, list) or len(cells) != 1 or not isinstance(cells[0], Mapping):
            raise ValueError(f"fold runtime artifact must contain exactly one cell: {path}")
        cell = cells[0]
        key = (int(cell["repeat_index"]), int(cell["fold_index"]))
        if key in output:
            raise ValueError(f"duplicate fold runtime artifact for {source.spec.name}: {key}")
        output[key] = cell
    return output


def _training_runtime_rows(
    static: LoadedSource,
    all_role: LoadedSource,
) -> list[dict[str, Any]]:
    left = _fold_runtime_map(static)
    right = _fold_runtime_map(all_role)
    if not left and not right:
        return [
            {
                "status": "fold_runtime_artifacts_not_available",
                "static_source": static.spec.name,
                "all_role_source": all_role.spec.name,
            }
        ]
    if set(left) != set(right):
        raise ValueError("static/all-role fold runtime artifact rosters differ")
    output: list[dict[str, Any]] = []
    for repeat, fold in sorted(left):
        static_cell = left[(repeat, fold)]
        all_cell = right[(repeat, fold)]
        static_fit = _mapping(static_cell.get("fitted_provenance"), "static fitted provenance")
        all_fit = _mapping(all_cell.get("fitted_provenance"), "all-role fitted provenance")
        static_participants = tuple(str(value) for value in static_fit["fitted_participant_ids"])
        all_participants = tuple(str(value) for value in all_fit["fitted_participant_ids"])
        if static_participants != all_participants:
            raise ValueError(f"outer-training participant roster differs for {(repeat, fold)}")
        output.append(
            {
                "status": "paired",
                "repeat": repeat,
                "fold": fold,
                "split_seed": int(static_cell["split_seed"]),
                "training_seed": int(static_cell["training_seed"]),
                "outer_training_participant_count": len(static_participants),
                "outer_membership_hash_equal": (
                    static_fit.get("outer_membership_hash")
                    == all_fit.get("outer_membership_hash")
                ),
                "static_class_counts": _parameter_text(static_fit.get("class_counts")),
                "all_role_class_counts": _parameter_text(all_fit.get("class_counts")),
                "static_class_weight_vector": _parameter_text(
                    static_fit.get("class_weight_vector")
                ),
                "all_role_class_weight_vector": _parameter_text(
                    all_fit.get("class_weight_vector")
                ),
                "class_count_basis": str(static_fit.get("class_count_basis")),
                "static_preprocessing_hash": str(static_cell.get("preprocessing_hash")),
                "all_role_preprocessing_hash": str(all_cell.get("preprocessing_hash")),
                "preprocessing_hash_equal": (
                    static_cell.get("preprocessing_hash")
                    == all_cell.get("preprocessing_hash")
                ),
                "static_retained_train_record_count": int(
                    static_cell["retained_train_record_count"]
                ),
                "all_role_retained_train_record_count": int(
                    all_cell["retained_train_record_count"]
                ),
            }
        )
    return output


def _component_rows(plan: RoleScopePlan) -> list[dict[str, Any]]:
    return [
        {
            "component_type": "classifier",
            "component_id": plan.static_source.case_id,
            "input_data": (
                f"{plan.expected_participants} participants; "
                f"{list(plan.static_source.training_role_families)} role families; "
                "persisted file/participant OOF"
            ),
            "fixed_parameters": "complete source config: source_configs/static_training_resolved_config.yaml",
            "study_action": "read predictions only; no fitting",
        },
        {
            "component_type": "classifier",
            "component_id": plan.all_role_source.case_id,
            "input_data": (
                f"{plan.expected_participants} participants; "
                f"{list(plan.all_role_source.training_role_families)} role families; "
                "persisted file/participant OOF"
            ),
            "fixed_parameters": "complete source config: source_configs/all_role_training_resolved_config.yaml",
            "study_action": "read predictions only; no fitting",
        },
        {
            "component_type": "aggregation",
            "component_id": plan.balance_line,
            "input_data": "persisted file-level class-probability vectors",
            "fixed_parameters": "file mean -> canonical role-family mean -> equal role-family mean; quality weighting off",
            "study_action": "exact replay and B/R-only post-hoc reaggregation",
        },
        {
            "component_type": "inference",
            "component_id": "participant_cluster_inference",
            "input_data": (
                f"{plan.expected_participants} paired participant clusters carrying "
                f"all {len(plan.expected_repeats)} repeats"
            ),
            "fixed_parameters": (
                f"bootstrap={plan.bootstrap_resamples}; permutation={plan.permutation_resamples}; "
                f"seed={plan.inference_seed}; alpha={plan.alpha}; Holm within metric"
            ),
            "study_action": "paired uncertainty and hypothesis testing",
        },
    ]


def _backup(output_dir: Path) -> None:
    backup = output_dir / "result_backup"
    backup.mkdir(parents=True, exist_ok=False)
    included: list[dict[str, Any]] = []
    for relative in (
        "study_plan.yaml",
        "STUDY_SUMMARY.md",
        "STUDY_SUMMARY.html",
        "RESULT_INTERPRETATION.md",
        "REPORT_METHODS.md",
        "TEST_COMPONENTS.md",
        "study_summary.json",
    ):
        source = output_dir / relative
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for directory in ("tables", "figures", "derived_oof", "source_configs"):
        source = output_dir / directory
        if source.is_dir():
            shutil.copytree(source, backup / directory)
    for path in sorted(backup.rglob("*")):
        if path.is_file():
            included.append(
                {
                    "path": str(path.relative_to(backup)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    (backup / "backup_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ppg_frailty.result_backup.v1",
                "file_count": len(included),
                "files": included,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _outputs_index(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "outputs_index.json":
            rows.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    (output_dir / "outputs_index.json").write_text(
        json.dumps(
            {
                "schema_version": "ppg_frailty.outputs_index.v1",
                "file_count": len(rows),
                "files": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run_role_scope_decomposition(
    plan_path: str | Path,
    *,
    pipeline_root: str | Path,
    output_root: str | Path,
) -> Path:
    """Run a prediction-locked three-cell role-scope decomposition."""

    plan = load_role_scope_plan(plan_path, pipeline_root=pipeline_root)
    static = _load_source(plan.static_source, plan)
    all_role = _load_source(plan.all_role_source, plan)
    _assert_source_pair(static, all_role, plan)

    replay_static = _aggregate_roles(
        static.file_rows, plan.static_aggregation_roles, balance_line=plan.balance_line
    )
    replay_all = _aggregate_roles(
        all_role.file_rows, plan.all_aggregation_roles, balance_line=plan.balance_line
    )
    static_replay_error = _assert_probability_replay(
        static.subject_rows,
        replay_static,
        tolerance=plan.probability_tolerance,
        source_name=static.spec.name,
    )
    all_replay_error = _assert_probability_replay(
        all_role.subject_rows,
        replay_all,
        tolerance=plan.probability_tolerance,
        source_name=all_role.spec.name,
    )
    cross_static = _aggregate_roles(
        all_role.file_rows,
        plan.static_aggregation_roles,
        balance_line=plan.balance_line,
    )
    _validate_subject_rows(cross_static, plan=plan, source_name="derived_cross_static")
    cells = {
        "A_static_train_static_aggregate": static.subject_rows,
        "C_all_train_static_aggregate": cross_static,
        "D_all_train_all_aggregate": all_role.subject_rows,
    }
    cell_rows, repeat_rows = _cell_metric_rows(cells, plan)
    contrast_rows = _contrast_rows(cells, plan)
    per_class_rows, confusion_rows = _per_class_rows(cells, plan)
    per_class_summary_rows = _per_class_summary_rows(per_class_rows, plan)
    provenance_rows = _provenance_rows((static, all_role))
    ranking_rows = _ranking_evidence_rows(plan, static, all_role)
    parameter_rows = _source_parameter_rows(static, all_role)
    runtime_rows = _training_runtime_rows(static, all_role)
    component_rows = _component_rows(plan)
    dynamic_weight = (
        len(set(plan.all_aggregation_roles) - set(plan.static_aggregation_roles))
        / len(plan.all_aggregation_roles)
    )

    now = datetime.now().astimezone()
    output_dir = Path(output_root).resolve() / f"{now:%Y%m%d_%H%M%S}_{plan.output_slug}"
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(plan.plan_path, output_dir / "study_plan.yaml")
    source_configs = output_dir / "source_configs"
    source_configs.mkdir(parents=True, exist_ok=False)
    shutil.copy2(
        static.resolved_config_path,
        source_configs / "static_training_resolved_config.yaml",
    )
    shutil.copy2(
        all_role.resolved_config_path,
        source_configs / "all_role_training_resolved_config.yaml",
    )
    tables = {
        "factorial_cells": cell_rows,
        "repeat_metrics": repeat_rows,
        "contrasts": contrast_rows,
        "per_class": per_class_rows,
        "per_class_summary": per_class_summary_rows,
        "confusions": confusion_rows,
        "provenance": provenance_rows,
        "ranking_evidence": ranking_rows,
        "source_parameters": parameter_rows,
        "training_runtime_comparison": runtime_rows,
        "test_components": component_rows,
    }
    for name, rows in tables.items():
        _write_rows(output_dir / "tables" / f"{name}.csv", rows)
    pairs = (
        [
            {"table": "factorial_cells.csv", "figure": "role_scope_matrix.png"},
            {"table": "contrasts.csv", "figure": "decomposition_contributions.png"},
            {"table": "repeat_metrics.csv", "figure": "paired_repeat_contributions.png"},
            {"table": "per_class_summary.csv", "figure": "per_class_f1.png"},
        ]
        if plan.write_static_figures
        else []
    )
    _write_rows(output_dir / "tables" / "table_figure_pairs.csv", pairs)
    write_table_column_definitions(
        output_dir / "tables",
        csv_directory=output_dir / "tables",
    )
    if plan.write_excel_workbook:
        write_excel_workbook_from_csv_directory(
            output_dir / "tables" / "report_tables.xlsx",
            output_dir / "tables",
        )
    derived_oof_path = write_oof_parquet(
        cross_static,
        output_dir
        / "derived_oof"
        / "all_role_trained_static_aggregation_subject_predictions.parquet",
    )
    (output_dir / "derived_oof" / "derivation_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "ppg_frailty.posthoc_oof_derivation.v1",
                "scientific_role": SCIENTIFIC_ROLE,
                "no_retraining": True,
                "source_file_oof": str(all_role.file_prediction_path),
                "source_file_oof_sha256": all_role.file_sha256,
                "source_config_sha256": all_role.config_sha256,
                "study_plan_sha256": _sha256(plan.plan_path),
                "selected_role_families": list(plan.static_aggregation_roles),
                "aggregation_rule": plan.balance_line,
                "quality_weighting": False,
                "derived_oof_sha256": _sha256(derived_oof_path),
                "row_count": len(cross_static),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if plan.write_static_figures:
        _plots(
            output_dir,
            cell_rows,
            contrast_rows,
            repeat_rows,
            per_class_summary_rows,
            plan,
        )
    _write_reports(
        output_dir,
        plan,
        cell_rows,
        contrast_rows,
        per_class_summary_rows,
        provenance_rows,
        ranking_rows,
        dynamic_weight,
        plan.write_static_figures,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "study_id": plan.study_id,
        "scientific_role": SCIENTIFIC_ROLE,
        "status": "complete_three_observable_cells_fourth_cell_unavailable",
        "no_retraining": True,
        "selection_eligible": False,
        "path": "A_static_static_to_C_all_static_to_D_all_all",
        "dynamic_role_family_weight": dynamic_weight,
        "fourth_cell_available": False,
        "fourth_cell_reason": (
            "static-trained OOF contains B/R only and the source case has no persisted "
            "DL checkpoint for S/W inference"
        ),
        "static_replay_max_abs_probability_error": static_replay_error,
        "all_role_replay_max_abs_probability_error": all_replay_error,
        "cell_metrics": cell_rows,
        "contrasts": contrast_rows,
        "source_provenance": provenance_rows,
        "ranking_evidence": ranking_rows,
    }
    (output_dir / "study_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    if plan.write_result_backup:
        _backup(output_dir)
    _outputs_index(output_dir)
    return output_dir


__all__ = [
    "LoadedSource",
    "RoleScopePlan",
    "SCHEMA_VERSION",
    "SCIENTIFIC_ROLE",
    "SourceSpec",
    "load_role_scope_plan",
    "run_role_scope_decomposition",
]

"""Read-only seed, split, and artifact reproducibility audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


PASS, FAIL, NOT_VERIFIABLE = "PASS", "FAIL", "NOT_VERIFIABLE"


@dataclass(frozen=True)
class ReproducibilityAudit:
    schema_version: str
    status: str
    summary: Mapping[str, Any]
    case_rows: tuple[Mapping[str, Any], ...]
    cell_rows: tuple[Mapping[str, Any], ...]
    split_rows: tuple[Mapping[str, Any], ...]
    issues: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _int(value: Any) -> int | None:
    try:
        return None if value is None or isinstance(value, bool) else int(value)
    except (TypeError, ValueError):
        return None


def audit_study_reproducibility(collected: Any) -> ReproducibilityAudit:
    """Summarize evidence once; this audit never blocks analysis."""
    manifest = collected.manifest if isinstance(collected.manifest, Mapping) else {}
    plan = collected.plan if isinstance(collected.plan, Mapping) else {}
    cases = [row for row in manifest.get("cases", ()) if isinstance(row, Mapping)]
    records = {str(row.get("case_id")): row for row in collected.case_records}
    cells = []
    seen: set[tuple[str, int, int]] = set()
    issues = []
    for source in collected.cell_rows:
        case, repeat, fold = str(source.get("case_id", "")), _int(source.get("repeat")), _int(source.get("fold"))
        if repeat is None or fold is None:
            continue
        key = (case, repeat, fold)
        if key in seen:
            issues.append(
                {
                    "severity": "error",
                    "code": "duplicate_cell",
                    "case_id": case,
                    "repeat": repeat,
                    "fold": fold,
                    "message": "multiple cell metric rows",
                }
            )
        seen.add(key)
        prediction_rows = [
            row
            for row in collected.subject_oof_rows
            if str(row.get("case_id")) == case and _int(row.get("repeat")) == repeat and _int(row.get("fold")) == fold
        ]
        cells.append(
            {
                "case_id": case,
                "repeat": repeat,
                "fold": fold,
                "split_seed": source.get("split_seed"),
                "training_seed": source.get("training_seed"),
                "status": source.get("status"),
                "participant_prediction_count": len(prediction_rows),
                "config_hashes": sorted(
                    {str(row.get("config_hash")) for row in prediction_rows if row.get("config_hash")}
                ),
                "fold_hashes": sorted({str(row.get("fold_hash")) for row in prediction_rows if row.get("fold_hash")}),
            }
        )
    execution = plan.get("execution", {}) if isinstance(plan.get("execution"), Mapping) else {}
    repeats = (
        tuple(map(int, execution.get("repeats", ()))) if isinstance(execution.get("repeats"), (list, tuple)) else ()
    )
    folds = tuple(map(int, execution.get("folds", ()))) if isinstance(execution.get("folds"), (list, tuple)) else ()
    case_ids = tuple(str(row.get("case_id")) for row in cases)
    if not case_ids:
        issues.append(
            {
                "severity": "not_verifiable",
                "code": "case_roster_unavailable",
                "message": "manifest case roster is unavailable",
            }
        )
    planned = {(case, repeat, fold) for case in case_ids for repeat in repeats for fold in folds}
    for case, repeat, fold in sorted(planned - seen):
        issues.append(
            {
                "severity": "not_verifiable",
                "code": "cell_not_observed",
                "case_id": case,
                "repeat": repeat,
                "fold": fold,
                "message": "planned cell has no metric row",
            }
        )
    split_rows = []
    for repeat, fold in sorted({(row["repeat"], row["fold"]) for row in cells}):
        selected = [row for row in cells if row["repeat"] == repeat and row["fold"] == fold]
        split_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "case_ids": sorted({str(row["case_id"]) for row in selected}),
                "split_seeds": sorted(
                    {int(row["split_seed"]) for row in selected if _int(row.get("split_seed")) is not None}
                ),
                "fold_hashes": sorted({value for row in selected for value in row["fold_hashes"]}),
            }
        )
    case_rows = tuple(
        {
            "case_id": case,
            "status": records.get(case, {}).get("status", "not_observed"),
            "planned_cell_count": len(repeats) * len(folds) if repeats and folds else None,
            "observed_cell_count": sum(row["case_id"] == case for row in cells),
            "participant_prediction_count": sum(
                int(row["participant_prediction_count"]) for row in cells if row["case_id"] == case
            ),
        }
        for case in case_ids
    )
    status = FAIL if any(row["severity"] == "error" for row in issues) else NOT_VERIFIABLE if issues else PASS
    summary = {
        "audit_status": status,
        "planned_case_count": len(case_ids),
        "planned_repeats": list(repeats) or None,
        "planned_folds": list(folds) or None,
        "planned_cell_count": len(planned) if planned else None,
        "observed_cell_count": len(cells),
        "error_count": sum(row["severity"] == "error" for row in issues),
        "not_verifiable_count": sum(row["severity"] != "error" for row in issues),
        "scope": "persisted_manifest_cell_and_prediction_evidence",
        "training_or_report_gate": False,
    }
    return ReproducibilityAudit(
        "ppg_frailty.reporting.reproducibility_audit.v2",
        status,
        summary,
        case_rows,
        tuple(cells),
        tuple(split_rows),
        tuple(issues),
    )


__all__ = ["PASS", "FAIL", "NOT_VERIFIABLE", "ReproducibilityAudit", "audit_study_reproducibility"]

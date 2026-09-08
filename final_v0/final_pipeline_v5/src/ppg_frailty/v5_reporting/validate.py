"""Single input-validation boundary for artifact-only analysis."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import LoadedReportData, ReportContractError, ReportRequest, ValidationIssue, ValidationReport


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    output: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        output.update(_flatten(item, path))
    return output


def changed_config_paths(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[str, ...]:
    left, right = _flatten(reference), _flatten(candidate)
    ignored = {"config_id", "case_id", "name"}
    return tuple(
        sorted(path for path in left.keys() | right.keys()
               if path.split(".")[-1] not in ignored and left.get(path) != right.get(path)))


def _roster(rows: tuple[Mapping[str, Any], ...], case: str) -> set[tuple[Any, ...]]:
    return {(str(row.get("participant_id")), int(row.get("repeat", 0)), int(row.get("fold", 0)), int(row.get("label")))
            for row in rows if str(row.get("case_id")) == case}


def validate_report_data(data: LoadedReportData, request: ReportRequest) -> ValidationReport:
    issues: list[ValidationIssue] = []
    cases = data.case_ids
    if set(cases) != set(data.evaluation_scope_by_case):
        raise ReportContractError("loaded case roster is inconsistent")
    if request.mode == "test":
        outer = [case for case in cases if data.evaluation_scope_by_case[case] != "independent_test"]
        if outer:
            raise ReportContractError(f"test mode rejects outer-OOF input: {outer}")
        for case in cases:
            evidence = data.manifest_by_case.get(case, {}).get("independent_test_evidence")
            if not isinstance(evidence, Mapping) or evidence.get("rosters_disjoint") is not True:
                raise ReportContractError(f"independent-test evidence is incomplete for {case}")
    elif any(data.evaluation_scope_by_case[case] != "outer_oof" for case in cases):
        raise ReportContractError("single/comparison/ablation modes require outer-OOF inputs")

    records = {(row.case_id, row.layer): row for row in data.artifact_records}
    for case in cases:
        participant = tuple(row for row in data.layer_rows["participant"] if str(row.get("case_id")) == case)
        if not participant:
            raise ReportContractError(f"participant predictions are required for {case}")
        ensemble = any(
            str(row.get("prediction_kind", "")) != "single_model" or row.get("member_training_seeds")
            for row in participant)
        member_record = records.get((case, "member"))
        if member_record is None:
            if data.source_kind == "v2_study" and case in data.legacy_v2_cases and not ensemble:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "legacy_v2_member_artifact_absent_non_ensemble",
                        "verified single-model V2 study predates the explicit empty member artifact",
                        case,
                    ))
            else:
                raise ReportContractError(f"{case} requires a member artifact")
        if ensemble and not data.layer_rows["member"]:
            raise ReportContractError(f"ensemble case {case} requires a member artifact with predictions")
        keys = [(row.get("participant_id"), row.get("repeat"), row.get("fold")) for row in participant]
        if len(keys) != len(set(keys)):
            raise ReportContractError(f"duplicate participant prediction key in {case}")

    if request.mode in {"comparison", "ablation"}:
        reference = str(request.reference_case)
        expected = _roster(data.layer_rows["participant"], reference)
        for case in cases:
            if case != reference and _roster(data.layer_rows["participant"], case) != expected:
                raise ReportContractError(f"paired roster mismatch: {reference} versus {case}")
    if request.mode == "ablation":
        reference = data.config_by_case.get(str(request.reference_case), {})
        declared = set(request.factor_paths)
        for case in cases:
            if case == request.reference_case:
                continue
            changed = set(changed_config_paths(reference, data.config_by_case.get(case, {})))
            undeclared = changed - declared
            if undeclared:
                raise ReportContractError(f"ablation {case} changes undeclared parameter paths: {sorted(undeclared)}")
            missing = declared - changed
            if missing:
                raise ReportContractError(f"ablation {case} does not change declared factor paths: {sorted(missing)}")

    status = "passed_with_warnings" if issues else "passed"
    return ValidationReport(status, tuple(issues), cases, len(data.artifact_records),
                            sum(len(rows) for rows in data.layer_rows.values()))


__all__ = ["changed_config_paths", "validate_report_data"]

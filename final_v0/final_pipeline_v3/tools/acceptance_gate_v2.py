#!/usr/bin/env python3
"""Non-scientific V2 release gate: contracts, configs, CLI, and safe tests only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _historical_inventory_check() -> dict[str, Any]:
    archive = ROOT / "historical/v1_transition"
    inventory_path = archive / "INVENTORY.json"
    try:
        payload = json.loads(
            inventory_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {value}")
            ),
        )
        expected = payload["files"]
        observed = []
        for path in sorted(archive.rglob("*")):
            if path.is_file() and path != inventory_path:
                observed.append(
                    {
                        "path": path.relative_to(archive).as_posix(),
                        "bytes": path.stat().st_size,
                        "archived_content_sha256": _sha(path),
                        "disposition": "inactive_v1_transition_provenance_only",
                    }
                )
        canonical = json.dumps(
            observed, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        tree_hash = hashlib.sha256(canonical).hexdigest()
        valid = (
            payload.get("active_use_prohibited") is True
            and payload.get("source_byte_equivalence_claim") is False
            and payload.get("scientific_evidence_claim") is False
            and int(payload.get("file_count", -1)) == len(observed)
            and expected == observed
            and payload.get("content_tree_sha256") == tree_hash
        )
        return {
            "name": "historical_v1_transition_inventory",
            "status": "passed" if valid else "failed",
            "file_count": len(observed),
            "content_tree_sha256": tree_hash,
            "inventory_sha256": _sha(inventory_path),
            "active_use_prohibited": True,
        }
    except Exception as error:
        return {
            "name": "historical_v1_transition_inventory",
            "status": "failed",
            "error": repr(error),
        }


def _command(command: list[str], *, environment: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "status": "passed" if completed.returncode == 0 else "failed",
        "duration_s": round(time.perf_counter() - started, 9),
        "stdout_sha256": hashlib.sha256(
            completed.stdout.encode("utf-8")
        ).hexdigest(),
        "stderr_sha256": hashlib.sha256(
            completed.stderr.encode("utf-8")
        ).hexdigest(),
        "stdout_tail": completed.stdout.splitlines()[-3:],
        "stderr_tail": completed.stderr.splitlines()[-3:],
    }


def _active_source_snapshot_sha256() -> str:
    """Use the same active-input hash as every scientific execution gate."""

    source_root = str(ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from ppg_frailty.scientific_gate import source_snapshot_sha256

    return source_snapshot_sha256(ROOT)


def _safe_suite_check(environment: dict[str, str]) -> dict[str, Any]:
    """Run safe tests and retain a hash-bound report summary in acceptance."""

    with tempfile.TemporaryDirectory(prefix=".v2_safe_report_", dir=ROOT) as temporary:
        report_path = Path(temporary) / "safe_test_report_v2.json"
        result = _command(
            [
                sys.executable,
                str(ROOT / "tools/run_test_suite.py"),
                "--suite",
                "safe",
                "--verbosity",
                "1",
                "--report",
                str(report_path),
            ],
            environment=environment,
        )
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            report_evidence = {
                "sha256": _sha(report_path),
                "schema_version": payload.get("schema_version"),
                "suite": payload.get("suite"),
                "status": payload.get("status"),
                "counts": payload.get("counts"),
                "test_source_snapshot": payload.get("test_source_snapshot"),
            }
        except (OSError, ValueError, json.JSONDecodeError) as error:
            report_evidence = {"error": repr(error)}
        result["command"][-1] = "<TEMP_SAFE_REPORT>"
        result["safe_report"] = report_evidence
        if (
            result["status"] != "passed"
            or report_evidence.get("status") != "passed"
            or report_evidence.get("suite") != "safe"
            or not isinstance(report_evidence.get("sha256"), str)
        ):
            result["status"] = "failed"
        return result


def _write(path: Path, payload: dict[str, Any]) -> None:
    target = path.resolve()
    target.relative_to(ROOT.resolve())
    if target.exists():
        raise FileExistsError(f"acceptance output overwrite forbidden: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _normalise_materialisation_index(
    payload: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Remove only the isolated root spelling before identity comparison."""

    spellings = (str(output_root), output_root.as_posix())

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): visit(item) for key, item in sorted(value.items())}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            normalised = value
            for spelling in spellings:
                normalised = normalised.replace(
                    spelling,
                    "<ISOLATED_OUTPUT_ROOT>",
                )
            return normalised
        return value

    return visit(payload)


def _isolated_data_materialisation_check(
    environment: dict[str, str],
) -> dict[str, Any]:
    """Materialise twice without recomputing splits and compare every index row."""

    materializer = ROOT / "tools/materialize_data_contracts.py"
    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".v2_acceptance_", dir=ROOT) as temporary:
        base = Path(temporary)
        for name in ("first", "second"):
            output_root = base / name
            command = [
                sys.executable,
                str(materializer),
                "--output-root",
                str(output_root),
                "--isolated-nonoverwrite",
            ]
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            try:
                index = json.loads(completed.stdout)
            except json.JSONDecodeError:
                index = {}
            runs.append(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "duration_s": round(time.perf_counter() - started, 9),
                    "index": index,
                    "normalised": _normalise_materialisation_index(
                        index,
                        output_root,
                    ),
                    "stderr_tail": completed.stderr.splitlines()[-3:],
                }
            )
    scientific_flags = (
        "training_executed",
        "ablation_executed",
        "ptt_benchmark_executed",
    )
    valid = all(
        run["returncode"] == 0
        and run["index"].get("status") == "passed"
        and all(run["index"].get(flag) is False for flag in scientific_flags)
        for run in runs
    )
    identical = (
        bool(runs[0]["normalised"])
        and runs[0]["normalised"] == runs[1]["normalised"]
    )
    return {
        "name": "isolated_build_data_identity_and_idempotence",
        "status": "passed" if valid and identical else "failed",
        "isolated_nonoverwrite": True,
        "frozen_split_recomputed": False,
        "materialisations_identical": identical,
        "artifact_count": len(
            runs[0]["index"].get("generated_artifacts", {})
        ),
        "report_count": len(runs[0]["index"].get("reports", {})),
        "runs": [
            {
                key: value
                for key, value in run.items()
                if key not in {"index", "normalised"}
            }
            for run in runs
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "src")
    active_source_before = _active_source_snapshot_sha256()
    checks = [
        _command(
            [sys.executable, str(ROOT / "tools/validate_v2.py")],
            environment=environment,
        ),
        _command(
            [
                sys.executable,
                "-m",
                "ppg_frailty.cli",
                "validate",
                "--all-configs",
            ],
            environment=environment,
        ),
        _safe_suite_check(environment),
    ]
    checks.append(_isolated_data_materialisation_check(environment))
    checks.append(_historical_inventory_check())
    checks.append(
        _command(
            [
                sys.executable,
                "-m",
                "unittest",
                (
                    "tests.models.test_architectures."
                    "ReviewedArchitectureTests."
                    "test_factory_covers_four_representation_modes"
                ),
            ],
            environment=environment,
        )
    )
    materializer = ROOT / "tools/materialize_data_contracts.py"
    required = {
        "materializer": materializer,
        "decision_profile": ROOT / "configs/v2_decision_profile.yaml",
        "dependency_locks": ROOT / "locks/profiles.lock.json",
        "internal_materialization_report": (
            ROOT / "reports/internal_manifest_v2_report.json"
        ),
        "historical_v1_inventory": (
            ROOT / "historical/v1_transition/INVENTORY.json"
        ),
    }
    forbidden_active = (
        ROOT / "tools/acceptance_gate.py",
        ROOT / "tools/run_cpu_ci.py",
        ROOT / "tools/validate_v1.py",
        ROOT / "tools/sync_tracking.py",
        ROOT / "reports/data_contract_report.json",
        ROOT / "reports/external_data_contract_report.json",
    )
    stale_active_evidence = sorted(
        path
        for base in (
            ROOT / "artifacts/acceptance",
            ROOT / "artifacts/test_reports",
        )
        if base.is_dir()
        for path in base.rglob("*.json")
        if "current" in path.name or "manual" in path.name
    )
    structural = {
        "status": (
            "passed"
            if all(path.is_file() for path in required.values())
            and not any(path.exists() for path in forbidden_active)
            and not stale_active_evidence
            else "failed"
        ),
        "artifacts": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha(path) if path.is_file() else None,
            }
            for name, path in required.items()
        },
        "build_data_executes_splitter": any(
            token in materializer.read_text(encoding="utf-8")
            for token in ("StratifiedGroupKFold(", "GroupKFold(", "train_test_split(")
        ),
        "forbidden_active_v1_paths": [
            path.relative_to(ROOT).as_posix()
            for path in forbidden_active
            if path.exists()
        ],
        "stale_active_current_or_manual_evidence": [
            path.relative_to(ROOT).as_posix()
            for path in stale_active_evidence
        ],
    }
    if structural["build_data_executes_splitter"]:
        structural["status"] = "failed"
    checks.append(structural)
    active_source_after = _active_source_snapshot_sha256()
    checks.append(
        {
            "name": "active_v2_source_snapshot_stable_during_acceptance",
            "status": (
                "passed"
                if active_source_before == active_source_after
                else "failed"
            ),
            "snapshot_algorithm": (
                "scientific_gate.source_snapshot_sha256(src,configs,tools,"
                "requirements,locks,pyproject.toml)"
            ),
            "before_sha256": active_source_before,
            "after_sha256": active_source_after,
            "git_commit_claim": False,
            "validity": "point_in_time_exact_active_source_bytes_only",
        }
    )
    passed = all(item["status"] == "passed" for item in checks)
    report = {
        "schema_version": "ppg_frailty.acceptance_gate.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "passed" if passed else "failed",
        "checks": checks,
        "scope": (
            "non_scientific_contract_config_cli_safe_tests_"
            "build_data_identity_four_mode_dispatch"
        ),
        "training_executed": False,
        "ablation_executed": False,
        "full_5x5_executed": False,
        "ptt_benchmark_executed": False,
        "independent_test_claim": False,
        "active_source_snapshot_sha256": active_source_after,
        "git_commit_claim": False,
        "evidence_validity": (
            "only_for_the_exact_active_source_snapshot_and_embedded_safe_report"
        ),
    }
    if arguments.output is not None:
        _write(arguments.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

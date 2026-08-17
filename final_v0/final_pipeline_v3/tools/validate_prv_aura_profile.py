#!/usr/bin/env python3
"""Source-bound fixed-PPI smoke for the isolated Aura 1.0.2 profile."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

from ppg_frailty.contracts import to_strict_json_value
from ppg_frailty.features.prv_backend_compare import (
    fixed_ppi_fixtures,
    run_prv_backend_comparison,
)
from ppg_frailty.provenance import sha256_file, stable_payload_sha256


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "src/ppg_frailty/features/prv_backend_compare.py"
EXPECTED_VERSIONS = {
    "hrv-analysis": "1.0.2",
    "nolds": "0.6.2",
    "astropy": "5.2.2",
    "numpy": "1.26.4",
}
EXPECTED_PREFIX_BASENAME = "prv_aura_hrv102_py311_v2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    target = arguments.output.resolve()
    target.relative_to(ROOT)
    if target.exists():
        raise FileExistsError(f"Aura validation output overwrite forbidden: {target}")
    observed_versions = {
        name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS
    }
    if observed_versions != EXPECTED_VERSIONS:
        raise RuntimeError(
            f"Aura exact runtime version mismatch: {observed_versions}"
        )
    if Path(sys.prefix).resolve().name != EXPECTED_PREFIX_BASENAME:
        raise RuntimeError("Aura smoke requires the named isolated runtime")
    result = run_prv_backend_comparison(
        backends=("aura_hrv_analysis",),
        fixture_ids=tuple(fixed_ppi_fixtures()),
    )
    backend_rows = [
        backend
        for fixture in result["fixtures"]
        for backend in fixture["backends"]
    ]
    if (
        result["status"] != "diagnostic_success_not_exact_profile_evidence"
        or len(result["fixtures"]) != 5
        or len(backend_rows) != 5
        or any(row["status"] != "success" for row in backend_rows)
        or any(row["package_version"] != "1.0.2" for row in backend_rows)
        or any(row["cleaner_applied"] is not False for row in backend_rows)
        or any(row["classifier_integrated"] is not False for row in backend_rows)
    ):
        raise RuntimeError("Aura fixed-PPI adapter smoke failed")
    validator = Path(__file__).resolve()
    payload = {
        "schema_version": "ppg_frailty.prv_aura_profile_smoke.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "passed",
        "scope": "fixed_ppi_function_outputs_only",
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "environment_prefix_basename": Path(sys.prefix).resolve().name,
            "versions": observed_versions,
        },
        "adapter": {
            "path": ADAPTER.relative_to(ROOT).as_posix(),
            "bytes": ADAPTER.stat().st_size,
            "sha256": sha256_file(ADAPTER),
        },
        "validator": {
            "path": validator.relative_to(ROOT).as_posix(),
            "bytes": validator.stat().st_size,
            "sha256": sha256_file(validator),
        },
        "comparison": result,
        "cleaner_applied": False,
        "classifier_integrated": False,
        "scientific_training_executed": False,
        "ablation_executed": False,
        "full_5x5_executed": False,
        "ptt_benchmark_executed": False,
    }
    payload["payload_sha256"] = stable_payload_sha256(payload)
    normalized = to_strict_json_value(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            normalized,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(normalized, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

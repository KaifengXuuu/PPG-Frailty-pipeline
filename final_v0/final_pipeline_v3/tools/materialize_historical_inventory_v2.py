#!/usr/bin/env python3
"""Freeze an immutable byte inventory of the inactive V1-transition archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "historical" / "v1_transition"
OUTPUT = ARCHIVE / "INVENTORY.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not ARCHIVE.is_dir():
        raise FileNotFoundError(f"historical archive missing: {ARCHIVE}")
    if OUTPUT.exists():
        raise FileExistsError(f"historical inventory overwrite forbidden: {OUTPUT}")
    files = []
    for path in sorted(ARCHIVE.rglob("*")):
        if path.is_file() and path != OUTPUT:
            files.append(
                {
                    "path": path.relative_to(ARCHIVE).as_posix(),
                    "bytes": path.stat().st_size,
                    "archived_content_sha256": _sha256(path),
                    "disposition": "inactive_v1_transition_provenance_only",
                }
            )
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload = {
        "schema_version": "ppg_frailty.historical_inventory.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "frozen_inactive_historical_archive",
        "active_use_prohibited": True,
        "source_byte_equivalence_claim": False,
        "scientific_evidence_claim": False,
        "file_count": len(files),
        "content_tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in (
        "schema_version", "status", "file_count", "content_tree_sha256"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

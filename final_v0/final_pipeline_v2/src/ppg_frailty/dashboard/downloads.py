"""Safe in-memory exports for preview and completed study artifacts."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping


def preview_csv_bytes(time_s: Any, traces: Mapping[str, Any]) -> bytes:
    import pandas as pd

    payload: dict[str, Any] = {"time_s": time_s}
    payload.update({str(key): value for key, value in traces.items()})
    return pd.DataFrame(payload).to_csv(index=False).encode("utf-8")


def preview_metadata_bytes(metadata: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(metadata), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def study_zip_bytes(study_dir: str | Path, *, studies_root: str | Path) -> bytes:
    root = Path(studies_root).resolve()
    study = Path(study_dir).resolve()
    study.relative_to(root)
    if not study.is_dir():
        raise FileNotFoundError(study)
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(study.rglob("*")):
            if path.is_file() and not path.is_symlink():
                archive.write(path, arcname=path.relative_to(study.parent).as_posix())
    return output.getvalue()

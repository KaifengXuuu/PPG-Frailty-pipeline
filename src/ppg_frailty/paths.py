"""Separate packaged pipeline resources from the optional external data tree."""

from __future__ import annotations

import os
from pathlib import Path


def pipeline_resource(relative: str | Path) -> Path:
    """Locate a shipped manifest or split independently of the checkout depth."""
    return (Path(__file__).resolve().parents[2] / relative).resolve()


def resolve_repository_root(pipeline_root: str | Path | None = None) -> Path:
    """Use an explicit data root, the historical enclosing tree, or this checkout.

    ``PPG_FRAILTY_DATA_ROOT`` points to the directory containing the original
    dataset directories and any optional historical evidence, not to a dataset
    subdirectory. Packaged configs, outputs and weights stay in the pipeline.
    """
    configured = os.environ.get("PPG_FRAILTY_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(pipeline_root).resolve() if pipeline_root is not None else pipeline_resource(".")
    return root.parent.parent if root.parent.name == "final_v0" else root

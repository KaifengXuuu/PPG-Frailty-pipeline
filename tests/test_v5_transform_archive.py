from __future__ import annotations

from pathlib import Path

import pytest

from ppg_frailty.training.bundle import FrozenRepresentationTransformArchive


_DIGEST = "a" * 64
_PARTICIPANTS = tuple(f"P{index:02d}" for index in range(1, 30))
_RAW_IMU_NOOP = {
    "schema_version": "not_applicable_all8_window_normalized_v1",
    "artifact_sha256": None,
    "fitted_on_participant_ids": (),
    "channel_schema": ("A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"),
    "valid_count": None,
    "strategy": "none_after_all8_per_window_robust",
    "parameters": None,
}


def _archive(*, provenance: dict[str, object]) -> FrozenRepresentationTransformArchive:
    return FrozenRepresentationTransformArchive(
        representation_mode="raw",
        input_schema_hash=_DIGEST,
        fitted_on_participant_ids=_PARTICIPANTS,
        fitted_artifacts={},
        provenance=provenance,
        source_records_hash=_DIGEST,
        dataset_hash=_DIGEST,
    )


def test_raw_archive_accepts_v2_explicit_noop_without_fitted_object() -> None:
    archive = _archive(provenance={"raw_imu": dict(_RAW_IMU_NOOP)})
    assert archive.fitted_artifacts == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "drift"),
        ("artifact_sha256", _DIGEST),
        ("fitted_on_participant_ids", _PARTICIPANTS),
        ("strategy", "outer_train_robust"),
        ("parameters", {}),
    ],
)
def test_raw_archive_rejects_missing_object_without_exact_noop_sentinel(
    field: str, value: object
) -> None:
    provenance = dict(_RAW_IMU_NOOP)
    provenance[field] = value
    with pytest.raises(ValueError, match="transform set"):
        _archive(provenance={"raw_imu": provenance})

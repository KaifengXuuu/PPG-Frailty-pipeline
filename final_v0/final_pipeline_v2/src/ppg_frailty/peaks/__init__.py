"""脉搏事件、间期和匹配门面 / Pulse event, interval, and matching facade."""

from .aboy_project import DETECTOR_NAME
from .intervals import PrvResult, compute_prv
from .pairing import (
    BeatPairAudit,
    BeatPairingResult,
    DUAL_WAVELENGTH_PAIRING_SCHEMA_VERSION,
    EventMatchMetrics,
    match_events,
    pair_dual_wavelength_beats,
    select_reference_wavelength,
)
from .resolver import (
    ABLATION_DETECTOR_ID,
    CANONICAL_DETECTOR_ID,
    REGISTERED_DETECTOR_IDS,
    detect_pulses,
    detect_pulses_per_wavelength,
    resolve_detector_id,
)

__all__ = [
    "ABLATION_DETECTOR_ID",
    "BeatPairAudit",
    "BeatPairingResult",
    "CANONICAL_DETECTOR_ID",
    "DETECTOR_NAME",
    "DUAL_WAVELENGTH_PAIRING_SCHEMA_VERSION",
    "EventMatchMetrics",
    "PrvResult",
    "REGISTERED_DETECTOR_IDS",
    "compute_prv",
    "detect_pulses",
    "detect_pulses_per_wavelength",
    "match_events",
    "pair_dual_wavelength_beats",
    "resolve_detector_id",
    "select_reference_wavelength",
]

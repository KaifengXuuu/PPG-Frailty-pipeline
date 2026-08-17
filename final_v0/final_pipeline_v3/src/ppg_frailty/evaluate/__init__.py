"""规范评价门面 / Canonical evaluation facade.

中文：汇总唯一聚合、指标、OOF 与校准入口。English: Export the sole aggregation, metrics, OOF and calibration entries.
"""

from .aggregate import StrictAggregationResult, aggregate_hierarchy_strict
from .benchmark import PairedMetricDelta, paired_metric_delta, summarize_repeats
from .calibration import TemperatureCalibrator, fit_temperature
from .metrics import ParticipantMetrics, evaluate_participant_probabilities
from .oof import OofContractAudit, validate_oof_contract

__all__ = [
    "OofContractAudit", "PairedMetricDelta", "ParticipantMetrics",
    "StrictAggregationResult", "TemperatureCalibrator", "aggregate_hierarchy_strict",
    "evaluate_participant_probabilities", "fit_temperature", "paired_metric_delta",
    "summarize_repeats", "validate_oof_contract",
]

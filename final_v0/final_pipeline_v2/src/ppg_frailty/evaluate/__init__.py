"""规范评价门面 / Canonical evaluation facade.

中文：汇总唯一聚合、指标、OOF 与校准入口。English: Export the sole aggregation, metrics, OOF and calibration entries.
"""

from .aggregate import StrictAggregationResult, aggregate_hierarchy_strict
from .benchmark import PairedMetricDelta, paired_metric_delta, summarize_repeats
from .calibration import TemperatureCalibrator, fit_temperature
from .decision_bias_oracle import (
    BiasOracleResult,
    DecisionBiasOraclePlan,
    enumerate_simplex_biases,
    load_decision_bias_oracle_plan,
    load_participant_oracle_dataset,
    run_decision_bias_oracle,
    search_decision_bias_oracle,
)
from .metrics import ParticipantMetrics, evaluate_participant_probabilities
from .oof import OofContractAudit, validate_oof_contract
from .role_scope_decomposition import (
    LoadedSource,
    RoleScopePlan,
    SourceSpec,
    load_role_scope_plan,
    run_role_scope_decomposition,
)

__all__ = [
    "BiasOracleResult", "DecisionBiasOraclePlan", "OofContractAudit",
    "PairedMetricDelta", "ParticipantMetrics",
    "StrictAggregationResult", "TemperatureCalibrator", "aggregate_hierarchy_strict",
    "enumerate_simplex_biases", "evaluate_participant_probabilities",
    "fit_temperature", "load_decision_bias_oracle_plan",
    "load_participant_oracle_dataset", "paired_metric_delta",
    "LoadedSource", "RoleScopePlan", "SourceSpec", "load_role_scope_plan",
    "run_decision_bias_oracle", "run_role_scope_decomposition", "search_decision_bias_oracle",
    "summarize_repeats", "validate_oof_contract",
]

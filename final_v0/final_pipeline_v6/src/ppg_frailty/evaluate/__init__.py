"""Calibration and specialized evaluation entry points."""

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
from .role_scope_decomposition import (
    LoadedSource,
    RoleScopePlan,
    SourceSpec,
    load_role_scope_plan,
    run_role_scope_decomposition,
)

__all__ = [
    "BiasOracleResult",
    "DecisionBiasOraclePlan",
    "PairedMetricDelta",
    "TemperatureCalibrator",
    "enumerate_simplex_biases",
    "fit_temperature",
    "load_decision_bias_oracle_plan",
    "load_participant_oracle_dataset",
    "paired_metric_delta",
    "LoadedSource",
    "RoleScopePlan",
    "SourceSpec",
    "load_role_scope_plan",
    "run_decision_bias_oracle",
    "run_role_scope_decomposition",
    "search_decision_bias_oracle",
    "summarize_repeats",
]

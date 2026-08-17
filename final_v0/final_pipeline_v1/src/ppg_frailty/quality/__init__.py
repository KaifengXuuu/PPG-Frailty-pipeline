"""规范 SQI 门面 / Canonical endpoint-SQI facade.

中文：稳定导出 direct Q_rate/Q_morph 与 non-identity rate-only 路由检查。
English: Stable exports for direct endpoint SQI and rate-only route enforcement.
"""

from .components import QualityComponent, QualityEndpoint, QualityResult, QualityState, component_rows
from .endpoint_sqi import SqiCalibrator, SqiConfig, evaluate_quality, fit_sqi_calibrator, quality_component_scores
from .routing import assert_quality_route

__all__ = [
    "QualityComponent", "QualityEndpoint", "QualityResult", "QualityState",
    "SqiCalibrator", "SqiConfig", "assert_quality_route", "component_rows",
    "evaluate_quality", "fit_sqi_calibrator", "quality_component_scores",
]
